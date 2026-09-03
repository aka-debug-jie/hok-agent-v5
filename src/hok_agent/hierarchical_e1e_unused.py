from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import torch

from hok_agent.hierarchical_e1c_anchor import load_anchor_contract, load_anchor_report
from hok_agent.hierarchical_e1c_clip import _load_manifest, _session_frames
from hok_agent.hierarchical_e1d_checkpoint import verify_checkpoint_bundle
from hok_agent.hierarchical_e1d_clip import E1dClipError, select_matched_negative
from hok_agent.hierarchical_e1d_clip import load_contract as load_clip_contract
from hok_agent.hierarchical_e1d_crystal import E1dError, locate_transition
from hok_agent.hierarchical_e1d_crystal import load_contract as load_crystal_contract
from hok_agent.hierarchical_e1d_test import _evaluate, _model, _tensor

SCHEMA = "hok-agent-hierarchical-event-e1e-unused-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1e-unused-report-v1"


class E1eError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path) -> tuple[dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    selection = cast(dict[str, object], payload["selection"])
    evaluation = cast(dict[str, object], payload["evaluation"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_unused_train_dev_diagnostic"
        or selection.get("source_splits") != ["train", "dev"]
        or selection.get("result_page_anchored") is not False
        or selection.get("test_allowed") is not False
        or evaluation.get("continue_after_ineligible_session") is not True
        or evaluation.get("retraining_allowed") is not False
        or evaluation.get("threshold_tuning_allowed") is not False
        or claim.get("formal_test_replacement") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("integration_allowed") is not False
    ):
        raise E1eError("E1e contract boundary differs")
    return payload, _sha(_canonical(payload))


def run_unused_audit(
    contract_path: Path,
    anchor_contract_path: Path,
    anchor_report_path: Path,
    checkpoint_contract_path: Path,
    checkpoint_bundle: Path,
    consumed_test_report: Path,
    crystal_contract_path: Path,
    clip_contract_path: Path,
    target_dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract, contract_sha = load_contract(contract_path)
    bindings = cast(dict[str, object], contract["bindings"])
    selection = cast(dict[str, object], contract["selection"])
    evaluation = cast(dict[str, object], contract["evaluation"])
    bundle = verify_checkpoint_bundle(checkpoint_bundle, checkpoint_contract_path)
    failed_test = cast(dict[str, object], json.loads(consumed_test_report.read_text()))
    anchor_config, _anchor, anchor_sha = load_anchor_contract(anchor_contract_path)
    anchor_report = load_anchor_report(anchor_report_path, anchor_contract_path)
    crystal_config, _crystal, crystal_sha = load_crystal_contract(crystal_contract_path)
    clip_config, _clip, clip_sha = load_clip_contract(clip_contract_path)
    manifest = _load_manifest(target_dataset / "manifest.json")
    if (
        bindings["anchor_report_sha256"] != anchor_report["report_sha256"]
        or bindings["checkpoint_bundle_sha256"] != bundle["bundle_sha256"]
        or bindings["consumed_test_failure_sha256"] != failed_test.get("report_sha256")
        or failed_test.get("rerun_allowed") is not False
        or bindings["target_manifest_sha256"] != manifest["manifest_sha256"]
        or bindings["crystal_contract_sha256"] != crystal_sha
        or bindings["clip_contract_sha256"] != clip_sha
    ):
        raise E1eError("E1e evidence binding differs")
    anchor_rows = cast(list[dict[str, object]], anchor_report["rows"])
    unused_ids = {
        cast(str, row["session_id"]): row["split"]
        for row in anchor_rows
        if not bool(row["anchored"]) and row["split"] in {"train", "dev"}
    }
    manifest_sessions = cast(list[dict[str, object]], manifest["sessions"])
    selected: dict[str, str] = {}
    for row in manifest_sessions:
        session = cast(str, row["session_hash"])
        anonymous = _sha(f"e1c-anchor:{anchor_sha}:{session}".encode())
        if anonymous in unused_ids:
            selected[session] = unused_ids[anonymous]
    clips: list[np.ndarray] = []
    labels: list[int] = []
    eligible_by_split: Counter[str] = Counter()
    ineligible: Counter[str] = Counter()
    for session, split in sorted(selected.items()):
        frames, times, _hashes = _session_frames(target_dataset, manifest, session)
        try:
            event_index, evidence = locate_transition(frames, times, crystal_config)
            if not bool(evidence["accepted"]):
                ineligible["NO_VISUAL_CONSENSUS"] += 1
                continue
            negative_index = select_matched_negative(frames, event_index, clip_config)
            positive = frames[
                event_index - clip_config.before : event_index + clip_config.after + 1
            ]
            negative = frames[
                negative_index - clip_config.before : negative_index + clip_config.after + 1
            ]
            if len(positive) != clip_config.sequence_frames:
                raise E1dClipError("positive sequence incomplete")
        except E1dError:
            ineligible["NO_PRE_RESULT_SEQUENCE"] += 1
            continue
        except E1dClipError:
            ineligible["NO_MATCHED_PAIR"] += 1
            continue
        anonymous = _sha(f"e1e:{contract_sha}:{session}".encode())
        pair = [(negative, 0), (positive, 1)]
        if int(anonymous[-1], 16) % 2:
            pair.reverse()
        for clip, label in pair:
            clips.append(clip)
            labels.append(label)
        eligible_by_split[split] += 1
    minimum_train = int(cast(int, selection["minimum_eligible_train_pairs"]))
    minimum_dev = int(cast(int, selection["minimum_eligible_dev_pairs"]))
    enough_support = (
        eligible_by_split["train"] >= minimum_train and eligible_by_split["dev"] >= minimum_dev
    )
    metrics: dict[str, dict[str, float]] = {}
    if clips:
        inputs = _tensor(clips)
        targets = torch.tensor(labels, dtype=torch.long)
        threshold = float(cast(float, evaluation["probability_threshold"]))
        metrics = {
            mode: _evaluate(_model(checkpoint_bundle, mode), inputs, targets, mode, threshold)
            for mode in ("temporal", "last_frame", "shuffled")
        }
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "E1E_UNUSED_SESSION_DIAGNOSTIC_COMPLETE"
        if enough_support
        else "E1E_UNUSED_SESSION_SUPPORT_INSUFFICIENT",
        "contract_sha256": contract_sha,
        "checkpoint_bundle_sha256": bundle["bundle_sha256"],
        "selected_sessions": len(selected),
        "eligible_pairs": dict(eligible_by_split),
        "ineligible_reasons": dict(ineligible),
        "metrics": metrics,
        "formal_test_replacement": False,
        "training_called": False,
        "threshold_tuned": False,
        "test_frames_opened": False,
        "semantic_accuracy_verified": False,
        "reward_allowed": False,
        "integration_allowed": False,
        "promotion_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload))
    if output_dir.exists() or output_dir.is_symlink():
        raise E1eError("E1e output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / "report.json").write_bytes(_canonical(payload) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E1e unused train/dev diagnostic")
    names = (
        "contract",
        "anchor-contract",
        "anchor-report",
        "checkpoint-contract",
        "checkpoint-bundle",
        "consumed-test-report",
        "crystal-contract",
        "clip-contract",
        "target-dataset",
        "output-dir",
    )
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_unused_audit(
        args.contract,
        args.anchor_contract,
        args.anchor_report,
        args.checkpoint_contract,
        args.checkpoint_bundle,
        args.consumed_test_report,
        args.crystal_contract,
        args.clip_contract,
        args.target_dataset,
        args.output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
