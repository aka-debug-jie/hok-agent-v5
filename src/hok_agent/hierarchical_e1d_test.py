from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors.torch import load_file

from hok_agent.hierarchical_e1c_clip import _load_manifest, _session_frames
from hok_agent.hierarchical_e1d_checkpoint import verify_checkpoint_bundle
from hok_agent.hierarchical_e1d_clip import load_contract as load_clip_contract
from hok_agent.hierarchical_e1d_clip import select_matched_negative
from hok_agent.hierarchical_e1d_crystal import load_contract as load_crystal_contract
from hok_agent.hierarchical_e1d_crystal import locate_transition
from hok_agent.hierarchical_e1d_probe import Mode, TerminalProbe, _macro_f1, _permute

SCHEMA = "hok-agent-hierarchical-event-e1d-test-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1d-test-report-v1"


class E1dTestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TestConfig:
    sessions_expected: int
    probability_threshold: float
    minimum_pairs: int
    minimum_macro_f1: float
    minimum_last_margin: float
    minimum_shuffle_margin: float


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_test_contract(path: Path) -> tuple[TestConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    evaluation = cast(dict[str, object], payload["evaluation"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_one_shot_test_contract"
        or payload.get("test_authorized_by_owner") is not True
        or evaluation.get("retrain_or_threshold_tuning_after_test") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
    ):
        raise E1dTestError("test contract boundary differs")
    config = TestConfig(
        int(cast(int, payload["test_sessions_expected"])),
        float(cast(float, evaluation["probability_threshold"])),
        int(cast(int, evaluation["minimum_test_pairs"])),
        float(cast(float, evaluation["minimum_temporal_macro_f1"])),
        float(cast(float, evaluation["minimum_gain_over_last_frame"])),
        float(cast(float, evaluation["minimum_gain_over_shuffled_frames"])),
    )
    return config, payload, _sha(_canonical(payload))


def _model(bundle: Path, mode: Mode) -> TerminalProbe:
    model = TerminalProbe(mode)
    model.load_state_dict(load_file(bundle / f"{mode}.safetensors", device="cpu"))
    model.eval()
    return model


def _tensor(frames: list[np.ndarray]) -> torch.Tensor:
    array = np.stack(frames)[:, :, ::4, ::4, :]
    return torch.from_numpy(array.copy()).permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0


def _evaluate(
    model: TerminalProbe, clips: torch.Tensor, labels: torch.Tensor, mode: Mode, threshold: float
) -> dict[str, float]:
    inputs = _permute(clips) if mode == "shuffled" else clips
    with torch.no_grad():
        confidence = torch.softmax(model(inputs), dim=1)[:, 1]
        predictions = (confidence >= threshold).long()
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "macro_f1": _macro_f1(labels, predictions),
        "positive_recall": float((predictions[labels == 1] == 1).float().mean()),
        "negative_specificity": float((predictions[labels == 0] == 0).float().mean()),
    }


def run_test(
    contract_path: Path,
    checkpoint_contract_path: Path,
    checkpoint_bundle: Path,
    crystal_contract_path: Path,
    clip_contract_path: Path,
    target_dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_test_contract(contract_path)
    bundle = verify_checkpoint_bundle(checkpoint_bundle, checkpoint_contract_path)
    crystal_config, _crystal, crystal_sha = load_crystal_contract(crystal_contract_path)
    clip_config, _clip, clip_sha = load_clip_contract(clip_contract_path)
    manifest = _load_manifest(target_dataset / "manifest.json")
    if (
        contract["checkpoint_bundle_sha256"] != bundle["bundle_sha256"]
        or contract["crystal_contract_sha256"] != crystal_sha
        or contract["clip_contract_sha256"] != clip_sha
        or contract["target_manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise E1dTestError("test evidence binding differs")
    expected_hashes = cast(dict[str, str], contract["checkpoint_sha256"])
    actual = cast(dict[str, dict[str, object]], bundle["checkpoints"])
    if any(actual[mode]["sha256"] != expected_hashes[mode] for mode in expected_hashes):
        raise E1dTestError("test checkpoint hash differs")
    session_rows = cast(list[dict[str, object]], manifest["sessions"])
    test_sessions = [
        cast(str, row["session_hash"]) for row in session_rows if row["split"] == "test"
    ]
    if len(test_sessions) != config.sessions_expected:
        raise E1dTestError("test session count differs")
    clips: list[np.ndarray] = []
    labels: list[int] = []
    accepted_ids: list[str] = []
    for session in sorted(test_sessions):
        frames, times, _hashes = _session_frames(target_dataset, manifest, session)
        event_index, evidence = locate_transition(frames, times, crystal_config)
        if not bool(evidence["accepted"]):
            continue
        try:
            negative_index = select_matched_negative(frames, event_index, clip_config)
        except ValueError:
            continue
        positive = frames[event_index - clip_config.before : event_index + clip_config.after + 1]
        negative = frames[
            negative_index - clip_config.before : negative_index + clip_config.after + 1
        ]
        if len(positive) != clip_config.sequence_frames:
            continue
        anonymous = _sha(f"e1d-test:{contract_sha}:{session}".encode())
        pair = [(negative, 0), (positive, 1)]
        if int(anonymous[-1], 16) % 2:
            pair.reverse()
        for clip, label in pair:
            clips.append(clip)
            labels.append(label)
        accepted_ids.append(anonymous)
    if not clips:
        raise E1dTestError("test produced no weak-label pairs")
    inputs = _tensor(clips)
    targets = torch.tensor(labels, dtype=torch.long)
    metrics = {
        mode: _evaluate(
            _model(checkpoint_bundle, mode), inputs, targets, mode, config.probability_threshold
        )
        for mode in ("temporal", "last_frame", "shuffled")
    }
    temporal = metrics["temporal"]["macro_f1"]
    checks = {
        "minimum_pairs": len(accepted_ids) >= config.minimum_pairs,
        "temporal_macro_f1": temporal >= config.minimum_macro_f1,
        "last_frame_margin": temporal - metrics["last_frame"]["macro_f1"]
        >= config.minimum_last_margin,
        "shuffle_margin": temporal - metrics["shuffled"]["macro_f1"]
        >= config.minimum_shuffle_margin,
        "both_classes": set(labels) == {0, 1},
        "split_overlap": len(set(test_sessions)) == len(test_sessions),
    }
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "E1D_ONE_SHOT_TEST_PASSED"
        if all(checks.values())
        else "E1D_ONE_SHOT_TEST_FAILED",
        "contract_sha256": contract_sha,
        "checkpoint_bundle_sha256": bundle["bundle_sha256"],
        "test_sessions_opened": len(test_sessions),
        "test_pairs": len(accepted_ids),
        "metrics": metrics,
        "checks": checks,
        "accepted_session_ids": accepted_ids,
        "training_called": False,
        "threshold_tuned_after_test": False,
        "automatic_weak_label_generalization_only": True,
        "semantic_accuracy_verified": False,
        "win_loss_verified": False,
        "reward_allowed": False,
        "promotion_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload))
    if output_dir.exists() or output_dir.is_symlink():
        raise E1dTestError("test output already exists")
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
    parser = argparse.ArgumentParser(description="One-shot E1d weak-label test")
    for name in (
        "contract",
        "checkpoint-contract",
        "checkpoint-bundle",
        "crystal-contract",
        "clip-contract",
        "target-dataset",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_test(
        args.contract,
        args.checkpoint_contract,
        args.checkpoint_bundle,
        args.crystal_contract,
        args.clip_contract,
        args.target_dataset,
        args.output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
