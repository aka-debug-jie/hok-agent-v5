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

from hok_agent.hierarchical_e1c_anchor import load_anchor_contract, load_anchor_report
from hok_agent.hierarchical_e1c_clip import _load_manifest, _selected_sessions, _session_frames
from hok_agent.hierarchical_e1d_crystal import load_contract as load_crystal_contract
from hok_agent.hierarchical_e1d_crystal import locate_transition

SCHEMA = "hok-agent-hierarchical-event-e1d-clip-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1d-clip-report-v1"


class E1dClipError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ClipConfig:
    sequence_frames: int
    before: int
    after: int
    exclusion_radius: int
    search_frames: int
    minimum_train_pairs: int
    minimum_dev_pairs: int
    maximum_dev_ordinal_accuracy: float


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[ClipConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    clip = cast(dict[str, object], payload["clip"])
    model = cast(dict[str, object], payload["model_input"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_event_centered_clip_materialization"
        or model.get("fields") != ["rgb_sequence"]
        or any(model.get(key) is not False for key in model if key != "fields")
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
        or claim.get("video_test_allowed") is not False
    ):
        raise E1dClipError("E1d clip boundary differs")
    config = ClipConfig(
        int(cast(int, clip["sequence_frames"])),
        int(cast(int, clip["positive_before_frames"])),
        int(cast(int, clip["positive_after_frames"])),
        int(cast(int, clip["negative_exclusion_radius_frames"])),
        int(cast(int, clip["negative_search_frames"])),
        int(cast(int, gate["minimum_train_pairs"])),
        int(cast(int, gate["minimum_dev_pairs"])),
        float(cast(float, gate["maximum_dev_ordinal_accuracy"])),
    )
    if config.before + config.after + 1 != config.sequence_frames:
        raise E1dClipError("E1d clip sequence length differs")
    return config, payload, _sha(_canonical(payload))


def select_matched_negative(
    frames: np.ndarray,
    event_index: int,
    config: ClipConfig,
) -> int:
    reference = frames[event_index - config.before].astype(np.float32)
    reference_gray = reference.mean(axis=2)
    reference_brightness = float(reference_gray.mean())
    reference_edge = float(
        np.abs(np.diff(reference_gray, axis=0)).mean()
        + np.abs(np.diff(reference_gray, axis=1)).mean()
    )
    first = max(config.before, event_index - config.search_frames)
    last = event_index - config.exclusion_radius - config.after
    candidates: list[tuple[float, int]] = []
    for center in range(first, last + 1):
        window = frames[center - config.before : center + config.after + 1].astype(np.float32)
        white = (window.min(axis=3) > 200).mean()
        gray = window[config.before].mean(axis=2)
        temporal = np.abs(window[1:].mean(axis=3) - window[:-1].mean(axis=3)).mean() / 255.0
        if white >= 0.55 or temporal >= 0.3:
            continue
        edge = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
        score = (
            abs(float(gray.mean()) - reference_brightness) / 255.0
            + abs(edge - reference_edge) / 255.0
        )
        candidates.append((score, center))
    if not candidates:
        raise E1dClipError("E1d clip has no matched negative")
    return min(candidates)[1]


def materialize(
    contract_path: Path,
    crystal_contract_path: Path,
    crystal_report_path: Path,
    anchor_contract_path: Path,
    anchor_report_path: Path,
    target_dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    crystal_config, _crystal_contract, crystal_sha = load_crystal_contract(crystal_contract_path)
    crystal_report = cast(dict[str, object], json.loads(crystal_report_path.read_text()))
    if (
        contract["crystal_report_sha256"] != crystal_report.get("report_sha256")
        or crystal_report.get("contract_sha256") != crystal_sha
    ):
        raise E1dClipError("E1d crystal report binding differs")
    accepted = {
        cast(str, row["session_id"])
        for row in cast(list[dict[str, object]], crystal_report["rows"])
        if bool(row["accepted"])
    }
    _a, _b, anchor_sha = load_anchor_contract(anchor_contract_path)
    anchor_report = load_anchor_report(anchor_report_path, anchor_contract_path)
    manifest = _load_manifest(target_dataset / "manifest.json")
    anchored = _selected_sessions(manifest, anchor_report, anchor_sha)
    selected = {
        session: split
        for session, split in anchored.items()
        if _sha(f"e1d:{crystal_sha}:{session}".encode()) in accepted
    }
    clips: dict[str, list[np.ndarray]] = {"train": [], "dev": []}
    labels: dict[str, list[int]] = {"train": [], "dev": []}
    sessions: dict[str, list[str]] = {"train": [], "dev": []}
    incomplete = 0
    for session, split in sorted(selected.items()):
        frames, times, _hashes = _session_frames(target_dataset, manifest, session)
        event_index, evidence = locate_transition(frames, times, crystal_config)
        if (
            not bool(evidence["accepted"])
            or event_index < config.before
            or event_index + config.after >= len(frames)
        ):
            incomplete += 1
            continue
        try:
            negative_index = select_matched_negative(frames, event_index, config)
        except E1dClipError:
            incomplete += 1
            continue
        positive = frames[event_index - config.before : event_index + config.after + 1]
        negative = frames[negative_index - config.before : negative_index + config.after + 1]
        anonymous = _sha(f"e1d-clip:{contract_sha}:{session}".encode())
        pair = [(negative, 0), (positive, 1)]
        if int(anonymous[-1], 16) % 2:
            pair.reverse()
        for clip, label in pair:
            clips[split].append(clip)
            labels[split].append(label)
            sessions[split].append(anonymous)
    pair_counts = {split: len(clips[split]) // 2 for split in ("train", "dev")}
    ordinal = {}
    for split in ("train", "dev"):
        predictions = np.tile([0, 1], pair_counts[split])
        ordinal[split] = float((predictions == np.asarray(labels[split])).mean())
    checks = {
        "train_pairs": pair_counts["train"] >= config.minimum_train_pairs,
        "dev_pairs": pair_counts["dev"] >= config.minimum_dev_pairs,
        "dev_ordinal": ordinal["dev"] <= config.maximum_dev_ordinal_accuracy,
        "cross_split": not (set(sessions["train"]) & set(sessions["dev"])),
    }
    if output_dir.exists():
        raise E1dClipError("E1d clip output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        shard_hashes = {}
        for split in ("train", "dev"):
            shard = staging / f"{split}.npz"
            np.savez_compressed(
                shard,
                rgb_sequence=np.stack(clips[split]),
                label=np.asarray(labels[split], dtype=np.int8),
                session_id=np.asarray(sessions[split], dtype="<U64"),
            )
            shard_hashes[split] = _sha(shard.read_bytes())
        payload: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "status": "E1D_CLIP_MATERIALIZATION_PASSED"
            if all(checks.values())
            else "E1D_CLIP_MATERIALIZATION_FAILED",
            "contract_sha256": contract_sha,
            "crystal_report_sha256": crystal_report["report_sha256"],
            "pair_counts": pair_counts,
            "incomplete_sessions": incomplete,
            "ordinal_accuracy": ordinal,
            "checks": checks,
            "shard_sha256": shard_hashes,
            "model_input_fields": ["rgb_sequence"],
            "semantic_accuracy_verified": False,
            "win_loss_verified": False,
            "reward_allowed": False,
            "promotion_allowed": False,
            "video_test_opened": False,
        }
        payload["report_sha256"] = _sha(_canonical(payload))
        (staging / "report.json").write_bytes(_canonical(payload) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "contract",
        "crystal-contract",
        "crystal-report",
        "anchor-contract",
        "anchor-report",
        "target-dataset",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = materialize(
        args.contract,
        args.crystal_contract,
        args.crystal_report,
        args.anchor_contract,
        args.anchor_report,
        args.target_dataset,
        args.output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
