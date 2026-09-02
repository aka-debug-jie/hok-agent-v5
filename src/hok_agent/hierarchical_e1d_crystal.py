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

SCHEMA = "hok-agent-hierarchical-event-e1d-crystal-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1d-crystal-report-v1"


class E1dError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CrystalConfig:
    final_similarity_max: float
    search_last_frames: int
    minimum_temporal_difference: float
    minimum_white_fraction: float
    minimum_pre_post_change: float
    radius: int
    minimum_train_candidates: int
    minimum_dev_candidates: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[CrystalConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    features = cast(dict[str, object], payload["features"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "exploratory_visual_consensus_audit"
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
        or claim.get("video_test_allowed") is not False
    ):
        raise E1dError("E1d contract boundary differs")
    config = CrystalConfig(
        float(cast(float, features["final_state_similarity_max"])),
        int(cast(int, features["search_last_frames"])),
        float(cast(float, features["minimum_temporal_difference"])),
        float(cast(float, features["minimum_white_fraction"])),
        float(cast(float, features["minimum_pre_post_change"])),
        int(cast(int, features["pre_post_radius_frames"])),
        int(cast(int, gate["minimum_train_candidates"])),
        int(cast(int, gate["minimum_dev_candidates"])),
    )
    return config, payload, _sha(_canonical(payload))


def detect_transition(
    frames: np.ndarray,
    timestamps: np.ndarray,
    config: CrystalConfig,
) -> dict[str, object]:
    _candidate, result = locate_transition(frames, timestamps, config)
    return result


def locate_transition(
    frames: np.ndarray,
    timestamps: np.ndarray,
    config: CrystalConfig,
) -> tuple[int, dict[str, object]]:
    rgb = frames.astype(np.float32)
    final_difference = np.abs(rgb - rgb[-1]).mean(axis=(1, 2, 3)) / 255.0
    onset = len(rgb) - 1
    while onset > 0 and final_difference[onset - 1] < config.final_similarity_max:
        onset -= 1
    before_result = rgb[:onset]
    if len(before_result) <= config.radius * 2:
        raise E1dError("E1d session has no pre-result sequence")
    center = before_result[:, 16:112, 24:104]
    gray = center.mean(axis=3)
    temporal = np.zeros(len(gray))
    temporal[1:] = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2)) / 255.0
    white = (center.min(axis=3) > 200).mean(axis=(1, 2))
    start = max(config.radius, len(gray) - config.search_last_frames)
    indices = np.arange(start, len(gray))
    candidate = int(indices[np.argmax(temporal[indices] + 0.5 * white[indices])])
    pre = gray[candidate - config.radius]
    post = rgb[min(len(rgb) - 1, candidate + config.radius), 16:112, 24:104].mean(axis=2)
    change = float(np.abs(post - pre).mean() / 255.0)
    accepted = bool(
        temporal[candidate] >= config.minimum_temporal_difference
        and white[candidate] >= config.minimum_white_fraction
        and change >= config.minimum_pre_post_change
    )
    return candidate, {
        "accepted": accepted,
        "candidate_seconds_before_stable_state": float(
            (timestamps[onset] - timestamps[candidate]) / 1000.0
        ),
        "temporal_difference": float(temporal[candidate]),
        "white_fraction": float(white[candidate]),
        "pre_post_change": change,
    }


def audit(
    contract_path: Path,
    anchor_contract_path: Path,
    anchor_report_path: Path,
    target_dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, _contract, contract_sha = load_contract(contract_path)
    _a, _b, anchor_sha = load_anchor_contract(anchor_contract_path)
    anchor_report = load_anchor_report(anchor_report_path, anchor_contract_path)
    manifest = _load_manifest(target_dataset / "manifest.json")
    sessions = _selected_sessions(manifest, anchor_report, anchor_sha)
    rows = []
    for session, split in sorted(sessions.items()):
        frames, times, _hashes = _session_frames(target_dataset, manifest, session)
        row = detect_transition(frames, times, config)
        row.update({"session_id": _sha(f"e1d:{contract_sha}:{session}".encode()), "split": split})
        rows.append(row)
    train = sum(row["split"] == "train" and bool(row["accepted"]) for row in rows)
    dev = sum(row["split"] == "dev" and bool(row["accepted"]) for row in rows)
    checks = {
        "train_support": train >= config.minimum_train_candidates,
        "dev_support": dev >= config.minimum_dev_candidates,
    }
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "E1D_CRYSTAL_CONSENSUS_PASSED"
            if all(checks.values())
            else "E1D_CRYSTAL_CONSENSUS_FAILED"
        ),
        "contract_sha256": contract_sha,
        "train_candidates": train,
        "dev_candidates": dev,
        "checks": checks,
        "rows": rows,
        "semantic_accuracy_verified": False,
        "win_loss_verified": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "video_test_opened": False,
        "gpu_used": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload))
    if output_dir.exists():
        raise E1dError("E1d output exists")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--anchor-contract", type=Path, required=True)
    parser.add_argument("--anchor-report", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit(
        args.contract,
        args.anchor_contract,
        args.anchor_report,
        args.target_dataset,
        args.output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
