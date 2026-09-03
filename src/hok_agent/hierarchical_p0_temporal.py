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

from hok_agent.hierarchical_e1c_clip import _load_manifest, _session_frames

SCHEMA = "hok-agent-hierarchical-p0-temporal-data-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p0-temporal-data-report-v1"


class P0TemporalError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TemporalDataConfig:
    train_sessions: int
    dev_sessions: int
    windows_per_session: int
    sequence_frames: int
    minimum_train_pairs: int
    minimum_dev_pairs: int
    maximum_dev_ordinal_accuracy: float


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path) -> tuple[TemporalDataConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    selection = cast(dict[str, object], payload["selection"])
    pair = cast(dict[str, object], payload["pair"])
    model = cast(dict[str, object], payload["model_input"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_middle_shuffle_materialization"
        or selection.get("test_allowed") is not False
        or pair.get("first_frame_fixed") is not True
        or pair.get("last_frame_fixed") is not True
        or pair.get("same_frame_multiset") is not True
        or model.get("fields") != ["rgb_sequence"]
        or model.get("timestamp_allowed") is not False
        or claim.get("p0_initialization_allowed") is not False
        or claim.get("reward_allowed") is not False
    ):
        raise P0TemporalError("P0 temporal data boundary differs")
    config = TemporalDataConfig(
        int(cast(int, selection["train_sessions"])),
        int(cast(int, selection["dev_sessions"])),
        int(cast(int, selection["windows_per_session"])),
        int(cast(int, selection["sequence_frames"])),
        int(cast(int, gate["minimum_train_pairs"])),
        int(cast(int, gate["minimum_dev_pairs"])),
        float(cast(float, gate["maximum_dev_ordinal_accuracy"])),
    )
    return config, payload, _sha(_canonical(payload))


def middle_shuffle(frames: np.ndarray, seed: int) -> np.ndarray:
    if len(frames) < 3:
        raise P0TemporalError("temporal pair requires at least three frames")
    order = np.arange(len(frames))
    generator = np.random.default_rng(seed)
    order[1:-1] = generator.permutation(order[1:-1])
    return frames[order]


def _selected_sessions(
    manifest: dict[str, object],
    contract_sha: str,
    config: TemporalDataConfig,
) -> dict[str, tuple[str, ...]]:
    rows = cast(list[dict[str, object]], manifest["sessions"])
    result = {}
    for split, count in (("train", config.train_sessions), ("dev", config.dev_sessions)):
        sessions = [cast(str, row["session_hash"]) for row in rows if row["split"] == split]
        sessions.sort(key=lambda value: _sha(f"{contract_sha}:{value}".encode()))
        result[split] = tuple(sessions[:count])
    return result


def materialize(
    contract_path: Path,
    target_dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    manifest = _load_manifest(target_dataset / "manifest.json")
    if manifest["manifest_sha256"] != contract["target_manifest_sha256"]:
        raise P0TemporalError("P0 temporal manifest binding differs")
    selected = _selected_sessions(manifest, contract_sha, config)
    clips: dict[str, list[np.ndarray]] = {"train": [], "dev": []}
    labels: dict[str, list[int]] = {"train": [], "dev": []}
    session_ids: dict[str, list[str]] = {"train": [], "dev": []}
    pair_counts = {"train": 0, "dev": 0}
    multiset_mismatch = 0
    for split in ("train", "dev"):
        for session in selected[split]:
            frames, _times, _hashes = _session_frames(target_dataset, manifest, session)
            available = len(frames) - config.sequence_frames
            if available <= 0:
                continue
            starts = []
            for ordinal in range(config.windows_per_session):
                digest = _sha(f"{contract_sha}:{session}:{ordinal}".encode())
                start = int(digest[:16], 16) % available
                while start in starts:
                    start = (start + config.sequence_frames) % available
                starts.append(start)
                original = frames[start : start + config.sequence_frames]
                shuffled = middle_shuffle(original, int(digest[16:32], 16))
                original_hashes = sorted(_sha(frame.tobytes()) for frame in original)
                shuffled_hashes = sorted(_sha(frame.tobytes()) for frame in shuffled)
                multiset_mismatch += int(original_hashes != shuffled_hashes)
                anonymous = _sha(f"p0-temporal:{contract_sha}:{session}:{ordinal}".encode())
                pair = [(original, 0), (shuffled, 1)]
                if int(anonymous[-1], 16) % 2:
                    pair.reverse()
                for clip, label in pair:
                    clips[split].append(clip)
                    labels[split].append(label)
                    session_ids[split].append(anonymous)
                pair_counts[split] += 1
    ordinal_accuracy = {}
    for split in ("train", "dev"):
        predictions = np.tile([0, 1], pair_counts[split])
        ordinal_accuracy[split] = float(
            (predictions == np.asarray(labels[split], dtype=np.int64)).mean()
        )
    gate = cast(dict[str, object], contract["gate"])
    checks = {
        "train_pairs": pair_counts["train"] >= config.minimum_train_pairs,
        "dev_pairs": pair_counts["dev"] >= config.minimum_dev_pairs,
        "dev_ordinal": ordinal_accuracy["dev"] <= config.maximum_dev_ordinal_accuracy,
        "frame_multiset": multiset_mismatch
        <= int(cast(int, gate["maximum_pair_frame_multiset_mismatch"])),
        "split_disjoint": not (set(session_ids["train"]) & set(session_ids["dev"])),
    }
    if output_dir.exists() or output_dir.is_symlink():
        raise P0TemporalError("P0 temporal output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        shard_hashes = {}
        for split in ("train", "dev"):
            path = staging / f"{split}.npz"
            np.savez_compressed(
                path,
                rgb_sequence=np.stack(clips[split]),
                label=np.asarray(labels[split], dtype=np.int8),
                session_id=np.asarray(session_ids[split], dtype="<U64"),
            )
            shard_hashes[split] = _sha(path.read_bytes())
        payload: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "status": (
                "P0_TEMPORAL_DATA_PASSED" if all(checks.values()) else "P0_TEMPORAL_DATA_FAILED"
            ),
            "contract_sha256": contract_sha,
            "target_manifest_sha256": manifest["manifest_sha256"],
            "selected_sessions": {
                "train": config.train_sessions,
                "dev": config.dev_sessions,
            },
            "pair_counts": pair_counts,
            "ordinal_accuracy": ordinal_accuracy,
            "frame_multiset_mismatch": multiset_mismatch,
            "checks": checks,
            "shard_sha256": shard_hashes,
            "model_input_fields": ["rgb_sequence"],
            "video_test_opened": False,
            "p0_initialization_allowed": False,
            "reward_allowed": False,
            "promotion_allowed": False,
        }
        payload["report_sha256"] = _sha(_canonical(payload))
        (staging / "report.json").write_bytes(_canonical(payload) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0 chronological/middle-shuffled data")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(materialize(args.contract, args.target_dataset, args.output_dir), sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
