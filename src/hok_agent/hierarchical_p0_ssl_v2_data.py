from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from hok_agent.hierarchical_e1c_clip import _load_manifest

SCHEMA = "hok-agent-hierarchical-p0-temporal-ssl-v2-data-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p0-temporal-ssl-v2-data-report-v1"
INDEX_SCHEMA = "hok-agent-hierarchical-p0-temporal-ssl-v2-index-v1"


class P0SSLV2DataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DataConfig:
    required_sessions: int
    windows_per_session: int
    maximum_shards_per_session: int
    sequence_frames: int
    frame_stride: int
    maximum_adjacent_gap_ms: int
    minimum_sessions: int
    minimum_pairs: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[DataConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    selection = cast(dict[str, object], payload["selection"])
    window = cast(dict[str, object], payload["window"])
    model = cast(dict[str, object], payload["model_input"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_train_only_temporal_index"
        or selection.get("split") != "train"
        or selection.get("video_dev_allowed") is not False
        or selection.get("video_test_allowed") is not False
        or window.get("first_frame_fixed") is not True
        or window.get("last_frame_fixed") is not True
        or window.get("same_frame_multiset") is not True
        or model.get("fields") != ["rgb_sequence"]
        or any(
            model.get(field) is not False
            for field in ("timestamp_allowed", "session_id_allowed", "shard_reference_allowed")
        )
        or claim.get("p0_initialization_allowed") is not False
        or claim.get("policy_training_allowed") is not False
        or claim.get("reward_allowed") is not False
    ):
        raise P0SSLV2DataError("P0 SSL v2 data boundary differs")
    config = DataConfig(
        int(cast(int, selection["required_sessions"])),
        int(cast(int, selection["windows_per_session"])),
        int(cast(int, selection["maximum_shards_per_session"])),
        int(cast(int, window["sequence_frames"])),
        int(cast(int, window["frame_stride"])),
        int(cast(int, window["maximum_adjacent_gap_ms"])),
        int(cast(int, gate["minimum_sessions"])),
        int(cast(int, gate["minimum_pairs"])),
    )
    if min(
        config.required_sessions,
        config.windows_per_session,
        config.maximum_shards_per_session,
        config.sequence_frames,
        config.frame_stride,
    ) <= 0:
        raise P0SSLV2DataError("P0 SSL v2 data dimensions must be positive")
    return config, payload, _sha(_canonical(payload))


def middle_permutation(length: int, seed: int) -> np.ndarray:
    if length < 3:
        raise P0SSLV2DataError("temporal sequence requires at least three frames")
    order = np.arange(length, dtype=np.int64)
    generator = np.random.default_rng(seed)
    order[1:-1] = generator.permutation(order[1:-1])
    if np.array_equal(order[1:-1], np.arange(1, length - 1)):
        order[1:-1] = np.roll(order[1:-1], 1)
    return order


def selected_train_sessions(manifest: dict[str, object]) -> tuple[str, ...]:
    sessions = cast(list[dict[str, object]], manifest["sessions"])
    return tuple(
        sorted(
            cast(str, row["session_hash"]) for row in sessions if row["split"] == "train"
        )
    )


def _selected_shards(
    rows: list[dict[str, object]], count: int, contract_sha: str, session: str
) -> list[dict[str, object]]:
    if len(rows) <= count:
        return rows
    boundaries = np.linspace(0, len(rows), count + 1, dtype=np.int64)
    selected = []
    for partition, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        width = int(stop - start)
        digest = _sha(f"{contract_sha}:{session}:shard:{partition}".encode())
        selected.append(rows[int(start) + int(digest[:16], 16) % width])
    return selected


def _read_shard(
    dataset: Path, row: dict[str, object], session: str
) -> tuple[np.ndarray, np.ndarray]:
    path = dataset / "shards" / Path(cast(str, row["path"])).name
    data = path.read_bytes()
    if _sha(data) != row["sha256"]:
        raise P0SSLV2DataError("P0 SSL v2 shard hash differs")
    with np.load(io.BytesIO(data), allow_pickle=False) as shard:
        if set(shard["session_hash"].tolist()) != {session} or set(shard["split"].tolist()) != {
            "train"
        }:
            raise P0SSLV2DataError("P0 SSL v2 opened a non-train shard")
        return shard["timestamp_ms"].copy(), shard["frame_hash"].copy()


def _valid_starts(
    timestamps: np.ndarray,
    frame_hashes: np.ndarray,
    config: DataConfig,
) -> list[int]:
    offsets = np.arange(config.sequence_frames) * config.frame_stride
    limit = len(timestamps) - int(offsets[-1])
    valid = []
    for start in range(max(0, limit)):
        indices = start + offsets
        times = timestamps[indices]
        hashes = frame_hashes[indices]
        if int(np.diff(times).max(initial=0)) <= config.maximum_adjacent_gap_ms and len(
            set(hashes.tolist())
        ) == config.sequence_frames:
            valid.append(start)
    return valid


def materialize_index(
    contract_path: Path, target_dataset: Path, output_dir: Path
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    manifest = _load_manifest(target_dataset / "manifest.json")
    if manifest["manifest_sha256"] != contract["target_manifest_sha256"]:
        raise P0SSLV2DataError("P0 SSL v2 manifest binding differs")
    sessions = selected_train_sessions(manifest)
    if len(sessions) != config.required_sessions:
        raise P0SSLV2DataError("P0 SSL v2 train-session count differs")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cast(list[dict[str, object]], manifest["shards"]):
        if row["split"] == "train":
            session_hashes = cast(list[str], row["session_hashes"])
            if len(session_hashes) != 1:
                raise P0SSLV2DataError("P0 SSL v2 requires one train session per shard")
            grouped[session_hashes[0]].append(row)
    windows: list[dict[str, object]] = []
    invalid_windows = 0
    for session in sessions:
        rows = [
            row
            for row in grouped[session]
            if int(cast(int, row["row_count"]))
            >= 1 + (config.sequence_frames - 1) * config.frame_stride
        ]
        chosen = _selected_shards(
            rows,
            min(config.maximum_shards_per_session, len(rows)),
            contract_sha,
            session,
        )
        if not chosen:
            raise P0SSLV2DataError("P0 SSL v2 session has no eligible shard")
        per_shard = [config.windows_per_session // len(chosen)] * len(chosen)
        for index in range(config.windows_per_session % len(chosen)):
            per_shard[index] += 1
        anonymous_session = _sha(f"p0-ssl-v2:{contract_sha}:{session}".encode())
        for shard_ordinal, (row, required) in enumerate(
            zip(chosen, per_shard, strict=True)
        ):
            timestamps, frame_hashes = _read_shard(target_dataset, row, session)
            starts = _valid_starts(timestamps, frame_hashes, config)
            if len(starts) < required:
                invalid_windows += required - len(starts)
                continue
            generator = np.random.default_rng(
                int(_sha(f"{contract_sha}:{session}:starts:{shard_ordinal}".encode())[:16], 16)
            )
            offsets = np.arange(config.sequence_frames) * config.frame_stride
            selected_starts = generator.choice(starts, required, replace=False)
            for local_ordinal, start in enumerate(selected_starts):
                ordinal = len(windows)
                digest = _sha(f"{contract_sha}:{session}:{shard_ordinal}:{local_ordinal}".encode())
                permutation = middle_permutation(config.sequence_frames, int(digest[:16], 16))
                windows.append(
                    {
                        "window_id": _sha(f"p0-ssl-v2-window:{digest}".encode()),
                        "session_id": anonymous_session,
                        "shard": Path(cast(str, row["path"])).name,
                        "shard_sha256": row["sha256"],
                        "frame_indices": (int(start) + offsets).tolist(),
                        "middle_permutation": permutation.tolist(),
                        "pair_reversed": bool(int(digest[-1], 16) % 2),
                        "ordinal": ordinal,
                    }
                )
    identities = {
        (cast(str, row["shard"]), tuple(cast(list[int], row["frame_indices"]))) for row in windows
    }
    duplicate_windows = len(windows) - len(identities)
    checks = {
        "sessions": len({cast(str, row["session_id"]) for row in windows})
        >= config.minimum_sessions,
        "pairs": len(windows) >= config.minimum_pairs,
        "invalid_windows": invalid_windows == 0,
        "duplicate_windows": duplicate_windows == 0,
        "train_only": True,
    }
    passed = all(checks.values())
    index_payload: dict[str, object] = {
        "schema_version": INDEX_SCHEMA,
        "contract_sha256": contract_sha,
        "target_manifest_sha256": manifest["manifest_sha256"],
        "windows": windows,
    }
    index_payload["index_sha256"] = _sha(_canonical(index_payload))
    if output_dir.exists() or output_dir.is_symlink():
        raise P0SSLV2DataError("P0 SSL v2 output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / "index.json").write_bytes(_canonical(index_payload) + b"\n")
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "status": (
                "P0_TEMPORAL_SSL_V2_DATA_PASSED"
                if passed
                else "P0_TEMPORAL_SSL_V2_DATA_FAILED"
            ),
            "contract_sha256": contract_sha,
            "target_manifest_sha256": manifest["manifest_sha256"],
            "train_sessions": len({cast(str, row["session_id"]) for row in windows}),
            "pair_count": len(windows),
            "invalid_windows": invalid_windows,
            "duplicate_windows": duplicate_windows,
            "checks": checks,
            "index_sha256": index_payload["index_sha256"],
            "model_input_fields": ["rgb_sequence"],
            "video_dev_opened": False,
            "video_test_opened": False,
            "ssl_training_allowed": passed,
            "p0_initialization_allowed": False,
            "policy_training_allowed": False,
            "reward_allowed": False,
            "promotion_allowed": False,
        }
        report["report_sha256"] = _sha(_canonical(report))
        (staging / "report.json").write_bytes(_canonical(report) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0 temporal SSL v2 train-only index")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize_index(args.contract, args.target_dataset, args.output_dir),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
