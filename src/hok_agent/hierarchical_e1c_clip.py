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

CONTRACT_SCHEMA = "hok-agent-hierarchical-event-e1c-clip-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1c-clip-report-v1"
TARGET_MANIFEST_SCHEMA = "hok-agent-v5-manifest-v2"
LABELS = ("FAR_NONTERMINAL", "NEAR_NONTERMINAL", "TERMINAL_TRANSITION_CANDIDATE")


class E1cClipError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ClipConfig:
    sequence_frames: int
    windows_seconds_before_anchor: tuple[tuple[int, int], ...]
    maximum_target_gap_ms: int
    minimum_complete_train_triplets: int
    minimum_complete_dev_triplets: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_clip_contract(path: Path) -> tuple[ClipConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise E1cClipError("E1c clip contract schema differs")
    anchor = cast(dict[str, object], payload["anchor_binding"])
    clip = cast(dict[str, object], payload["clip"])
    model = cast(dict[str, object], payload["model_input"])
    gate = cast(dict[str, object], payload["materialization_gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("status") != "frozen_anchor_preceding_clip_materialization"
        or anchor.get("anchor_frame_allowed_in_clip") is not False
        or model.get("fields") != ["rgb_sequence"]
        or any(
            model.get(field) is not False
            for field in (
                "timestamp_allowed",
                "relative_offset_allowed",
                "anchor_frame_allowed",
                "session_id_allowed",
                "ocr_allowed",
            )
        )
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
        or claim.get("video_test_allowed") is not False
    ):
        raise E1cClipError("E1c clip boundary differs")
    config = ClipConfig(
        sequence_frames=int(cast(int, clip["sequence_frames"])),
        windows_seconds_before_anchor=(
            (
                int(cast(int, clip["far_negative_start_seconds_before_anchor"])),
                int(cast(int, clip["far_negative_end_seconds_before_anchor"])),
            ),
            (
                int(cast(int, clip["near_negative_start_seconds_before_anchor"])),
                int(cast(int, clip["near_negative_end_seconds_before_anchor"])),
            ),
            (
                int(cast(int, clip["positive_start_seconds_before_anchor"])),
                int(cast(int, clip["positive_end_seconds_before_anchor"])),
            ),
        ),
        maximum_target_gap_ms=int(cast(int, clip["maximum_target_gap_ms"])),
        minimum_complete_train_triplets=int(cast(int, gate["minimum_complete_train_triplets"])),
        minimum_complete_dev_triplets=int(cast(int, gate["minimum_complete_dev_triplets"])),
    )
    if config.sequence_frames <= 1 or any(
        start <= end for start, end in config.windows_seconds_before_anchor
    ):
        raise E1cClipError("E1c clip windows are invalid")
    return config, payload, _sha(_canonical(payload))


def select_window_indices(
    timestamps_ms: np.ndarray,
    anchor_ms: int,
    start_seconds_before: int,
    end_seconds_before: int,
    sequence_frames: int,
    maximum_gap_ms: int,
) -> np.ndarray:
    targets = np.linspace(
        anchor_ms - start_seconds_before * 1000,
        anchor_ms - end_seconds_before * 1000,
        sequence_frames,
    )
    positions = np.searchsorted(timestamps_ms, targets)
    positions = np.clip(positions, 1, len(timestamps_ms) - 1)
    left = positions - 1
    choose_left = np.abs(timestamps_ms[left] - targets) <= np.abs(
        timestamps_ms[positions] - targets
    )
    selected = np.where(choose_left, left, positions).astype(np.int64)
    gaps = np.abs(timestamps_ms[selected] - targets)
    if len(set(selected.tolist())) != sequence_frames or float(gaps.max()) > maximum_gap_ms:
        raise E1cClipError("E1c clip lacks continuous target frames")
    return selected


def _load_manifest(path: Path) -> dict[str, object]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    supplied = payload.get("manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload.get("schema_version") != TARGET_MANIFEST_SCHEMA or supplied != _sha(
        _canonical(unsigned)
    ):
        raise E1cClipError("E1c target manifest is invalid")
    return payload


def _selected_sessions(
    manifest: dict[str, object],
    anchor_report: dict[str, object],
    anchor_contract_sha256: str,
) -> dict[str, str]:
    anchored = {
        cast(str, row["session_id"]): cast(str, row["split"])
        for row in cast(list[dict[str, object]], anchor_report["rows"])
        if bool(row["anchored"])
    }
    selected: dict[str, str] = {}
    for row in cast(list[dict[str, object]], manifest["sessions"]):
        session_hash = cast(str, row["session_hash"])
        anonymous = _sha(f"e1c-anchor:{anchor_contract_sha256}:{session_hash}".encode())
        if anonymous in anchored:
            split = cast(str, row["split"])
            if split not in {"train", "dev"} or split != anchored[anonymous]:
                raise E1cClipError("E1c anchor split binding differs")
            selected[session_hash] = split
    if len(selected) != len(anchored):
        raise E1cClipError("E1c anchor sessions are missing from target manifest")
    return selected


def _session_frames(
    dataset: Path,
    manifest: dict[str, object],
    session_hash: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shard_rows = [
        row
        for row in cast(list[dict[str, object]], manifest["shards"])
        if session_hash in cast(list[str], row["session_hashes"])
    ]
    frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    hashes: list[np.ndarray] = []
    for row in shard_rows[-4:]:
        path = dataset / "shards" / Path(cast(str, row["path"])).name
        with np.load(path, allow_pickle=False) as data:
            mask = data["session_hash"] == session_hash
            frames.append(data["frames"][mask])
            times.append(data["timestamp_ms"][mask])
            hashes.append(data["frame_hash"][mask])
    if not frames:
        raise E1cClipError("E1c session has no target frames")
    rgb = np.concatenate(frames)
    timestamps = np.concatenate(times)
    frame_hashes = np.concatenate(hashes)
    order = np.argsort(timestamps)
    return rgb[order], timestamps[order], frame_hashes[order]


def materialize_clips(
    contract_path: Path,
    anchor_contract_path: Path,
    anchor_report_path: Path,
    target_dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha256 = load_clip_contract(contract_path)
    _anchor_config, _anchor_contract, anchor_contract_sha256 = load_anchor_contract(
        anchor_contract_path
    )
    anchor_report = load_anchor_report(anchor_report_path, anchor_contract_path)
    binding = cast(dict[str, object], contract["anchor_binding"])
    if binding["report_sha256"] != anchor_report["report_sha256"]:
        raise E1cClipError("E1c anchor report binding differs")
    manifest = _load_manifest(target_dataset / "manifest.json")
    selected = _selected_sessions(manifest, anchor_report, anchor_contract_sha256)
    split_clips: dict[str, list[np.ndarray]] = {"train": [], "dev": []}
    split_labels: dict[str, list[int]] = {"train": [], "dev": []}
    split_sessions: dict[str, list[str]] = {"train": [], "dev": []}
    incomplete: list[str] = []
    anchor_overlap = 0
    for sequence, (session_hash, split) in enumerate(sorted(selected.items()), 1):
        frames, timestamps, frame_hashes = _session_frames(target_dataset, manifest, session_hash)
        anchor_ms = int(timestamps[-1])
        anchor_hash = str(frame_hashes[-1])
        session_clips: list[np.ndarray] = []
        try:
            for start, end in config.windows_seconds_before_anchor:
                indices = select_window_indices(
                    timestamps,
                    anchor_ms,
                    start,
                    end,
                    config.sequence_frames,
                    config.maximum_target_gap_ms,
                )
                anchor_overlap += int(anchor_hash in set(frame_hashes[indices].tolist()))
                session_clips.append(frames[indices])
        except E1cClipError:
            incomplete.append(_sha(f"e1c-clip:{contract_sha256}:{session_hash}".encode()))
            continue
        anonymous = _sha(f"e1c-clip:{contract_sha256}:{session_hash}".encode())
        for label, clip in enumerate(session_clips):
            split_clips[split].append(clip)
            split_labels[split].append(label)
            split_sessions[split].append(anonymous)
        if sequence % 10 == 0 or sequence == len(selected):
            print(
                json.dumps(
                    {
                        "clip_sessions_complete": sequence,
                        "incomplete": len(incomplete),
                        "total": len(selected),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    counts = {split: len(split_clips[split]) // 3 for split in ("train", "dev")}
    gate = cast(dict[str, object], contract["materialization_gate"])
    checks = {
        "train_complete_triplets": counts["train"] >= config.minimum_complete_train_triplets,
        "dev_complete_triplets": counts["dev"] >= config.minimum_complete_dev_triplets,
        "anchor_frame_overlap": anchor_overlap
        <= int(cast(int, gate["maximum_anchor_frame_overlap"])),
        "cross_split_sessions": not (set(split_sessions["train"]) & set(split_sessions["dev"])),
    }
    if output_dir.exists() or output_dir.is_symlink():
        raise E1cClipError("E1c clip output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        shard_hashes: dict[str, str] = {}
        for split in ("train", "dev"):
            shard = staging / f"{split}.npz"
            np.savez_compressed(
                shard,
                rgb_sequence=np.stack(split_clips[split]),
                label=np.asarray(split_labels[split], dtype=np.int8),
                session_id=np.asarray(split_sessions[split], dtype="<U64"),
            )
            shard_hashes[split] = _sha(shard.read_bytes())
        payload: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "status": (
                "E1C_CLIP_MATERIALIZATION_PASSED"
                if all(checks.values())
                else "E1C_CLIP_MATERIALIZATION_FAILED"
            ),
            "contract_sha256": contract_sha256,
            "anchor_report_sha256": anchor_report["report_sha256"],
            "target_manifest_sha256": manifest["manifest_sha256"],
            "complete_triplets": counts,
            "incomplete_session_ids": sorted(incomplete),
            "anchor_frame_overlap": anchor_overlap,
            "checks": checks,
            "shard_sha256": shard_hashes,
            "model_input_fields": ["rgb_sequence"],
            "video_test_opened": False,
            "semantic_accuracy_verified": False,
            "win_loss_verified": False,
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


def load_clip_report(report_path: Path, contract_path: Path) -> dict[str, object]:
    if report_path.is_symlink() or not report_path.is_file():
        raise E1cClipError("E1c clip report must be a regular file")
    data = report_path.read_bytes()
    payload = cast(dict[str, object], json.loads(data))
    _config, _contract, contract_sha256 = load_clip_contract(contract_path)
    supplied = payload.get("report_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    if (
        payload.get("schema_version") != REPORT_SCHEMA
        or payload.get("contract_sha256") != contract_sha256
        or supplied != _sha(_canonical(unsigned))
        or data != _canonical(payload) + b"\n"
        or payload.get("model_input_fields") != ["rgb_sequence"]
        or payload.get("video_test_opened") is not False
        or payload.get("reward_allowed") is not False
        or payload.get("promotion_allowed") is not False
    ):
        raise E1cClipError("E1c clip report is invalid")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline E1c dynamic clip materializer")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--anchor-contract", type=Path, required=True)
    parser.add_argument("--anchor-report", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = materialize_clips(
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
