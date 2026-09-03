from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import av
import numpy as np

from hok_agent.mobile_testbed import (
    MOVEMENTS,
    ObservationROIs,
    _movement_teacher_contract,
    load_observation_rois,
    movement_teacher_decision,
)
from hok_agent.operation_policy import _resize_crops
from hok_agent.pre_ingest import _candidate, _scan

SCHEMA = "hok-agent-hierarchical-p1-movement-teacher-audit-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p1-movement-teacher-audit-report-v1"
DIRECTIONS = MOVEMENTS[1:]


class P1MovementAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AuditConfig:
    segment_fractions: tuple[float, ...]
    frames_per_segment: int
    sample_period_ms: int
    confirmation_frames: int
    minimum_detection_coverage: float
    minimum_detected_session_fraction: float
    minimum_samples: tuple[int, int]
    minimum_sessions: tuple[int, int]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[AuditConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    sampling = cast(dict[str, object], payload["sampling"])
    labels = cast(dict[str, object], payload["labels"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_raw_video_minimap_teacher_audit_repair1"
        or sampling.get("splits") != ["train", "dev"]
        or sampling.get("video_test_allowed") is not False
        or labels.get("vocabulary") != list(DIRECTIONS)
        or labels.get("meaning") != "teacher_recommended_direction_not_observed_human_action"
        or labels.get("wait_label_allowed") is not False
        or labels.get("human_labels_used") is not False
        or labels.get("semantic_accuracy_verified") is not False
        or claim.get("policy_bundle_assembly_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("device_input_allowed") is not False
    ):
        raise P1MovementAuditError("P1 Movement audit boundary differs")
    repair = cast(dict[str, object], payload["repair_history"])
    if (
        repair.get("repairs_allowed") != 1
        or repair.get("repairs_used") != 1
        or repair.get("repair") != "apply_mp4_display_matrix_before_roi_crop"
        or repair.get("sampling_teacher_or_gate_changed") is not False
    ):
        raise P1MovementAuditError("P1 Movement audit repair differs")
    fractions = tuple(float(value) for value in cast(list[float], sampling["segment_fractions"]))
    config = AuditConfig(
        fractions,
        int(cast(int, sampling["frames_per_segment"])),
        int(cast(int, sampling["sample_period_ms"])),
        int(cast(int, sampling["direction_confirmation_frames"])),
        float(cast(float, gate["minimum_detection_coverage"])),
        float(cast(float, gate["minimum_detected_session_fraction"])),
        (
            int(cast(int, gate["minimum_stable_train_samples_per_direction"])),
            int(cast(int, gate["minimum_stable_dev_samples_per_direction"])),
        ),
        (
            int(cast(int, gate["minimum_train_sessions_per_direction"])),
            int(cast(int, gate["minimum_dev_sessions_per_direction"])),
        ),
    )
    if (
        len(fractions) != sampling.get("segments_per_session")
        or any(not 0 < value < 1 for value in fractions)
        or min(config.frames_per_segment, config.sample_period_ms, config.confirmation_frames) <= 0
    ):
        raise P1MovementAuditError("P1 Movement audit sampling differs")
    return config, payload, _sha(_canonical(payload))


def stable_directions(values: list[str | None], confirmation_frames: int) -> list[str | None]:
    stable: list[str | None] = []
    pending: str | None = None
    count = 0
    for value in values:
        if value is None:
            pending, count = None, 0
            stable.append(None)
            continue
        if value == pending:
            count += 1
        else:
            pending, count = value, 1
        stable.append(value if count >= confirmation_frames else None)
    return stable


def display_rotation(matrix: bytes) -> int:
    if len(matrix) != 36:
        raise P1MovementAuditError("MP4 display matrix size differs")
    values = np.frombuffer(matrix, dtype="<i4")
    angle = round(float(np.degrees(-np.arctan2(values[1], values[0])))) % 360
    if angle not in {0, 90, 180, 270}:
        raise P1MovementAuditError("MP4 display rotation differs")
    return angle


def _frame_rotation(frame: av.VideoFrame) -> int:
    side_data = cast(Any, frame.side_data)
    matrices = [
        memoryview(value).tobytes()
        for value in side_data
        if getattr(value.type, "name", "") == "DISPLAYMATRIX"
    ]
    if len(matrices) > 1:
        raise P1MovementAuditError("multiple MP4 display matrices")
    return display_rotation(matrices[0]) if matrices else 0


def _decode_segment(path: Path, fraction: float, config: AuditConfig) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if container.duration is None or container.duration <= 0:
            return frames
        stream = container.streams.video[0]
        target_seconds = container.duration / 1_000_000 * fraction
        container.seek(int(target_seconds * 1_000_000), backward=True)
        next_seconds = target_seconds
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            seconds = float(frame.pts * frame.time_base)
            if seconds + 1e-9 < next_seconds:
                continue
            image = frame.to_ndarray(format="rgb24")
            image = np.rot90(image, k=-(_frame_rotation(frame) // 90))
            if image.shape[1] <= image.shape[0]:
                return []
            frames.append(image)
            next_seconds = seconds + config.sample_period_ms / 1000
            if len(frames) == config.frames_per_segment:
                break
    return frames


def _session_labels(
    path: Path,
    config: AuditConfig,
    teacher: dict[str, object],
    rois: ObservationROIs,
) -> tuple[int, Counter[str], Counter[str], list[float]]:
    total = 0
    raw_counts: Counter[str] = Counter()
    stable_counts: Counter[str] = Counter()
    confidences: list[float] = []
    for fraction in config.segment_fractions:
        frames = _decode_segment(path, fraction, config)
        total += len(frames)
        if not frames:
            continue
        minimaps = _resize_crops(np.stack(frames), rois.minimap, rois)
        previous: tuple[float, float] | None = None
        raw: list[str | None] = []
        for frame in minimaps:
            decision = movement_teacher_decision(frame, teacher, previous)
            raw.append(None if decision is None else decision.movement)
            if decision is not None:
                raw_counts[decision.movement] += 1
                confidences.append(decision.confidence)
                previous = decision.player_yx
        stable_counts.update(
            value
            for value in stable_directions(raw, config.confirmation_frames)
            if value is not None
        )
    return total, raw_counts, stable_counts, confidences


def audit(
    contract_path: Path,
    video_root: Path,
    cohort_path: Path,
    pre_ingest_path: Path,
    teacher_contract_path: Path,
    observation_rois_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    cohort = cast(dict[str, object], json.loads(cohort_path.read_text()))
    pre_ingest = cast(dict[str, object], json.loads(pre_ingest_path.read_text()))
    if (
        cohort.get("cohort_sha256") != contract["cohort_sha256"]
        or pre_ingest.get("pre_ingest_sha256") != contract["pre_ingest_sha256"]
    ):
        raise P1MovementAuditError("P1 Movement source evidence differs")
    teacher, teacher_sha = _movement_teacher_contract(teacher_contract_path)
    if teacher_sha != contract["movement_teacher_contract_sha256"]:
        raise P1MovementAuditError("P1 Movement teacher contract differs")
    rois, roi_sha = load_observation_rois(observation_rois_path)
    paths = {_candidate(path, video_root).candidate_id: path for path in _scan(video_root)}
    splits = cast(dict[str, str], cohort["session_splits"])
    if set(paths) != set(splits):
        raise P1MovementAuditError("P1 Movement raw-video identity differs")
    counts = {split: Counter({name: 0 for name in DIRECTIONS}) for split in ("train", "dev")}
    supports = {split: Counter({name: 0 for name in DIRECTIONS}) for split in ("train", "dev")}
    detections = Counter({"train": 0, "dev": 0})
    sampled = Counter({"train": 0, "dev": 0})
    detected_sessions = Counter({"train": 0, "dev": 0})
    rows: list[dict[str, object]] = []
    for identity in sorted(paths):
        split = splits[identity]
        if split == "test":
            continue
        total, raw, stable, confidences = _session_labels(
            paths[identity], config, teacher, rois
        )
        sampled[split] += total
        detections[split] += sum(raw.values())
        if raw:
            detected_sessions[split] += 1
        for direction in DIRECTIONS:
            counts[split][direction] += stable[direction]
            supports[split][direction] += int(stable[direction] > 0)
        rows.append(
            {
                "session_id": _sha(f"p1-movement:{contract_sha}:{identity}".encode()),
                "split": split,
                "sampled_frames": total,
                "detections": sum(raw.values()),
                "stable_direction_counts": dict(stable),
                "mean_confidence": round(float(np.mean(confidences)), 8) if confidences else None,
            }
        )
    expected_sessions = Counter(value for value in splits.values() if value != "test")
    coverage = {
        split: detections[split] / sampled[split] if sampled[split] else 0.0
        for split in ("train", "dev")
    }
    session_fraction = {
        split: detected_sessions[split] / expected_sessions[split]
        for split in ("train", "dev")
    }
    checks = {
        "detection_coverage": all(
            coverage[split] >= config.minimum_detection_coverage for split in ("train", "dev")
        ),
        "detected_session_fraction": all(
            session_fraction[split] >= config.minimum_detected_session_fraction
            for split in ("train", "dev")
        ),
        "stable_direction_samples": all(
            counts[split][direction] >= config.minimum_samples[index]
            for index, split in enumerate(("train", "dev"))
            for direction in DIRECTIONS
        ),
        "direction_session_support": all(
            supports[split][direction] >= config.minimum_sessions[index]
            for index, split in enumerate(("train", "dev"))
            for direction in DIRECTIONS
        ),
        "split_counts": expected_sessions == Counter({"train": 103, "dev": 23}),
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "P1_MOVEMENT_TEACHER_AUDIT_PASSED"
            if passed
            else "P1_MOVEMENT_TEACHER_AUDIT_FAILED"
        ),
        "contract_sha256": contract_sha,
        "cohort_sha256": cohort["cohort_sha256"],
        "pre_ingest_sha256": pre_ingest["pre_ingest_sha256"],
        "movement_teacher_contract_sha256": teacher_sha,
        "observation_rois_sha256": roi_sha,
        "sampled_frames": dict(sampled),
        "detections": dict(detections),
        "detection_coverage": coverage,
        "detected_sessions": dict(detected_sessions),
        "detected_session_fraction": session_fraction,
        "stable_direction_counts": {split: dict(counts[split]) for split in ("train", "dev")},
        "direction_session_support": {split: dict(supports[split]) for split in ("train", "dev")},
        "sessions": rows,
        "checks": checks,
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "display_matrix_rotation_applied": True,
        "video_test_opened": False,
        "movement_head_training_allowed": passed,
        "policy_bundle_assembly_allowed": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "device_input_allowed": False,
    }
    report["report_sha256"] = _sha(_canonical(report))
    if output_dir.exists() or output_dir.is_symlink():
        raise P1MovementAuditError("P1 Movement audit output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / "report.json").write_bytes(_canonical(report) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 Movement raw-video teacher audit")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--pre-ingest", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
    parser.add_argument("--observation-rois", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            audit(
                args.contract,
                args.video_root,
                args.cohort,
                args.pre_ingest,
                args.teacher_contract,
                args.observation_rois,
                args.output_dir,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
