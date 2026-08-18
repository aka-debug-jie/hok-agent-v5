"""Basic-only deterministic offline candidate gate over owner-provided video-dev."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.mobile_testbed import (
    ABILITIES,
    GuardWatchdog,
    ScrcpyV4L2,
    _model_frame,
    _new_large_output,
    _open_device_guard,
    load_layout,
)
from hok_agent.t8 import (
    _canonical,
    _large_existing,
    _large_new,
    _load_v2_adapter,
    _read_object,
    _retrospective_content_box,
    _retrospective_load_session,
    _retrospective_target_index,
    _sha,
)
from hok_agent.t8_v4 import _normalize_teacher_frame, _self_hash
from hok_agent.t8_v5 import (
    DATASET_SCHEMA as V5_DATASET_SCHEMA,
)
from hok_agent.t8_v5 import (
    DIAGNOSIS_SCHEMA as V5_DIAGNOSIS_SCHEMA,
)
from hok_agent.t8_v5 import (
    _button_box,
    _fractional_crop,
    _verified_dataset,
    verify_t8_v5_contract,
)

CONTRACT_SCHEMA: Final = "hok-agent-t8-basic-mvp-contract-v1"
REPLAY_SCHEMA: Final = "hok-agent-t8-basic-mvp-offline-replay-v1"
EVENT_SCHEMA: Final = "hok-agent-t8-basic-mvp-offline-event-v1"
SHADOW_CONTRACT_SCHEMA: Final = "hok-agent-t8-basic-mvp-shadow-contract-v1"
SHADOW_SCHEMA: Final = "hok-agent-t8-basic-mvp-read-only-shadow-v1"


class T8BasicMVPError(ValueError):
    pass


def verify_t8_basic_mvp_contract(path: Path) -> dict[str, object]:
    value = _read_object(path, "T8 Basic MVP contract is unreadable")
    if value.get("schema_version") != CONTRACT_SCHEMA or value.get("contract_sha256") != _self_hash(
        value, "contract_sha256"
    ):
        raise T8BasicMVPError("T8 Basic MVP contract identity is invalid")
    expected: dict[str, object] = {
        "source_component": "t8_v5_basic_correct_roi_linear",
        "source_overall_gate_required": False,
        "source_basic_gate_required": True,
        "split": "dev",
        "infer_hz": 5,
        "enemy_rule_probability_threshold": 0.8,
        "basic_model_probability_threshold": 0.8,
        "positive_confirmation_frames": 3,
        "maximum_confirmation_gap_ms": 300,
        "global_refractory_ms": 1500,
        "maximum_candidates_per_minute": 10,
        "maximum_candidates_per_session": 20,
        "maximum_consecutive_identical_candidates": 3,
        "minimum_total_candidates": 5,
        "minimum_screen_mean": 8.0,
        "minimum_screen_standard_deviation": 5.0,
        "frozen_reference_ms": 1000,
        "video_test_access_allowed": False,
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "control_output": False,
        "device_input_allowed": False,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise T8BasicMVPError("T8 Basic MVP frozen contract values differ")
    return {
        "schema_version": "hok-agent-t8-basic-mvp-contract-check-v1",
        "status": "PASSED",
        "contract_sha256": value["contract_sha256"],
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "video_test_accessed": False,
        "control_output": False,
        "device_input_allowed": False,
    }


def verify_t8_basic_mvp_shadow_contract(path: Path, base_contract_sha256: str) -> dict[str, object]:
    value = _read_object(path, "T8 Basic MVP Shadow contract is unreadable")
    if (
        value.get("schema_version") != SHADOW_CONTRACT_SCHEMA
        or value.get("contract_sha256") != _self_hash(value, "contract_sha256")
        or value.get("base_contract_sha256") != base_contract_sha256
        or value.get("run_seconds") != 300.0
        or value.get("infer_hz") != 5
        or value.get("stream_fps") != 30
        or value.get("minimum_period_coverage") != 0.95
        or value.get("maximum_p95_decision_seconds") != 0.15
        or value.get("minimum_candidates") != 1
        or value.get("maximum_candidates_per_minute") != 10
        or value.get("input_commands_sent") != 0
        or value.get("control_output") is not False
        or value.get("device_input_allowed") is not False
    ):
        raise T8BasicMVPError("T8 Basic MVP Shadow contract differs")
    return value


def _component_identity(
    *,
    feature_root: Path,
    v5_contract: Path,
    training_report: Path,
    model_path: Path,
    adapter_checkpoint: Path,
    layout_sha256: str,
) -> tuple[dict[str, torch.Tensor], Mapping[str, torch.Tensor], dict[str, object]]:
    v5_checked = verify_t8_v5_contract(v5_contract)
    features = _large_existing(feature_root)
    manifest = _verified_dataset(features, cast(str, v5_checked["experiment_sha256"]))
    report_path = _large_existing(training_report)
    report = _read_object(report_path, "T8-v5 ROI training report is unreadable")
    model = _large_existing(model_path)
    adapter = _large_existing(adapter_checkpoint)
    gates = report.get("formal_head_gates")
    decision = report.get("decision")
    basic = gates.get("basic_attack_button_visual_enabled") if isinstance(gates, dict) else None
    if (
        report.get("schema_version") != V5_DIAGNOSIS_SCHEMA
        or report.get("dataset_manifest_sha256") != manifest.get("manifest_sha256")
        or report.get("model_sha256") != _sha(model)
        or not isinstance(basic, dict)
        or basic.get("passed") is not True
        or not isinstance(decision, dict)
        or decision.get("formal_heads_passed") is not False
        or decision.get("promotion_allowed") is not False
        or report.get("semantic_accuracy_verified") is not False
        or report.get("control_output") is not False
        or manifest.get("schema_version") != V5_DATASET_SCHEMA
        or manifest.get("adapter_sha256") != _sha(adapter)
        or manifest.get("layout_sha256") != layout_sha256
    ):
        raise T8BasicMVPError("T8 Basic MVP source component identity is not admissible")
    state = load_file(model, device="cpu")
    required = {
        "heads.1.weight": (1, 512),
        "heads.1.bias": (1,),
    }
    if any(key not in state or tuple(state[key].shape) != shape for key, shape in required.items()):
        raise T8BasicMVPError("T8 Basic MVP basic head tensors differ")
    encoder_state, adapter_meta = _load_v2_adapter(adapter, torch.device("cpu"))
    return (
        state,
        encoder_state,
        {
            "feature_manifest_sha256": manifest["manifest_sha256"],
            "training_report_sha256": _sha(report_path),
            "model_sha256": _sha(model),
            "adapter_sha256": _sha(adapter),
            "adapter_source_sha256": adapter_meta.get("v5_source_model_sha256"),
            "basic_gate": basic,
        },
    )


def _sample_indices(timestamps: np.ndarray, infer_hz: int) -> np.ndarray:
    if timestamps.ndim != 1 or timestamps.dtype != np.int64 or len(timestamps) == 0:
        raise T8BasicMVPError("T8 Basic MVP source timestamps are invalid")
    step = 1000.0 / infer_hz
    due = np.arange(float(timestamps[0]), float(timestamps[-1]) + 1.0, step)
    indices = np.searchsorted(timestamps, due, side="left")
    indices = indices[indices < len(timestamps)]
    return np.unique(indices).astype(np.int64)


def _screen_valid(frame: np.ndarray, minimum_mean: float, minimum_std: float) -> bool:
    return bool(float(frame.mean()) >= minimum_mean and float(frame.std()) >= minimum_std)


def _enemy_probability(frames: np.ndarray) -> np.ndarray:
    scene = frames[:, 8:108, 15:108].astype(np.int16)
    red, green, blue = (scene[..., index] for index in range(3))
    mask = (red > 140) & (red - green > 45) & (red - blue > 25)
    pixels = mask.sum(axis=(1, 2))
    row_max = mask.sum(axis=2).max(axis=1)
    strength = np.maximum(pixels / 400.0, row_max / 11.0)
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-4.0 * (strength - 1.0))))


def _basic_probabilities(
    *,
    frames: np.ndarray,
    basic_point: tuple[float, float],
    encoder: nn.Module,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            rois = np.stack(
                [
                    _fractional_crop(frame, _button_box(basic_point, 0.08))
                    for frame in frames[start : start + batch_size]
                ]
            )
            tensor = torch.from_numpy(rois).to(device).permute(0, 3, 1, 2).float().div(255.0)
            features = cast(torch.Tensor, encoder(tensor))
            logits = nn.functional.linear(features, weight.to(device), bias.to(device)).squeeze(1)
            values.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(values)


def _write_output(output: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        with (staging / "events.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary["events_sha256"] = _sha(staging / "events.jsonl")
        summary["summary_sha256"] = hashlib.sha256(_canonical(summary)).hexdigest()
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def run_t8_basic_mvp_offline_replay(
    *,
    contract_path: Path,
    v5_contract: Path,
    feature_root: Path,
    target_root: Path,
    training_report: Path,
    model_path: Path,
    adapter_checkpoint: Path,
    layout_path: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, object]:
    checked = verify_t8_basic_mvp_contract(contract_path)
    contract = _read_object(contract_path, "T8 Basic MVP contract is unreadable")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise T8BasicMVPError("T8 Basic MVP replay device is unavailable")
    if batch_size < 1:
        raise T8BasicMVPError("T8 Basic MVP replay batch size is invalid")
    layout, layout_sha = load_layout(layout_path)
    basic_point = layout.buttons[ABILITIES[1]]
    if basic_point is None:
        raise T8BasicMVPError("T8 Basic MVP layout lacks basic attack")
    state, encoder_state, component = _component_identity(
        feature_root=feature_root,
        v5_contract=v5_contract,
        training_report=training_report,
        model_path=model_path,
        adapter_checkpoint=adapter_checkpoint,
        layout_sha256=layout_sha,
    )
    target_device = torch.device(device)
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(target_device).eval()
    target_base, target_sha, sessions = _retrospective_target_index(target_root, "dev")
    if len(sessions) != 23:
        raise T8BasicMVPError("T8 Basic MVP requires 23 frozen video-dev sessions")
    feature_manifest = _read_object(
        _large_existing(feature_root) / "manifest.json",
        "T8 Basic MVP feature manifest is unreadable",
    )
    if target_sha != feature_manifest.get("target_manifest_sha256"):
        raise T8BasicMVPError("T8 Basic MVP target manifest differs")
    infer_hz = cast(int, contract["infer_hz"])
    rows: list[dict[str, object]] = []
    session_reports: list[dict[str, object]] = []
    total_candidates = 0
    invalid_candidates = 0
    rate_violations = 0
    latencies: list[float] = []
    for identity, source_rows in sessions:
        source_frames, timestamps, _hashes = _retrospective_load_session(
            target_base, "dev", identity, source_rows
        )
        canonical, _orientation, content_box = _retrospective_content_box(source_frames)
        indices = _sample_indices(timestamps, infer_hz)
        normalized = np.stack(
            [_normalize_teacher_frame(canonical[int(index)], content_box) for index in indices]
        )
        inference_start = time.monotonic()
        basic_probability = _basic_probabilities(
            frames=normalized,
            basic_point=basic_point,
            encoder=encoder,
            weight=state["heads.1.weight"],
            bias=state["heads.1.bias"],
            device=target_device,
            batch_size=batch_size,
        )
        enemy_probability = _enemy_probability(normalized)
        latencies.append((time.monotonic() - inference_start) / max(len(indices), 1))
        accepted_times: deque[int] = deque()
        stable_count = 0
        previous_time: int | None = None
        last_candidate = -(10**12)
        consecutive = 0
        session_candidates = 0
        for offset, frame_index in enumerate(indices):
            timestamp = int(timestamps[frame_index])
            history_index = max(
                int(
                    np.asarray(
                        np.searchsorted(
                            timestamps,
                            timestamp - cast(int, contract["frozen_reference_ms"]),
                            side="right",
                        )
                    ).item()
                )
                - 1,
                0,
            )
            valid = _screen_valid(
                normalized[offset],
                cast(float, contract["minimum_screen_mean"]),
                cast(float, contract["minimum_screen_standard_deviation"]),
            )
            frozen = bool(
                timestamp - int(timestamps[history_index])
                >= cast(int, contract["frozen_reference_ms"]) - 100
                and np.array_equal(canonical[frame_index], canonical[history_index])
            )
            raw_candidate = bool(
                valid
                and not frozen
                and enemy_probability[offset]
                >= cast(float, contract["enemy_rule_probability_threshold"])
                and basic_probability[offset]
                >= cast(float, contract["basic_model_probability_threshold"])
            )
            gap_ok = bool(
                previous_time is not None
                and timestamp - previous_time <= cast(int, contract["maximum_confirmation_gap_ms"])
            )
            stable_count = stable_count + 1 if raw_candidate and gap_ok else int(raw_candidate)
            if not raw_candidate:
                consecutive = 0
            while accepted_times and timestamp - accepted_times[0] >= 60_000:
                accepted_times.popleft()
            stable = stable_count >= cast(int, contract["positive_confirmation_frames"])
            refractory_ok = timestamp - last_candidate >= cast(
                int, contract["global_refractory_ms"]
            )
            rate_ok = len(accepted_times) < cast(int, contract["maximum_candidates_per_minute"])
            session_ok = session_candidates < cast(int, contract["maximum_candidates_per_session"])
            repetition_ok = consecutive < cast(
                int, contract["maximum_consecutive_identical_candidates"]
            )
            accepted = bool(stable and refractory_ok and rate_ok and session_ok and repetition_ok)
            if accepted:
                accepted_times.append(timestamp)
                last_candidate = timestamp
                session_candidates += 1
                total_candidates += 1
                consecutive += 1
            invalid_candidates += int(accepted and (not valid or frozen))
            rate_violations += int(accepted and not rate_ok)
            if not valid:
                reason = "INVALID_SCREEN"
            elif frozen:
                reason = "FROZEN_SCREEN"
            elif enemy_probability[offset] < cast(
                float, contract["enemy_rule_probability_threshold"]
            ):
                reason = "NO_CONFIDENT_ENEMY_CUE"
            elif basic_probability[offset] < cast(
                float, contract["basic_model_probability_threshold"]
            ):
                reason = "BASIC_NOT_CONFIDENTLY_ENABLED"
            elif not stable:
                reason = "PENDING_CONFIRMATION"
            elif not refractory_ok:
                reason = "REFRACTORY"
            elif not rate_ok:
                reason = "PER_MINUTE_LIMIT"
            elif not session_ok:
                reason = "SESSION_LIMIT"
            elif not repetition_ok:
                reason = "REPETITION_LIMIT"
            else:
                reason = "CANDIDATE_BASIC_ATTACK"
            rows.append(
                {
                    "schema_version": EVENT_SCHEMA,
                    "session_hash": identity,
                    "timestamp_ms": timestamp,
                    "frame_sha256": hashlib.sha256(normalized[offset].tobytes()).hexdigest(),
                    "screen_valid": valid,
                    "frozen_screen": frozen,
                    "enemy_rule_probability": round(float(enemy_probability[offset]), 8),
                    "basic_model_probability": round(float(basic_probability[offset]), 8),
                    "stable": stable,
                    "candidate": "candidate_basic_attack" if accepted else "wait",
                    "rejection_reason": reason,
                    "input_sent": False,
                }
            )
            previous_time = timestamp
        session_reports.append(
            {
                "session_hash": identity,
                "sampled_frames": len(indices),
                "duration_seconds": round(
                    (int(timestamps[indices[-1]]) - int(timestamps[indices[0]])) / 1000.0, 3
                ),
                "candidates": session_candidates,
            }
        )
    black = np.zeros((128, 128, 3), dtype=np.uint8)
    gray = np.full((128, 128, 3), 127, dtype=np.uint8)
    negative_controls_passed = not _screen_valid(
        black,
        cast(float, contract["minimum_screen_mean"]),
        cast(float, contract["minimum_screen_standard_deviation"]),
    ) and not _screen_valid(
        gray,
        cast(float, contract["minimum_screen_mean"]),
        cast(float, contract["minimum_screen_standard_deviation"]),
    )
    maximum_rate = max(
        (
            cast(int, report["candidates"])
            / max(cast(float, report["duration_seconds"]) / 60.0, 1e-9)
            for report in session_reports
        ),
        default=0.0,
    )
    strict = bool(
        total_candidates >= cast(int, contract["minimum_total_candidates"])
        and invalid_candidates == 0
        and rate_violations == 0
        and maximum_rate <= cast(int, contract["maximum_candidates_per_minute"])
        and negative_controls_passed
    )
    summary: dict[str, object] = {
        "schema_version": REPLAY_SCHEMA,
        "status": "PASSED" if strict else "FAILED",
        "strict_passed": strict,
        "contract_sha256": checked["contract_sha256"],
        **component,
        "target_manifest_sha256": target_sha,
        "layout_sha256": layout_sha,
        "sessions": len(sessions),
        "sampled_frames": len(rows),
        "total_candidates": total_candidates,
        "minimum_total_candidates": contract["minimum_total_candidates"],
        "invalid_screen_candidates": invalid_candidates,
        "rate_limit_violations": rate_violations,
        "maximum_session_candidate_rate_per_minute": round(maximum_rate, 8),
        "negative_controls_passed": negative_controls_passed,
        "mean_model_seconds_per_frame": round(float(np.mean(latencies)), 8),
        "session_qc": session_reports,
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "raw_rgb_persisted": False,
        "raw_video_or_source_paths_persisted": False,
        "video_test_accessed": False,
        "input_commands_sent": 0,
        "control_output": False,
        "shadow_allowed": strict,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    _write_output(output, rows, summary)
    return summary


def run_t8_basic_mvp_shadow(
    *,
    serial: str,
    video_node: Path,
    base_contract_path: Path,
    shadow_contract_path: Path,
    offline_summary: Path,
    v5_contract: Path,
    feature_root: Path,
    training_report: Path,
    model_path: Path,
    adapter_checkpoint: Path,
    layout_path: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, object]:
    base_checked = verify_t8_basic_mvp_contract(base_contract_path)
    base = _read_object(base_contract_path, "T8 Basic MVP contract is unreadable")
    shadow = verify_t8_basic_mvp_shadow_contract(
        shadow_contract_path, cast(str, base_checked["contract_sha256"])
    )
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise T8BasicMVPError("T8 Basic MVP Shadow device is unavailable")
    if batch_size < 1:
        raise T8BasicMVPError("T8 Basic MVP Shadow batch size is invalid")
    offline_path = _large_existing(offline_summary)
    offline = _read_object(offline_path, "T8 Basic MVP offline summary is unreadable")
    if (
        offline.get("schema_version") != REPLAY_SCHEMA
        or offline.get("status") != "PASSED"
        or offline.get("strict_passed") is not True
        or offline.get("contract_sha256") != base_checked["contract_sha256"]
        or offline.get("summary_sha256") != _self_hash(offline, "summary_sha256")
        or offline.get("shadow_allowed") is not True
        or offline.get("input_commands_sent") != 0
        or offline.get("control_output") is not False
        or offline.get("device_input_allowed") is not False
    ):
        raise T8BasicMVPError("T8 Basic MVP offline evidence does not admit Shadow")
    layout, layout_sha = load_layout(layout_path)
    basic_point = layout.buttons[ABILITIES[1]]
    if basic_point is None or offline.get("layout_sha256") != layout_sha:
        raise T8BasicMVPError("T8 Basic MVP Shadow layout differs")
    state, encoder_state, component = _component_identity(
        feature_root=feature_root,
        v5_contract=v5_contract,
        training_report=training_report,
        model_path=model_path,
        adapter_checkpoint=adapter_checkpoint,
        layout_sha256=layout_sha,
    )
    if any(offline.get(key) != component[key] for key in component):
        raise T8BasicMVPError("T8 Basic MVP Shadow component identity differs")
    guard = _open_device_guard(serial)
    if (guard.width, guard.height) != (layout.width, layout.height):
        raise T8BasicMVPError("T8 Basic MVP Shadow display differs from layout")
    output = _new_large_output(output_dir)
    target_device = torch.device(device)
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(target_device).eval()
    stream = ScrcpyV4L2(guard.serial, video_node, cast(int, shadow["stream_fps"]))
    watchdog = GuardWatchdog(guard)
    rows: list[dict[str, object]] = []
    delays: list[float] = []
    recent_candidates: deque[int] = deque()
    history: deque[tuple[int, str]] = deque()
    stable_count = 0
    previous_timestamp: int | None = None
    last_candidate = -(10**12)
    consecutive = 0
    invalid_candidates = 0
    failure: str | None = None
    run_seconds = cast(float, shadow["run_seconds"])
    infer_hz = cast(int, shadow["infer_hz"])
    expected_cycles = int(run_seconds * infer_hz)
    started = next_due = 0.0
    try:
        stream.start()
        watchdog.start()
        started = next_due = time.monotonic()
        while len(rows) < expected_cycles and time.monotonic() - started < run_seconds:
            now = time.monotonic()
            if now < next_due:
                time.sleep(min(next_due - now, 0.01))
                continue
            scheduled = next_due
            next_due += 1.0 / infer_hz
            watchdog.ensure_fresh()
            timestamp_ns, raw_frame = stream.frame_with_timestamp()
            frame = _model_frame(raw_frame)
            timestamp = timestamp_ns // 1_000_000
            decision_start = time.monotonic()
            valid = _screen_valid(
                frame,
                cast(float, base["minimum_screen_mean"]),
                cast(float, base["minimum_screen_standard_deviation"]),
            )
            frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
            history.append((timestamp, frame_hash))
            while history and timestamp - history[0][0] > cast(int, base["frozen_reference_ms"]):
                history.popleft()
            frozen = bool(
                history
                and timestamp - history[0][0] >= cast(int, base["frozen_reference_ms"]) - 100
                and history[0][1] == frame_hash
            )
            basic_probability = float(
                _basic_probabilities(
                    frames=frame[None],
                    basic_point=basic_point,
                    encoder=encoder,
                    weight=state["heads.1.weight"],
                    bias=state["heads.1.bias"],
                    device=target_device,
                    batch_size=batch_size,
                )[0]
            )
            enemy_probability = float(_enemy_probability(frame[None])[0])
            raw_candidate = bool(
                valid
                and not frozen
                and enemy_probability >= cast(float, base["enemy_rule_probability_threshold"])
                and basic_probability >= cast(float, base["basic_model_probability_threshold"])
            )
            gap_ok = bool(
                previous_timestamp is not None
                and timestamp - previous_timestamp <= cast(int, base["maximum_confirmation_gap_ms"])
            )
            stable_count = stable_count + 1 if raw_candidate and gap_ok else int(raw_candidate)
            if not raw_candidate:
                consecutive = 0
            while recent_candidates and timestamp - recent_candidates[0] >= 60_000:
                recent_candidates.popleft()
            stable = stable_count >= cast(int, base["positive_confirmation_frames"])
            refractory_ok = timestamp - last_candidate >= cast(int, base["global_refractory_ms"])
            rate_ok = len(recent_candidates) < cast(int, shadow["maximum_candidates_per_minute"])
            repetition_ok = consecutive < cast(
                int, base["maximum_consecutive_identical_candidates"]
            )
            accepted = bool(stable and refractory_ok and rate_ok and repetition_ok)
            if accepted:
                recent_candidates.append(timestamp)
                last_candidate = timestamp
                consecutive += 1
            invalid_candidates += int(accepted and (not valid or frozen))
            if not valid:
                reason = "INVALID_SCREEN"
            elif frozen:
                reason = "FROZEN_SCREEN"
            elif enemy_probability < cast(float, base["enemy_rule_probability_threshold"]):
                reason = "NO_CONFIDENT_ENEMY_CUE"
            elif basic_probability < cast(float, base["basic_model_probability_threshold"]):
                reason = "BASIC_NOT_CONFIDENTLY_ENABLED"
            elif not stable:
                reason = "PENDING_CONFIRMATION"
            elif not refractory_ok:
                reason = "REFRACTORY"
            elif not rate_ok:
                reason = "PER_MINUTE_LIMIT"
            elif not repetition_ok:
                reason = "REPETITION_LIMIT"
            else:
                reason = "CANDIDATE_BASIC_ATTACK"
            delays.append(time.monotonic() - scheduled)
            rows.append(
                {
                    "schema_version": SHADOW_SCHEMA,
                    "sequence": len(rows),
                    "frame_sha256": frame_hash,
                    "screen_valid": valid,
                    "frozen_screen": frozen,
                    "enemy_rule_probability": round(enemy_probability, 8),
                    "basic_model_probability": round(basic_probability, 8),
                    "stable": stable,
                    "candidate": "candidate_basic_attack" if accepted else "wait",
                    "rejection_reason": reason,
                    "decision_seconds": round(time.monotonic() - decision_start, 8),
                    "input_sent": False,
                }
            )
            previous_timestamp = timestamp
    except Exception as exc:
        failure = str(exc)
    finally:
        watchdog.stop()
        stream.close()
    duration = time.monotonic() - started if started else 0.0
    candidates = sum(row["candidate"] == "candidate_basic_attack" for row in rows)
    coverage = len(rows) / expected_cycles
    p95 = float(np.percentile(np.asarray(delays), 95)) if delays else None
    candidate_rate = candidates / max(duration / 60.0, 1e-9)
    strict = bool(
        failure is None
        and coverage >= cast(float, shadow["minimum_period_coverage"])
        and p95 is not None
        and p95 <= cast(float, shadow["maximum_p95_decision_seconds"])
        and candidates >= cast(int, shadow["minimum_candidates"])
        and candidate_rate <= cast(int, shadow["maximum_candidates_per_minute"])
        and invalid_candidates == 0
    )
    summary: dict[str, object] = {
        "schema_version": SHADOW_SCHEMA,
        "status": "PASSED" if strict else "FAILED",
        "strict_passed": strict,
        "base_contract_sha256": base_checked["contract_sha256"],
        "shadow_contract_sha256": shadow["contract_sha256"],
        "offline_summary_sha256": _sha(offline_path),
        **component,
        "layout_sha256": layout_sha,
        "scheduled_cycles": expected_cycles,
        "completed_cycles": len(rows),
        "period_coverage": round(coverage, 8),
        "duration_seconds": round(duration, 8),
        "candidates": candidates,
        "candidate_rate_per_minute": round(candidate_rate, 8),
        "invalid_screen_candidates": invalid_candidates,
        "p95_scheduled_to_decision_seconds": None if p95 is None else round(p95, 8),
        "failure": failure,
        "raw_rgb_persisted": False,
        "input_commands_sent": 0,
        "control_output": False,
        "device_input_allowed": False,
        "probe_allowed": strict,
    }
    _write_output(output, rows, summary)
    return summary
