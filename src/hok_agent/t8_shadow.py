"""Read-only live diagnostics for a trained T8 model."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections import Counter, deque
from pathlib import Path
from typing import cast

import numpy as np

from hok_agent.mobile_testbed import (
    ABILITIES,
    AIMS,
    MOVEMENTS,
    TARGETS,
    AdbInputPipe,
    FactorizedAction,
    GuardWatchdog,
    ScrcpyV4L2,
    _execute_action,
    _guarded_send,
    _model_frame,
    _new_large_output,
    _open_device_guard,
    load_layout,
)
from hok_agent.t8 import (
    CONFIDENCE_THRESHOLD,
    EVALUATION_SCHEMA,
    FRAME_COUNT,
    HOLD_VALUES,
    V25_SPLIT_SCHEMA,
    V26_EVALUATION_SCHEMA,
    V26_GATE_DECISION_THRESHOLD,
    V26_SELECTION_SCHEMA,
    _canonical,
    _combat_session_views,
    _head_metrics,
    _model_metadata,
    _switch_rate,
    _v25_shard_loader,
    _V25FrameCache,
    _v26_selected_test_rows,
    open_t8_predictor,
    open_t8_v26_predictor,
    open_t8_v26_stream_predictor,
)

SCHEMA = "hok-agent-t8-shadow-v2"
V26_SHADOW_SCHEMA = "hok-agent-t8-v2.6-read-only-shadow-v1"
V26_REPLAY_SHADOW_SCHEMA = "hok-agent-t8-v2.6-sealed-replay-shadow-v1"
V26_SHADOW_CONFIDENCE_THRESHOLD = 0.45
V26_SHADOW_ENTROPY_THRESHOLD = 0.80
V26_PROBE_SCHEMA = "hok-agent-t8-v2.6-bounded-20-action-probe-v1"
V26_PROBE_MINIMUM_INTERVAL_SECONDS = 0.50
V26_PROBE_MAX_IDENTICAL_ACTIONS = 3
V26_PROBE_SCENE_WARMUP_SECONDS = 5.0


class T8ShadowError(ValueError):
    pass


def _publish(path: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as raw:
        staging = Path(raw)
        events = staging / "events.jsonl"
        events.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary["events_sha256"] = hashlib.sha256(events.read_bytes()).hexdigest()
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(path)


def _action(labels: np.ndarray) -> dict[str, object]:
    return {
        "movement": MOVEMENTS[int(labels[0])],
        "ability": ABILITIES[int(labels[1])],
        "aim": AIMS[int(labels[2])],
        "target": TARGETS[int(labels[3])],
        "hold_ms": HOLD_VALUES[int(labels[4])],
    }


def _decode_candidate(candidate: dict[str, object]) -> FactorizedAction | None:
    movement, ability = candidate["movement"], candidate["ability"]
    aim, target, hold = candidate.get("aim"), candidate.get("target"), candidate.get("hold_ms")
    if (
        not isinstance(movement, str)
        or not isinstance(ability, str)
        or not isinstance(aim, str)
        or not isinstance(target, str)
        or not isinstance(hold, int)
    ):
        return None
    if movement != "wait" and ability != "none":
        return None
    try:
        return FactorizedAction(
            movement=movement,
            ability=ability,
            aim=aim,
            target=target,
            hold_ms=hold,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _accepted_offline_evidence(offline_report: Path, model_path: Path, layout_sha256: str) -> None:
    offline = _read_json(offline_report, "T8 offline evaluation is unreadable")
    if (
        offline.get("schema_version") != EVALUATION_SCHEMA
        or offline.get("status") != "PASSED"
        or offline.get("strict_passed") is not True
        or offline.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest()
        or offline.get("layout_sha256") != layout_sha256
    ):
        raise T8ShadowError("T8 offline gate did not admit Shadow")


def _accepted_v26_shadow_evidence(
    offline_report: Path,
    split_path: Path,
    model_path: Path,
    layout_sha256: str,
) -> float:
    offline = _read_json(offline_report, "T8-v2.6 offline evaluation is unreadable")
    split = _read_json(split_path, "T8-v2.6 split is unreadable")
    unsigned = {key: value for key, value in split.items() if key != "split_sha256"}
    switch = offline.get("switch_rate")
    expected_switch = switch.get("predicted") if isinstance(switch, dict) else None
    if (
        offline.get("schema_version") != V26_EVALUATION_SCHEMA
        or offline.get("status") != "SEALED_OFFLINE_EVALUATION_PASSED"
        or offline.get("strict_passed") is not True
        or offline.get("shadow_allowed") is not True
        or offline.get("test_accessed") is not True
        or offline.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest()
        or split.get("schema_version") != V25_SPLIT_SCHEMA
        or split.get("split_sha256") != offline.get("split_sha256")
        or split.get("split_sha256") != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or split.get("layout_sha256") != layout_sha256
        or split.get("gate_decision_threshold") != V26_GATE_DECISION_THRESHOLD
        or not isinstance(expected_switch, (int, float))
        or not 0 <= float(expected_switch) <= 1
    ):
        raise T8ShadowError("T8-v2.6 offline gate did not admit Shadow")
    return float(expected_switch)


def _accepted_v26_probe_evidence(
    *,
    selection_path: Path,
    offline_report: Path,
    shadow_summary: Path,
    split_path: Path,
    model_path: Path,
    layout_sha256: str,
) -> None:
    _accepted_v26_shadow_evidence(offline_report, split_path, model_path, layout_sha256)
    selection = _read_json(selection_path, "T8-v2.6 selection is unreadable")
    shadow = _read_json(shadow_summary, "T8-v2.6 Shadow summary is unreadable")
    unsigned = {key: value for key, value in selection.items() if key != "selection_sha256"}
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if (
        selection.get("schema_version") != V26_SELECTION_SCHEMA
        or selection.get("status") != "THREE_SEED_MODEL_SELECTED"
        or selection.get("selected_seed") != 1
        or selection.get("selected_model") != "seed-1/model-seed-1.safetensors"
        or selection.get("selected_model_sha256") != model_sha256
        or selection.get("selection_sha256")
        != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or shadow.get("schema_version") != V26_REPLAY_SHADOW_SCHEMA
        or shadow.get("status") != "PASSED"
        or shadow.get("strict_passed") is not True
        or shadow.get("source_session") != "session-011"
        or shadow.get("model_sha256") != model_sha256
        or shadow.get("layout_sha256") != layout_sha256
        or shadow.get("split_sha256") != selection.get("split_sha256")
        or shadow.get("input_commands_sent") != 0
        or shadow.get("control_output") is not False
    ):
        raise T8ShadowError("T8-v2.6 evidence did not admit the bounded probe")


def _v26_probe_action(label: int) -> FactorizedAction:
    if label not in (1, 2, 3):
        raise T8ShadowError("T8-v2.6 probe candidate is not an allowed combat action")
    return FactorizedAction(ability=ABILITIES[label])


def _v26_scene_candidate_admitted(
    label: int, stable: bool, confidence: float, entropy: float
) -> bool:
    return bool(
        label in (1, 2, 3)
        and stable
        and confidence >= V26_SHADOW_CONFIDENCE_THRESHOLD
        and entropy <= V26_SHADOW_ENTROPY_THRESHOLD
    )


def _read_json(path: Path, message: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T8ShadowError(message) from exc
    if not isinstance(value, dict):
        raise T8ShadowError(message)
    return value


def _accepted_probe_evidence(
    training_report: Path,
    offline_report: Path,
    shadow_summary: Path,
    model_path: Path,
    layout_sha256: str,
) -> None:
    try:
        training = _read_json(training_report, "T8 admission evidence is unreadable")
        shadow = _read_json(shadow_summary, "T8 admission evidence is unreadable")
    except T8ShadowError:
        raise
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    _accepted_offline_evidence(offline_report, model_path, layout_sha256)
    listed = training.get("seeds")
    admitted = isinstance(listed, list) and any(
        isinstance(row, dict) and row.get("sha256") == model_sha256 for row in listed
    )
    if (
        training.get("status") != "TRAINED_PENDING_OFFLINE_EVALUATION"
        or training.get("layout_sha256") != layout_sha256
        or not admitted
        or shadow.get("status") != "PASSED"
        or shadow.get("strict_passed") is not True
        or shadow.get("model_sha256") != model_sha256
        or shadow.get("layout_sha256") != layout_sha256
    ):
        raise T8ShadowError("T8 strict offline and Shadow gates did not admit execution")


def run_t8_shadow(
    *,
    serial: str,
    model_path: Path,
    offline_report: Path,
    layout_path: Path,
    video_node: Path,
    output_dir: Path,
    device: str,
    stream_fps: int = 30,
    infer_hz: int = 10,
    run_seconds: float = 300.0,
) -> dict[str, object]:
    if (
        output_dir.exists()
        or not 1 <= infer_hz <= 30
        or not 1 <= stream_fps <= 60
        or run_seconds != 300.0
    ):
        raise T8ShadowError("invalid read-only Shadow bounds")
    try:
        output_dir = _new_large_output(output_dir)
        guard = _open_device_guard(serial)
    except ValueError as exc:
        raise T8ShadowError(str(exc)) from exc
    layout, layout_sha256 = load_layout(layout_path)
    if (guard.width, guard.height) != (layout.width, layout.height):
        raise T8ShadowError("T8 layout does not match the active display")
    _accepted_offline_evidence(offline_report, model_path, layout_sha256)
    predictor = open_t8_predictor(model_path, layout_sha256, device)
    try:
        ood_threshold = float(_model_metadata(model_path)["ood_threshold"])
    except (KeyError, ValueError) as exc:
        raise T8ShadowError("T8 model OOD threshold is invalid") from exc
    stream = ScrcpyV4L2(guard.serial, video_node, stream_fps)
    rows: list[dict[str, object]] = []
    windows: deque[np.ndarray] = deque(maxlen=FRAME_COUNT)
    delays: list[float] = []
    ood_values: list[float] = []
    expected_cycles = int(run_seconds * infer_hz)
    start, next_due = time.monotonic(), time.monotonic()
    failure: str | None = None
    try:
        stream.start()
        while len(rows) < expected_cycles and time.monotonic() - start < run_seconds:
            now = time.monotonic()
            if now < next_due:
                time.sleep(min(next_due - now, 0.01))
                continue
            scheduled = next_due
            next_due += 1 / infer_hz
            guard.check()
            frame = _model_frame(stream.frame())
            windows.append(frame)
            if len(windows) < FRAME_COUNT:
                windows.extendleft([frame] * (FRAME_COUNT - len(windows)))
            labels, confidence, ood = predictor(np.stack([np.stack(tuple(windows))]))
            action = _action(labels[0])
            stable = len(rows) >= 4 and sum(row["candidate"] == action for row in rows[-4:]) >= 3
            delays.append(time.monotonic() - scheduled)
            ood_values.append(float(ood[0]))
            rows.append(
                {
                    "schema_version": SCHEMA,
                    "sequence": len(rows),
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "candidate": action,
                    "confidence": round(float(confidence[0]), 8),
                    "ood_distance": round(float(ood[0]), 8),
                    "stable": stable,
                    "advisory_action": "ABSTAIN",
                    "abstain_reason": "T8_READ_ONLY_SHADOW",
                    "control_output": False,
                }
            )
    except Exception as exc:
        failure = str(exc)
    finally:
        stream.close()
    p95 = float(np.percentile(np.asarray(delays), 95)) if delays else None
    stable_ratio = sum(bool(row["stable"]) for row in rows) / len(rows) if rows else 0.0
    confidence_ratio = (
        sum(float(cast(float, row["confidence"])) >= CONFIDENCE_THRESHOLD for row in rows)
        / len(rows)
        if rows
        else 0.0
    )
    ood_inlier_ratio = (
        sum(value <= ood_threshold for value in ood_values) / len(ood_values) if ood_values else 0.0
    )
    coverage = len(rows) / expected_cycles
    strict = (
        failure is None
        and coverage >= 0.99
        and p95 is not None
        and p95 <= 0.1
        and stable_ratio >= 0.9
        and confidence_ratio >= 0.9
        and ood_inlier_ratio >= 0.9
    )
    summary: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "PASSED" if strict else "FAILED",
        "disposition": "T8_READ_ONLY_SHADOW",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "layout_sha256": layout_sha256,
        "device": device,
        "scheduled_cycles": expected_cycles,
        "completed_cycles": len(rows),
        "period_coverage": round(coverage, 8),
        "p95_scheduled_to_candidate_seconds": None if p95 is None else round(p95, 8),
        "stable_ratio": round(stable_ratio, 8),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "confidence_ratio": round(confidence_ratio, 8),
        "ood_threshold": round(ood_threshold, 8),
        "ood_inlier_ratio": round(ood_inlier_ratio, 8),
        "failure": failure,
        "strict_passed": strict,
        "raw_frames_persisted": False,
        "control_output": False,
        "advisory_output": False,
        "target_intent_status": "NOT_IMPLEMENTED",
    }
    _publish(output_dir, rows, summary)
    if failure is not None:
        raise T8ShadowError("Shadow stopped: " + failure)
    return summary


def run_t8_v26_shadow(
    *,
    serial: str,
    model_path: Path,
    offline_report: Path,
    split_path: Path,
    layout_path: Path,
    video_node: Path,
    output_dir: Path,
    device: str,
    stream_fps: int = 30,
    infer_hz: int = 10,
    run_seconds: float = 300.0,
) -> dict[str, object]:
    if (
        output_dir.exists()
        or not 1 <= infer_hz <= 10
        or not 1 <= stream_fps <= 60
        or run_seconds != 300.0
    ):
        raise T8ShadowError("invalid T8-v2.6 read-only Shadow bounds")
    try:
        output_dir = _new_large_output(output_dir)
        guard = _open_device_guard(serial)
    except ValueError as exc:
        raise T8ShadowError(str(exc)) from exc
    layout, layout_sha256 = load_layout(layout_path)
    if (guard.width, guard.height) != (layout.width, layout.height):
        raise T8ShadowError("T8-v2.6 layout does not match the active display")
    expected_switch = _accepted_v26_shadow_evidence(
        offline_report, split_path, model_path, layout_sha256
    )
    predictor = open_t8_v26_stream_predictor(model_path, device)
    stream = ScrcpyV4L2(guard.serial, video_node, stream_fps)
    watchdog = GuardWatchdog(guard)
    rows: list[dict[str, object]] = []
    delays: list[float] = []
    expected_cycles = int(run_seconds * infer_hz)
    start, next_due = 0.0, 0.0
    failure: str | None = None
    try:
        stream.start()
        watchdog.start()
        start = next_due = time.monotonic()
        while len(rows) < expected_cycles and time.monotonic() - start < run_seconds:
            now = time.monotonic()
            if now < next_due:
                time.sleep(min(next_due - now, 0.01))
                continue
            scheduled = next_due
            next_due += 1 / infer_hz
            watchdog.ensure_fresh()
            frame = _model_frame(stream.frame())
            views = _combat_session_views(frame[None], (0, 0, 128, 128))[0]
            labels, confidence, entropy = predictor(views)
            current_confidence = float(confidence[0])
            current_entropy = float(entropy[0])
            abstain = (
                current_confidence < V26_SHADOW_CONFIDENCE_THRESHOLD
                or current_entropy > V26_SHADOW_ENTROPY_THRESHOLD
            )
            delays.append(time.monotonic() - scheduled)
            rows.append(
                {
                    "schema_version": V26_SHADOW_SCHEMA,
                    "sequence": len(rows),
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "combat_candidate": ABILITIES[int(labels[0])],
                    "confidence": round(current_confidence, 8),
                    "normalized_predictive_entropy": round(current_entropy, 8),
                    "advisory_action": "ABSTAIN",
                    "abstain_reason": (
                        "T8_V26_LOW_CONFIDENCE_OR_ENTROPY_PROXY"
                        if abstain
                        else "T8_V26_READ_ONLY_SHADOW"
                    ),
                    "control_output": False,
                }
            )
    except Exception as exc:
        failure = str(exc)
    finally:
        watchdog.stop()
        stream.close()
    coverage = len(rows) / expected_cycles
    p95 = float(np.percentile(np.asarray(delays), 95)) if delays else None
    confidence_ratio = (
        sum(
            float(cast(float, row["confidence"])) >= V26_SHADOW_CONFIDENCE_THRESHOLD
            for row in rows
        )
        / len(rows)
        if rows
        else 0.0
    )
    entropy_inlier_ratio = (
        sum(
            float(cast(float, row["normalized_predictive_entropy"]))
            <= V26_SHADOW_ENTROPY_THRESHOLD
            for row in rows
        )
        / len(rows)
        if rows
        else 0.0
    )
    switches = sum(
        rows[index]["combat_candidate"] != rows[index - 1]["combat_candidate"]
        for index in range(1, len(rows))
    )
    switch_rate = switches / (len(rows) - 1) if len(rows) > 1 else 0.0
    switch_error = abs(switch_rate - expected_switch)
    strict = bool(
        failure is None
        and coverage >= 0.95
        and p95 is not None
        and p95 <= 0.15
        and confidence_ratio >= 0.50
        and entropy_inlier_ratio >= 0.50
        and switch_error <= 0.15
    )
    summary: dict[str, object] = {
        "schema_version": V26_SHADOW_SCHEMA,
        "status": "PASSED" if strict else "FAILED",
        "strict_passed": strict,
        "disposition": "T8_V26_READ_ONLY_SHADOW",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "layout_sha256": layout_sha256,
        "split_sha256": _read_json(split_path, "T8-v2.6 split is unreadable")["split_sha256"],
        "scheduled_cycles": expected_cycles,
        "completed_cycles": len(rows),
        "period_coverage": round(coverage, 8),
        "p95_scheduled_to_candidate_seconds": None if p95 is None else round(p95, 8),
        "confidence_threshold": V26_SHADOW_CONFIDENCE_THRESHOLD,
        "confidence_ratio": round(confidence_ratio, 8),
        "ood_status": "PREDICTIVE_ENTROPY_PROXY_NOT_TRAINED_OOD",
        "entropy_threshold": V26_SHADOW_ENTROPY_THRESHOLD,
        "entropy_inlier_ratio": round(entropy_inlier_ratio, 8),
        "offline_predicted_switch_rate": round(expected_switch, 8),
        "shadow_predicted_switch_rate": round(switch_rate, 8),
        "switch_rate_absolute_error": round(switch_error, 8),
        "thresholds": {
            "period_coverage": 0.95,
            "p95_latency_seconds": 0.15,
            "confidence_ratio": 0.50,
            "entropy_inlier_ratio": 0.50,
            "switch_rate_absolute_error": 0.15,
        },
        "failure": failure,
        "raw_frames_persisted": False,
        "input_commands_sent": 0,
        "control_output": False,
        "advisory_output": False,
    }
    _publish(output_dir, rows, summary)
    if failure is not None:
        raise T8ShadowError("T8-v2.6 Shadow stopped: " + failure)
    return summary


def run_t8_v26_replay_shadow(
    *,
    dataset_root: Path,
    split_path: Path,
    run_root: Path,
    selection_path: Path,
    offline_report: Path,
    layout_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, object]:
    if output_dir.exists():
        raise T8ShadowError("T8-v2.6 replay Shadow output already exists")
    try:
        output_dir = _new_large_output(output_dir)
        layout, layout_sha256 = load_layout(layout_path)
        root, rows, _selection, model_path = _v26_selected_test_rows(
            dataset_root, split_path, run_root, selection_path
        )
    except ValueError as exc:
        raise T8ShadowError(str(exc)) from exc
    expected_switch = _accepted_v26_shadow_evidence(
        offline_report, split_path, model_path, layout_sha256
    )
    selected_rows = [row for row in rows if row["session"] == "session-011"]
    if not selected_rows:
        raise T8ShadowError("T8-v2.6 sealed replay session is unavailable")
    predictor = open_t8_v26_predictor(model_path, device)
    loader = _v25_shard_loader(_V25FrameCache())
    events: list[dict[str, object]] = []
    predicted: list[int] = []
    labels: list[int] = []
    confidence_values: list[float] = []
    entropy_values: list[float] = []
    latencies: list[float] = []
    for row in selected_rows:
        views, _shifted, current_labels = loader(root, row)
        for ordinal in range(len(views)):
            started = time.perf_counter()
            current_predicted, current_confidence, current_entropy = predictor(
                views[ordinal : ordinal + 1]
            )
            latency = time.perf_counter() - started
            prediction = int(current_predicted[0])
            label = int(current_labels[ordinal])
            confidence = float(current_confidence[0])
            entropy = float(current_entropy[0])
            predicted.append(prediction)
            labels.append(label)
            confidence_values.append(confidence)
            entropy_values.append(entropy)
            latencies.append(latency)
            events.append(
                {
                    "schema_version": V26_REPLAY_SHADOW_SCHEMA,
                    "sequence": len(events),
                    "frame_sha256": hashlib.sha256(views[ordinal, -1, 0].tobytes()).hexdigest(),
                    "sealed_executed_action": ABILITIES[label],
                    "combat_candidate": ABILITIES[prediction],
                    "confidence": round(confidence, 8),
                    "normalized_predictive_entropy": round(entropy, 8),
                    "control_output": False,
                }
            )
    predicted_array = np.asarray(predicted, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=np.int64)
    metrics = _head_metrics(predicted_array, labels_array, 4)
    recall = cast(list[float], metrics["per_class_recall"])
    true_switch = _switch_rate(labels_array[:, None], (len(labels_array),))
    predicted_switch = _switch_rate(predicted_array[:, None], (len(predicted_array),))
    switch_error = abs(predicted_switch - true_switch)
    confidence_ratio = float(
        np.mean(np.asarray(confidence_values) >= V26_SHADOW_CONFIDENCE_THRESHOLD)
    )
    entropy_array = np.asarray(entropy_values)
    entropy_finite = bool(np.all(np.isfinite(entropy_array)))
    entropy_inlier_ratio = float(
        np.mean(entropy_array <= V26_SHADOW_ENTROPY_THRESHOLD)
    )
    p95 = float(np.percentile(np.asarray(latencies), 95))
    passed = bool(
        cast(float, metrics["accuracy"]) >= 0.45
        and cast(float, metrics["macro_f1"]) >= 0.50
        and min(recall[1:]) >= 0.35
        and p95 <= 0.15
        and confidence_ratio >= 0.50
        and entropy_finite
        and entropy_inlier_ratio >= 0.50
        and switch_error <= 0.10
    )
    summary: dict[str, object] = {
        "schema_version": V26_REPLAY_SHADOW_SCHEMA,
        "status": "PASSED" if passed else "FAILED",
        "strict_passed": passed,
        "disposition": "T8_V26_SEALED_FIVE_MINUTE_SESSION_REPLAY_SHADOW",
        "source_session": "session-011",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "layout_sha256": layout_sha256,
        "split_sha256": _read_json(split_path, "T8-v2.6 split is unreadable")["split_sha256"],
        "samples": len(events),
        "metrics": metrics,
        "p95_prediction_seconds": round(p95, 8),
        "confidence_threshold": V26_SHADOW_CONFIDENCE_THRESHOLD,
        "confidence_ratio": round(confidence_ratio, 8),
        "ood_status": "PREDICTIVE_ENTROPY_PROXY_NOT_TRAINED_OOD",
        "entropy_threshold": V26_SHADOW_ENTROPY_THRESHOLD,
        "entropy_finite": entropy_finite,
        "entropy_inlier_ratio": round(entropy_inlier_ratio, 8),
        "offline_two_session_predicted_switch_rate": round(expected_switch, 8),
        "true_switch_rate": round(true_switch, 8),
        "predicted_switch_rate": round(predicted_switch, 8),
        "switch_rate_absolute_error": round(switch_error, 8),
        "thresholds": {
            "accuracy": 0.45,
            "macro_f1": 0.50,
            "minimum_action_recall": 0.35,
            "p95_prediction_seconds": 0.15,
            "confidence_ratio": 0.50,
            "entropy_inlier_ratio": 0.50,
            "switch_rate_absolute_error": 0.10,
        },
        "raw_frames_persisted": False,
        "input_commands_sent": 0,
        "control_output": False,
        "advisory_output": False,
    }
    _publish(output_dir, events, summary)
    return summary


def run_t8_v26_execute_probe(
    *,
    serial: str,
    model_path: Path,
    selection_path: Path,
    offline_report: Path,
    shadow_summary: Path,
    split_path: Path,
    layout_path: Path,
    video_node: Path,
    output_dir: Path,
    device: str,
    stream_fps: int = 30,
    infer_hz: int = 10,
    run_seconds: float = 60.0,
    max_actions: int = 20,
) -> dict[str, object]:
    if (
        output_dir.exists()
        or run_seconds != 60.0
        or max_actions != 20
        or infer_hz != 10
        or not 10 <= stream_fps <= 60
    ):
        raise T8ShadowError("T8-v2.6 probe requires the frozen 60-second/20-action bounds")
    try:
        output_dir = _new_large_output(output_dir)
        guard = _open_device_guard(serial)
    except ValueError as exc:
        raise T8ShadowError(str(exc)) from exc
    layout, layout_sha256 = load_layout(layout_path)
    if (
        (guard.width, guard.height) != (layout.width, layout.height)
        or any(layout.buttons[name] is None for name in ABILITIES[1:4])
    ):
        raise T8ShadowError("T8-v2.6 probe requires the complete three-button layout")
    _accepted_v26_probe_evidence(
        selection_path=selection_path,
        offline_report=offline_report,
        shadow_summary=shadow_summary,
        split_path=split_path,
        model_path=model_path,
        layout_sha256=layout_sha256,
    )
    predictor = open_t8_v26_stream_predictor(model_path, device)
    stream = ScrcpyV4L2(guard.serial, video_node, stream_fps)
    pipe = AdbInputPipe(guard.serial)
    watchdog = GuardWatchdog(guard)
    rows: list[dict[str, object]] = []
    delays: list[float] = []
    executed = 0
    failure: str | None = None
    last_sent = 0.0
    last_ability = ""
    identical_actions = 0
    scene_ready = False
    started = next_due = 0.0
    try:
        stream.start()
        watchdog.start()
        started = next_due = time.monotonic()
        while time.monotonic() - started < run_seconds and executed < max_actions:
            now = time.monotonic()
            if now < next_due:
                time.sleep(min(next_due - now, 0.01))
                continue
            scheduled = next_due
            next_due = max(next_due + 1 / infer_hz, now)
            watchdog.ensure_fresh()
            frame = _model_frame(stream.frame())
            views = _combat_session_views(frame[None], (0, 0, 128, 128))[0]
            labels, confidence, entropy = predictor(views)
            label = int(labels[0])
            candidate = ABILITIES[label]
            current_confidence = float(confidence[0])
            current_entropy = float(entropy[0])
            stable = len(rows) >= 3 and all(
                row["candidate"] == candidate for row in rows[-3:]
            )
            elapsed = time.monotonic() - started
            scene_ready = scene_ready or _v26_scene_candidate_admitted(
                label, stable, current_confidence, current_entropy
            )
            warmup_complete = elapsed >= V26_PROBE_SCENE_WARMUP_SECONDS
            if warmup_complete and not scene_ready:
                raise T8ShadowError(
                    "T8-v2.6 live scene produced no admitted combat candidate during warmup"
                )
            interval_ready = time.monotonic() - last_sent >= V26_PROBE_MINIMUM_INTERVAL_SECONDS
            repetition_ready = (
                candidate != last_ability
                or identical_actions < V26_PROBE_MAX_IDENTICAL_ACTIONS
            )
            accepted = bool(
                label in (1, 2, 3)
                and warmup_complete
                and stable
                and current_confidence >= V26_SHADOW_CONFIDENCE_THRESHOLD
                and current_entropy <= V26_SHADOW_ENTROPY_THRESHOLD
                and interval_ready
                and repetition_ready
            )
            sent = False
            if accepted:
                action = _v26_probe_action(label)
                sent = _execute_action(
                    action,
                    layout,
                    guard.width,
                    guard.height,
                    _guarded_send(guard, pipe.send),
                )
            if sent:
                executed += 1
                last_sent = time.monotonic()
                if candidate == last_ability:
                    identical_actions += 1
                else:
                    last_ability = candidate
                    identical_actions = 1
            if not warmup_complete:
                rejection = "SCENE_WARMUP"
            elif label == 0:
                rejection = "WAIT"
            elif not stable:
                rejection = "UNSTABLE"
            elif current_confidence < V26_SHADOW_CONFIDENCE_THRESHOLD:
                rejection = "LOW_CONFIDENCE"
            elif current_entropy > V26_SHADOW_ENTROPY_THRESHOLD:
                rejection = "HIGH_ENTROPY"
            elif not interval_ready:
                rejection = "RATE_LIMIT"
            elif not repetition_ready:
                rejection = "IDENTICAL_ACTION_LIMIT"
            else:
                rejection = "NONE"
            delays.append(time.monotonic() - scheduled)
            rows.append(
                {
                    "schema_version": V26_PROBE_SCHEMA,
                    "sequence": len(rows),
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "candidate": candidate,
                    "confidence": round(current_confidence, 8),
                    "normalized_predictive_entropy": round(current_entropy, 8),
                    "stable": stable,
                    "rejection": rejection,
                    "input_sent": sent,
                }
            )
    except Exception as exc:
        failure = str(exc)
    finally:
        watchdog.stop()
        pipe.close()
        stream.close()
    duration = time.monotonic() - started if started else 0.0
    p95 = float(np.percentile(np.asarray(delays), 95)) if delays else None
    sent_actions = [cast(str, row["candidate"]) for row in rows if row["input_sent"]]
    strict = bool(
        failure is None
        and executed == max_actions
        and len(sent_actions) == executed
        and all(action in ABILITIES[1:4] for action in sent_actions)
    )
    summary: dict[str, object] = {
        "schema_version": V26_PROBE_SCHEMA,
        "status": "PASSED" if strict else "FAILED",
        "strict_passed": strict,
        "disposition": "T8_V26_BOUNDED_SELF_BUILT_APP_PROBE",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "layout_sha256": layout_sha256,
        "split_sha256": _read_json(split_path, "T8-v2.6 split is unreadable")["split_sha256"],
        "duration_seconds": round(duration, 8),
        "inference_cycles": len(rows),
        "executed_actions": executed,
        "max_actions": max_actions,
        "executed_action_counts": dict(Counter(sent_actions)),
        "minimum_action_interval_seconds": V26_PROBE_MINIMUM_INTERVAL_SECONDS,
        "maximum_consecutive_identical_actions": V26_PROBE_MAX_IDENTICAL_ACTIONS,
        "scene_warmup_seconds": V26_PROBE_SCENE_WARMUP_SECONDS,
        "scene_ready": scene_ready,
        "p95_scheduled_to_decision_seconds": None if p95 is None else round(p95, 8),
        "failure": failure,
        "unexpected_or_out_of_vocabulary_actions": 0,
        "raw_frames_persisted": False,
        "coordinates_persisted": False,
        "control_output": executed > 0,
    }
    _publish(output_dir, rows, summary)
    return summary


def run_t8_execute_probe(
    *,
    serial: str,
    model_path: Path,
    layout_path: Path,
    video_node: Path,
    training_report: Path,
    offline_report: Path,
    shadow_summary: Path,
    output_dir: Path,
    device: str,
    stream_fps: int = 30,
    infer_hz: int = 10,
    run_seconds: float = 60.0,
    max_actions: int = 20,
) -> dict[str, object]:
    if (
        not 1 <= run_seconds <= 60
        or not 1 <= max_actions <= 20
        or not 1 <= infer_hz <= 30
        or not 1 <= stream_fps <= 60
    ):
        raise T8ShadowError("T8 probe bounds must be within the accepted cap")
    try:
        output_dir = _new_large_output(output_dir)
        guard = _open_device_guard(serial)
    except ValueError as exc:
        raise T8ShadowError(str(exc)) from exc
    layout, layout_sha256 = load_layout(layout_path)
    if (guard.width, guard.height) != (layout.width, layout.height):
        raise T8ShadowError("T8 layout does not match the active display")
    _accepted_probe_evidence(
        training_report, offline_report, shadow_summary, model_path, layout_sha256
    )
    predictor = open_t8_predictor(model_path, layout_sha256, device)
    stream, pipe = ScrcpyV4L2(guard.serial, video_node, stream_fps), AdbInputPipe(guard.serial)
    windows: deque[np.ndarray] = deque(maxlen=FRAME_COUNT)
    rows: list[dict[str, object]] = []
    start, next_due, executed = time.monotonic(), time.monotonic(), 0
    try:
        stream.start()
        while time.monotonic() - start < run_seconds and executed < max_actions:
            if time.monotonic() < next_due:
                time.sleep(0.01)
                continue
            next_due += 1 / infer_hz
            frame = _model_frame(stream.frame())
            windows.append(frame)
            if len(windows) < FRAME_COUNT:
                windows.extendleft([frame] * (FRAME_COUNT - len(windows)))
            labels, confidence, ood = predictor(np.stack([np.stack(tuple(windows))]))
            candidate = _action(labels[0])
            action = _decode_candidate(candidate)
            stable = len(rows) >= 4 and sum(row["candidate"] == candidate for row in rows[-4:]) >= 3
            accepted = (
                stable
                and float(confidence[0]) >= 0.8
                and float(ood[0]) <= 1.0
                and action is not None
            )
            sent = (
                False
                if not accepted or action is None
                else _execute_action(
                    action, layout, frame.shape[1], frame.shape[0], _guarded_send(guard, pipe.send)
                )
            )
            executed += int(sent)
            rows.append(
                {
                    "schema_version": SCHEMA,
                    "sequence": len(rows),
                    "candidate": candidate,
                    "stable": stable,
                    "confidence": round(float(confidence[0]), 8),
                    "ood_distance": round(float(ood[0]), 8),
                    "input_sent": sent,
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                }
            )
    finally:
        pipe.close()
        stream.close()
    summary: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "PASSED" if executed > 0 else "FAILED",
        "disposition": "T8_BOUNDED_SELF_BUILT_APP_PROBE",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "layout_sha256": layout_sha256,
        "duration_seconds": round(time.monotonic() - start, 8),
        "executed_actions": executed,
        "max_actions": max_actions,
        "raw_frames_persisted": False,
        "control_output": executed > 0,
        "target_intent_status": "NOT_IMPLEMENTED",
    }
    _publish(output_dir, rows, summary)
    return summary
