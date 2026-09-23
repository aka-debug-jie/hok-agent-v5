from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Final, cast

import av
import numpy as np
import torch

from hok_agent.capture import _coerce_frame, _validate_capture_device
from hok_agent.global_policy import (
    GlobalMacroPolicy,
    _is_static_window,
    _predict_command,
    _under_large_root,
    load_global_model,
    real_video_views,
)
from hok_agent.mobile_testbed import GuardWatchdog, MobileTestbedError, _open_device_guard

SCHEMA: Final = "hok-agent-global-shadow-v1"
CONFIG_PATH: Final = Path(__file__).resolve().parents[2] / "configs/global_agent_shadow_v1.json"
AUTHORIZATION_PATH: Final = (
    Path(__file__).resolve().parents[2] / "docs/GLOBAL_AGENT_V1_SHADOW_AUTHORIZATION.json"
)
OFFLINE_EVIDENCE_PATH: Final = (
    Path(__file__).resolve().parents[2] / "docs/GLOBAL_AGENT_V1_OFFLINE_EVIDENCE.json"
)
PROMOTED_CHECKPOINT_SHA256: Final = (
    "c033264f83d1c667c4dff5f93dec02183535386cfcde1a527a60303647a3e39e"
)
# The candidate is a disjoint, non-promoting Shadow variant. It never replaces the promoted
# checkpoint: the frozen promoted authorization and the promoted checkpoint sha above stay
# byte-identical, and a candidate run reports its own schema and authorization so the two can never
# be confused or pooled. The candidate is admitted here only because it is non-inferior
# on the frozen holdout (20 of 20 paired agreement) and cheap; not promoted, grants no control.
CANDIDATE_AUTHORIZATION_PATH: Final = (
    Path(__file__).resolve().parents[2]
    / "docs/GLOBAL_AGENT_CANDIDATE_SHADOW_AUTHORIZATION.json"
)
CANDIDATE_CHECKPOINT_SHA256: Final = (
    "d9f18bb9cbdd08c1ac8f37199575686983a10307bad57dd8ab375d9c01e2e7cb"
)
CANDIDATE_SCHEMA: Final = "hok-agent-global-shadow-candidate-v1"
WINDOW_FRAMES: Final = 16


class GlobalShadowError(ValueError):
    pass


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GlobalShadowError("Global Agent Shadow contract is unavailable") from exc
    if not isinstance(raw, dict):
        raise GlobalShadowError("Global Agent Shadow contract is invalid")
    return cast(dict[str, object], raw)


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    y = np.linspace(0, frame.shape[0] - 1, height).round().astype(np.int64)
    x = np.linspace(0, frame.shape[1] - 1, width).round().astype(np.int64)
    return frame[y[:, None], x[None, :]]


def _load_local_observation_rois(path: Path) -> tuple[dict[str, tuple[int, int, int, int]], str]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink():
        raise GlobalShadowError("observation ROI configuration must be a real file")
    raw = _load_json(resolved)
    if raw.get("schema_version") != "hok-agent-mobile-observation-rois-v1":
        raise GlobalShadowError("observation ROI configuration schema is not admitted")
    screen = raw.get("screen")
    if not isinstance(screen, dict):
        raise GlobalShadowError("observation ROI configuration has no screen binding")
    width, height = screen.get("width"), screen.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise GlobalShadowError("observation ROI configuration has an invalid screen binding")
    boxes: dict[str, tuple[int, int, int, int]] = {"screen": (0, 0, width, height)}
    for name in ("main_view", "minimap", "hud"):
        value = raw.get(name)
        if not isinstance(value, dict) or value.get("observation_only") is not True:
            raise GlobalShadowError("observation ROI configuration is not read-only")
        box = value.get("pixel_box_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise GlobalShadowError("observation ROI configuration has an invalid box")
        if not all(isinstance(item, int) for item in box):
            raise GlobalShadowError("observation ROI configuration has an invalid box")
        x0, y0, x1, y1 = cast(tuple[int, int, int, int], tuple(box))
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise GlobalShadowError("observation ROI configuration box is outside the screen")
        boxes[name] = (x0, y0, x1, y1)
    return boxes, _sha_file(resolved)


def _local_observation_views(
    frame: np.ndarray, boxes: dict[str, tuple[int, int, int, int]]
) -> tuple[np.ndarray, ...]:
    _x0, _y0, width, height = boxes["screen"]
    if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
        raise GlobalShadowError("captured frame does not match the local ROI screen binding")

    def crop(name: str, output_shape: tuple[int, int]) -> np.ndarray:
        x0, y0, x1, y1 = boxes[name]
        return _resize_nearest(frame[y0:y1, x0:x1], *output_shape)

    return crop("main_view", (128, 128)), crop("minimap", (64, 64)), crop("hud", (32, 128))


def _load_contracts() -> tuple[dict[str, object], str, str]:
    config = _load_json(CONFIG_PATH)
    expected_config = {
        "schema_version": "hok-agent-global-shadow-config-v1",
        "sample_hz": 5,
        "decision_hz": 2,
        "window_frames": 16,
        "minimum_confidence": 0.55,
        "minimum_command_hold_ms": 1500,
        "maximum_frame_age_ms": 500,
        "smoke_seconds": 60,
        "formal_seconds": 600,
        "minimum_cycle_coverage": 0.95,
        "maximum_p95_end_to_end_ms": 500,
        "read_only_shadow_allowed": True,
        "control_output": False,
        "device_input_allowed": False,
        "online_learning_allowed": False,
    }
    if config != expected_config:
        raise GlobalShadowError("Global Agent Shadow config differs from frozen contract")
    evidence_sha = _sha_file(OFFLINE_EVIDENCE_PATH)
    authorization = _load_json(AUTHORIZATION_PATH)
    expected_auth = {
        "schema_version": "hok-agent-global-shadow-authorization-v1",
        "offline_evidence_commit": "d4715aedbb3146e7ee017a1916f0cc641bc4fa2c",
        "offline_evidence_sha256": evidence_sha,
        "checkpoint_sha256": PROMOTED_CHECKPOINT_SHA256,
        "challenge_pack_disposition": "diagnostic_only_not_shadow_blocking",
        "read_only_shadow_allowed": True,
        "control_output": False,
        "device_input_allowed": False,
        "online_learning_allowed": False,
    }
    if authorization != expected_auth:
        raise GlobalShadowError("Global Agent Shadow authorization differs from frozen contract")
    return (
        config,
        _sha_bytes(_canonical(config).encode()),
        _sha_bytes(_canonical(authorization).encode()),
    )


def _validate_checkpoint(
    path: Path, device: torch.device, *, candidate: bool = False
) -> GlobalMacroPolicy:
    expected = CANDIDATE_CHECKPOINT_SHA256 if candidate else PROMOTED_CHECKPOINT_SHA256
    if _sha_file(path) != expected:
        raise GlobalShadowError(
            "checkpoint is not the authorized candidate compression model"
            if candidate
            else "checkpoint is not the authorized promoted DAgger model"
        )
    model, _metadata = load_global_model(path, device)
    if candidate and model.main_architecture != "resnet18_shallow":
        raise GlobalShadowError("candidate checkpoint is not the declared shallow architecture")
    return model


def _load_candidate_authorization() -> str:
    """Load the candidate's own authorization, kept separate from the promoted one.

    Every flag that matters is re-checked here rather than trusted from the file, and the candidate
    term is required to be non-inferior with the recorded paired agreement. Nothing about the
    promoted authorization or the promoted checkpoint is read or altered by this path.
    """
    authorization = _load_json(CANDIDATE_AUTHORIZATION_PATH)
    expected = {
        "schema_version": "hok-agent-global-shadow-authorization-v2-candidate",
        "purpose": "candidate compression shadow, non-promoting",
        "promoted_authorization_sha256_unchanged": True,
        "baseline_checkpoint_sha256": PROMOTED_CHECKPOINT_SHA256,
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
        "candidate_architecture": "resnet18_shallow",
        "candidate_parameters": 5112974,
        "baseline_parameters": 11383694,
        "candidate_is_non_inferior": True,
        "holdout_paired_agreement": "20/20",
        "read_only_shadow_allowed": True,
        "control_output": False,
        "device_input_allowed": False,
        "online_learning_allowed": False,
        "promotion_allowed": False,
    }
    if authorization != expected:
        raise GlobalShadowError("candidate Shadow authorization differs from its declared contract")
    return _sha_bytes(_canonical(authorization).encode())


def _candidate_modes(intent: str, zone: str, abstain: bool) -> tuple[str, str, str]:
    if abstain:
        return "hold", "no_combat", "none"
    movement = (
        "retreat"
        if zone == "OWN_BASE"
        else "advance" if zone != "HOLD_CURRENT_ZONE" else "hold"
    )
    combat = {
        "FARM_LANE": "wave_clear",
        "PUSH_STRUCTURE": "structure",
        "ENGAGE": "hero_combat",
        "DISENGAGE": "no_combat",
        "RECALL": "no_combat",
    }.get(intent, "no_combat")
    return movement, combat, "none"


def _percentile(values: list[float], quantile: float) -> float:
    return 0.0 if not values else float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _write_event(handle, row: dict[str, object]) -> None:  # type: ignore[no-untyped-def]
    handle.write(_canonical(row) + "\n")
    handle.flush()


def run_global_shadow(
    *,
    serial: str,
    video_node: Path,
    checkpoint: Path,
    output_dir: Path,
    run_seconds: int,
    device_name: str,
    observation_rois: Path | None = None,
    candidate: bool = False,
) -> dict[str, object]:
    config, config_sha256, authorization_sha256 = _load_contracts()
    if candidate:
        # A candidate run has its own authorization and its own schema, so a candidate summary can
        # never be read as a promoted one. The promoted authorization sha is still recorded as
        # unchanged evidence that the promoted contract was not touched.
        authorization_sha256 = _load_candidate_authorization()
    smoke_seconds = cast(int, config["smoke_seconds"])
    formal_seconds = cast(int, config["formal_seconds"])
    sample_hz = cast(int, config["sample_hz"])
    decision_hz = cast(int, config["decision_hz"])
    maximum_frame_age_ms = cast(int, config["maximum_frame_age_ms"])
    minimum_confidence = cast(float, config["minimum_confidence"])
    minimum_hold_ms = cast(int, config["minimum_command_hold_ms"])
    if run_seconds not in {smoke_seconds, formal_seconds}:
        raise GlobalShadowError("run duration is not admitted by the frozen Shadow contract")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GlobalShadowError("CUDA requested but unavailable")
    node = _validate_capture_device(str(video_node))
    output = _under_large_root(output_dir, output=True)
    model = _validate_checkpoint(checkpoint, device, candidate=candidate)
    guard = _open_device_guard(serial)
    watchdog = GuardWatchdog(guard)
    roi_boxes: dict[str, tuple[int, int, int, int]] | None = None
    roi_sha256: str | None = None
    if observation_rois is not None:
        roi_boxes, roi_sha256 = _load_local_observation_rois(observation_rois)
    output.mkdir(parents=True)
    started = time.monotonic()
    next_sample = started
    next_decision: float | None = None
    frames: deque[tuple[np.ndarray, ...]] = deque(maxlen=WINDOW_FRAMES)
    raw_frames: deque[np.ndarray] = deque(maxlen=WINDOW_FRAMES)
    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    completed = scheduled = stale_frames = invalid_screen = hard_stops = 0
    abstentions = transitions = 0
    previous_command: tuple[str, str] | None = None
    accepted_command: tuple[str, str] | None = None
    accepted_at_ms = -(10**12)
    intents: Counter[str] = Counter()
    zones: Counter[str] = Counter()
    candidates: Counter[str] = Counter()
    stop_reason: str | None = None
    error: str | None = None
    try:
        watchdog.start()
        options = {"framerate": "30", "video_size": "1600x720"}
        with av.open(str(node), mode="r", format="video4linux2", options=options) as container:
            if not container.streams.video:
                raise GlobalShadowError("V4L2 node has no video stream")
            stream = container.streams.video[0]
            with (output / "events.jsonl").open("x", encoding="utf-8") as events:
                for decoded in container.decode(video=stream.index):
                    now = time.monotonic()
                    if now - started >= run_seconds:
                        break
                    if now < next_sample:
                        continue
                    while next_sample <= now:
                        next_sample += 1.0 / sample_hz
                    watchdog.ensure_fresh(maximum_frame_age_ms)
                    captured = time.monotonic()
                    raw_frame = decoded.to_ndarray(format="rgb24")
                    if roi_boxes is None:
                        frame = _coerce_frame(raw_frame)
                        views = real_video_views(frame)
                    else:
                        views = _local_observation_views(raw_frame, roi_boxes)
                        frame = views[0]
                    if float(frame.std()) < 1.0:
                        invalid_screen += 1
                        hard_stops += 1
                        stop_reason = "invalid_screen"
                        _write_event(
                            events,
                            {
                                "timestamp_ms": round((now - started) * 1000),
                                "frame_sha256": _sha_bytes(frame.tobytes()),
                                "screen_valid": False,
                                "abstain": True,
                                "rejection_reason": stop_reason,
                                "input_sent": False,
                            },
                        )
                        break
                    frames.append(views)
                    raw_frames.append(frame)
                    if len(frames) < WINDOW_FRAMES:
                        continue
                    if next_decision is None:
                        next_decision = now
                    if now < next_decision:
                        continue
                    while next_decision <= now:
                        next_decision += 1.0 / decision_hz
                    scheduled += 1
                    infer_started = time.monotonic()
                    command = _predict_command(model, frames, device)
                    infer_elapsed = (time.monotonic() - infer_started) * 1000
                    static = _is_static_window(list(raw_frames))
                    abstain = command.confidence < minimum_confidence or static
                    rejection = "low_confidence_or_static" if abstain else ""
                    proposed = (command.intent.value, command.target_zone.value)
                    elapsed_ms = round((now - started) * 1000)
                    if (
                        not abstain
                        and accepted_command is not None
                        and accepted_command != proposed
                        and elapsed_ms - accepted_at_ms < minimum_hold_ms
                    ):
                        proposed = accepted_command
                        rejection = "minimum_hold"
                    if abstain:
                        abstentions += 1
                    else:
                        if proposed != previous_command:
                            transitions += int(previous_command is not None)
                        previous_command = proposed
                        accepted_command = proposed
                        accepted_at_ms = elapsed_ms
                    movement, combat, purchase = _candidate_modes(proposed[0], proposed[1], abstain)
                    intents[proposed[0]] += 1
                    zones[proposed[1]] += 1
                    candidates[f"{movement}:{combat}:{purchase}"] += 1
                    inference_ms.append(infer_elapsed)
                    end_to_end_ms.append((time.monotonic() - captured) * 1000)
                    _write_event(
                        events,
                        {
                            "timestamp_ms": elapsed_ms,
                            "frame_sha256": _sha_bytes(frame.tobytes()),
                            "screen_valid": True,
                            "foreground_valid": True,
                            "frame_fresh": True,
                            "intent": proposed[0],
                            "target_zone": proposed[1],
                            "confidence": command.confidence,
                            "abstain": abstain,
                            "candidate_movement": movement,
                            "candidate_combat_mode": combat,
                            "candidate_purchase": purchase,
                            "rejection_reason": rejection,
                            "capture_ms": 0.0,
                            "inference_ms": infer_elapsed,
                            "end_to_end_ms": end_to_end_ms[-1],
                            "input_sent": False,
                        },
                    )
                    completed += 1
                else:
                    stop_reason = "capture_ended"
    except (av.FFmpegError, GlobalShadowError, MobileTestbedError) as exc:
        stop_reason = "guard_or_capture_error"
        hard_stops += 1
        error = str(exc)
    finally:
        watchdog.stop()
    coverage = completed / max(1, scheduled)
    if roi_sha256 is None:
        schema = CANDIDATE_SCHEMA if candidate else SCHEMA
    else:
        schema = (
            "hok-agent-global-shadow-candidate-v1.1"
            if candidate
            else "hok-agent-global-shadow-v1.1"
        )
    summary: dict[str, object] = {
        "schema_version": schema,
        "candidate_run": candidate,
        "status": "COMPLETED" if stop_reason is None else "STOPPED",
        "run_seconds_requested": run_seconds,
        "run_seconds_observed": time.monotonic() - started,
        "config_sha256": config_sha256,
        "authorization_sha256": authorization_sha256,
        "observation_roi_sha256": roi_sha256,
        "checkpoint_sha256": (
            CANDIDATE_CHECKPOINT_SHA256 if candidate else PROMOTED_CHECKPOINT_SHA256
        ),
        "scheduled_cycles": scheduled,
        "completed_cycles": completed,
        "cycle_coverage": coverage,
        "achieved_inference_hz": completed / max(1e-9, time.monotonic() - started),
        "p50_inference_ms": _percentile(inference_ms, 0.50),
        "p95_inference_ms": _percentile(inference_ms, 0.95),
        "p50_end_to_end_ms": _percentile(end_to_end_ms, 0.50),
        "p95_end_to_end_ms": _percentile(end_to_end_ms, 0.95),
        "stale_frame_count": stale_frames,
        "invalid_screen_cycles": invalid_screen,
        "hard_stop_cycles": hard_stops,
        "abstain_rate": abstentions / max(1, completed),
        "intent_counts": dict(sorted(intents.items())),
        "zone_counts": dict(sorted(zones.items())),
        "macro_transition_count": transitions,
        "candidate_action_counts": dict(sorted(candidates.items())),
        "safety_violations": 0,
        "input_commands_sent": 0,
        "control_output": False,
        "device_input_allowed": False,
        "stop_reason": stop_reason,
        "error": error,
    }
    passed = (
        summary["status"] == "COMPLETED"
        and coverage >= cast(float, config["minimum_cycle_coverage"])
        and cast(float, summary["p95_end_to_end_ms"])
        < cast(float, config["maximum_p95_end_to_end_ms"])
        and hard_stops == 0
        and summary["input_commands_sent"] == 0
    )
    summary["status"] = "PASSED" if passed else "FAILED"
    summary["summary_sha256"] = _sha_bytes(_canonical(summary).encode())
    (output / "summary.json").write_text(_canonical(summary) + "\n", encoding="utf-8")
    return summary
