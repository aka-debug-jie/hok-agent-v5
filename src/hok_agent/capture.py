# ruff: noqa: E501
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import queue
import re
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import av
import numpy as np

from hok_agent import pixel, shadow

SCHEMA = "pixelarena-shadow-live-v1"
ACTION_LABELS = ("wait", "forward", "backward", "attack_hero", "attack_tower", "attack_crystal")
DENIED_ROOTS = (Path("/proc"), Path("/sys"))


class CaptureError(ValueError):
    pass


FrameSource = Callable[[], Any]
FrameRecord = tuple[float, np.ndarray]
EventSink = Callable[[dict[str, object]], None]
MetricReport = dict[str, float | int]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_capture_device(raw: str | None) -> Path:
    if raw is None:
        raise CaptureError("input must be an explicit /dev/videoN node")
    value = raw.strip()
    if not re.fullmatch(r"/dev/video[0-9]+", value):
        raise CaptureError("input must be exactly a /dev/videoN character device path")
    path = Path(value)
    try:
        info = path.lstat()
    except OSError as exc:
        raise CaptureError("capture device is unavailable") from exc
    if path.is_symlink():
        raise CaptureError("capture device must not be a symlink")
    if not stat.S_ISCHR(info.st_mode):
        raise CaptureError("input must be a V4L2 character device")
    resolved = path.resolve(strict=True)
    if any(root in resolved.parents for root in DENIED_ROOTS):
        raise CaptureError("cannot read from pseudo-filesystem nodes")
    node = Path("/sys/class/video4linux") / resolved.name
    if not node.exists():
        raise CaptureError("input must be an available V4L2 character device")
    return resolved


def _validate_model(model_raw: str) -> Path:
    value = model_raw.strip()
    if not value or value == "-" or value.isdigit() or "://" in value:
        raise CaptureError("model must be a local file path")
    if value.startswith(("file:", "http:", "https:", "tcp:", "udp:", "rtsp:")):
        raise CaptureError("network URI and pseudo-device paths are not allowed")
    path = Path(value)
    try:
        info = path.lstat()
    except OSError as exc:
        raise CaptureError("model file is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise CaptureError("model must be a non-symlink regular file")
    if path.suffix.lower() != ".safetensors":
        raise CaptureError("model must be a safetensors file")
    resolved = path.resolve()
    if any(root in resolved.parents for root in DENIED_ROOTS):
        raise CaptureError("invalid model path")
    return resolved


def _validate_output(output_raw: Path) -> Path:
    if output_raw.exists() or output_raw.is_symlink():
        raise CaptureError("output directory already exists")
    parent = output_raw.parent.resolve()
    if any(parent == root or root in parent.parents for root in DENIED_ROOTS):
        raise CaptureError("output must be on a regular local filesystem")
    parent.mkdir(parents=True, exist_ok=True)
    return output_raw


def _parse_size(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", raw.strip())
    if not match:
        raise CaptureError("capture-size must be WIDTHxHEIGHT")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0 or width > 10000 or height > 10000:
        raise CaptureError("capture-size must be positive and bounded")
    return width, height


def _resize_for_model(frame: np.ndarray) -> np.ndarray:
    if frame.shape[0] == 128 and frame.shape[1] == 128:
        return frame
    row_indices = np.linspace(0, frame.shape[0] - 1, 128).astype(np.int64)
    col_indices = np.linspace(0, frame.shape[1] - 1, 128).astype(np.int64)
    return frame[row_indices[:, None], col_indices[None, :], :]


def _coerce_frame(raw: Any) -> np.ndarray:
    if isinstance(raw, tuple):
        if len(raw) != 2:
            raise CaptureError("frame source tuple must be (timestamp, frame)")
        raw = raw[1]
    if not isinstance(raw, np.ndarray):
        raise CaptureError("frame source must yield uint8 ndarray")
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise CaptureError("frame must be HxWx3")
    if raw.dtype != np.uint8:
        raise CaptureError("frame must be uint8")
    return np.ascontiguousarray(_resize_for_model(raw))


def _coerce_source_frame(item: Any) -> FrameRecord:
    if isinstance(item, tuple):
        if len(item) != 2:
            raise CaptureError("frame source tuple must be (timestamp, frame)")
        timestamp, frame = item
        return (float(timestamp), _coerce_frame(frame))
    return (time.monotonic(), _coerce_frame(item))


def _iter_injected_source(source: FrameSource) -> Iterator[FrameRecord]:
    if not callable(source):
        raise CaptureError("injectable frame source must be callable")
    while True:
        value = source()
        if value is None:
            return
        yield _coerce_source_frame(value)


def _latest_frame_put(target: queue.Queue[FrameRecord], payload: FrameRecord) -> None:
    try:
        target.put_nowait(payload)
        return
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            target.get_nowait()
        target.put_nowait(payload)


def _latest_frame_get(target: queue.Queue[FrameRecord]) -> FrameRecord | None:
    latest: FrameRecord | None = None
    while True:
        try:
            latest = target.get_nowait()
        except queue.Empty:
            return latest


def _producer_from_injection(source: FrameSource, output: queue.Queue[FrameRecord], stop: threading.Event, done: threading.Event, errors: list[BaseException]) -> threading.Thread:
    def _run() -> None:
        try:
            for timestamp, frame in _iter_injected_source(source):
                if stop.is_set():
                    return
                _latest_frame_put(output, (timestamp, frame))
        except BaseException as exc:  # pragma: no cover - defensive path
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _producer_from_device(device: Path, capture_size: tuple[int, int], source_report: dict[str, object], output: queue.Queue[FrameRecord], stop: threading.Event, done: threading.Event, errors: list[BaseException], fps: int) -> threading.Thread:
    width, height = capture_size

    def _run() -> None:
        options = {"input_format": "v4l2", "framerate": str(fps), "video_size": f"{width}x{height}"}
        try:
            with av.open(str(device), mode="r", format="video4linux2", options=options) as container:
                if not container.streams.video:
                    raise CaptureError("device has no video stream")
                stream = container.streams.video[0]
                source_report["actual_width"] = int(stream.width or width)
                source_report["actual_height"] = int(stream.height or height)
                if stream.average_rate is None:
                    source_report["average_rate"] = None
                else:
                    source_report["average_rate"] = float(stream.average_rate)
                previous_ts: float | None = None
                intervals: list[float] = []
                for frame in container.decode(video=stream.index):
                    if stop.is_set():
                        return
                    captured_at = time.monotonic()
                    if source_report["average_rate"] is None and frame.pts is not None and frame.time_base is not None:
                        current_ts = float(frame.pts * frame.time_base)
                        if previous_ts is not None:
                            delta = current_ts - previous_ts
                            if delta > 0:
                                intervals.append(delta)
                        previous_ts = current_ts
                    rgb = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
                    resized = _resize_for_model(np.ascontiguousarray(rgb, dtype=np.uint8))
                    _latest_frame_put(output, (captured_at, resized))
                if source_report["average_rate"] is None and intervals:
                    mean_interval = sum(intervals) / len(intervals)
                    if mean_interval > 0:
                        source_report["average_rate"] = 1.0 / mean_interval
        except BaseException as exc:  # pragma: no cover - hardware-dependent
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _metrics(values: list[float]) -> MetricReport:
    if not values:
        return {"count": 0, "min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    ordered = sorted(values)
    arr = np.array(values, dtype=np.float64)
    return {"count": len(values), "min_ms": float(ordered[0]), "max_ms": float(ordered[-1]), "mean_ms": float(arr.mean()), "p50_ms": float(np.quantile(arr, 0.50)), "p90_ms": float(np.quantile(arr, 0.90)), "p95_ms": float(np.quantile(arr, 0.95)), "p99_ms": float(np.quantile(arr, 0.99))}


def _validate_frozen_model(path: Path) -> str:
    model_sha256 = _sha_file(path)
    if model_sha256 != shadow.FROZEN_V3_MODEL_SHA256:
        raise CaptureError("model is not the frozen promoted V3 checkpoint")
    return model_sha256


def _load_frozen_predictor(path: Path, device: str) -> tuple[Callable[[np.ndarray], tuple[list[int], list[float]]], int, str]:
    with shadow._open_checked(path) as handle:
        model_sha256 = shadow._sha_handle(handle)
        if model_sha256 != shadow.FROZEN_V3_MODEL_SHA256:
            raise CaptureError("model is not the frozen promoted V3 checkpoint")
        descriptor = Path(f"/proc/self/fd/{handle.fileno()}")
        predict, seed = pixel.open_rgb_predictor(descriptor, device)
    return predict, seed, model_sha256


def run_shadow_live(input_device: str | None, model_raw: str, output_dir: Path, *, device: str = "cpu", capture_size: str = "1920x1080", capture_fps: int = 60, infer_hz: int = 10, max_frames: int | None = None, run_seconds: float | None = None, frame_source: FrameSource | None = None, event_sink: EventSink | None = None) -> dict[str, object]:
    if max_frames is not None and max_frames < 0:
        raise CaptureError("max_frames must be >= 0")
    if run_seconds is not None and run_seconds <= 0:
        raise CaptureError("run_seconds must be > 0")
    if infer_hz <= 0:
        raise CaptureError("infer-hz must be > 0")
    if capture_fps <= 0:
        raise CaptureError("capture-fps must be > 0")
    if frame_source is None and input_device is None:
        raise CaptureError("frame source or /dev/videoN input is required")
    if frame_source is None and max_frames is None and run_seconds is None:
        raise CaptureError("live capture requires max_frames or run_seconds")

    model = _validate_model(model_raw)
    predict, model_seed, model_sha256 = _load_frozen_predictor(model, device)
    output = _validate_output(output_dir)
    size = _parse_size(capture_size)
    device_path = _validate_capture_device(input_device) if frame_source is None else None

    frame_queue: queue.Queue[FrameRecord] = queue.Queue(maxsize=1)
    stop = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []
    rows: list[dict[str, object]] = []
    on_time_events: list[float] = []
    latency_ms: list[float] = []
    schedule_ticks = 0
    on_time_ticks = 0
    source_frames = 0
    source_report: dict[str, object] = {"is_simulated": frame_source is not None}

    if frame_source is not None:
        producer = _producer_from_injection(frame_source, frame_queue, stop, done, errors)
    else:
        if device_path is None:
            raise CaptureError("validated capture device is unavailable")
        producer = _producer_from_device(device_path, size, source_report, frame_queue, stop, done, errors, capture_fps)

    interval = 1.0 / float(infer_hz)
    next_tick = time.monotonic()
    start = next_tick
    while True:
        elapsed = time.monotonic() - start
        if run_seconds is not None and elapsed >= run_seconds:
            break
        if max_frames is not None and len(rows) >= max_frames:
            break

        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        schedule_time = next_tick
        deadline = schedule_time + interval
        next_tick += interval
        schedule_ticks += 1

        latest = _latest_frame_get(frame_queue)
        if latest is None:
            if done.is_set():
                break
            time.sleep(0.001)
            continue

        captured_at, frame = latest
        if frame.shape != (128, 128, 3):
            raise CaptureError("model capture requires 128x128 frames")

        source_frames += 1
        inference_start = time.monotonic()
        predictions, confidences = predict(frame[None, ...])
        inference_done = time.monotonic()
        inference_ms = (inference_done - inference_start) * 1000.0
        on_time_events.append(inference_ms)

        row: dict[str, object] = {"schema_version": SCHEMA, "sequence": len(rows), "source_frame_timestamp": captured_at, "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(), "raw_model_hypothesis": ACTION_LABELS[int(predictions[0])], "confidence": round(float(confidences[0]), 8), "advisory_action": "ABSTAIN", "abstain_reason": "UNVALIDATED_COMMERCIAL_DOMAIN", "control_output": False, "queue_delay_ms": round((inference_start - captured_at) * 1000.0, 4), "inference_ms": round(inference_ms, 4)}
        if event_sink is not None:
            event_sink(row)
        terminal_done = time.monotonic()
        capture_to_terminal_ms = (terminal_done - captured_at) * 1000.0
        row["capture_to_terminal_ms"] = round(capture_to_terminal_ms, 4)
        latency_ms.append(capture_to_terminal_ms)
        if terminal_done <= deadline:
            on_time_ticks += 1
        rows.append(row)

    elapsed_seconds = time.monotonic() - start

    inference_metrics = _metrics(on_time_events)
    end_to_end_metrics = _metrics(latency_ms)
    on_time_ratio = on_time_ticks / schedule_ticks if schedule_ticks else 0.0
    source_is_simulated = bool(source_report["is_simulated"])
    raw_width = source_report.get("actual_width", 0)
    raw_height = source_report.get("actual_height", 0)
    stream_width = raw_width if isinstance(raw_width, int) else 0
    stream_height = raw_height if isinstance(raw_height, int) else 0
    stream_rate = source_report.get("average_rate")
    formal_stream_match = False
    if not source_is_simulated and stream_width == 1920 and stream_height == 1080 and isinstance(stream_rate, (int, float)) and math.isclose(float(stream_rate), 60.0, rel_tol=0.05, abs_tol=0.5):
        formal_stream_match = True

    if len(rows) == 0:
        status = "FAILED"
        disposition = "LIVE_SHADOW_FAILED"
    elif not source_is_simulated and size == (1920, 1080) and capture_fps == 60 and infer_hz == 10 and run_seconds is not None and math.isclose(run_seconds, 600.0, rel_tol=0.0, abs_tol=1e-9) and elapsed_seconds >= run_seconds and schedule_ticks >= int(run_seconds * infer_hz) and len(rows) / schedule_ticks >= 0.99 and on_time_ratio >= 0.99 and end_to_end_metrics["p95_ms"] <= 100.0 and formal_stream_match and event_sink is not None:
        status = "FORMAL_PASSED"
        disposition = "FORMAL_LIVE_SHADOW"
    else:
        status = "SMOKE_PASSED"
        disposition = "SMOKE_LIVE_SHADOW"

    stop.set()
    producer.join(timeout=1.0)

    if errors:
        raise CaptureError(f"capture failed: {errors[0]}")

    summary: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": status,
        "disposition": disposition,
        "real_domain_validated": False,
        "advisory_action": "ABSTAIN",
        "control_output": False,
        "model_sha256": "sha256:" + model_sha256,
        "model_seed": model_seed,
        "captured_frames": source_frames,
        "analyzed_frames": len(rows),
        "abstain_count": len(rows),
        "advisory_count": 0,
        "schedule": {"target_infer_hz": infer_hz, "target_interval_s": interval, "target_ticks": schedule_ticks, "actual_infer_ticks": len(rows), "elapsed_seconds": round(elapsed_seconds, 6), "inference_coverage": round(len(rows) / schedule_ticks if schedule_ticks else 0.0, 6), "terminal_output_enabled": event_sink is not None},
        "on_time": {"late_ticks": schedule_ticks - on_time_ticks, "on_time_ticks": on_time_ticks, "on_time_ratio": round(on_time_ratio, 6)},
        "latency": {"capture_to_terminal_ms": end_to_end_metrics, "inference_ms": inference_metrics},
        "capture_size": f"{size[0]}x{size[1]}",
        "source": {
            "is_simulated": source_is_simulated,
            **({"stream_width": stream_width, "stream_height": stream_height, "stream_average_rate": stream_rate} if stream_width and stream_height else {"stream_average_rate": stream_rate}),
        },
    }

    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        events_path = staging / "events.jsonl"
        summary_path = staging / "summary.json"
        events_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        summary_path.write_text(summary_text, encoding="utf-8")
        staging.rename(output)
    return summary
