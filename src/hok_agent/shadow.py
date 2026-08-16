# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import BinaryIO

import av
import numpy as np

from hok_agent.pixel import infer_rgb_frames

SCHEMA = "pixelarena-shadow-diagnostic-v1"
FROZEN_V3_MODEL_SHA256 = "df511e9b19327886da359400055dcc99aad6520a495c6d5e0495031c86b44eed"
VIDEO_FORMATS = {".avi": "avi", ".mkv": "matroska", ".mov": "mov", ".mp4": "mov", ".webm": "matroska"}
ACTION_LABELS = ("wait", "forward", "backward", "attack_hero", "attack_tower", "attack_crystal")
DENIED_ROOTS = (Path("/dev"), Path("/proc"), Path("/sys"))


class ShadowError(ValueError):
    pass


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_handle(handle: BinaryIO) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _local_regular_file(raw: str, suffixes: set[str]) -> Path:
    value = raw.strip()
    lowered = value.lower()
    if not value or value == "-" or value.isdigit() or "://" in value:
        raise ShadowError("input must be a local regular file path")
    if lowered.startswith(("file:", "http:", "https:", "rtsp:", "tcp:", "udp:")):
        raise ShadowError("URI and device inputs are not allowed")
    path = Path(value)
    try:
        info = path.lstat()
    except OSError as exc:
        raise ShadowError("local input file is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ShadowError("input must be a non-symlink regular file")
    resolved = path.resolve(strict=True)
    if any(resolved == root or root in resolved.parents for root in DENIED_ROOTS):
        raise ShadowError("device and pseudo-filesystem paths are not allowed")
    if resolved.suffix.lower() not in suffixes:
        raise ShadowError("unsupported local file type")
    return resolved


def _open_checked(path: Path) -> BinaryIO:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ShadowError("opened input is not a regular file")
        return os.fdopen(fd, "rb")
    except OSError as exc:
        raise ShadowError("local input could not be opened read-only") from exc


def _output_path(raw: Path) -> Path:
    if raw.exists() or raw.is_symlink():
        raise ShadowError("output directory already exists")
    parent = raw.parent.resolve()
    if any(parent == root or root in parent.parents for root in DENIED_ROOTS):
        raise ShadowError("output must be on a regular local filesystem")
    parent.mkdir(parents=True, exist_ok=True)
    return parent / raw.name


def _decode_video(source: BinaryIO, container_format: str, sample_every: int, max_frames: int) -> tuple[np.ndarray, list[tuple[int, int | None]], int]:
    if not 1 <= sample_every <= 30 or not 1 <= max_frames <= 300:
        raise ShadowError("sample-every must be 1..30 and max-frames must be 1..300")
    images: list[np.ndarray] = []
    positions: list[tuple[int, int | None]] = []
    decoded = 0
    try:
        source.seek(0)
        with av.open(source, mode="r", format=container_format, options={"protocol_whitelist": ""}) as container:
            if not container.streams.video:
                raise ShadowError("video contains no video stream")
            for index, frame in enumerate(container.decode(video=0)):
                decoded += 1
                if index % sample_every:
                    continue
                image = frame.reformat(width=128, height=128, format="rgb24").to_ndarray()
                images.append(np.ascontiguousarray(image, dtype=np.uint8))
                positions.append((index, None if frame.pts is None else int(frame.pts)))
                if len(images) == max_frames:
                    break
    except av.error.FFmpegError as exc:
        raise ShadowError("local video decode failed") from exc
    if not images:
        raise ShadowError("video produced no sampled frames")
    return np.stack(images), positions, decoded


def analyze_video(video_raw: str, model_raw: str, output_raw: Path, device: str = "cpu", sample_every: int = 5, max_frames: int = 300) -> dict[str, object]:
    """Analyze a local recording without producing a commercial-client action."""
    video = _local_regular_file(video_raw, set(VIDEO_FORMATS))
    model = _local_regular_file(model_raw, {".safetensors"})
    output = _output_path(output_raw)
    with _open_checked(video) as source, _open_checked(model) as model_source:
        before = os.fstat(source.fileno())
        source_hash = _sha_handle(source)
        model_hash = _sha_handle(model_source)
        if model_hash != FROZEN_V3_MODEL_SHA256:
            raise ShadowError("model is not the frozen promoted V3 checkpoint")
        frames, positions, decoded = _decode_video(source, VIDEO_FORMATS[video.suffix.lower()], sample_every, max_frames)
        model_fd = Path(f"/proc/self/fd/{model_source.fileno()}")
        predictions, confidences, model_seed = infer_rgb_frames(model_fd, frames, device)
        after = os.fstat(source.fileno())
    rows = []
    for sequence, ((source_frame, pts), prediction, confidence, frame) in enumerate(zip(positions, predictions, confidences, frames, strict=True)):
        rows.append({"schema_version": SCHEMA, "sequence": sequence, "source_frame": source_frame, "pts": pts, "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(), "raw_model_hypothesis": ACTION_LABELS[prediction], "confidence": round(float(confidence), 8), "advisory_action": "ABSTAIN", "abstain_reason": "UNVALIDATED_COMMERCIAL_DOMAIN", "control_output": False})
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ShadowError("input changed during analysis")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        predictions_path = staging / "predictions.jsonl"
        predictions_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        summary: dict[str, object] = {
            "schema_version": SCHEMA,
            "status": "PASSED",
            "disposition": "OFFLINE_SHADOW_DIAGNOSTIC",
            "source_sha256": source_hash,
            "model_sha256": model_hash,
            "model_seed": model_seed,
            "device": device,
            "decoded_frames": decoded,
            "analyzed_frames": len(rows),
            "abstain_count": len(rows),
            "advisory_count": 0,
            "predictions_sha256": _sha_file(predictions_path),
            "real_domain_validated": False,
            "promotion_eligible": False,
            "commercial_client_action_output": False,
            "hok_capability_claim": False,
            "gamecore_equivalence_claim": False,
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.rename(output)
    return summary
