from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

CaptureSource = Literal["pixelarena", "offline_video", "v4l2", "self_built_mobile_test_app"]
ViewName = Literal["main", "minimap", "hud", "equipment"]

_CAPTURE_SOURCES = {"pixelarena", "offline_video", "v4l2", "self_built_mobile_test_app"}
_REQUIRED_POLICY_VIEWS = {"main", "minimap", "hud"}


class FramePacketRecord(TypedDict):
    observation_id: str
    capture_start_ns: int
    capture_end_ns: int
    capture_source_class: CaptureSource
    frame_bundle_ref: str
    view_sha256: dict[str, str]
    capture_latency_ms: float
    source_locator_persisted: Literal[False]


@dataclass(frozen=True, slots=True)
class RgbView:
    name: ViewName
    rgb: bytes
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        height, width, channels = self.shape
        if height <= 0 or width <= 0 or channels != 3:
            raise ValueError("RGB view shape must be positive HxWx3")
        if len(self.rgb) != height * width * channels:
            raise ValueError("RGB view bytes do not match its declared shape")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.rgb).hexdigest()


@dataclass(frozen=True, slots=True)
class FramePacket:
    observation_id: str
    capture_start_ns: int
    capture_end_ns: int
    capture_source_class: CaptureSource
    frame_bundle_ref: str
    views: tuple[RgbView, ...]

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must not be empty")
        if self.capture_start_ns < 0 or self.capture_end_ns < self.capture_start_ns:
            raise ValueError("capture timestamps are invalid")
        if self.capture_source_class not in _CAPTURE_SOURCES:
            raise ValueError("capture source is outside the FrameBus contract")
        if not self.frame_bundle_ref or Path(self.frame_bundle_ref).name != self.frame_bundle_ref:
            raise ValueError("frame_bundle_ref must be an anonymous basename")
        names = [view.name for view in self.views]
        if len(names) != len(set(names)):
            raise ValueError("FramePacket view names must be unique")
        if not _REQUIRED_POLICY_VIEWS.issubset(names):
            raise ValueError("FramePacket must contain main, minimap, and hud views")

    @property
    def capture_latency_ms(self) -> float:
        return (self.capture_end_ns - self.capture_start_ns) / 1_000_000.0

    def view(self, name: ViewName) -> RgbView:
        return next(view for view in self.views if view.name == name)

    def to_record(self) -> FramePacketRecord:
        return {
            "observation_id": self.observation_id,
            "capture_start_ns": self.capture_start_ns,
            "capture_end_ns": self.capture_end_ns,
            "capture_source_class": self.capture_source_class,
            "frame_bundle_ref": self.frame_bundle_ref,
            "view_sha256": {view.name: view.sha256 for view in self.views},
            "capture_latency_ms": self.capture_latency_ms,
            "source_locator_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    generation: int
    packet: FramePacket


class LatestFrameBus:
    """Thread-safe single-slot FrameBus with no frame backlog."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._generation = 0
        self._latest: FramePacket | None = None

    def publish(self, packet: FramePacket) -> FrameSnapshot:
        with self._condition:
            if self._latest is not None:
                if packet.observation_id == self._latest.observation_id:
                    raise ValueError("observation_id must be unique")
                if packet.capture_end_ns <= self._latest.capture_end_ns:
                    raise ValueError("FrameBus capture time must increase")
            self._generation += 1
            self._latest = packet
            snapshot = FrameSnapshot(self._generation, packet)
            self._condition.notify_all()
            return snapshot

    def snapshot(self) -> FrameSnapshot | None:
        with self._condition:
            if self._latest is None:
                return None
            return FrameSnapshot(self._generation, self._latest)

    def wait_for_newer(
        self,
        after_generation: int,
        timeout_s: float | None,
    ) -> FrameSnapshot | None:
        if after_generation < 0:
            raise ValueError("after_generation must be non-negative")
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._generation > after_generation,
                timeout=timeout_s,
            )
            if not ready or self._latest is None:
                return None
            return FrameSnapshot(self._generation, self._latest)
