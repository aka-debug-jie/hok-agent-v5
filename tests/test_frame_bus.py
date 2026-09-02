from __future__ import annotations

from pathlib import Path

import pytest

from hok_agent.frame_bus import FramePacket, LatestFrameBus, RgbView


def _view(name: str, value: int = 0) -> RgbView:
    return RgbView(name=name, rgb=bytes([value]) * 12, shape=(2, 2, 3))  # type: ignore[arg-type]


def _packet(sequence: int) -> FramePacket:
    start = sequence * 1_000_000
    return FramePacket(
        observation_id=f"obs-{sequence:03d}",
        capture_start_ns=start,
        capture_end_ns=start + 500_000,
        capture_source_class="pixelarena",
        frame_bundle_ref=f"obs-{sequence:03d}.npz",
        views=(_view("main", sequence), _view("minimap", sequence), _view("hud", sequence)),
    )


def test_frame_packet_emits_hash_only_record() -> None:
    packet = _packet(1)
    record = packet.to_record()
    assert record["observation_id"] == "obs-001"
    assert record["capture_latency_ms"] == 0.5
    assert set(record["view_sha256"]) == {"main", "minimap", "hud"}
    assert all(len(value) == 64 for value in record["view_sha256"].values())
    assert record["source_locator_persisted"] is False


def test_frame_packet_requires_policy_views_and_anonymous_reference() -> None:
    with pytest.raises(ValueError, match="main, minimap, and hud"):
        FramePacket("obs", 1, 2, "pixelarena", "obs.npz", (_view("main"),))
    with pytest.raises(ValueError, match="anonymous basename"):
        FramePacket(
            "obs",
            1,
            2,
            "pixelarena",
            str(Path("private") / "obs.npz"),
            (_view("main"), _view("minimap"), _view("hud")),
        )


def test_latest_frame_bus_replaces_old_frames_without_backlog() -> None:
    bus = LatestFrameBus()
    first = bus.publish(_packet(1))
    bus.publish(_packet(2))
    third = bus.publish(_packet(3))
    snapshot = bus.snapshot()
    assert first.generation == 1
    assert third.generation == 3
    assert snapshot is not None
    assert snapshot.packet.observation_id == "obs-003"
    assert bus.wait_for_newer(2, timeout_s=0) == third
    assert bus.wait_for_newer(3, timeout_s=0) is None


def test_latest_frame_bus_rejects_duplicate_or_older_capture() -> None:
    bus = LatestFrameBus()
    bus.publish(_packet(2))
    with pytest.raises(ValueError, match="capture time must increase"):
        bus.publish(_packet(1))
    duplicate = FramePacket(
        observation_id="obs-002",
        capture_start_ns=3_000_000,
        capture_end_ns=3_500_000,
        capture_source_class="pixelarena",
        frame_bundle_ref="duplicate.npz",
        views=(_view("main"), _view("minimap"), _view("hud")),
    )
    with pytest.raises(ValueError, match="observation_id must be unique"):
        bus.publish(duplicate)

