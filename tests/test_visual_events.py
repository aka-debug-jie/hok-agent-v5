from __future__ import annotations

import pytest

from hok_agent.visual_events import (
    EventEngineIdentity,
    EventSource,
    ExactOnceEventFusion,
    LifeState,
    MatchState,
    VisualEvent,
    VisualEventType,
    VisualState,
)


def _death_event(event_id: str = "event-001", dedup_key: str = "death-001") -> VisualEvent:
    return VisualEvent(
        event_id=event_id,
        event_type=VisualEventType.DEATH,
        start_ns=10,
        end_ns=20,
        old_value=False,
        new_value=True,
        delta=None,
        confidence=0.9,
        roi_id="death_banner",
        dedup_key=dedup_key,
        source=EventSource.RGB_FUSION,
    )


def test_visual_state_is_rgb_hypothesis_with_bounded_values() -> None:
    state = VisualState(
        observation_id="obs-001",
        timestamp_ns=20,
        match_state=MatchState.IN_PROGRESS,
        life_state=LifeState.ALIVE,
        self_hp_ratio=0.75,
        match_confidence=0.9,
        life_confidence=0.8,
        self_hp_confidence=0.7,
    )
    assert state.self_hp_ratio == 0.75
    with pytest.raises(ValueError, match="self_hp_ratio"):
        VisualState("obs", 1, MatchState.UNKNOWN, LifeState.UNKNOWN, 1.1, 0.0, 0.0, 0.0)


def test_exact_once_fusion_rejects_duplicate_id_or_dedup_key() -> None:
    fusion = ExactOnceEventFusion(EventEngineIdentity("event-v0", "0" * 64))
    first = _death_event()
    assert fusion.accept(first) is True
    assert fusion.accept(first) is False
    assert fusion.accept(_death_event("event-002", "death-001")) is False
    assert fusion.accept(_death_event("event-002", "death-002")) is True
    assert [event.event_id for event in fusion.accepted] == ["event-001", "event-002"]


def test_event_fusion_resets_only_at_episode_boundary() -> None:
    fusion = ExactOnceEventFusion(EventEngineIdentity("event-v0", "1" * 64))
    event = _death_event()
    fusion.accept(event)
    fusion.reset_episode()
    assert fusion.accepted == ()
    assert fusion.accept(event) is True
    assert fusion.accepted[0].to_record()["source"] == "rgb_fusion"

