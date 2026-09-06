from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import pytest

from hok_agent.frame_bus import FramePacket, FramePacketRecord, RgbView, ViewName
from hok_agent.transition_store import (
    ExecutedActionRecord,
    HierarchicalTransitionRecord,
    PolicyProposalRecord,
    ProposalBundleRecord,
    ReplayRecord,
    RewardComponentsRecord,
    RewardRecord,
    UnifiedTransitionStore,
    validate_transition,
)
from hok_agent.visual_events import EventSource, VisualEvent, VisualEventType


def _frame(observation_id: str, start_ns: int, end_ns: int) -> FramePacketRecord:
    def view(name: ViewName) -> RgbView:
        return RgbView(name, b"\x00" * 12, (2, 2, 3))

    packet = FramePacket(
        observation_id=observation_id,
        capture_start_ns=start_ns,
        capture_end_ns=end_ns,
        capture_source_class="pixelarena",
        frame_bundle_ref=f"{observation_id}.npz",
        views=(view("main"), view("minimap"), view("hud")),
    )
    return packet.to_record()


def _proposal(
    observation_id: str,
    value: str,
    *,
    applied_observation_id: str | None = None,
    carried_forward: bool = False,
    decision_start_ns: int = 3_000_000,
    decision_end_ns: int = 4_000_000,
) -> PolicyProposalRecord:
    return {
        "observation_id": observation_id,
        "applied_observation_id": applied_observation_id or observation_id,
        "carried_forward": carried_forward,
        "value": value,
        "confidence": 0.9,
        "decision_start_ns": decision_start_ns,
        "decision_end_ns": decision_end_ns,
        "valid_until_ns": 20_000_000,
        "policy_bundle_version": "bundle-v0",
    }


def _transition(*, terminal: bool = False) -> HierarchicalTransitionRecord:
    proposals: ProposalBundleRecord = {
        "macro": _proposal("obs-000", "HOLD"),
        "movement": _proposal("obs-000", "NONE"),
        "combat": _proposal("obs-000", "WAIT"),
    }
    executed: ExecutedActionRecord = {
        "requested_movement": "NONE",
        "applied_movement": "NONE",
        "requested_combat": "WAIT",
        "applied_combat": "WAIT",
        "movement_command": "NOOP",
        "combat_command": "NOOP",
        "dispatch_start_ns": 5_000_000,
        "dispatch_ack_ns": 6_000_000,
        "first_attempt_status": "not_attempted",
        "retry_status": "not_attempted",
        "retry_count": 0,
        "final_status": "noop",
    }
    events = []
    event_ids: list[str] = []
    components: RewardComponentsRecord = {
        "terminal": 0.0,
        "death": 0.0,
        "self_hp_delta": 0.0,
        "tower_damage": 0.0,
    }
    if terminal:
        event = VisualEvent(
            event_id="win-001",
            event_type=VisualEventType.WIN,
            start_ns=7_000_000,
            end_ns=9_000_000,
            old_value="IN_PROGRESS",
            new_value="WIN",
            delta=None,
            confidence=1.0,
            roi_id="terminal",
            dedup_key="episode-win",
            source=EventSource.RGB_OCR,
        )
        events.append(event.to_record())
        event_ids.append(event.event_id)
        components["terminal"] = 1.0
    reward: RewardRecord = {
        "reward_version": "reward-v0",
        "components": components,
        "total": sum(components.values()),
        "event_ids": event_ids,
    }
    replay: ReplayRecord = {"source": "sim", "failure_tags": [], "priority": 1.0}
    return {
        "schema_version": "hok-agent-hierarchical-transition-v0",
        "episode_id": "episode-001",
        "step_id": 0,
        "policy_bundle_version": "bundle-v0",
        "policy_bundle_sha256": "0" * 64,
        "event_engine_version": "event-v0",
        "event_engine_sha256": "1" * 64,
        "observation": _frame("obs-000", 1_000_000, 2_000_000),
        "proposals": proposals,
        "executed_action": executed,
        "settle_end_ns": 7_000_000,
        "next_observation": _frame("obs-001", 8_000_000, 9_000_000),
        "events": events,
        "reward": reward,
        "done": terminal,
        "terminal_reason": "WIN" if terminal else "NOT_DONE",
        "causal_order_valid": True,
        "training_eligible": True,
        "replay": replay,
    }


def test_valid_transition_matches_the_serial_causal_step() -> None:
    result = validate_transition(_transition())
    assert result.valid is True
    assert result.causal_order_valid is True


def test_transition_store_commits_terminal_before_episode_exit(tmp_path: Path) -> None:
    path = tmp_path / "replay.sqlite3"
    with UnifiedTransitionStore(path) as store:
        stored = store.append(_transition(terminal=True))
        assert stored.validation.valid is True
        assert store.count(training_only=True) == 1
        loaded = store.load_episode("episode-001")
        assert len(loaded) == 1
        assert loaded[0]["done"] is True
        assert loaded[0]["terminal_reason"] == "WIN"
        assert loaded[0]["reward"]["event_ids"] == ["win-001"]


def test_transition_store_is_transactional_and_reopenable(tmp_path: Path) -> None:
    path = tmp_path / "replay.sqlite3"
    with UnifiedTransitionStore(path) as store:
        store.append(_transition())
        with pytest.raises(sqlite3.IntegrityError):
            store.append(_transition())
        assert store.count() == 1
    with UnifiedTransitionStore(path) as reopened:
        assert reopened.count() == 1
        assert reopened.load_episode("episode-001")[0]["step_id"] == 0


def test_invalid_causal_transition_is_preserved_but_not_trainable(tmp_path: Path) -> None:
    row = _transition()
    row["next_observation"]["capture_start_ns"] = 6_500_000
    with UnifiedTransitionStore(tmp_path / "replay.sqlite3") as store:
        stored = store.append(row)
        assert stored.validation.causal_order_valid is False
        assert "next_capture_before_settle" in stored.validation.errors
        assert stored.payload["training_eligible"] is False
        assert "next_capture_before_settle" in stored.payload["ineligibility_reasons"]
        assert store.count() == 1
        assert store.count(training_only=True) == 0


def test_intentionally_nontraining_transition_keeps_causal_episode_chain(tmp_path: Path) -> None:
    first = _transition()
    first["training_eligible"] = False
    first["ineligibility_reasons"] = ["semantic_accuracy_unverified"]
    second = copy.deepcopy(first)
    second["step_id"] = 1
    second["observation"] = _frame("obs-001", 8_000_000, 9_000_000)
    second["next_observation"] = _frame("obs-002", 15_000_000, 16_000_000)
    for proposal in second["proposals"].values():
        proposal["observation_id"] = "obs-001"
        proposal["applied_observation_id"] = "obs-001"
        proposal["decision_start_ns"] = 10_000_000
        proposal["decision_end_ns"] = 11_000_000
    second["executed_action"]["dispatch_start_ns"] = 12_000_000
    second["executed_action"]["dispatch_ack_ns"] = 13_000_000
    second["settle_end_ns"] = 14_000_000
    with UnifiedTransitionStore(tmp_path / "replay.sqlite3") as store:
        assert store.append(first).validation.valid is True
        stored = store.append(second)
        assert stored.validation.valid is True
        assert stored.validation.causal_order_valid is True
        assert stored.payload["training_eligible"] is False
        assert store.count() == 2 and store.count(training_only=True) == 0


def test_causally_invalid_nontraining_transition_still_poison_next_chain(tmp_path: Path) -> None:
    first = _transition()
    first["next_observation"]["capture_start_ns"] = 6_500_000
    first["training_eligible"] = False
    second = copy.deepcopy(_transition())
    second["step_id"] = 1
    second["observation"] = _frame("obs-001", 8_000_000, 9_000_000)
    second["next_observation"] = _frame("obs-002", 15_000_000, 16_000_000)
    for proposal in second["proposals"].values():
        proposal["observation_id"] = "obs-001"
        proposal["applied_observation_id"] = "obs-001"
        proposal["decision_start_ns"] = 10_000_000
        proposal["decision_end_ns"] = 11_000_000
    second["executed_action"]["dispatch_start_ns"] = 12_000_000
    second["executed_action"]["dispatch_ack_ns"] = 13_000_000
    second["settle_end_ns"] = 14_000_000
    with UnifiedTransitionStore(tmp_path / "replay.sqlite3") as store:
        assert store.append(first).validation.causal_order_valid is False
        stored = store.append(second)
        assert "previous_transition_ineligible" in stored.validation.errors
        assert stored.validation.causal_order_valid is False


def test_stale_proposal_is_rejected() -> None:
    row = _transition()
    row["proposals"]["macro"]["valid_until_ns"] = 4_500_000
    result = validate_transition(row)
    assert "macro_proposal_stale" in result.errors


def test_carried_macro_keeps_its_source_observation(tmp_path: Path) -> None:
    first = _transition()
    second = copy.deepcopy(first)
    second["step_id"] = 1
    second["observation"] = _frame("obs-001", 8_000_000, 9_000_000)
    second["next_observation"] = _frame("obs-002", 15_000_000, 16_000_000)
    second["proposals"] = {
        "macro": _proposal(
            "obs-000",
            "HOLD",
            applied_observation_id="obs-001",
            carried_forward=True,
        ),
        "movement": _proposal(
            "obs-001",
            "NONE",
            decision_start_ns=10_000_000,
            decision_end_ns=11_000_000,
        ),
        "combat": _proposal(
            "obs-001",
            "WAIT",
            decision_start_ns=10_000_000,
            decision_end_ns=11_000_000,
        ),
    }
    second["executed_action"]["dispatch_start_ns"] = 12_000_000
    second["executed_action"]["dispatch_ack_ns"] = 13_000_000
    second["settle_end_ns"] = 14_000_000
    with UnifiedTransitionStore(tmp_path / "replay.sqlite3") as store:
        assert store.append(first).validation.valid is True
        stored = store.append(second)
        assert stored.validation.valid is True
        assert store.count(training_only=True) == 2


def test_broken_episode_chain_is_preserved_but_not_trainable(tmp_path: Path) -> None:
    first = _transition()
    second = copy.deepcopy(first)
    second["step_id"] = 1
    second["observation"] = _frame("wrong-observation", 8_000_000, 9_000_000)
    second["next_observation"] = _frame("obs-002", 15_000_000, 16_000_000)
    for proposal in second["proposals"].values():
        proposal["observation_id"] = "wrong-observation"
        proposal["applied_observation_id"] = "wrong-observation"
        proposal["decision_start_ns"] = 10_000_000
        proposal["decision_end_ns"] = 11_000_000
    second["executed_action"]["dispatch_start_ns"] = 12_000_000
    second["executed_action"]["dispatch_ack_ns"] = 13_000_000
    second["settle_end_ns"] = 14_000_000
    with UnifiedTransitionStore(tmp_path / "replay.sqlite3") as store:
        store.append(first)
        stored = store.append(second)
        assert "episode_observation_chain_broken" in stored.validation.errors
        assert stored.payload["training_eligible"] is False


def test_episode_cannot_continue_after_terminal(tmp_path: Path) -> None:
    terminal = _transition(terminal=True)
    next_row = copy.deepcopy(_transition())
    next_row["step_id"] = 1
    next_row["observation"] = _frame("obs-001", 8_000_000, 9_000_000)
    next_row["next_observation"] = _frame("obs-002", 15_000_000, 16_000_000)
    for proposal in next_row["proposals"].values():
        proposal["observation_id"] = "obs-001"
        proposal["applied_observation_id"] = "obs-001"
        proposal["decision_start_ns"] = 10_000_000
        proposal["decision_end_ns"] = 11_000_000
    next_row["executed_action"]["dispatch_start_ns"] = 12_000_000
    next_row["executed_action"]["dispatch_ack_ns"] = 13_000_000
    next_row["settle_end_ns"] = 14_000_000
    with UnifiedTransitionStore(tmp_path / "replay.sqlite3") as store:
        store.append(terminal)
        stored = store.append(next_row)
        assert "episode_continues_after_terminal" in stored.validation.errors
        assert stored.payload["training_eligible"] is False


def test_mismatched_head_observation_is_rejected_from_training(tmp_path: Path) -> None:
    row = _transition()
    row["proposals"]["movement"]["applied_observation_id"] = "old-observation"
    with UnifiedTransitionStore(tmp_path / "replay.sqlite3") as store:
        stored = store.append(row)
        assert "movement_applied_observation_id_mismatch" in stored.validation.errors
        assert stored.payload["training_eligible"] is False
