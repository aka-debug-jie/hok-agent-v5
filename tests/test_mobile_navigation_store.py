# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from hok_agent import mobile_navigation_store as store_runner
from hok_agent.frame_bus import FramePacket, RgbView
from hok_agent.mobile_testbed import MobileTestbedError, _goal_navigation_contract
from hok_agent.transition_store import UnifiedTransitionStore, validate_transition

ROOT = Path(__file__).resolve().parents[1]
STORE_CONTRACT = ROOT / "configs" / "movement_goal_navigation_store_v1.json"


def _store_block() -> dict[str, object]:
    contract, sha = _goal_navigation_contract(STORE_CONTRACT)
    return store_runner._store_contract(contract, sha)


def _views(step_id: int) -> tuple[tuple[RgbView, ...], dict[str, np.ndarray]]:
    arrays = {
        "main": np.full((4, 4, 3), step_id, dtype=np.uint8),
        "minimap": np.full((4, 4, 3), step_id + 1, dtype=np.uint8),
        "hud": np.full((2, 2, 3), step_id + 2, dtype=np.uint8),
        "equipment": np.full((2, 2, 3), step_id + 3, dtype=np.uint8),
    }
    views = tuple(
        RgbView(cast("object", name), value.tobytes(), value.shape)  # type: ignore[arg-type]
        for name, value in arrays.items()
    )
    return views, arrays


def _packet(tmp_path: Path, episode_id: str, step_id: int, start_ns: int, end_ns: int) -> FramePacket:
    views, arrays = _views(step_id)
    basename = f"{episode_id}-{step_id:04d}.npz"
    store_runner._write_bundle(tmp_path, basename, arrays)
    return FramePacket(
        observation_id=f"{episode_id}-obs-{step_id:04d}",
        capture_start_ns=start_ns,
        capture_end_ns=end_ns,
        capture_source_class="self_built_mobile_test_app",
        frame_bundle_ref=basename,
        views=views,
    )


def _chain(tmp_path: Path, steps: int, episode_id: str = "ep-test") -> list[FramePacket]:
    """Build a contiguous observation chain; each next capture follows the previous settle."""
    packets: list[FramePacket] = []
    start_ns = 1_000
    for step_id in range(steps + 1):
        end_ns = start_ns + 1_000
        packets.append(_packet(tmp_path, episode_id, step_id, start_ns, end_ns))
        start_ns = end_ns + 6_000_000
    return packets


def _row(
    observation: FramePacket,
    next_observation: FramePacket,
    *,
    step_id: int,
    done: bool,
    reason: str,
    end_kind: str,
) -> dict:
    base = observation.capture_end_ns
    row = store_runner._transition(
        store_contract=_store_block(),
        episode_id=observation.observation_id.rsplit("-obs-", 1)[0],
        step_id=step_id,
        observation=observation,
        next_observation=next_observation,
        requested="E",
        applied="E",
        owner="geometry_rule",
        reason="geometry_rule",
        movement_command="DOWN",
        dispatch_start_ns=base + 3_000_000,
        dispatch_ack_ns=base + 4_000_000,
        settle_end_ns=base + 5_000_000,
        final_status="acknowledged",
        retry_count=0,
        done=done,
        terminal_reason=reason,
        end_kind=end_kind,
    )
    return cast(dict, row)


def test_store_contract_validates_and_hashes_two_distinct_artifacts() -> None:
    block = _store_block()
    assert block["replay_source"] == "controller"
    assert block["maximum_steps"] > 0
    assert len(cast(str, block["policy_bundle_sha256"])) == 64
    assert len(cast(str, block["event_engine_sha256"])) == 64
    assert block["policy_bundle_sha256"] != block["event_engine_sha256"]


def test_store_contract_rejects_a_bad_block(tmp_path: Path) -> None:
    contract, sha = _goal_navigation_contract(STORE_CONTRACT)
    for key, value in (("replay_source", "pixelarena"), ("settle_ms", -1), ("maximum_steps", 0)):
        tampered = json.loads(json.dumps(contract))
        cast(dict, tampered["store"])[key] = value
        with pytest.raises(MobileTestbedError):
            store_runner._store_contract(cast(dict, tampered), sha)
    missing = json.loads(json.dumps(contract))
    cast(dict, missing["store"]).pop("reward_version")
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(cast(dict, missing), sha)


def test_movement_command_covers_every_transition() -> None:
    command = store_runner._movement_command
    assert command("STOP", "E") == "DOWN"
    assert command("E", "E") == "KEEP"
    assert command("E", "NE") == "MOVE"
    assert command("E", "STOP") == "UP"
    assert command("STOP", "STOP") == "NOOP"


def test_router_masks_keep_requested_and_applied_separate() -> None:
    route = store_runner._route
    assert route("east", known=True, death=False, outside_region=False) == (
        "east",
        "geometry_rule",
        "geometry_rule",
    )
    assert route("east", known=True, death=True, outside_region=False)[0] == "wait"
    assert route("east", known=True, death=False, outside_region=True)[0] == "wait"
    assert route("east", known=False, death=False, outside_region=False) == (
        "wait",
        "deterministic_router",
        "unknown_position",
    )
    assert store_runner.JOYSTICK_TO_STORE_DIRECTION["wait"] == "STOP"


def test_direction_vocabulary_round_trips() -> None:
    mapping = store_runner.JOYSTICK_TO_STORE_DIRECTION
    assert mapping["wait"] == "STOP"
    assert mapping["north_west"] == "NW"
    assert set(mapping.values()) == {"STOP", "N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    assert store_runner.STORE_TO_JOYSTICK_DIRECTION["SE"] == "south_east"


def test_transition_satisfies_the_frozen_causal_contract(tmp_path: Path) -> None:
    packets = _chain(tmp_path, 1)
    row = _row(packets[0], packets[1], step_id=0, done=False, reason="NOT_DONE", end_kind="NOT_DONE")
    validation = validate_transition(cast("object", row))  # type: ignore[arg-type]
    assert validation.errors == ()
    assert validation.causal_order_valid is True
    assert row["training_eligible"] is False
    assert row["events"] == []
    assert row["reward"]["components"]["terminal"] == 0.0
    assert row["observation"]["capture_source_class"] == "self_built_mobile_test_app"


def test_terminal_transition_carries_the_arrival_reward(tmp_path: Path) -> None:
    packets = _chain(tmp_path, 1)
    row = _row(
        packets[0],
        packets[1],
        step_id=0,
        done=True,
        reason="NAVIGATION_GOAL_REACHED",
        end_kind="TERMINATED",
    )
    validation = validate_transition(cast("object", row))  # type: ignore[arg-type]
    assert validation.errors == ()
    assert row["reward"]["components"]["terminal"] == 1.0
    assert row["reward"]["total"] == 1.0
    assert row["episode_end_kind"] == "TERMINATED"
    assert row["replay"]["source"] == "controller"


def test_transition_rejects_a_broken_causal_chain(tmp_path: Path) -> None:
    packets = _chain(tmp_path, 1)
    row = _row(packets[0], packets[1], step_id=0, done=False, reason="NOT_DONE", end_kind="NOT_DONE")
    cast(dict, row["executed_action"])["dispatch_start_ns"] = 1
    validation = validate_transition(cast("object", row))  # type: ignore[arg-type]
    assert validation.valid is False
    assert validation.causal_order_valid is False


def test_write_bundle_is_atomic_and_rejects_different_content(tmp_path: Path) -> None:
    views, arrays = _views(0)
    store_runner._write_bundle(tmp_path, "bundle.npz", arrays)
    assert (tmp_path / "bundle.npz").exists()
    store_runner._write_bundle(tmp_path, "bundle.npz", arrays)
    changed = dict(arrays)
    changed["main"] = np.full((4, 4, 3), 9, dtype=np.uint8)
    with pytest.raises(MobileTestbedError):
        store_runner._write_bundle(tmp_path, "bundle.npz", changed)
    with pytest.raises(MobileTestbedError):
        store_runner._write_bundle(tmp_path, "bundle.npz", {"main": arrays["main"]})


def test_verify_recovers_a_committed_episode_and_reports_damage(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    database = tmp_path / "transitions.sqlite3"
    episode_id = "ep-test"
    packets = _chain(frames, 2, episode_id)
    rows = [
        _row(packets[0], packets[1], step_id=0, done=False, reason="NOT_DONE", end_kind="NOT_DONE"),
        _row(
            packets[1],
            packets[2],
            step_id=1,
            done=True,
            reason="NAVIGATION_GOAL_REACHED",
            end_kind="TERMINATED",
        ),
    ]
    with UnifiedTransitionStore(database) as store:
        for row in rows:
            stored = store.append(cast("object", row))  # type: ignore[arg-type]
            assert stored.validation.valid, stored.validation.errors
    report = store_runner.verify_mobile_navigation_episode(
        store_path=database, episode_id=episode_id, frame_root=frames
    )
    assert report["recoverable"] is True
    assert report["transitions"] == 2
    assert report["terminal_reason"] == "NAVIGATION_GOAL_REACHED"

    (frames / f"{episode_id}-0001.npz").unlink()
    damaged = store_runner.verify_mobile_navigation_episode(
        store_path=database, episode_id=episode_id, frame_root=frames
    )
    assert damaged["recoverable"] is False
    assert any("frame_bundle_missing" in item for item in cast(list[str], damaged["findings"]))

    empty = tmp_path / "other.sqlite3"
    with UnifiedTransitionStore(empty):
        pass
    missing = store_runner.verify_mobile_navigation_episode(
        store_path=empty, episode_id=episode_id, frame_root=frames
    )
    assert missing["findings"] == ["episode_missing"]


def test_verify_requires_a_terminal_transition(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    database = tmp_path / "transitions.sqlite3"
    packets = _chain(frames, 1)
    with UnifiedTransitionStore(database) as store:
        store.append(  # type: ignore[arg-type]
            cast(
                "object",
                _row(
                    packets[0],
                    packets[1],
                    step_id=0,
                    done=False,
                    reason="NOT_DONE",
                    end_kind="NOT_DONE",
                ),
            )
        )
    report = store_runner.verify_mobile_navigation_episode(
        store_path=database, episode_id="ep-test", frame_root=frames
    )
    assert report["recoverable"] is False
    assert "terminal_transition_missing" in cast(list[str], report["findings"])


def test_store_lists_episodes_and_reports_integrity(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    database = tmp_path / "transitions.sqlite3"
    with UnifiedTransitionStore(database) as store:
        assert store.episode_ids() == ()
        assert store.integrity() == "ok"
        packets = _chain(frames, 1, "ep-a")
        store.append(  # type: ignore[arg-type]
            cast(
                "object",
                _row(
                    packets[0],
                    packets[1],
                    step_id=0,
                    done=True,
                    reason="NAVIGATION_GOAL_REACHED",
                    end_kind="TERMINATED",
                ),
            )
        )
        assert store.episode_ids() == ("ep-a",)


def test_verify_store_covers_every_episode_and_integrity(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    database = tmp_path / "transitions.sqlite3"
    with UnifiedTransitionStore(database) as store:
        for episode_id in ("ep-a", "ep-b"):
            packets = _chain(frames, 1, episode_id)
            stored = store.append(  # type: ignore[arg-type]
                cast(
                    "object",
                    _row(
                        packets[0],
                        packets[1],
                        step_id=0,
                        done=True,
                        reason="NAVIGATION_GOAL_REACHED",
                        end_kind="TERMINATED",
                    ),
                )
            )
            assert stored.validation.valid, stored.validation.errors
    report = store_runner.verify_mobile_navigation_store(
        store_path=database, frame_root=frames
    )
    assert report["recoverable"] is True
    assert report["episodes"] == 2
    assert report["episode_ids"] == ["ep-a", "ep-b"]
    assert report["transitions"] == 2
    assert report["store_integrity"] == "ok"
    assert report["findings"] == []

    empty = tmp_path / "empty.sqlite3"
    with UnifiedTransitionStore(empty):
        pass
    blank = store_runner.verify_mobile_navigation_store(
        store_path=empty, frame_root=frames
    )
    assert blank["recoverable"] is False
    assert blank["episodes"] == 0


def test_batch_runner_rejects_a_bad_episode_count(tmp_path: Path) -> None:
    with pytest.raises(MobileTestbedError):
        store_runner.run_mobile_navigation_episodes(
            serial="unused",
            contract_path=STORE_CONTRACT,
            visual_layout_path=tmp_path,
            execution_layout_path=tmp_path,
            observation_rois_path=tmp_path,
            output_dir=tmp_path / "out",
            episodes=0,
        )


def test_batch_episode_ids_are_distinct_and_ordered() -> None:
    contract, sha = _goal_navigation_contract(STORE_CONTRACT)
    prefix = cast(str, cast(dict, contract["store"])["episode_prefix"])
    base = store_runner._episode_id(prefix, sha)
    ids = [f"{base}-{ordinal:02d}" for ordinal in range(1, 4)]
    assert len(set(ids)) == 3
    assert ids[0] < ids[1] < ids[2]
    assert all(item.startswith(base) for item in ids)


def test_router_output_is_always_a_joystick_direction() -> None:
    """Regression: the router must stay in the joystick vocabulary for every mask."""
    for death in (False, True):
        for outside_region in (False, True):
            for known in (False, True):
                applied, owner, reason = store_runner._route(
                    "east", known=known, death=death, outside_region=outside_region
                )
                assert applied in store_runner.JOYSTICK_TO_STORE_DIRECTION, (applied, reason)
                assert store_runner.STORE_TO_JOYSTICK_DIRECTION[
                    store_runner.JOYSTICK_TO_STORE_DIRECTION[applied]
                ] == applied
                assert owner in {"geometry_rule", "deterministic_router"}


def test_episode_outcome_ends_a_sustained_localisation_gap() -> None:
    outcome = store_runner._episode_outcome
    assert outcome(
        arrived=False,
        death=False,
        outside_region=False,
        missing_streak=3,
        maximum_gap=10,
        budget_exhausted=False,
    ) == (False, "NOT_DONE", "NOT_DONE", None)
    assert outcome(
        arrived=False,
        death=False,
        outside_region=False,
        missing_streak=11,
        maximum_gap=10,
        budget_exhausted=False,
    ) == (True, "CAPTURE_FAILURE", "ERROR", "localization_gap")
    assert outcome(
        arrived=False,
        death=False,
        outside_region=False,
        missing_streak=99,
        maximum_gap=0,
        budget_exhausted=False,
    ) == (False, "NOT_DONE", "NOT_DONE", None)


def test_episode_outcome_priority_keeps_arrival_and_safety_first() -> None:
    outcome = store_runner._episode_outcome
    assert outcome(
        arrived=True,
        death=True,
        outside_region=True,
        missing_streak=99,
        maximum_gap=10,
        budget_exhausted=True,
    )[1] == "NAVIGATION_GOAL_REACHED"
    assert outcome(
        arrived=False,
        death=True,
        outside_region=False,
        missing_streak=0,
        maximum_gap=10,
        budget_exhausted=False,
    ) == (True, "SAFETY_STOP", "ERROR", "death_or_ended_screen")
    assert outcome(
        arrived=False,
        death=False,
        outside_region=True,
        missing_streak=0,
        maximum_gap=10,
        budget_exhausted=False,
    ) == (True, "SAFETY_STOP", "ERROR", "outside_free_movement_region")


def test_episode_outcome_truncates_an_exhausted_budget() -> None:
    """A step or duration cap must still produce a terminal transition, never a dangling NOT_DONE."""
    outcome = store_runner._episode_outcome
    done, reason, end_kind, abort = outcome(
        arrived=False,
        death=False,
        outside_region=False,
        missing_streak=0,
        maximum_gap=10,
        budget_exhausted=True,
    )
    assert (done, reason, end_kind) == (True, "TIMEOUT", "TRUNCATED")
    assert abort == "step_or_duration_budget_exhausted"


def test_store_v2_contract_carries_the_localisation_gap_guard() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract

    contract, sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_store_v2.json"
    )
    assert len(sha) == 64
    block = cast(dict, contract["store"])
    assert block["maximum_localization_gap_frames"] == 10
    v1, _ = _goal_navigation_contract(ROOT / "configs/movement_goal_navigation_store_v1.json")
    assert "maximum_localization_gap_frames" not in cast(dict, v1["store"])


def test_recovery_direction_sweeps_the_declared_order() -> None:
    order = ["north", "east", "south", "west"]
    assert policy_direction(0, order) == "north"
    assert policy_direction(3, order) == "west"
    assert policy_direction(4, order) == "north"
    assert policy_direction(11, order) == "west"
    with pytest.raises(store_runner.MobileTestbedError):
        store_runner._recovery_direction(0, [])
    with pytest.raises(store_runner.MobileTestbedError):
        store_runner._recovery_direction(-1, order)


def policy_direction(attempt: int, order: list[str]) -> str:
    return store_runner._recovery_direction(attempt, order)


def test_recovery_suspends_the_gap_failure_while_it_runs() -> None:
    """A bounded sweep must be allowed to run past the plain gap limit, then fail if it cannot recover."""
    outcome = store_runner._episode_outcome
    assert outcome(
        arrived=False,
        death=False,
        outside_region=False,
        missing_streak=40,
        maximum_gap=10,
        budget_exhausted=False,
        recovery_active=True,
    ) == (False, "NOT_DONE", "NOT_DONE", None)
    assert outcome(
        arrived=False,
        death=False,
        outside_region=False,
        missing_streak=40,
        maximum_gap=10,
        budget_exhausted=False,
        recovery_active=False,
    ) == (True, "CAPTURE_FAILURE", "ERROR", "localization_gap")


def test_store_v3_contract_declares_the_bounded_recovery() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract

    contract, sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_store_v3.json"
    )
    assert len(sha) == 64
    recovery = cast(dict, contract["recovery"])
    assert recovery["mode"] == "bounded_direction_sweep"
    assert len(recovery["direction_order"]) == 8
    assert recovery["maximum_attempts_per_event"] > recovery["trigger_missing_frames"]


def test_store_v3_contract_rejects_a_broken_recovery_block(tmp_path: Path) -> None:

    value = json.loads(
        (ROOT / "configs/movement_goal_navigation_store_v3.json").read_text(encoding="utf-8")
    )
    tampered = tmp_path / "store-v3.json"
    for mutate in (
        lambda item: item["recovery"].__setitem__("mode", "ad_hoc"),
        lambda item: item["recovery"].__setitem__("direction_order", ["north", "up"]),
        lambda item: item["recovery"].__setitem__("direction_order", ["north"]),
        lambda item: item["recovery"].__setitem__("maximum_attempts_per_event", 0),
    ):
        value = json.loads(
            (ROOT / "configs/movement_goal_navigation_store_v3.json").read_text(encoding="utf-8")
        )
        mutate(value)
        tampered.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(MobileTestbedError):
            store_runner._store_contract(value, "0" * 64)
    v1 = store_runner._store_contract(
        json.loads(
            (ROOT / "configs/movement_goal_navigation_store_v1.json").read_text(encoding="utf-8")
        ),
        "0" * 64,
    )
    assert v1.get("recovery") is None
