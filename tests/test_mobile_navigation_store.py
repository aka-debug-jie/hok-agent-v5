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
from hok_agent.traversability import discover_route_b_runs

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
        no_advance=False,
        maximum_gap=10,
        budget_exhausted=False,
    ) == (False, "NOT_DONE", "NOT_DONE", None)
    assert outcome(
        arrived=False,
        death=False,
        outside_region=False,
        missing_streak=11,
        no_advance=False,
        maximum_gap=10,
        budget_exhausted=False,
    ) == (True, "CAPTURE_FAILURE", "ERROR", "localization_gap")
    assert outcome(
        arrived=False,
        death=False,
        outside_region=False,
        missing_streak=99,
        no_advance=False,
        maximum_gap=0,
        budget_exhausted=False,
    ) == (False, "NOT_DONE", "NOT_DONE", None)


def test_localisation_cutoff_is_the_stop_rule_not_a_death_classifier() -> None:
    """Pin the boundary: the 11th missing step is the stop rule, not a life-state observation.

    The route B diagnosis rests on this. A sustained localisation gap ends the episode one step
    past ``maximum_gap`` and is reported as a capture failure; it carries no death information,
    so a run that ends on its 11th blind step has not been classified as a dead hero.
    """
    outcome = store_runner._episode_outcome
    common = {
        "arrived": False,
        "death": False,
        "outside_region": False,
        "no_advance": False,
        "maximum_gap": 10,
        "budget_exhausted": False,
    }
    assert outcome(missing_streak=10, **common) == (False, "NOT_DONE", "NOT_DONE", None)
    assert outcome(missing_streak=11, **common) == (
        True,
        "CAPTURE_FAILURE",
        "ERROR",
        "localization_gap",
    )
    # Only a positive life-state observation may produce the death reason; a gap never does.
    assert outcome(missing_streak=99, **common)[3] != "death_or_ended_screen"
    death = dict(common, death=True)
    assert outcome(missing_streak=99, **death) == (
        True,
        "SAFETY_STOP",
        "ERROR",
        "death_or_ended_screen",
    )


def test_episode_outcome_priority_keeps_arrival_and_safety_first() -> None:
    outcome = store_runner._episode_outcome
    assert outcome(
        arrived=True,
        death=True,
        outside_region=True,
        missing_streak=99,
        no_advance=False,
        maximum_gap=10,
        budget_exhausted=True,
    )[1] == "NAVIGATION_GOAL_REACHED"
    assert outcome(
        arrived=False,
        death=True,
        outside_region=False,
        missing_streak=0,
        no_advance=False,
        maximum_gap=10,
        budget_exhausted=False,
    ) == (True, "SAFETY_STOP", "ERROR", "death_or_ended_screen")
    assert outcome(
        arrived=False,
        death=False,
        outside_region=True,
        missing_streak=0,
        no_advance=False,
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
        no_advance=False,
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


def test_rotate_direction_wraps_the_eight_way_order() -> None:
    assert store_runner._rotate_direction("north", 1) == "north_east"
    assert store_runner._rotate_direction("north", -1) == "north_west"
    assert store_runner._rotate_direction("north", 8) == "north"
    assert store_runner._rotate_direction("west", 3) == "north_east"
    with pytest.raises(MobileTestbedError):
        store_runner._rotate_direction("wait", 1)


def test_progress_guard_offset_holds_each_bearing_for_the_hold_window() -> None:
    offsets = [1, -1, 2, -2]
    assert store_runner._progress_guard_offset(0, 3, offsets) == 1
    assert store_runner._progress_guard_offset(2, 3, offsets) == 1
    assert store_runner._progress_guard_offset(3, 3, offsets) == -1
    assert store_runner._progress_guard_offset(11, 3, offsets) == -2
    with pytest.raises(MobileTestbedError):
        store_runner._progress_guard_offset(12, 3, offsets)
    with pytest.raises(MobileTestbedError):
        store_runner._progress_guard_offset(0, 0, offsets)


def test_route_b_v2_contract_declares_the_progress_guard() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract

    contract, sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v2.json"
    )
    assert len(sha) == 64
    guard = cast(dict, contract["progress_guard"])
    assert guard["mode"] == "bounded_bearing_escape"
    assert guard["confirmation_steps"] >= 2
    assert set(guard["escape_offsets_sectors"]) == {1, -1, 2, -2}
    assert guard["escape_hold_steps"] >= 2
    resolved = store_runner._store_contract(contract, sha)
    assert resolved["progress_guard"] == guard
    v1, v1_sha = _goal_navigation_contract(ROOT / "configs/movement_goal_navigation_route_b_v1.json")
    assert "progress_guard" not in v1
    assert store_runner._store_contract(v1, v1_sha)["progress_guard"] is None


def test_route_b_v2_contract_rejects_a_broken_progress_guard() -> None:
    value = json.loads(
        (ROOT / "configs/movement_goal_navigation_route_b_v2.json").read_text(encoding="utf-8")
    )
    for mutate in (
        lambda item: item["progress_guard"].__setitem__("mode", "ad_hoc"),
        lambda item: item["progress_guard"].__setitem__("escape_offsets_sectors", []),
        lambda item: item["progress_guard"].__setitem__("escape_offsets_sectors", [9]),
        lambda item: item["progress_guard"].__setitem__("confirmation_steps", 0),
        lambda item: item["progress_guard"].__setitem__("escape_hold_steps", 0),
    ):
        value = json.loads(
            (ROOT / "configs/movement_goal_navigation_route_b_v2.json").read_text(encoding="utf-8")
        )
        mutate(value)
        with pytest.raises(MobileTestbedError):
            store_runner._store_contract(value, "0" * 64)


REGION = {"minimum_y": 35.0, "maximum_y": 95.0, "minimum_x": 35.0, "maximum_x": 95.0}


def test_region_filter_keeps_the_desired_direction_when_it_is_safe() -> None:
    safe = store_runner._region_safe_direction((60.0, 60.0), "east", REGION, 6.0, 4.0)
    assert safe == "east"


def test_region_filter_masks_a_step_that_would_leave_the_region() -> None:
    """From x=93 an east step exits the region, so the filter must pick another bearing."""
    masked = store_runner._region_safe_direction((60.0, 93.0), "east", REGION, 6.0, 4.0)
    assert masked != "east"
    step_y, step_x = store_runner._DIRECTION_STEPS[masked]
    assert 35.0 + 4.0 <= 60.0 + step_y * 6.0 <= 95.0 - 4.0
    assert 35.0 + 4.0 <= 93.0 + step_x * 6.0 <= 95.0 - 4.0


def test_region_filter_turns_a_corner_into_a_follow_along_bearing() -> None:
    """At the bottom-right corner both east and south are unsafe, so a lateral bearing is chosen."""
    masked = store_runner._region_safe_direction((93.0, 93.0), "south_east", REGION, 6.0, 4.0)
    assert masked in {"west", "north", "north_west", "south_west", "north_east"}


def test_region_filter_heads_inward_from_outside_the_region() -> None:
    inward = store_runner._region_safe_direction((60.0, 20.0), "west", REGION, 6.0, 4.0)
    step_y, step_x = store_runner._DIRECTION_STEPS[inward]
    assert 20.0 + step_x * 6.0 > 20.0
    assert inward in {"east", "north_east", "south_east"}


def test_region_filter_passes_wait_through_unchanged() -> None:
    assert store_runner._region_safe_direction((60.0, 60.0), "wait", REGION, 6.0, 4.0) == "wait"


def test_route_b_v3_contract_declares_the_region_filter() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract

    contract, sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v3.json"
    )
    assert len(sha) == 64
    region_filter = cast(dict, contract["region_filter"])
    assert region_filter["mode"] == "feasible_direction_within_region"
    assert region_filter["nominal_step_pixels"] > 0 and region_filter["margin_pixels"] > 0
    resolved = store_runner._store_contract(contract, sha)
    assert resolved["region_filter"] == region_filter
    assert cast(dict, contract["progress_guard"])["mode"] == "bounded_bearing_escape"


def test_route_b_v3_contract_rejects_a_broken_region_filter() -> None:
    value = json.loads(
        (ROOT / "configs/movement_goal_navigation_route_b_v3.json").read_text(encoding="utf-8")
    )
    for mutate in (
        lambda item: item["region_filter"].__setitem__("mode", "ad_hoc"),
        lambda item: item["region_filter"].__setitem__("nominal_step_pixels", 0),
        lambda item: item["region_filter"].__setitem__("margin_pixels", -1),
    ):
        value = json.loads(
            (ROOT / "configs/movement_goal_navigation_route_b_v3.json").read_text(encoding="utf-8")
        )
        mutate(value)
        with pytest.raises(MobileTestbedError):
            store_runner._store_contract(value, "0" * 64)


def test_approach_hold_shrinks_with_the_remaining_distance() -> None:
    tiers = [(8.0, 200), (16.0, 400), (32.0, 800)]
    pick = store_runner._approach_hold_ms
    assert pick(3.0, tiers, 1200) == 200
    assert pick(8.0, tiers, 1200) == 200
    assert pick(12.0, tiers, 1200) == 400
    assert pick(20.0, tiers, 1200) == 800
    assert pick(60.0, tiers, 1200) == 1200
    assert pick(None, tiers, 1200) == 1200
    assert pick(5.0, [], 1200) == 1200


def test_route_b_v4_contract_declares_the_deceleration_table() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract

    contract, sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v4.json"
    )
    assert len(sha) == 64
    approach = cast(dict, contract["final_approach"])
    assert approach["mode"] == "declared_deceleration"
    tiers = sorted(
        (float(item["maximum_distance_pixels"]), int(item["hold_ms"])) for item in approach["tiers"]
    )
    assert tiers[0][0] < tiers[-1][0]
    assert tiers[0][1] < tiers[-1][1]
    assert approach["default_hold_ms"] >= tiers[-1][1]
    resolved = store_runner._store_contract(contract, sha)
    assert resolved["final_approach"] == approach
    assert cast(dict, contract["region_filter"])["mode"] == "feasible_direction_within_region"


def test_route_b_v4_contract_rejects_a_broken_deceleration_table() -> None:
    value = json.loads(
        (ROOT / "configs/movement_goal_navigation_route_b_v4.json").read_text(encoding="utf-8")
    )
    for mutate in (
        lambda item: item["final_approach"].__setitem__("mode", "ad_hoc"),
        lambda item: item["final_approach"].__setitem__("tiers", []),
        lambda item: item["final_approach"].__setitem__("default_hold_ms", 0),
        lambda item: item["final_approach"].__setitem__(
            "tiers",
            [
                {"maximum_distance_pixels": 8.0, "hold_ms": 800},
                {"maximum_distance_pixels": 16.0, "hold_ms": 200},
            ],
        ),
    ):
        value = json.loads(
            (ROOT / "configs/movement_goal_navigation_route_b_v4.json").read_text(encoding="utf-8")
        )
        mutate(value)
        with pytest.raises(MobileTestbedError):
            store_runner._store_contract(value, "0" * 64)


def test_path_reference_projects_and_leads_along_the_leg() -> None:
    # route entries are (y, x); this leg runs along x from (50, 50) to (50, 80)
    route = [(50.0, 50.0), (50.0, 80.0)]
    start = (50.0, 50.0)
    reference = store_runner._path_reference((60.0, 50.0), route, 1, start, 10.0)
    assert abs(reference[0] - 50.0) < 1e-9
    assert abs(reference[1] - 60.0) < 1e-6
    at_end = store_runner._path_reference((50.0, 79.0), route, 1, start, 10.0)
    assert abs(at_end[1] - 80.0) < 1e-9
    before = store_runner._path_reference((50.0, 40.0), route, 1, start, 10.0)
    assert before[1] >= 50.0


def test_planner_maximises_progress_and_never_leaves_the_region() -> None:
    route = [(50.0, 50.0), (60.0, 60.0)]
    region = {"minimum_y": 35.0, "maximum_y": 95.0, "minimum_x": 35.0, "maximum_x": 95.0}
    position = (50.0, 50.0)
    chosen, reference = store_runner._planner_direction(
        position,
        route,
        1,
        position,
        region,
        nominal_step_pixels=6.0,
        margin_pixels=4.0,
        lookahead_pixels=10.0,
        stall_active=False,
        stall_bias=0.6,
    )
    aim_y = reference[0] - position[0]
    aim_x = reference[1] - position[1]
    norm = (aim_y * aim_y + aim_x * aim_x) ** 0.5
    aim_y /= norm
    aim_x /= norm
    best = max(
        store_runner._DIRECTION_STEPS.items(),
        key=lambda item: item[1][0] * aim_y + item[1][1] * aim_x,
    )[0]
    assert chosen == best
    corner, _ = store_runner._planner_direction(
        (93.0, 93.0),
        route,
        1,
        position,
        region,
        nominal_step_pixels=6.0,
        margin_pixels=4.0,
        lookahead_pixels=10.0,
        stall_active=False,
        stall_bias=0.6,
    )
    step_y, step_x = store_runner._DIRECTION_STEPS[corner]
    assert 39.0 <= 93.0 + step_y * 6.0 <= 91.0
    assert 39.0 <= 93.0 + step_x * 6.0 <= 91.0


def test_planner_stall_mode_rewards_a_lateral_component() -> None:
    route = [(50.0, 50.0), (50.0, 80.0)]
    region = {"minimum_y": 35.0, "maximum_y": 95.0, "minimum_x": 35.0, "maximum_x": 95.0}
    position = (50.0, 60.0)
    straight, reference = store_runner._planner_direction(
        position, route, 1, (50.0, 50.0), region,
        nominal_step_pixels=6.0, margin_pixels=4.0, lookahead_pixels=10.0,
        stall_active=False, stall_bias=0.6,
    )
    stalled, _ = store_runner._planner_direction(
        position, route, 1, (50.0, 50.0), region,
        nominal_step_pixels=6.0, margin_pixels=4.0, lookahead_pixels=10.0,
        stall_active=True, stall_bias=0.6,
    )
    aim_y = reference[0] - position[0]
    aim_x = reference[1] - position[1]
    norm = (aim_y * aim_y + aim_x * aim_x) ** 0.5
    aim_y /= norm
    aim_x /= norm
    assert set(store_runner._DIRECTION_STEPS) == set(store_runner._MOVEMENT_ORDER)
    for step_y, step_x in store_runner._DIRECTION_STEPS.values():
        assert (step_y * step_y + step_x * step_x) > 0.0
    straight_lateral = abs(
        store_runner._DIRECTION_STEPS[straight][0] * aim_x
        - store_runner._DIRECTION_STEPS[straight][1] * aim_y
    )
    stalled_lateral = abs(
        store_runner._DIRECTION_STEPS[stalled][0] * aim_x
        - store_runner._DIRECTION_STEPS[stalled][1] * aim_y
    )
    assert stalled_lateral > straight_lateral


def test_route_b_v6_contract_declares_the_single_arbiter_planner() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract

    contract, sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v6.json"
    )
    assert len(sha) == 64
    planner = cast(dict, contract["planner"])
    assert planner["mode"] == "path_progress_with_feasibility"
    assert planner["lookahead_pixels"] > 0 and 0.0 < planner["stall_bias"] < 1.0
    guard = cast(dict, contract["progress_guard"])
    assert guard["mode"] == "stall_trigger_only"
    assert "escape_offsets_sectors" not in guard
    assert "region_filter" not in contract
    resolved = store_runner._store_contract(contract, sha)
    assert resolved["planner"] == planner


def test_route_b_v6_rejects_two_arbiters() -> None:
    value = json.loads(
        (ROOT / "configs/movement_goal_navigation_route_b_v6.json").read_text(encoding="utf-8")
    )
    value["region_filter"] = {
        "mode": "feasible_direction_within_region",
        "nominal_step_pixels": 6.0,
        "margin_pixels": 4.0,
    }
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(value, "0" * 64)
    value = json.loads(
        (ROOT / "configs/movement_goal_navigation_route_b_v6.json").read_text(encoding="utf-8")
    )
    value["progress_guard"]["escape_offsets_sectors"] = [1, -1]
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(value, "0" * 64)


def test_stall_trigger_guard_has_no_escape_schedule() -> None:
    """A stall-trigger-only guard must not require the escape schedule the planner replaces."""
    from hok_agent.mobile_testbed import _goal_navigation_contract

    contract, sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v6.json"
    )
    resolved = store_runner._store_contract(contract, sha)
    guard = cast(dict, resolved["progress_guard"])
    assert guard["mode"] == "stall_trigger_only"
    assert "escape_offsets_sectors" not in guard
    assert resolved["planner"] is not None


def test_retrace_direction_reverses_the_last_known_displacement() -> None:
    """The declared recovery walks back over ground the hero has just covered."""
    retrace = store_runner._retrace_direction
    region = {"minimum_y": 35.0, "maximum_y": 95.0, "minimum_x": 35.0, "maximum_x": 95.0}
    # the hero walked west, so the retreat bearing is east
    assert retrace((50.0, 80.0), (50.0, 86.0), region, 6.0) == "east"
    # the hero walked south, so the retreat bearing is north
    assert retrace((50.0, 80.0), (44.0, 80.0), region, 6.0) == "north"
    # a south-west approach retraces north-east
    assert retrace((50.0, 80.0), (46.0, 84.0), region, 6.0) == "north_east"


def test_retrace_direction_declines_without_a_usable_displacement() -> None:
    """A standing hero, or one below the displacement floor, has no bearing to reverse."""
    retrace = store_runner._retrace_direction
    region = {"minimum_y": 35.0, "maximum_y": 95.0, "minimum_x": 35.0, "maximum_x": 95.0}
    assert retrace((50.0, 80.0), (50.0, 80.0), region, 6.0) is None
    assert retrace((50.0, 80.0), (50.2, 80.1), region, 6.0) is None
    # the measured batch-10 episode-01 terminal onset: the hero held its final approach, so the
    # last two known positions differ by 0.02 px and there is nothing to retrace
    assert retrace((49.21, 87.52), (49.19, 87.53), region, 6.0) is None


def test_retrace_direction_declines_a_step_that_would_leave_the_declared_region() -> None:
    """The recovery declines rather than guessing when the retreat would leave the region.

    The check is against the declared region itself, which is the invariant ``outside_region``
    already enforces, so a retreat can never trip that guard.
    """
    retrace = store_runner._retrace_direction
    region = {"minimum_y": 35.0, "maximum_y": 95.0, "minimum_x": 35.0, "maximum_x": 95.0}
    # reversing east from x=80 lands at 86, inside the declared region
    assert retrace((50.0, 80.0), (50.0, 84.0), region, 6.0) == "east"
    # the same bearing becomes unsafe as the retreat deepens: 80 + 3*6 = 98 leaves the region
    assert retrace((50.0, 80.0), (50.0, 84.0), region, 6.0, 2) is None
    # and a reversal that would overshoot the eastern edge from x=91 is declined outright
    assert retrace((50.0, 91.0), (50.0, 95.0), region, 6.0) is None


def test_router_applies_the_declared_recovery_only_while_unknown() -> None:
    """The recovery is a Router masking rule, not a new policy layer."""
    route = store_runner._route
    assert route(
        "east", known=False, death=False, outside_region=False,
        recovery=("west", "unknown_recovery_retrace"),
    ) == ("west", "deterministic_router", "unknown_recovery_retrace")
    assert route(
        "east", known=False, death=False, outside_region=False,
        recovery=("west", "unknown_recovery_waypoint"),
    ) == ("west", "deterministic_router", "unknown_recovery_waypoint")
    # a located hero never takes a recovery bearing
    assert route(
        "east", known=True, death=False, outside_region=False,
        recovery=("west", "unknown_recovery_retrace"),
    ) == ("east", "geometry_rule", "geometry_rule")
    # without a declared recovery the wait-on-unknown behaviour is unchanged
    assert route("east", known=False, death=False, outside_region=False) == (
        "wait",
        "deterministic_router",
        "unknown_position",
    )
    # death and region safety still outrank the recovery
    assert (
        route(
            "east", known=False, death=True, outside_region=False,
            recovery=("west", "unknown_recovery_waypoint"),
        )[2]
        == "death_or_ended_screen"
    )
    assert (
        route(
            "east", known=False, death=False, outside_region=True,
            recovery=("west", "unknown_recovery_waypoint"),
        )[2]
        == "outside_free_movement_region"
    )


def test_unknown_recovery_step_is_bounded_and_resets_after_a_fix() -> None:
    """Pin the sequencing: trigger, budget, decline and the reset a returned fix must cause."""
    step = store_runner._unknown_recovery_step
    region = {"minimum_y": 35.0, "maximum_y": 95.0, "minimum_x": 35.0, "maximum_x": 95.0}
    config = {
        "mode": "bounded_retrace_or_waypoint",
        "trigger_after_missing_frames": 3,
        "maximum_recovery_steps": 6,
        "hold_ms": 800,
        "nominal_step_pixels": 6.0,
    }
    target = (50.0, 80.0)
    # a roomy anchor: the hero walked west into x=50, so the retreat bearing is east with room
    previous_known, last_known = (50.0, 56.0), (50.0, 50.0)
    # no anchor at all: the recovery cannot act, which is the whole all-blind-from-start class
    assert step(
        last_known=None,
        previous_known=None,
        target=target,
        missing_streak=5,
        recovery_steps=0,
        config=config,
        region=region,
    ) == (None, False)
    # before the trigger, the Router keeps waiting: not eligible at all
    assert step(
        last_known=last_known,
        previous_known=previous_known,
        target=target,
        missing_streak=2,
        recovery_steps=0,
        config=config,
        region=region,
    ) == (None, False)
    # at the trigger the retreat starts, and stays eligible while the declared budget holds
    assert step(
        last_known=last_known,
        previous_known=previous_known,
        target=target,
        missing_streak=3,
        recovery_steps=0,
        config=config,
        region=region,
    ) == (("east", "unknown_recovery_retrace"), True)
    assert step(
        last_known=last_known,
        previous_known=previous_known,
        target=target,
        missing_streak=8,
        recovery_steps=5,
        config=config,
        region=region,
    ) == (("east", "unknown_recovery_retrace"), True)
    # past the declared budget the recovery stops trying, so the gap guard owns the ending
    assert step(
        last_known=last_known,
        previous_known=previous_known,
        target=target,
        missing_streak=9,
        recovery_steps=6,
        config=config,
        region=region,
    ) == (None, False)
    # an eligible step that would leave the region is reported as declined, not as waiting
    assert step(
        last_known=(50.0, 91.0),
        previous_known=(50.0, 95.0),
        target=(50.0, 150.0),
        missing_streak=4,
        recovery_steps=0,
        config=config,
        region=region,
    ) == (None, True)
    # without a declared block nothing changes for the existing contracts
    assert step(
        last_known=last_known,
        previous_known=previous_known,
        target=target,
        missing_streak=9,
        recovery_steps=0,
        config=None,
        region=region,
    ) == (None, False)
    # a returned fix zeroes recovery_steps in the caller, so the next loss retraces from depth 0
    assert step(
        last_known=last_known,
        previous_known=previous_known,
        target=target,
        missing_streak=3,
        recovery_steps=0,
        config=config,
        region=region,
    )[0] == ("east", "unknown_recovery_retrace")


def test_unknown_recovery_re_aims_at_the_waypoint_when_there_is_nothing_to_retrace() -> None:
    """The measured replacement: a hero that held its final approach still gets a bounded action.

    These are the recorded batch-10 terminal-loss coordinates. The hero was stationary at the last
    known position, so the retrace has no displacement to reverse and v7 declined every eligible
    step; the waypoint fallback turns the same steps into a bounded, region-checked re-aim.
    """
    step = store_runner._unknown_recovery_step
    region = {"minimum_y": 35.0, "maximum_y": 95.0, "minimum_x": 35.0, "maximum_x": 95.0}
    base = {
        "trigger_after_missing_frames": 3,
        "maximum_recovery_steps": 6,
        "hold_ms": 800,
        "nominal_step_pixels": 6.0,
    }
    # the recorded episode-01 onset: last two known positions differ by 0.02 px
    stationary_last, stationary_previous = (49.21, 87.52), (49.19, 87.53)
    target = (50.0, 80.0)
    # v7 declined here; v8 re-aims at the waypoint, which lies west with ample region room
    assert step(
        last_known=stationary_last,
        previous_known=stationary_previous,
        target=target,
        missing_streak=3,
        recovery_steps=0,
        config={**base, "mode": "bounded_retrace_or_waypoint"},
        region=region,
    ) == (("west", "unknown_recovery_waypoint"), True)
    # the pure-retrace mode still declines that same step, so the version difference is the fix
    assert step(
        last_known=stationary_last,
        previous_known=stationary_previous,
        target=target,
        missing_streak=3,
        recovery_steps=0,
        config={**base, "mode": "bounded_retrace"},
        region=region,
    ) == (None, True)
    # the fallback is region-checked at depth too: 87.52 - 8*6 = 39.52 is still inside, and
    # one step deeper the predicted x leaves the declared region, so it stops rather than guessing
    deep = {**base, "mode": "bounded_retrace_or_waypoint", "maximum_recovery_steps": 9}
    assert step(
        last_known=stationary_last,
        previous_known=stationary_previous,
        target=target,
        missing_streak=9,
        recovery_steps=7,
        config=deep,
        region=region,
    ) == (("west", "unknown_recovery_waypoint"), True)
    assert step(
        last_known=stationary_last,
        previous_known=stationary_previous,
        target=target,
        missing_streak=10,
        recovery_steps=8,
        config=deep,
        region=region,
    ) == (None, True)
    assert step(
        last_known=(50.0, 40.0),
        previous_known=(50.0, 40.0),
        target=(50.0, 20.0),
        missing_streak=4,
        recovery_steps=0,
        config={**base, "mode": "bounded_retrace_or_waypoint"},
        region=region,
    ) == (None, True)
    # a target the hero is already standing on yields no bearing, so the step declines
    assert step(
        last_known=(50.0, 80.0),
        previous_known=(50.0, 80.0),
        target=(50.0, 80.0),
        missing_streak=4,
        recovery_steps=0,
        config={**base, "mode": "bounded_retrace_or_waypoint"},
        region=region,
    ) == (None, True)


def test_route_b_v9_declares_the_press_release_band() -> None:
    """v9 makes the declared approach hold bound the press instead of only the sampling period."""
    from hok_agent.mobile_testbed import _goal_navigation_contract

    v8, v8_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v8.json"
    )
    v9, v9_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v9.json"
    )
    assert v8_sha == "3c41d5bbbfb1ccd18115953c8d3a5427db9b31edfabbe2ee4c7f23aa914fd0f4"
    assert "press_release_maximum_distance_pixels" not in cast(dict, v8["final_approach"])
    assert "backlog_free_maximum_pointer_messages" not in cast(dict, v8["store"])
    approach = cast(dict, v9["final_approach"])
    # the press band must sit inside the declared deceleration tiers
    tiers = [cast(dict, item)["maximum_distance_pixels"] for item in cast(list, approach["tiers"])]
    assert approach["press_release_maximum_distance_pixels"] == 16.0
    assert approach["press_release_maximum_distance_pixels"] <= max(tiers)
    assert approach["mode"] == "declared_deceleration"
    # a pulse step issues one press and one release
    assert cast(dict, v9["store"])["backlog_free_maximum_pointer_messages"] == 3
    # nothing else moved: the planner, recovery, targets, gates and gap guard are carried over
    ignored = {"final_approach", "store", "purpose", "route_id", "contract_sha256"}
    assert {k: v for k, v in v9.items() if k not in ignored} == {
        k: v for k, v in v8.items() if k not in ignored
    }
    resolved = store_runner._store_contract(v9, v9_sha)
    assert (
        cast(dict, resolved["final_approach"])["press_release_maximum_distance_pixels"] == 16.0
    )
    assert resolved["backlog_free_maximum_pointer_messages"] == 3


def test_route_b_v9_rejects_a_press_band_outside_the_tiers_or_a_zero_ceiling() -> None:
    value = cast(
        dict,
        json.loads(
            (ROOT / "configs/movement_goal_navigation_route_b_v9.json").read_text(encoding="utf-8")
        ),
    )
    cast(dict, value["final_approach"])["press_release_maximum_distance_pixels"] = 64.0
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(value, "0" * 64)
    value = cast(
        dict,
        json.loads(
            (ROOT / "configs/movement_goal_navigation_route_b_v9.json").read_text(encoding="utf-8")
        ),
    )
    cast(dict, value["final_approach"])["press_release_maximum_distance_pixels"] = 0.0
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(value, "0" * 64)
    value = cast(
        dict,
        json.loads(
            (ROOT / "configs/movement_goal_navigation_route_b_v9.json").read_text(encoding="utf-8")
        ),
    )
    value["store"]["backlog_free_maximum_pointer_messages"] = 0
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(value, "0" * 64)


def test_route_b_v10_prefers_the_waypoint_bearing_inside_the_declared_band() -> None:
    """Measured on the v9 run: near the target the retrace walks the hero away, so the band wins."""
    from hok_agent.mobile_testbed import _goal_navigation_contract

    v9, v9_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v9.json"
    )
    v10, v10_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v10.json"
    )
    assert v9_sha == "03339a7b57e52cfd5724bb354b07d036caf13bb6b63d05cb0f3887ddcdfa802a"
    assert "waypoint_bearing_within_distance_pixels" not in cast(dict, v9["unknown_recovery"])
    recovery_v9 = cast(dict, v9["unknown_recovery"])
    recovery_v10 = cast(dict, v10["unknown_recovery"])
    assert recovery_v10["waypoint_bearing_within_distance_pixels"] == 16.0
    # only the band is added: the recovery, the press band, the planner and the gates are carried over
    assert {k: v for k, v in recovery_v10.items() if k != "waypoint_bearing_within_distance_pixels"} == {
        k: v for k, v in recovery_v9.items() if k != "waypoint_bearing_within_distance_pixels"
    }
    ignored = {"unknown_recovery", "purpose", "route_id", "contract_sha256"}
    assert {k: v for k, v in v10.items() if k not in ignored} == {
        k: v for k, v in v9.items() if k not in ignored
    }
    assert store_runner._store_contract(v10, v10_sha)["unknown_recovery"] == recovery_v10
    # a non-positive band is rejected
    value = cast(
        dict,
        json.loads(
            (ROOT / "configs/movement_goal_navigation_route_b_v10.json").read_text(encoding="utf-8")
        ),
    )
    cast(dict, value["unknown_recovery"])["waypoint_bearing_within_distance_pixels"] = 0.0
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(value, "0" * 64)


def test_unknown_recovery_prefers_the_waypoint_bearing_inside_the_band() -> None:
    step = store_runner._unknown_recovery_step
    region = {"minimum_y": 35.0, "maximum_y": 95.0, "minimum_x": 35.0, "maximum_x": 95.0}
    config = {
        "mode": "bounded_retrace_or_waypoint",
        "trigger_after_missing_frames": 3,
        "maximum_recovery_steps": 6,
        "hold_ms": 800,
        "nominal_step_pixels": 6.0,
        "waypoint_bearing_within_distance_pixels": 16.0,
    }
    # the recorded v9 final-leg onset: 6.6 px from the waypoint, so the re-aim wins over the retrace
    assert step(
        last_known=(56.6, 80.1),
        previous_known=(57.0, 79.5),
        target=(50.0, 80.0),
        missing_streak=3,
        recovery_steps=0,
        config=config,
        region=region,
    ) == (("north", "unknown_recovery_waypoint"), True)
    # farther out the retrace is still preferred, because escaping the blind area is the point
    assert step(
        last_known=(80.0, 80.0),
        previous_known=(79.0, 79.0),
        target=(50.0, 80.0),
        missing_streak=3,
        recovery_steps=0,
        config=config,
        region=region,
    )[0][1] == "unknown_recovery_retrace"
    # the band is measured from the last known position to the current waypoint: at exactly the
    # declared 16 px it still applies, and just outside it the retrace returns
    assert step(
        last_known=(66.0, 80.0),
        previous_known=(67.0, 79.0),
        target=(50.0, 80.0),
        missing_streak=3,
        recovery_steps=0,
        config=config,
        region=region,
    )[0][1] == "unknown_recovery_waypoint"
    assert step(
        last_known=(68.0, 80.0),
        previous_known=(67.0, 79.0),
        target=(50.0, 80.0),
        missing_streak=3,
        recovery_steps=0,
        config=config,
        region=region,
    )[0][1] == "unknown_recovery_retrace"


def test_route_b_v11_declares_the_direct_final_approach() -> None:
    """v11 aims at the waypoint itself inside the declared approach band instead of the path reference."""
    from hok_agent.mobile_testbed import _goal_navigation_contract

    v10, v10_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v10.json"
    )
    v11, v11_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v11.json"
    )
    assert v10_sha == "43297e0dd42a88809d6ff4dfe795d698160054e22c0f27361b76f0c327ce7e77"
    assert "direct_bearing" not in cast(dict, v10["final_approach"])
    assert cast(dict, v11["final_approach"])["direct_bearing"] is True
    # the declared approach band is the boundary the direct bearing uses
    assert cast(dict, v11)["final_approach_distance_pixels"] == 12.0
    assert store_runner._store_contract(v11, v11_sha)["final_approach"] == cast(
        dict, v11["final_approach"]
    )
    # a non-boolean value is rejected
    value = cast(
        dict,
        json.loads(
            (ROOT / "configs/movement_goal_navigation_route_b_v11.json").read_text(encoding="utf-8")
        ),
    )
    cast(dict, value["final_approach"])["direct_bearing"] = "yes"
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(value, "0" * 64)


def test_no_advance_guard_detects_a_dead_screen_and_ignores_live_steps() -> None:
    """A held bearing with a frozen localised position is a dead state, not a slow approach."""
    guard = store_runner._no_advance_detected
    # the measured dead run: 12 applied steps with the position inside a 0.2 px box
    anchor = None
    steps = 0
    detected = False
    for _ in range(12):
        anchor, steps, detected = guard(anchor, steps, (78.4, 55.5), True, 12, 1.0)
    assert detected is True and steps == 12
    # one more step without travel keeps it latched
    assert guard(anchor, steps, (78.5, 55.4), True, 12, 1.0)[2] is True
    # the measured working runs travel at least 2.76 px per 12 steps, so they must never trip
    anchor = None
    steps = 0
    detected = False
    for index in range(12):
        anchor, steps, detected = guard(anchor, steps, (78.0 - index * 0.25, 55.5), True, 12, 1.0)
    assert detected is False
    # travel beyond the declared bound resets the run rather than latching
    anchor, steps, detected = guard(None, 0, (60.0, 60.0), True, 12, 1.0)
    for _ in range(10):
        anchor, steps, detected = guard(anchor, steps, (60.0, 60.0), True, 12, 1.0)
    assert detected is False and steps == 11
    anchor, steps, detected = guard(anchor, steps, (60.0, 65.0), True, 12, 1.0)
    assert detected is False and steps == 1 and anchor == (60.0, 65.0)
    # a blind step or a released bearing resets the run, so a lost marker is never a dead screen
    anchor, steps, _ = guard(None, 0, (70.0, 70.0), True, 12, 1.0)
    anchor, steps, detected = guard(anchor, steps, None, True, 12, 1.0)
    assert (anchor, steps, detected) == (None, 0, False)
    anchor, steps, detected = guard(anchor, steps, (70.0, 70.0), False, 12, 1.0)
    assert (anchor, steps, detected) == (None, 0, False)
    # an alternating blind/live pattern can never accumulate to the window
    anchor = None
    steps = 0
    detected = False
    for index in range(40):
        anchor, steps, detected = guard(
            anchor, steps, None if index % 3 == 0 else (70.0, 70.0), True, 12, 1.0
        )
    assert detected is False


def test_episode_outcome_reports_a_no_advance_stall() -> None:
    """The stall is a distinct outcome, and arrival and safety still outrank it."""
    outcome = store_runner._episode_outcome
    common = {
        "death": False,
        "outside_region": False,
        "no_advance": False,
        "missing_streak": 0,
        "maximum_gap": 10,
        "budget_exhausted": False,
    }
    assert outcome(arrived=False, **{**common, "no_advance": True}) == (
        True,
        "ACTION_FAILURE",
        "ERROR",
        "no_advance_detected",
    )
    # a stall must never be reported as a captured-budget or a localisation-gap ending
    assert outcome(arrived=False, **{**common, "no_advance": True})[3] != "localization_gap"
    assert outcome(arrived=False, **{**common, "no_advance": True})[1] != "TIMEOUT"
    # arrival and the safety stops keep their priority
    assert outcome(arrived=True, **{**common, "no_advance": True})[1] == "NAVIGATION_GOAL_REACHED"
    assert outcome(arrived=False, **{**common, "no_advance": True, "death": True})[3] == (
        "death_or_ended_screen"
    )
    assert outcome(arrived=False, **{**common, "no_advance": True, "outside_region": True})[3] == (
        "outside_free_movement_region"
    )
    # and a normal episode is untouched by the new input
    assert outcome(arrived=False, **common) == (False, "NOT_DONE", "NOT_DONE", None)


def test_route_b_v14_declares_the_no_advance_guard() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract

    v13, v13_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v13.json"
    )
    v14, v14_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v14.json"
    )
    assert v13_sha == "393dd82ba7c62d0d6a09dcec48e8d2defd2e2e53919525cd30e93cb093e7c193"
    assert "no_advance_guard" not in v13
    guard = cast(dict, v14["no_advance_guard"])
    assert guard["mode"] == "flat_localised_position"
    assert guard["window_steps"] == 12
    assert guard["maximum_travel_pixels"] == 1.0
    # the working runs travel at least 2.76 px per window, so the bound has real headroom
    assert guard["maximum_travel_pixels"] < 2.76
    # nothing else moved: the targets, planner, press band, recovery and gates are carried over
    ignored = {"no_advance_guard", "purpose", "route_id", "contract_sha256"}
    assert {k: v for k, v in v14.items() if k not in ignored} == {
        k: v for k, v in v13.items() if k not in ignored
    }
    assert cast(dict, v14)["targets_minimap_xy"] == [[50.0, 50.0], [80.0, 50.0], [80.0, 80.0], [50.0, 70.0]]
    assert store_runner._store_contract(v14, v14_sha)["no_advance_guard"] == guard
    for mutation in (
        {"mode": "anything"},
        {"window_steps": 1},
        {"window_steps": "twelve"},
        {"maximum_travel_pixels": 0.0},
    ):
        value = cast(
            dict,
            json.loads(
                (ROOT / "configs/movement_goal_navigation_route_b_v14.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        cast(dict, value["no_advance_guard"]).update(mutation)
        with pytest.raises(MobileTestbedError):
            store_runner._store_contract(value, "0" * 64)


def test_commit_approach_holds_the_bearing_for_the_declared_window() -> None:
    """The finest press tier is near the position noise floor, so a held bearing must not re-aim."""
    commit = store_runner._commit_approach
    committed: str | None = None
    steps = 0
    held: list[bool] = []
    requested: list[str] = []
    for index in range(6):
        asked = ("N", "NE", "E", "SE", "S", "SW")[index]
        applied, committed, steps, was_held = commit(asked, committed, steps, 3, in_band=True)
        requested.append(applied)
        held.append(was_held)
    # the first step chooses, the next two hold, then a new choice is made
    assert requested == ["N", "N", "N", "SE", "SE", "SE"]
    assert held == [False, True, True, False, True, True]
    # leaving the declared band clears the commitment, and a commitment of one is a no-op
    assert commit("N", "NE", 2, 3, in_band=False) == ("N", None, 0, False)
    assert commit("N", "NE", 2, 1, in_band=True) == ("N", None, 0, False)


def _grid_block() -> dict[str, object]:
    return {
        "mode": "measured_grid_mask",
        "minimum_rate_per_100ms": 0.08,
        "nominal_step_pixels": 6.0,
        "grid": {
            "schema_version": "hok-agent-traversability-grid-v1",
            "cell_pixels": 4.0,
            "minimum_samples": 3,
            "unbounded_movement_ms": 890.0,
            "cells": {
                # the measured v14 stall cell: north is ineffective, north-east is normal
                "14:11": {
                    "N": {"n": 125, "rate": 0.0711},
                    "NE": {"n": 132, "rate": 0.304},
                },
                # too few samples to judge: must never be masked
                "20:20": {"N": {"n": 2, "rate": 0.01}},
            },
        },
    }


def test_traversability_mask_removes_only_measured_ineffective_bearings() -> None:
    from hok_agent.traversability import (
        traversability_coverage,
        traversability_masked_direction,
        validate_traversability,
    )

    block = _grid_block()
    validate_traversability(block)
    mask = traversability_masked_direction
    # the hero stands in the stall cell: an ineffective north press is replaced by the nearest
    # bearing that was measured effective
    assert mask((57.0, 45.0), "N", block) == "NE"
    # a bearing that was measured effective is untouched
    assert mask((57.0, 45.0), "NE", block) == "NE"
    # a cell with no data at all fails open
    assert mask((66.0, 66.0), "N", block) == "N"
    # and a cell whose sample count is below the declared minimum also fails open
    assert mask((81.0, 81.0), "N", block) == "N"
    # a bearing outside the eight-way vocabulary is returned unchanged
    assert mask((57.0, 45.0), "wait", block) == "wait"
    assert traversability_coverage(block)["cell_bearing_above_minimum_samples"] == 2


def test_traversability_grid_normalises_the_estimate_by_the_declared_press(tmp_path: Path) -> None:
    """A short bounded press also moves the hero little, so the estimate must divide it out."""
    from hok_agent.traversability import build_traversability_grid

    run = tmp_path / "route-b-batch-synthetic" / "episode-01"
    run.mkdir(parents=True)
    rows = [
        # a 400 ms bounded press that moved 0.8 px -> 0.2 px per 100 ms
        {"step_id": 0, "position": [56.0, 44.0], "applied_movement": "N", "approach_press_ms": 400},
        {"step_id": 1, "position": [55.2, 44.0], "applied_movement": "NE", "approach_press_ms": 400},
        {"step_id": 2, "position": [55.2, 44.8], "applied_movement": "NE", "approach_press_ms": 400},
        # an unbounded step that moved 1.78 px -> 0.2 px per 100 ms
        {"step_id": 3, "position": [55.2, 46.58]},
    ]
    (run / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    grid = build_traversability_grid(
        runs_root=tmp_path,
        runs=("route-b-batch-synthetic",),
        cell_pixels=4.0,
        minimum_samples=1,
    )
    cells = cast(dict, grid["cells"])
    north = cells["14:11"]["N"]
    assert north["n"] == 1
    assert abs(north["rate"] - 0.2) < 0.01
    # discovery is for a human picking a source list, and never reaches a build
    assert discover_route_b_runs(tmp_path) == ("route-b-batch-synthetic",)


def test_traversability_build_needs_an_explicit_and_present_source_list(tmp_path: Path) -> None:
    """A grid must depend on a declared input list, never on whichever runs happen to exist."""
    from hok_agent.traversability import build_traversability_grid

    build = build_traversability_grid
    with pytest.raises(MobileTestbedError):
        build(runs_root=tmp_path, runs=(), cell_pixels=4.0, minimum_samples=1)
    with pytest.raises(MobileTestbedError):
        build(runs_root=tmp_path, runs=("route-b-batch-absent",), cell_pixels=4.0, minimum_samples=1)
    assert discover_route_b_runs(tmp_path / "nowhere") == ()


def test_traversability_block_pins_the_sources_it_was_built_from(tmp_path: Path) -> None:
    """The block carries its own source list, so a later contract regenerates without a glob."""
    from hok_agent.traversability import (
        make_traversability_block,
        verify_traversability_grid,
    )

    run = tmp_path / "route-b-batch-pinned" / "episode-01"
    run.mkdir(parents=True)
    rows = [
        {"step_id": 0, "position": [56.0, 44.0], "applied_movement": "N", "approach_press_ms": 400},
        {"step_id": 1, "position": [55.2, 44.0]},
    ]
    (run / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    block = make_traversability_block(
        runs_root=tmp_path,
        runs=("route-b-batch-pinned",),
        cell_pixels=4.0,
        minimum_samples=1,
        minimum_rate_per_100ms=0.08,
        nominal_step_pixels=6.0,
    )
    assert block["source_runs"] == ["route-b-batch-pinned"]
    checked = verify_traversability_grid(
        runs_root=tmp_path, runs=("route-b-batch-pinned",), block=block
    )
    assert checked["matches_frozen_grid"] is True
    # a grid rebuilt from a different source set must not be reported as matching
    other = tmp_path / "route-b-batch-other" / "episode-01"
    other.mkdir(parents=True)
    (other / "steps.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"step_id": 0, "position": [80.0, 80.0], "applied_movement": "S"},
                {"step_id": 1, "position": [83.0, 80.0]},
            ]
        ),
        encoding="utf-8",
    )
    drifted = verify_traversability_grid(
        runs_root=tmp_path, runs=("route-b-batch-other",), block=block
    )
    assert drifted["matches_frozen_grid"] is False
    assert drifted["differing_cell_count"] > 0


def test_mask_joystick_bearing_converts_between_the_two_vocabularies() -> None:
    """The grid is keyed by store bearings, so a joystick name must never reach it directly.

    This is the defect the first v15 device run exposed: the mask was wired to the joystick
    vocabulary, the grid matched none of it, and the mask silently did nothing for a whole run.
    """
    from hok_agent.traversability import traversability_masked_direction

    block = _grid_block()
    mask = store_runner._mask_joystick_bearing
    assert mask((57.0, 45.0), "north", block) == "north_east"
    assert mask((57.0, 45.0), "north_east", block) == "north_east"
    assert mask((66.0, 66.0), "north", block) == "north"
    assert mask((57.0, 45.0), "wait", block) == "wait"
    assert mask((57.0, 45.0), "not_a_bearing", block) == "not_a_bearing"
    # the raw grid call with a joystick name does nothing, which is exactly why the helper exists
    assert traversability_masked_direction((57.0, 45.0), "north", block) == "north"


def test_route_b_v15_declares_the_mask_and_the_commitment() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract
    from hok_agent.traversability import traversability_coverage, validate_traversability

    v14, v14_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v14.json"
    )
    v15, v15_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v15.json"
    )
    assert v14_sha == "caa0a58d67b323b7613b9b73b7053ebbdad6959dee2c9a175312756907acef89"
    assert "traversability" not in v14
    assert "commitment_steps" not in cast(dict, v14["final_approach"])
    # the grid is frozen with the contract, so the run and the grid cannot drift apart
    block = cast(dict, v15["traversability"])
    validate_traversability(block)
    coverage = traversability_coverage(block)
    assert coverage["cells"] > 100
    assert coverage["cell_bearing_above_minimum_samples"] > 100
    assert cast(dict, v15["final_approach"])["commitment_steps"] == 3
    # the grid must flag the measured stall and must not flag a bearing it barely sampled
    assert cast(dict, block["grid"])["cells"]["14:11"]["N"]["rate"] < block["minimum_rate_per_100ms"]
    assert cast(dict, block["grid"])["cells"]["14:11"]["NE"]["rate"] > block["minimum_rate_per_100ms"]
    # nothing else moved
    ignored = {"traversability", "final_approach", "purpose", "route_id", "contract_sha256"}
    assert {k: v for k, v in v15.items() if k not in ignored} == {
        k: v for k, v in v14.items() if k not in ignored
    }
    resolved = store_runner._store_contract(v15, v15_sha)
    assert resolved["traversability"] == block
    # a malformed grid must be rejected rather than silently ignored
    for mutation in (
        {"mode": "anything"},
        {"minimum_rate_per_100ms": 0.0},
    ):
        value = cast(
            dict,
            json.loads(
                (ROOT / "configs/movement_goal_navigation_route_b_v15.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        cast(dict, value["traversability"]).update(mutation)
        with pytest.raises(MobileTestbedError):
            store_runner._store_contract(value, "0" * 64)
    broken = cast(
        dict,
        json.loads(
            (ROOT / "configs/movement_goal_navigation_route_b_v15.json").read_text(encoding="utf-8")
        ),
    )
    cast(dict, cast(dict, broken["traversability"])["grid"])["cell_pixels"] = 0
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(broken, "0" * 64)


def _route_b_contract(name: str) -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads((ROOT / "configs" / name).read_text(encoding="utf-8")),
    )


def test_route_b_v8_declares_the_measured_replacement_bearing() -> None:
    """v7 stays on disk as the recorded inert experiment; v8 changes only the bearing rule."""
    from hok_agent.mobile_testbed import _goal_navigation_contract

    v7, v7_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v7.json"
    )
    v8, v8_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v8.json"
    )
    assert v7_sha == "13465a1194a666ccc2de774fb631fd7d20909959a8ac86053304204eb6da0642"
    recovery_v7 = cast(dict, v7["unknown_recovery"])
    recovery_v8 = cast(dict, v8["unknown_recovery"])
    assert recovery_v7["mode"] == "bounded_retrace"
    assert recovery_v8["mode"] == "bounded_retrace_or_waypoint"
    # only the bearing rule differs inside the block
    assert {k: v for k, v in recovery_v8.items() if k != "mode"} == {
        k: v for k, v in recovery_v7.items() if k != "mode"
    }
    # and only the recovery differs between the two contracts
    ignored = {"unknown_recovery", "purpose", "route_id", "contract_sha256"}
    assert {k: v for k, v in v8.items() if k not in ignored} == {
        k: v for k, v in v7.items() if k not in ignored
    }
    gap = cast(dict, v8["store"])["maximum_localization_gap_frames"]
    assert recovery_v8["trigger_after_missing_frames"] + recovery_v8["maximum_recovery_steps"] <= gap
    assert store_runner._store_contract(v8, v8_sha)["unknown_recovery"] == recovery_v8


def test_route_b_v8_rejects_a_recovery_that_outlives_the_guard() -> None:
    for trigger, maximum in ((3, 8), (11, 1)):
        value = _route_b_contract("movement_goal_navigation_route_b_v8.json")
        recovery = cast(dict, value["unknown_recovery"])
        recovery["trigger_after_missing_frames"] = trigger
        recovery["maximum_recovery_steps"] = maximum
        with pytest.raises(MobileTestbedError):
            store_runner._store_contract(value, "0" * 64)
    for mutation in (
        {"mode": "spin_in_place"},
        {"maximum_recovery_steps": 0},
        {"trigger_after_missing_frames": 0},
        {"hold_ms": 0},
        {"nominal_step_pixels": 0.0},
    ):
        value = _route_b_contract("movement_goal_navigation_route_b_v8.json")
        cast(dict, value["unknown_recovery"]).update(mutation)
        with pytest.raises(MobileTestbedError):
            store_runner._store_contract(value, "0" * 64)


def test_route_b_v7_declares_the_bounded_retrace_recovery() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract

    contract, sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v7.json"
    )
    assert len(sha) == 64
    recovery = cast(dict, contract["unknown_recovery"])
    assert recovery["mode"] == "bounded_retrace"
    assert recovery["trigger_after_missing_frames"] == 3
    assert recovery["maximum_recovery_steps"] == 6
    assert recovery["hold_ms"] > 0
    assert recovery["nominal_step_pixels"] > 0
    gap = cast(dict, contract["store"])["maximum_localization_gap_frames"]
    assert recovery["trigger_after_missing_frames"] + recovery["maximum_recovery_steps"] <= gap
    # v7 keeps the v6 planner and adds nothing that replaces it
    assert cast(dict, contract["planner"])["mode"] == "path_progress_with_feasibility"
    resolved = store_runner._store_contract(contract, sha)
    assert resolved["unknown_recovery"] == recovery


def test_route_b_v7_keeps_v6_unchanged() -> None:
    from hok_agent.mobile_testbed import _goal_navigation_contract

    v6, v6_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v6.json"
    )
    assert v6_sha == "fcb4d8086a91eae4f1f84c4a8dff173b5f95527f534e49d69fdcd32a9e41400c"
    assert "unknown_recovery" not in v6
    # v7 is preserved on disk as the recorded inert-bearing experiment
    v7, v7_sha = _goal_navigation_contract(
        ROOT / "configs/movement_goal_navigation_route_b_v7.json"
    )
    assert v7_sha == "13465a1194a666ccc2de774fb631fd7d20909959a8ac86053304204eb6da0642"
    assert cast(dict, v7["unknown_recovery"])["mode"] == "bounded_retrace"
