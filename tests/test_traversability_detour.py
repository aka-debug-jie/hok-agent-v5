"""The declared bounded grid detour: the capability the cold-start limit named.

The mask can only remove a bearing, so it cannot plan the two-step way around a cell where the
goal-directed bearing does not work. These tests pin the planner's contract - bounded, shortest,
deterministic, mask-respecting, region-respecting - and replay it against the frozen v15 grid
and the cells the recorded cold-start failures actually stalled in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from hok_agent import mobile_navigation_store as store_runner
from hok_agent import mobile_testbed
from hok_agent.mobile_testbed import MobileTestbedError, _goal_navigation_contract
from hok_agent.traversability import (
    BEARING_STEPS,
    MOVEMENT_ORDER,
    plan_grid_detour,
    tier_restricted_vectors,
    validate_detour,
)

REGION = {"minimum_y": 35, "maximum_y": 95, "minimum_x": 35, "maximum_x": 95}


def _block(cells: dict[str, dict[str, dict[str, float]]]) -> dict[str, object]:
    return {
        "mode": "measured_grid_mask",
        "minimum_rate_per_100ms": 0.08,
        "nominal_step_pixels": 6.0,
        "source_runs": ["synthetic"],
        "coverage": {},
        "grid": {
            "schema_version": "hok-agent-traversability-grid-v1",
            "cell_pixels": 4.0,
            "minimum_samples": 3,
            "cells": cells,
        },
    }


def _measured(**bearings: float) -> dict[str, dict[str, float]]:
    return {bearing: {"n": 10, "rate": rate} for bearing, rate in bearings.items()}


def _walk(start: tuple[float, float], bearings: tuple[str, ...]) -> tuple[int, int]:
    row = int(start[0] // 4.0)
    column = int(start[1] // 4.0)
    for bearing in bearings:
        step_y, step_x = BEARING_STEPS[bearing]
        row += int(round(step_y))
        column += int(round(step_x))
    return row, column


def test_plan_grid_detour_steps_around_the_bearing_the_mask_removes() -> None:
    """North is obstructed at the start and next cell, so the plan must go sideways first."""
    cells = {
        "15:12": _measured(N=0.01, NE=0.3, E=0.3, NW=0.3),
        "14:12": _measured(N=0.01, NE=0.3, NW=0.3),
    }
    plan = plan_grid_detour(
        position=(60.0, 50.0),
        goal=(50.0, 50.0),
        block=_block(cells),
        region=REGION,
        maximum_steps=8,
    )
    assert plan is not None
    assert plan.bearings == ("NE", "N", "NW")
    assert _walk((60.0, 50.0), plan.bearings) == (12, 12)
    # the first step is not the bearing the mask removed
    assert plan.bearings[0] != "N"
    # the unmeasured cells on the way are reported rather than hidden
    assert plan.unknown_steps == 2


def test_plan_grid_detour_takes_the_straight_line_when_nothing_is_obstructed() -> None:
    plan = plan_grid_detour(
        position=(60.0, 50.0),
        goal=(50.0, 50.0),
        block=_block({}),
        region=REGION,
        maximum_steps=8,
    )
    assert plan is not None
    assert plan.bearings == ("N", "N", "N")
    assert plan.unknown_steps == 3


def test_plan_grid_detour_returns_none_when_no_path_fits_the_declared_cap() -> None:
    """A cap that is smaller than the only path is a recorded outcome, not a silent fallback."""
    cells = {
        "15:12": _measured(N=0.01, NE=0.01, E=0.01, SE=0.01, S=0.01, SW=0.01, W=0.01, NW=0.01),
    }
    plan = plan_grid_detour(
        position=(60.0, 50.0),
        goal=(50.0, 50.0),
        block=_block(cells),
        region=REGION,
        maximum_steps=4,
    )
    assert plan is None


def test_plan_grid_detour_never_leaves_the_declared_region() -> None:
    """A goal outside the free movement region has no admissible path."""
    plan = plan_grid_detour(
        position=(50.0, 50.0),
        goal=(10.0, 10.0),
        block=_block({}),
        region=REGION,
        maximum_steps=20,
    )
    assert plan is None


def test_plan_grid_detour_requires_a_declared_bearing_permutation() -> None:
    with pytest.raises(MobileTestbedError):
        plan_grid_detour(
            position=(60.0, 50.0),
            goal=(50.0, 50.0),
            block=_block({}),
            region=REGION,
            maximum_steps=8,
            bearing_order=("N", "N", "E", "SE", "S", "SW", "W", "NW"),
        )
    with pytest.raises(MobileTestbedError):
        plan_grid_detour(
            position=(60.0, 50.0),
            goal=(50.0, 50.0),
            block=_block({}),
            region=REGION,
            maximum_steps=0,
        )


def test_the_detour_plan_vocabulary_is_convertible_to_the_joystick_vocabulary() -> None:
    """The planner and the Router use disjoint vocabularies, and the boundary must be total.

    This pins the second instance of the vocabulary trap: the planner returns `NE`, the Router and
    the joystick want `north_east`, and the two overlap on nothing. Handing a plan bearing straight
    to the Router would silently mask nothing and then send an invalid direction, which is exactly
    what the first device attempt of the detour did.
    """
    for bearing in MOVEMENT_ORDER:
        joystick = store_runner.STORE_TO_JOYSTICK_DIRECTION[bearing]
        assert joystick in mobile_testbed.MOVEMENTS
        assert store_runner.JOYSTICK_TO_STORE_DIRECTION[joystick] == bearing
        assert joystick not in MOVEMENT_ORDER


def test_validate_detour_requires_every_declared_number() -> None:
    good: dict[str, object] = {
        "schema_version": "hok-agent-declared-detour-v1",
        "mode": "bounded_grid_bfs",
        "maximum_steps": 8,
        "maximum_attempts_per_episode": 2,
        "trigger_stall_steps": 6,
        "trigger_stall_progress_pixels": 2.0,
        "confirmation_steps": 3,
        "confirmation_minimum_progress_pixels": 2.0,
        "bearing_order": list(MOVEMENT_ORDER),
    }
    assert validate_detour(dict(good))["mode"] == "bounded_grid_bfs"
    for broken in (
        {**good, "mode": "grid"},
        {**good, "schema_version": "other"},
        {**good, "maximum_steps": 0},
        {**good, "maximum_attempts_per_episode": 0},
        {**good, "trigger_stall_steps": 0},
        {**good, "trigger_stall_progress_pixels": 0.0},
        {**good, "confirmation_steps": 0},
        {**good, "confirmation_minimum_progress_pixels": 0.0},
        {**good, "bearing_order": ["N"]},
    ):
        with pytest.raises(MobileTestbedError):
            validate_detour(broken)


def test_route_b_v18_contract_declares_the_bounded_grid_detour() -> None:
    config = Path(__file__).resolve().parents[1] / "configs"
    contract, _sha = _goal_navigation_contract(
        config / "movement_goal_navigation_route_b_v18.json"
    )
    detour = cast(dict[str, object], contract["detour"])
    assert detour["mode"] == "bounded_grid_bfs"
    assert detour["maximum_steps"] == 8
    assert detour["maximum_attempts_per_episode"] == 2
    assert detour["bearing_order"] == list(MOVEMENT_ORDER)
    # the grid the detour searches is the same frozen v15 grid the mask reads
    assert contract["traversability"] is not None


def test_a_declared_detour_without_a_grid_is_refused() -> None:
    base = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "movement_goal_navigation_route_b_v18.json"
        ).read_text(encoding="utf-8")
    )
    del base["traversability"]
    with pytest.raises(MobileTestbedError):
        store_runner._store_contract(base, "0" * 64)


def test_episode_outcome_reports_an_exhausted_detour_rather_than_a_timeout() -> None:
    outcome = store_runner._episode_outcome
    common = {
        "arrived": False,
        "death": False,
        "outside_region": False,
        "no_advance": False,
        "missing_streak": 0,
        "maximum_gap": 10,
    }
    assert outcome(**common, budget_exhausted=False, detour_exhausted=True) == (
        True,
        "DETOUR_FAILURE",
        "ERROR",
        "detour_exhausted",
    )
    # a budget that is also spent does not mask the detour failure
    assert (
        outcome(**common, budget_exhausted=True, detour_exhausted=True)[3] == "detour_exhausted"
    )
    # a lost marker is still a capture failure, and stays above the detour reason
    assert (
        outcome(
            **{**common, "missing_streak": 99}, budget_exhausted=False, detour_exhausted=True
        )[3]
        == "localization_gap"
    )


def test_the_frozen_v15_grid_gives_the_recorded_cold_start_stalls_a_detour() -> None:
    """Replay the planner on the real frozen grid and the cells the failures actually stalled in.

    The v17 attempt stalls in cell 14:12, which is exactly the cell where the frozen grid removes
    north: its only better bearing is north-east, which moves away from the goal, and the greedy
    objective will not take that two-step detour. This is the offline check the protocol requires
    before any device session is spent, and it uses the recorded grid rather than a synthetic one.
    """
    config = Path(__file__).resolve().parents[1] / "configs"
    contract, _sha = _goal_navigation_contract(
        config / "movement_goal_navigation_route_b_v15.json"
    )
    block = cast(dict[str, object], contract["traversability"])
    region = cast(dict[str, object], contract["free_movement_region"])
    expected = {
        "15:12": ("N", "NE", "NW"),
        "14:12": ("NE", "NW"),
        "14:11": ("NE", "N"),
    }
    for cell, bearings in expected.items():
        row, column = (int(part) for part in cell.split(":"))
        position = ((row + 0.5) * 4.0, (column + 0.5) * 4.0)
        plan = plan_grid_detour(
            position=position,
            goal=(50.0, 50.0),
            block=block,
            region=region,
            maximum_steps=8,
        )
        assert plan is not None, cell
        assert plan.bearings == bearings, cell
        assert _walk(position, plan.bearings) == (12, 12), cell
        assert plan.unknown_steps == 0, cell


def test_tier_restricted_vectors_report_the_slide_the_projection_hides(tmp_path: Path) -> None:
    """The mask decides on one scalar; the vector shows what the press actually did.

    At cell 14:12 the frozen grid removes north on a 0.0548 projection while the measured mean
    displacement is a 1.70 px slide west. This pins the measurement that makes that visible, at the
    Router's bounded press tier only, and pins that an unbounded press is not mixed in.
    """
    episode = tmp_path / "route-b-batch-synthetic" / "episode-01"
    episode.mkdir(parents=True)
    rows = [
        {"position": [59.0, 50.0], "applied_movement": "N", "approach_press_ms": 400},
        {"position": [58.8, 49.5], "applied_movement": "N", "approach_press_ms": 400},
        {"position": [58.6, 49.0], "applied_movement": "N", "approach_press_ms": 400},
        {"position": [58.4, 48.5], "applied_movement": "N", "approach_press_ms": 400},
        {"position": [58.2, 48.0], "applied_movement": "N"},
        {"position": [57.0, 47.5], "applied_movement": "N"},
    ]
    (episode / "steps.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    measured = tier_restricted_vectors(tmp_path, ("route-b-batch-synthetic",), 4.0)
    cells = cast(dict[str, dict[str, dict[str, float]]], measured["cells"])
    north = cells["14:12"]["N"]
    assert north["n"] == 4
    assert north["mean_delta_y"] == pytest.approx(-0.20)
    assert north["mean_delta_x"] == pytest.approx(-0.50)
    assert north["projection_rate"] == pytest.approx(0.05)
    # the projection scores the press as barely responsive while it actually moved 0.54 px per press
    assert north["magnitude_rate"] == pytest.approx(0.1346, abs=1e-3)
    assert north["magnitude_rate"] > north["projection_rate"]
    assert north["mean_press_ms"] == 400.0


def test_probe_contract_accepts_a_stationarity_preflight_and_refuses_a_useless_one(
    tmp_path: Path,
) -> None:
    """The pre-flight is optional so earlier probe contracts load, but a declared one must bite.

    The wall-corridor traverse spent two full sessions and then failed because the hero was walking
    during the idle windows - measured idle drift 0.4551 px per 100 ms against bounded press rates
    of 0.05 to 0.18 - and the analyser subtracts the idle rate it measures, so a moving hero raises
    the floor above every press. A pre-flight that cannot fail on anything would be worse than none.
    """
    import json as _json

    from hok_agent import mobile_testbed as testbed

    probe_path = (
        Path(__file__).resolve().parents[1] / "configs" / "movement_active_probe_v14.json"
    )
    base = _json.loads(probe_path.read_text(encoding="utf-8"))

    def write(block: object) -> Path:
        payload = dict(base)
        if block is None:
            payload.pop("stationarity_preflight", None)
        else:
            payload["stationarity_preflight"] = block
        unsigned = {k: v for k, v in payload.items() if k != "contract_sha256"}
        payload["contract_sha256"] = hashlib.sha256(
            _json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        path = Path(base_path) / f"probe-{abs(hash(str(block)))}.json"
        path.write_text(_json.dumps(payload), encoding="utf-8")
        return path

    base_path = tmp_path
    declared = {"frames": 8, "maximum_travel_pixels": 1.5}
    contract, _sha = testbed._active_probe_contract(write(declared))
    assert contract["stationarity_preflight"] == declared
    # absence is allowed so the earlier contracts keep loading
    contract, _sha = testbed._active_probe_contract(write(None))
    assert "stationarity_preflight" not in contract
    for broken in (
        {"frames": 1, "maximum_travel_pixels": 1.5},
        {"frames": 8, "maximum_travel_pixels": 0.0},
        {"frames": 8, "maximum_travel_pixels": -1.0},
        {"frames": 8},
        {},
    ):
        with pytest.raises(testbed.MobileTestbedError):
            testbed._active_probe_contract(write(broken))


def test_probe_contract_may_declare_a_press_subset(tmp_path: Path) -> None:
    """A directional sweep gives up cancellation on purpose, and the declaration must be exact.

    In the wall corridor south is responsive while north is not, so the balanced eight-direction
    schedule cancels less well in one axis and walks the hero away from the wall. A contract may
    therefore narrow what it presses, but only to a unique subset of the vocabulary it still
    declares in full, so the analysis vocabulary is unchanged and only the schedule is narrowed.
    """
    import json as _json

    from hok_agent import mobile_testbed as testbed

    probe_path = (
        Path(__file__).resolve().parents[1] / "configs" / "movement_active_probe_v15.json"
    )
    base = _json.loads(probe_path.read_text(encoding="utf-8"))
    assert base["press_directions"] == ["north", "north_east", "north_west"]

    def write(block: object) -> Path:
        payload = dict(base)
        if block is None:
            payload.pop("press_directions", None)
        else:
            payload["press_directions"] = block
        unsigned = {k: v for k, v in payload.items() if k != "contract_sha256"}
        payload["contract_sha256"] = hashlib.sha256(
            _json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        path = tmp_path / f"probe-subset-{abs(hash(str(block)))}.json"
        path.write_text(_json.dumps(payload), encoding="utf-8")
        return path

    northward = ["north", "north_east", "north_west"]
    contract, _sha = testbed._active_probe_contract(write(northward))
    assert contract["press_directions"] == northward
    # the full vocabulary is still declared, so the analysis side is untouched
    assert contract["directions"] == ["north", "north_east", "east", "south_east", "south",
                                      "south_west", "west", "north_west"]
    # absence keeps the balanced schedule, so the earlier contracts still load
    contract, _sha = testbed._active_probe_contract(write(None))
    assert "press_directions" not in contract
    # a unique non-empty subset of the declared vocabulary, and nothing else
    for broken in ([], ["north", "north"], ["up"], ["north", 3]):
        with pytest.raises(testbed.MobileTestbedError):
            testbed._active_probe_contract(write(broken))
    # the schedule presses only the declared subset, in the declared order
    subset, _sha = testbed._active_probe_contract(write(["north"]))
    schedule = testbed.plan_active_probe_schedule(subset)
    pulses = [entry["direction"] for entry in schedule if entry["kind"] == "pulse"]
    assert set(pulses) == {"north"}
    assert len(pulses) == subset["pulses_per_direction"]
