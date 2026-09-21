"""Focused tests for the controlled response probe analyser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from hok_agent import mobile_navigation_store as store_runner
from hok_agent.mobile_testbed import MobileTestbedError
from hok_agent.traversability import traversability_masked_direction
from hok_agent.traversability_probe import (
    analyse_probe_pulses,
    explain_probe_grid,
    probe_grid_block,
    validate_probe_analysis,
)

ROOT = Path(__file__).resolve().parents[1]
CELL_PIXELS = 4.0
MINIMUM_PULSES = 3
IDLE_MARGIN = 1.0
MOVEMENT_MS = 2500.0


def _pulse(
    direction: str,
    *,
    dy: float,
    dx: float,
    position: tuple[float, float] = (57.0, 45.0),
) -> dict[str, object]:
    return {
        "position": list(position),
        "direction": direction,
        "dy": dy,
        "dx": dx,
        "movement_ms": MOVEMENT_MS,
    }


def _analyse(
    observations: list[dict[str, object]], *, idle_rate_per_100ms: float = 0.0
) -> dict[str, object]:
    return analyse_probe_pulses(
        observations,
        cell_pixels=CELL_PIXELS,
        minimum_pulses=MINIMUM_PULSES,
        idle_margin_pixels=IDLE_MARGIN,
        idle_rate_per_100ms=idle_rate_per_100ms,
    )


def test_probe_analysis_separates_a_blocked_bearing_from_a_responsive_one() -> None:
    observations = [
        # north is the direction the planner pressed 52 times into at this cell: it only drifts
        *[_pulse("N", dy=-0.30, dx=0.0) for _ in range(4)],
        # north-east was measured three to four times more effective in the same cell
        *[_pulse("NE", dy=-3.0, dx=3.0) for _ in range(4)],
    ]
    # a realistic measured idle bound: 0.33 px of drift over a 2500 ms control window
    analysis = _analyse(observations, idle_rate_per_100ms=0.0132)
    cells = cast(dict, analysis["cells"])
    entry = cells["14:11"]
    assert entry["N"]["above_idle"] == 0
    assert entry["N"]["blocked"] is True
    assert entry["N"]["rate"] < 0.08
    assert entry["NE"]["above_idle"] == 4
    assert entry["NE"]["blocked"] is False
    assert entry["NE"]["rate"] > 0.08
    diagnostics = cast(dict, analysis["diagnostics"])
    assert diagnostics["cell_bearing_observations"] == 2
    assert diagnostics["cell_bearing_blocked"] == 1


def test_probe_analysis_rejects_drift_dressed_as_movement() -> None:
    """The defect this analyser exists to prevent: a grid reads idle drift as commanded movement.

    The earlier probe measured a pulse-hold median of 0.344 px against 0.317 px of idle drift. When
    the measured idle bound is that large, a press whose displacement is explained by it must
    contribute no effective rate at all rather than a small one a threshold might accept.
    """
    # 0.33 px over 2500 ms is 0.0132 px per 100 ms, so that is the honest idle bound here
    drifting = [_pulse("N", dy=-0.33, dx=0.0) for _ in range(4)]
    drifted = cast(dict, cast(dict, _analyse(drifting, idle_rate_per_100ms=0.0132)["cells"]))
    entry = drifted["14:11"]["N"]
    assert entry["above_idle"] == 0
    assert entry["blocked"] is True
    assert entry["rate"] < 0.001
    # The same pulses with no measured idle bound would have been credited as movement.
    credited = cast(dict, cast(dict, _analyse(drifting)["cells"])["14:11"])["N"]
    assert credited["median_press_rate"] > 0.013
    assert credited["above_idle"] == 0 or credited["rate"] > entry["rate"]


def test_probe_analysis_leaves_under_sampled_bearings_unknown() -> None:
    """Two pulses cannot block a bearing: the mask's own sample floor must still fail open."""
    observations = [_pulse("N", dy=-0.3, dx=0.0) for _ in range(2)]
    entry = cast(dict, cast(dict, _analyse(observations)["cells"])["14:11"])["N"]
    assert entry["n"] == 2
    assert entry["above_idle"] == 0
    assert entry["blocked"] is False
    block = _composed_block(observations)
    grid_entry = cast(dict, cast(dict, block["grid"])["cells"])["14:11"]["N"]
    assert grid_entry["n"] == 2
    # the mask's sample floor is untouched by the probe, so two pulses cannot remove a bearing
    assert int(cast(int, block["grid"]["minimum_samples"])) == MINIMUM_PULSES
    assert traversability_masked_direction((57.0, 45.0), "N", block) == "N"


def test_probe_analysis_rejects_a_bearing_outside_the_vocabulary() -> None:
    with pytest.raises(MobileTestbedError):
        _analyse([_pulse("north", dy=-1.0, dx=0.0)])
    with pytest.raises(MobileTestbedError):
        analyse_probe_pulses(
            [dict(_pulse("N", dy=-1.0, dx=0.0), movement_ms=0.0)],
            cell_pixels=CELL_PIXELS,
            minimum_pulses=MINIMUM_PULSES,
            idle_margin_pixels=IDLE_MARGIN,
            idle_rate_per_100ms=0.0,
        )


def _composed_block(observations: list[dict[str, object]]) -> dict[str, object]:
    return probe_grid_block(
        _analyse(observations),
        cell_pixels=CELL_PIXELS,
        press_ms=MOVEMENT_MS,
        idle_rate_per_100ms=0.0132,
        minimum_samples=MINIMUM_PULSES,
        minimum_rate_per_100ms=0.08,
        nominal_step_pixels=6.0,
        source_sessions=["probe-session-synthetic"],
    )


def test_probe_grid_is_consumed_by_the_existing_mask_without_a_mask_change() -> None:
    """Matching the frozen grid shape means the Router mask needs no probe special case."""
    observations = [
        *[_pulse("N", dy=-0.30, dx=0.0) for _ in range(4)],
        *[_pulse("NE", dy=-3.0, dx=3.0) for _ in range(4)],
    ]
    block = _composed_block(observations)
    assert block["mode"] == "measured_grid_mask"
    assert cast(dict, block["probe"])["schema_version"] == "hok-agent-traversability-probe-v1"
    assert block["coverage"] == {
        "cells": 1,
        "cell_bearing_observations": 2,
        "cell_bearing_above_minimum_samples": 2,
    }
    # the measured-ineffective bearing is masked, the responsive one is left alone
    assert traversability_masked_direction((57.0, 45.0), "N", block) == "NE"
    assert traversability_masked_direction((57.0, 45.0), "NE", block) == "NE"
    # and it survives the joystick vocabulary conversion the runtime actually uses
    assert store_runner._mask_joystick_bearing((57.0, 45.0), "north", block) == "north_east"
    # a cell the probe never visited still fails open
    assert traversability_masked_direction((80.0, 80.0), "N", block) == "N"


def test_probe_analysis_and_grid_validation_reject_malformed_parameters() -> None:
    with pytest.raises(MobileTestbedError):
        validate_probe_analysis({"mode": "measured_probe", "cell_pixels": 4.0})
    validate_probe_analysis(
        {
            "mode": "measured_probe",
            "cell_pixels": 4.0,
            "minimum_pulses": 3,
            "idle_margin_pixels": 1.0,
        }
    )
    assert explain_probe_grid({})["cell_bearing_observations"] == 0
    assert explain_probe_grid({})["above_idle_fraction"] == 0.0
    # a block built from a probe still satisfies the frozen grid validator the mask is bound to
    from hok_agent.traversability import validate_traversability

    validate_traversability(_composed_block([_pulse("N", dy=-3.0, dx=0.0)]))


def test_probe_analysis_attributes_pulses_to_the_cell_the_hero_stood_in() -> None:
    """Attribution is by press position, matching how the incidental grid was aggregated."""
    observations = [
        _pulse("N", dy=-3.0, dx=0.0, position=(57.0, 45.0)),
        _pulse("N", dy=-3.0, dx=0.0, position=(59.0, 45.0)),
    ]
    cells = cast(dict, _analyse(observations)["cells"])
    assert sorted(cells) == ["14:11"]
    assert cells["14:11"]["N"]["n"] == 2


def test_existing_probe_contracts_keep_their_recorded_digests() -> None:
    """A new probe version must never overwrite an old one.

    The traversability work first wrote its contract as v4 without checking the directory, which
    silently replaced an existing v4 that three earlier batches were recorded against. Nothing
    failed, because no test pinned it. These digests pin the neighbouring contracts so the next
    version has to check for a free name instead of assuming one.
    """
    digests = {
        "configs/movement_active_probe_v4.json": (
            "9adbfcf52d3519510bf785b195bc32a2edc0d516229714783b61d966ab582666"
        ),
        "configs/movement_active_probe_v10.json": (
            "3ec3dbcce5fee4c658623adf1c502bf851303eee99114a148a6168a44ed7a619"
        ),
    }
    for name, digest in digests.items():
        assert json.loads((ROOT / name).read_text())["contract_sha256"] == digest


def test_probe_contract_v10_declares_the_paired_and_balanced_schedule() -> None:
    """The probe must declare a direction-balanced schedule with interleaved idle controls."""
    contract = json.loads((ROOT / "configs/movement_active_probe_v10.json").read_text())
    assert contract["direction_order"] == "opposite-pairs"
    assert contract["directions"] == [
        "north",
        "north_east",
        "east",
        "south_east",
        "south",
        "south_west",
        "west",
        "north_west",
    ]
    assert contract["training_allowed"] is False
    assert contract["test_allowed"] is False
    # controls are interleaved, and the measurement pairs each press with a matched idle window
    assert contract["control_windows"] >= 8
    assert contract["measurement"]["baseline"] == "last_analysis_frame_before_press_ack"
    analysis = cast(dict, contract["traversability_analysis"])
    assert analysis["mode"] == "measured_probe"
    assert analysis["cell_pixels"] == 4.0
    assert analysis["idle_margin_pixels"] > 0
    # the session stays bounded
    budgets = cast(dict, contract["budgets"])
    assert budgets["maximum_duration_seconds_per_session"] <= 480
    pulses = int(contract["pulses_per_direction"]) * len(contract["directions"])
    hold = int(contract["hold_ms"])
    per_pulse = hold + int(contract["observation_ms"]) + int(contract["inter_pulse_gap_ms"])
    assert pulses * per_pulse + contract["control_windows"] * (
        int(contract["control_window_ms"]) + int(contract["inter_pulse_gap_ms"])
    ) <= budgets["maximum_duration_seconds_per_session"] * 1000
