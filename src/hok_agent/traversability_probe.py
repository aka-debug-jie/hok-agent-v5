"""Turn a controlled response probe into a per-cell traversability grid.

The first traversability grid was aggregated from transitions that the planner produced while it
was already failing, so its samples are biased exactly where and in the directions the planner
preferred: at the cell where the v14 stage died, north carried 125 samples and north was the
ineffective direction. A probe replaces that with a declared, direction-balanced schedule, and it
separates the commanded response from idle drift with interleaved control windows.

Drift separation is the part that has to be right, and the first version of this module got it
wrong in a way only the real numbers exposed. It paired each press with the window immediately
before it, which is the previous pulse's own observation window; under an opposite-pairs schedule
that window holds the opposite bearing's residual motion, so subtracting it doubled the estimate
instead of correcting it and reported 0.8 px per 100 ms where the raw projections were 0.39. The
idle baseline must come from a window in which nothing is commanded, so this module takes the
measured idle speed from the contract's declared control windows and uses that.

The earlier forensics measured a pulse-hold median of 0.344 px against 0.317 px of idle drift, so
the margin matters: a bearing counts as responsive only when its rate exceeds the measured idle
bound and its effective displacement also clears a declared cushion.

The output is the same frozen grid shape that the mask in `hok_agent.traversability` already
consumes, so a probe grid replaces an incidental grid in a contract without changing the
mask. It is a strictly better estimate of the same quantity: drift-subtracted, direction-balanced,
and sampled at a declared press duration rather than at whatever duration the planner happened to
choose.
"""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from hok_agent.mobile_testbed import MobileTestbedError
from hok_agent.traversability import (
    BEARING_STEPS,
    MOVEMENT_ORDER,
    TRAVERSABILITY_SCHEMA,
    traversability_coverage,
)

PROBE_TRAVERSABILITY_SCHEMA = "hok-agent-traversability-probe-v1"


def validate_probe_analysis(block: object) -> dict[str, object]:
    """Validate the declared probe analysis parameters."""
    if (
        not isinstance(block, dict)
        or block.get("mode") != "measured_probe"
        or not isinstance(block.get("cell_pixels"), (int, float))
        or float(cast(float, block["cell_pixels"])) <= 0
        or not isinstance(block.get("minimum_pulses"), int)
        or int(cast(int, block["minimum_pulses"])) < 1
        or not isinstance(block.get("idle_margin_pixels"), (int, float))
        or float(cast(float, block["idle_margin_pixels"])) < 0
    ):
        raise MobileTestbedError("traversability probe analysis differs")
    return block


def explain_probe_grid(cells: dict[str, dict[str, dict[str, object]]]) -> dict[str, object]:
    """Summarise how much of the probe result is actually measured rather than inherited drift."""
    observations = 0
    above_idle = 0
    blocked = 0
    for entry in cells.values():
        for measured in entry.values():
            observations += 1
            if int(cast(int, measured["above_idle"])) > 0:
                above_idle += 1
            if bool(measured["blocked"]):
                blocked += 1
    return {
        "cell_bearing_observations": observations,
        "cell_bearing_above_idle": above_idle,
        "cell_bearing_blocked": blocked,
        "above_idle_fraction": (above_idle / observations) if observations else 0.0,
    }


def analyse_probe_pulses(
    observations: list[dict[str, object]],
    *,
    cell_pixels: float,
    minimum_pulses: int,
    idle_margin_pixels: float,
    idle_rate_per_100ms: float,
) -> dict[str, object]:
    """Aggregate paired pulse observations into a drift-corrected per-cell, per-bearing grid.

    Each observation carries the localised position at press, the commanded bearing in the recorded
    store vocabulary, the displacement over the press window, and the declared press duration that
    produced it. `idle_rate_per_100ms` is measured from the contract's control windows, in which
    nothing is commanded.

    The reported rate subtracts that measured idle bound and clamps at zero, so a bearing that only
    drifts contributes no effective movement. That keeps the estimate monotone in the thing the
    mask asks about - whether pressing this way moves the hero - without inventing a threshold
    fitted to a schedule.
    """
    samples: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        direction = str(observation["direction"])
        if direction not in BEARING_STEPS:
            raise MobileTestbedError(f"traversability probe bearing differs: {direction}")
        position = cast(list[float] | tuple[float, float], observation["position"])
        movement_ms = float(cast(float, observation["movement_ms"]))
        if movement_ms <= 0:
            raise MobileTestbedError("traversability probe movement duration differs")
        step_y, step_x = BEARING_STEPS[direction]
        projection = (
            float(cast(float, observation["dy"])) * step_y
            + float(cast(float, observation["dx"])) * step_x
        )
        press_rate = projection / movement_ms * 100.0
        effective_rate = max(0.0, press_rate - idle_rate_per_100ms)
        effective_pixels = effective_rate * movement_ms / 100.0
        key = f"{int(float(position[0]) // cell_pixels)}:{int(float(position[1]) // cell_pixels)}"
        samples[key][direction].append(
            {
                "rate": effective_rate,
                "above": 1.0 if effective_pixels >= idle_margin_pixels else 0.0,
                "press_rate": press_rate,
            }
        )
    cells: dict[str, dict[str, dict[str, object]]] = {}
    for key in sorted(samples):
        entry: dict[str, dict[str, object]] = {}
        for direction in MOVEMENT_ORDER:
            values = samples[key].get(direction)
            if not values:
                continue
            count = len(values)
            above = int(sum(value["above"] for value in values))
            rate = sum(value["rate"] for value in values) / count
            entry[direction] = {
                "n": count,
                "rate": round(rate, 4),
                "above_idle": above,
                "median_press_rate": round(
                    sorted(value["press_rate"] for value in values)[count // 2], 4
                ),
                # A bearing is blocked only when it was measured enough times and never once beat
                # the idle band. Too few pulses stays unknown, which the mask reads as passable.
                "blocked": count >= minimum_pulses and above == 0,
            }
        if entry:
            cells[key] = entry
    return {
        "cells": cells,
        "diagnostics": explain_probe_grid(cells),
    }


def probe_grid_block(
    analysis: dict[str, object],
    *,
    cell_pixels: float,
    press_ms: float,
    idle_rate_per_100ms: float,
    minimum_samples: int,
    minimum_rate_per_100ms: float,
    nominal_step_pixels: float,
    source_sessions: list[str],
) -> dict[str, object]:
    """Assemble the analyser output into the frozen grid shape the mask already consumes.

    `minimum_samples` is the mask's own sample floor and is deliberately separate from the
    analyser's `minimum_pulses`: a bearing whose probe verdict is blocked still reports its real
    pulse count, so the mask reaches its `blocked` verdict through the same rule it already applies
    to an incidental grid, with no special case for probe data.
    """
    cells = cast(dict[str, dict[str, dict[str, object]]], analysis["cells"])
    grid_cells: dict[str, dict[str, dict[str, object]]] = {}
    for key, entry in cells.items():
        grid_entry: dict[str, dict[str, object]] = {}
        for direction, measured in entry.items():
            grid_entry[direction] = {
                "n": int(cast(int, measured["n"])),
                "rate": float(cast(float, measured["rate"])),
                # kept beside the two fields the mask reads, so the evidence for a verdict is
                # persisted with the verdict instead of only in a transient analysis object
                "above_idle": int(cast(int, measured["above_idle"])),
                "median_press_rate": float(cast(float, measured["median_press_rate"])),
            }
        grid_cells[key] = grid_entry
    block: dict[str, object] = {
        "mode": "measured_grid_mask",
        "minimum_rate_per_100ms": minimum_rate_per_100ms,
        "nominal_step_pixels": nominal_step_pixels,
        "probe": {
            "schema_version": PROBE_TRAVERSABILITY_SCHEMA,
            "cell_pixels": cell_pixels,
            # The rate denominator is the probe's own declared press, so probe rates are in the
            # probe tier and must never be compared with the mixed-tier rates of an incidental
            # grid. Recording it here is what stops that comparison being made by accident.
            "press_ms": press_ms,
            "idle_rate_per_100ms": idle_rate_per_100ms,
            "source_sessions": list(source_sessions),
            "diagnostics": analysis["diagnostics"],
        },
        "grid": {
            "schema_version": TRAVERSABILITY_SCHEMA,
            "cell_pixels": cell_pixels,
            "minimum_samples": minimum_samples,
            "cells": grid_cells,
        },
    }
    block["coverage"] = traversability_coverage(block)
    return block
