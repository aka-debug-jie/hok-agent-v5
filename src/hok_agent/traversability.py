"""Measured traversability: an offline grid of "did the commanded bearing actually move the hero".

Motivation, measured on the route B runs: the stall that blocks the consecutive stages is a
*directional* obstruction, not a lost signal and not a spent budget. At the cell where the v14
three-consecutive stage died, a north press produced 0.071 px per 100 ms against a baseline of
0.187-0.325 px per 100 ms, while the north-east press in the same cell produced 0.304. The hero
pressed north 52 times into a direction that was three to four times less effective than the
baseline.

The signal needs no reward and no label - only the recorded pair (position, applied bearing) ->
next position - so the data-source limit that closed the R learning route does not apply here. This
module turns that recorded pair into a frozen grid, and the Router consults it as a passability
mask in exactly the way it already consults the declared region filter.

Two deliberate properties:

- the estimate is normalised by the declared bounded press duration, because a short bounded press
  also yields a small displacement and would otherwise be misread as a wall;
- the mask fails open. A cell or bearing with too few samples is *passable*, so a partly covered
  grid can only remove a bearing that was measured to be ineffective, never invent one.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from hok_agent.mobile_testbed import MobileTestbedError

TRAVERSABILITY_SCHEMA = "hok-agent-traversability-grid-v1"
BEARING_STEPS = {
    "N": (-1.0, 0.0),
    "NE": (-0.7071067811865476, 0.7071067811865476),
    "E": (0.0, 1.0),
    "SE": (0.7071067811865476, 0.7071067811865476),
    "S": (1.0, 0.0),
    "SW": (0.7071067811865476, -0.7071067811865476),
    "W": (0.0, -1.0),
    "NW": (-0.7071067811865476, -0.7071067811865476),
}
MOVEMENT_ORDER = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
UNBOUNDED_MOVEMENT_MS = 890.0


def _cell_key(position: tuple[float, float], cell_pixels: float) -> str:
    return f"{int(position[0] // cell_pixels)}:{int(position[1] // cell_pixels)}"


def discover_route_b_runs(runs_root: Path, run_prefix: str = "route-b-batch-") -> tuple[str, ...]:
    """List candidate run directories. Discovery is for a human choosing a source list, never for
    a build: a glob makes the grid depend on whatever batches happen to exist, so a re-run after
    new batches would silently change a grid that a contract already froze."""
    if not runs_root.is_dir():
        return ()
    return tuple(
        sorted(
            name
            for name in os.listdir(runs_root)
            if name.startswith(run_prefix) and (runs_root / name).is_dir()
        )
    )


def build_traversability_grid(
    *,
    runs_root: Path,
    runs: Sequence[str],
    cell_pixels: float,
    minimum_samples: int,
) -> dict[str, object]:
    """Aggregate recorded transitions into a per-cell, per-bearing response rate.

    Only consecutive *localised* steps with an applied bearing are used. The rate is the
    displacement projected onto the commanded bearing, divided by the movement time that produced
    it: the declared bounded press duration when there was one, and the measured unbounded movement
    time otherwise. Rate is reported per 100 ms so the tiers are comparable.
    """
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    if not runs:
        raise MobileTestbedError("a traversability build needs an explicit, non-empty run list")
    for run in runs:
        if not (runs_root / run).is_dir():
            raise MobileTestbedError(f"traversability source run is missing: {run}")
        for episode in sorted(os.listdir(runs_root / run)):
            log = runs_root / run / episode / "steps.jsonl"
            if not log.is_file():
                continue
            steps = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for earlier, later in zip(steps, steps[1:], strict=False):
                if earlier["position"] is None or later["position"] is None:
                    continue
                bearing = earlier["applied_movement"]
                if bearing not in BEARING_STEPS:
                    continue
                press = earlier.get("approach_press_ms")
                movement_ms = float(press) if press else UNBOUNDED_MOVEMENT_MS
                if movement_ms <= 0:
                    continue
                step_y, step_x = BEARING_STEPS[bearing]
                delta_y = later["position"][0] - earlier["position"][0]
                delta_x = later["position"][1] - earlier["position"][1]
                projection = delta_y * step_y + delta_x * step_x
                key = _cell_key(
                    (float(earlier["position"][0]), float(earlier["position"][1])), cell_pixels
                )
                samples[key][bearing].append(projection / movement_ms * 100.0)
    cells: dict[str, object] = {}
    for key in sorted(samples):
        entry: dict[str, object] = {}
        for bearing in MOVEMENT_ORDER:
            values = samples[key].get(bearing)
            if not values:
                continue
            entry[bearing] = {
                "n": len(values),
                "rate": round(sum(values) / len(values), 4),
            }
        # Every observed cell is kept: the per-bearing sample minimum is applied by the mask,
        # so a single stray sample can never remove a bearing.
        if entry:
            cells[key] = entry
    return {
        "schema_version": TRAVERSABILITY_SCHEMA,
        "cell_pixels": cell_pixels,
        "minimum_samples": minimum_samples,
        "unbounded_movement_ms": UNBOUNDED_MOVEMENT_MS,
        "cells": cells,
    }


def make_traversability_block(
    *,
    runs_root: Path,
    runs: Sequence[str],
    cell_pixels: float,
    minimum_samples: int,
    minimum_rate_per_100ms: float,
    nominal_step_pixels: float,
) -> dict[str, object]:
    """Build a complete declared traversability block, including the source list it was built from.

    The block carries its own pinned source runs, so a later contract can regenerate it without
    depending on which batches happen to exist at the time.
    """
    grid = build_traversability_grid(
        runs_root=runs_root, runs=runs, cell_pixels=cell_pixels, minimum_samples=minimum_samples
    )
    block: dict[str, object] = {
        "mode": "measured_grid_mask",
        "minimum_rate_per_100ms": minimum_rate_per_100ms,
        "nominal_step_pixels": nominal_step_pixels,
        "source_runs": list(runs),
        "coverage": traversability_coverage({"grid": grid}),
        "grid": grid,
    }
    return validate_traversability(block)


def verify_traversability_grid(
    *,
    runs_root: Path,
    runs: Sequence[str],
    block: dict[str, object],
) -> dict[str, object]:
    """Rebuild the grid from the pinned runs and compare it with the frozen one.

    A mismatch is a failure, not a warning: it means the recorded grid can no longer be regenerated
    from its declared inputs, so the run and its grid have drifted apart.
    """
    grid = cast(dict[str, object], block["grid"])
    rebuilt = build_traversability_grid(
        runs_root=runs_root,
        runs=runs,
        cell_pixels=float(cast(float, grid["cell_pixels"])),
        minimum_samples=int(cast(int, grid["minimum_samples"])),
    )
    frozen_cells = cast(dict[str, object], grid["cells"])
    rebuilt_cells = cast(dict[str, object], rebuilt["cells"])
    differing = sorted(
        key
        for key in set(frozen_cells) | set(rebuilt_cells)
        if frozen_cells.get(key) != rebuilt_cells.get(key)
    )
    return {
        "matches_frozen_grid": not differing,
        "source_runs": list(runs),
        "frozen_cells": len(frozen_cells),
        "rebuilt_cells": len(rebuilt_cells),
        "differing_cells": differing[:20],
        "differing_cell_count": len(differing),
    }


def validate_traversability(block: object) -> dict[str, object]:
    """Validate the declared traversability block, raising on anything ambiguous."""
    if not isinstance(block, dict):
        raise MobileTestbedError("mobile navigation store traversability differs")
    grid = block.get("grid")
    if (
        block.get("mode") != "measured_grid_mask"
        or not isinstance(grid, dict)
        or grid.get("schema_version") != TRAVERSABILITY_SCHEMA
        or not isinstance(grid.get("cell_pixels"), (int, float))
        or float(cast(float, grid["cell_pixels"])) <= 0
        or not isinstance(grid.get("minimum_samples"), int)
        or int(cast(int, grid["minimum_samples"])) < 1
        or not isinstance(grid.get("cells"), dict)
        or not isinstance(block.get("minimum_rate_per_100ms"), (int, float))
        or float(cast(float, block["minimum_rate_per_100ms"])) <= 0
        or not isinstance(block.get("nominal_step_pixels"), (int, float))
        or float(cast(float, block["nominal_step_pixels"])) <= 0
    ):
        raise MobileTestbedError("mobile navigation store traversability differs")
    cells = cast(dict[str, object], grid["cells"])
    for entry in cells.values():
        if not isinstance(entry, dict) or not any(
            isinstance(bearing, str) and bearing in BEARING_STEPS for bearing in entry
        ):
            raise MobileTestbedError("mobile navigation store traversability differs")
        for bearing, value in cast(dict[str, object], entry).items():
            if (
                bearing not in BEARING_STEPS
                or not isinstance(value, dict)
                or not isinstance(cast(dict[str, object], value).get("n"), int)
                or int(cast(int, cast(dict[str, object], value)["n"])) < 1
                or not isinstance(cast(dict[str, object], value).get("rate"), (int, float))
            ):
                raise MobileTestbedError("mobile navigation store traversability differs")
    return block


def traversability_masked_direction(
    position: tuple[float, float],
    desired: str,
    block: dict[str, object],
) -> str:
    """Mask a bearing that the frozen grid measured to be ineffective in the current cell.

    The lookup is the cell the hero is standing in, because that is how the samples were
    attributed: "from here, this bearing produced this rate". A missing cell, a missing bearing or
    too few samples all count as passable, so the mask can only remove a bearing that was actually
    measured to be ineffective.
    """
    if desired not in BEARING_STEPS:
        return desired
    grid = cast(dict[str, object], block["grid"])
    cells = cast(dict[str, object], grid["cells"])
    cell_pixels = float(cast(float, grid["cell_pixels"]))
    minimum_samples = int(cast(int, grid["minimum_samples"]))
    minimum_rate = float(cast(float, block["minimum_rate_per_100ms"]))

    def blocked(bearing: str) -> bool:
        entry = cells.get(_cell_key(position, cell_pixels))
        if not isinstance(entry, dict):
            return False
        measured = cast(dict[str, object], entry).get(bearing)
        if not isinstance(measured, dict):
            return False
        if int(cast(int, cast(dict[str, object], measured)["n"])) < minimum_samples:
            return False
        return float(cast(float, cast(dict[str, object], measured)["rate"])) < minimum_rate

    if not blocked(desired):
        return desired
    index = MOVEMENT_ORDER.index(desired)
    for offset in (1, -1, 2, -2, 3, -3, 4):
        candidate = MOVEMENT_ORDER[(index + offset) % len(MOVEMENT_ORDER)]
        if not blocked(candidate):
            return candidate
    return desired


def tier_restricted_rates(
    runs_root: Path, runs: Sequence[str], cell_pixels: float
) -> dict[str, object]:
    """Per-cell, per-bearing rates restricted to the bounded presses the Router itself issues.

    Normalising by the declared press makes an unbounded sample comparable in units, but it does not
    make the tiers interchangeable: at cell 14:11 a bounded north press moves the hero at 0.060 px
    per 100 ms while a 2500 ms north press over the same city block moves it at 0.315. The long
    press works through a partial obstruction that the short one never gets past, so a mask that
    guards the Router has to be judged on the bounded tier alone. Restricting here is what makes the
    comparison matched: at 14:11 the mask removes north on 125 bounded samples while keeping
    north-east at 0.307 on 137 of them, so the contrast is same-cell, same-tier and near-equal-n.
    This is evidence from the recorded transitions rather than from a new session, and it is
    reported separately so the mixed-tier grid the mask reads is never mistaken for it.
    """
    every: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    bounded: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if not (runs_root / run).is_dir():
            raise MobileTestbedError(f"traversability source run is missing: {run}")
        for episode in sorted(os.listdir(runs_root / run)):
            log = runs_root / run / episode / "steps.jsonl"
            if not log.is_file():
                continue
            rows = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for earlier, later in zip(rows, rows[1:], strict=False):
                if earlier["position"] is None or later["position"] is None:
                    continue
                direction = earlier["applied_movement"]
                if direction not in BEARING_STEPS:
                    continue
                press = earlier.get("approach_press_ms")
                movement_ms = float(press) if press else UNBOUNDED_MOVEMENT_MS
                if movement_ms <= 0:
                    continue
                step_y, step_x = BEARING_STEPS[direction]
                projection = (
                    (later["position"][0] - earlier["position"][0]) * step_y
                    + (later["position"][1] - earlier["position"][1]) * step_x
                )
                key = _cell_key(
                    (float(earlier["position"][0]), float(earlier["position"][1])), cell_pixels
                )
                value = projection / movement_ms * 100.0
                every[key][direction].append(value)
                if press:
                    bounded[key][direction].append(value)
    cells: dict[str, object] = {}
    for key in sorted(set(every) | set(bounded)):
        entry: dict[str, object] = {}
        for direction in MOVEMENT_ORDER:
            all_values = every[key].get(direction, [])
            bounded_values = bounded[key].get(direction, [])
            if not all_values:
                continue
            entry[direction] = {
                "n": len(all_values),
                "rate": round(sum(all_values) / len(all_values), 4),
                "bounded_n": len(bounded_values),
                "bounded_rate": (
                    round(sum(bounded_values) / len(bounded_values), 4) if bounded_values else None
                ),
            }
        if entry:
            cells[key] = entry
    return {"cell_pixels": cell_pixels, "cells": cells}


def traversability_coverage(block: dict[str, object]) -> dict[str, object]:
    """Report how much of the grid is actually known, so partial coverage is never hidden."""
    grid = cast(dict[str, object], block["grid"])
    cells = cast(dict[str, object], grid["cells"])
    minimum_samples = int(cast(int, grid["minimum_samples"]))
    observations = 0
    effective = 0
    for entry in cells.values():
        for value in cast(dict[str, object], entry).values():
            observations += 1
            measured = cast(dict[str, object], value)
            if int(cast(int, measured["n"])) >= minimum_samples:
                effective += 1
    return {
        "cells": len(cells),
        "cell_bearing_observations": observations,
        "cell_bearing_above_minimum_samples": effective,
    }


def load_traversability(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MobileTestbedError("traversability grid is unavailable") from exc
    return validate_traversability(value)


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])
