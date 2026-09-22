"""Store-bound single-episode device navigation runner (Hierarchical Policy v0 stage L1).

The passed device navigation chain wrote ``observations.jsonl`` and ``summary.json`` only. This
module drives the same closed-loop rule through the single ``UnifiedTransitionStore`` so that one
complete episode's capture, action, reward, replay and recovery come from the same run:

    FrameBus FramePacket per observation (main/minimap/hud hashed derived views)
    -> deterministic Router (requested versus applied movement, masks, command)
    -> synchronous dispatch with an acknowledgement timestamp
    -> frozen settle interval
    -> next observation
    -> atomic transition append, terminal transition written before the episode ends
    -> reload verifier that re-checks step continuity and every committed frame bundle

Two boundaries are deliberate and must not be read as more than they are:

- This module adds no input transport. It drives the device only through the existing guarded
  ``mobile_testbed`` primitives (``_open_device_guard``, ``ScrcpyControlSession``,
  ``PersistentJoystick``, ``GuardWatchdog``), keeps the same serial, foreground-package, display,
  identity, layout and ROI gates, and releases the joystick on every exit path.
- No visual event is claimed on this episode. The only available RGB event engine (E1a health) is
  frozen with ``mobile_capture_allowed=false`` and ``reward_allowed=false``, and the transition
  event vocabulary has no navigation type. The event slot and its engine identity are carried and
  hashed; the arrival itself is carried by ``terminal_reason=NAVIGATION_GOAL_REACHED``.
- ``navigation_context`` is omitted because its published contract requires
  ``goal_source=simulator_config`` and ``simulation_time_ms=step_id*100``, which are simulator
  semantics. The declared goal and the observed positions are written to ``steps.jsonl`` beside
  the Store instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from hok_agent.frame_bus import FramePacket, RgbView
from hok_agent.mobile_testbed import (
    ACTIVE_PROBE_GUARD_INTERVAL_SECONDS,
    ACTIVE_PROBE_GUARD_STALENESS_MS,
    ACTIVE_PROBE_TOUCH_SETTLE_SECONDS,
    GOAL_NAVIGATION_LOOP_SLEEP_SECONDS,
    DeviceGuard,
    GuardWatchdog,
    MobileTestbedError,
    ObservationROIs,
    PersistentJoystick,
    ScrcpyControlSession,
    TouchOperation,
    _death_replay_visible,
    _goal_navigation_contract,
    _goal_navigation_direction,
    _goal_navigation_tracked_cue,
    _GoalNavigationTemplateState,
    _new_large_output,
    _observation_roi_frame,
    _open_device_guard,
    load_layout,
    load_observation_rois,
)
from hok_agent.transition_store import (
    EpisodeEndKind,
    ExecutedActionRecord,
    HierarchicalTransitionRecord,
    MovementAction,
    PolicyProposalRecord,
    ProposalBundleRecord,
    ReplayRecord,
    RewardRecord,
    TerminalReason,
    UnifiedTransitionStore,
    validate_transition,
)
from hok_agent.traversability import (
    plan_grid_detour,
    traversability_masked_direction,
    validate_detour,
    validate_traversability,
)

MOBILE_NAVIGATION_STORE_SCHEMA = "hok-agent-mobile-navigation-store-session-v1"
MOBILE_NAVIGATION_STORE_BATCH_SCHEMA = "hok-agent-mobile-navigation-store-batch-v1"
MOBILE_NAVIGATION_PLACEMENT_ROUTE_SCHEMA = "hok-agent-mobile-navigation-placement-route-v1"
MOBILE_NAVIGATION_STORE_CONTRACT_SCHEMA = "movement-goal-navigation-store-contract-v1"
MOBILE_NAVIGATION_STORE_REQUIRED_FIELDS = (
    "policy_bundle_version",
    "event_engine_version",
    "reward_version",
    "replay_source",
    "episode_prefix",
    "maximum_steps",
    "settle_ms",
)
STORE_VIEW_NAMES = ("main", "minimap", "hud", "equipment")
JOYSTICK_TO_STORE_DIRECTION = {
    "wait": "STOP",
    "north": "N",
    "north_east": "NE",
    "east": "E",
    "south_east": "SE",
    "south": "S",
    "south_west": "SW",
    "west": "W",
    "north_west": "NW",
}
STORE_TO_JOYSTICK_DIRECTION = {
    value: key for key, value in JOYSTICK_TO_STORE_DIRECTION.items()
}


def _store_contract(contract: dict[str, object], contract_sha: str) -> dict[str, object]:
    block = contract.get("store")
    if (
        not isinstance(block, dict)
        or any(key not in block for key in MOBILE_NAVIGATION_STORE_REQUIRED_FIELDS)
        or block["replay_source"] not in {"controller", "online"}
        or not isinstance(block["maximum_steps"], int)
        or int(block["maximum_steps"]) <= 0
        or not isinstance(block["settle_ms"], int)
        or int(block["settle_ms"]) < 0
        or any(
            not isinstance(block[key], str) or not cast(str, block[key])
            for key in (
                "policy_bundle_version",
                "event_engine_version",
                "reward_version",
                "episode_prefix",
            )
        )
        or block.get("schema_version") != MOBILE_NAVIGATION_STORE_CONTRACT_SCHEMA
    ):
        raise MobileTestbedError("mobile navigation store contract differs")
    planner_raw = contract.get("planner")
    planner = cast(dict[str, object] | None, planner_raw)
    if planner_raw is not None and (
        not isinstance(planner_raw, dict)
        or planner_raw.get("mode") != "path_progress_with_feasibility"
        or any(
            not isinstance(planner_raw.get(key), (int, float))
            or float(cast(float, planner_raw[key])) <= 0
            for key in (
                "nominal_step_pixels",
                "margin_pixels",
                "lookahead_pixels",
                "stall_bias",
            )
        )
        or contract.get("region_filter") is not None
        or (
            isinstance(contract.get("progress_guard"), dict)
            and cast(dict[str, object], contract["progress_guard"]).get("escape_offsets_sectors")
            is not None
        )
    ):
        raise MobileTestbedError("mobile navigation store planner differs")
    guard_raw = contract.get("progress_guard")
    if guard_raw is not None and not isinstance(guard_raw, dict):
        raise MobileTestbedError("mobile navigation store progress guard differs")
    progress_guard = cast(dict[str, object] | None, guard_raw)
    if progress_guard is not None and progress_guard.get("mode") == "stall_trigger_only":
        if any(
            not isinstance(progress_guard.get(key), (int, float))
            or float(cast(float, progress_guard[key])) <= 0
            for key in (
                "minimum_improvement_pixels",
                "confirmation_steps",
                "stall_steps",
                "maximum_events_per_episode",
            )
        ):
            raise MobileTestbedError("mobile navigation store progress guard differs")
    elif progress_guard is not None:
        offsets = progress_guard.get("escape_offsets_sectors")
        if (
            progress_guard.get("mode") != "bounded_bearing_escape"
            or not isinstance(offsets, list)
            or not cast(list[object], offsets)
            or any(
                not isinstance(item, int) or abs(int(item)) > 3
                for item in cast(list[object], offsets)
            )
            or any(
                not isinstance(progress_guard.get(key), (int, float))
                or float(cast(float, progress_guard[key])) <= 0
                for key in (
                    "minimum_improvement_pixels",
                    "confirmation_steps",
                    "escape_hold_steps",
                    "maximum_events_per_episode",
                )
            )
        ):
            raise MobileTestbedError("mobile navigation store progress guard differs")
    filter_raw = contract.get("region_filter")
    if filter_raw is not None and not isinstance(filter_raw, dict):
        raise MobileTestbedError("mobile navigation store region filter differs")
    region_filter = cast(dict[str, object] | None, filter_raw)
    if region_filter is not None and (
        region_filter.get("mode") != "feasible_direction_within_region"
        or any(
            not isinstance(region_filter.get(key), (int, float))
            or float(cast(float, region_filter[key])) <= 0
            for key in ("nominal_step_pixels", "margin_pixels")
        )
    ):
        raise MobileTestbedError("mobile navigation store region filter differs")
    recovery_raw = contract.get("unknown_recovery")
    if recovery_raw is not None and not isinstance(recovery_raw, dict):
        raise MobileTestbedError("mobile navigation store unknown recovery differs")
    unknown_recovery = cast(dict[str, object] | None, recovery_raw)
    if recovery_raw is not None:
        recovery_block = cast(dict[str, object], recovery_raw)
        trigger = recovery_block.get("trigger_after_missing_frames")
        maximum = recovery_block.get("maximum_recovery_steps")
        hold = recovery_block.get("hold_ms")
        if (
            recovery_block.get("mode") not in {"bounded_retrace", "bounded_retrace_or_waypoint"}
            or not isinstance(trigger, int)
            or not isinstance(maximum, int)
            or not isinstance(hold, int)
            or int(trigger) <= 0
            or int(maximum) <= 0
            or int(hold) <= 0
            or any(
                not isinstance(recovery_block.get(key), (int, float))
                or float(cast(float, recovery_block[key])) <= 0
                for key in ("nominal_step_pixels",)
            )
            or (
                recovery_block.get("waypoint_bearing_within_distance_pixels") is not None
                and (
                    not isinstance(
                        recovery_block["waypoint_bearing_within_distance_pixels"], (int, float)
                    )
                    or float(
                        cast(float, recovery_block["waypoint_bearing_within_distance_pixels"])
                    )
                    <= 0
                )
            )
        ):
            raise MobileTestbedError("mobile navigation store unknown recovery differs")
        # The recovery must fit inside the declared localisation guard, so a failed retreat can
        # never extend the episode past the gap the guard already bounds.
        declared_gap = block.get("maximum_localization_gap_frames")
        if (
            not isinstance(declared_gap, int)
            or int(trigger) + int(maximum) > int(declared_gap)
        ):
            raise MobileTestbedError("mobile navigation store unknown recovery differs")
    approach_raw = contract.get("final_approach")
    final_approach = cast(dict[str, object] | None, approach_raw)
    if approach_raw is not None:
        if (
            not isinstance(approach_raw, dict)
            or approach_raw.get("mode") != "declared_deceleration"
        ):
            raise MobileTestbedError("mobile navigation store final approach differs")
        tiers = approach_raw.get("tiers")
        default_hold = approach_raw.get("default_hold_ms")
        if (
            not isinstance(tiers, list)
            or not cast(list[object], tiers)
            or not isinstance(default_hold, int)
            or int(default_hold) <= 0
        ):
            raise MobileTestbedError("mobile navigation store final approach differs")
        ordered: list[tuple[float, int]] = []
        for item in cast(list[object], tiers):
            if not isinstance(item, dict):
                raise MobileTestbedError("mobile navigation store final approach differs")
            entry = cast(dict[str, object], item)
            distance = entry.get("maximum_distance_pixels")
            hold = entry.get("hold_ms")
            if (
                not isinstance(distance, (int, float))
                or not isinstance(hold, int)
                or float(distance) <= 0
                or hold <= 0
            ):
                raise MobileTestbedError("mobile navigation store final approach differs")
            ordered.append((float(distance), int(hold)))
        ordered.sort()
        if any(later[1] < earlier[1] for earlier, later in zip(ordered, ordered[1:], strict=False)):
            raise MobileTestbedError("mobile navigation store final approach differs")
        exit_margin = approach_raw.get("direct_bearing_exit_margin_pixels")
        if exit_margin is not None and (
            not isinstance(exit_margin, (int, float)) or float(exit_margin) < 0
        ):
            raise MobileTestbedError("mobile navigation store final approach differs")
        commitment = approach_raw.get("commitment_steps")
        if commitment is not None and (not isinstance(commitment, int) or int(commitment) < 1):
            raise MobileTestbedError("mobile navigation store final approach differs")
        stall_commitment = (
            cast(dict[str, object], planner_raw).get("stall_commitment_steps")
            if planner_raw is not None
            else None
        )
        if stall_commitment is not None and (
            not isinstance(stall_commitment, int) or int(stall_commitment) < 1
        ):
            raise MobileTestbedError("mobile navigation store planner stall commitment differs")
        direct_bearing = approach_raw.get("direct_bearing")
        if direct_bearing is not None and not isinstance(direct_bearing, bool):
            raise MobileTestbedError("mobile navigation store final approach differs")
        press_distance = approach_raw.get("press_release_maximum_distance_pixels")
        if press_distance is not None and (
            not isinstance(press_distance, (int, float))
            or float(press_distance) <= 0
            or float(press_distance) > ordered[-1][0]
        ):
            # A press-release band outside the declared deceleration tiers would shorten a press
            # that no tier describes, so it is rejected rather than silently introduced.
            raise MobileTestbedError("mobile navigation store final approach differs")
    if contract.get("traversability") is not None:
        validate_traversability(contract["traversability"])
    if contract.get("detour") is not None:
        # The detour searches the frozen grid, so it has no meaning without one. A contract that
        # declares a detour and no grid is refused rather than silently planning blind.
        if contract.get("traversability") is None:
            raise MobileTestbedError("detour requires a traversability grid")
        validate_detour(contract["detour"])
    if contract.get("masked_persistence") is not None:
        # This rule holds a bearing the mask removed, so it also has no meaning without the mask.
        if contract.get("traversability") is None:
            raise MobileTestbedError("masked persistence requires a traversability grid")
        _validate_masked_persistence(contract["masked_persistence"])
    advance_raw = contract.get("no_advance_guard")
    no_advance_guard = cast(dict[str, object] | None, advance_raw)
    if advance_raw is not None and (
        not isinstance(advance_raw, dict)
        or advance_raw.get("mode") != "flat_localised_position"
        or not isinstance(advance_raw.get("window_steps"), int)
        or not isinstance(advance_raw.get("maximum_travel_pixels"), (int, float))
        or int(advance_raw["window_steps"]) <= 1
        or float(cast(float, advance_raw["maximum_travel_pixels"])) <= 0
    ):
        raise MobileTestbedError("mobile navigation store no-advance guard differs")
    backlog_raw = block.get("backlog_free_maximum_pointer_messages")
    if backlog_raw is not None and (not isinstance(backlog_raw, int) or int(backlog_raw) < 1):
        raise MobileTestbedError("mobile navigation store contract differs")
    resolved = dict(block)
    resolved["progress_guard"] = progress_guard
    resolved["planner"] = planner
    resolved["final_approach"] = final_approach
    resolved["region_filter"] = region_filter
    resolved["unknown_recovery"] = unknown_recovery
    resolved["no_advance_guard"] = no_advance_guard
    resolved["traversability"] = cast(
        dict[str, object] | None, contract.get("traversability")
    )
    resolved["policy_bundle_sha256"] = contract_sha
    resolved["event_engine_sha256"] = hashlib.sha256(
        json.dumps(block, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return resolved


def _episode_id(prefix: str, contract_sha: str) -> str:
    return f"{prefix}-{contract_sha[:12]}"


def _write_bundle(directory: Path, basename: str, arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != set(STORE_VIEW_NAMES):
        raise MobileTestbedError("mobile navigation store views differ")
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / basename
    if bundle.exists():
        try:
            with np.load(bundle, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(
                    not np.array_equal(saved[name], value) for name, value in arrays.items()
                ):
                    raise ValueError("existing frame bundle content differs")
        except (OSError, ValueError) as exc:
            raise MobileTestbedError(f"invalid existing frame bundle: {basename}") from exc
        return
    temporary = bundle.with_name(f".{bundle.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            main=arrays["main"],
            minimap=arrays["minimap"],
            hud=arrays["hud"],
            equipment=arrays["equipment"],
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, bundle)


def _packet(
    frame: np.ndarray,
    rois: object,
    *,
    directory: Path,
    episode_id: str,
    step_id: int,
    capture_start_ns: int,
    capture_end_ns: int,
) -> FramePacket:
    arrays = {
        "main": np.ascontiguousarray(_observation_roi_frame(frame, rois.main_view)),  # type: ignore[attr-defined]
        "minimap": np.ascontiguousarray(_observation_roi_frame(frame, rois.minimap)),  # type: ignore[attr-defined]
        "hud": np.ascontiguousarray(_observation_roi_frame(frame, rois.hud)),  # type: ignore[attr-defined]
        "equipment": np.ascontiguousarray(
            _observation_roi_frame(frame, rois.recommended_equipment)  # type: ignore[attr-defined]
        ),
    }
    basename = f"{episode_id}-{step_id:04d}.npz"
    _write_bundle(directory, basename, arrays)
    views = tuple(
        RgbView(
            cast(Literal["main", "minimap", "hud", "equipment"], name),
            value.tobytes(),
            cast(tuple[int, int, int], value.shape),
        )
        for name, value in arrays.items()
    )
    return FramePacket(
        observation_id=f"{episode_id}-obs-{step_id:04d}",
        capture_start_ns=capture_start_ns,
        capture_end_ns=capture_end_ns,
        capture_source_class="self_built_mobile_test_app",
        frame_bundle_ref=basename,
        views=views,
    )


def _proposal(observation_id: str, value: str, start_ns: int, version: str) -> PolicyProposalRecord:
    return {
        "observation_id": observation_id,
        "applied_observation_id": observation_id,
        "carried_forward": False,
        "value": value,
        "confidence": 1.0,
        "decision_start_ns": start_ns,
        "decision_end_ns": start_ns + 1_000_000,
        "valid_until_ns": start_ns + 5_000_000_000,
        "policy_bundle_version": version,
    }


def _movement_command(previous: str, current: str) -> str:
    if current == "STOP":
        return "UP" if previous != "STOP" else "NOOP"
    if previous == "STOP":
        return "DOWN"
    return "KEEP" if previous == current else "MOVE"


def _route(
    requested: str,
    *,
    known: bool,
    death: bool,
    outside_region: bool,
    recovery: tuple[str, str] | None = None,
) -> tuple[str, str, str]:
    """Deterministic Router: masks the geometry proposal and keeps requested versus applied.

    Both the request and the result stay in the joystick vocabulary; the store vocabulary is
    applied at the transition boundary only. ``recovery`` is a declared bounded
    ``(bearing, reason)`` pair that the Router may apply instead of waiting while the marker is
    unknown; it is still the Router that decides, so the geometry proposal stays masked as before.
    """
    if death:
        return "wait", "deterministic_router", "death_or_ended_screen"
    if outside_region:
        return "wait", "deterministic_router", "outside_free_movement_region"
    if not known:
        if recovery is not None:
            bearing, reason = recovery
            return bearing, "deterministic_router", reason
        return "wait", "deterministic_router", "unknown_position"
    return requested, "geometry_rule", "geometry_rule"


def _transition(
    *,
    store_contract: dict[str, object],
    episode_id: str,
    step_id: int,
    observation: FramePacket,
    next_observation: FramePacket,
    requested: str,
    applied: str,
    owner: str,
    reason: str,
    movement_command: str,
    dispatch_start_ns: int,
    dispatch_ack_ns: int,
    settle_end_ns: int,
    final_status: str,
    retry_count: int,
    done: bool,
    terminal_reason: str,
    end_kind: str,
) -> HierarchicalTransitionRecord:
    version = cast(str, store_contract["policy_bundle_version"])
    decision_start = observation.capture_end_ns + 1_000_000
    proposals: ProposalBundleRecord = {
        "macro": _proposal(observation.observation_id, "HOLD", decision_start, version),
        "movement": _proposal(observation.observation_id, applied, decision_start, version),
        "combat": _proposal(observation.observation_id, "WAIT", decision_start, version),
    }
    executed: ExecutedActionRecord = {
        "requested_movement": cast(MovementAction, requested),
        "applied_movement": cast(MovementAction, applied),
        "requested_combat": "WAIT",
        "applied_combat": "WAIT",
        "movement_command": cast(
            Literal["DOWN", "MOVE", "UP", "KEEP", "NOOP"], movement_command
        ),
        "combat_command": "NOOP",
        "dispatch_start_ns": dispatch_start_ns,
        "dispatch_ack_ns": dispatch_ack_ns,
        "first_attempt_status": "acknowledged" if retry_count == 0 else "failed",
        "retry_status": "not_attempted" if retry_count == 0 else "acknowledged",
        "retry_count": retry_count,
        "final_status": cast(
            Literal["acknowledged", "rejected", "failed", "noop"], final_status
        ),
    }
    reward: RewardRecord = {
        "reward_version": cast(str, store_contract["reward_version"]),
        "components": {
            "terminal": 1.0 if done else 0.0,
            "death": 0.0,
            "self_hp_delta": 0.0,
            "tower_damage": 0.0,
        },
        "total": 1.0 if done else 0.0,
        "event_ids": [],
    }
    replay: ReplayRecord = {
        "source": cast(
            Literal["demo", "sim", "controller", "online", "offline_video"],
            store_contract["replay_source"],
        ),
        "failure_tags": [] if done else ["in_progress"],
        "priority": 1.0 if done else 0.5,
    }
    return {
        "schema_version": "hok-agent-hierarchical-transition-v0",
        "episode_id": episode_id,
        "step_id": step_id,
        "policy_bundle_version": version,
        "policy_bundle_sha256": cast(str, store_contract["policy_bundle_sha256"]),
        "event_engine_version": cast(str, store_contract["event_engine_version"]),
        "event_engine_sha256": cast(str, store_contract["event_engine_sha256"]),
        "observation": observation.to_record(),
        "proposals": proposals,
        "executed_action": executed,
        "settle_end_ns": settle_end_ns,
        "next_observation": next_observation.to_record(),
        "events": [],
        "reward": reward,
        "done": done,
        "terminal_reason": cast(TerminalReason, terminal_reason),
        "episode_end_kind": cast(EpisodeEndKind, end_kind),
        "causal_order_valid": True,
        "training_eligible": False,
        "ineligibility_reasons": [
            "single_episode_runtime_binding_only",
            "no_visual_event_claimed",
            f"router:{owner}:{reason}",
        ],
        "replay": replay,
    }


_DIRECTION_STEPS = {
    "north": (-1.0, 0.0),
    "north_east": (-0.7071067811865476, 0.7071067811865476),
    "east": (0.0, 1.0),
    "south_east": (0.7071067811865476, 0.7071067811865476),
    "south": (1.0, 0.0),
    "south_west": (0.7071067811865476, -0.7071067811865476),
    "west": (0.0, -1.0),
    "north_west": (-0.7071067811865476, -0.7071067811865476),
}

_MOVEMENT_ORDER = (
    "north", "north_east", "east", "south_east",
    "south", "south_west", "west", "north_west",
)


def _rotate_direction(direction: str, sectors: int) -> str:
    """Rotate a joystick direction by whole 45-degree sectors, wrapping around."""
    if direction not in _MOVEMENT_ORDER:
        raise MobileTestbedError("cannot rotate a non-direction")
    index = _MOVEMENT_ORDER.index(direction)
    return _MOVEMENT_ORDER[(index + sectors) % len(_MOVEMENT_ORDER)]


def _path_reference(
    position: tuple[float, float],
    route: list[tuple[float, float]],
    waypoint_index: int,
    start: tuple[float, float],
    lookahead_pixels: float,
) -> tuple[float, float]:
    """The point on the current leg the planner aims at: the path projection plus a lookahead."""
    index = min(max(waypoint_index, 0), len(route) - 1)
    leg_start = start if index == 0 else route[index - 1]
    leg_end = route[index]
    span_y = leg_end[0] - leg_start[0]
    span_x = leg_end[1] - leg_start[1]
    length_squared = span_y * span_y + span_x * span_x
    if length_squared <= 1e-9:
        return leg_end
    along = (
        (position[0] - leg_start[0]) * span_y + (position[1] - leg_start[1]) * span_x
    ) / length_squared
    along = min(max(along, 0.0), 1.0)
    length = float(np.sqrt(length_squared))
    lead = min(1.0, along + lookahead_pixels / length)
    return (leg_start[0] + span_y * lead, leg_start[1] + span_x * lead)


def _planner_direction(
    position: tuple[float, float],
    route: list[tuple[float, float]],
    waypoint_index: int,
    start: tuple[float, float],
    region: dict[str, object],
    *,
    nominal_step_pixels: float,
    margin_pixels: float,
    lookahead_pixels: float,
    stall_active: bool,
    stall_bias: float,
) -> tuple[str, tuple[float, float]]:
    """One arbiter: maximise progress along the path subject to staying inside the region.

    Feasibility and progress live in the same objective, so no layer can override another. While the
    progress guard reports a stall the objective additionally rewards a lateral component, which is
    a sidestep along the objective rather than a rotation of the bearing.
    """
    reference = _path_reference(position, route, waypoint_index, start, lookahead_pixels)
    aim_y = reference[0] - position[0]
    aim_x = reference[1] - position[1]
    aim_norm = float(np.hypot(aim_y, aim_x))
    if aim_norm <= 1e-9:
        return "wait", reference
    aim_y /= aim_norm
    aim_x /= aim_norm
    minimum_y = float(cast(float, region["minimum_y"])) + margin_pixels
    maximum_y = float(cast(float, region["maximum_y"])) - margin_pixels
    minimum_x = float(cast(float, region["minimum_x"])) + margin_pixels
    maximum_x = float(cast(float, region["maximum_x"])) - margin_pixels
    scored: list[tuple[bool, float, str]] = []
    for name, (step_y, step_x) in _DIRECTION_STEPS.items():
        target_y = position[0] + step_y * nominal_step_pixels
        target_x = position[1] + step_x * nominal_step_pixels
        feasible = (
            minimum_y <= target_y <= maximum_y and minimum_x <= target_x <= maximum_x
        )
        progress = step_y * aim_y + step_x * aim_x
        if stall_active:
            lateral = abs(step_y * aim_x - step_x * aim_y)
            progress = (1.0 - stall_bias) * progress + stall_bias * lateral
        scored.append((feasible, progress, name))
    feasible_steps = [item for item in scored if item[0]]
    if feasible_steps:
        return max(feasible_steps, key=lambda item: item[1])[2], reference
    return max(scored, key=lambda item: item[1])[2], reference


def _mask_joystick_bearing(
    position: tuple[float, float],
    joystick_bearing: str,
    block: dict[str, object],
) -> str:
    """Apply the measured grid to a joystick-vocabulary bearing and return a joystick bearing.

    The grid is keyed by the recorded store vocabulary, so the caller must not hand a joystick name
    straight to it: the two vocabularies overlap on nothing, and a direct call silently masks
    nothing at all. This is the single conversion point.
    """
    stored = JOYSTICK_TO_STORE_DIRECTION.get(joystick_bearing)
    if stored is None:
        return joystick_bearing
    masked = traversability_masked_direction(position, stored, block)
    return STORE_TO_JOYSTICK_DIRECTION.get(masked, joystick_bearing)


def _commit_approach(
    requested: str,
    committed: str | None,
    committed_steps: int,
    commitment: int,
    in_band: bool,
) -> tuple[str, str | None, int, bool]:
    """Hold a chosen approach bearing for the declared commitment, or choose a new one.

    Returns the bearing to request, the new commitment state, and whether this step was a held one.
    Outside the declared band, or with a commitment of one, everything is as it was.
    """
    if commitment <= 1 or not in_band:
        return requested, None, 0, False
    if committed is not None and committed_steps < commitment:
        return committed, committed, committed_steps + 1, True
    return requested, requested, 1, False


def _death_confirmed(
    banner_streak: int,
    recent_positions: Sequence[tuple[float, float] | None],
    confirmation_steps: int,
    maximum_travel_pixels: float,
) -> bool:
    """Confirm a death from a sustained banner and a hero that is not moving.

    The banner colour test is a raw measurement of "something red and white is in the box". It is
    true on red-brown terrain, on in-match announcements and on the death-replay prompt alike, so it
    cannot be the stop on its own: the 2026-09-22 rebind stopped an episode while the hero was
    walking at about three pixels per step. A death is a state, though, and a dead hero does not
    move, so the banner must hold for the declared number of consecutive observations and the
    localised position must stay within the declared travel over the declared window. A lost marker
    counts as no travel, because a missing cue is itself consistent with death and the alternative
    would be to require localisation during the one state that removes it.
    """
    if banner_streak < confirmation_steps:
        return False
    fixed = [item for item in recent_positions if item is not None]
    if len(fixed) < 2:
        return True
    travel = math.hypot(fixed[-1][0] - fixed[0][0], fixed[-1][1] - fixed[0][1])
    return travel <= maximum_travel_pixels


def _no_progress_stall(
    window: Sequence[float], steps: int, maximum_progress_pixels: float
) -> bool:
    """Whether the goal distance improved by no more than the declared bound over ``steps``.

    A stall in this route is an absence of progress, not an absence of motion: the recorded failure
    oscillates inside a couple of pixels, so a flat-position rule never engages, which is how the
    v16 commitment and the detour's first device attempt both failed. Measuring the improvement in
    goal distance instead
    """
    if len(window) < steps or steps < 1:
        return False
    recent = list(window[-steps:])
    return recent[0] - min(recent) <= maximum_progress_pixels


def _goal_bearing(position: tuple[float, float], target: tuple[float, float]) -> str:
    """The joystick-vocabulary bearing from a position toward a target.

    The mask is asked about the planner's proposal, and that proposal changes for the planner's own
    reasons, so whether the mask fired this step is not the same question as whether the
    here". This derives the latter directly from the position and the target, using the same
    eight-sector rounding the recovery path already uses.
    """
    angle = math.atan2(target[1] - position[1], -(target[0] - position[0]))
    return _MOVEMENT_ORDER[round(angle / (math.pi / 4)) % len(_MOVEMENT_ORDER)]


def _validate_masked_persistence(block: object) -> dict[str, object]:
    """Validate a declared masked-bearing persistence block, raising on anything ambiguous.

    This rule deliberately keeps a bearing the frozen grid removed, so every number that bounds it
    has to be declared: how long it may be held, how many times it may be tried in an episode, what
    counts as the stall that activates it, and how much progress confirms it. A mistyped field is a
    load failure rather than a default.
    """
    if (
        not isinstance(block, dict)
        or block.get("schema_version") != "hok-agent-declared-masked-persistence-v1"
        or block.get("mode") != "hold-the-masked-goal-bearing"
    ):
        raise MobileTestbedError("mobile navigation store masked persistence differs")
    for key in ("maximum_steps", "maximum_activations_per_episode", "trigger_stall_steps"):
        if not isinstance(block.get(key), int) or int(cast(int, block[key])) < 1:
            raise MobileTestbedError("mobile navigation store masked persistence differs")
    for key in ("trigger_stall_progress_pixels", "confirmation_minimum_progress_pixels"):
        if not isinstance(block.get(key), (int, float)) or float(cast(float, block[key])) <= 0.0:
            raise MobileTestbedError("mobile navigation store masked persistence differs")
    bearings = block.get("persistence_bearings")
    if (
        not isinstance(bearings, list)
        or not bearings
        or not all(isinstance(item, str) and item in _MOVEMENT_ORDER for item in bearings)
        or len(set(bearings)) != len(bearings)
    ):
        raise MobileTestbedError("mobile navigation store masked persistence differs")
    return block


def _no_advance_detected(
    previous_anchor: tuple[float, float] | None,
    previous_steps: int,
    position: tuple[float, float] | None,
    advanced: bool,
    window_steps: int,
    maximum_travel_pixels: float,
) -> tuple[tuple[float, float] | None, int, bool]:
    """Track a run of applied-bearing steps whose localised position never travels.

    Returns the updated anchor, the updated run length and whether the declared window was
    exceeded. A step that is blind, or that applies no bearing, resets the run, so a lost marker, a
    deliberate release or a fine final-approach pulse can never be read as a dead screen.
    """
    if not advanced or position is None:
        return None, 0, False
    if previous_anchor is None:
        return position, 1, False
    travel = math.hypot(position[0] - previous_anchor[0], position[1] - previous_anchor[1])
    if travel > maximum_travel_pixels:
        return position, 1, False
    steps = previous_steps + 1
    return previous_anchor, steps, steps >= window_steps


def _approach_hold_ms(
    distance: float | None, tiers: list[tuple[float, int]], default_hold_ms: int
) -> int:
    """Declared deceleration: the tier whose distance bound covers the remaining distance."""
    if distance is None:
        return default_hold_ms
    for maximum_distance, hold_ms in sorted(tiers, key=lambda item: item[0]):
        if distance <= maximum_distance:
            return hold_ms
    return default_hold_ms


def _region_safe_direction(
    position: tuple[float, float],
    desired: str,
    region: dict[str, object],
    nominal_step_pixels: float,
    margin_pixels: float,
) -> str:
    """Mask any applied direction whose nominal next step would leave the free-movement region.

    The nearest safe bearing is used when the desired one is unsafe, which turns a wall into a
    follow-along behaviour. If no single step can return inside, the filter heads for the region
    centre rather than picking arbitrarily.
    """
    if desired not in _DIRECTION_STEPS:
        return desired
    minimum_y = float(cast(float, region["minimum_y"])) + margin_pixels
    maximum_y = float(cast(float, region["maximum_y"])) - margin_pixels
    minimum_x = float(cast(float, region["minimum_x"])) + margin_pixels
    maximum_x = float(cast(float, region["maximum_x"])) - margin_pixels

    def predicted(name: str) -> tuple[float, float]:
        step_y, step_x = _DIRECTION_STEPS[name]
        return (
            position[0] + step_y * nominal_step_pixels,
            position[1] + step_x * nominal_step_pixels,
        )

    def inside(point: tuple[float, float]) -> bool:
        return minimum_y <= point[0] <= maximum_y and minimum_x <= point[1] <= maximum_x

    if inside(predicted(desired)):
        return desired
    index = _MOVEMENT_ORDER.index(desired)
    for offset in (1, -1, 2, -2, 3, -3, 4):
        candidate = _MOVEMENT_ORDER[(index + offset) % len(_MOVEMENT_ORDER)]
        if inside(predicted(candidate)):
            return candidate
    centre_y = (
        float(cast(float, region["minimum_y"])) + float(cast(float, region["maximum_y"]))
    ) / 2
    centre_x = (
        float(cast(float, region["minimum_x"])) + float(cast(float, region["maximum_x"]))
    ) / 2
    return min(
        _MOVEMENT_ORDER,
        key=lambda name: abs(predicted(name)[0] - centre_y) + abs(predicted(name)[1] - centre_x),
    )


_RETRACE_MINIMUM_DISPLACEMENT = 0.5


def _retrace_direction(
    last_known: tuple[float, float],
    previous_known: tuple[float, float],
    region: dict[str, object],
    nominal_step_pixels: float,
    retreat_steps: int = 0,
) -> str | None:
    """The reversed, quantized bearing of the last known displacement, or None when it is unsafe.

    Retracing walks back over ground the hero has just occupied, so it is the cheapest bounded way
    out of a detection blind area while the marker is unknown. The bearing is the exact reversal of
    the last known displacement, quantized to the eight-way vocabulary. The step is predicted from
    the last known position at the depth already retreated, and the recovery declines to act rather
    than guess when the predicted point would leave the declared region, or when the last two known
    positions imply no usable displacement. The check is against the declared region itself, which
    is the same invariant ``outside_region`` enforces, so a retreat can never trip that guard.
    """
    delta_y = previous_known[0] - last_known[0]
    delta_x = previous_known[1] - last_known[1]
    if (
        abs(delta_y) < _RETRACE_MINIMUM_DISPLACEMENT
        and abs(delta_x) < _RETRACE_MINIMUM_DISPLACEMENT
    ):
        return None
    angle = math.atan2(delta_x, -delta_y)
    sector = round(angle / (math.pi / 4)) % 8
    bearing = _MOVEMENT_ORDER[sector]
    step_y, step_x = _DIRECTION_STEPS[bearing]
    depth = float(retreat_steps + 1) * nominal_step_pixels
    predicted_y = last_known[0] + step_y * depth
    predicted_x = last_known[1] + step_x * depth
    if not (
        float(cast(float, region["minimum_y"]))
        <= predicted_y
        <= float(cast(float, region["maximum_y"]))
        and float(cast(float, region["minimum_x"]))
        <= predicted_x
        <= float(cast(float, region["maximum_x"]))
    ):
        return None
    return bearing


def _toward_target_direction(
    last_known: tuple[float, float],
    target: tuple[float, float],
    region: dict[str, object],
    nominal_step_pixels: float,
    retreat_steps: int = 0,
) -> str | None:
    """The quantized bearing toward the current waypoint, region-checked at the given depth.

    This is the fallback the declared recovery uses when there is nothing to retrace, which the
    offline replay shows is the common case: a hero that lost its marker while holding its final
    approach has no usable displacement to reverse. The bearing is a plain re-aim at the declared
    waypoint, so it never invents a destination, and it is declined when the predicted step would
    leave the declared region, exactly as the retrace is.
    """
    bearing = _goal_navigation_direction(last_known, target, None, 0)
    if bearing == "wait":
        return None
    step_y, step_x = _DIRECTION_STEPS[bearing]
    depth = float(retreat_steps + 1) * nominal_step_pixels
    predicted_y = last_known[0] + step_y * depth
    predicted_x = last_known[1] + step_x * depth
    if not (
        float(cast(float, region["minimum_y"]))
        <= predicted_y
        <= float(cast(float, region["maximum_y"]))
        and float(cast(float, region["minimum_x"]))
        <= predicted_x
        <= float(cast(float, region["maximum_x"]))
    ):
        return None
    return bearing


def _unknown_recovery_step(
    *,
    last_known: tuple[float, float] | None,
    previous_known: tuple[float, float] | None,
    target: tuple[float, float] | None,
    missing_streak: int,
    recovery_steps: int,
    config: dict[str, object] | None,
    region: dict[str, object],
) -> tuple[tuple[str, str] | None, bool]:
    """Decide one unknown step of the declared recovery.

    Returns ``(recovery, eligible)``, where ``recovery`` is ``(bearing, reason)`` to apply or None.
    ``eligible`` is True only once the declared trigger has fired and the declared step budget still
    allows an attempt, so the caller can tell "waiting" from "tried and declined".

    The declared rule is: retrace the reversed last-known displacement whenever that is usable and
    region-safe, and otherwise - only in ``bounded_retrace_or_waypoint`` - re-aim at the current
    waypoint. Inside ``waypoint_bearing_within_distance_pixels`` the re-aim wins outright, because a
    measured run showed the retrace walking the hero away from the waypoint it was about to reach on
    the short blind windows that sit on the final leg. Both bearings are declined when the predicted
    step would leave the declared region, so an unanchored episode, or one with no room left, still
    waits rather than guessing.
    """
    if config is None or last_known is None:
        return None, False
    if missing_streak < int(cast(int, config["trigger_after_missing_frames"])):
        return None, False
    if recovery_steps >= int(cast(int, config["maximum_recovery_steps"])):
        return None, False
    nominal = float(cast(float, config["nominal_step_pixels"]))
    band_raw = config.get("waypoint_bearing_within_distance_pixels")
    within_band = (
        target is not None
        and band_raw is not None
        and math.hypot(last_known[0] - target[0], last_known[1] - target[1])
        <= float(cast(float, band_raw))
    )
    if not within_band and previous_known is not None:
        bearing = _retrace_direction(last_known, previous_known, region, nominal, recovery_steps)
        if bearing is not None:
            return (bearing, "unknown_recovery_retrace"), True
    if config["mode"] == "bounded_retrace_or_waypoint" and target is not None:
        bearing = _toward_target_direction(last_known, target, region, nominal, recovery_steps)
        if bearing is not None:
            return (bearing, "unknown_recovery_waypoint"), True
    return None, True


def _progress_guard_offset(escape_step: int, hold_steps: int, offsets: list[int]) -> int:
    """The bearing offset for one step of a bounded escape schedule."""
    if hold_steps <= 0 or not offsets or escape_step < 0:
        raise MobileTestbedError("progress guard escape schedule is invalid")
    index = escape_step // hold_steps
    if index >= len(offsets):
        raise MobileTestbedError("progress guard escape schedule is exhausted")
    return offsets[index]


def _episode_outcome(
    *,
    arrived: bool,
    death: bool,
    outside_region: bool,
    no_advance: bool,
    missing_streak: int,
    maximum_gap: int,
    budget_exhausted: bool,
    detour_exhausted: bool = False,
) -> tuple[bool, str, str, str | None]:
    """Decide whether the episode ends and why.

    A sustained localisation gap is a capture failure, not a timeout: without this a run that can
    never localise silently burns its whole duration while sending no input, which is both a wasted
    attempt and misleading evidence. An exhausted detour budget is reported as its own failure too,
    so a route that keeps meeting a bearing the mask removes is never quietly relabelled a timeout.
    """
    if arrived:
        return True, "NAVIGATION_GOAL_REACHED", "TERMINATED", None
    if death:
        return True, "SAFETY_STOP", "ERROR", "death_or_ended_screen"
    if outside_region:
        return True, "SAFETY_STOP", "ERROR", "outside_free_movement_region"
    if no_advance:
        # The applied action produced no advance at all, which is a different failure from a lost
        # marker: the hero is localised, a bearing is held, and the world is not moving. Ending here
        # is an explicit truncation of a dead state rather than a spent budget.
        return True, "ACTION_FAILURE", "ERROR", "no_advance_detected"
    if maximum_gap > 0 and missing_streak > maximum_gap:
        return True, "CAPTURE_FAILURE", "ERROR", "localization_gap"
    if detour_exhausted:
        # Every declared detour was spent and the goal-directed bearing is still removed while the
        # hero is still not advancing, so the route genuinely cannot get through rather than merely
        # having run out of clock. A lost marker stays a capture failure above this.
        return True, "DETOUR_FAILURE", "ERROR", "detour_exhausted"
    if budget_exhausted:
        # An episode must always end with a terminal transition; a step or duration cap is a
        # truncated episode, not a non-terminal one, which the reload verifier would reject.
        return True, "TIMEOUT", "TRUNCATED", "step_or_duration_budget_exhausted"
    return False, "NOT_DONE", "NOT_DONE", None


@dataclass(frozen=True, slots=True)
class _NavigationDevice:
    """One opened guard, scrcpy session, joystick and watchdog, shared by two contract phases.

    A composed placement-then-route run must not open a second guard between the phases: the whole
    point is that the start the placement reaches is the start the route leaves from. The device is
    therefore resolved once and handed to each phase's runtime.
    """

    guard: DeviceGuard
    session: ScrcpyControlSession
    joystick: PersistentJoystick
    watchdog: GuardWatchdog


@dataclass(frozen=True, slots=True)
class _NavigationRuntime:
    contract: dict[str, object]
    contract_sha: str
    store_contract: dict[str, object]
    visual_sha: str
    execution_sha: str
    rois_sha: str
    rois: ObservationROIs
    targets: list[tuple[float, float]]
    tolerance: float
    hold_ms: int
    approach_hold_ms: int
    approach_distance: float
    hysteresis: int
    region: dict[str, object]
    region_limit: int
    settle_ns: int
    maximum_steps: int
    maximum_gap: int
    maximum_duration_seconds: float
    progress_guard: dict[str, object] | None
    region_filter: dict[str, object] | None
    planner: dict[str, object] | None
    unknown_recovery: dict[str, object] | None
    approach_press_distance: float | None
    approach_direct_bearing: bool
    no_advance_guard: dict[str, object] | None
    traversability: dict[str, object] | None
    detour: dict[str, object] | None
    masked_persistence: dict[str, object] | None
    approach_commitment: int
    approach_exit_margin: float
    stall_commitment: int
    death_confirmation_steps: int
    death_stationary_window_steps: int
    death_maximum_travel_pixels: float
    backlog_free_maximum_messages: int
    approach_tiers: list[tuple[float, int]]
    approach_default_hold_ms: int
    guard: DeviceGuard
    session: ScrcpyControlSession
    joystick: PersistentJoystick
    watchdog: GuardWatchdog
    frames_dir: Path
    database: Path
    enable_input: bool


def _prepare_navigation_runtime(
    *,
    serial: str,
    contract_path: Path,
    visual_layout_path: Path,
    execution_layout_path: Path,
    observation_rois_path: Path,
    output: Path,
    enable_input: bool,
    device: _NavigationDevice | None = None,
) -> _NavigationRuntime:
    contract, contract_sha = _goal_navigation_contract(contract_path)
    store_contract = _store_contract(contract, contract_sha)
    visual_layout, visual_sha = load_layout(visual_layout_path)
    execution_layout, execution_sha = load_layout(execution_layout_path)
    rois, rois_sha = load_observation_rois(observation_rois_path)
    if visual_layout.width != execution_layout.width:
        raise MobileTestbedError("mobile navigation store layouts differ")
    if device is None:
        guard = _open_device_guard(serial)
        session = ScrcpyControlSession(guard.serial, 30)
        joystick = PersistentJoystick(execution_layout, guard.width, guard.height)
        watchdog = GuardWatchdog(guard, ACTIVE_PROBE_GUARD_INTERVAL_SECONDS)
    else:
        # A composed run passes the already-opened device so both phases share one session.
        guard = device.guard
        session = device.session
        joystick = device.joystick
        watchdog = device.watchdog
    if (guard.width, guard.height) != (visual_layout.width, visual_layout.height):
        raise MobileTestbedError("mobile navigation store display differs")
    if rois.width != guard.width or rois.height != guard.height:
        raise MobileTestbedError("mobile navigation store rois differ")
    targets = [
        (
            float(cast(float, cast(list[object], item)[0])),
            float(cast(float, cast(list[object], item)[1])),
        )
        for item in cast(list[object], contract["targets_minimap_xy"])
    ]
    tolerance = float(cast(float, contract["arrival_tolerance_pixels"]))
    hold_ms = int(cast(int, contract["direction_hold_ms"]))
    region = cast(dict[str, object], contract["free_movement_region"])
    return _NavigationRuntime(
        contract=contract,
        contract_sha=contract_sha,
        store_contract=store_contract,
        visual_sha=visual_sha,
        execution_sha=execution_sha,
        rois_sha=rois_sha,
        rois=rois,
        targets=targets,
        tolerance=tolerance,
        hold_ms=hold_ms,
        approach_hold_ms=int(cast(int, contract.get("final_approach_hold_ms", hold_ms))),
        approach_distance=float(
            cast(float, contract.get("final_approach_distance_pixels", tolerance))
        ),
        hysteresis=int(cast(int, contract.get("direction_hysteresis_sectors", 0))),
        region=region,
        region_limit=int(cast(int, region.get("maximum_consecutive_violation_frames", 10))),
        settle_ns=int(cast(int, store_contract["settle_ms"])) * 1_000_000,
        maximum_steps=int(cast(int, store_contract["maximum_steps"])),
        maximum_gap=int(cast(int, store_contract.get("maximum_localization_gap_frames", 0))),
        maximum_duration_seconds=float(
            cast(float, contract.get("maximum_duration_seconds", 0.0))
        ),
        progress_guard=cast(dict[str, object] | None, store_contract.get("progress_guard")),
        region_filter=cast(dict[str, object] | None, store_contract.get("region_filter")),
        planner=cast(dict[str, object] | None, store_contract.get("planner")),
        unknown_recovery=cast(dict[str, object] | None, contract.get("unknown_recovery")),
        approach_press_distance=(
            None
            if store_contract.get("final_approach") is None
            or cast(dict[str, object], store_contract["final_approach"]).get(
                "press_release_maximum_distance_pixels"
            )
            is None
            else float(
                cast(
                    float,
                    cast(dict[str, object], store_contract["final_approach"])[
                        "press_release_maximum_distance_pixels"
                    ],
                )
            )
        ),
        approach_direct_bearing=(
            store_contract.get("final_approach") is not None
            and bool(
                cast(dict[str, object], store_contract["final_approach"]).get("direct_bearing")
            )
        ),
        no_advance_guard=cast(dict[str, object] | None, contract.get("no_advance_guard")),
        traversability=cast(dict[str, object] | None, contract.get("traversability")),
        detour=cast(dict[str, object] | None, contract.get("detour")),
        masked_persistence=cast(dict[str, object] | None, contract.get("masked_persistence")),
        approach_exit_margin=float(
            cast(
                float,
                (
                    cast(dict[str, object], store_contract["final_approach"]).get(
                        "direct_bearing_exit_margin_pixels", 0.0
                    )
                    if store_contract.get("final_approach") is not None
                    else 0.0
                ),
            )
        ),
        stall_commitment=int(
            cast(
                int,
                (
                    cast(dict[str, object], contract["planner"]).get("stall_commitment_steps", 1)
                    if contract.get("planner") is not None
                    else 1
                ),
            )
        ),
        death_confirmation_steps=rois.death_confirmation_steps,
        death_stationary_window_steps=rois.death_stationary_window_steps,
        death_maximum_travel_pixels=rois.death_maximum_travel_pixels,
        approach_commitment=int(
            cast(
                int,
                (
                    cast(dict[str, object], store_contract["final_approach"]).get(
                        "commitment_steps", 1
                    )
                    if store_contract.get("final_approach") is not None
                    else 1
                ),
            )
        ),
        backlog_free_maximum_messages=int(
            cast(int, store_contract.get("backlog_free_maximum_pointer_messages", 2))
        ),
        approach_tiers=(
            [
                (
                    float(cast(float, cast(dict[str, object], item)["maximum_distance_pixels"])),
                    int(cast(int, cast(dict[str, object], item)["hold_ms"])),
                )
                for item in cast(
                    list[object],
                    cast(dict[str, object], store_contract["final_approach"])["tiers"],
                )
            ]
            if store_contract.get("final_approach") is not None
            else []
        ),
        approach_default_hold_ms=(
            int(
                cast(
                    int,
                    cast(dict[str, object], store_contract["final_approach"])["default_hold_ms"],
                )
            )
            if store_contract.get("final_approach") is not None
            else 0
        ),
        guard=guard,
        session=session,
        joystick=joystick,
        watchdog=watchdog,
        frames_dir=output / "frames",
        database=output / "transitions.sqlite3",
        enable_input=enable_input,
    )


def _run_episode(
    runtime: _NavigationRuntime, store: UnifiedTransitionStore, episode_id: str
) -> dict[str, object]:
    contract = runtime.contract
    guard = runtime.guard
    joystick = runtime.joystick
    watchdog = runtime.watchdog
    session = runtime.session
    rois = runtime.rois
    region = runtime.region
    state = _GoalNavigationTemplateState()
    steps: list[dict[str, object]] = []
    previous_position: tuple[float, float] | None = None
    region_streak = 0
    missing_streak = 0
    start_position: tuple[float, float] | None = None
    progress_guard = runtime.progress_guard
    guard_offsets = (
        [
            int(cast(int, item))
            for item in cast(list[object], progress_guard["escape_offsets_sectors"])
        ]
        if progress_guard is not None
        and progress_guard.get("mode") == "bounded_bearing_escape"
        else []
    )
    guard_hold = (
        int(cast(int, progress_guard["escape_hold_steps"]))
        if progress_guard is not None and progress_guard.get("mode") == "bounded_bearing_escape"
        else 1
        if progress_guard is not None
        else 0
    )
    guard_confirmation = (
        int(cast(int, progress_guard["confirmation_steps"]))
        if progress_guard is not None
        else 0
    )
    guard_improvement = (
        float(cast(float, progress_guard["minimum_improvement_pixels"]))
        if progress_guard is not None
        else 0.0
    )
    guard_max_events = (
        int(cast(int, progress_guard["maximum_events_per_episode"]))
        if progress_guard is not None
        else 0
    )
    guard_escape_steps = (
        len(guard_offsets) * guard_hold
        if guard_offsets
        else int(cast(int, progress_guard["stall_steps"])) * guard_hold
        if progress_guard is not None and progress_guard.get("mode") == "stall_trigger_only"
        else 0
    )
    best_distance: float | None = None
    no_progress_streak = 0
    escape_active = False
    escape_steps = 0
    guard_events = 0
    guard_escape_steps_total = 0
    region_filter_masked_steps = 0
    recovery_block = runtime.unknown_recovery
    recovery_hold_ms = int(cast(int, recovery_block["hold_ms"])) if recovery_block else 0
    retrace_from: tuple[float, float] | None = None
    retrace_to: tuple[float, float] | None = None
    recovery_steps = 0
    recovery_steps_total = 0
    recovery_events = 0
    recovery_declined_steps = 0
    approach_press_steps = 0
    advance_guard = runtime.no_advance_guard
    advance_window = int(cast(int, advance_guard["window_steps"])) if advance_guard else 0
    advance_travel = (
        float(cast(float, advance_guard["maximum_travel_pixels"])) if advance_guard else 0.0
    )
    committed_approach: str | None = None
    committed_steps = 0
    committed_stall: str | None = None
    stall_commitment_steps = 0
    was_in_approach_band = False
    traversability_masked_steps = 0
    approach_committed_steps = 0
    stall_committed_steps = 0
    advance_anchor: tuple[float, float] | None = None
    advance_steps = 0
    no_advance = False
    waypoint_index = 0
    pointer_messages = 0
    # The joystick is a physical pointer that outlives an episode, so the runner must adopt its
    # real direction instead of assuming a released pointer. Assuming STOP while the touch is
    # still down makes the first command a phantom DOWN that emits no message.
    previous_applied = JOYSTICK_TO_STORE_DIRECTION.get(joystick.direction, "STOP")
    previous_joystick = joystick.direction
    retry_total = 0
    failure: str | None = None
    arrived = False
    terminal_reason = "NOT_DONE"
    end_kind = "NOT_DONE"
    abort_reason: str | None = None
    started = time.monotonic()
    episode_deadline = started + runtime.maximum_duration_seconds
    last_position: tuple[float, float] | None = None
    committed: tuple[HierarchicalTransitionRecord, ...] = ()
    # A death is a state, not a single frame. The banner colour test alone fires on any red UI or
    # terrain that passes under the declared box, so it is confirmed only after it holds for the
    # declared number of consecutive observations and the hero has not travelled within the declared
    # window. A dead hero cannot move; the 2026-09-22 rebind stopped while the hero was walking.
    banner_streak = 0
    recent_positions: list[tuple[float, float] | None] = []
    death = False
    banner_steps = 0
    death_steps = 0
    # Declared detour state. The phase is explicit so a plan cannot be half-applied, and every
    # counter is reported, because a detour that never fires and a detour that fires and fails must
    # not look the same in the record.
    detour_phase = "idle"
    detour_plan_bearings: list[str] = []
    detour_index = 0
    detour_attempts = 0
    detour_steps = 0
    detour_unknown_steps = 0
    detour_effective_events = 0
    detour_confirm_left = 0
    detour_confirm_baseline: float | None = None
    stall_distance_window: list[float] = []
    detour_exhausted = False
    # Bounded masked-bearing persistence: the state is explicit so a hold cannot be half-applied,
    # and every counter is reported, because a rule that keeps a mask-removed bearing has to show
    # in the record rather than inferred from the outcome.
    persistence_left = 0
    persistence_activations = 0
    persistence_steps = 0
    persistence_effective_events = 0
    persistence_exhausted_events = 0
    persistence_index = 0
    # One window serves every declared rule that needs a stall; each rule reads its own declared
    # own declared number of steps out of it, so a contract may declare one rule or both.
    stall_window_length = max(
        [
            int(cast(int, block["trigger_stall_steps"]))
            for block in (runtime.detour, runtime.masked_persistence)
            if block is not None
        ]
        or [0]
    )

    def dispatch(operations: list[TouchOperation]) -> tuple[int, int, int, int, str]:
        nonlocal pointer_messages, retry_total
        start_ns = time.monotonic_ns()
        if not runtime.enable_input:
            return start_ns, start_ns, start_ns + runtime.settle_ns, 0, "noop"
        try:
            for index, operation in enumerate(operations):
                if index:
                    time.sleep(ACTIVE_PROBE_TOUCH_SETTLE_SECONDS)
                watchdog.ensure_fresh_or_refresh(ACTIVE_PROBE_GUARD_STALENESS_MS)
                session.touch(operation, guard.width, guard.height)
                pointer_messages += 1
        except Exception:
            retry_total += 1
            try:
                for operation in operations:
                    watchdog.ensure_fresh_or_refresh(ACTIVE_PROBE_GUARD_STALENESS_MS)
                    session.touch(operation, guard.width, guard.height)
                    pointer_messages += 1
            except Exception as exc:  # noqa: BLE001
                raise MobileTestbedError(f"mobile navigation store dispatch failed: {exc}") from exc
        ack_ns = time.monotonic_ns()
        return start_ns, ack_ns, ack_ns + runtime.settle_ns, retry_total and 1, "acknowledged"

    try:
        step_id = 0
        pending: tuple[int, int, int, np.ndarray] | None = None
        while step_id < runtime.maximum_steps:
            watchdog.ensure_fresh_or_refresh(ACTIVE_PROBE_GUARD_STALENESS_MS)
            if pending is None:
                capture_start_ns = time.monotonic_ns()
                frame_timestamp_ns, frame = session.frame()
                capture_end_ns = time.monotonic_ns()
            else:
                capture_start_ns, capture_end_ns, frame_timestamp_ns, frame = pending
                pending = None
            banner = _death_replay_visible(frame, rois)
            minimap = _observation_roi_frame(frame, rois.minimap)
            state.previous = previous_position
            position = _goal_navigation_tracked_cue(minimap, contract, state, step_id)
            previous_position = state.previous
            last_position = position
            banner_streak = banner_streak + 1 if banner else 0
            recent_positions.append(position)
            if len(recent_positions) > runtime.death_stationary_window_steps:
                recent_positions.pop(0)
            death = _death_confirmed(
                banner_streak,
                recent_positions,
                runtime.death_confirmation_steps,
                runtime.death_maximum_travel_pixels,
            )
            if banner:
                banner_steps += 1
            if death:
                death_steps += 1
            known = position is not None
            missing_streak = 0 if known else missing_streak + 1
            if known:
                if recovery_steps:
                    # The retreat walked the hero backwards, so the progress guard must not read
                    # that as a stall. Clearing the baseline rescores progress from the new fix.
                    best_distance = None
                    no_progress_streak = 0
                    escape_active = False
                    escape_steps = 0
                retrace_from, retrace_to = retrace_to, position
                recovery_steps = 0
            outside_region = False
            if position is not None:
                inside = (
                    float(cast(float, region["minimum_y"]))
                    <= position[0]
                    <= float(cast(float, region["maximum_y"]))
                    and float(cast(float, region["minimum_x"]))
                    <= position[1]
                    <= float(cast(float, region["maximum_x"]))
                )
                region_streak = 0 if inside else region_streak + 1
                outside_region = region_streak > runtime.region_limit
            target = runtime.targets[waypoint_index]
            distance = (
                None
                if position is None
                else float(np.hypot(position[0] - target[0], position[1] - target[1]))
            )
            # Two declared rules need a stall, and it must be no-progress, not no-motion. The v16
            # v16 attempt failed because its stall rule never engaged, and the detour's first device
            # attempt did the same with a one-pixel bound: the failure oscillates inside a couple of
            # pixels, so the hero keeps moving while the goal distance barely changes. This window
            # window measures the improvement in goal distance instead so oscillation cannot hide a
            # stall, and a lost marker empties it, since an unknown position is not evidence either
            if stall_window_length:
                if distance is None:
                    stall_distance_window = []
                else:
                    stall_distance_window.append(distance)
                    if len(stall_distance_window) > stall_window_length:
                        stall_distance_window.pop(0)
            if progress_guard is not None and distance is not None:
                improved = best_distance is None or distance <= best_distance - guard_improvement
                if improved:
                    best_distance = distance
                    no_progress_streak = 0
                    if escape_active:
                        escape_active = False
                        escape_steps = 0
                else:
                    no_progress_streak += 1
                if (
                    not escape_active
                    and no_progress_streak >= guard_confirmation
                    and guard_events < guard_max_events
                ):
                    escape_active = True
                    escape_steps = 0
                    guard_events += 1
                    best_distance = distance
                    no_progress_streak = 0
            # The direct-approach aim is chosen by band membership, and membership flickers when
            # the hero sits on the boundary: the cold-start failure spent forty-five steps between
            # 9.2 and 12.4 px against a declared 12.0, alternating north and north-east because
            # inside the band it aims at the waypoint and outside it aims at the path lookahead.
            # That is not the bearing hysteresis removed in v12 - it is the boundary of the band -
            # so entering and leaving use different distances, by the declared exit margin.
            if position is None or death or outside_region or distance is None:
                in_approach_band = False
            elif distance <= runtime.approach_distance:
                in_approach_band = True
            else:
                in_approach_band = bool(
                    was_in_approach_band
                    and distance <= runtime.approach_distance + runtime.approach_exit_margin
                )
            was_in_approach_band = in_approach_band
            requested_joystick = "wait"
            planner = runtime.planner
            if planner is not None and position is not None and not death and not outside_region:
                if start_position is None:
                    start_position = position
                if runtime.approach_direct_bearing and in_approach_band:
                    # Inside the declared final approach the aim is the waypoint itself. The planner
                    # otherwise aims at its path reference plus a lookahead, which at a few pixels
                    # out keeps correcting across the leg instead of closing the remaining gap.
                    # Hysteresis is deliberately not applied here: the first version of this rule
                    # passed the previous bearing, which pinned the aim to a neighbouring sector
                    # whose measured y response was about zero, so the hero could only move across.
                    requested_joystick = _goal_navigation_direction(position, target)
                else:
                    requested_joystick, _reference = _planner_direction(
                    position,
                    runtime.targets,
                    waypoint_index,
                    start_position,
                    region,
                    nominal_step_pixels=float(cast(float, planner["nominal_step_pixels"])),
                    margin_pixels=float(cast(float, planner["margin_pixels"])),
                    lookahead_pixels=float(cast(float, planner["lookahead_pixels"])),
                    stall_active=escape_active,
                    stall_bias=float(cast(float, planner["stall_bias"])),
                )
            elif position is not None and not death and not outside_region:
                requested_joystick = _goal_navigation_direction(
                    position, target, previous_joystick, runtime.hysteresis
                )
            recovery: tuple[str, str] | None = None
            if not known and not death and not outside_region:
                recovery, eligible = _unknown_recovery_step(
                    last_known=retrace_to,
                    previous_known=retrace_from,
                    target=target,
                    missing_streak=missing_streak,
                    recovery_steps=recovery_steps,
                    config=recovery_block,
                    region=region,
                )
                if eligible:
                    if recovery is None:
                        recovery_declined_steps += 1
                    else:
                        if recovery_steps == 0:
                            recovery_events += 1
                        recovery_steps += 1
                        recovery_steps_total += 1
            # in_approach_band was computed above, with the declared exit margin applied.
            traversability_masked_here = False
            masked_away_bearing: str | None = None
            if (
                runtime.traversability is not None
                and position is not None
                and not death
                and not outside_region
            ):
                masked_joystick = _mask_joystick_bearing(
                    position, requested_joystick, runtime.traversability
                )
                if masked_joystick != requested_joystick:
                    traversability_masked_steps += 1
                    masked_away_bearing = requested_joystick
                    requested_joystick = masked_joystick
                    traversability_masked_here = True
            # The finest press tier moves the hero about 0.65 px, near the position noise floor,
            # so re-aiming every step lets the steps cancel. Hold the chosen bearing for the
            # declared commitment instead of re-deciding from a noisy position.
            (
                requested_joystick,
                committed_approach,
                committed_steps,
                was_committed,
            ) = _commit_approach(
                requested_joystick,
                committed_approach,
                committed_steps,
                runtime.approach_commitment,
                in_approach_band,
            )
            if was_committed:
                approach_committed_steps += 1
            # The stall response has the same failure mode the approach had. The planner rewards a
            # lateral component with an absolute value, so the two opposite sidesteps score equally
            # and it can alternate between them: on the cold-start failure the hero oscillated in x
            # for about 25 s inside a 2 px box, alternating north-east and west, making no forward
            # progress at all. Holding the chosen sidestep for the declared commitment is the same
            # fix that closed the fine-approach stall, applied to the stall response instead.
            if runtime.stall_commitment > 1 and escape_active and not in_approach_band:
                (
                    requested_joystick,
                    committed_stall,
                    stall_commitment_steps,
                    was_stall_committed,
                ) = _commit_approach(
                    requested_joystick,
                    committed_stall,
                    stall_commitment_steps,
                    runtime.stall_commitment,
                    True,
                )
                if was_stall_committed:
                    stall_committed_steps += 1
            else:
                committed_stall = None
                stall_commitment_steps = 0
            # The declared detour outranks the geometry proposal and the approach commitment,
            # because the v16 attempt showed the stall happens inside the approach band where the
            # commitment deliberately holds its aim. It does not outrank the mask: a planned bearing
            # is re-masked before it is applied, so the detour can only pick a bearing the frozen
            # grid permits. Nothing here reads the device or the model.
            # The declared masked-bearing persistence deliberately keeps a bearing the frozen grid
            # removed, so it is bounded on every axis and reported: it activates only when the mask
            # has just removed the goal bearing and the hero has made no progress, it may be tried
            # only the declared number of times per episode and held only the declared steps, and it
            # ends when the goal distance improves by the declared amount, which means the creep
            # worked. The corridor measurement justifies it: at the stall cell
            # the mask removes north on a 0.0548 projection while north is the only bearing with
            # a northward component and the hero creeps about 0.6 px per press by the wall, so the
            # mask removes the only press that moves. Nothing here reads the device or the model.
            if (
                runtime.masked_persistence is not None
                and detour_phase == "idle"
                and position is not None
                and not death
                and not outside_region
            ):
                persistence_applied = False
                persistence = runtime.masked_persistence
                sweep = cast(list[str], persistence["persistence_bearings"])
                goal_bearing = _goal_bearing(position, target)
                if (
                    persistence_left > 0
                    and runtime.traversability is not None
                    and _mask_joystick_bearing(position, goal_bearing, runtime.traversability)
                    == goal_bearing
                ):
                    # The goal bearing is usable in this cell now, measured on the bearing itself
                    # rather than on the planner's proposal, so the sweep did what it was for and
                    # normal planning resumes.
                    persistence_effective_events += 1
                    persistence_left = 0
                if (
                    persistence_left <= 0
                    and persistence_activations
                    < int(cast(int, persistence["maximum_activations_per_episode"]))
                    and traversability_masked_here
                    and masked_away_bearing is not None
                    and _no_progress_stall(
                        stall_distance_window,
                        int(cast(int, persistence["trigger_stall_steps"])),
                        float(cast(float, persistence["trigger_stall_progress_pixels"])),
                    )
                ):
                    persistence_activations += 1
                    persistence_left = int(cast(int, persistence["maximum_steps"]))
                    persistence_index = 0
                    committed_approach = None
                    committed_steps = 0
                if persistence_left > 0:
                    # Sweep the declared upward bearings rather than holding one. The directional
                    # probe is what shows this is the escaping motion: cycling north, north-east and
                    # north-west carried the hero five cells up the corridor, while holding one
                    # masked bearing did not. The sweep is bounded by the declared step count.
                    requested_joystick = sweep[persistence_index % len(sweep)]
                    persistence_applied = True
                    persistence_index += 1
                    committed_approach = None
                    committed_steps = 0
                    persistence_steps += 1
                    persistence_left -= 1
                    if persistence_left == 0:
                        # The declared sweep ran out while the goal bearing was still blocked, so
                        # this activation is recorded as exhausted and normal masking resumes.
                        persistence_exhausted_events += 1
            detour_bearing: str | None = None
            if (
                runtime.detour is not None
                and runtime.traversability is not None
                and position is not None
                and not death
                and not outside_region
            ):
                detour = runtime.detour
                traversability_block = runtime.traversability
                attempts_cap = int(cast(int, detour["maximum_attempts_per_episode"]))
                stalled = _no_progress_stall(
                    stall_distance_window,
                    int(cast(int, detour["trigger_stall_steps"])),
                    float(cast(float, detour["trigger_stall_progress_pixels"])),
                )
                if detour_phase == "running":
                    detour_bearing = detour_plan_bearings[detour_index]
                    detour_index += 1
                    if detour_index >= len(detour_plan_bearings):
                        detour_phase = "confirming"
                        detour_confirm_left = int(cast(int, detour["confirmation_steps"]))
                        detour_confirm_baseline = distance
                elif detour_phase == "confirming":
                    progressed = (
                        distance is not None
                        and detour_confirm_baseline is not None
                        and distance
                        <= detour_confirm_baseline
                        - float(cast(float, detour["confirmation_minimum_progress_pixels"]))
                    )
                    if progressed:
                        # The bearing is usable again and the goal distance fell over the declared
                        # confirmation window, so the detour did what it was planned to do.
                        detour_effective_events += 1
                        detour_phase = "idle"
                    else:
                        detour_confirm_left -= 1
                        if detour_confirm_left <= 0:
                            detour_phase = "idle"
                if detour_phase == "idle" and traversability_masked_here and stalled:
                    if detour_attempts < attempts_cap:
                        plan = plan_grid_detour(
                            position=position,
                            goal=target,
                            block=traversability_block,
                            region=region,
                            maximum_steps=int(cast(int, detour["maximum_steps"])),
                            bearing_order=tuple(cast(list[str], detour["bearing_order"])),
                        )
                        detour_attempts += 1
                        if plan is not None and plan.bearings:
                            # The planner speaks the recorded store vocabulary while the Router and
                            # the joystick speak the joystick vocabulary, and the two overlap on
                            # nothing. Convert once, here, at the boundary: handing a raw plan
                            # bearing on would both mask nothing and send an invalid direction.
                            detour_plan_bearings = [
                                STORE_TO_JOYSTICK_DIRECTION[bearing] for bearing in plan.bearings
                            ]
                            detour_unknown_steps += plan.unknown_steps
                            detour_index = 1
                            detour_bearing = detour_plan_bearings[0]
                            if detour_index >= len(detour_plan_bearings):
                                detour_phase = "confirming"
                                detour_confirm_left = int(cast(int, detour["confirmation_steps"]))
                                detour_confirm_baseline = distance
                            else:
                                detour_phase = "running"
                    else:
                        # Every declared attempt is spent and the same trigger still holds, so the
                        # route cannot get through rather than merely having run out of clock.
                        detour_exhausted = True
                if detour_bearing is not None:
                    requested_joystick = _mask_joystick_bearing(
                        position, detour_bearing, traversability_block
                    )
                    committed_approach = None
                    committed_steps = 0
                    detour_steps += 1
            applied_joystick, owner, reason = _route(
                requested_joystick,
                known=known,
                death=death,
                outside_region=outside_region,
                recovery=recovery,
            )
            if detour_bearing is not None and reason == "geometry_rule":
                owner = "deterministic_router"
                reason = "detour_applied"
            elif persistence_applied and reason == "geometry_rule":
                owner = "deterministic_router"
                reason = "persistence_hold"
            recovery_applied = reason in {"unknown_recovery_retrace", "unknown_recovery_waypoint"}
            if escape_active and progress_guard is not None and not death:
                if (
                    runtime.planner is None
                    and applied_joystick != "wait"
                    and escape_steps < guard_escape_steps
                ):
                    offset = _progress_guard_offset(escape_steps, guard_hold, guard_offsets)
                    applied_joystick = _rotate_direction(applied_joystick, offset)
                    owner = "deterministic_router"
                    reason = "progress_guard_escape"
                escape_steps += 1
                guard_escape_steps_total += 1
                if escape_steps >= guard_escape_steps:
                    escape_active = False
                    escape_steps = 0
            region_filter = runtime.region_filter
            region_filter_applied = False
            if (
                runtime.planner is None
                and region_filter is not None
                and position is not None
                and not death
                and applied_joystick != "wait"
            ):
                masked = _region_safe_direction(
                    position,
                    applied_joystick,
                    region,
                    float(cast(float, region_filter["nominal_step_pixels"])),
                    float(cast(float, region_filter["margin_pixels"])),
                )
                if masked != applied_joystick:
                    applied_joystick = masked
                    region_filter_applied = True
                    region_filter_masked_steps += 1
                    owner = "deterministic_router"
                    reason = "region_filter_masked"
            requested = JOYSTICK_TO_STORE_DIRECTION[requested_joystick]
            applied = JOYSTICK_TO_STORE_DIRECTION[applied_joystick]
            command = _movement_command(previous_applied, applied)
            step_hold_ms = (
                _approach_hold_ms(
                    distance, runtime.approach_tiers, runtime.approach_default_hold_ms
                )
                if runtime.approach_tiers
                else runtime.approach_hold_ms
                if distance is not None and distance <= runtime.approach_distance
                else runtime.hold_ms
            )
            before_messages = pointer_messages
            dispatched_direction = applied_joystick if command in {"DOWN", "MOVE", "UP"} else None
            operations = (
                joystick.set_direction(dispatched_direction)
                if dispatched_direction is not None
                else []
            )
            (
                dispatch_start_ns,
                dispatch_ack_ns,
                settle_end_ns,
                retry_count,
                status,
            ) = dispatch(operations)
            approach_press_ms = 0
            if (
                runtime.approach_press_distance is not None
                and distance is not None
                and distance <= runtime.approach_press_distance
                and applied_joystick != "wait"
            ):
                # The declared deceleration has to bound the press itself. The persistent joystick
                # holds the touch down until its direction changes, so without this explicit release
                # the declared hold only sets the sampling period: the hero travels for the whole
                # step, and near a 4 px arrival tolerance that makes the final leg oscillate instead
                # of converging. The press lasts exactly the declared tier hold, then the bearing is
                # released and the step is not extended afterwards.
                approach_press_ms = step_hold_ms
                time.sleep(approach_press_ms / 1000)
                (
                    _release_start_ns,
                    _release_ack_ns,
                    release_settle_end_ns,
                    _release_retries,
                    release_status,
                ) = dispatch(joystick.set_direction("wait"))
                if release_status not in {"acknowledged", "noop"}:
                    raise MobileTestbedError("mobile navigation store approach release failed")
                settle_end_ns = release_settle_end_ns
                approach_press_steps += 1
                previous_applied = "STOP"
            else:
                previous_applied = applied
            step_messages = pointer_messages - before_messages
            previous_joystick = applied_joystick
            while time.monotonic_ns() < settle_end_ns:
                watchdog.ensure_fresh_or_refresh(ACTIVE_PROBE_GUARD_STALENESS_MS)
                time.sleep(GOAL_NAVIGATION_LOOP_SLEEP_SECONDS)
            next_start_ns = time.monotonic_ns()
            next_timestamp_ns, next_frame = session.frame()
            next_end_ns = time.monotonic_ns()
            if next_end_ns < settle_end_ns:
                raise MobileTestbedError("mobile navigation store next capture precedes settle")
            pending = (next_start_ns, next_end_ns, next_timestamp_ns, next_frame)
            observation = _packet(
                frame,
                rois,
                directory=runtime.frames_dir,
                episode_id=episode_id,
                step_id=step_id,
                capture_start_ns=capture_start_ns,
                capture_end_ns=capture_end_ns,
            )
            next_observation = _packet(
                next_frame,
                rois,
                directory=runtime.frames_dir,
                episode_id=episode_id,
                step_id=step_id + 1,
                capture_start_ns=next_start_ns,
                capture_end_ns=next_end_ns,
            )
            if distance is not None and distance <= runtime.tolerance:
                waypoint_index += 1
                if waypoint_index >= len(runtime.targets):
                    arrived = True
            advance_anchor, advance_steps, no_advance = _no_advance_detected(
                advance_anchor,
                advance_steps,
                position,
                applied_joystick != "wait" and not death and not outside_region,
                advance_window,
                advance_travel,
            )
            done, terminal_reason, end_kind, abort_reason = _episode_outcome(
                arrived=arrived,
                death=death,
                outside_region=outside_region,
                no_advance=no_advance,
                missing_streak=missing_streak,
                maximum_gap=runtime.maximum_gap,
                budget_exhausted=time.monotonic() >= episode_deadline
                or step_id + 1 >= runtime.maximum_steps,
                detour_exhausted=detour_exhausted,
            )
            row = _transition(
                store_contract=runtime.store_contract,
                episode_id=episode_id,
                step_id=step_id,
                observation=observation,
                next_observation=next_observation,
                requested=requested,
                applied=applied,
                owner=owner,
                reason=reason,
                movement_command=command,
                dispatch_start_ns=dispatch_start_ns,
                dispatch_ack_ns=dispatch_ack_ns,
                settle_end_ns=settle_end_ns,
                final_status=status,
                retry_count=retry_count,
                done=done,
                terminal_reason=terminal_reason,
                end_kind=end_kind,
            )
            stored = store.append(row)
            if not stored.validation.valid:
                raise MobileTestbedError(
                    "mobile navigation store transition invalid: "
                    + ",".join(stored.validation.errors)
                )
            steps.append(
                {
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "frame_timestamp_ns": frame_timestamp_ns,
                    "goal_index": waypoint_index,
                    "goal_xy": list(
                        runtime.targets[min(waypoint_index, len(runtime.targets) - 1)]
                    ),
                    "position": None if position is None else [position[0], position[1]],
                    "distance_pixels": distance,
                    "requested_movement": requested,
                    "applied_movement": applied,
                    "router_owner": owner,
                    "router_reason": reason,
                    "movement_command": command,
                    "dispatched_direction": dispatched_direction,
                    "dispatch_start_ns": dispatch_start_ns,
                    "dispatch_ack_ns": dispatch_ack_ns,
                    "settle_end_ns": settle_end_ns,
                    "pointer_messages": step_messages,
                    "retry_count": retry_count,
                    "final_status": status,
                    "done": done,
                    "abort_reason": abort_reason,
                    "region_filter_applied": region_filter_applied,
                    "progress_guard_escape": escape_active,
                    "progress_guard_escape_step": escape_steps,
                    "region_filter_masked_steps": region_filter_masked_steps,
                    "unknown_recovery_events": recovery_events,
                    "unknown_recovery_steps": recovery_steps_total,
                    "unknown_recovery_depth": recovery_steps,
                    "unknown_recovery_declined_steps": recovery_declined_steps,
                    "approach_press_ms": approach_press_ms,
                    "no_advance_steps": advance_steps,
                    "traversability_masked_steps": traversability_masked_steps,
                    "approach_committed_steps": approach_committed_steps,
                    "stall_committed_steps": stall_committed_steps,
                    "progress_guard_events": guard_events,
                    "terminal_reason": terminal_reason,
                    "training_eligible": stored.payload["training_eligible"],
                }
            )
            if done:
                break
            step_id += 1
            # A pulse step already released the bearing and waited out the release settle, so it is
            # not extended further; otherwise the declared hold or the recovery hold owns the wait.
            hold = (
                0
                if approach_press_ms
                else recovery_hold_ms
                if recovery_applied
                else step_hold_ms
            )
            deadline = time.monotonic() + hold / 1000
            while time.monotonic() < deadline:
                watchdog.ensure_fresh_or_refresh(ACTIVE_PROBE_GUARD_STALENESS_MS)
                time.sleep(GOAL_NAVIGATION_LOOP_SLEEP_SECONDS)
        # Release the pointer between episodes so each episode starts from rest rather than
        # inheriting a held bearing, and count the messages like any other dispatch.
        with suppress(Exception):
            dispatch(joystick.set_direction("wait"))
        committed = store.load_episode(episode_id)
    except Exception as exc:
        failure = str(exc)
        with suppress(Exception):
            committed = store.load_episode(episode_id)
    terminal_rows = [row for row in committed if row["done"]]
    max_messages = max((cast(int, row["pointer_messages"]) for row in steps), default=0)
    return {
        "schema_version": MOBILE_NAVIGATION_STORE_SCHEMA,
        "status": "PASSED" if failure is None and arrived else "FAILED",
        "episode_id": episode_id,
        "store_transitions": len(committed),
        "terminal_transitions": len(terminal_rows),
        "terminal_reason": terminal_reason,
        "episode_end_kind": end_kind,
        "steps": len(steps),
        "arrived": arrived,
        "waypoints_reached": waypoint_index,
        "final_position": None if last_position is None else list(last_position),
        "input_commands_sent": pointer_messages,
        "input_enabled": runtime.enable_input,
        "max_actions_per_step": max_messages,
        "retries": retry_total,
        "steps_without_ack": sum(
            1 for row in steps if row["final_status"] not in {"acknowledged", "noop"}
        ),
        "backlog_free": max_messages <= runtime.backlog_free_maximum_messages
        and all(row["final_status"] in {"acknowledged", "noop"} for row in steps),
        "failure": failure,
        "abort_reason": abort_reason,
        "progress_guard_events": guard_events,
        "progress_guard_escape_steps": guard_escape_steps_total,
        "region_filter_masked_steps": region_filter_masked_steps,
        "unknown_recovery_events": recovery_events,
        "unknown_recovery_steps": recovery_steps_total,
        "unknown_recovery_declined_steps": recovery_declined_steps,
        "approach_press_steps": approach_press_steps,
        "no_advance_events": int(no_advance),
        "traversability_masked_steps": traversability_masked_steps,
        "approach_committed_steps": approach_committed_steps,
        # A banner that is seen and rejected is the evidence that the confirmation policy is doing
        # work rather than the run simply never meeting a banner, so both counts are reported.
        "death_banner_steps": banner_steps,
        "death_confirmed_steps": death_steps,
        "detour_attempts": detour_attempts,
        "detour_steps": detour_steps,
        "detour_unknown_steps": detour_unknown_steps,
        "detour_effective_events": detour_effective_events,
        "detour_exhausted": int(detour_exhausted),
        "masked_persistence_activations": persistence_activations,
        "masked_persistence_steps": persistence_steps,
        "masked_persistence_effective_events": persistence_effective_events,
        "masked_persistence_exhausted_events": persistence_exhausted_events,
        "duration_seconds": round(time.monotonic() - started, 8),
        "step_rows": steps,
    }


def _open_runtime(runtime: _NavigationRuntime) -> None:
    runtime.session.start()
    if runtime.session.frame_size != (runtime.guard.width, runtime.guard.height):
        raise MobileTestbedError("mobile navigation store scrcpy frame size differs")
    runtime.watchdog.start()


def _close_runtime(runtime: _NavigationRuntime, pointer_messages: int) -> None:
    try:
        if runtime.enable_input:
            for operation in runtime.joystick.release():
                runtime.session.touch(operation, runtime.guard.width, runtime.guard.height)
    finally:
        runtime.watchdog.stop()
        runtime.session.close()


def _base_summary(runtime: _NavigationRuntime) -> dict[str, object]:
    return {
        "schema_version": MOBILE_NAVIGATION_STORE_SCHEMA,
        "contract_sha256": runtime.contract_sha,
        "visual_layout_sha256": runtime.visual_sha,
        "execution_layout_sha256": runtime.execution_sha,
        "observation_rois_sha256": runtime.rois_sha,
        "store_path": runtime.database.name,
        "input_enabled": runtime.enable_input,
        "raw_frames_persisted": False,
        "derived_views_persisted": True,
        "training_eligible": False,
        "events_claimed": 0,
        "event_engine_version": runtime.store_contract["event_engine_version"],
    }


def run_mobile_navigation_episode(
    *,
    serial: str,
    contract_path: Path,
    visual_layout_path: Path,
    execution_layout_path: Path,
    observation_rois_path: Path,
    output_dir: Path,
    enable_input: bool = True,
) -> dict[str, object]:
    output = _new_large_output(output_dir)
    runtime = _prepare_navigation_runtime(
        serial=serial,
        contract_path=contract_path,
        visual_layout_path=visual_layout_path,
        execution_layout_path=execution_layout_path,
        observation_rois_path=observation_rois_path,
        output=output,
        enable_input=enable_input,
    )
    episode_id = _episode_id(
        cast(str, runtime.store_contract["episode_prefix"]), runtime.contract_sha
    )
    _open_runtime(runtime)
    try:
        with UnifiedTransitionStore(runtime.database) as store:
            episode = _run_episode(runtime, store, episode_id)
    finally:
        _close_runtime(runtime, cast(int, episode.get("input_commands_sent", 0)))
    steps = cast(list[dict[str, object]], episode.pop("step_rows"))
    summary = _base_summary(runtime) | episode
    (output / "steps.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in steps), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_mobile_navigation_episodes(
    *,
    serial: str,
    contract_path: Path,
    visual_layout_path: Path,
    execution_layout_path: Path,
    observation_rois_path: Path,
    output_dir: Path,
    episodes: int,
    enable_input: bool = True,
) -> dict[str, object]:
    """Run several consecutive episodes on one device session and one Store (L2 gate)."""
    if episodes <= 0:
        raise MobileTestbedError("mobile navigation store episode count is invalid")
    output = _new_large_output(output_dir)
    runtime = _prepare_navigation_runtime(
        serial=serial,
        contract_path=contract_path,
        visual_layout_path=visual_layout_path,
        execution_layout_path=execution_layout_path,
        observation_rois_path=observation_rois_path,
        output=output,
        enable_input=enable_input,
    )
    prefix = cast(str, runtime.store_contract["episode_prefix"])
    episode_ids = [
        f"{_episode_id(prefix, runtime.contract_sha)}-{ordinal:02d}"
        for ordinal in range(1, episodes + 1)
    ]
    results: list[dict[str, object]] = []
    pointer_messages = 0
    _open_runtime(runtime)
    try:
        with UnifiedTransitionStore(runtime.database) as store:
            for ordinal, episode_id in enumerate(episode_ids, start=1):
                episode = _run_episode(runtime, store, episode_id)
                steps = cast(list[dict[str, object]], episode.pop("step_rows"))
                directory = output / f"episode-{ordinal:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "steps.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in steps),
                    encoding="utf-8",
                )
                summary = _base_summary(runtime) | episode
                (directory / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                pointer_messages += cast(int, episode.get("input_commands_sent", 0))
                results.append(summary)
            verified = [
                verify_mobile_navigation_episode(
                    store_path=runtime.database,
                    episode_id=episode_id,
                    frame_root=runtime.frames_dir,
                )
                for episode_id in episode_ids
            ]
            integrity = store.integrity()
            bindings = {
                (
                    row["policy_bundle_sha256"],
                    row["event_engine_sha256"],
                    row["policy_bundle_version"],
                )
                for row in store.load_episode(episode_ids[0])
            }
    finally:
        _close_runtime(runtime, pointer_messages)
    findings = sorted(
        {finding for report in verified for finding in cast(list[str], report["findings"])}
    )
    backlog_free = all(bool(row["backlog_free"]) for row in results)
    max_actions = max(
        (cast(int, row["max_actions_per_step"]) for row in results), default=0
    )
    batch: dict[str, object] = _base_summary(runtime) | {
        "schema_version": MOBILE_NAVIGATION_STORE_BATCH_SCHEMA,
        "status": "PASSED"
        if all(bool(row["arrived"]) for row in results)
        and backlog_free
        and not findings
        and integrity == "ok"
        and len(bindings) == 1
        else "FAILED",
        "episodes": episodes,
        "arrivals": sum(1 for row in results if bool(row["arrived"])),
        "arrival_rate": sum(1 for row in results if bool(row["arrived"])) / episodes,
        "transitions_total": sum(cast(int, row["store_transitions"]) for row in results),
        "terminal_transitions_total": sum(
            cast(int, row["terminal_transitions"]) for row in results
        ),
        "input_commands_sent": pointer_messages,
        "max_actions_per_step": max_actions,
        "retries_total": sum(cast(int, row["retries"]) for row in results),
        "steps_without_ack": sum(cast(int, row["steps_without_ack"]) for row in results),
        "backlog_free": backlog_free,
        "frame_reference_findings": findings,
        "store_integrity": integrity,
        "policy_binding_versions": len(bindings),
        "binding_stable": len(bindings) == 1,
        "episode_summaries": results,
    }
    (output / "batch-summary.json").write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return batch


def _placement_start_gate(
    *,
    placement: dict[str, object],
    declared_start: tuple[float, float],
    tolerance: float,
) -> dict[str, object]:
    """Decide whether the route may start, from the placement episode's own recorded outcome.

    This is a pre-action check, not an independent truth claim: it reads the position the frozen
    green-ring cue localised at the placement's last step and compares it with the declared start.
    A placement that did not arrive, lost its marker, or stopped outside the declared start
    tolerance refuses the route; the caller then never starts the route phase, so a bad start
    cannot be retried forever.
    """
    final = placement.get("final_position")
    if not isinstance(final, list) or len(final) != 2:
        return {
            "started_route": False,
            "reason": "PLACEMENT_POSITION_MISSING",
            "distance_pixels": None,
        }
    distance = math.hypot(
        float(cast(float, final[0])) - declared_start[0],
        float(cast(float, final[1])) - declared_start[1],
    )
    if not bool(placement.get("arrived")):
        return {
            "started_route": False,
            "reason": "PLACEMENT_NOT_ARRIVED",
            "distance_pixels": distance,
        }
    if distance > tolerance:
        return {
            "started_route": False,
            "reason": "PLACEMENT_OUTSIDE_START_TOLERANCE",
            "distance_pixels": distance,
        }
    return {
        "started_route": True,
        "reason": "PLACEMENT_CONFIRMED",
        "distance_pixels": distance,
    }


def _placement_route_summary(
    *,
    placement: dict[str, object],
    gate: dict[str, object],
    route: dict[str, object] | None,
) -> dict[str, object]:
    """Separate denominators, because a placement arrival is not a route success.

    `session_attempts` counts one composed attempt. A placement-only arrival can raise
    `placement_successes` and nothing else: `route_started`, `route_successes` and
    `end_to_end_successes` all stay zero when the route phase did not run or did not arrive.
    """
    placed = bool(gate.get("started_route"))
    routed = route is not None
    arrived = routed and bool(cast(dict[str, object], route).get("arrived"))
    return {
        "session_attempts": 1,
        "placement_successes": 1 if placed else 0,
        "route_started": 1 if routed else 0,
        "route_successes": 1 if arrived else 0,
        "end_to_end_successes": 1 if (placed and arrived) else 0,
        "placement_seconds": placement.get("duration_seconds"),
        "route_seconds": None if route is None else route.get("duration_seconds"),
        "placement_episode_id": placement.get("episode_id"),
        "route_episode_id": None if route is None else route.get("episode_id"),
        "start_gate": gate,
    }


def run_mobile_navigation_placement_route(
    *,
    serial: str,
    placement_contract_path: Path,
    route_contract_path: Path,
    visual_layout_path: Path,
    execution_layout_path: Path,
    observation_rois_path: Path,
    output_dir: Path,
    enable_input: bool = True,
) -> dict[str, object]:
    """Place, confirm the start, then run the unchanged route, all on one session and one Store.

    Both phases share the opened device and the single `UnifiedTransitionStore`. The placement
    phase's declared single target is the declared start. The route starts only when the placement
    arrived and its recorded final position is inside the declared start tolerance. The route
    contract is not modified: its own arrival gate is untouched.
    """
    output = _new_large_output(output_dir)
    placement_runtime = _prepare_navigation_runtime(
        serial=serial,
        contract_path=placement_contract_path,
        visual_layout_path=visual_layout_path,
        execution_layout_path=execution_layout_path,
        observation_rois_path=observation_rois_path,
        output=output,
        enable_input=enable_input,
    )
    placement_targets = placement_runtime.targets
    if len(placement_targets) != 1:
        raise MobileTestbedError(
            "mobile navigation placement route needs exactly one placement target"
        )
    declared_start = placement_targets[0]
    start_tolerance = float(
        cast(
            float,
            placement_runtime.contract.get(
                "start_gate_tolerance_pixels", placement_runtime.tolerance
            ),
        )
    )
    device = _NavigationDevice(
        guard=placement_runtime.guard,
        session=placement_runtime.session,
        joystick=placement_runtime.joystick,
        watchdog=placement_runtime.watchdog,
    )
    route_runtime = _prepare_navigation_runtime(
        serial=serial,
        contract_path=route_contract_path,
        visual_layout_path=visual_layout_path,
        execution_layout_path=execution_layout_path,
        observation_rois_path=observation_rois_path,
        output=output,
        enable_input=enable_input,
        device=device,
    )
    placement_prefix = cast(str, placement_runtime.store_contract["episode_prefix"])
    route_prefix = cast(str, route_runtime.store_contract["episode_prefix"])
    # The phase is recoverable from the episode id, so a placement step is never read as a route
    # step without a Store schema change.
    placement_episode_id = (
        f"{_episode_id(placement_prefix, placement_runtime.contract_sha)}-placement"
    )
    route_episode_id = f"{_episode_id(route_prefix, route_runtime.contract_sha)}-route"
    placement: dict[str, object]
    route: dict[str, object] | None = None
    gate: dict[str, object]
    pointer_messages = 0
    _open_runtime(placement_runtime)
    try:
        with UnifiedTransitionStore(placement_runtime.database) as store:
            placement = _run_episode(placement_runtime, store, placement_episode_id)
            (output / "placement-steps.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in cast(list[dict[str, object]], placement.pop("step_rows"))
                ),
                encoding="utf-8",
            )
            gate = _placement_start_gate(
                placement=placement,
                declared_start=declared_start,
                tolerance=start_tolerance,
            )
            pointer_messages += cast(int, placement.get("input_commands_sent", 0))
            if bool(gate["started_route"]):
                route = _run_episode(route_runtime, store, route_episode_id)
                (output / "route-steps.jsonl").write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for row in cast(list[dict[str, object]], route.pop("step_rows"))
                    ),
                    encoding="utf-8",
                )
                pointer_messages += cast(int, route.get("input_commands_sent", 0))
    finally:
        _close_runtime(placement_runtime, pointer_messages)
    counters = _placement_route_summary(placement=placement, gate=gate, route=route)
    episodes = [placement] if route is None else [placement, route]
    verified = [
        verify_mobile_navigation_episode(
            store_path=placement_runtime.database,
            episode_id=cast(str, episode["episode_id"]),
            frame_root=placement_runtime.frames_dir,
        )
        for episode in episodes
    ]
    findings = sorted(
        {finding for report in verified for finding in cast(list[str], report["findings"])}
    )
    with UnifiedTransitionStore(placement_runtime.database) as store:
        integrity = store.integrity()
    end_to_end = cast(int, counters["end_to_end_successes"]) == 1
    summary: dict[str, object] = _base_summary(placement_runtime) | counters | {
        "schema_version": MOBILE_NAVIGATION_PLACEMENT_ROUTE_SCHEMA,
        "status": "PASSED" if end_to_end and not findings and integrity == "ok" else "FAILED",
        "setup_failure": (
            None
            if bool(gate["started_route"])
            else cast(str, gate["reason"])
        ),
        "declared_start_xy": [declared_start[0], declared_start[1]],
        "start_gate_tolerance_pixels": start_tolerance,
        "route_contract_sha256": route_runtime.contract_sha,
        "placement_contract_sha256": placement_runtime.contract_sha,
        "input_commands_sent": pointer_messages,
        "frame_reference_findings": findings,
        "store_integrity": integrity,
        "placement_summary": placement,
        "route_summary": route,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def verify_mobile_navigation_store(*, store_path: Path, frame_root: Path) -> dict[str, object]:
    """Verify every episode in one Store plus its integrity (L2 gate)."""
    with UnifiedTransitionStore(store_path) as store:
        integrity = store.integrity()
        episode_ids = store.episode_ids()
    reports = [
        verify_mobile_navigation_episode(
            store_path=store_path, episode_id=episode_id, frame_root=frame_root
        )
        for episode_id in episode_ids
    ]
    findings = sorted(
        {finding for report in reports for finding in cast(list[str], report["findings"])}
    )
    return {
        "store_path": store_path.name,
        "episodes": len(episode_ids),
        "episode_ids": list(episode_ids),
        "transitions": sum(cast(int, report["transitions"]) for report in reports),
        "store_integrity": integrity,
        "recoverable": integrity == "ok" and not findings and bool(episode_ids),
        "findings": findings,
        "reports": reports,
    }


def verify_mobile_navigation_episode(
    *, store_path: Path, episode_id: str, frame_root: Path
) -> dict[str, object]:
    """Reload one committed episode and re-check its causal chain and frame bundles."""
    with UnifiedTransitionStore(store_path) as store:
        rows = store.load_episode(episode_id)
    findings: list[str] = []
    if not rows:
        findings.append("episode_missing")
    for index, row in enumerate(rows):
        if row["step_id"] != index:
            findings.append(f"step_gap_at_{index}")
        validation = validate_transition(row)
        if not validation.valid:
            findings.extend(validation.errors)
        if row["done"] and index != len(rows) - 1:
            findings.append("episode_continues_after_terminal")
        if index and rows[index - 1]["next_observation"]["observation_id"] != row["observation"][
            "observation_id"
        ]:
            findings.append(f"observation_chain_broken_at_{index}")
        for label in ("observation", "next_observation"):
            packet = row[label]
            basename = packet["frame_bundle_ref"]
            bundle = frame_root / basename
            if not bundle.exists():
                findings.append(f"frame_bundle_missing:{basename}")
                continue
            with np.load(bundle, allow_pickle=False) as saved:
                for view, expected in packet["view_sha256"].items():
                    if view not in saved.files:
                        findings.append(f"frame_view_missing:{basename}:{view}")
                        continue
                    actual = hashlib.sha256(
                        np.ascontiguousarray(saved[view]).tobytes()
                    ).hexdigest()
                    if actual != expected:
                        findings.append(f"frame_view_hash_mismatch:{basename}:{view}")
    if rows and not rows[-1]["done"]:
        findings.append("terminal_transition_missing")
    return {
        "episode_id": episode_id,
        "transitions": len(rows),
        "terminal_reason": rows[-1]["terminal_reason"] if rows else None,
        "recoverable": not findings,
        "findings": sorted(set(findings)),
    }


PANEL_CAPTURE_SCHEMA = "hok-agent-panel-capture-v1"


def capture_panel_samples(
    *,
    serial: str,
    visual_layout_path: Path,
    observation_rois_path: Path,
    output_dir: Path,
    episode_id: str,
    seconds: float,
    sample_hz: float,
) -> dict[str, object]:
    """Read-only bounded capture of the panel ROI for R1.

    This function deliberately constructs no input sender: it has no joystick, sends no touch and
    cannot dispatch anything. It writes only the derived panel view and its capture timestamps.
    """
    if seconds <= 0 or sample_hz <= 0:
        raise MobileTestbedError("panel capture duration and rate must be positive")
    if sample_hz > 30:
        raise MobileTestbedError("panel capture rate is above the capture ceiling")
    visual_layout, visual_sha = load_layout(visual_layout_path)
    rois, rois_sha = load_observation_rois(observation_rois_path)
    output = _new_large_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    guard = _open_device_guard(serial)
    if (guard.width, guard.height) != (visual_layout.width, visual_layout.height):
        raise MobileTestbedError("panel capture display differs")
    if rois.width != guard.width or rois.height != guard.height:
        raise MobileTestbedError("panel capture rois differ")
    session = ScrcpyControlSession(guard.serial, 30)
    watchdog = GuardWatchdog(guard, ACTIVE_PROBE_GUARD_INTERVAL_SECONDS)
    views: list[np.ndarray] = []
    capture_ns: list[int] = []
    interval = 1.0 / sample_hz
    failure: str | None = None
    started = 0.0
    try:
        session.start()
        if session.frame_size != (guard.width, guard.height):
            raise MobileTestbedError("panel capture scrcpy frame size differs")
        watchdog.start()
        started = time.monotonic()
        deadline = started + seconds
        next_at = started
        while time.monotonic() < deadline:
            watchdog.ensure_fresh_or_refresh(ACTIVE_PROBE_GUARD_STALENESS_MS)
            timestamp_ns, frame = session.frame()
            views.append(
                np.ascontiguousarray(
                    _observation_roi_frame(frame, rois.recommended_equipment)
                )
            )
            capture_ns.append(int(timestamp_ns))
            next_at += interval
            sleep_for = next_at - time.monotonic()
            while sleep_for > 0:
                time.sleep(min(sleep_for, GOAL_NAVIGATION_LOOP_SLEEP_SECONDS))
                sleep_for = next_at - time.monotonic()
    except Exception as exc:  # noqa: BLE001
        failure = str(exc)
    finally:
        watchdog.stop()
        session.close()
    shards: list[dict[str, object]] = []
    if views:
        shards_dir = output / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        for index in range(0, len(views), 256):
            name = f"panel-{index // 256:04d}.npz"
            np.savez_compressed(
                shards_dir / name,
                equipment=np.stack(views[index : index + 256]),
                capture_ns=np.asarray(capture_ns[index : index + 256], dtype=np.int64),
            )
            shards.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256((shards_dir / name).read_bytes()).hexdigest(),
                    "rows": len(views[index : index + 256]),
                }
            )
    summary: dict[str, object] = {
        "schema_version": PANEL_CAPTURE_SCHEMA,
        "status": "PASSED" if failure is None and views else "FAILED",
        "episode_id": episode_id,
        "purpose": "r1_panel_gating_samples",
        "visual_layout_sha256": visual_sha,
        "observation_rois_sha256": rois_sha,
        "seconds": seconds,
        "sample_hz": sample_hz,
        "samples": len(views),
        "shards": shards,
        "device_input_sent": False,
        "input_sender_constructed": False,
        "derived_views_persisted": True,
        "failure": failure,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
