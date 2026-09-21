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
import os
import time
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

MOBILE_NAVIGATION_STORE_SCHEMA = "hok-agent-mobile-navigation-store-session-v1"
MOBILE_NAVIGATION_STORE_BATCH_SCHEMA = "hok-agent-mobile-navigation-store-batch-v1"
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
    resolved = dict(block)
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
    requested: str, *, known: bool, death: bool, outside_region: bool
) -> tuple[str, str, str]:
    """Deterministic Router: masks the geometry proposal and keeps requested versus applied.

    Both the request and the result stay in the joystick vocabulary; the store vocabulary is
    applied at the transition boundary only.
    """
    if death:
        return "wait", "deterministic_router", "death_or_ended_screen"
    if outside_region:
        return "wait", "deterministic_router", "outside_free_movement_region"
    if not known:
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
) -> _NavigationRuntime:
    contract, contract_sha = _goal_navigation_contract(contract_path)
    store_contract = _store_contract(contract, contract_sha)
    visual_layout, visual_sha = load_layout(visual_layout_path)
    execution_layout, execution_sha = load_layout(execution_layout_path)
    rois, rois_sha = load_observation_rois(observation_rois_path)
    if visual_layout.width != execution_layout.width:
        raise MobileTestbedError("mobile navigation store layouts differ")
    guard = _open_device_guard(serial)
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
        guard=guard,
        session=ScrcpyControlSession(guard.serial, 30),
        joystick=PersistentJoystick(execution_layout, guard.width, guard.height),
        watchdog=GuardWatchdog(guard, ACTIVE_PROBE_GUARD_INTERVAL_SECONDS),
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
    previous_applied = "STOP"
    previous_joystick = "wait"
    region_streak = 0
    waypoint_index = 0
    pointer_messages = 0
    retry_total = 0
    failure: str | None = None
    arrived = False
    terminal_reason = "NOT_DONE"
    end_kind = "NOT_DONE"
    started = time.monotonic()
    last_position: tuple[float, float] | None = None
    committed: tuple[HierarchicalTransitionRecord, ...] = ()

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
            death = _death_replay_visible(frame, rois)
            minimap = _observation_roi_frame(frame, rois.minimap)
            state.previous = previous_position
            position = _goal_navigation_tracked_cue(minimap, contract, state, step_id)
            previous_position = state.previous
            last_position = position
            known = position is not None
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
            requested_joystick = "wait"
            if position is not None and not death and not outside_region:
                requested_joystick = _goal_navigation_direction(
                    position, target, previous_joystick, runtime.hysteresis
                )
            applied_joystick, owner, reason = _route(
                requested_joystick, known=known, death=death, outside_region=outside_region
            )
            requested = JOYSTICK_TO_STORE_DIRECTION[requested_joystick]
            applied = JOYSTICK_TO_STORE_DIRECTION[applied_joystick]
            command = _movement_command(previous_applied, applied)
            before_messages = pointer_messages
            operations = (
                joystick.set_direction(applied_joystick)
                if command in {"DOWN", "MOVE", "UP"}
                else []
            )
            (
                dispatch_start_ns,
                dispatch_ack_ns,
                settle_end_ns,
                retry_count,
                status,
            ) = dispatch(operations)
            step_messages = pointer_messages - before_messages
            previous_applied = applied
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
            if arrived:
                terminal_reason = "NAVIGATION_GOAL_REACHED"
                end_kind = "TERMINATED"
            elif death or outside_region:
                terminal_reason = "SAFETY_STOP"
                end_kind = "ERROR"
            done = arrived or death or outside_region
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
                    "dispatch_start_ns": dispatch_start_ns,
                    "dispatch_ack_ns": dispatch_ack_ns,
                    "settle_end_ns": settle_end_ns,
                    "pointer_messages": step_messages,
                    "retry_count": retry_count,
                    "final_status": status,
                    "done": done,
                    "terminal_reason": terminal_reason,
                    "training_eligible": stored.payload["training_eligible"],
                }
            )
            if done:
                break
            step_id += 1
            hold = (
                runtime.approach_hold_ms
                if distance is not None and distance <= runtime.approach_distance
                else runtime.hold_ms
            )
            deadline = time.monotonic() + hold / 1000
            while time.monotonic() < deadline:
                watchdog.ensure_fresh_or_refresh(ACTIVE_PROBE_GUARD_STALENESS_MS)
                time.sleep(GOAL_NAVIGATION_LOOP_SLEEP_SECONDS)
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
        "backlog_free": max_messages <= 2
        and all(row["final_status"] in {"acknowledged", "noop"} for row in steps),
        "failure": failure,
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
