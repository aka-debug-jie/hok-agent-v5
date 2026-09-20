from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np

from hok_agent.movement_mvp import StageAMovement, _movement_command
from hok_agent.movement_real_rgb import (
    _file_sha256,
    _filtered_player_positions,
    _goal_direction,
    _load_bound_json,
    _mask_components,
    _object_sha256,
    _player_candidates,
)

_MOVEMENT_VECTORS = {
    "north": (-1.0, 0.0),
    "north_east": (-1.0, 1.0),
    "east": (0.0, 1.0),
    "south_east": (1.0, 1.0),
    "south": (1.0, 0.0),
    "south_west": (1.0, -1.0),
    "west": (0.0, -1.0),
    "north_west": (-1.0, -1.0),
}


def _valid_run_lengths(positions: list[tuple[float, float] | None]) -> list[int]:
    runs: list[int] = []
    current = 0
    for position in positions:
        if position is None:
            current = 0
        elif current == 0:
            current = 1
            runs.append(current)
        else:
            current += 1
            runs[-1] = current
    return runs


def run_partial_navigation_shadow(
    contract_path: Path,
    prior_summary_path: Path,
    session_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    started = time.monotonic()
    contract = _load_bound_json(contract_path, "contract_sha256")
    prior = _load_bound_json(prior_summary_path, "summary_sha256")
    if (
        contract.get("schema_version") != "movement-partial-navigation-shadow-v1"
        or contract.get("training_allowed") is not False
        or contract.get("test_allowed") is not False
        or contract.get("device_input_allowed") is not False
        or prior.get("status") != "DATA_SOURCE_LIMITED"
        or prior.get("visual_demonstrator_complete") is not True
        or prior.get("localization_improved") is not False
        or _file_sha256(prior_summary_path) != contract.get("prior_summary_file_sha256")
        or prior.get("summary_sha256") != contract.get("prior_summary_sha256")
        or prior.get("contract_sha256") != contract.get("prior_contract_sha256")
    ):
        raise ValueError("partial navigation Shadow contract differs")
    if session_root.is_symlink() or not session_root.is_dir():
        raise ValueError("partial navigation Shadow session root differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("partial navigation Shadow output already exists")

    declaration = cast(dict[str, object], contract["session"])
    basename = str(declaration["basename"])
    directory = session_root / basename
    source_summary_path = directory / "summary.json"
    if (
        basename != "teacher-session-002"
        or Path(basename).name != basename
        or _file_sha256(source_summary_path) != declaration["summary_sha256"]
    ):
        raise ValueError("partial navigation Shadow source identity differs")
    source_summary = _load_bound_json(source_summary_path, "summary_sha256")
    if (
        source_summary.get("status") != "PASSED"
        or source_summary.get("derived_roi_rgb_persisted") is not True
        or source_summary.get("raw_frames_persisted") is not False
    ):
        raise ValueError("partial navigation Shadow source summary differs")

    frame_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    opened_shards: list[dict[str, object]] = []
    for row in cast(list[dict[str, object]], source_summary["observation_shards"]):
        shard_name = str(row["path"])
        shard_path = directory / "shards" / shard_name
        if (
            Path(shard_name).name != shard_name
            or shard_path.is_symlink()
            or _file_sha256(shard_path) != row["sha256"]
        ):
            raise ValueError("partial navigation Shadow shard differs")
        with np.load(shard_path, allow_pickle=False) as shard:
            frame_parts.append(shard["minimap_rgb"].copy())
            timestamp_parts.append(shard["scheduled_elapsed_ms"].copy())
        opened_shards.append(
            {"basename": shard_name, "rows": row["rows"], "sha256": row["sha256"]}
        )
    frames = np.concatenate(frame_parts)
    timestamps = np.concatenate(timestamp_parts)
    expected_frames = int(cast(int, contract["expected_frames"]))
    frame_period_ms = int(cast(int, contract["frame_period_ms"]))
    if (
        frames.shape != (expected_frames, 128, 128, 3)
        or frames.dtype != np.uint8
        or timestamps.shape != (expected_frames,)
        or not np.all(np.diff(timestamps) == frame_period_ms)
    ):
        raise ValueError("partial navigation Shadow frame cache differs")

    cue_contract: dict[str, object] = {
        "color": contract["color"],
        "components": contract["components"],
    }
    exclusion = cast(
        tuple[int, int, int, int],
        tuple(map(int, cast(list[int], contract["excluded_ui_xyxy"]))),
    )
    _candidates, positions = _filtered_player_positions(frames, cue_contract, exclusion)
    goal_xy = cast(list[float], contract["goal_xy_relative"])
    goal_yx = (round(goal_xy[1] * 127), round(goal_xy[0] * 127))
    stop_radius = float(cast(float, contract["stop_radius_pixels"]))
    previous: StageAMovement = "STOP"
    shadow_rows: list[dict[str, object]] = []
    proposal_counts: Counter[str] = Counter()
    command_counts: Counter[str] = Counter()
    direction_switches = 0
    previous_valid: str | None = None
    for index, (timestamp, position) in enumerate(zip(timestamps, positions, strict=True)):
        stop_reason: str | None
        if position is None:
            proposal = None
            shadow_action: StageAMovement = "STOP"
            stop_reason = "PLAYER_UNKNOWN"
            previous_valid = None
            owner = "deterministic_router"
        else:
            proposal = _goal_direction(position, goal_yx, stop_radius)
            shadow_action = cast(StageAMovement, proposal)
            stop_reason = "GOAL_REACHED" if proposal == "STOP" else None
            if previous_valid is not None and proposal != previous_valid:
                direction_switches += 1
            previous_valid = proposal
            proposal_counts[proposal] += 1
            owner = (
                "deterministic_router"
                if proposal == "STOP"
                else "deterministic_executor"
                if proposal == previous
                else "geometry_rule"
            )
        command = _movement_command(previous, shadow_action)
        command_counts[command] += 1
        shadow_rows.append(
            {
                "session": basename,
                "frame_index": index,
                "scheduled_elapsed_ms": int(timestamp),
                "localization_state": "direct" if position is not None else "unknown",
                "player_yx": list(position) if position is not None else None,
                "goal_xy_relative": goal_xy,
                "goal_source": "explicit_shadow_config",
                "proposal": proposal,
                "shadow_action": shadow_action,
                "movement_command": command,
                "decision_owner": owner,
                "stop_reason": stop_reason,
                "executed_action": None,
                "transition_written": False,
            }
        )
        previous = shadow_action

    runs = _valid_run_lengths(positions)
    direct = sum(position is not None for position in positions)
    measured: dict[str, object] = {
        "direct_observations": direct,
        "unknown_observations": len(positions) - direct,
        "valid_runs": len(runs),
        "longest_valid_run_frames": max(runs, default=0),
        "proposal_counts": dict(sorted(proposal_counts.items())),
        "command_counts": dict(sorted(command_counts.items())),
    }
    if measured != contract["expected"]:
        raise ValueError("partial navigation Shadow frozen counts differ")
    prior_session = next(
        row
        for row in cast(list[dict[str, object]], prior["sessions"])
        if row["session"] == basename
    )
    if (
        prior_session["samples_total"] != len(positions)
        or prior_session["direct_observations"] != direct
    ):
        raise ValueError("partial navigation Shadow differs from N1")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        rows_path = staging / "shadow.jsonl"
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in shadow_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        summary: dict[str, object] = {
            "schema_version": "movement-partial-navigation-shadow-report-v1",
            "status": "PARTIAL_OFFLINE_SHADOW_COMPLETE_DATA_SOURCE_LIMITED",
            "contract_sha256": contract["contract_sha256"],
            "prior_summary_sha256": prior["summary_sha256"],
            "session": basename,
            "frames": len(shadow_rows),
            "duration_ms": len(shadow_rows) * frame_period_ms,
            "localization_coverage": direct / len(shadow_rows),
            **measured,
            "longest_valid_run_ms": max(runs, default=0) * frame_period_ms,
            "direction_switches_within_valid_runs": direction_switches,
            "opened_shards": opened_shards,
            "shadow_file_sha256": _file_sha256(rows_path),
            "shadow_file_bytes": rows_path.stat().st_size,
            "source_rgb_persisted": False,
            "recorded_future_used_as_action_effect": False,
            "executed_actions": 0,
            "transitions_written": 0,
            "input_commands_sent": 0,
            "training_called": False,
            "test_frames_read": 0,
            "gpu_seconds": 0,
            "runtime_wall_seconds": time.monotonic() - started,
        }
        summary["summary_sha256"] = _object_sha256(summary)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return summary


def _response_candidate_pairs(
    start_candidates: list[tuple[float, float, float]],
    end_candidates: list[tuple[float, float, float]],
    exclusion: tuple[int, int, int, int],
) -> list[dict[str, object]]:
    x0, y0, x1, y1 = exclusion

    def group(
        candidates: list[tuple[float, float, float]], *, inside: bool
    ) -> list[tuple[float, float, float]]:
        return [
            candidate
            for candidate in candidates
            if (x0 <= candidate[1] < x1 and y0 <= candidate[0] < y1) is inside
        ]

    rows: list[dict[str, object]] = []
    for name, inside in (("interior", False), ("fixed_ui", True)):
        starts = group(start_candidates, inside=inside)
        ends = group(end_candidates, inside=inside)
        if len(starts) == len(ends) == 1:
            rows.append(
                {
                    "candidate_group": name,
                    "start_yx": list(starts[0][:2]),
                    "end_yx": list(ends[0][:2]),
                }
            )
    return rows


def _response_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    projections = [float(cast(float, row["projection_pixels"])) for row in rows]
    displacements = [float(cast(float, row["displacement_pixels"])) for row in rows]
    orthogonal = [float(cast(float, row["orthogonal_pixels"])) for row in rows]
    directions = Counter(str(row["action"]) for row in rows)
    return {
        "pairs": len(rows),
        "directions": dict(sorted(directions.items())),
        "responsive_pairs": sum(bool(row["responsive"]) for row in rows),
        "responsive_fraction": (
            sum(bool(row["responsive"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "positive_projection_fraction": (
            sum(value > 0.0 for value in projections) / len(rows) if rows else 0.0
        ),
        "median_projection_pixels": float(np.median(projections)) if rows else 0.0,
        "median_displacement_pixels": float(np.median(displacements)) if rows else 0.0,
        "median_orthogonal_pixels": float(np.median(orthogonal)) if rows else 0.0,
    }


def run_action_response_identity_audit(
    contract_path: Path,
    prior_summary_path: Path,
    session_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    started = time.monotonic()
    contract = _load_bound_json(contract_path, "contract_sha256")
    prior = _load_bound_json(prior_summary_path, "summary_sha256")
    movement_names = cast(list[str], contract.get("movement_names"))
    if (
        contract.get("schema_version") != "movement-action-response-identity-audit-v1"
        or contract.get("training_allowed") is not False
        or contract.get("test_allowed") is not False
        or contract.get("device_input_allowed") is not False
        or movement_names != ["wait", *_MOVEMENT_VECTORS]
        or prior.get("status") != "DATA_SOURCE_LIMITED"
        or prior.get("visual_demonstrator_complete") is not True
        or _file_sha256(prior_summary_path) != contract.get("prior_summary_file_sha256")
        or prior.get("summary_sha256") != contract.get("prior_summary_sha256")
    ):
        raise ValueError("action-response identity audit contract differs")
    if session_root.is_symlink() or not session_root.is_dir():
        raise ValueError("action-response identity audit session root differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("action-response identity audit output already exists")

    frame_period_ms = int(cast(int, contract["frame_period_ms"]))
    response_lag_ms = int(cast(int, contract["response_lag_ms"]))
    expected_frames = int(cast(int, contract["expected_frames_per_session"]))
    if frame_period_ms != 200 or response_lag_ms != 1000:
        raise ValueError("action-response identity audit timing differs")
    lag_frames = response_lag_ms // frame_period_ms
    exclusion = cast(
        tuple[int, int, int, int],
        tuple(map(int, cast(list[int], contract["excluded_ui_xyxy"]))),
    )
    cue_contract: dict[str, object] = {
        "color": contract["color"],
        "components": contract["components"],
    }
    minimum_projection = float(cast(float, contract["minimum_projection_pixels"]))
    pair_rows: list[dict[str, object]] = []
    action_events: dict[str, int] = {}
    opened_shards: list[dict[str, object]] = []
    for declaration in cast(list[dict[str, object]], contract["sessions"]):
        basename = str(declaration["basename"])
        directory = session_root / basename
        summary_path = directory / "summary.json"
        if (
            Path(basename).name != basename
            or _file_sha256(summary_path) != declaration["summary_sha256"]
        ):
            raise ValueError("action-response identity audit session differs")
        summary = _load_bound_json(summary_path, "summary_sha256")
        frame_parts: list[np.ndarray] = []
        timestamp_parts: list[np.ndarray] = []
        movement_parts: list[np.ndarray] = []
        sent_parts: list[np.ndarray] = []
        for source_row in cast(list[dict[str, object]], summary["observation_shards"]):
            shard_name = str(source_row["path"])
            shard_path = directory / "shards" / shard_name
            if (
                Path(shard_name).name != shard_name
                or shard_path.is_symlink()
                or _file_sha256(shard_path) != source_row["sha256"]
            ):
                raise ValueError("action-response identity audit shard differs")
            with np.load(shard_path, allow_pickle=False) as shard:
                frame_parts.append(shard["minimap_rgb"].copy())
                timestamp_parts.append(shard["scheduled_elapsed_ms"].copy())
                movement_parts.append(shard["movement_id"].copy())
                sent_parts.append(shard["movement_input_sent"].copy())
            opened_shards.append(
                {"session": basename, "basename": shard_name, "sha256": source_row["sha256"]}
            )
        frames = np.concatenate(frame_parts)
        timestamps = np.concatenate(timestamp_parts)
        movement_ids = np.concatenate(movement_parts).astype(np.int64)
        sent = np.concatenate(sent_parts).astype(bool)
        if (
            frames.shape != (expected_frames, 128, 128, 3)
            or timestamps.shape != (expected_frames,)
            or movement_ids.shape != (expected_frames,)
            or sent.shape != (expected_frames,)
            or not np.all(np.diff(timestamps) == frame_period_ms)
        ):
            raise ValueError("action-response identity audit arrays differ")
        events = 0
        for index in range(len(frames) - lag_frames):
            action_id = int(movement_ids[index])
            if not sent[index] or action_id == 0:
                continue
            events += 1
            action = movement_names[action_id]
            start_candidates = _player_candidates(frames[index], cue_contract)
            end_candidates = _player_candidates(frames[index + lag_frames], cue_contract)
            for pair in _response_candidate_pairs(start_candidates, end_candidates, exclusion):
                start_y, start_x = map(float, cast(list[float], pair["start_yx"]))
                end_y, end_x = map(float, cast(list[float], pair["end_yx"]))
                delta_y, delta_x = end_y - start_y, end_x - start_x
                vector_y, vector_x = _MOVEMENT_VECTORS[action]
                norm = math.hypot(vector_y, vector_x)
                projection = (delta_y * vector_y + delta_x * vector_x) / norm
                orthogonal = abs(delta_y * vector_x - delta_x * vector_y) / norm
                pair_rows.append(
                    {
                        "session": basename,
                        "event_frame": index,
                        "response_frame": index + lag_frames,
                        "event_elapsed_ms": int(timestamps[index]),
                        "action": action,
                        **pair,
                        "projection_pixels": projection,
                        "displacement_pixels": math.hypot(delta_y, delta_x),
                        "orthogonal_pixels": orthogonal,
                        "responsive": projection >= minimum_projection,
                    }
                )
        action_events[basename] = events

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        grouped[f"{row['session']}:{row['candidate_group']}"].append(row)
    session_metrics = {
        key: _response_metrics(rows) for key, rows in sorted(grouped.items())
    }
    session002_interior = session_metrics.get("teacher-session-002:interior", _response_metrics([]))
    fixed_rows = [row for row in pair_rows if row["candidate_group"] == "fixed_ui"]
    fixed_metrics = _response_metrics(fixed_rows)
    other_interior_pairs = sum(
        int(cast(int, metrics["pairs"]))
        for key, metrics in session_metrics.items()
        if key.endswith(":interior") and not key.startswith("teacher-session-002:")
    )
    gates = cast(dict[str, object], contract["gates"])
    checks = {
        "session002_interior_pairs": int(cast(int, session002_interior["pairs"]))
        >= int(cast(int, gates["minimum_session002_interior_pairs"])),
        "session002_direction_support": len(cast(dict[str, int], session002_interior["directions"]))
        >= int(cast(int, gates["minimum_session002_directions"])),
        "session002_responsive_fraction": float(
            cast(float, session002_interior["responsive_fraction"])
        )
        >= float(cast(float, gates["minimum_session002_responsive_fraction"])),
        "session002_median_projection": float(
            cast(float, session002_interior["median_projection_pixels"])
        )
        >= float(cast(float, gates["minimum_session002_median_projection_pixels"])),
        "fixed_ui_responsive_fraction": float(cast(float, fixed_metrics["responsive_fraction"]))
        <= float(cast(float, gates["maximum_fixed_ui_responsive_fraction"])),
        "fixed_ui_median_displacement": float(
            cast(float, fixed_metrics["median_displacement_pixels"])
        )
        <= float(cast(float, gates["maximum_fixed_ui_median_displacement_pixels"])),
    }
    passed = all(checks.values())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        pairs_path = staging / "pairs.jsonl"
        with pairs_path.open("w", encoding="utf-8") as handle:
            for row in pair_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        report: dict[str, object] = {
            "schema_version": "movement-action-response-identity-audit-report-v1",
            "status": (
                "ACTION_RESPONSE_SEPARATES_FIXED_UI_SESSION002_ONLY"
                if passed
                else "ACTION_RESPONSE_IDENTITY_NOT_SEPARABLE"
            ),
            "contract_sha256": contract["contract_sha256"],
            "prior_summary_sha256": prior["summary_sha256"],
            "action_events": action_events,
            "action_events_total": sum(action_events.values()),
            "candidate_pairs": len(pair_rows),
            "session_metrics": session_metrics,
            "fixed_ui_aggregate": fixed_metrics,
            "other_sessions_interior_pairs": other_interior_pairs,
            "checks": checks,
            "pairs_file_sha256": _file_sha256(pairs_path),
            "pairs_file_bytes": pairs_path.stat().st_size,
            "opened_shards": opened_shards,
            "candidate_generation_uses_action": False,
            "action_used_for_response_scoring_only": True,
            "semantic_player_identity_verified": False,
            "multi_session_identity_verified": False,
            "active_probe_design_supported": passed,
            "training_called": False,
            "test_frames_read": 0,
            "device_input_commands_sent": 0,
            "gpu_seconds": 0,
            "runtime_wall_seconds": time.monotonic() - started,
        }
        report["report_sha256"] = _object_sha256(report)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return report


_PROBE_DIRECTIONS = (
    "north",
    "north_east",
    "east",
    "south_east",
    "south",
    "south_west",
    "west",
    "north_west",
)


def _probe_unique_position(
    candidates: list[tuple[float, float, float]],
) -> tuple[float, float] | None:
    return (candidates[0][0], candidates[0][1]) if len(candidates) == 1 else None


def _probe_run_lengths(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in mask:
        if bool(flag):
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _probe_frame_candidates(
    frames: np.ndarray,
    cue_contract: dict[str, object],
    exclusion: tuple[int, int, int, int],
) -> tuple[
    list[list[tuple[float, float, float]]],
    list[list[tuple[float, float, float]]],
]:
    x0, y0, x1, y1 = exclusion
    interior: list[list[tuple[float, float, float]]] = []
    fixed: list[list[tuple[float, float, float]]] = []
    for frame in frames:
        frame_interior: list[tuple[float, float, float]] = []
        frame_fixed: list[tuple[float, float, float]] = []
        for candidate in _player_candidates(frame, cue_contract):
            if x0 <= candidate[1] < x1 and y0 <= candidate[0] < y1:
                frame_fixed.append(candidate)
            else:
                frame_interior.append(candidate)
        interior.append(frame_interior)
        fixed.append(frame_fixed)
    return interior, fixed


def _probe_last_frame(
    times: np.ndarray, analysis: np.ndarray, target_ms: int, gap_ms: int
) -> int | None:
    index = int(np.searchsorted(times, target_ms, side="right")) - 1
    if index < 0 or not bool(analysis[index]) or int(times[index]) < target_ms - gap_ms:
        return None
    return index


def _probe_event_contaminated(
    kind: str,
    direction_id: int,
    press_ack_ms: int,
    release_ack_ms: int,
    end_ms: int,
    times: np.ndarray,
    movement_ids: np.ndarray,
    sent: np.ndarray,
) -> bool:
    for index in np.flatnonzero((times >= press_ack_ms) & (times <= end_ms)):
        if not bool(sent[index]):
            continue
        if kind == "control":
            return True
        if int(times[index]) <= release_ack_ms and int(movement_ids[index]) == direction_id:
            continue
        return True
    return False


def _probe_event_metrics(
    event: dict[str, object],
    *,
    direction_ids: dict[str, int],
    observation_ms: int,
    gap_ms: int,
    minimum_projection: float,
    times: np.ndarray,
    analysis: np.ndarray,
    interior: list[list[tuple[float, float, float]]],
    fixed: list[list[tuple[float, float, float]]],
    movement_ids: np.ndarray,
    sent: np.ndarray,
) -> dict[str, object]:
    kind = str(event["kind"])
    direction = str(event["direction"]) if kind == "pulse" else None
    direction_id = direction_ids.get(direction, 0) if direction is not None else 0
    if kind == "pulse":
        press_ack_ms = int(cast(int, event["press_ack_ms"]))
        release_ack_ms = int(cast(int, event["release_ack_ms"]))
        end_target_ms = release_ack_ms + observation_ms
    else:
        press_ack_ms = int(cast(int, event["window_start_ms"]))
        release_ack_ms = press_ack_ms
        end_target_ms = int(cast(int, event["window_end_ms"]))
    baseline_index = _probe_last_frame(times, analysis, press_ack_ms, gap_ms)
    end_index = _probe_last_frame(times, analysis, end_target_ms, gap_ms)
    contaminated = bool(event.get("contaminated")) or _probe_event_contaminated(
        kind,
        direction_id,
        press_ack_ms,
        release_ack_ms,
        end_target_ms,
        times,
        movement_ids,
        sent,
    )
    hard_stop = bool(event.get("hard_stop"))
    row: dict[str, object] = {
        "kind": kind,
        "direction": direction,
        "press_ack_ms": press_ack_ms,
        "release_ack_ms": release_ack_ms,
        "response_end_ms": end_target_ms,
        "baseline_frame": baseline_index,
        "response_frame": end_index,
        "contaminated": contaminated,
        "hard_stop": hard_stop,
        "fate": "outside_analysis_window",
        "projection_pixels": None,
        "displacement_pixels": None,
        "orthogonal_pixels": None,
        "responsive": False,
        "fixed_ui_projection_pixels": None,
        "fixed_ui_displacement_pixels": None,
        "fixed_ui_responsive": False,
    }
    if baseline_index is None or end_index is None:
        return row
    baseline_interior = interior[baseline_index]
    end_interior = interior[end_index]
    if hard_stop or contaminated:
        row["fate"] = "contaminated"
    elif not baseline_interior:
        row["fate"] = "no_candidate_baseline"
    elif not end_interior:
        row["fate"] = "no_candidate_observation"
    elif len(baseline_interior) > 1:
        row["fate"] = "ambiguous_baseline"
    elif len(end_interior) > 1:
        row["fate"] = "ambiguous_observation"
    else:
        row["fate"] = "paired"
    fate = str(row["fate"])
    baseline_position = _probe_unique_position(baseline_interior)
    end_position = _probe_unique_position(end_interior)
    if baseline_position is not None and end_position is not None:
        delta_y = end_position[0] - baseline_position[0]
        delta_x = end_position[1] - baseline_position[1]
        row["displacement_pixels"] = math.hypot(delta_y, delta_x)
        if kind == "pulse" and direction is not None:
            vector_y, vector_x = _MOVEMENT_VECTORS[direction]
            norm = math.hypot(vector_y, vector_x)
            projection = (delta_y * vector_y + delta_x * vector_x) / norm
            row["projection_pixels"] = projection
            row["orthogonal_pixels"] = abs(delta_y * vector_x - delta_x * vector_y) / norm
            row["responsive"] = fate == "paired" and projection >= minimum_projection
    baseline_fixed = _probe_unique_position(fixed[baseline_index])
    end_fixed = _probe_unique_position(fixed[end_index])
    if (
        kind == "pulse"
        and direction is not None
        and baseline_fixed is not None
        and end_fixed is not None
    ):
        delta_y = end_fixed[0] - baseline_fixed[0]
        delta_x = end_fixed[1] - baseline_fixed[1]
        vector_y, vector_x = _MOVEMENT_VECTORS[direction]
        norm = math.hypot(vector_y, vector_x)
        projection = (delta_y * vector_y + delta_x * vector_x) / norm
        row["fixed_ui_projection_pixels"] = projection
        row["fixed_ui_displacement_pixels"] = math.hypot(delta_y, delta_x)
        row["fixed_ui_responsive"] = projection >= minimum_projection
    return row


def _probe_session_metrics(
    directory: Path,
    contract: dict[str, object],
    opened_shards: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary = _load_bound_json(directory / "summary.json", "summary_sha256")
    if (
        summary.get("schema_version") != contract["session_schema_version"]
        or summary.get("contract_sha256") != contract["contract_sha256"]
        or summary.get("derived_roi_rgb_persisted") is not True
        or summary.get("raw_frames_persisted") is not False
    ):
        raise ValueError("active probe session summary differs")
    frame_parts: list[np.ndarray] = []
    scheduled_parts: list[np.ndarray] = []
    elapsed_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    allowed_parts: list[np.ndarray] = []
    movement_parts: list[np.ndarray] = []
    sent_parts: list[np.ndarray] = []
    for source_row in cast(list[dict[str, object]], summary["observation_shards"]):
        shard_name = str(source_row["path"])
        shard_path = directory / "shards" / shard_name
        if (
            Path(shard_name).name != shard_name
            or shard_path.is_symlink()
            or _file_sha256(shard_path) != source_row["sha256"]
        ):
            raise ValueError("active probe session shard differs")
        with np.load(shard_path, allow_pickle=False) as shard:
            frame_parts.append(shard["minimap_rgb"].copy())
            scheduled_parts.append(shard["scheduled_elapsed_ms"].copy())
            elapsed_parts.append(shard["frame_elapsed_ms"].copy())
            valid_parts.append(shard["screen_valid"].copy())
            allowed_parts.append(shard["operation_allowed"].copy())
            movement_parts.append(shard["movement_id"].copy())
            sent_parts.append(shard["movement_input_sent"].copy())
        opened_shards.append(
            {"session": directory.name, "basename": shard_name, "sha256": source_row["sha256"]}
        )
    frames = np.concatenate(frame_parts)
    scheduled = np.concatenate(scheduled_parts).astype(np.int64)
    elapsed = np.concatenate(elapsed_parts).astype(np.int64)
    screen_valid = np.concatenate(valid_parts).astype(bool)
    operation_allowed = np.concatenate(allowed_parts).astype(bool)
    movement_ids = np.concatenate(movement_parts).astype(np.int64)
    sent = np.concatenate(sent_parts).astype(bool)
    if (
        frames.ndim != 4
        or frames.shape[1:] != (128, 128, 3)
        or frames.dtype != np.uint8
        or scheduled.shape != (frames.shape[0],)
        or elapsed.shape != (frames.shape[0],)
        or screen_valid.shape != (frames.shape[0],)
        or operation_allowed.shape != (frames.shape[0],)
        or movement_ids.shape != (frames.shape[0],)
        or sent.shape != (frames.shape[0],)
        or not np.all(np.diff(elapsed) > 0)
    ):
        raise ValueError("active probe session arrays differ")
    event_lines = (directory / "pulses.jsonl").read_text(encoding="utf-8").splitlines()
    events: list[dict[str, object]] = []
    for line in event_lines:
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("active probe event is not an object")
        events.append(cast(dict[str, object], payload))
    pulse_events = [event for event in events if event.get("kind") == "pulse"]
    control_events = [event for event in events if event.get("kind") == "control"]
    if len(pulse_events) + len(control_events) != len(events):
        raise ValueError("active probe event kind differs")
    for event in pulse_events:
        if (
            str(event.get("direction")) not in _MOVEMENT_VECTORS
            or not all(
                isinstance(event.get(key), (int, float, bool))
                for key in ("press_ack_ms", "release_ack_ms")
            )
            or int(cast(int, event["release_ack_ms"])) < int(cast(int, event["press_ack_ms"]))
        ):
            raise ValueError("active probe pulse event differs")
    for event in control_events:
        if (
            not all(
                isinstance(event.get(key), (int, float, bool))
                for key in ("window_start_ms", "window_end_ms")
            )
            or int(cast(int, event["window_end_ms"]))
            <= int(cast(int, event["window_start_ms"]))
        ):
            raise ValueError("active probe control event differs")
    if (
        int(cast(int, summary.get("pulses_dispatched", len(pulse_events)))) != len(pulse_events)
        or int(cast(int, summary.get("control_windows_dispatched", len(control_events))))
        != len(control_events)
        or int(cast(int, summary["input_commands_sent"])) < 0
    ):
        raise ValueError("active probe event counts differ")

    cue_contract: dict[str, object] = {
        "color": contract["color"],
        "components": contract["components"],
    }
    exclusion = cast(
        tuple[int, int, int, int],
        tuple(map(int, cast(list[int], contract["excluded_ui_xyxy"]))),
    )
    interior, fixed = _probe_frame_candidates(frames, cue_contract, exclusion)
    analysis = screen_valid & operation_allowed
    resolved = np.asarray([len(row) == 1 for row in interior], dtype=bool)
    ambiguous = np.asarray([len(row) > 1 for row in interior], dtype=bool)
    localized = analysis & resolved
    unresolved = analysis & ~resolved
    analysis_frames = int(analysis.sum())
    analysis_coverage = int(localized.sum()) / analysis_frames if analysis_frames else 0.0
    valid_runs = _probe_run_lengths(localized)
    unresolved_runs = _probe_run_lengths(unresolved)
    frame_period_ms = int(cast(int, contract["frame_period_ms"]))
    components = cast(dict[str, object], contract["components"])
    maximum_jump = float(cast(float, components["maximum_pair_l1_distance"]))
    identity_switch_events = 0
    previous: tuple[float, float] | None = None
    for index in range(len(frames)):
        if not bool(analysis[index]):
            previous = None
            continue
        current = _probe_unique_position(interior[index])
        if current is None:
            previous = None
            continue
        if (
            previous is not None
            and abs(current[0] - previous[0]) + abs(current[1] - previous[1]) > maximum_jump
        ):
            identity_switch_events += 1
        previous = current

    measurement = cast(dict[str, object], contract["measurement"])
    gates = cast(dict[str, object], contract["gates_per_session"])
    direction_ids = {
        name: index for index, name in enumerate(cast(list[str], contract["movement_names"]))
    }
    event_rows: list[dict[str, object]] = []
    for event in events:
        row = _probe_event_metrics(
            event,
            direction_ids=direction_ids,
            observation_ms=int(cast(int, contract["observation_ms"])),
            gap_ms=int(cast(int, measurement["maximum_gap_to_analysis_frame_ms"])),
            minimum_projection=float(cast(float, gates["minimum_projection_pixels"])),
            times=elapsed,
            analysis=analysis,
            interior=interior,
            fixed=fixed,
            movement_ids=movement_ids,
            sent=sent,
        )
        row["session"] = directory.name
        event_rows.append(row)

    pulse_rows = [row for row in event_rows if row["kind"] == "pulse"]
    control_rows = [row for row in event_rows if row["kind"] == "control"]
    fates = Counter(str(row["fate"]) for row in pulse_rows)
    paired_rows = [row for row in pulse_rows if row["fate"] == "paired"]
    responsive_rows = [row for row in paired_rows if bool(row["responsive"])]
    paired_fraction = len(paired_rows) / len(pulse_rows) if pulse_rows else 0.0
    direction_correct_fraction = (
        len(responsive_rows) / len(paired_rows) if paired_rows else 0.0
    )
    paired_directions = Counter(str(row["direction"]) for row in paired_rows)
    pulse_projections = [
        float(cast(float, row["projection_pixels"]))
        for row in paired_rows
        if row["projection_pixels"] is not None
    ]
    control_displacements = [
        float(cast(float, row["displacement_pixels"]))
        for row in control_rows
        if row["fate"] == "paired" and row["displacement_pixels"] is not None
    ]
    fixed_rows = [row for row in pulse_rows if row["fixed_ui_displacement_pixels"] is not None]
    fixed_responsive = [row for row in fixed_rows if bool(row["fixed_ui_responsive"])]
    fixed_ui_responsive_fraction = len(fixed_responsive) / len(fixed_rows) if fixed_rows else 0.0
    pulse_median_projection = float(np.median(pulse_projections)) if pulse_projections else 0.0
    control_p95 = (
        float(np.quantile(np.asarray(control_displacements), 0.95))
        if control_displacements
        else 0.0
    )
    longest_valid_run_seconds = max(valid_runs, default=0) * frame_period_ms / 1000.0
    longest_unresolved_run_seconds = (
        max(unresolved_runs, default=0) * frame_period_ms / 1000.0
    )
    checks = {
        "dispatched_pulses": len(pulse_rows) >= int(cast(int, gates["minimum_dispatched_pulses"])),
        "paired_fraction": paired_fraction >= float(cast(float, gates["minimum_paired_fraction"])),
        "balanced_directions": len(paired_directions)
        >= int(cast(int, gates["minimum_balanced_directions"])),
        "direction_correct_fraction": direction_correct_fraction
        >= float(cast(float, gates["minimum_direction_correct_fraction"])),
        "pulse_median_projection": pulse_median_projection
        >= float(cast(float, gates["minimum_projection_pixels"])),
        "pulse_vs_control": (pulse_median_projection - control_p95)
        >= float(cast(float, gates["minimum_pulse_median_minus_control_p95_pixels"])),
        "fixed_ui_responsive_fraction": fixed_ui_responsive_fraction
        <= float(cast(float, gates["maximum_fixed_ui_responsive_fraction"])),
        "analysis_coverage": analysis_coverage
        >= float(cast(float, gates["minimum_analysis_coverage_fraction"])),
        "unknown_streak": longest_unresolved_run_seconds
        <= float(cast(float, gates["maximum_unknown_streak_seconds"])),
        "valid_run": longest_valid_run_seconds
        >= float(cast(float, gates["minimum_valid_run_seconds"])),
        "identity_switch_events": identity_switch_events
        <= int(cast(int, gates["maximum_identity_switch_events"])),
    }
    region = contract.get("free_movement_region")
    region_violation_frames = 0
    region_maximum_streak = 0
    if isinstance(region, dict):
        minimum_y = float(cast(float, region["minimum_y"]))
        maximum_y = float(cast(float, region["maximum_y"]))
        minimum_x = float(cast(float, region["minimum_x"]))
        maximum_x = float(cast(float, region["maximum_x"]))
        streak = 0
        for index in range(len(frames)):
            position = _probe_unique_position(interior[index]) if bool(localized[index]) else None
            if position is None or (
                minimum_y <= position[0] <= maximum_y
                and minimum_x <= position[1] <= maximum_x
            ):
                streak = 0
                continue
            streak += 1
            region_violation_frames += 1
            region_maximum_streak = max(region_maximum_streak, streak)
        checks["free_movement_region"] = region_maximum_streak <= int(
            cast(int, region["maximum_consecutive_violation_frames"])
        )
    maximum_frame_gap = contract.get("maximum_frame_gap_ms")
    if isinstance(maximum_frame_gap, (int, float)) and len(elapsed) > 1:
        checks["capture_stall"] = float(np.diff(elapsed).max()) <= float(maximum_frame_gap)
    maximum_press_drift = contract.get("maximum_press_start_drift_ms")
    if isinstance(maximum_press_drift, (int, float)):
        drifts = [
            int(cast(int, event["press_ack_ms"])) - int(cast(int, event["press_scheduled_ms"]))
            for event in pulse_events
            if "press_scheduled_ms" in event
        ]
        checks["press_start_drift"] = (
            max(drifts) <= float(maximum_press_drift) if drifts else True
        )
    metrics: dict[str, object] = {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "passed": all(checks.values()),
        "frames": int(frames.shape[0]),
        "analysis_frames": analysis_frames,
        "analysis_coverage": analysis_coverage,
        "ambiguous_frames": int((analysis & ambiguous).sum()),
        "longest_valid_run_seconds": longest_valid_run_seconds,
        "longest_unresolved_run_seconds": longest_unresolved_run_seconds,
        "identity_switch_events": identity_switch_events,
        "pulses_dispatched": len(pulse_rows),
        "control_windows": len(control_rows),
        "fates": dict(sorted(fates.items())),
        "paired": len(paired_rows),
        "paired_fraction": paired_fraction,
        "responsive_pairs": len(responsive_rows),
        "direction_correct_fraction": direction_correct_fraction,
        "paired_directions": dict(sorted(paired_directions.items())),
        "pulse_median_projection_pixels": pulse_median_projection,
        "control_p95_displacement_pixels": control_p95,
        "fixed_ui_pairs": len(fixed_rows),
        "fixed_ui_responsive_fraction": fixed_ui_responsive_fraction,
        "hard_stops": sum(bool(row["hard_stop"]) for row in pulse_rows),
        "checks": checks,
    }
    if isinstance(region, dict):
        metrics["region_violation_frames"] = region_violation_frames
        metrics["region_maximum_streak"] = region_maximum_streak
    return metrics, event_rows


def run_active_probe_audit(
    contract_path: Path, session_root: Path, output_dir: Path
) -> dict[str, object]:
    started = time.monotonic()
    contract = _load_bound_json(contract_path, "contract_sha256")
    directions = tuple(cast(list[str], contract.get("directions", [])))
    if (
        contract.get("schema_version") != "movement-active-probe-contract-v1"
        or contract.get("training_allowed") is not False
        or contract.get("test_allowed") is not False
        or directions != _PROBE_DIRECTIONS
    ):
        raise ValueError("active probe contract differs")
    if session_root.is_symlink() or not session_root.is_dir():
        raise ValueError("active probe session root differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("active probe output already exists")
    sessions_required = int(cast(int, contract["sessions_required"]))
    directories = sorted(
        path
        for path in session_root.iterdir()
        if path.is_dir() and (path / "summary.json").is_file()
    )
    if not directories or len(directories) > sessions_required:
        raise ValueError("active probe session count differs")
    opened_shards: list[dict[str, object]] = []
    session_metrics: dict[str, object] = {}
    event_rows: list[dict[str, object]] = []
    input_commands_sent = 0
    for directory in directories:
        metrics, rows = _probe_session_metrics(directory, contract, opened_shards)
        session_metrics[directory.name] = metrics
        event_rows.extend(rows)
        summary = _load_bound_json(directory / "summary.json", "summary_sha256")
        input_commands_sent += int(cast(int, summary["input_commands_sent"]))
    passed_sessions = [
        name
        for name, metrics in session_metrics.items()
        if bool(cast(dict[str, object], metrics)["passed"])
    ]
    found = len(directories)
    identity_control_verified = found >= sessions_required and len(passed_sessions) == found
    if identity_control_verified:
        status = "ACTIVE_PROBE_IDENTITY_AND_CONTROL_VERIFIED"
    elif found < sessions_required and len(passed_sessions) == found:
        status = "ACTIVE_PROBE_SINGLE_SESSION_ONLY"
    else:
        status = "ACTIVE_PROBE_GATES_FAILED"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        pulses_path = staging / "pulses.jsonl"
        with pulses_path.open("w", encoding="utf-8") as handle:
            for row in event_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        report: dict[str, object] = {
            "schema_version": "movement-active-probe-audit-report-v1",
            "status": status,
            "contract_sha256": contract["contract_sha256"],
            "sessions_required": sessions_required,
            "sessions_found": found,
            "session_metrics": session_metrics,
            "sessions_passed": sorted(passed_sessions),
            "identity_control_verified": identity_control_verified,
            "semantic_player_identity_verified": identity_control_verified,
            "multi_session_identity_verified": identity_control_verified,
            "navigation_available": False,
            "pulses_file_sha256": _file_sha256(pulses_path),
            "pulses_file_bytes": pulses_path.stat().st_size,
            "opened_shards": opened_shards,
            "device_input_commands_sent": input_commands_sent,
            "training_called": False,
            "test_frames_read": 0,
            "gpu_seconds": 0,
            "runtime_wall_seconds": time.monotonic() - started,
        }
        report["report_sha256"] = _object_sha256(report)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return report


_FORENSIC_NEAR_DUPLICATE_THRESHOLD = 0.5
_FORENSIC_BASE_REGION_Y = 110.0


def _green_only_candidates(
    frame: np.ndarray, contract: dict[str, object]
) -> list[tuple[float, float]]:
    rgb = frame.astype(np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    color = cast(dict[str, object], contract["color"])
    config = cast(dict[str, object], contract["components"])
    mask = (
        (green > int(cast(int, color["green_minimum"])))
        & (green - red > int(cast(int, color["green_red_margin"])))
        & (green - blue > int(cast(int, color["green_blue_margin"])))
    )
    green_size = cast(list[int], config["green_size"])
    green_extent = cast(list[int], config["green_extent"])
    positions: list[tuple[float, float]] = []
    for item in _mask_components(mask):
        if (
            green_size[0] <= item[0] <= green_size[1]
            and green_extent[0] <= item[3] <= green_extent[1]
            and green_extent[0] <= item[4] <= green_extent[1]
            and 5 < item[1] < 123
            and 5 < item[2] < 123
        ):
            positions.append((item[1], item[2]))
    return positions


def _forensic_delta(
    times: np.ndarray,
    positions: np.ndarray,
    start_ms: int,
    end_ms: int,
    gap_ms: int,
) -> dict[str, object] | None:
    start_index = int(np.searchsorted(times, start_ms, side="right")) - 1
    end_index = int(np.searchsorted(times, end_ms, side="right")) - 1
    if (
        start_index < 0
        or end_index < 0
        or int(times[start_index]) < start_ms - gap_ms
        or int(times[end_index]) < end_ms - gap_ms
    ):
        return None
    start_y, start_x = float(positions[start_index][0]), float(positions[start_index][1])
    end_y, end_x = float(positions[end_index][0]), float(positions[end_index][1])
    return {
        "start_ms": int(times[start_index]),
        "end_ms": int(times[end_index]),
        "start_frame": start_index,
        "end_frame": end_index,
        "dy": end_y - start_y,
        "dx": end_x - start_x,
    }


def _rank_auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    greater = 0.0
    for value in positive:
        for other in negative:
            if value > other:
                greater += 1.0
            elif value == other:
                greater += 0.5
    return greater / (len(positive) * len(negative))


def _distribution(values: list[float]) -> dict[str, object]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "p90": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def _load_forensic_session(
    directory: Path, contract: dict[str, object]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary = _load_bound_json(directory / "summary.json", "summary_sha256")
    if (
        summary.get("schema_version") != contract["session_schema_version"]
        or summary.get("contract_sha256") != contract["contract_sha256"]
        or summary.get("derived_roi_rgb_persisted") is not True
        or summary.get("raw_frames_persisted") is not False
    ):
        raise ValueError("active probe forensics session summary differs")
    frame_parts: list[np.ndarray] = []
    elapsed_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    allowed_parts: list[np.ndarray] = []
    movement_parts: list[np.ndarray] = []
    sent_parts: list[np.ndarray] = []
    opened: list[dict[str, object]] = []
    for source_row in cast(list[dict[str, object]], summary["observation_shards"]):
        shard_name = str(source_row["path"])
        shard_path = directory / "shards" / shard_name
        if (
            Path(shard_name).name != shard_name
            or shard_path.is_symlink()
            or _file_sha256(shard_path) != source_row["sha256"]
        ):
            raise ValueError("active probe forensics shard differs")
        with np.load(shard_path, allow_pickle=False) as shard:
            frame_parts.append(shard["minimap_rgb"].copy())
            elapsed_parts.append(shard["frame_elapsed_ms"].copy())
            valid_parts.append(shard["screen_valid"].copy())
            allowed_parts.append(shard["operation_allowed"].copy())
            movement_parts.append(shard["movement_id"].copy())
            sent_parts.append(shard["movement_input_sent"].copy())
        opened.append(
            {"session": directory.name, "basename": shard_name, "sha256": source_row["sha256"]}
        )
    frames = np.concatenate(frame_parts)
    times = np.concatenate(elapsed_parts).astype(np.int64)
    screen_valid = np.concatenate(valid_parts).astype(bool)
    operation_allowed = np.concatenate(allowed_parts).astype(bool)
    movement_ids = np.concatenate(movement_parts).astype(np.int64)
    sent = np.concatenate(sent_parts).astype(bool)
    if (
        frames.ndim != 4
        or frames.shape[1:] != (128, 128, 3)
        or frames.dtype != np.uint8
        or times.shape != (frames.shape[0],)
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("active probe forensics arrays differ")
    events: list[dict[str, object]] = []
    for line in (directory / "pulses.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("active probe forensics event is not an object")
        events.append(cast(dict[str, object], payload))
    bundle: dict[str, object] = {
        "frames": frames,
        "times": times,
        "screen_valid": screen_valid,
        "operation_allowed": operation_allowed,
        "movement_ids": movement_ids,
        "sent": sent,
        "events": events,
        "summary": summary,
    }
    return bundle, opened


def _forensic_indicators(
    pulse_hold_projections: list[float],
    idle_projections: list[float],
    control_displacements: list[float],
    tracked_motion: dict[str, object],
    green_motion: dict[str, object],
    coverage: dict[str, object],
) -> dict[str, object]:
    hold_abs = [abs(value) for value in pulse_hold_projections]
    idle_abs = [abs(value) for value in idle_projections]
    nonzero = sum(1 for value in pulse_hold_projections if value != 0.0)
    return {
        "pulse_hold_signal_over_idle_auc": _rank_auc(hold_abs, idle_abs),
        "pulse_hold_signal_over_control_auc": _rank_auc(hold_abs, control_displacements),
        "pulse_hold_nonzero_fraction": nonzero / len(pulse_hold_projections)
        if pulse_hold_projections
        else None,
        "pulse_hold_median_abs_pixels": float(np.median(hold_abs)) if hold_abs else None,
        "idle_median_abs_pixels": float(np.median(idle_abs)) if idle_abs else None,
        "tracked_blob_frame_step": tracked_motion,
        "green_only_blob_frame_step": green_motion,
        "effective_update_coverage": coverage,
    }


def run_active_probe_forensics(
    contract_path: Path, session_root: Path, output_dir: Path
) -> dict[str, object]:
    started = time.monotonic()
    contract = _load_bound_json(contract_path, "contract_sha256")
    directions = tuple(cast(list[str], contract.get("directions", [])))
    if (
        contract.get("schema_version") != "movement-active-probe-contract-v1"
        or contract.get("training_allowed") is not False
        or contract.get("test_allowed") is not False
        or directions != _PROBE_DIRECTIONS
    ):
        raise ValueError("active probe forensics contract differs")
    if session_root.is_symlink() or not session_root.is_dir():
        raise ValueError("active probe forensics session root differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("active probe forensics output already exists")
    sessions_required = int(cast(int, contract["sessions_required"]))
    directories = sorted(
        path
        for path in session_root.iterdir()
        if path.is_dir() and (path / "summary.json").is_file()
    )
    if not directories or len(directories) > sessions_required:
        raise ValueError("active probe forensics session count differs")
    measurement = cast(dict[str, object], contract["measurement"])
    gap_ms = int(cast(int, measurement["maximum_gap_to_analysis_frame_ms"]))
    observation_ms = int(cast(int, contract["observation_ms"]))
    inter_gap_ms = int(cast(int, contract["inter_pulse_gap_ms"]))
    cue_contract: dict[str, object] = {
        "color": contract["color"],
        "components": contract["components"],
    }
    exclusion = cast(
        tuple[int, int, int, int],
        tuple(map(int, cast(list[int], contract["excluded_ui_xyxy"]))),
    )
    duplicate_threshold = _FORENSIC_NEAR_DUPLICATE_THRESHOLD
    movement_names = list(cast(list[str], contract["movement_names"]))
    session_metrics: dict[str, object] = {}
    opened_shards: list[dict[str, object]] = []
    pooled_hold_projections: list[float] = []
    pooled_idle_projections: list[float] = []
    pooled_control_displacements: list[float] = []
    pooled_tracked_motion: dict[str, object] = {}
    pooled_green_motion: dict[str, object] = {}
    pooled_coverage: dict[str, object] = {}
    for directory in directories:
        bundle, opened = _load_forensic_session(directory, contract)
        opened_shards.extend(opened)
        frames = cast(np.ndarray, bundle["frames"])
        times = cast(np.ndarray, bundle["times"])
        events = cast(list[dict[str, object]], bundle["events"])
        movement_ids = cast(np.ndarray, bundle["movement_ids"])
        interior, _fixed = _probe_frame_candidates(frames, cue_contract, exclusion)
        tracked: list[tuple[float, float]] = []
        green_counts: list[int] = []
        green_sets: list[list[tuple[float, float]]] = []
        for index, frame in enumerate(frames):
            position = _probe_unique_position(interior[index])
            tracked.append(position if position is not None else (float("nan"), float("nan")))
            greens = _green_only_candidates(frame, cue_contract)
            green_sets.append(greens)
            green_counts.append(len(greens))
        tracked_array = np.asarray(tracked, dtype=np.float64)
        localized = ~np.isnan(tracked_array[:, 0])
        localized_times = times[localized]
        localized_positions = tracked_array[localized]
        tracked_steps = (
            np.hypot(
                np.diff(localized_positions[:, 0]), np.diff(localized_positions[:, 1])
            )
            if len(localized_positions) > 1
            else np.asarray([])
        )
        green_track: list[tuple[float, float] | None] = []
        previous: tuple[float, float] | None = None
        for greens in green_sets:
            if not greens:
                green_track.append(None)
                previous = None
                continue
            if previous is None:
                choice = greens[0]
            else:
                reference = previous
                choice = greens[0]
                best = abs(choice[0] - reference[0]) + abs(choice[1] - reference[1])
                for candidate in greens[1:]:
                    distance = abs(candidate[0] - reference[0]) + abs(
                        candidate[1] - reference[1]
                    )
                    if distance < best:
                        choice = candidate
                        best = distance
            green_track.append(choice)
            previous = choice
        green_steps: list[float] = []
        for index in range(1, len(green_track)):
            current = green_track[index]
            previous_green = green_track[index - 1]
            if current is None or previous_green is None:
                continue
            green_steps.append(
                math.hypot(current[0] - previous_green[0], current[1] - previous_green[1])
            )
        pulse_events = [event for event in events if event.get("kind") == "pulse"]
        control_events = [event for event in events if event.get("kind") == "control"]
        per_direction: dict[str, dict[str, object]] = {}
        command_hold_fractions: list[float] = []
        effective_updates: list[float] = []
        for event in pulse_events:
            direction = str(event["direction"])
            vector_y, vector_x = _MOVEMENT_VECTORS[direction]
            norm = math.hypot(vector_y, vector_x)
            direction_id = movement_names.index(direction)
            press_ack_ms = int(cast(int, event["press_ack_ms"]))
            release_ack_ms = int(cast(int, event["release_ack_ms"]))
            hold = _forensic_delta(
                localized_times, localized_positions, press_ack_ms, release_ack_ms, gap_ms
            )
            observation = _forensic_delta(
                localized_times,
                localized_positions,
                press_ack_ms,
                release_ack_ms + observation_ms,
                gap_ms,
            )
            idle = _forensic_delta(
                localized_times,
                localized_positions,
                press_ack_ms - inter_gap_ms - observation_ms,
                press_ack_ms - inter_gap_ms,
                gap_ms,
            )
            entry = per_direction.setdefault(
                direction,
                {
                    "pulses": 0,
                    "hold_projections": [],
                    "hold_displacements": [],
                    "observation_displacements": [],
                },
            )
            entry["pulses"] = int(cast(int, entry["pulses"])) + 1
            if hold is not None:
                projection = (
                    float(cast(float, hold["dy"])) * vector_y
                    + float(cast(float, hold["dx"])) * vector_x
                ) / norm
                pooled_hold_projections.append(projection)
                cast(list[float], entry["hold_projections"]).append(projection)
                cast(list[float], entry["hold_displacements"]).append(
                    math.hypot(float(cast(float, hold["dy"])), float(cast(float, hold["dx"])))
                )
                hold_index = np.flatnonzero((times >= press_ack_ms) & (times <= release_ack_ms))
                if len(hold_index):
                    commanded = movement_ids[hold_index] == direction_id
                    command_hold_fractions.append(float(commanded.mean()))
                    if len(hold_index) > 1:
                        diffs = np.abs(
                            np.diff(frames[hold_index].astype(np.int16), axis=0)
                        ).mean(axis=(1, 2, 3))
                        effective_updates.append(float((diffs >= duplicate_threshold).mean()))
            if observation is not None:
                cast(list[float], entry["observation_displacements"]).append(
                    math.hypot(
                        float(cast(float, observation["dy"])),
                        float(cast(float, observation["dx"])),
                    )
                )
            if idle is not None:
                pooled_idle_projections.append(
                    (float(cast(float, idle["dy"])) * vector_y
                     + float(cast(float, idle["dx"])) * vector_x) / norm
                )
        for event in control_events:
            delta = _forensic_delta(
                localized_times,
                localized_positions,
                int(cast(int, event["window_start_ms"])),
                int(cast(int, event["window_end_ms"])),
                gap_ms,
            )
            if delta is not None:
                pooled_control_displacements.append(
                    math.hypot(float(cast(float, delta["dy"])), float(cast(float, delta["dx"])))
                )
        mean_diffs = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
        gaps = np.diff(times)
        stalls = [
            {"start_ms": int(times[index]), "end_ms": int(times[index + 1])}
            for index in np.flatnonzero(gaps > 1000)
        ]
        base_frames = np.flatnonzero(localized & (tracked_array[:, 0] >= _FORENSIC_BASE_REGION_Y))
        per_direction_summary = {
            name: {
                "pulses": int(cast(int, entry["pulses"])),
                "hold_projection_median": float(
                    np.median(cast(list[float], entry["hold_projections"]))
                )
                if cast(list[float], entry["hold_projections"])
                else None,
                "hold_displacement_median": float(
                    np.median(cast(list[float], entry["hold_displacements"]))
                )
                if cast(list[float], entry["hold_displacements"])
                else None,
                "observation_displacement_median": float(
                    np.median(cast(list[float], entry["observation_displacements"]))
                )
                if cast(list[float], entry["observation_displacements"])
                else None,
            }
            for name, entry in sorted(per_direction.items())
        }
        hold_projection_values: list[float] = []
        for entry in per_direction.values():
            hold_projection_values.extend(cast(list[float], entry["hold_projections"]))
        session_metrics[directory.name] = {
            "frames": int(frames.shape[0]),
            "localized_frames": int(localized.sum()),
            "pulses": len(pulse_events),
            "control_windows": len(control_events),
            "per_direction": per_direction_summary,
            "hold_projection": _distribution(hold_projection_values),
            "tracked_frame_step": _distribution([float(value) for value in tracked_steps]),
            "green_only_frame_step": _distribution([float(value) for value in green_steps]),
            "green_only_candidates_per_frame": {
                "zero": int(sum(1 for count in green_counts if count == 0)),
                "one": int(sum(1 for count in green_counts if count == 1)),
                "many": int(sum(1 for count in green_counts if count > 1)),
            },
            "near_duplicate_fraction": float((mean_diffs < duplicate_threshold).mean())
            if len(mean_diffs)
            else 0.0,
            "frame_gap_ms": {
                "median": float(np.median(gaps)),
                "p90": float(np.quantile(gaps, 0.9)),
                "maximum": int(gaps.max()),
                "stalls_over_1000ms": stalls,
            },
            "command_hold_fraction": _distribution(command_hold_fractions),
            "effective_update_fraction_in_hold": _distribution(effective_updates),
            "base_region_frames": int(len(base_frames)),
            "base_region_first_ms": int(times[base_frames[0]]) if len(base_frames) else None,
            "trajectory_sample": [
                {
                    "ms": int(times[index]),
                    "y": float(tracked_array[index][0]),
                    "x": float(tracked_array[index][1]),
                }
                for index in range(0, len(times), max(1, len(times) // 12))
                if localized[index]
            ],
        }
        pooled_tracked_motion = _distribution([float(value) for value in tracked_steps])
        pooled_green_motion = _distribution([float(value) for value in green_steps])
        pooled_coverage = _distribution(effective_updates)
    report: dict[str, object] = {
        "schema_version": "movement-active-probe-forensics-report-v1",
        "contract_sha256": contract["contract_sha256"],
        "sessions_found": len(directories),
        "session_metrics": session_metrics,
        "pooled": {
            "hold_projection": _distribution(pooled_hold_projections),
            "idle_projection": _distribution(pooled_idle_projections),
            "control_displacement": _distribution(pooled_control_displacements),
            "indicators": _forensic_indicators(
                pooled_hold_projections,
                pooled_idle_projections,
                pooled_control_displacements,
                pooled_tracked_motion,
                pooled_green_motion,
                pooled_coverage,
            ),
        },
        "opened_shards": opened_shards,
        "device_input_commands_sent": 0,
        "training_called": False,
        "test_frames_read": 0,
        "gpu_seconds": 0,
        "runtime_wall_seconds": time.monotonic() - started,
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return report
