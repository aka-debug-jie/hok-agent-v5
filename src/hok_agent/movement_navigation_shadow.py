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
