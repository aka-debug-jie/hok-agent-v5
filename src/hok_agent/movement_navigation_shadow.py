from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
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
)


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
