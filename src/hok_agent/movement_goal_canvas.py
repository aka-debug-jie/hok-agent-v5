from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np

from hok_agent.movement_mvp import (
    MOVEMENT_ACTIONS,
    StageAMovement,
    rule_movement_in_range,
    stage_c_arena,
    stage_c_scenarios,
    to_arena_action,
)
from hok_agent.rich_arena import ArenaConfig, RichPixelArena, move_action, wait_action


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(path: Path, field: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid goal canvas JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"goal canvas JSON root differs: {path.name}")
    payload = cast(dict[str, object], value)
    supplied = str(payload.pop(field, ""))
    calculated = _object_sha256(payload)
    payload[field] = supplied
    if supplied != calculated:
        raise ValueError(f"goal canvas JSON hash differs: {path.name}")
    return payload


def _point(position: tuple[int, int]) -> tuple[int, int]:
    return 12 + round(position[0] * 104 / 14), 32 + (position[1] - 2) * 32


def render_goal_minimap(
    position: tuple[int, int],
    goal: tuple[int, int],
    render_seed: int,
    marker: dict[str, object],
) -> np.ndarray:
    rng = np.random.default_rng(render_seed)
    frame = np.empty((128, 128, 3), dtype=np.uint8)
    frame[:] = np.asarray((22, 38, 28), dtype=np.uint8)
    noise = rng.integers(-4, 5, size=(128, 128, 1), dtype=np.int16)
    frame = cast(np.ndarray, np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    frame[30:99:32, 8:120] = (48, 58, 42)
    frame[8:120, 10:118:26] = (42, 52, 38)
    player_x, player_y = _point(position)
    frame[player_y - 4 : player_y + 5, player_x - 4 : player_x + 5] = (55, 195, 235)
    goal_x, goal_y = _point(goal)
    radius = int(cast(int, marker["radius"]))
    thickness = int(cast(int, marker["thickness"]))
    color = np.asarray(cast(list[int], marker["rgb"]), dtype=np.uint8)
    for y in range(goal_y - radius, goal_y + radius + 1):
        for x in range(goal_x - radius, goal_x + radius + 1):
            squared = (x - goal_x) ** 2 + (y - goal_y) ** 2
            if (radius - thickness) ** 2 <= squared <= radius**2:
                frame[y, x] = color
    return frame


def goal_canvas_geometry_movement(frame: np.ndarray) -> StageAMovement:
    player_y, player_x = np.where(np.all(frame == (55, 195, 235), axis=2))
    goal_y, goal_x = np.where(np.all(frame == (245, 225, 45), axis=2))
    if not len(player_x) or not len(goal_x):
        raise ValueError("goal canvas geometry requires player and target marker")
    delta_x = round((float(goal_x.mean()) - float(player_x.mean())) / (104 / 14))
    delta_y = round((float(goal_y.mean()) - float(player_y.mean())) / 32)
    return rule_movement_in_range((0, 0), (delta_x, delta_y))


def _warmup(position: tuple[int, int], goal: tuple[int, int]) -> tuple[str, str]:
    for outgoing, incoming, dx in (("east", "west", 1), ("west", "east", -1)):
        destination = (position[0] + dx, position[1])
        if 0 <= destination[0] < 15 and destination != goal:
            return outgoing, incoming
    raise ValueError("goal canvas warmup movement is unavailable")


def _causal_clip(
    position: tuple[int, int], goal: tuple[int, int], render_seed: int, marker: dict[str, object]
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    arena = RichPixelArena(
        ArenaConfig(
            max_ticks=32,
            blue_start=position,
            red_start=(13, 2),
            tower_damage=0,
            minion_damage=0,
        )
    )
    arena.reset(render_seed)
    outgoing, incoming = _warmup(position, goal)
    frames: list[np.ndarray] = []

    def observe(count: int) -> None:
        for _ in range(count):
            raw = cast(dict[str, int], arena.observe("blue")["self_position"])
            frames.append(render_goal_minimap((raw["x"], raw["y"]), goal, render_seed, marker))

    observe(4)
    arena.step(move_action(outgoing), wait_action())
    observe(4)
    arena.step(move_action(incoming), wait_action())
    observe(4)
    arena.step(move_action(outgoing), wait_action())
    observe(2)
    arena.step(move_action(incoming), wait_action())
    observe(2)
    before_raw = cast(dict[str, int], arena.observe("blue")["self_position"])
    before = (before_raw["x"], before_raw["y"])
    action = rule_movement_in_range(before, goal)
    arena.step(to_arena_action(action), wait_action())
    after_raw = cast(dict[str, int], arena.observe("blue")["self_position"])
    return np.stack(frames), before, (after_raw["x"], after_raw["y"])


def materialize_goal_canvas_overfit32(
    contract_path: Path, goal_canvas_report_path: Path, output_dir: Path
) -> dict[str, object]:
    contract = _load_bound(contract_path, "contract_sha256")
    prior = _load_bound(goal_canvas_report_path, "report_sha256")
    if (
        contract.get("schema_version") != "movement-goal-canvas-overfit32-contract-v1"
        or prior.get("status")
        != "GOAL_CANVAS_GENERATION_PASSED_SELF_LOCALIZATION_UNRESOLVED"
        or contract.get("goal_canvas_contract_sha256") != prior.get("contract_sha256")
        or contract.get("goal_canvas_report_sha256") != prior.get("report_sha256")
        or contract.get("test_allowed") is not False
        or contract.get("formal_training_allowed") is not False
        or contract.get("r2_allowed") is not False
        or contract.get("device_input_allowed") is not False
    ):
        raise ValueError("goal canvas overfit32 contract binding differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("goal canvas overfit32 output already exists")
    direction_positions: dict[StageAMovement, tuple[tuple[int, int], tuple[int, int]]] = {
        "N": ((7, 4), (7, 2)),
        "S": ((7, 2), (7, 4)),
        "W": ((8, 3), (6, 3)),
        "E": ((6, 3), (8, 3)),
        "NW": ((8, 4), (6, 2)),
        "NE": ((6, 4), (8, 2)),
        "SW": ((8, 2), (6, 4)),
        "SE": ((6, 2), (8, 4)),
    }
    marker = cast(dict[str, object], contract["marker"])
    samples: list[tuple[StageAMovement, tuple[int, int], tuple[int, int], int]] = []
    for replicate in range(int(cast(int, contract["direction_samples"]))):
        for action in MOVEMENT_ACTIONS[1:]:
            current, goal = direction_positions[action]
            samples.append((action, current, goal, 100 + replicate * 17 + len(samples)))
    stop_goals = ((7, 2), (8, 3), (7, 4), (6, 3)) * 2
    samples.extend(
        ("STOP", (7, 3), goal, 1000 + index) for index, goal in enumerate(stop_goals)
    )
    clips: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    changed_windows = 0
    for index, (expected, current, goal, render_seed) in enumerate(samples):
        clip, before, after = _causal_clip(current, goal, render_seed, marker)
        action = rule_movement_in_range(before, goal)
        if action != expected or (before != after) != (expected != "STOP"):
            raise ValueError("goal canvas overfit32 action causality differs")
        clips.append(clip)
        labels.append(MOVEMENT_ACTIONS.index(expected))
        episode_ids.append(f"goal-canvas-overfit-{index:02d}")
        changed_windows += int(
            any(not np.array_equal(clip[frame], clip[frame + 1]) for frame in range(15))
        )
    counterfactual_changes = 0
    for action in MOVEMENT_ACTIONS[1:]:
        center = direction_positions[action][0]
        first_goal = direction_positions[action][1]
        second_goal = center
        first = render_goal_minimap(center, first_goal, 77, marker)
        second = render_goal_minimap(center, second_goal, 77, marker)
        counterfactual_changes += int(
            not np.array_equal(first, second)
            and rule_movement_in_range(center, first_goal)
            != rule_movement_in_range(center, second_goal)
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        dataset_path = staging / "overfit32.npz"
        with dataset_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                rgb_sequence=np.stack(clips).astype(np.uint8),
                label=np.asarray(labels, dtype=np.int64),
                episode_id=np.asarray(episode_ids),
                action=np.asarray([MOVEMENT_ACTIONS[label] for label in labels]),
                contract_sha256=np.asarray([contract["contract_sha256"]]),
                goal_canvas_report_sha256=np.asarray([prior["report_sha256"]]),
            )
            handle.flush()
            os.fsync(handle.fileno())
        counts = {action: labels.count(index) for index, action in enumerate(MOVEMENT_ACTIONS)}
        report: dict[str, object] = {
            "schema_version": "movement-goal-canvas-overfit32-data-report-v1",
            "status": "PASSED",
            "contract_sha256": contract["contract_sha256"],
            "goal_canvas_report_sha256": prior["report_sha256"],
            "dataset_sha256": _file_sha256(dataset_path),
            "samples": len(samples),
            "frames_per_sample": 16,
            "class_counts": counts,
            "unique_episode_ids": len(set(episode_ids)),
            "windows_with_state_change": changed_windows,
            "counterfactual_goal_changes": counterfactual_changes,
            "model_input": contract["actor_input"],
            "structured_coordinates_in_model_input": False,
            "direction_arrow_in_model_input": False,
            "action_executed_after_window": True,
            "real_rgb_training_frames": 0,
            "test_frames_read": 0,
            "formal_training_allowed": False,
            "r2_allowed": False,
            "device_input_commands_sent": 0,
        }
        report["report_sha256"] = _object_sha256(report)
        path = staging / "report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return report


def _teacher_episode(
    scenario: dict[str, object],
    maximum_steps: int,
    step_duration_ms: int,
    stop_confirmation_steps: int,
    marker: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    goal = cast(tuple[int, int], tuple(cast(list[int], scenario["goal"])))
    render_seed = int(cast(int, scenario["render_seed"]))
    arena = stage_c_arena(scenario, maximum_steps, navigation_only=True)
    frames: list[np.ndarray] = []
    labels: list[int] = []
    timestamps: list[int] = []
    stop_streak = 0
    for step in range(maximum_steps):
        observation = arena.observe("blue")
        position_raw = cast(dict[str, int], observation["self_position"])
        position = (position_raw["x"], position_raw["y"])
        frames.append(render_goal_minimap(position, goal, render_seed, marker))
        timestamps.append(step * step_duration_ms)
        action = rule_movement_in_range(position, goal)
        labels.append(MOVEMENT_ACTIONS.index(action))
        arena.step(to_arena_action(action), wait_action())
        stop_streak = stop_streak + 1 if action == "STOP" else 0
        if stop_streak >= stop_confirmation_steps:
            return (
                np.stack(frames).astype(np.uint8),
                np.asarray(labels, dtype=np.int64),
                np.asarray(timestamps, dtype=np.int64),
                True,
            )
    return (
        np.stack(frames).astype(np.uint8),
        np.asarray(labels, dtype=np.int64),
        np.asarray(timestamps, dtype=np.int64),
        False,
    )


def materialize_goal_canvas_trajectories(
    contract_path: Path,
    overfit_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract = _load_bound(contract_path, "contract_sha256")
    try:
        overfit = json.loads(overfit_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("goal canvas overfit report is invalid") from exc
    if (
        contract.get("schema_version") != "movement-goal-canvas-stage-c-contract-v1"
        or _file_sha256(overfit_report_path) != contract.get("overfit_report_sha256")
        or overfit.get("status") != "PASSED"
        or overfit.get("dataset_sha256") != contract.get("overfit_dataset_sha256")
        or overfit.get("diagnostic_checkpoint_reusable_for_formal_training") is not False
        or contract.get("diagnostic_checkpoint_loaded") is not False
        or contract.get("test_allowed") is not False
        or contract.get("holdout_allowed") is not False
        or contract.get("r2_allowed") is not False
        or contract.get("device_input_allowed") is not False
    ):
        raise ValueError("goal canvas stage C contract binding differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("goal canvas stage C output already exists")
    stage = cast(dict[str, object], contract["stage_c"])
    maximum_steps = int(cast(int, stage["maximum_episode_steps"]))
    step_duration_ms = int(cast(int, stage["step_duration_ms"]))
    stop_steps = int(cast(int, stage["stop_confirmation_steps"]))
    marker = cast(dict[str, object], contract["marker"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        episodes_dir = staging / "episodes"
        episodes_dir.mkdir()
        counts: Counter[str] = Counter()
        successes: Counter[str] = Counter()
        label_counts = {
            split: {action: 0 for action in MOVEMENT_ACTIONS} for split in ("train", "dev")
        }
        episodes: list[dict[str, object]] = []
        for scenario in stage_c_scenarios():
            split = str(scenario["split"])
            basename = f"{split}-{counts[split]:03d}.npz"
            frames, labels, timestamps, success = _teacher_episode(
                scenario, maximum_steps, step_duration_ms, stop_steps, marker
            )
            path = episodes_dir / basename
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    frames=frames,
                    labels=labels,
                    frame_timestamps_ms=timestamps,
                    window_end=np.arange(len(labels), dtype=np.int64),
                )
                handle.flush()
                os.fsync(handle.fileno())
            for label in labels.tolist():
                label_counts[split][MOVEMENT_ACTIONS[int(label)]] += 1
            episodes.append(
                {
                    **scenario,
                    "basename": basename,
                    "steps": len(labels),
                    "supervised_windows": len(labels),
                    "teacher_success": success,
                    "artifact_sha256": _file_sha256(path),
                }
            )
            counts[split] += 1
            successes[split] += int(success)
        overlap = len(
            {str(row["scenario_id"]) for row in episodes if row["split"] == "train"}
            & {str(row["scenario_id"]) for row in episodes if row["split"] == "dev"}
        )
        expected = {
            "train": int(cast(int, stage["train_episodes"])),
            "dev": int(cast(int, stage["dev_episodes"])),
        }
        passed = (
            dict(counts) == expected
            and dict(successes) == expected
            and overlap == 0
            and all(
                sum(
                    row["split"] == split and row["initial_action"] == action
                    for row in episodes
                )
                == required
                for split, required in (("train", 8), ("dev", 3))
                for action in MOVEMENT_ACTIONS[1:]
            )
        )
        manifest_without_hash: dict[str, object] = {
            "schema_version": "movement-mvp-stage-c-trajectories-v0",
            "contract_sha256": contract["contract_sha256"],
            "config_sha256": _file_sha256(contract_path),
            "step_duration_ms": step_duration_ms,
            "sequence_frames": int(cast(int, stage["sequence_frames"])),
            "navigation_only": True,
            "stop_confirmation_steps": stop_steps,
            "actor_input": contract["actor_input"],
            "diagnostic_checkpoint_loaded": False,
            "episodes": episodes,
        }
        manifest = {
            **manifest_without_hash,
            "manifest_sha256": _object_sha256(manifest_without_hash),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report: dict[str, object] = {
            "schema_version": "movement-goal-canvas-stage-c-data-report-v1",
            "status": "PASSED" if passed else "FAILED",
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "episode_counts": dict(counts),
            "teacher_successes": dict(successes),
            "label_counts": label_counts,
            "scenario_overlap": overlap,
            "actor_input": contract["actor_input"],
            "frames_stored_once_per_episode": True,
            "windows_stored_as_indices": True,
            "diagnostic_checkpoint_loaded": False,
            "real_rgb_training_frames": 0,
            "test_frames_read": 0,
            "holdout_opened": False,
            "r2_allowed": False,
            "device_input_commands_sent": 0,
        }
        report["report_sha256"] = _object_sha256(report)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not passed:
            raise ValueError("goal canvas stage C trajectory gate failed")
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            staging.rmdir()
        raise
    return report
