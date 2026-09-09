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
    _movement_command,
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


CHANGE_MODEL_ACTIONS = MOVEMENT_ACTIONS[1:]
CHANGE_EVENTS = ("macro_goal_version_changed", "stuck_recovery", "death_respawn_reset")


def change_only_route(
    previous_action: StageAMovement,
    goal_frame: np.ndarray | None,
    event: str,
    *,
    terminal: bool = False,
) -> dict[str, object]:
    """Apply a geometry teacher only at change events; runtime models replace that teacher."""
    if terminal:
        return {"model_invoked": False, "model_target": None, "applied_action": "STOP",
                "executor_command": _movement_command(previous_action, "STOP"),
                "owner": "deterministic_router", "reason": "terminal"}
    if goal_frame is None:
        return {"model_invoked": False, "model_target": None, "applied_action": "STOP",
                "executor_command": _movement_command(previous_action, "STOP"),
                "owner": "deterministic_router", "reason": "unknown_goal"}
    desired = goal_canvas_geometry_movement(goal_frame)
    if desired == "STOP":
        return {"model_invoked": False, "model_target": None, "applied_action": "STOP",
                "executor_command": _movement_command(previous_action, "STOP"),
                "owner": "deterministic_router", "reason": "goal_reached"}
    if event not in CHANGE_EVENTS:
        return {"model_invoked": False, "model_target": None,
                "applied_action": previous_action,
                "executor_command": _movement_command(previous_action, previous_action),
                "owner": "deterministic_executor", "reason": "persist"}
    return {"model_invoked": True, "model_target": desired, "applied_action": desired,
            "executor_command": _movement_command(previous_action, desired),
            "owner": "change_policy", "reason": event}


def run_change_only_contract(
    contract_path: Path, persistence_report_path: Path, output_dir: Path,
) -> dict[str, object]:
    contract = _load_bound(contract_path, "contract_sha256")
    persistence = _load_bound(persistence_report_path, "report_sha256")
    expected = {
        "schema_version": "movement-change-only-contract-v1",
        "persistence_report_file_sha256": _file_sha256(persistence_report_path),
        "step_duration_ms": 100,
        "model_action_order": list(CHANGE_MODEL_ACTIONS),
        "model_invocation_events": list(CHANGE_EVENTS),
        "persistence_owner": "deterministic_executor",
        "stop_owner": "deterministic_router",
        "actor_input": "16_frame_rgb_with_hollow_macro_goal_ring",
        "previous_action_is_actor_input": False,
        "unknown_goal_action": "STOP", "goal_reached_action": "STOP",
        "terminal_action": "STOP", "model_training_allowed": False,
        "checkpoint_promotion_allowed": False, "device_input_allowed": False,
        "video_dev_allowed": False, "video_test_allowed": False,
    }
    unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if (unsigned != expected
            or persistence.get("status") != "DETERMINISTIC_DIRECTION_PERSISTENCE_EXACT"):
        raise ValueError("change-only contract differs")
    if output_dir.exists():
        raise ValueError("change-only output exists")
    marker = {"radius": 7, "thickness": 2, "rgb": [245, 225, 45]}
    direction_positions: dict[StageAMovement, tuple[tuple[int, int], tuple[int, int]]] = {
        "N": ((7, 4), (7, 2)), "S": ((7, 2), (7, 4)),
        "W": ((8, 3), (6, 3)), "E": ((6, 3), (8, 3)),
        "NW": ((8, 4), (6, 2)), "NE": ((6, 4), (8, 2)),
        "SW": ((8, 2), (6, 4)), "SE": ((6, 2), (8, 4)),
    }
    opposite: dict[StageAMovement, StageAMovement] = {
        "N": "S", "S": "N", "W": "E", "E": "W",
        "NW": "SE", "NE": "SW", "SW": "NE", "SE": "NW",
    }
    changes: list[dict[str, object]] = []
    keeps: list[dict[str, object]] = []
    same_direction_changes: list[dict[str, object]] = []
    for index, action in enumerate(CHANGE_MODEL_ACTIONS):
        position, goal = direction_positions[action]
        frame = render_goal_minimap(position, goal, 900 + index, marker)
        change = change_only_route(opposite[action], frame, "macro_goal_version_changed")
        keep = change_only_route(action, frame, "none")
        same = change_only_route(action, frame, "macro_goal_version_changed")
        if (change["model_target"] != action or change["executor_command"] != "MOVE"
                or keep["applied_action"] != action or keep["executor_command"] != "KEEP"
                or same["model_target"] != action or same["executor_command"] != "KEEP"):
            raise ValueError("change-only direction case differs")
        frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
        changes.append({"expected": action, "frame_sha256": frame_hash, **change})
        keeps.append({"expected": action, "frame_sha256": frame_hash, **keep})
        same_direction_changes.append({"expected": action, "frame_sha256": frame_hash, **same})
    reached = render_goal_minimap((7, 3), (7, 3), 999, marker)
    stops = {
        "goal_reached": change_only_route("E", reached, "macro_goal_version_changed"),
        "unknown_goal": change_only_route("E", None, "macro_goal_version_changed"),
        "terminal": change_only_route("E", reached, "none", terminal=True),
    }
    if any(row["applied_action"] != "STOP" or row["model_invoked"]
           for row in stops.values()):
        raise ValueError("change-only stop ownership differs")
    report: dict[str, object] = {
        "schema_version": "movement-change-only-contract-report-v1",
        "contract_sha256": contract["contract_sha256"],
        "persistence_report_file_sha256": _file_sha256(persistence_report_path),
        "movement_goal_canvas_source_sha256": _file_sha256(Path(__file__)),
        "movement_runtime_source_sha256": _file_sha256(Path(__file__).with_name("movement_mvp.py")),
        "status": "CHANGE_ONLY_MOVEMENT_CONTRACT_PASSED",
        "model_action_order": list(CHANGE_MODEL_ACTIONS),
        "change_cases": changes, "persist_cases": keeps,
        "same_direction_goal_change_cases": same_direction_changes, "stop_cases": stops,
        "model_invocation_count": len(changes) + len(same_direction_changes),
        "persistence_count": len(keeps), "router_stop_count": len(stops),
        "model_outputs_stop": False, "previous_action_is_actor_input": False,
        "model_runs": 0, "training_allowed": False, "checkpoint_allowed": False,
        "input_commands_sent": 0, "video_dev_opened": False, "video_test_opened": False,
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".movement-change-only-", dir=output_dir.parent))
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return report


_CHANGE_VECTOR: dict[StageAMovement, tuple[int, int]] = {
    "N": (0, -2), "S": (0, 2), "W": (-2, 0), "E": (2, 0),
    "NW": (-2, -2), "NE": (2, -2), "SW": (-2, 2), "SE": (2, 2),
}


def _change_geometry(
    action: StageAMovement, x_shift: int, horizontal_y: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    y = (4 if action in {"N", "NW", "NE"}
         else 2 if action in {"S", "SW", "SE"}
         else 3 if horizontal_y is None else horizontal_y)
    position = (7 + x_shift, y)
    dx, dy = _CHANGE_VECTOR[action]
    return position, (position[0] + dx, position[1] + dy)


def _old_change_goal(
    position: tuple[int, int], desired: StageAMovement, offset: int,
) -> tuple[StageAMovement, tuple[int, int]]:
    for step in range(1, len(CHANGE_MODEL_ACTIONS) + 1):
        candidate = CHANGE_MODEL_ACTIONS[(offset + step) % len(CHANGE_MODEL_ACTIONS)]
        if candidate == desired:
            continue
        dx, dy = _CHANGE_VECTOR[candidate]
        goal = (position[0] + dx, position[1] + dy)
        if 0 <= goal[0] < 15 and 2 <= goal[1] <= 4:
            return candidate, goal
    raise ValueError("change event has no valid old goal")


def _change_event_samples(
    split: str, episodes: int, events_per_episode: int, marker: dict[str, object], seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    clips: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    rows: list[dict[str, object]] = []
    for episode in range(episodes):
        episode_id = f"change-{split}-{episode:03d}"
        for event in range(events_per_episode):
            label = (episode * events_per_episode + event) % len(CHANGE_MODEL_ACTIONS)
            action = CHANGE_MODEL_ACTIONS[label]
            position, new_goal = _change_geometry(action, (episode % 5) - 2)
            old_action, old_goal = _old_change_goal(position, action, episode + event)
            render_seed = seed + episode * 101 + event * 19
            frames = [
                render_goal_minimap(position, old_goal, render_seed + frame, marker)
                for frame in range(15)
            ]
            frames.append(render_goal_minimap(position, new_goal, render_seed + 15, marker))
            clip = np.stack(frames).astype(np.uint8)
            old_geometry = [goal_canvas_geometry_movement(frame) for frame in clip[:-1]]
            if (goal_canvas_geometry_movement(clip[-1]) != action
                    or any(value != old_action for value in old_geometry)
                    or old_action == action):
                raise ValueError("change event geometry differs")
            alternative, alternative_goal = _old_change_goal(position, action, episode + event + 3)
            counterfactual = render_goal_minimap(
                position, alternative_goal, render_seed + 15, marker
            )
            if (alternative == action or np.array_equal(counterfactual, clip[-1])
                    or goal_canvas_geometry_movement(counterfactual) == action):
                raise ValueError("change event counterfactual differs")
            clips.append(clip)
            labels.append(label)
            episode_ids.append(episode_id)
            rows.append({
                "sample_id": f"{episode_id}-{event}", "episode_id": episode_id,
                "event_id": event, "goal_version_before": event,
                "goal_version_after": event + 1, "position": list(position),
                "old_goal": list(old_goal), "new_goal": list(new_goal),
                "old_direction": old_action, "target_direction": action,
                "render_seed": render_seed,
                "rgb_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
            })
    return (np.stack(clips), np.asarray(labels, dtype=np.int64),
            np.asarray(episode_ids), rows)


def _change_event_samples_position_v2(
    split: str, episodes: int, marker: dict[str, object], seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    x_values = (-1, 0, 1) if split == "train" else (-2, 2)
    horizontal_y = (2, 3, 4)
    occurrences = dict.fromkeys(CHANGE_MODEL_ACTIONS, 0)
    clips: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    rows: list[dict[str, object]] = []
    for episode in range(episodes):
        episode_id = f"change-position-{split}-{episode:03d}"
        for event in range(4):
            label = (episode * 4 + event) % len(CHANGE_MODEL_ACTIONS)
            action = CHANGE_MODEL_ACTIONS[label]
            occurrence = occurrences[action]
            occurrences[action] += 1
            row_y = horizontal_y[occurrence % len(horizontal_y)] if action in {"E", "W"} else None
            position, new_goal = _change_geometry(
                action, x_values[occurrence % len(x_values)], row_y
            )
            old_action, old_goal = _old_change_goal(position, action, episode + event)
            render_seed = seed + episode * 103 + event * 23
            frames = [render_goal_minimap(position, old_goal, render_seed + i, marker)
                      for i in range(15)]
            frames.append(render_goal_minimap(position, new_goal, render_seed + 15, marker))
            clip = np.stack(frames).astype(np.uint8)
            old_geometry = [goal_canvas_geometry_movement(frame) for frame in clip[:-1]]
            if (goal_canvas_geometry_movement(clip[-1]) != action
                    or any(value != old_action for value in old_geometry)
                    or old_action == action):
                raise ValueError("position-v2 change geometry differs")
            alternative, alternative_goal = _old_change_goal(
                position, action, episode + event + 3
            )
            counterfactual = render_goal_minimap(
                position, alternative_goal, render_seed + 15, marker
            )
            if (alternative == action or np.array_equal(counterfactual, clip[-1])
                    or goal_canvas_geometry_movement(counterfactual) == action):
                raise ValueError("position-v2 counterfactual differs")
            clips.append(clip)
            labels.append(label)
            episode_ids.append(episode_id)
            rows.append({
                "sample_id": f"{episode_id}-{event}", "episode_id": episode_id,
                "event_id": event, "position_group": position[0],
                "position": list(position), "old_goal": list(old_goal),
                "new_goal": list(new_goal), "old_direction": old_action,
                "target_direction": action, "render_seed": render_seed,
                "rgb_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
            })
    return (np.stack(clips), np.asarray(labels, dtype=np.int64),
            np.asarray(episode_ids), rows)


def validate_change_event_dataset(output_dir: Path) -> dict[str, object]:
    manifest = _load_bound(output_dir / "manifest.json", "manifest_sha256")
    dataset_path = output_dir / "change-events.npz"
    if dataset_path.is_symlink() or _file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("change event dataset hash differs")
    source_sets: dict[str, set[str]] = {}
    summaries: dict[str, object] = {}
    all_hashes: list[str] = []
    with np.load(dataset_path, allow_pickle=False) as data:
        for split, samples, episodes in (("train", 256, 64), ("dev", 96, 24)):
            clips = data[f"{split}_rgb"]
            labels = data[f"{split}_label"]
            episode_ids = data[f"{split}_episode_id"]
            if (clips.shape != (samples, 16, 128, 128, 3) or clips.dtype != np.uint8
                    or labels.shape != (samples,) or episode_ids.shape != (samples,)):
                raise ValueError(f"{split} change event arrays differ")
            counts = np.bincount(labels, minlength=8)
            if not np.all(counts == samples // 8):
                raise ValueError(f"{split} change event labels are not balanced")
            for clip, label in zip(clips, labels, strict=True):
                action = CHANGE_MODEL_ACTIONS[int(label)]
                old_actions = {goal_canvas_geometry_movement(frame) for frame in clip[:-1]}
                if (goal_canvas_geometry_movement(clip[-1]) != action
                        or len(old_actions) != 1 or action in old_actions):
                    raise ValueError(f"{split} change event RGB target differs")
                all_hashes.append(hashlib.sha256(clip.tobytes()).hexdigest())
            source_sets[split] = set(map(str, episode_ids.tolist()))
            if len(source_sets[split]) != episodes:
                raise ValueError(f"{split} change event episode groups differ")
            summaries[split] = {
                "samples": samples, "episodes": episodes,
                "class_counts": dict(zip(CHANGE_MODEL_ACTIONS, counts.tolist(), strict=True)),
            }
    if source_sets["train"] & source_sets["dev"] or len(set(all_hashes)) != 352:
        raise ValueError("change event split leakage or duplicate clips")
    return {"status": "CHANGE_EVENT_DATASET_VALIDATED", "splits": summaries,
            "episode_overlap": 0, "unique_clips": 352, "stop_in_dataset": False,
            "training_allowed": False}


def materialize_change_event_dataset(
    contract_path: Path, change_report_path: Path, output_dir: Path,
    *, verify_only: bool = False,
) -> dict[str, object]:
    if verify_only:
        return validate_change_event_dataset(output_dir)
    contract = _load_bound(contract_path, "contract_sha256")
    change = _load_bound(change_report_path, "report_sha256")
    if (
        contract.get("schema_version") != "movement-change-event-dataset-contract-v1"
        or contract.get("change_contract_report_file_sha256") != _file_sha256(change_report_path)
        or change.get("status") != "CHANGE_ONLY_MOVEMENT_CONTRACT_PASSED"
        or contract.get("action_order") != list(CHANGE_MODEL_ACTIONS)
        or contract.get("sequence_frames") != 16 or contract.get("old_goal_frames") != 15
        or contract.get("new_goal_frames") != 1 or contract.get("step_duration_ms") != 100
        or contract.get("events_per_episode") != 4
        or contract.get("train_episodes") != 64 or contract.get("dev_episodes") != 24
        or contract.get("train_samples") != 256 or contract.get("dev_samples") != 96
        or contract.get("previous_action_is_actor_input") is not False
        or contract.get("stop_in_dataset") is not False
        or contract.get("model_training_allowed") is not False
        or contract.get("device_input_allowed") is not False
        or contract.get("video_dev_allowed") is not False
        or contract.get("video_test_allowed") is not False
    ):
        raise ValueError("change event dataset contract differs")
    if output_dir.exists():
        raise ValueError("change event dataset output exists")
    marker = cast(dict[str, object], contract["marker"])
    train = _change_event_samples("train", 64, 4, marker, 10_000)
    dev = _change_event_samples("dev", 24, 4, marker, 90_000)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".movement-change-events-", dir=output_dir.parent))
    dataset_path = staging / "change-events.npz"
    with dataset_path.open("wb") as handle:
        np.savez_compressed(
            handle, train_rgb=train[0], train_label=train[1], train_episode_id=train[2],
            dev_rgb=dev[0], dev_label=dev[1], dev_episode_id=dev[2],
        )
        handle.flush()
        os.fsync(handle.fileno())
    manifest: dict[str, object] = {
        "schema_version": "movement-change-event-dataset-manifest-v1",
        "dataset": dataset_path.name, "dataset_sha256": _file_sha256(dataset_path),
        "contract_sha256": contract["contract_sha256"],
        "change_contract_report_file_sha256": _file_sha256(change_report_path),
        "action_order": list(CHANGE_MODEL_ACTIONS), "samples": {"train": train[3], "dev": dev[3]},
        "input_shape": [16, 128, 128, 3], "current_goal_frame_included": True,
        "old_goal_frames": 15, "new_goal_frames": 1,
        "direction_arrow_in_input": False, "previous_action_in_input": False,
        "stop_in_dataset": False, "real_rgb_frames": 0, "training_allowed": False,
        "checkpoint_allowed": False, "input_commands_sent": 0,
        "video_dev_opened": False, "video_test_opened": False,
    }
    manifest["manifest_sha256"] = _object_sha256(manifest)
    (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    staging.rename(output_dir)
    return validate_change_event_dataset(output_dir)


def validate_change_event_dataset_v2(output_dir: Path) -> dict[str, object]:
    manifest = _load_bound(output_dir / "manifest.json", "manifest_sha256")
    dataset_path = output_dir / "change-events-position-v2.npz"
    if dataset_path.is_symlink() or _file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("position-v2 dataset hash differs")
    rows = cast(dict[str, list[dict[str, object]]], manifest["samples"])
    position_sets: dict[str, set[tuple[int, int]]] = {}
    all_hashes: list[str] = []
    summaries: dict[str, object] = {}
    with np.load(dataset_path, allow_pickle=False) as data:
        for split, samples, episodes in (("train", 256, 64), ("dev", 96, 24)):
            clips = data[f"{split}_rgb"]
            labels = data[f"{split}_label"]
            episode_ids = data[f"{split}_episode_id"]
            if (clips.shape != (samples, 16, 128, 128, 3) or clips.dtype != np.uint8
                    or labels.shape != (samples,) or episode_ids.shape != (samples,)
                    or len(rows[split]) != samples):
                raise ValueError(f"{split} position-v2 arrays differ")
            counts = np.bincount(labels, minlength=8)
            if not np.all(counts == samples // 8):
                raise ValueError(f"{split} position-v2 labels are not balanced")
            positions = {
                cast(tuple[int, int], tuple(map(int, cast(list[int], row["position"]))))
                for row in rows[split]
            }
            position_sets[split] = positions
            expected_x = {6, 7, 8} if split == "train" else {5, 9}
            horizontal_rows = {
                cast(list[int], row["position"])[1]
                for row in rows[split] if row["target_direction"] in {"E", "W"}
            }
            if {point[0] for point in positions} != expected_x or horizontal_rows != {2, 3, 4}:
                raise ValueError(f"{split} position-v2 parameter coverage differs")
            for clip, label in zip(clips, labels, strict=True):
                action = CHANGE_MODEL_ACTIONS[int(label)]
                if goal_canvas_geometry_movement(clip[-1]) != action:
                    raise ValueError(f"{split} position-v2 RGB label differs")
                all_hashes.append(hashlib.sha256(clip.tobytes()).hexdigest())
            if len(set(map(str, episode_ids.tolist()))) != episodes:
                raise ValueError(f"{split} position-v2 episodes differ")
            summaries[split] = {
                "samples": samples, "episodes": episodes,
                "current_x": sorted({point[0] for point in positions}),
                "horizontal_current_y": sorted(horizontal_rows),
                "class_counts": dict(zip(CHANGE_MODEL_ACTIONS, counts.tolist(), strict=True)),
            }
    if position_sets["train"] & position_sets["dev"] or len(set(all_hashes)) != 352:
        raise ValueError("position-v2 holdout overlap or duplicate clips")
    return {"status": "CHANGE_EVENT_POSITION_V2_DATASET_VALIDATED", "splits": summaries,
            "position_overlap": 0, "unique_clips": 352, "stop_in_dataset": False,
            "training_allowed": False}


def materialize_change_event_dataset_v2(
    contract_path: Path, failed_replay_report_path: Path, source_dataset_path: Path,
    output_dir: Path, *, verify_only: bool = False,
) -> dict[str, object]:
    if verify_only:
        return validate_change_event_dataset_v2(output_dir)
    contract = _load_bound(contract_path, "contract_sha256")
    failed = _load_bound(failed_replay_report_path, "report_sha256")
    if (
        contract.get("schema_version") != "movement-change-event-position-dataset-contract-v2"
        or contract.get("failed_replay_report_file_sha256") !=
        _file_sha256(failed_replay_report_path)
        or contract.get("source_dataset_sha256") != _file_sha256(source_dataset_path)
        or failed.get("failure") != "ROUTE_PARAMETER_GENERALIZATION_FAILED_PRECHECK"
        or failed.get("arena_steps_executed") != 0
        or contract.get("action_order") != list(CHANGE_MODEL_ACTIONS)
        or contract.get("sequence_frames") != 16 or contract.get("old_goal_frames") != 15
        or contract.get("new_goal_frames") != 1 or contract.get("events_per_episode") != 4
        or contract.get("train_episodes") != 64 or contract.get("dev_episodes") != 24
        or contract.get("train_samples") != 256 or contract.get("dev_samples") != 96
        or contract.get("train_current_x") != [6, 7, 8]
        or contract.get("dev_current_x") != [5, 9]
        or contract.get("horizontal_current_y") != [2, 3, 4]
        or contract.get("position_group_overlap_allowed") is not False
        or contract.get("previous_action_is_actor_input") is not False
        or contract.get("stop_in_dataset") is not False
        or contract.get("model_training_allowed") is not False
        or contract.get("device_input_allowed") is not False
        or contract.get("video_dev_allowed") is not False
        or contract.get("video_test_allowed") is not False
    ):
        raise ValueError("position-v2 dataset contract differs")
    if output_dir.exists():
        raise ValueError("position-v2 output exists")
    marker = cast(dict[str, object], contract["marker"])
    train = _change_event_samples_position_v2("train", 64, marker, 410_000)
    dev = _change_event_samples_position_v2("dev", 24, marker, 510_000)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".movement-change-position-v2-", dir=output_dir.parent))
    dataset_path = staging / "change-events-position-v2.npz"
    with dataset_path.open("wb") as handle:
        np.savez_compressed(
            handle, train_rgb=train[0], train_label=train[1], train_episode_id=train[2],
            dev_rgb=dev[0], dev_label=dev[1], dev_episode_id=dev[2],
        )
        handle.flush()
        os.fsync(handle.fileno())
    manifest: dict[str, object] = {
        "schema_version": "movement-change-event-position-dataset-manifest-v2",
        "dataset": dataset_path.name, "dataset_sha256": _file_sha256(dataset_path),
        "contract_sha256": contract["contract_sha256"],
        "failed_replay_report_file_sha256": _file_sha256(failed_replay_report_path),
        "source_dataset_sha256": _file_sha256(source_dataset_path),
        "action_order": list(CHANGE_MODEL_ACTIONS), "samples": {"train": train[3], "dev": dev[3]},
        "input_shape": [16, 128, 128, 3], "train_current_x": [6, 7, 8],
        "dev_current_x": [5, 9], "horizontal_current_y": [2, 3, 4],
        "position_group_overlap": 0, "previous_action_in_input": False,
        "stop_in_dataset": False, "real_rgb_frames": 0, "training_allowed": False,
        "checkpoint_allowed": False, "input_commands_sent": 0,
        "video_dev_opened": False, "video_test_opened": False,
    }
    manifest["manifest_sha256"] = _object_sha256(manifest)
    (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    staging.rename(output_dir)
    return validate_change_event_dataset_v2(output_dir)


def _warmup(position: tuple[int, int], goal: tuple[int, int]) -> tuple[str, str]:
    for outgoing, incoming, dx in (("east", "west", 1), ("west", "east", -1)):
        destination = (position[0] + dx, position[1])
        if 0 <= destination[0] < 15 and destination != goal:
            return outgoing, incoming
    raise ValueError("goal canvas warmup movement is unavailable")


def _causal_clip(
    position: tuple[int, int], goal: tuple[int, int], render_seed: int, marker: dict[str, object]
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    frames, _player_xy, _goal_xy, before, after = _localized_causal_clip(
        position, goal, render_seed, marker
    )
    return frames, before, after


def _localized_causal_clip(
    position: tuple[int, int], goal: tuple[int, int], render_seed: int, marker: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
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
    player_xy: list[tuple[int, int]] = []
    goal_xy = _point(goal)

    def observe(count: int) -> None:
        for _ in range(count):
            raw = cast(dict[str, int], arena.observe("blue")["self_position"])
            current = (raw["x"], raw["y"])
            frames.append(render_goal_minimap(current, goal, render_seed, marker))
            player_xy.append(_point(current))

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
    return (
        np.stack(frames),
        np.asarray(player_xy, dtype=np.float32),
        np.repeat(np.asarray(goal_xy, dtype=np.float32)[None], len(frames), axis=0),
        before,
        (after_raw["x"], after_raw["y"]),
    )


def _overfit_samples(
    direction_samples: int,
) -> list[tuple[StageAMovement, tuple[int, int], tuple[int, int], int]]:
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
    samples: list[tuple[StageAMovement, tuple[int, int], tuple[int, int], int]] = []
    for replicate in range(direction_samples):
        for action in MOVEMENT_ACTIONS[1:]:
            current, goal = direction_positions[action]
            samples.append((action, current, goal, 100 + replicate * 17 + len(samples)))
    stop_goals = ((7, 2), (8, 3), (7, 4), (6, 3)) * 2
    samples.extend(
        ("STOP", (7, 3), goal, 1000 + index) for index, goal in enumerate(stop_goals)
    )
    return samples


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
    marker = cast(dict[str, object], contract["marker"])
    samples = _overfit_samples(int(cast(int, contract["direction_samples"])))
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
    direction_positions = {action: (current, goal) for action, current, goal, _seed in samples[:24]}
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


def materialize_localized_overfit32(
    contract_path: Path,
    failed_relational_report_path: Path,
    source_dataset_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract = _load_bound(contract_path, "contract_sha256")
    try:
        failed = json.loads(failed_relational_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("failed relational report is invalid") from exc
    if (
        contract.get("schema_version") != "movement-goal-canvas-localized-contract-v1"
        or _file_sha256(failed_relational_report_path)
        != contract.get("failed_relational_report_sha256")
        or failed.get("status") != "FAILED"
        or failed.get("next_stage_allowed") is not False
        or _file_sha256(source_dataset_path) != contract.get("source_overfit_dataset_sha256")
        or contract.get("formal_training_allowed") is not False
        or contract.get("test_allowed") is not False
        or contract.get("holdout_allowed") is not False
        or contract.get("r2_allowed") is not False
        or contract.get("device_input_allowed") is not False
    ):
        raise ValueError("localized overfit32 contract binding differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("localized overfit32 output already exists")
    marker = cast(dict[str, object], contract["marker"])
    samples = _overfit_samples(int(cast(int, contract["direction_samples"])))
    clips: list[np.ndarray] = []
    player_sequences: list[np.ndarray] = []
    goal_sequences: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    for index, (expected, current, goal, render_seed) in enumerate(samples):
        clip, player_xy, goal_xy, before, after = _localized_causal_clip(
            current, goal, render_seed, marker
        )
        if (
            len(player_xy) != 16
            or len(goal_xy) != 16
            or rule_movement_in_range(before, goal) != expected
            or (before != after) != (expected != "STOP")
        ):
            raise ValueError("localized overfit32 causality differs")
        clips.append(clip)
        player_sequences.append(player_xy)
        goal_sequences.append(goal_xy)
        labels.append(MOVEMENT_ACTIONS.index(expected))
        episode_ids.append(f"localized-goal-canvas-{index:02d}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        dataset_path = staging / "overfit32-localized.npz"
        with dataset_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                rgb_sequence=np.stack(clips).astype(np.uint8),
                label=np.asarray(labels, dtype=np.int64),
                player_xy_sequence=np.stack(player_sequences).astype(np.float32),
                goal_xy_sequence=np.stack(goal_sequences).astype(np.float32),
                episode_id=np.asarray(episode_ids),
                contract_sha256=np.asarray([contract["contract_sha256"]]),
            )
            handle.flush()
            os.fsync(handle.fileno())
        report: dict[str, object] = {
            "schema_version": "movement-goal-canvas-localized-data-report-v1",
            "status": "PASSED",
            "contract_sha256": contract["contract_sha256"],
            "failed_relational_report_sha256": _file_sha256(
                failed_relational_report_path
            ),
            "source_overfit_dataset_sha256": _file_sha256(source_dataset_path),
            "dataset_sha256": _file_sha256(dataset_path),
            "samples": len(samples),
            "frames_per_sample": 16,
            "class_counts": {
                action: labels.count(index) for index, action in enumerate(MOVEMENT_ACTIONS)
            },
            "automatic_targets": contract["automatic_targets"],
            "actor_inputs": contract["actor_inputs"],
            "coordinate_labels_in_actor_input": False,
            "unique_episode_ids": len(set(episode_ids)),
            "real_rgb_training_frames": 0,
            "test_frames_read": 0,
            "formal_training_allowed": False,
            "r2_allowed": False,
            "device_input_commands_sent": 0,
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
