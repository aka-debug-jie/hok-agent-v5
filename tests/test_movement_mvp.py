from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from hok_agent.movement_mvp import (
    MOVEMENT_ACTIONS,
    load_navigation_config,
    mark_visible_target,
    materialize_overfit32,
    materialize_stage_c_trajectories,
    rgb_geometry_movement,
    rule_movement,
    run_stage_a,
    stage_c_scenarios,
    to_arena_action,
)
from hok_agent.transition_store import UnifiedTransitionStore, validate_transition

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "movement_mvp.json"


def test_stage_a_action_order_preserves_rich_directions_after_stop() -> None:
    assert MOVEMENT_ACTIONS == ("STOP", "N", "S", "W", "E", "NW", "NE", "SW", "SE")
    assert [to_arena_action(action).direction for action in MOVEMENT_ACTIONS[1:]] == [
        "north",
        "south",
        "west",
        "east",
        "northwest",
        "northeast",
        "southwest",
        "southeast",
    ]
    assert rule_movement((2, 4), (8, 4)) == "E"
    assert rule_movement((8, 4), (8, 4)) == "STOP"


def test_stage_a_rule_trajectory_moves_then_commits_stop(tmp_path: Path) -> None:
    summary = run_stage_a(CONFIG, tmp_path / "stage-a")
    assert summary["status"] == "PASSED"
    assert summary["actions"] == ["E", "E", "E", "E", "E", "E", "STOP"]
    assert summary["positions"] == [
        (2, 4),
        (3, 4),
        (4, 4),
        (5, 4),
        (6, 4),
        (7, 4),
        (8, 4),
        (8, 4),
    ]
    assert summary["position_changed"] is True
    assert summary["terminal_reason"] == "NAVIGATION_GOAL_REACHED"
    assert summary["episode_end_kind"] == "TERMINATED"
    assert summary["terminal_transition_committed"] is True
    assert summary["reward_total"] == 0.0
    assert summary["input_commands_sent"] == 0
    assert len(tuple((tmp_path / "stage-a").glob("*.npz"))) == 8


def test_stage_a_timeout_and_end_categories_are_distinct(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["max_steps"] = 2
    config = tmp_path / "timeout.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    summary = run_stage_a(config, tmp_path / "timeout")
    assert summary["status"] == "FAILED"
    assert summary["terminal_reason"] == "TIMEOUT"
    assert summary["episode_end_kind"] == "TRUNCATED"
    assert summary["terminal_transition_committed"] is True

    with UnifiedTransitionStore(tmp_path / "timeout" / "replay.sqlite3") as store:
        row = store.load_episode(str(summary["episode_id"]))[-1]
    for reason, kind in (("VIDEO_EOF", "TRUNCATED"), ("ACTION_FAILURE", "ERROR")):
        candidate = copy.deepcopy(row)
        candidate["terminal_reason"] = reason  # type: ignore[typeddict-item]
        candidate["episode_end_kind"] = kind  # type: ignore[typeddict-item]
        assert validate_transition(candidate).valid
    row["episode_end_kind"] = "TERMINATED"
    assert "episode_end_kind_mismatch" in validate_transition(row).errors


def test_stage_a_config_is_fixed_to_houyi_blue_bottom() -> None:
    config = load_navigation_config(CONFIG)
    assert (config.hero, config.role, config.side, config.lane) == (
        "houyi",
        "marksman",
        "blue",
        "bottom",
    )


def test_stage_b_materializes_causal_balanced_overfit32(tmp_path: Path) -> None:
    output = tmp_path / "overfit32"
    report = materialize_overfit32(CONFIG, output)
    assert report["samples"] == 32
    assert report["class_counts"] == {
        "STOP": 8,
        "N": 3,
        "S": 3,
        "W": 3,
        "E": 3,
        "NW": 3,
        "NE": 3,
        "SW": 3,
        "SE": 3,
    }
    assert report["unique_episode_ids"] == 32
    assert report["windows_with_state_change"] == 32
    assert report["action_executed_after_window"] is True
    with np.load(output / "overfit32.npz", allow_pickle=False) as data:
        assert data["rgb_sequence"].shape == (32, 16, 128, 128, 3)
        assert data["rgb_sequence"].dtype == np.uint8
        assert sorted(data["label"].tolist()).count(0) == 8


def test_goal_marker_uses_visible_rgb_target_category() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:11, 6:9] = (225, 70, 65)
    rgb[20:23, 24:27] = (220, 120, 65)
    hero = mark_visible_target(rgb, "opponent_hero")
    tower = mark_visible_target(rgb, "enemy_tower")
    assert not np.array_equal(hero, tower)
    assert np.array_equal(rgb[8:11, 6:9], hero[8:11, 6:9])


def test_stage_c_scenarios_are_balanced_and_disjoint() -> None:
    rows = stage_c_scenarios()
    assert sum(row["split"] == "train" for row in rows) == 64
    assert sum(row["split"] == "dev" for row in rows) == 24
    train = {str(row["scenario_id"]) for row in rows if row["split"] == "train"}
    dev = {str(row["scenario_id"]) for row in rows if row["split"] == "dev"}
    assert not train & dev
    for split, count in (("train", 8), ("dev", 3)):
        assert {
            action: sum(
                row["split"] == split and row["initial_action"] == action for row in rows
            )
            for action in MOVEMENT_ACTIONS[1:]
        } == {action: count for action in MOVEMENT_ACTIONS[1:]}


def test_stage_c_stores_episode_frames_and_window_indices(tmp_path: Path) -> None:
    output = tmp_path / "stage-c"
    report = materialize_stage_c_trajectories(CONFIG, output)
    assert report["status"] == "PASSED"
    assert report["episode_counts"] == {"train": 64, "dev": 24}
    assert report["teacher_successes"] == {"train": 64, "dev": 24}
    assert report["scenario_overlap"] == 0
    assert len(tuple((output / "episodes").glob("*.npz"))) == 88
    with np.load(output / "episodes" / "train-000.npz", allow_pickle=False) as data:
        assert len(data["frames"]) == len(data["labels"])
        assert np.array_equal(data["window_end"], np.arange(len(data["labels"])))
        assert np.array_equal(data["frame_timestamps_ms"], np.arange(len(data["labels"])) * 100)


def test_rgb_geometry_baseline_reads_self_and_marked_target() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[30:33, 20:23] = (55, 195, 235)
    rgb[17:20, 36:39] = (225, 70, 65)
    marked = mark_visible_target(rgb, "opponent_hero")
    assert rgb_geometry_movement(marked) == "NE"
