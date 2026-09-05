from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from hok_agent.movement_mvp import (
    MOVEMENT_ACTIONS,
    _canonical_sha256,
    load_navigation_config,
    mark_visible_target,
    materialize_overfit32,
    materialize_stage_c_trajectories,
    rgb_geometry_movement,
    rule_movement,
    run_rule_batch,
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


def test_rule_batch_resumes_without_replaying_completed_episodes(tmp_path: Path) -> None:
    output = tmp_path / "batch"
    first = run_rule_batch(CONFIG, output, 1)
    episode_id = first["episode_summaries"][0]["episode_id"]
    with UnifiedTransitionStore(output / "replay.sqlite3") as store:
        original = store.load_episode(episode_id)
    assert len(original) == 9
    assert [row["done"] for row in original] == [False] * 8 + [True]
    with pytest.raises(ValueError, match="use --resume"):
        run_rule_batch(CONFIG, output, 3)
    third = run_rule_batch(CONFIG, output, 3, resume=True)
    assert third["milestones"] == [1, 3]
    assert third["transitions"] == 27
    with UnifiedTransitionStore(output / "replay.sqlite3") as store:
        assert store.load_episode(episode_id) == original
        for episode in third["episode_summaries"]:
            rows = store.load_episode(episode["episode_id"])
            assert rows[-1]["done"]
            assert all((output / row["observation"]["frame_bundle_ref"]).is_file() for row in rows)
    assert len({row["episode_id"] for row in third["episode_summaries"]}) == 3
    assert third["input_commands_sent"] == 0


def _batch_rows(output: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with UnifiedTransitionStore(output / "replay.sqlite3") as store:
        for ordinal in range(10):
            rows.extend(store.load_episode(f"movement-rule-seed-0-episode-{ordinal:03d}"))
    return tuple(rows)


def test_rule_batch_mid_episode_resume_matches_uninterrupted(tmp_path: Path) -> None:
    interrupted, continuous = tmp_path / "interrupted", tmp_path / "continuous"
    paused = run_rule_batch(CONFIG, interrupted, 10, step_budget=4)
    assert paused["status"] == "PAUSED"
    assert paused["completed_episodes"] == 0
    assert paused["current_episode_id"] == "movement-rule-seed-0-episode-000"
    assert paused["next_step"] == 4
    resumed = run_rule_batch(CONFIG, interrupted, 10, resume=True)
    direct = run_rule_batch(CONFIG, continuous, 10)
    assert resumed["status"] == direct["status"] == "PASSED"
    assert resumed["delivery_grade"] == "R0_RULE_OFFLINE"
    assert resumed["transitions"] == direct["transitions"] == 90
    assert resumed["terminal_transitions"] == 10
    assert resumed["recovered_transition_count"] == 4
    assert resumed["resume_count"] == 1
    assert resumed["sqlite_integrity"] == "ok"
    assert resumed["transition_content_sha256"] == direct["transition_content_sha256"]
    assert resumed["frame_view_manifest_sha256"] == direct["frame_view_manifest_sha256"]
    assert _batch_rows(interrupted) == _batch_rows(continuous)


def test_rule_batch_recovers_committed_row_and_atomic_orphan_frame(tmp_path: Path) -> None:
    committed = tmp_path / "committed"
    with pytest.raises(RuntimeError, match="committed transition"):
        run_rule_batch(CONFIG, committed, 1, interrupt_after_commits=4)
    assert not (committed / "batch-summary.json").exists()
    resumed = run_rule_batch(CONFIG, committed, 1, resume=True)
    assert resumed["transitions"] == 9
    assert [row["step_id"] for row in _batch_rows(committed)] == list(range(9))

    orphan = tmp_path / "orphan"
    with pytest.raises(RuntimeError, match="atomic frame"):
        run_rule_batch(CONFIG, orphan, 1, interrupt_after_frames=5)
    assert len(_batch_rows(orphan)) == 4
    assert (orphan / "movement-rule-seed-0-episode-000-005.npz").is_file()
    resumed = run_rule_batch(CONFIG, orphan, 1, resume=True)
    assert resumed["transitions"] == 9

    incomplete = tmp_path / "incomplete"
    run_rule_batch(CONFIG, incomplete, 1, step_budget=4)
    temporary = incomplete / ".movement-rule-seed-0-episode-000-005.npz.tmp"
    temporary.write_bytes(b"incomplete")
    resumed = run_rule_batch(CONFIG, incomplete, 1, resume=True)
    assert resumed["transitions"] == 9


def test_rule_batch_rejects_changed_bindings_and_committed_evidence(tmp_path: Path) -> None:
    output = tmp_path / "bound"
    run_rule_batch(CONFIG, output, 10, step_budget=4)
    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["seed"] = 1
    changed_config = tmp_path / "changed.json"
    changed_config.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="resume contract differs"):
        run_rule_batch(changed_config, output, 10, resume=True)

    contract_path = output / "run-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["source_sha256"]["rich_arena"] = "0" * 64
    contract.pop("run_contract_sha256")
    contract["run_contract_sha256"] = _canonical_sha256(contract)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="resume contract differs"):
        run_rule_batch(CONFIG, output, 10, resume=True)

    frame_output = tmp_path / "frame"
    run_rule_batch(CONFIG, frame_output, 10, step_budget=4)
    frame = frame_output / "movement-rule-seed-0-episode-000-002.npz"
    frame.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid existing frame bundle"):
        run_rule_batch(CONFIG, frame_output, 10, resume=True)

    row_output = tmp_path / "row"
    run_rule_batch(CONFIG, row_output, 10, step_budget=4)
    with sqlite3.connect(row_output / "replay.sqlite3") as connection:
        encoded = connection.execute(
            "SELECT payload_json FROM transitions WHERE episode_id = ? AND step_id = 1",
            ("movement-rule-seed-0-episode-000",),
        ).fetchone()[0]
        payload = json.loads(encoded)
        payload["executed_action"]["applied_movement"] = "W"
        connection.execute(
            "UPDATE transitions SET payload_json = ? WHERE episode_id = ? AND step_id = 1",
            (json.dumps(payload), "movement-rule-seed-0-episode-000"),
        )
    with pytest.raises(ValueError, match="fixed rule"):
        run_rule_batch(CONFIG, row_output, 10, resume=True)


def test_rule_batch_requires_resume_and_cannot_shrink(tmp_path: Path) -> None:
    output = tmp_path / "partial"
    run_rule_batch(CONFIG, output, 10, step_budget=4)
    with pytest.raises(ValueError, match="use --resume"):
        run_rule_batch(CONFIG, output, 10)
    run_rule_batch(CONFIG, output, 3, resume=True)
    with pytest.raises(ValueError, match="cannot shrink"):
        run_rule_batch(CONFIG, output, 1, resume=True)


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
            action: sum(row["split"] == split and row["initial_action"] == action for row in rows)
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
