from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.movement_goal_canvas import (
    CHANGE_MODEL_ACTIONS,
    _change_event_samples,
    _change_event_samples_position_v2,
    _object_sha256,
    change_only_route,
    materialize_goal_canvas_overfit32,
    materialize_goal_canvas_trajectories,
    materialize_localized_overfit32,
    render_goal_minimap,
)
from hok_agent.movement_mvp_train import TrajectoryWindowDataset, _rollout

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "movement_goal_canvas_overfit32_v1.json"
STAGE_C_CONTRACT = ROOT / "configs" / "movement_goal_canvas_stage_c_v1.json"
LOCALIZED_CONTRACT = ROOT / "configs" / "movement_goal_canvas_localized_v1.json"
CHANGE_REPLAY_CONTRACT = ROOT / "configs" / "movement_change_replay_v1.json"


def test_change_only_route_separates_policy_persistence_and_stop() -> None:
    marker = {"radius": 7, "thickness": 2, "rgb": [245, 225, 45]}
    east = render_goal_minimap((6, 3), (8, 3), 1, marker)
    reached = render_goal_minimap((7, 3), (7, 3), 2, marker)
    changed = change_only_route("W", east, "macro_goal_version_changed")
    same = change_only_route("E", east, "macro_goal_version_changed")
    persisted = change_only_route("E", east, "none")
    assert changed["model_target"] == "E" and changed["executor_command"] == "MOVE"
    assert same["model_target"] == "E" and same["executor_command"] == "KEEP"
    assert not persisted["model_invoked"] and persisted["executor_command"] == "KEEP"
    for stopped in (
        change_only_route("E", reached, "macro_goal_version_changed"),
        change_only_route("E", None, "macro_goal_version_changed"),
        change_only_route("E", east, "none", terminal=True),
    ):
        assert not stopped["model_invoked"] and stopped["applied_action"] == "STOP"
        assert stopped["owner"] == "deterministic_router"
    assert "STOP" not in CHANGE_MODEL_ACTIONS and len(CHANGE_MODEL_ACTIONS) == 8


def test_change_event_samples_are_balanced_grouped_and_goal_conditioned() -> None:
    marker = {"radius": 7, "thickness": 2, "rgb": [245, 225, 45]}
    clips, labels, episodes, rows = _change_event_samples("tiny", 4, 4, marker, 700)
    assert clips.shape == (16, 16, 128, 128, 3)
    assert np.bincount(labels, minlength=8).tolist() == [2] * 8
    assert len(set(episodes.tolist())) == 4 and len(rows) == 16
    assert all(not np.array_equal(clip[-2], clip[-1]) for clip in clips)
    assert all(row["old_direction"] != row["target_direction"] for row in rows)
    assert not any(row["target_direction"] == "STOP" for row in rows)


def test_position_v2_covers_horizontal_rows_with_disjoint_position_groups() -> None:
    marker = {"radius": 7, "thickness": 2, "rgb": [245, 225, 45]}
    train = _change_event_samples_position_v2("train", 6, marker, 800)
    dev = _change_event_samples_position_v2("dev", 6, marker, 1800)
    for output in (train, dev):
        assert output[0].shape == (24, 16, 128, 128, 3)
        assert np.bincount(output[1], minlength=8).tolist() == [3] * 8
        rows = output[3]
        horizontal_rows = {
            row["position"][1] for row in rows if row["target_direction"] in {"E", "W"}
        }
        assert horizontal_rows == {2, 3, 4}
    train_positions = {tuple(row["position"]) for row in train[3]}
    dev_positions = {tuple(row["position"]) for row in dev[3]}
    assert {point[0] for point in train_positions} == {6, 7, 8}
    assert {point[0] for point in dev_positions} == {5, 9}
    assert not train_positions & dev_positions


def test_change_replay_routes_are_two_cell_segments_and_cover_all_directions() -> None:
    from hok_agent.movement_mvp import MOVEMENT_ACTIONS, rule_movement

    contract = json.loads(CHANGE_REPLAY_CONTRACT.read_text())
    supplied = contract.pop("contract_sha256")
    assert supplied == _object_sha256(contract)
    directions = []
    for route in contract["routes"]:
        position = tuple(route["start"])
        for raw_goal in route["goals"]:
            goal = tuple(raw_goal)
            directions.append(rule_movement(position, goal))
            assert max(abs(goal[0] - position[0]), abs(goal[1] - position[1])) == 2
            position = goal
    assert set(directions) == set(MOVEMENT_ACTIONS[1:])
    assert contract["expected_model_invocations"] == 40
    assert contract["expected_keep_steps"] == contract["expected_router_stops"] == 40


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    prior = {
        "schema_version": "movement-real-rgb-goal-canvas-report-v2",
        "status": "GOAL_CANVAS_GENERATION_PASSED_SELF_LOCALIZATION_UNRESOLVED",
        "contract_sha256": "243aa47aa164af2e9783a9ad08aa7cd372c602537704a75685374c72c9704e28",
    }
    prior["report_sha256"] = _object_sha256(prior)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["goal_canvas_report_sha256"] = prior["report_sha256"]
    contract.pop("contract_sha256")
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path, prior_path


def test_goal_canvas_overfit32_is_balanced_causal_and_rgb_only(tmp_path: Path) -> None:
    contract, prior = _inputs(tmp_path)
    output = tmp_path / "dataset"
    report = materialize_goal_canvas_overfit32(contract, prior, output)
    assert report["status"] == "PASSED"
    assert report["samples"] == 32
    assert report["frames_per_sample"] == 16
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
    assert report["windows_with_state_change"] == 32
    assert report["counterfactual_goal_changes"] == 8
    assert report["structured_coordinates_in_model_input"] is False
    assert report["direction_arrow_in_model_input"] is False
    assert report["action_executed_after_window"] is True
    assert report["real_rgb_training_frames"] == 0
    with np.load(output / "overfit32.npz", allow_pickle=False) as data:
        assert data["rgb_sequence"].shape == (32, 16, 128, 128, 3)
        assert data["rgb_sequence"].dtype == np.uint8
        assert len(set(data["episode_id"].tolist())) == 32
        assert sorted(data["label"].tolist()) == [
            *([0] * 8),
            *[label for label in range(1, 9) for _ in range(3)],
        ]


def test_goal_canvas_changes_only_with_goal_and_rejects_prior_tampering(tmp_path: Path) -> None:
    contract, prior = _inputs(tmp_path)
    marker = json.loads(contract.read_text(encoding="utf-8"))["marker"]
    north = render_goal_minimap((7, 3), (7, 2), 9, marker)
    south = render_goal_minimap((7, 3), (7, 4), 9, marker)
    assert not np.array_equal(north, south)
    assert np.array_equal(north, render_goal_minimap((7, 3), (7, 2), 9, marker))
    prior.write_text(prior.read_text(encoding="utf-8") + " ", encoding="utf-8")
    payload = json.loads(prior.read_text(encoding="utf-8"))
    payload["status"] = "PASSED"
    prior.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        materialize_goal_canvas_overfit32(contract, prior, tmp_path / "blocked")


def test_goal_canvas_stage_c_trajectories_are_action_driven_and_disjoint(
    tmp_path: Path,
) -> None:
    overfit = {
        "status": "PASSED",
        "dataset_sha256": "8b7104f55b71fc6a54b9be3d7c6204c50d1d357288c2e2bc30e9abc14ac33cde",
        "diagnostic_checkpoint_reusable_for_formal_training": False,
    }
    overfit_path = tmp_path / "overfit-report.json"
    overfit_path.write_text(json.dumps(overfit), encoding="utf-8")
    contract = json.loads(STAGE_C_CONTRACT.read_text(encoding="utf-8"))
    contract["overfit_report_sha256"] = hashlib.sha256(overfit_path.read_bytes()).hexdigest()
    contract.pop("contract_sha256")
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "stage-c-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "trajectories"
    report = materialize_goal_canvas_trajectories(contract_path, overfit_path, output)
    assert report["status"] == "PASSED"
    assert report["episode_counts"] == {"train": 64, "dev": 24}
    assert report["teacher_successes"] == {"train": 64, "dev": 24}
    assert report["scenario_overlap"] == 0
    assert report["diagnostic_checkpoint_loaded"] is False
    dataset = TrajectoryWindowDataset(output, "train")
    assert len(dataset) > 64
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    scenario = next(row for row in manifest["episodes"] if row["split"] == "dev")
    with np.load(output / "episodes" / scenario["basename"], allow_pickle=False) as episode:
        assert episode["labels"][-3:].tolist() == [0, 0, 0]
        assert np.array_equal(
            episode["frame_timestamps_ms"], np.arange(len(episode["labels"])) * 100
        )
    result = _rollout(
        scenario,
        "geometry",
        128,
        16,
        7001,
        stop_confirmation_steps=3,
        navigation_only=True,
        goal_canvas=True,
        marker=contract["marker"],
    )
    assert result["status"] == "success"


def test_localized_overfit32_adds_targets_without_actor_leakage(tmp_path: Path) -> None:
    source_contract, prior = _inputs(tmp_path)
    source = tmp_path / "source"
    materialize_goal_canvas_overfit32(source_contract, prior, source)
    failed = {"status": "FAILED", "next_stage_allowed": False}
    failed_path = tmp_path / "failed-relational.json"
    failed_path.write_text(json.dumps(failed), encoding="utf-8")
    contract = json.loads(LOCALIZED_CONTRACT.read_text(encoding="utf-8"))
    contract["failed_relational_report_sha256"] = hashlib.sha256(
        failed_path.read_bytes()
    ).hexdigest()
    contract["source_overfit_dataset_sha256"] = hashlib.sha256(
        (source / "overfit32.npz").read_bytes()
    ).hexdigest()
    contract.pop("contract_sha256")
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "localized-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "localized"
    report = materialize_localized_overfit32(
        contract_path, failed_path, source / "overfit32.npz", output
    )
    assert report["status"] == "PASSED"
    assert report["automatic_targets"] == ["player_xy", "goal_xy"]
    assert report["actor_inputs"] == ["rgb_sequence"]
    assert report["coordinate_labels_in_actor_input"] is False
    with np.load(output / "overfit32-localized.npz", allow_pickle=False) as data:
        assert data["player_xy_sequence"].shape == (32, 16, 2)
        assert data["goal_xy_sequence"].shape == (32, 16, 2)
        frame = data["rgb_sequence"][0, 0]
        y, x = np.where(np.all(frame == (55, 195, 235), axis=2))
        assert np.allclose(data["player_xy_sequence"][0, 0], [x.mean(), y.mean()])
