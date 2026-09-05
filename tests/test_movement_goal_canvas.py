from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.movement_goal_canvas import (
    _object_sha256,
    materialize_goal_canvas_overfit32,
    materialize_goal_canvas_trajectories,
    render_goal_minimap,
)
from hok_agent.movement_mvp_train import TrajectoryWindowDataset, _rollout

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "movement_goal_canvas_overfit32_v1.json"
STAGE_C_CONTRACT = ROOT / "configs" / "movement_goal_canvas_stage_c_v1.json"


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
