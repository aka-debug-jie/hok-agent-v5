from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.movement_goal_canvas import (
    _object_sha256,
    materialize_goal_canvas_overfit32,
    render_goal_minimap,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "movement_goal_canvas_overfit32_v1.json"


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
