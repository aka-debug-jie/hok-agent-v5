from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.t8_v4 import STATE_NAMES
from hok_agent.t8_v5 import T8V5Error, roi_views, verify_t8_v5_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t8_v5_roi_experiment_v1.json"


def test_t8_v5_contract_freezes_roi_only_non_promoting_route() -> None:
    result = verify_t8_v5_contract(CONTRACT)
    assert result["status"] == "PASSED"
    assert result["human_labels_used"] is False
    assert result["semantic_accuracy_verified"] is False
    assert result["video_test_accessed"] is False
    assert result["promotion_allowed"] is False
    assert result["control_output"] is False
    assert result["device_input_allowed"] is False


def test_t8_v5_rehashed_gate_drift_still_fails(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["minimum_gain_over_wrong_roi"] = 0.14
    value["experiment_sha256"] = hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "experiment_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(T8V5Error, match="frozen contract values differ"):
        verify_t8_v5_contract(path)


def test_t8_v5_roi_views_are_head_isolated_and_deterministic() -> None:
    frames = np.zeros((1, 128, 128, 3), dtype=np.uint8)
    frames[:, :, :92] = 17
    points = [(0.78, 0.72), (0.87, 0.64), (0.93, 0.54)]
    for index, (x_value, y_value) in enumerate(points):
        x, y = round(x_value * 127), round(y_value * 127)
        frames[:, y - 5 : y + 6, x - 5 : x + 6] = 60 + index * 60
    first_correct, first_wrong = roi_views(frames, points)
    second_correct, second_wrong = roi_views(frames, points)
    assert first_correct.shape == (1, len(STATE_NAMES), 128, 128, 3)
    assert np.array_equal(first_correct, second_correct)
    assert np.array_equal(first_wrong, second_wrong)
    assert not np.array_equal(first_correct[:, 0], first_wrong[:, 0])
    for index in range(1, len(STATE_NAMES)):
        assert not np.array_equal(first_correct[:, index], first_wrong[:, index])


def test_t8_v5_roi_contract_excludes_tcn_and_skill2_gate() -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert "causal_tcn" not in value["models"]
    assert value["formal_gate_heads"] == list(STATE_NAMES[:3])
    assert value["diagnostic_only_heads"] == [STATE_NAMES[3]]
