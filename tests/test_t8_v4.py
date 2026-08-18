from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.t8_v4 import (
    STATE_NAMES,
    T8V4Error,
    _normalize_teacher_frame,
    _restore_teacher_frame,
    consensus_labels,
    deterministic_photometric_views,
    spatial_mask,
    spatial_swap,
    verify_t8_v4_contracts,
)

ROOT = Path(__file__).resolve().parents[1]


def _contracts() -> dict[str, Path]:
    return {
        "observation_contract": ROOT / "game_rules/observation_contract_v2.json",
        "candidate_contract": ROOT / "game_rules/candidate_action_contract_v1.json",
        "weak_supervision_contract": ROOT / "configs/t8_v4_weak_supervision_v1.json",
        "experiment_contract": ROOT / "configs/t8_v4_experiment_plan_v1.json",
    }


def test_t8_v4_frozen_contracts_pass_and_disable_control() -> None:
    checked = verify_t8_v4_contracts(**_contracts())
    assert checked["status"] == "PASSED"
    assert checked["human_labels_used"] is False
    assert checked["video_test_accessed"] is False
    assert checked["promotion_allowed"] is False
    assert checked["control_output"] is False
    assert checked["device_input_allowed"] is False


def test_t8_v4_contract_hash_and_threshold_drift_fail(tmp_path: Path) -> None:
    paths = _contracts()
    weak = json.loads(paths["weak_supervision_contract"].read_text(encoding="utf-8"))
    weak["teacher_confidence_threshold"] = 0.79
    changed = tmp_path / "weak.json"
    changed.write_text(json.dumps(weak), encoding="utf-8")
    paths["weak_supervision_contract"] = changed
    with pytest.raises(T8V4Error, match="identity is invalid"):
        verify_t8_v4_contracts(**paths)

    weak["weak_supervision_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in weak.items() if key != "weak_supervision_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    changed.write_text(json.dumps(weak), encoding="utf-8")
    with pytest.raises(T8V4Error, match="frozen contract values differ"):
        verify_t8_v4_contracts(**paths)


def test_t8_v4_consensus_accepts_only_confident_stable_agreement() -> None:
    rule = np.full((4, 5, len(STATE_NAMES)), 0.9, dtype=np.float32)
    source = np.full_like(rule, 0.85)
    rule[1, :, 0] = 0.1
    source[1, :, 0] = 0.9
    rule[2, 2, 1] = 0.1
    source[3, :, 2] = 0.75
    labels, accepted, qc = consensus_labels(rule, source)
    assert accepted[0].tolist() == [1, 1, 1, 1]
    assert accepted[1, 0] == 0
    assert accepted[2, 1] == 0
    assert accepted[3, 2] == 0
    assert labels[accepted == 0].tolist() == [-1, -1, -1]
    assert qc["coverage_by_head"] != {}


def test_t8_v4_photometric_views_are_deterministic() -> None:
    frames = np.arange(2 * 8 * 8 * 3, dtype=np.uint8).reshape(2, 8, 8, 3)
    first = deterministic_photometric_views(frames)
    second = deterministic_photometric_views(frames)
    assert first.shape == (2, 5, 8, 8, 3)
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, 0], frames)


def test_t8_v4_teacher_frame_removes_letterbox_without_future_data() -> None:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[34:93] = 73
    normalized = _normalize_teacher_frame(frame, (0, 34, 128, 93))
    assert normalized.shape == frame.shape
    assert np.all(normalized == 73)
    restored = _restore_teacher_frame(normalized, frame, (0, 34, 128, 93), "stored")
    assert np.array_equal(restored, frame)


def test_t8_v4_spatial_interventions_touch_only_requested_regions() -> None:
    frames = np.full((2, 128, 128, 3), 7, dtype=np.uint8)
    donors = np.full_like(frames, 19)
    gameplay = spatial_mask(frames, "gameplay")
    hud = spatial_mask(frames, "hud")
    assert np.all(gameplay[:, :38] == 0)
    assert np.all(gameplay[:, 38:, 66:] == 7)
    assert np.all(hud[:, 38:, 66:] == 0)
    assert np.all(hud[:, :38] == 7)
    swapped = spatial_swap(frames, donors, "gameplay")
    assert np.all(swapped[:, :38] == 19)
    assert np.all(swapped[:, 38:, 66:] == 7)


def test_t8_v4_rejects_runtime_threshold_override() -> None:
    values = np.full((1, 5, len(STATE_NAMES)), 0.9, dtype=np.float32)
    with pytest.raises(T8V4Error, match="probability contract"):
        consensus_labels(values, values, threshold=0.7)
