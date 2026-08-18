from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hok_agent import operation_policy
from hok_agent.mobile_testbed import ObservationROIs


def test_operation_policy_contract_is_self_hashed_and_locked(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "configs/operation_policy_v1.json"
    result = operation_policy.verify_operation_policy_contract(source)
    assert result["status"] == "PASSED"
    assert result["learned_heads"] == ["movement", "combat"]
    assert result["deterministic_heads"] == ["purchase", "hard_stop"]
    assert result["device_input_allowed"] is False

    value = json.loads(source.read_text(encoding="utf-8"))
    value["teacher_confidence_threshold"] = 0.79
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(operation_policy.OperationPolicyError, match="contract differs"):
        operation_policy.verify_operation_policy_contract(changed)


def test_operation_direct_policy_contract_is_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    result = operation_policy.verify_operation_direct_policy_contract(
        root / "configs/operation_direct_policy_v1.json"
    )
    assert result["status"] == "PASSED"
    assert result["supervision_source"] == "executed_action"
    assert result["views"] == ["main_view", "hud", "minimap"]
    assert result["device_input_allowed"] is False


def test_operation_movement_policy_contract_binds_existing_combat_model() -> None:
    root = Path(__file__).resolve().parents[1]
    result = operation_policy.verify_operation_movement_policy_contract(
        root / "configs/operation_movement_policy_v1.json"
    )
    assert result["status"] == "PASSED"
    assert result["required_label_source"] == "rgb_minimap_teacher_v1"
    assert result["combat_model_sha256"].startswith("bce47dc1")
    assert result["device_input_allowed"] is False


def test_operation_policy_crop_uses_private_screen_mapping() -> None:
    frames = np.zeros((2, 20, 40, 3), dtype=np.uint8)
    frames[:, 5:15, 10:30] = 255
    rois = ObservationROIs(
        400,
        200,
        1,
        (100, 50, 300, 150),
        (0, 0, 100, 100),
        (200, 100, 400, 200),
        (0, 0, 10, 10),
        (0, 0, 10, 10),
        1,
        1,
    )
    cropped = operation_policy._resize_crops(frames, rois.main_view, rois)
    assert cropped.shape == (2, 128, 128, 3)
    assert float(cropped.mean()) == 255.0


def test_operation_policy_pairs_are_timestamped_and_hard_stop_filtered() -> None:
    timestamps = np.arange(8, dtype=np.int64) * 100
    source, target = operation_policy._future_indices(timestamps, 200)
    assert source.tolist() == [0, 1, 2, 3, 4, 5]
    assert target.tolist() == [2, 3, 4, 5, 6, 7]
    features = np.arange(8 * 512, dtype=np.float32).reshape(8, 512)
    session = operation_policy.EncodedSession(
        "a" * 64,
        features,
        features,
        features,
        timestamps,
        np.asarray([1, 1, 1, 2, 2, 2, 2, 2]),
        np.zeros(8, dtype=np.int64),
        np.asarray([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8),
    )
    paired, labels = operation_policy._movement_pairs(session, 200)
    assert paired.shape[1] == 1536
    assert labels.tolist() == [2, 2, 2]


def test_operation_policy_negative_cap_is_deterministic() -> None:
    features = np.arange(60, dtype=np.float32).reshape(20, 3)
    labels = np.asarray([0] * 16 + [1] * 4, dtype=np.int64)
    first = operation_policy._cap_negative(features, labels, 3, seed=7)
    second = operation_policy._cap_negative(features, labels, 3, seed=7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.bincount(first[1], minlength=2).tolist() == [12, 4]


def test_operation_policy_consensus_rejects_disagreement(monkeypatch) -> None:
    calls = 0

    def fake_predictions(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        labels = np.asarray([1, 2], dtype=np.int64)
        if calls == 2:
            labels[1] = 3
        return np.asarray([0, 1]), labels, np.asarray([0.9, 0.95], dtype=np.float32)

    monkeypatch.setattr(operation_policy, "_pair_predictions", fake_predictions)
    values = np.zeros((4, 512), dtype=np.float32)
    labels, confidence = operation_policy._consensus_labels(
        {},
        "movement",
        values,
        values,
        np.arange(4, dtype=np.int64) * 100,
        0.8,
        torch.device("cpu"),
        2,
        "b" * 64,
        3,
    )
    assert labels.tolist() == [1, -1, -1, -1]
    assert confidence[0] >= 0.89


def test_operation_policy_model_shapes_and_test_fail_closed() -> None:
    for kind in ("last_frame", "pool_mlp", "causal_tcn"):
        model = operation_policy._PolicyModel(kind, 1024).eval()
        with torch.no_grad():
            movement, combat = model(torch.zeros((2, 16, 1024)))
        assert movement.shape == (2, 9)
        assert combat.shape == (2, 5)
    with pytest.raises(operation_policy.OperationPolicyError, match="video-train or video-dev"):
        next(operation_policy._target_sessions(Path("missing"), {}, "test"))


def test_operation_direct_examples_keep_causal_windows_and_sparse_heads() -> None:
    rows = 40
    features = np.zeros((rows, 512, 4, 4), dtype=np.float16)
    movement = np.asarray(([1] * 20) + ([2] * 20), dtype=np.int64)
    combat = np.zeros(rows, dtype=np.int64)
    combat[[20, 30]] = [1, 2]
    session = operation_policy.EncodedSession(
        "c" * 64,
        features,
        features,
        features,
        np.arange(rows, dtype=np.int64) * 200,
        movement,
        combat,
        np.zeros(rows, dtype=np.uint8),
    )
    examples = operation_policy._direct_examples([session], 3, seed=0)
    assert examples.windows.shape[1:] == (16, 1536)
    assert set(examples.movement_id.tolist()) == {1, 2}
    assert np.sum(examples.combat_id > 0) == 2
    assert np.sum(examples.combat_id == 0) == 6
    assert np.sum(examples.movement_transition) == 1


def test_operation_movement_examples_require_teacher_source_and_are_causal() -> None:
    rows = 40
    features = np.zeros((rows, 512, 4, 4), dtype=np.float16)
    movement = np.asarray(([1] * 20) + ([2] * 20), dtype=np.int64)
    session = operation_policy.EncodedSession(
        "d" * 64,
        features,
        features,
        features,
        np.arange(rows, dtype=np.int64) * 200,
        movement,
        np.zeros(rows, dtype=np.int64),
        np.zeros(rows, dtype=np.uint8),
        np.ones(rows, dtype=np.float32),
        np.ones(rows, dtype=np.uint8),
    )
    examples = operation_policy._movement_examples([session], 3, seed=0)
    assert examples.windows.shape == (25, 16, 1024)
    assert examples.labels.tolist().count(1) == 5
    assert examples.labels.tolist().count(2) == 20
    assert int(np.sum(examples.transitions)) == 1
    for kind in ("last_frame", "pool_mlp", "causal_tcn"):
        model = operation_policy._MovementModel(kind).eval()
        with torch.no_grad():
            assert model(torch.zeros((2, 16, 1024))).shape == (2, 9)
