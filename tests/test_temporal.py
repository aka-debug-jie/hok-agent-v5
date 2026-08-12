from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from hok_agent import temporal

HASHES = ("1" * 64, "2" * 64, "3" * 64, "4" * 64)
FROZEN = {
    "kappa": 0.70,
    "overall_precision": 0.85,
    "per_class_precision": 0.75,
    "coverage": 0.30,
    "ood_false_accept": 0.05,
    "baseline_delta": 0.05,
    "source_accuracy_drop": 0.02,
    "source_recall_drop": 0.05,
}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _release(path: Path, threshold: float = 0.75) -> Path:
    payload: dict[str, object] = {
        "schema_version": "hok-agent-v5-release-v1",
        "model_sha256": HASHES[0],
        "alignment_sha256": HASHES[1],
        "audit_sha256": HASHES[2],
        "config_sha256": HASHES[3],
        "overall_pass": True,
        "allowed_classes": list(temporal.ACTION_NAMES),
        "class_thresholds": {name: threshold for name in temporal.ACTION_NAMES},
        "thresholds": FROZEN,
        "thresholds_hash": hashlib.sha256(_json(FROZEN).encode()).hexdigest(),
    }
    payload["release_sha256"] = hashlib.sha256(_json(payload).encode()).hexdigest()
    path.write_text(_json(payload) + "\n", encoding="utf-8")
    return path


def _coach(path: Path) -> temporal.TemporalCoach:
    return temporal.TemporalCoach(
        release_path=path,
        expected_model_sha256=HASHES[0],
        expected_alignment_sha256=HASHES[1],
        expected_audit_sha256=HASHES[2],
        expected_config_sha256=HASHES[3],
    )


def test_rgb_only_forward_has_compact_heads_and_six_actions() -> None:
    output = temporal.TemporalModel()(torch.rand(2, 3, 64, 64))
    assert temporal.ACTION_NAMES == (
        "wait",
        "forward",
        "backward",
        "attack_hero",
        "attack_tower",
        "attack_crystal",
    )
    assert output["logits"].shape == (2, 6)
    assert output["hero_heatmaps"].shape == (2, 2, 8, 8)
    assert output["hud"].shape == (2, 4)
    with pytest.raises(TypeError):
        temporal.TemporalModel()(torch.rand(1, 3, 64, 64), legal_mask=torch.ones(6))  # type: ignore[call-arg]


def test_without_release_always_abstains_and_does_not_retain_state() -> None:
    coach = temporal.TemporalCoach()
    frame = torch.full((1, 5, 3, 64, 64), 0.4)
    output = coach(frame, torch.arange(5) * 100)
    assert output["advisory"] == [temporal.ABSTAIN]
    assert output["abstain_reason"] == ["NO_RELEASE"]
    assert output["metrics"]["release_binding_passed"] is False
    assert coach.model._runtime is None


def test_tamper_and_wrong_binding_fail_closed(tmp_path: Path) -> None:
    path = _release(tmp_path / "release.json")
    wrong = temporal.TemporalCoach(
        release_path=path,
        expected_model_sha256="f" * 64,
        expected_alignment_sha256=HASHES[1],
        expected_audit_sha256=HASHES[2],
        expected_config_sha256=HASHES[3],
    )
    assert wrong(torch.rand(1, 3, 64, 64))["abstain_reason"] == ["RELEASE_BINDING"]
    path.write_text(path.read_text().replace('"overall_pass":true', '"overall_pass":false'))
    assert _coach(path)(torch.rand(1, 3, 64, 64))["advisory"] == [temporal.ABSTAIN]


def test_valid_v5_release_still_abstains_without_v6_audit(tmp_path: Path) -> None:
    frames = torch.full((1, 5, 3, 64, 64), 0.4)
    timestamps = torch.arange(5) * 100
    open_output = _coach(_release(tmp_path / "release.json", 0.75))(frames, timestamps)
    assert open_output["advisory"] == [temporal.ABSTAIN]
    assert open_output["abstain_reason"] == ["V6_AUDIT_NOT_BOUND"]
    assert open_output["metrics"]["release_binding_passed"] is True


def test_pts_gap_segments_before_features_and_matches_fresh_segment() -> None:
    torch.manual_seed(7)
    joined = temporal.TemporalModel()
    torch.manual_seed(7)
    fresh = temporal.TemporalModel()
    frames = torch.full((1, 4, 3, 64, 64), 0.3)
    joined_output = joined(frames, torch.tensor([0, 100, 600, 700]))
    fresh_output = fresh(frames[:, 2:], torch.tensor([600, 700]))
    assert joined_output["reset_count"] == 1
    assert torch.equal(joined_output["frame_logits"][:, 2:], fresh_output["frame_logits"])
    assert torch.equal(
        joined_output["frame_tracking_quality"][:, 2:], fresh_output["frame_tracking_quality"]
    )


def test_tracker_uses_actual_timestamp_delta() -> None:
    output = temporal.TemporalModel()(torch.rand(1, 3, 3, 64, 64), torch.tensor([10, 130, 330]))
    assert output["frame_dt_ms"].tolist() == [[100.0, 120.0, 200.0]]


def test_tracking_gate_is_hard_and_missing_labels_fail_closed() -> None:
    passed = dict(label_count=300, pck=0.85, visibility_f1=0.90, hp_mae=0.10, skill_ready_f1=0.90)
    assert temporal.tracking_gate(**passed)
    assert not temporal.tracking_gate(**{**passed, "label_count": 299})
    for field, failing in (
        ("pck", 0.849),
        ("visibility_f1", 0.899),
        ("hp_mae", 0.101),
        ("skill_ready_f1", 0.899),
    ):
        assert not temporal.tracking_gate(**{**passed, field: failing})


def test_temporal_audit_gate_is_hard_and_missing_audit_fails_closed() -> None:
    passed = dict(
        sealed_count=200,
        overall_precision=0.85,
        unlocked_class_precision={"wait": 0.75},
        coverage=0.20,
        transition_false_advice=0.05,
        ood_false_advice=0.05,
        switch_reduction=0.50,
        median_delay_ms=300.0,
        p95_delay_ms=500.0,
        live_hz=10.0,
        live_p95_ms=100.0,
    )
    assert temporal.temporal_audit_gate(**passed)
    assert not temporal.temporal_audit_gate(**{**passed, "sealed_count": 199})
    for field, failing in (
        ("overall_precision", 0.849),
        ("coverage", 0.199),
        ("ood_false_advice", 0.051),
        ("switch_reduction", 0.499),
        ("p95_delay_ms", 501.0),
        ("live_hz", 9.99),
        ("live_p95_ms", 100.1),
    ):
        assert not temporal.temporal_audit_gate(**{**passed, field: failing})
