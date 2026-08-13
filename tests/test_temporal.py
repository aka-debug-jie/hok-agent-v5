# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from hok_agent import temporal


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return path


def _v5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: float = 0.75) -> tuple[Path, Path, temporal.alignment.BoundV5Release]:
    release, model = tmp_path / "v5-release.json", tmp_path / "v5-model.safetensors"
    release.write_bytes(b"v5-release")
    model.write_bytes(b"v5-model")
    binding = temporal.alignment.BoundV5Release(temporal._file_sha(release), temporal._file_sha(model), temporal.ACTION_NAMES, {name: threshold for name in temporal.ACTION_NAMES})
    def load_bound(release_path: Path, model_path: Path) -> object:
        assert (release_path, model_path) == (release, model)
        return SimpleNamespace(**binding.__dict__)
    monkeypatch.setattr(temporal.alignment, "load_bound_v5_release", load_bound, raising=False)
    return release, model, binding


def _checkpoint(path: Path, binding: temporal.alignment.BoundV5Release, *, ood_bias: float = -10.0, extra: bool = False) -> Path:
    model = temporal.TemporalModel()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.visual_encoder.heatmap_head.bias.fill_(10.0)
        model.logit_head.bias[0] = 10.0
        model.ood_head.bias.fill_(ood_bias)
    metadata = temporal._checkpoint_metadata(binding, "a" * 64, 7)
    if extra:
        metadata["summary"] = "untrusted"
    save_file(model.state_dict(), path, metadata=metadata)
    return path


def _tracking(path: Path, checkpoint_sha: str, *, count: int = 300, bad: bool = False, summary: bool = False, split_leak: bool = False) -> Path:
    rows = []
    for index in range(count):
        truth = [[0.1, 0.2], [0.8, 0.7]]
        split = "train" if index < 180 else "dev" if index < 240 else "test"
        rows.append({"frame_id": str(index), "session_hash": ("b" if split_leak else {"train": "b", "dev": "c", "test": "d"}[split]) * 64, "split": split, "predicted_centers": [[1.0, 1.0], [1.0, 1.0]] if bad else truth, "truth_centers": truth, "predicted_visibility": [True, True], "truth_visibility": [True, True], "predicted_hp": [0.4, 0.8], "truth_hp": [0.4, 0.8], "predicted_skill_ready": [index % 2 == 0, True], "truth_skill_ready": [index % 2 == 0, True]})
    payload: dict[str, object] = {"schema_version": temporal.TRACKING_SCHEMA, "checkpoint_sha256": checkpoint_sha, "rows": rows}
    if summary:
        payload["pck"] = 1.0
    return _write(path, payload)


def _audit(path: Path, checkpoint_sha: str, *, count: int = 200, confidence: float = 0.99) -> Path:
    rows = []
    for index in range(count):
        is_ood = index % 20 == 0
        def prediction(timestamp: int, ood: bool = is_ood) -> dict[str, object]:
            return {"timestamp_ms": timestamp, "action": "wait", "confidence": confidence, "ood_score": 1.0 if ood else 0.0, "tracking_quality": 1.0, "stable": True, "latency_ms": 10.0}
        annotations = [{"reviewer": reviewer, "observed_action": "wait", "validity": not is_ood} for reviewer in ("r1", "r2")]
        rows.append({"clip_id": str(index), "session_hash": "c" * 64, "annotations": annotations, "transition": index % 5 == 1, "ood": is_ood, "reference_event_ms": 0, "baseline_actions": ["wait", "forward"], "predictions": [prediction(0), prediction(100)]})
    return _write(path, {"schema_version": temporal.AUDIT_SCHEMA, "checkpoint_sha256": checkpoint_sha, "rows": rows})


def _artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, threshold: float = 0.75, tracking_count: int = 300, audit_count: int = 200, bad_tracking: bool = False, summary: bool = False, split_leak: bool = False, ood_bias: float = -10.0) -> tuple[dict[str, Path], temporal.alignment.BoundV5Release]:
    v5_release, v5_model, binding = _v5(tmp_path, monkeypatch, threshold)
    checkpoint = _checkpoint(tmp_path / "v6.safetensors", binding, ood_bias=ood_bias)
    checkpoint_sha = temporal._file_sha(checkpoint)
    paths = {"v5_release_path": v5_release, "v5_model_path": v5_model, "checkpoint_path": checkpoint, "tracking_evidence_path": _tracking(tmp_path / "tracking.json", checkpoint_sha, count=tracking_count, bad=bad_tracking, summary=summary, split_leak=split_leak), "temporal_audit_path": _audit(tmp_path / "audit.json", checkpoint_sha, count=audit_count, confidence=threshold), "temporal_release_path": tmp_path / "release.json"}
    return paths, binding


def _release(paths: dict[str, Path]) -> dict[str, object]:
    return temporal.create_v6_release(v5_release_path=paths["v5_release_path"], v5_model_path=paths["v5_model_path"], checkpoint_path=paths["checkpoint_path"], tracking_evidence_path=paths["tracking_evidence_path"], temporal_audit_path=paths["temporal_audit_path"], release_path=paths["temporal_release_path"])


def test_rgb_model_is_causal_six_class_and_tracks_actual_pts() -> None:
    output = temporal.TemporalModel()(torch.rand(1, 3, 3, 64, 64), torch.tensor([10, 130, 330]))
    assert output["logits"].shape == (1, 6)
    assert output["hero_heatmaps"].shape == (1, 2, 8, 8)
    assert output["hud"].shape == (1, 4)
    assert output["frame_dt_ms"].tolist() == [[100.0, 120.0, 200.0]]


def test_missing_any_path_abstains_without_forward_or_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(temporal.TemporalModel, "forward", lambda *args, **kwargs: pytest.fail("forward called"))
    output = temporal.TemporalCoach()(torch.zeros(2, 3, 64, 64))
    assert output["advisory"] == [temporal.ABSTAIN, temporal.ABSTAIN]
    assert output["control_output"] is False
    assert output["metrics"]["v6_release_binding_passed"] is False


@pytest.mark.parametrize("change", ["tracking_count", "audit_count", "bad_tracking", "summary", "split_leak"])
def test_raw_evidence_is_exact_and_recomputed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    values: dict[str, Any] = {"tracking_count": 300, "audit_count": 200, "bad_tracking": False, "summary": False, "split_leak": False}
    values[change] = {"tracking_count": 299, "audit_count": 199, "bad_tracking": True, "summary": True, "split_leak": True}[change]
    paths, _ = _artifacts(tmp_path, monkeypatch, **values)
    with pytest.raises(temporal.TemporalError):
        _release(paths)


def test_checkpoint_metadata_and_state_are_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, binding = _v5(tmp_path, monkeypatch)
    path = _checkpoint(tmp_path / "bad.safetensors", binding, extra=True)
    with pytest.raises(temporal.TemporalError):
        temporal._load_checkpoint(path, binding)
    metadata = temporal._checkpoint_metadata(binding, "a" * 64, 7)
    save_file({"unexpected": torch.zeros(1)}, path, metadata=metadata)
    with pytest.raises(temporal.TemporalError):
        temporal._load_checkpoint(path, binding)


def test_release_is_exclusive_self_hashed_and_v5_conservative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch)
    payload = _release(paths)
    assert payload["allowed_classes"] == ["wait"]
    assert payload["class_thresholds"] == {"wait": 0.75}
    signature = payload.pop("release_sha256")
    assert signature == temporal._sha(temporal._json(payload).encode())
    with pytest.raises(temporal.TemporalError):
        _release(paths)


def test_valid_runtime_is_eval_inference_only_and_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch)
    _release(paths)
    seen: list[bool] = []
    original = temporal.TemporalModel.forward
    def wrapped(model: temporal.TemporalModel, *args: object, **kwargs: object) -> dict[str, object]:
        seen.append(torch.is_inference_mode_enabled())
        return original(model, *args, **kwargs)  # type: ignore[arg-type,return-value]
    monkeypatch.setattr(temporal.TemporalModel, "forward", wrapped)
    coach = temporal.TemporalCoach(**paths)
    output = coach(torch.full((1, 5, 3, 64, 64), 0.4), torch.arange(5) * 100)
    assert output["advisory"] == ["wait"] and output["control_output"] is False
    assert seen == [True] and coach.model is not None and not coach.model.training


@pytest.mark.parametrize(("threshold", "ood_bias", "reason"), [(1.0, -10.0, "LOW_SCORE"), (0.75, 10.0, "OOD")])
def test_runtime_reapplies_release_and_ood_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: float, ood_bias: float, reason: str) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch, threshold=threshold, ood_bias=ood_bias)
    _release(paths)
    output = temporal.TemporalCoach(**paths)(torch.zeros(1, 5, 3, 64, 64), torch.arange(5) * 100)
    assert output["advisory"] == [temporal.ABSTAIN]
    assert output["abstain_reason"] == [reason]
