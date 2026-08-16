# ruff: noqa: E501, E701, E702
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from hok_agent import alignment, temporal


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8'); return path
def _v5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: float=0.75) -> tuple[dict[str, Any], temporal._V5Binding]:
    release, model, manifest, pre_ingest, privacy, attestation, confirmation, shard = (tmp_path / name for name in ('v5-release.json', 'v5-model.safetensors', 'manifest.json', 'pre-ingest.json', 'privacy.json', 'owner.json', 'confirmation.json', 'target.npz'))
    for (path, data) in ((release, b'v5-release'), (model, b'v5-model'), (manifest, b'manifest-v2'), (pre_ingest, b'pre-ingest'), (privacy, b'privacy'), (attestation, b'attestation'), (confirmation, b'confirmation'), (shard, b'sealed-shard')):
        path.write_bytes(data)
    sessions = [temporal._sha(f'session-{index}'.encode()) for index in range(12)]; splits = {session: 'train' if index < 8 else 'dev' if index < 10 else 'test' for index, session in enumerate(sessions)}; binding = temporal._V5Binding(release_sha256=temporal._file_sha(release), model_sha256=temporal._file_sha(model), manifest_sha256='a' * 64, split_binding_sha256=alignment.split_binding_hash(splits), pre_ingest_sha256='c' * 64, session_splits=splits, allowed_classes=temporal.ACTION_NAMES, class_thresholds={name: threshold for name in temporal.ACTION_NAMES}); expected = {path: path.read_bytes() for path in (manifest, pre_ingest, privacy, attestation, confirmation, shard)}
    def load_bound(release_path: Path, model_path: Path) -> object:
        assert (release_path, model_path) == (release, model); return binding
    def load_manifest(manifest_path: Path, pre_path: Path, privacy_path: Path, attestation_path: Path, confirmation_path: Path, shard_paths: Sequence[Path]) -> object:
        assert (manifest_path, pre_path, privacy_path, attestation_path, confirmation_path, tuple(shard_paths)) == (manifest, pre_ingest, privacy, attestation, confirmation, (shard,))
        if any((path.read_bytes() != data for path, data in expected.items())):
            raise alignment.AlignmentError('V5 path evidence changed')
        return SimpleNamespace(manifest_sha256=binding.manifest_sha256, split_binding_sha256=binding.split_binding_sha256, pre_ingest_sha256=binding.pre_ingest_sha256, session_splits=dict(splits))
    monkeypatch.setattr(alignment, 'load_bound_v5_release', load_bound, raising=False); monkeypatch.setattr(alignment, 'load_v5_manifest', load_manifest); return ({'v5_release_path': release, 'v5_model_path': model, 'v5_manifest_path': manifest, 'v5_pre_ingest_path': pre_ingest, 'v5_privacy_context_path': privacy, 'v5_owner_attestation_path': attestation, 'v5_owner_component_confirmation_path': confirmation, 'v5_shard_paths': (shard,)}, binding)
def _tracking_rows(count: int, session_splits: dict[str, str], *, split_leak: bool=False) -> list[tuple[str, str, str]]:
    by_split = {split: [session for session, value in session_splits.items() if value == split] for split in ('train', 'dev', 'test')}; rows = []
    for index in range(count):
        split = 'train' if index < 180 else 'dev' if index < 240 else 'test'; session = by_split['train'][0] if split_leak else by_split[split][index % len(by_split[split])]; rows.append((split, session, str(index)))
    return rows
def _tracking_split(path: Path, manifest: str, split_binding: str, rows: list[tuple[str, str, str]], *, session_splits: dict[str, str] | None=None) -> Path:
    mapping = session_splits or {session: split for split, session, _ in rows}; return _write(path, {'schema_version': temporal.TRACKING_SPLIT_SCHEMA, 'v5_manifest_sha256': manifest, 'v5_split_binding_sha256': split_binding, 'session_splits': mapping, 'tracking_row_identities': [f'{session}:{frame_id}' for _, session, frame_id in rows]})
def _tracking(path: Path, checkpoint_sha: str, rows: list[tuple[str, str, str]], *, bad: bool=False, summary: bool=False) -> Path:
    data = []
    for (split, session, frame_id) in rows:
        truth = [[0.1, 0.2], [0.8, 0.7]]; row = {'frame_id': frame_id, 'session_hash': session, 'split': split, 'predicted_centers': [[1.0, 1.0], [1.0, 1.0]] if bad else truth, 'truth_centers': truth, 'predicted_visibility': [True, True], 'truth_visibility': [True, True], 'predicted_hp': [0.4, 0.8], 'truth_hp': [0.4, 0.8], 'predicted_skill_ready': [int(frame_id) % 2 == 0, True], 'truth_skill_ready': [int(frame_id) % 2 == 0, True]}; data.append(row)
    payload: dict[str, object] = {'schema_version': temporal.TRACKING_SCHEMA, 'checkpoint_sha256': checkpoint_sha, 'rows': data}
    if summary:
        payload['pck'] = 1.0
    return _write(path, payload)
def _audit(path: Path, checkpoint_sha: str, *, count: int=200, confidence: float=0.99, session: str='d' * 64) -> Path:
    rows = []
    for index in range(count):
        is_ood = index % 20 == 0
        def prediction(timestamp: int, ood: bool=is_ood) -> dict[str, object]:
            return {'timestamp_ms': timestamp, 'action': 'wait', 'confidence': confidence, 'ood_score': 1.0 if ood else 0.0, 'tracking_quality': 1.0, 'stable': True, 'latency_ms': 10.0}
        annotations = [{'reviewer': reviewer, 'observed_action': 'wait', 'validity': not is_ood} for reviewer in ('r1', 'r2')]; rows.append({'clip_id': str(index), 'session_hash': session, 'annotations': annotations, 'transition': index % 5 == 1, 'ood': is_ood, 'reference_event_ms': 0, 'baseline_actions': ['wait', 'forward'], 'predictions': [prediction(0), prediction(100)]})
    return _write(path, {'schema_version': temporal.AUDIT_SCHEMA, 'checkpoint_sha256': checkpoint_sha, 'rows': rows})
def _checkpoint(path: Path, binding: temporal._V5Binding, training_artifact: Path, tracking_split: Path, *, ood_bias: float=-10.0, extra: bool=False) -> Path:
    model = temporal.TemporalModel()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        assert model.visual_encoder.heatmap_head.bias is not None; model.visual_encoder.heatmap_head.bias.fill_(10.0); model.logit_head.bias[0] = 10.0; model.ood_head.bias.fill_(ood_bias)
    metadata = temporal._checkpoint_metadata(binding, training_artifact, tracking_split, 7)
    if extra:
        metadata['summary'] = 'untrusted'
    save_file(model.state_dict(), path, metadata=metadata); return path
def _artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, threshold: float=0.75, tracking_count: int=300, audit_count: int=200, bad_tracking: bool=False, summary: bool=False, split_leak: bool=False, ood_bias: float=-10.0) -> tuple[dict[str, Any], temporal._V5Binding]:
    paths, binding = _v5(tmp_path, monkeypatch, threshold); training_artifact = tmp_path / 'training-artifact.bin'; training_artifact.write_bytes(b'training-artifact'); rows = _tracking_rows(tracking_count, binding.session_splits, split_leak=split_leak); split_path = _tracking_split(tmp_path / 'tracking_split.json', binding.manifest_sha256, binding.split_binding_sha256, rows, session_splits=binding.session_splits); checkpoint = _checkpoint(tmp_path / 'v6.safetensors', binding, training_artifact, split_path, ood_bias=ood_bias); checkpoint_sha = temporal._file_sha(checkpoint); paths.update({'training_artifact_path': training_artifact, 'checkpoint_path': checkpoint, 'tracking_split_path': split_path, 'tracking_evidence_path': _tracking(tmp_path / 'tracking.json', checkpoint_sha, rows, bad=bad_tracking, summary=summary), 'temporal_audit_path': _audit(tmp_path / 'audit.json', checkpoint_sha, count=audit_count, confidence=threshold, session=next((session for session, split in binding.session_splits.items() if split == 'test'))), 'temporal_release_path': tmp_path / 'release.json'}); return (paths, binding)
def _release(paths: dict[str, Any]) -> dict[str, object]:
    return temporal.create_v6_release(v5_release_path=paths['v5_release_path'], v5_model_path=paths['v5_model_path'], v5_manifest_path=paths['v5_manifest_path'], v5_shard_paths=paths['v5_shard_paths'], v5_pre_ingest_path=paths['v5_pre_ingest_path'], v5_privacy_context_path=paths['v5_privacy_context_path'], v5_owner_attestation_path=paths['v5_owner_attestation_path'], v5_owner_component_confirmation_path=paths['v5_owner_component_confirmation_path'], training_artifact_path=paths['training_artifact_path'], checkpoint_path=paths['checkpoint_path'], tracking_split_path=paths['tracking_split_path'], tracking_evidence_path=paths['tracking_evidence_path'], temporal_audit_path=paths['temporal_audit_path'], release_path=paths['temporal_release_path'])
def test_rgb_model_is_causal_six_class_and_tracks_actual_pts() -> None:
    output = temporal.TemporalModel()(torch.rand(1, 3, 3, 64, 64), torch.tensor([10, 130, 330])); assert output['logits'].shape == (1, 6); assert output['hero_heatmaps'].shape == (1, 2, 8, 8); assert output['hud'].shape == (1, 4); assert output['frame_dt_ms'].tolist() == [[100.0, 120.0, 200.0]]
def test_missing_any_path_abstains_without_forward_or_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(temporal.TemporalModel, 'forward', lambda *args, **kwargs: pytest.fail('forward called')); output = temporal.TemporalCoach()(torch.zeros(2, 3, 64, 64)); assert output['advisory'] == [temporal.ABSTAIN, temporal.ABSTAIN]; assert output['control_output'] is False; assert output['metrics']['v6_release_binding_passed'] is False
@pytest.mark.parametrize('change', ['tracking_count', 'audit_count', 'bad_tracking', 'summary', 'split_leak'])
def test_raw_evidence_is_exact_and_recomputed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    values: dict[str, Any] = {'tracking_count': 300, 'audit_count': 200, 'bad_tracking': False, 'summary': False, 'split_leak': False}; values[change] = {'tracking_count': 299, 'audit_count': 199, 'bad_tracking': True, 'summary': True, 'split_leak': True}[change]; paths, _ = _artifacts(tmp_path, monkeypatch, **values)
    with pytest.raises(temporal.TemporalError):
        _release(paths)


def test_public_output_is_frozen_abstain_no_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch)
    def leaked_forward(*args: object, **kwargs: object) -> dict[str, Any]:
        return {"advisory": ["forward"], "abstain_reason": [""], "control_output": True, "metrics": {}, "logits": torch.randn(1, 6), "hero_heatmaps": torch.zeros(1, 2, 8, 8), "hero_visibility": torch.ones(1, 2), "hud": torch.zeros(1, 4)}
    monkeypatch.setattr(temporal.TemporalModel, "forward", leaked_forward)
    _release(paths)
    output = temporal.TemporalCoach(**paths)(torch.zeros((1, 4, 3, 64, 64)))
    assert output["advisory"] == [temporal.ABSTAIN]
    assert output["control_output"] is False
    assert "logits" not in output and "hero_heatmaps" not in output and "hero_visibility" not in output and "hud" not in output and output["abstain_reason"] == [alignment.COLLAPSE_BLOCK]


@pytest.mark.parametrize("shape", [(3, 64, 64), (1, 64, 64), (1, 4, 64, 64), (1, 2, 4, 64, 64)])
def test_malformed_input_rejects_or_abstains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch)
    _release(paths)
    coach = temporal.TemporalCoach(**paths)
    with pytest.raises(ValueError):
        coach(torch.zeros(shape))

def test_checkpoint_metadata_and_state_are_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, binding = _artifacts(tmp_path, monkeypatch); path = _checkpoint(tmp_path / 'bad.safetensors', binding, paths['training_artifact_path'], paths['tracking_split_path'], extra=True)
    with pytest.raises(temporal.TemporalError):
        temporal._load_checkpoint(path, binding, paths['training_artifact_path'], paths['tracking_split_path'])
    metadata = temporal._checkpoint_metadata(binding, paths['training_artifact_path'], paths['tracking_split_path'], 7); save_file({'unexpected': torch.zeros(1)}, path, metadata=metadata)
    with pytest.raises(temporal.TemporalError):
        temporal._load_checkpoint(path, binding, paths['training_artifact_path'], paths['tracking_split_path'])
def test_release_is_exclusive_self_hashed_and_v5_conservative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch); payload = _release(paths); assert payload['overall_pass'] is False; assert payload['allowed_classes'] == [] and payload['class_thresholds'] == {}; signature = payload.pop('release_sha256'); assert signature == temporal._sha(temporal._json(payload).encode())
    with pytest.raises(temporal.TemporalError):
        _release(paths)
def test_valid_runtime_is_eval_inference_only_and_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch); _release(paths); seen: list[bool] = []; original = temporal.TemporalModel.forward
    def wrapped(model: temporal.TemporalModel, *args: object, **kwargs: object) -> dict[str, object]:
        seen.append(torch.is_inference_mode_enabled()); return original(model, *args, **kwargs)
    monkeypatch.setattr(temporal.TemporalModel, 'forward', wrapped); coach = temporal.TemporalCoach(**paths); output = coach(torch.full((1, 5, 3, 64, 64), 0.4), torch.arange(5) * 100); assert output['advisory'] == [temporal.ABSTAIN] and output['abstain_reason'] == [alignment.COLLAPSE_BLOCK]; assert seen == [True] and coach.model is not None and (not coach.model.training)
def test_temporal_audit_requires_v5_test_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, binding = _artifacts(tmp_path, monkeypatch); rows = _tracking_rows(300, binding.session_splits); paths['tracking_evidence_path'] = _tracking(tmp_path / 'tracking.json', temporal._file_sha(paths['checkpoint_path']), rows); paths['temporal_audit_path'] = _audit(tmp_path / 'audit.json', temporal._file_sha(paths['checkpoint_path']), session=next((session for session, split in binding.session_splits.items() if split == 'train')))
    with pytest.raises(temporal.TemporalError):
        _release(paths)
    coach = temporal.TemporalCoach(**paths); output = coach(torch.zeros(1, 5, 3, 64, 64), torch.arange(5) * 100); assert output['advisory'] == [temporal.ABSTAIN]; assert output['metrics']['v6_release_binding_passed'] is False
def test_tracking_split_mismatch_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, binding = _artifacts(tmp_path, monkeypatch); rows = _tracking_rows(300, binding.session_splits); bad_split = dict(binding.session_splits); bad_split[next((session for session, split in bad_split.items() if split == 'test'))] = 'train'; _tracking_split(paths['tracking_split_path'], binding.manifest_sha256, binding.split_binding_sha256, rows, session_splits=bad_split); _checkpoint(paths['checkpoint_path'], binding, paths['training_artifact_path'], paths['tracking_split_path']); checkpoint_sha = temporal._file_sha(paths['checkpoint_path']); paths['tracking_evidence_path'] = _tracking(paths['tracking_evidence_path'], checkpoint_sha, rows); paths['temporal_audit_path'] = _audit(paths['temporal_audit_path'], checkpoint_sha, session=next((session for session, split in binding.session_splits.items() if split == 'test')))
    with pytest.raises(temporal.TemporalError):
        _release(paths)
def test_checkpoint_training_artifact_tamper_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch); paths['training_artifact_path'].write_bytes(b'tampered-artifact')
    with pytest.raises(temporal.TemporalError):
        _release(paths)
@pytest.mark.parametrize('artifact', ['v5_manifest_path', 'v5_pre_ingest_path', 'v5_privacy_context_path', 'v5_owner_attestation_path', 'v5_owner_component_confirmation_path'])
def test_v5_path_tamper_clears_coach_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch); _release(paths); coach = temporal.TemporalCoach(**paths); first = coach(torch.zeros(1, 5, 3, 64, 64), torch.arange(5) * 100); assert first['advisory'] == [temporal.ABSTAIN] and first['metrics']['v6_release_binding_passed'] is True; generation = coach._reset_generation; paths[artifact].write_bytes(b'tampered'); changed = coach(torch.zeros(1, 3, 64, 64), 500); assert changed['advisory'] == [temporal.ABSTAIN] and changed['metrics']['v6_release_binding_passed'] is False; assert coach.model is None and coach._reset_generation > generation
def test_identical_reverification_preserves_temporal_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _artifacts(tmp_path, monkeypatch); _release(paths); coach = temporal.TemporalCoach(**paths); first = coach(torch.zeros(1, 5, 3, 64, 64), torch.arange(5) * 100); model, generation = (coach.model, coach._reset_generation); second = coach(torch.zeros(1, 3, 64, 64), 500); assert first['advisory'] == second['advisory'] == [temporal.ABSTAIN]; assert coach.model is model and coach._reset_generation == generation; assert coach.model is not None and coach.model._runtime is not None and (coach.model._runtime[0].last_timestamp_ms == 500)
