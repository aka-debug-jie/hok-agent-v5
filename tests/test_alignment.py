# ruff: noqa: E501
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import save as save_safetensors
from torchvision.models import resnet18

from hok_agent import alignment as a


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    value[field] = _sha(_json(value))
    return value


def _write(path: Path, value: object) -> Path:
    path.write_bytes(_json(value) + b"\n")
    return path


def _artifacts(tmp_path: Path) -> dict[str, object]:
    sessions = [_sha(f"s{i}".encode()) for i in range(12)]
    split_of = {session: ("train" if i < 8 else "dev" if i < 10 else "test") for i, session in enumerate(sessions)}
    frame, mask = np.full((16, 24, 3), 80, np.uint8), np.zeros((16, 24), np.bool_)
    shards, shard_rows = [], []
    for split in ("train", "dev", "test"):
        rows = []
        for session in (value for value in sessions if split_of[value] == split):
            times = (0, 100, 200) if session == sessions[0] else range(300) if session == sessions[10] else (0,)
            rows.extend({"session_hash": session, "timestamp_ms": time, "pts": time, "time_base": (1, 1000), "split": split, "source": "target", "frame": frame, "privacy_mask": mask} for time in times)
        made = a.write_npz_shards(rows, tmp_path / f"make-{split}", shard_size=len(rows))[0]
        path = made.replace(tmp_path / f"target-{split}.npz")
        shards.append(path)
        shard_rows.append({"path": path.name, "sha256": _sha(path.read_bytes()), "row_count": len(rows), "session_hashes": sorted({row["session_hash"] for row in rows}), "split": split, "source": "target"})
    manifest_value = {"schema_version": a.MANIFEST_SCHEMA, "sessions": [{"session_hash": session, "family_id": f"family-{i}", "parent_hash": None, "near_duplicate_hashes": [], "split": split_of[session]} for i, session in enumerate(sessions)], "shards": shard_rows}
    manifest = _write(tmp_path / "manifest.json", _seal(manifest_value, "manifest_sha256"))
    config_value = a.build_training_config(batch_size=2, epochs=1, mean_teacher_epochs=1)
    config = _write(tmp_path / "config.json", config_value)
    observation = {"side": "blue", "self_position": 2, "opponent_position": 8, "self_health": 5, "opponent_health": 5, "own_tower_health": 4, "enemy_tower_health": 4, "own_crystal_health": 6, "enemy_crystal_health": 6}
    source_frame, class_id = a.source_render_128_rgb(observation), 1
    source_alignment = _sha(_json([_sha(source_frame.tobytes()), class_id]))
    source_dataset = tmp_path / "source.npz"
    np.savez(source_dataset, frames=source_frame[None], class_id=np.asarray([class_id], np.int64), alignment_hash=np.asarray([source_alignment], dtype="<U64"))
    source_metadata_value = {"schema_version": a.SOURCE_SCHEMA, "manifest_sha256": manifest_value["manifest_sha256"], "config_sha256": config_value["config_sha256"], "dataset_path": source_dataset.name, "dataset_sha256": _sha(source_dataset.read_bytes()), "renderer_id": a.RENDERER_SPEC["id"], "renderer_sha256": a.source_renderer_hash(), "teacher_id": "causal-source-teacher-v1", "teacher_sha256": a.causal_source_teacher_hash(), "action_types": list(a.ACTION_TYPES), "action_schema_sha256": a.action_schema_hash(), "rows": [{"alignment_hash": source_alignment, "observation": observation, "legal_actions": ["wait", "forward"], "class_id": class_id}]}
    source_metadata = _write(tmp_path / "source.json", _seal(source_metadata_value, "source_metadata_sha256"))
    source_state = resnet18(weights=None, num_classes=6).state_dict()
    source_model_metadata = {"schema_version": a.MODEL_SCHEMA, "role": "v5_causal_source_teacher", "manifest_sha256": str(manifest_value["manifest_sha256"]), "config_sha256": str(config_value["config_sha256"]), "renderer_sha256": a.source_renderer_hash(), "teacher_sha256": a.causal_source_teacher_hash(), "action_schema_sha256": a.action_schema_hash(), "source_dataset_sha256": str(source_metadata_value["dataset_sha256"]), "source_metadata_sha256": str(source_metadata_value["source_metadata_sha256"])}
    source_model = tmp_path / "source.safetensors"
    source_model.write_bytes(save_safetensors(source_state, metadata=source_model_metadata))
    source_model_sha = _sha(source_model.read_bytes())
    adapted_metadata = {"schema_version": a.MODEL_SCHEMA, "role": "v5_simsiam_adapted", "manifest_sha256": str(manifest_value["manifest_sha256"]), "config_sha256": str(config_value["config_sha256"]), "renderer_sha256": a.source_renderer_hash(), "teacher_sha256": a.causal_source_teacher_hash(), "action_schema_sha256": a.action_schema_hash(), "source_model_sha256": source_model_sha, "source_dataset_sha256": str(source_metadata_value["dataset_sha256"]), "source_metadata_sha256": str(source_metadata_value["source_metadata_sha256"]), "collapse_metrics_sha256": "c" * 64}
    adapted_model = tmp_path / "adapted.safetensors"
    adapted_model.write_bytes(save_safetensors(a.ResNet18SimSiam(source_state).state_dict(), metadata=adapted_metadata))
    target = {(str(session), int(time)): str(alignment) for shard in a.load_npz_shards(shards) for session, time, alignment in zip(shard["session_hash"], shard["timestamp_ms"], shard["alignment_hash"], strict=True)}
    probability = [0.996, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008]
    prediction_rows = [{"session_hash": sessions[0], "anchor_timestamp_ms": 100, "frame_timestamp_ms": time, "frame_alignment_hash": target[(sessions[0], time)], "model_role": role, "view_id": view, "probs": probability, "ood_score": 0.0, "black_control_ok": True, "constant_control_ok": True, "cut": False} for role in ("source", "student") for view, time in (("t-100", 0), ("t", 100), ("t+100", 200))]
    predictions_value = {"schema_version": a.PREDICTION_SCHEMA, "manifest_sha256": manifest_value["manifest_sha256"], "source_model_sha256": source_model_sha, "adapted_model_sha256": _sha(adapted_model.read_bytes()), "config_sha256": config_value["config_sha256"], "rows": prediction_rows}
    predictions = _write(tmp_path / "predictions.json", _seal(predictions_value, "predictions_sha256"))
    return {"sessions": sessions, "split_of": split_of, "manifest_value": manifest_value, "manifest": manifest, "config_value": config_value, "config": config, "shards": shards, "source_dataset": source_dataset, "source_metadata": source_metadata, "source_model": source_model, "adapted_model": adapted_model, "predictions": predictions, "target": target}


def _mean_teacher(bundle: dict[str, object], tmp_path: Path) -> tuple[Path, Path, Path]:
    pseudo = tmp_path / "pseudo.npz"
    accepted, report = a.materialize_v5_pseudo(predictions_path=bundle["predictions"], source_metadata_path=bundle["source_metadata"], source_dataset_path=bundle["source_dataset"], manifest_path=bundle["manifest"], target_shards=bundle["shards"], config_path=bundle["config"], source_model_path=bundle["source_model"], adapted_model_path=bundle["adapted_model"], output_path=pseudo)  # type: ignore[arg-type]
    assert len(accepted.frames) == 1 and report.accepted == 1 and not report.filter_floor_met
    ema, ledger = tmp_path / "ema.safetensors", tmp_path / "round.json"
    result = a.run_mean_teacher_round(source_metadata_path=bundle["source_metadata"], source_dataset_path=bundle["source_dataset"], manifest_path=bundle["manifest"], target_shards=bundle["shards"], predictions_path=bundle["predictions"], pseudo_path=pseudo, source_model_path=bundle["source_model"], adapted_checkpoint=bundle["adapted_model"], config_path=bundle["config"], ema_checkpoint=ema, round_ledger=ledger)  # type: ignore[arg-type]
    assert result["round"] == 1 and result["collapse_status"] == a.COLLAPSE_BLOCK and ema.is_file() and ledger.is_file()
    with pytest.raises(a.AlignmentError, match="already exists"):
        a.run_mean_teacher_round(source_metadata_path=bundle["source_metadata"], source_dataset_path=bundle["source_dataset"], manifest_path=bundle["manifest"], target_shards=bundle["shards"], predictions_path=bundle["predictions"], pseudo_path=pseudo, source_model_path=bundle["source_model"], adapted_checkpoint=bundle["adapted_model"], config_path=bundle["config"], ema_checkpoint=ema, round_ledger=ledger)  # type: ignore[arg-type]
    return pseudo, ema, ledger


def test_causal_teacher_six_classes_and_renderer_has_no_tick_shortcut() -> None:
    base = {"side": "blue", "self_position": 2, "opponent_position": 8, "self_health": 6, "opponent_health": 6, "own_tower_health": 4, "enemy_tower_health": 4, "own_crystal_health": 6, "enemy_crystal_health": 6}
    cases = ((["wait"], 6, 6, "wait"), (["wait", "forward"], 6, 6, "forward"), (["wait", "backward"], 1, 6, "backward"), (["wait", "attack_hero"], 6, 1, "attack_hero"), (["wait", "attack_tower", "attack_hero"], 6, 6, "attack_tower"), (["wait", "attack_crystal", "attack_tower"], 6, 6, "attack_crystal"), (["wait", "attack_hero"], 1, 6, "wait"))
    for legal, self_hp, opponent_hp, expected in cases:
        observation = dict(base, self_health=self_hp, opponent_health=opponent_hp, tick=17)
        assert a._action_name(a.CausalSourceTeacher().select(observation, tuple(a._action_object(name) for name in legal))) == expected
    assert np.array_equal(a.source_render_128_rgb(dict(base, tick=1)), a.source_render_128_rgb(dict(base, tick=999)))


def test_strict_manifest_source_and_symlink_fail_closed(tmp_path: Path) -> None:
    bundle = _artifacts(tmp_path)
    manifest = a.load_v5_manifest(bundle["manifest"], bundle["shards"])  # type: ignore[arg-type]
    source = a.load_v5_source_dataset(bundle["source_metadata"], bundle["source_dataset"], bundle["manifest"], bundle["shards"], bundle["config"], bundle["source_model"])  # type: ignore[arg-type]
    assert len(manifest.session_splits) == 12 and source.frames.shape == (1, 128, 128, 3)
    bad = json.loads(json.dumps(bundle["manifest_value"]))
    bad["sessions"][0]["parent_hash"] = "f" * 64
    _write(tmp_path / "bad.json", _seal({key: value for key, value in bad.items() if key != "manifest_sha256"}, "manifest_sha256"))
    with pytest.raises(a.AlignmentError, match="known"):
        a.load_v5_manifest(tmp_path / "bad.json", bundle["shards"])  # type: ignore[arg-type]
    (tmp_path / "link.json").symlink_to(bundle["manifest"])
    with pytest.raises(a.AlignmentError, match="non-symlink"):
        a.load_v5_manifest(tmp_path / "link.json", bundle["shards"])  # type: ignore[arg-type]


def test_path_pseudo_mean_teacher_audit_and_bound_release(tmp_path: Path) -> None:
    bundle = _artifacts(tmp_path)
    _, ema, ledger = _mean_teacher(bundle, tmp_path)
    sessions, target = bundle["sessions"], bundle["target"]
    test_rows = [(sessions[10], time, target[(sessions[10], time)]) for time in range(300)]
    selection = "d" * 64
    sealed_rows = [{"clip_id": f"c{i:03d}", "session_hash": session, "timestamp_ms": time, "alignment_hash": alignment, "action": "wait", "accepted": True, "confidence": 0.999, "baselines": ["forward", "backward"]} for i, (session, time, alignment) in enumerate(test_rows)]
    sealed_value = {"schema_version": "hok-agent-v5-sealed-predictions-v1", "manifest_sha256": bundle["manifest_value"]["manifest_sha256"], "model_sha256": _sha(ema.read_bytes()), "config_sha256": bundle["config_value"]["config_sha256"], "selection_sha256": selection, "source_regression": {"accuracy_before": 0.9, "accuracy_after": 0.89, "recall_before": [0.9] * 6, "recall_after": [0.86] * 6}, "rows": sealed_rows}
    sealed = _write(tmp_path / "sealed.json", _seal(sealed_value, "predictions_sha256"))
    audit_rows = [{"clip_id": row["clip_id"], "session_hash": row["session_hash"], "timestamp_ms": row["timestamp_ms"], "alignment_hash": row["alignment_hash"], "reviewer": reviewer, "observed_action": "wait", "validity": True} for row in sealed_rows for reviewer in ("r1", "r2")]
    audit_value = {"schema_version": a.AUDIT_SCHEMA, "manifest_sha256": bundle["manifest_value"]["manifest_sha256"], "selection_sha256": selection, "rows": audit_rows}
    audit = _write(tmp_path / "audit.json", _seal(audit_value, "audit_sha256"))
    release_path = tmp_path / "release.json"
    gate = a.path_only_v5_release_gate(manifest_path=bundle["manifest"], target_shards=bundle["shards"], sealed_predictions_path=sealed, sealed_audit_path=audit, model_path=ema, mean_teacher_ledger_path=ledger, config_path=bundle["config"], release_path=release_path)  # type: ignore[arg-type]
    assert not gate.passed and gate.collapse_status == a.COLLAPSE_BLOCK and gate.release_path is None and not release_path.exists()
    duplicate_value = json.loads(json.dumps(sealed_value))
    duplicate_value["rows"][-1].update({key: duplicate_value["rows"][0][key] for key in ("session_hash", "timestamp_ms", "alignment_hash")})
    duplicate = _write(tmp_path / "duplicate.json", _seal({key: value for key, value in duplicate_value.items() if key != "predictions_sha256"}, "predictions_sha256"))
    with pytest.raises(a.AlignmentError, match="target/test"):
        a.path_only_v5_release_gate(manifest_path=bundle["manifest"], target_shards=bundle["shards"], sealed_predictions_path=duplicate, sealed_audit_path=audit, model_path=ema, mean_teacher_ledger_path=ledger, config_path=bundle["config"], release_path=release_path)  # type: ignore[arg-type]
    thresholds = {"kappa": 0.70, "overall_precision": 0.85, "per_class_precision": 0.75, "coverage": 0.30, "ood_false_accept": 0.05, "baseline_delta": 0.05, "source_accuracy_drop": 0.02, "source_recall_drop": 0.05}
    release_value = {"schema_version": a.RELEASE_SCHEMA, "model_sha256": _sha(ema.read_bytes()), "alignment_sha256": bundle["manifest_value"]["manifest_sha256"], "audit_sha256": audit_value["audit_sha256"], "config_sha256": bundle["config_value"]["config_sha256"], "overall_pass": True, "allowed_classes": ["wait"], "class_thresholds": {"wait": 0.9}, "thresholds": thresholds, "thresholds_hash": _sha(_json(thresholds))}
    release = _write(tmp_path / "bound.json", _seal(release_value, "release_sha256"))
    with pytest.raises(a.AlignmentError, match=a.COLLAPSE_BLOCK):
        a.load_bound_v5_release(release, ema)
    assert "audit" not in inspect.signature(a.train_shallow_simsiam).parameters and "reviewer" not in inspect.signature(a.run_mean_teacher_round).parameters
