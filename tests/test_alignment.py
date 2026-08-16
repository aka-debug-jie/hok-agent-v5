# ruff: noqa: E501, E701, E702
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save as save_safetensors
from torchvision.models import resnet18

from hok_agent import alignment as a
from hok_agent import cli
from hok_agent import pre_ingest as p


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


def _pre_ingest(
    tmp_path: Path, sessions: list[str], *, relation: str | None = None
) -> tuple[Path, dict[str, str]]:
    groups = [[value] for value in sessions]
    component_of = {
        candidate: _sha(_json(["component-v1", sorted(group)]))
        for group in groups
        for candidate in group
    }
    relations = []
    if relation:
        evidence = {
            "whole_file_sha256_equal": False,
            "compared_samples": 4,
            "matched_samples": 4,
            "shorter_coverage_ppm": 1000000,
            "match_fraction_ppm": 1000000,
            "offset_samples": 0,
            "median_dhash_hamming": 0,
            "median_luma_delta": 0,
            "evidence_sha256": "e" * 64,
        }
        relations = [
            {
                "left_candidate_id": left,
                "right_candidate_id": right,
                "relation": relation,
                "evidence": evidence,
            }
            for session in sessions[1:]
            if relation == "near_duplicate" or session == sessions[1]
            for left, right in [sorted((sessions[0], session))]
        ]
    blockers: list[str] = []
    disposition = p.READY if len(groups) >= p.MIN_CANDIDATES else "BLOCKED_LT_12_COMPONENTS"
    value = {
        "schema_version": p.SCHEMA,
        "algorithm_spec": p.ALGORITHM_SPEC,
        "relationship_mode": p.RELATIONSHIP_MODE,
        "candidate_count": len(sessions),
        "component_count": len(groups),
        "candidates": [
            {"candidate_id": session, "component_id": component_of[session], "pts_range_us": [0, 1]}
            for session in sessions
        ],
        "relations": relations,
        "component_of": component_of,
        "uncertain_relation_count": 0,
        "blockers": blockers,
        "review_status": p.REVIEW_STATUS,
        "disposition": disposition,
    }
    return (
        _write(tmp_path / f"pre-{relation or 'ready'}.json", _seal(value, "pre_ingest_sha256")),
        component_of,
    )


def _artifacts(tmp_path: Path) -> dict[str, object]:
    sessions = [_sha(f"s{i}".encode()) for i in range(12)]
    pre_ingest, component_of = _pre_ingest(tmp_path, sessions)
    pre_ingest_sha = p.load_pre_ingest(pre_ingest).pre_ingest_sha256
    component_splits = a._all_clean_components_lexicographic_ceil15pct_v1(
        sorted(set(component_of.values()))
    )
    split_of = {session: component_splits[component_of[session]] for session in sessions}
    train_sessions = [session for session in sessions if split_of[session] == "train"]
    anchor_session = train_sessions[0]
    long_train_session = train_sessions[1]
    owner_value = _seal(
        {
            "schema_version": a.OWNER_ATTESTATION_SCHEMA,
            "recording_owner": True,
            "local_research_only": True,
            "zero_redaction_authorized": True,
            "redistribution": False,
        },
        "owner_attestation_sha256",
    )
    owner = _write(tmp_path / "owner.json", owner_value)
    confirmation_value = _seal(
        {
            "schema_version": a.COMPONENT_COHORT_SCHEMA,
            "pre_ingest_sha256": pre_ingest_sha,
            "owner_attestation_sha256": owner_value["owner_attestation_sha256"],
            "component_hashes": sorted(set(component_of.values())),
        },
        "component_cohort_sha256",
    )
    confirmation = _write(tmp_path / "confirmation.json", confirmation_value)
    reviews = [
        _seal(
            {
                "session_hash": session,
                "component_hash": component_of[session],
                "zero_redaction_authorized": True,
            },
            "privacy_review_sha256",
        )
        for session in sessions
    ]
    privacy_value = _seal(
        {
            "schema_version": a.PRIVACY_CONTEXT_SCHEMA,
            "transform": a.PRIVACY_SPEC,
            "privacy_transform_sha256": a.privacy_transform_hash(),
            "owner_attestation_sha256": owner_value["owner_attestation_sha256"],
            "reviews": reviews,
        },
        "privacy_context_sha256",
    )
    privacy = _write(tmp_path / "privacy.json", privacy_value)
    frame = np.full((16, 24, 3), 80, np.uint8)
    shards, shard_rows = ([], [])
    for split in ("train", "dev", "test"):
        rows = []
        for session in (value for value in sessions if split_of[value] == split):
            if session == anchor_session:
                times = (0, 100, 200)
            elif session == long_train_session:
                times = range(300)
            else:
                times = (0,)
            rows.extend(
                {
                    "session_hash": session,
                    "timestamp_ms": time,
                    "pts": time,
                    "time_base": (1, 1000),
                    "rotation_degrees": 90,
                    "split": split,
                    "source": "target",
                    "frame": frame,
                }
                for time in times
            )
        made = a.write_npz_shards(
            rows,
            tmp_path / f"make-{split}",
            shard_size=len(rows),
            pre_ingest_path=pre_ingest,
            privacy_context_path=privacy,
            owner_attestation_path=owner,
            owner_component_confirmation_path=confirmation,
        )[0]
        path = made.replace(tmp_path / f"target-{split}.npz")
        shards.append(path)
        shard_rows.append(
            {
                "path": path.name,
                "sha256": _sha(path.read_bytes()),
                "row_count": len(rows),
                "session_hashes": sorted({row["session_hash"] for row in rows}),
                "split": split,
                "source": "target",
            }
        )
    manifest_value = {
        "schema_version": a.MANIFEST_SCHEMA,
        "pre_ingest_sha256": pre_ingest_sha,
        "component_cohort_sha256": confirmation_value["component_cohort_sha256"],
        "privacy_context_sha256": privacy_value["privacy_context_sha256"],
        "privacy_transform_sha256": privacy_value["privacy_transform_sha256"],
        "owner_attestation_sha256": owner_value["owner_attestation_sha256"],
        "split_binding_sha256": a.split_binding_hash(split_of),
        "sessions": [
            {
                "session_hash": session,
                "component_hash": component_of[session],
                "parent_hash": None,
                "near_duplicate_hashes": [],
                "split": split_of[session],
                "privacy_review_sha256": reviews[i]["privacy_review_sha256"],
            }
            for i, session in enumerate(sessions)
        ],
        "shards": shard_rows,
    }
    manifest = _write(tmp_path / "manifest.json", _seal(manifest_value, "manifest_sha256"))
    config_value = a.build_training_config(batch_size=2, epochs=1, mean_teacher_epochs=1)
    config = _write(tmp_path / "config.json", config_value)
    observation = {
        "side": "blue",
        "self_position": 2,
        "opponent_position": 8,
        "self_health": 5,
        "opponent_health": 5,
        "own_tower_health": 4,
        "enemy_tower_health": 4,
        "own_crystal_health": 6,
        "enemy_crystal_health": 6,
    }
    source_frame, class_id = (a.source_render_128_rgb(observation), 1)
    source_alignment = _sha(_json([_sha(source_frame.tobytes()), class_id]))
    source_dataset = tmp_path / "source.npz"
    np.savez(
        source_dataset,
        frames=source_frame[None],
        class_id=np.asarray([class_id], np.int64),
        alignment_hash=np.asarray([source_alignment], dtype="<U64"),
    )
    source_metadata_value = {
        "schema_version": a.SOURCE_SCHEMA,
        "manifest_sha256": manifest_value["manifest_sha256"],
        "config_sha256": config_value["config_sha256"],
        "dataset_path": source_dataset.name,
        "dataset_sha256": _sha(source_dataset.read_bytes()),
        "renderer_id": a.RENDERER_SPEC["id"],
        "renderer_sha256": a.source_renderer_hash(),
        "teacher_id": "causal-source-teacher-v1",
        "teacher_sha256": a.causal_source_teacher_hash(),
        "action_types": list(a.ACTION_TYPES),
        "action_schema_sha256": a.action_schema_hash(),
        "rows": [
            {
                "alignment_hash": source_alignment,
                "observation": observation,
                "legal_actions": ["wait", "forward"],
                "class_id": class_id,
            }
        ],
    }
    source_metadata = _write(
        tmp_path / "source.json", _seal(source_metadata_value, "source_metadata_sha256")
    )
    source_state = resnet18(weights=None, num_classes=6).state_dict()
    source_model_metadata = {
        "schema_version": a.MODEL_SCHEMA,
        "role": "v5_causal_source_teacher",
        "manifest_sha256": str(manifest_value["manifest_sha256"]),
        "split_binding_sha256": str(manifest_value["split_binding_sha256"]),
        "config_sha256": str(config_value["config_sha256"]),
        "renderer_sha256": a.source_renderer_hash(),
        "teacher_sha256": a.causal_source_teacher_hash(),
        "action_schema_sha256": a.action_schema_hash(),
        "source_dataset_sha256": str(source_metadata_value["dataset_sha256"]),
        "source_metadata_sha256": str(source_metadata_value["source_metadata_sha256"]),
    }
    source_model = tmp_path / "source.safetensors"
    source_model.write_bytes(save_safetensors(source_state, metadata=source_model_metadata))
    source_model_sha = _sha(source_model.read_bytes())
    adapted_metadata = {
        "schema_version": a.MODEL_SCHEMA,
        "role": "v5_simsiam_adapted",
        "manifest_sha256": str(manifest_value["manifest_sha256"]),
        "split_binding_sha256": str(manifest_value["split_binding_sha256"]),
        "config_sha256": str(config_value["config_sha256"]),
        "renderer_sha256": a.source_renderer_hash(),
        "teacher_sha256": a.causal_source_teacher_hash(),
        "action_schema_sha256": a.action_schema_hash(),
        "source_model_sha256": source_model_sha,
        "source_dataset_sha256": str(source_metadata_value["dataset_sha256"]),
        "source_metadata_sha256": str(source_metadata_value["source_metadata_sha256"]),
        "training_seed": "0",
        "collapse_metrics_sha256": "c" * 64,
        "collapse_metrics_json": "{}",
    }
    adapted_model = tmp_path / "adapted.safetensors"
    adapted_model.write_bytes(
        save_safetensors(a.ResNet18SimSiam(source_state).state_dict(), metadata=adapted_metadata)
    )
    target = {
        (str(session), int(time)): str(alignment)
        for shard in a.load_npz_shards(shards)
        for session, time, alignment in zip(
            shard["session_hash"], shard["timestamp_ms"], shard["alignment_hash"], strict=True
        )
    }
    probability = [0.996, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008]
    prediction_rows = [
        {
            "session_hash": anchor_session,
            "anchor_timestamp_ms": 100,
            "frame_timestamp_ms": time,
            "frame_alignment_hash": target[anchor_session, time],
            "model_role": role,
            "view_id": view,
            "probs": probability,
            "ood_score": 0.0,
            "black_control_ok": True,
            "constant_control_ok": True,
            "cut": False,
        }
        for role in ("source", "student")
        for view, time in (("t-100", 0), ("t", 100), ("t+100", 200))
    ]
    predictions_value = {
        "schema_version": a.PREDICTION_SCHEMA,
        "manifest_sha256": manifest_value["manifest_sha256"],
        "source_model_sha256": source_model_sha,
        "adapted_model_sha256": _sha(adapted_model.read_bytes()),
        "config_sha256": config_value["config_sha256"],
        "rows": prediction_rows,
    }
    predictions = _write(
        tmp_path / "predictions.json", _seal(predictions_value, "predictions_sha256")
    )
    return {
        "sessions": sessions,
        "pre_ingest": pre_ingest,
        "split_of": split_of,
        "manifest_value": manifest_value,
        "manifest": manifest,
        "privacy_value": privacy_value,
        "privacy": privacy,
        "owner_value": owner_value,
        "owner": owner,
        "confirmation_value": confirmation_value,
        "confirmation": confirmation,
        "config_value": config_value,
        "config": config,
        "shards": shards,
        "source_dataset": source_dataset,
        "source_metadata": source_metadata,
        "source_model": source_model,
        "adapted_model": adapted_model,
        "predictions": predictions,
        "target": target,
    }


def _mean_teacher(
    bundle: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    pseudo = tmp_path / "pseudo.npz"
    with pytest.raises(a.AlignmentError, match="model-generated evidence"):
        a.materialize_v5_pseudo(
            predictions_path=bundle["predictions"],
            source_metadata_path=bundle["source_metadata"],
            source_dataset_path=bundle["source_dataset"],
            manifest_path=bundle["manifest"],
            pre_ingest_path=bundle["pre_ingest"],
            privacy_context_path=bundle["privacy"],
            owner_attestation_path=bundle["owner"],
            owner_component_confirmation_path=bundle["confirmation"],
            target_shards=bundle["shards"],
            config_path=bundle["config"],
            source_model_path=bundle["source_model"],
            adapted_model_path=bundle["adapted_model"],
            output_path=pseudo,
        )
    accepted = a.AcceptedPseudoDataset(
        np.empty((1, 128, 128, 3), dtype=np.uint8),
        np.zeros(1, dtype=np.int64),
        pseudo,
        "a" * 64,
        str(bundle["manifest_value"]["manifest_sha256"]),
        "b" * 64,
    )
    monkeypatch.setattr(a, "load_accepted_pseudo_artifact", lambda *_args, **_kwargs: accepted)
    ema, ledger = (tmp_path / "ema.safetensors", tmp_path / "round.json")
    with pytest.raises(a.AlignmentError, match="at least 200"):
        a.run_mean_teacher_round(
            source_metadata_path=bundle["source_metadata"],
            source_dataset_path=bundle["source_dataset"],
            manifest_path=bundle["manifest"],
            pre_ingest_path=bundle["pre_ingest"],
            privacy_context_path=bundle["privacy"],
            owner_attestation_path=bundle["owner"],
            owner_component_confirmation_path=bundle["confirmation"],
            target_shards=bundle["shards"],
            predictions_path=bundle["predictions"],
            pseudo_path=pseudo,
            source_model_path=bundle["source_model"],
            adapted_checkpoint=bundle["adapted_model"],
            config_path=bundle["config"],
            ema_checkpoint=ema,
            round_ledger=ledger,
        )
    assert not pseudo.exists() and not ema.exists() and not ledger.exists()
    return pseudo


def test_adapted_checkpoint_metadata_records_seed_and_collapse_report_binding(
    tmp_path: Path
) -> None:
    bundle = _artifacts(tmp_path)
    _, metadata, _ = a._load_model(bundle["adapted_model"], "v5_simsiam_adapted")
    assert metadata["training_seed"] == "0"
    assert metadata["collapse_metrics_json"] == "{}"
    assert metadata["collapse_metrics_sha256"] == "c" * 64


def test_mean_teacher_round_rolls_back_ema_if_round_ledger_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))

    manifest = a.V5Manifest(
        path=tmp_path / "manifest.json",
        manifest_sha256="a" * 64,
        pre_ingest_sha256="b" * 64,
        component_cohort_sha256="c" * 64,
        privacy_context_sha256="d" * 64,
        owner_attestation_sha256="e" * 64,
        privacy_transform_sha256="g" * 64,
        split_binding_sha256="f" * 64,
        session_splits={},
        shard_paths=(),
        shard_sha256={},
        split_row_counts={"train": 1, "dev": 1, "test": 0},
    )
    config = {
        "batch_size": 200,
        "epochs": 50,
        "mean_teacher_epochs": 1,
        "ema_decay": 0.999,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "config_sha256": "f" * 64,
    }
    source = a.SourceDataset(
        frames=np.zeros((2, 128, 128, 3), dtype=np.uint8),
        labels=np.array([0, 1], dtype=np.int64),
        split=np.array(["train", "validation"], dtype="<U10"),
    )
    source_meta = {
        "source_dataset_sha256": "1" * 64,
        "source_metadata_sha256": "2" * 64,
    }
    source_sha = "3" * 64

    class _TinyTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(128 * 128 * 3, 6)

        def freeze_batch_norm(self) -> None:
            return None

        def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = self.linear(frames.reshape(len(frames), -1))
            zeros = torch.zeros((len(frames), 6), dtype=frames.dtype, device=frames.device)
            return logits, zeros, zeros

    pseudo = a.AcceptedPseudoDataset(
        np.zeros((200, 128, 128, 3), dtype=np.uint8),
        np.zeros(200, dtype=np.int64),
        tmp_path / "pseudo.npz",
        "0" * 64,
        "1" * 64,
        "2" * 64,
    )
    monkeypatch.setattr(a, "_source_bundle", lambda *_: (source, {}, source_meta, source_sha))
    monkeypatch.setattr(a, "load_v5_manifest", lambda *_: manifest)
    monkeypatch.setattr(a, "load_v5_training_config", lambda *_: config)
    monkeypatch.setattr(a, "_adapted_model", lambda *_: _TinyTeacher())
    monkeypatch.setattr(
        a,
        "_load_model",
        lambda *_: (
            {},
            {
                "manifest_sha256": manifest.manifest_sha256,
                "split_binding_sha256": manifest.split_binding_sha256,
                "config_sha256": config["config_sha256"],
                "source_model_sha256": source_sha,
                "source_dataset_sha256": source_meta["source_dataset_sha256"],
                "source_metadata_sha256": source_meta["source_metadata_sha256"],
                "collapse_metrics_sha256": "9" * 64,
            },
            "4" * 64,
        ),
    )
    monkeypatch.setattr(
        a,
        "load_accepted_pseudo_artifact",
        lambda *_args, **_kwargs: pseudo,
    )
    ema = root / "mean-teacher-ema.safetensors"
    ledger = root / "mean-teacher-ledger.json"
    write = a._write_exclusive

    def fail_on_ledger(path: Path, payload: bytes) -> None:
        if path == ledger:
            raise RuntimeError("round ledger write blocked")
        write(path, payload)

    monkeypatch.setattr(a, "_write_exclusive", fail_on_ledger)

    with pytest.raises(RuntimeError, match="round ledger write blocked"):
        a.run_mean_teacher_round(
            source_metadata_path=tmp_path / "source.json",
            source_dataset_path=tmp_path / "source.npz",
            manifest_path=tmp_path / "manifest.json",
            pre_ingest_path=tmp_path / "pre-ingest.json",
            privacy_context_path=tmp_path / "privacy.json",
            owner_attestation_path=tmp_path / "owner.json",
            owner_component_confirmation_path=tmp_path / "confirmation.json",
            target_shards=(),
            predictions_path=tmp_path / "predictions",
            pseudo_path=tmp_path / "pseudo.npz",
            source_model_path=tmp_path / "source.pt",
            adapted_checkpoint=tmp_path / "adapted.safetensors",
            config_path=tmp_path / "config.json",
            ema_checkpoint=ema,
            round_ledger=ledger,
            seed=0,
        )
    assert not ema.exists() and not ledger.exists()


def test_mean_teacher_round_uses_train_split_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))

    manifest = a.V5Manifest(
        path=tmp_path / "manifest.json",
        manifest_sha256="a" * 64,
        pre_ingest_sha256="b" * 64,
        component_cohort_sha256="c" * 64,
        privacy_context_sha256="d" * 64,
        owner_attestation_sha256="e" * 64,
        privacy_transform_sha256="g" * 64,
        split_binding_sha256="f" * 64,
        session_splits={},
        shard_paths=(),
        shard_sha256={},
        split_row_counts={"train": 1, "dev": 1, "test": 0},
    )
    config = {
        "batch_size": 2,
        "epochs": 1,
        "mean_teacher_epochs": 1,
        "ema_decay": 0.999,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "config_sha256": "f" * 64,
    }
    source = a.SourceDataset(
        frames=np.zeros((2, 128, 128, 3), dtype=np.uint8),
        labels=np.array([0, 6], dtype=np.int64),
        split=np.array(["train", "validation"], dtype="<U10"),
    )
    source_meta = {
        "source_dataset_sha256": "1" * 64,
        "source_metadata_sha256": "2" * 64,
    }
    source_sha = "3" * 64

    class _TinyTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(128 * 128 * 3, 6)

        def freeze_batch_norm(self) -> None:
            return None

        def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = self.linear(frames.reshape(len(frames), -1))
            zeros = torch.zeros((len(frames), 6), dtype=frames.dtype, device=frames.device)
            return logits, zeros, zeros

    pseudo = a.AcceptedPseudoDataset(
        np.zeros((200, 128, 128, 3), dtype=np.uint8),
        np.zeros(200, dtype=np.int64),
        tmp_path / "pseudo.npz",
        "0" * 64,
        "1" * 64,
        "2" * 64,
    )
    monkeypatch.setattr(a, "_source_bundle", lambda *_: (source, {}, source_meta, source_sha))
    monkeypatch.setattr(a, "load_v5_manifest", lambda *_: manifest)
    monkeypatch.setattr(a, "load_v5_training_config", lambda *_: config)
    monkeypatch.setattr(a, "_adapted_model", lambda *_: _TinyTeacher())
    monkeypatch.setattr(
        a,
        "_load_model",
        lambda *_: (
            {},
            {
                "manifest_sha256": manifest.manifest_sha256,
                "split_binding_sha256": manifest.split_binding_sha256,
                "config_sha256": config["config_sha256"],
                "source_model_sha256": source_sha,
                "source_dataset_sha256": source_meta["source_dataset_sha256"],
                "source_metadata_sha256": source_meta["source_metadata_sha256"],
                "collapse_metrics_sha256": "9" * 64,
            },
            "4" * 64,
        ),
    )
    monkeypatch.setattr(
        a,
        "load_accepted_pseudo_artifact",
        lambda *_args, **_kwargs: pseudo,
    )

    ema = root / "mean-teacher-ema.safetensors"
    ledger = root / "mean-teacher-ledger.json"

    a.run_mean_teacher_round(
        source_metadata_path=tmp_path / "source.json",
        source_dataset_path=tmp_path / "source.npz",
        manifest_path=tmp_path / "manifest.json",
        pre_ingest_path=tmp_path / "pre-ingest.json",
        privacy_context_path=tmp_path / "privacy.json",
        owner_attestation_path=tmp_path / "owner.json",
        owner_component_confirmation_path=tmp_path / "confirmation.json",
        target_shards=(),
        predictions_path=tmp_path / "predictions",
        pseudo_path=tmp_path / "pseudo.npz",
        source_model_path=tmp_path / "source.pt",
        adapted_checkpoint=tmp_path / "adapted.safetensors",
        config_path=tmp_path / "config.json",
        ema_checkpoint=ema,
        round_ledger=ledger,
        seed=0,
    )
    assert ema.exists() and ledger.exists()


def test_cli_materialize_pseudo_failed_status_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    cohort_dir = tmp_path / "cohort"
    for path in (source_dir, target_dir, cohort_dir):
        path.mkdir()

    (source_dir / "source.json").write_text(
        json.dumps({"selected_model_path": "model.safetensors"}),
        encoding="utf-8",
    )
    (source_dir / "model.safetensors").write_text("", encoding="utf-8")
    (target_dir / "manifest.json").write_text("{}", encoding="utf-8")

    pre_ingest = tmp_path / "pre_ingest.json"
    privacy = cohort_dir / "privacy-context.json"
    owner = cohort_dir / "owner-attestation.json"
    confirmation = cohort_dir / "component-cohort.json"
    for path in (pre_ingest, privacy, owner, confirmation):
        path.write_text("{}", encoding="utf-8")

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    output = tmp_path / "pseudo.npz"

    def fake_materialize(*_args: object, **_kwargs: object) -> tuple[
        a.AcceptedPseudoDataset, a.PseudoFilterReport
    ]:
        return (
            a.AcceptedPseudoDataset(
                np.empty((0, 128, 128, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
                output,
                "a" * 64,
                "b" * 64,
                "c" * 64,
            ),
            a.PseudoFilterReport(0, 0, {}, False),
        )

    monkeypatch.setattr(a, "materialize_v5_pseudo", fake_materialize)
    monkeypatch.setattr(cli, "_v5_source_model", lambda _path: source_dir / "model.safetensors")

    assert (
        cli.main(
            [
                "v5-materialize-pseudo",
                "--source-dir",
                str(source_dir),
                "--target-dir",
                str(target_dir),
                "--cohort-dir",
                str(cohort_dir),
                "--pre-ingest",
                str(pre_ingest),
                "--config",
                str(config),
                "--adapted-model",
                str(tmp_path / "adapted.safetensors"),
                "--predictions-dir",
                str(predictions),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 2
    )


def test_cli_train_simsiam_adapted_result_is_json_serializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    cohort_dir = tmp_path / "cohort"
    for path in (source_dir, target_dir, cohort_dir):
        path.mkdir()

    (source_dir / "source.json").write_text(
        json.dumps({"selected_model_path": "model.safetensors"}), encoding="utf-8"
    )
    (source_dir / "model.safetensors").write_text("", encoding="utf-8")
    (target_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (target_dir / "shards" / "target-00000.npz").parent.mkdir(parents=True, exist_ok=True)
    (target_dir / "shards" / "target-00000.npz").write_bytes(b"\x00")

    pre_ingest = tmp_path / "pre_ingest.json"
    privacy = cohort_dir / "privacy-context.json"
    owner = cohort_dir / "owner-attestation.json"
    confirmation = cohort_dir / "component-cohort.json"
    for path in (pre_ingest, privacy, owner, confirmation):
        path.write_text("{}", encoding="utf-8")

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output_checkpoint = tmp_path / "adapted.safetensors"

    fake_training = a.TrainingResult(
        output_checkpoint,
        "f" * 64,
        "a" * 64,
        {"seed": 123, "collapse_metrics": {}},
        False,
    )
    received: dict[str, object] = {}
    monkeypatch.setattr(
        a, "train_shallow_simsiam", lambda **kwargs: (received.update(kwargs), fake_training)[1]
    )
    monkeypatch.setattr(cli, "_v5_source_model", lambda _path: source_dir / "model.safetensors")

    assert (
        cli.main(
            [
                "v5-train-simsiam-adapted",
                "--source-dir",
                str(source_dir),
                "--target-dir",
                str(target_dir),
                "--cohort-dir",
                str(cohort_dir),
                "--pre-ingest",
                str(pre_ingest),
                "--config",
                str(config),
                "--output-checkpoint",
                str(output_checkpoint),
                "--device",
                "cpu",
                "--resume",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint"] == str(output_checkpoint)
    assert payload["metrics"]["seed"] == 123
    assert received["resume"] is True


def test_simsiam_resume_round_trip_restores_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    loss = model(torch.ones((2, 2))).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    binding: dict[str, object] = {
        "manifest_sha256": "a" * 64,
        "split_binding_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "source_model_sha256": "d" * 64,
        "training_seed": 7,
        "steps_per_epoch": 3,
    }
    resume = tmp_path / "resume.safetensors"
    a._write_atomic(
        resume,
        a._resume_bytes(
            model,
            optimizer,
            scheduler,
            {**binding, "completed_epochs": 1, "elapsed_seconds": 2.5},
            cuda=False,
        ),
    )
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01, weight_decay=0.001)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, 10)
    assert a._load_resume(
        resume, restored, restored_optimizer, restored_scheduler, binding, cuda=False
    ) == (1, 2.5)
    assert torch.equal(model.weight, restored.weight)
    assert torch.equal(model.bias, restored.bias)
    assert scheduler.last_epoch == restored_scheduler.last_epoch
    assert optimizer.param_groups[0]["lr"] == restored_optimizer.param_groups[0]["lr"]


def test_model_generated_predictions_are_18_view_bound_and_tamper_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _artifacts(tmp_path)
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    predictions = tmp_path / "model-predictions"
    result = a.generate_v5_model_predictions(
        source_metadata_path=bundle["source_metadata"],
        source_dataset_path=bundle["source_dataset"],
        manifest_path=bundle["manifest"],
        pre_ingest_path=bundle["pre_ingest"],
        privacy_context_path=bundle["privacy"],
        owner_attestation_path=bundle["owner"],
        owner_component_confirmation_path=bundle["confirmation"],
        target_shards=bundle["shards"],
        config_path=bundle["config"],
        source_model_path=bundle["source_model"],
        adapted_model_path=bundle["adapted_model"],
        output_dir=predictions,
        batch_size=2,
    )
    assert result["prediction_rows"] == 18 and result["human_labels_consumed"] is False
    payload = json.loads((predictions / "manifest.json").read_text())
    assert (
        payload["schema_version"] == a.MODEL_PREDICTION_SCHEMA
        and payload["inference_contract_sha256"] == a.prediction_inference_contract_hash()
    )
    arrays, _ = a._load_model_prediction_shard(predictions / "shards" / "prediction-00000.npz")
    assert set(arrays["augmentation_id"]) == set(a.AUGMENTATION_KEYS)
    pseudo, report = a.materialize_v5_pseudo(
        predictions_path=predictions,
        source_metadata_path=bundle["source_metadata"],
        source_dataset_path=bundle["source_dataset"],
        manifest_path=bundle["manifest"],
        pre_ingest_path=bundle["pre_ingest"],
        privacy_context_path=bundle["privacy"],
        owner_attestation_path=bundle["owner"],
        owner_component_confirmation_path=bundle["confirmation"],
        target_shards=bundle["shards"],
        config_path=bundle["config"],
        source_model_path=bundle["source_model"],
        adapted_model_path=bundle["adapted_model"],
        output_path=tmp_path / "model-pseudo.npz",
        prediction_batch_size=2,
    )
    assert not len(pseudo.frames) and not report.filter_floor_met
    prediction = predictions / "shards" / "prediction-00000.npz"
    prediction.write_bytes(prediction.read_bytes() + b"tamper")
    with pytest.raises(a.AlignmentError, match="model prediction"):
        a.materialize_v5_pseudo(
            predictions_path=predictions,
            source_metadata_path=bundle["source_metadata"],
            source_dataset_path=bundle["source_dataset"],
            manifest_path=bundle["manifest"],
            pre_ingest_path=bundle["pre_ingest"],
            privacy_context_path=bundle["privacy"],
            owner_attestation_path=bundle["owner"],
            owner_component_confirmation_path=bundle["confirmation"],
            target_shards=bundle["shards"],
            config_path=bundle["config"],
            source_model_path=bundle["source_model"],
            adapted_model_path=bundle["adapted_model"],
            output_path=tmp_path / "model-pseudo-tampered.npz",
            prediction_batch_size=2,
        )


def test_causal_teacher_six_classes_and_renderer_has_no_tick_shortcut() -> None:
    base = {
        "side": "blue",
        "self_position": 2,
        "opponent_position": 8,
        "self_health": 6,
        "opponent_health": 6,
        "own_tower_health": 4,
        "enemy_tower_health": 4,
        "own_crystal_health": 6,
        "enemy_crystal_health": 6,
    }
    cases = (
        (["wait"], 6, 6, "wait"),
        (["wait", "forward"], 6, 6, "forward"),
        (["wait", "backward"], 1, 6, "backward"),
        (["wait", "attack_hero"], 6, 1, "attack_hero"),
        (["wait", "attack_tower", "attack_hero"], 6, 6, "attack_tower"),
        (["wait", "attack_crystal", "attack_tower"], 6, 6, "attack_crystal"),
        (["wait", "attack_hero"], 1, 6, "wait"),
    )
    for legal, self_hp, opponent_hp, expected in cases:
        observation = dict(base, self_health=self_hp, opponent_health=opponent_hp, tick=17)
        assert (
            a._action_name(
                a.CausalSourceTeacher().select(
                    observation, tuple(a._action_object(name) for name in legal)
                )
            )
            == expected
        )
    assert np.array_equal(
        a.source_render_128_rgb(dict(base, tick=1)), a.source_render_128_rgb(dict(base, tick=999))
    )
    assert inspect.signature(a.source_render_128_rgb).parameters["render_seed"].default == 0
    assert not np.array_equal(
        a.source_render_128_rgb(base, render_seed=0), a.source_render_128_rgb(base, render_seed=1)
    )
    asymmetric = np.zeros((128, 128, 3), np.uint8)
    asymmetric[2:7, 11:19] = (1, 2, 3)
    assert np.array_equal(
        a.zero_redaction_letterbox_rgb(asymmetric, 90), np.rot90(asymmetric, k=-1)
    )


def test_pre_ingest_file_atomic_mode_never_decodes_and_emits_one_component_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.mp4").write_bytes(b"a")
    (tmp_path / "b.mp4").write_bytes(b"b")
    output = tmp_path / "pre-ingest.json"
    monkeypatch.setattr(p, "_decode", lambda *_args: pytest.fail("v2 must not decode video"))
    result = p.pre_ingest(tmp_path, output)
    evidence = p.load_pre_ingest(output)
    assert result["relationship_mode"] == p.RELATIONSHIP_MODE
    assert result["algorithm_spec"]["input_integrity"]["decode_enabled"] is False
    assert result["relations"] == []
    assert result["candidate_count"] == result["component_count"] == 2
    assert len(set(evidence.component_of.values())) == 2


def test_strict_manifest_source_and_symlink_fail_closed(tmp_path: Path) -> None:
    bundle = _artifacts(tmp_path)
    manifest = a.load_v5_manifest(
        bundle["manifest"],
        bundle["pre_ingest"],
        bundle["privacy"],
        bundle["owner"],
        bundle["confirmation"],
        bundle["shards"],
    )
    source = a.load_v5_source_dataset(
        bundle["source_metadata"],
        bundle["source_dataset"],
        bundle["manifest"],
        bundle["pre_ingest"],
        bundle["privacy"],
        bundle["owner"],
        bundle["confirmation"],
        bundle["shards"],
        bundle["config"],
        bundle["source_model"],
    )
    assert len(manifest.session_splits) == 12 and source.frames.shape == (1, 128, 128, 3)
    with pytest.raises(a.AlignmentError, match="cohort artifact is required"):
        a.load_v5_manifest(
            bundle["manifest"],
            bundle["pre_ingest"],
            bundle["privacy"],
            bundle["owner"],
            None,
            bundle["shards"],
        )
    bad_confirmation_value = json.loads(json.dumps(bundle["confirmation_value"]))
    bad_confirmation_value["component_cohort_sha256"] = "a" * 64
    bad_confirmation = _write(tmp_path / "confirmation-bad-hash.json", bad_confirmation_value)
    with pytest.raises(a.AlignmentError, match="component_cohort_sha256 mismatch"):
        a.load_v5_manifest(
            bundle["manifest"],
            bundle["pre_ingest"],
            bundle["privacy"],
            bundle["owner"],
            bad_confirmation,
            bundle["shards"],
        )
    mismatch_confirmation_value = json.loads(json.dumps(bundle["confirmation_value"]))
    mismatch_confirmation_value["component_hashes"][-1] = "f" * 64
    mismatch_confirmation_value["component_hashes"].sort()
    mismatch_confirmation = _write(
        tmp_path / "confirmation-component-mismatch.json",
        _seal(
            {
                key: value
                for key, value in mismatch_confirmation_value.items()
                if key != "component_cohort_sha256"
            },
            "component_cohort_sha256",
        ),
    )
    with pytest.raises(a.AlignmentError, match="absent from pre-ingest"):
        a.load_v5_manifest(
            bundle["manifest"],
            bundle["pre_ingest"],
            bundle["privacy"],
            bundle["owner"],
            mismatch_confirmation,
            bundle["shards"],
        )
    bad = json.loads(json.dumps(bundle["manifest_value"]))
    bad["sessions"][0]["parent_hash"] = "f" * 64
    _write(
        tmp_path / "bad.json",
        _seal(
            {key: value for key, value in bad.items() if key != "manifest_sha256"},
            "manifest_sha256",
        ),
    )
    with pytest.raises(a.AlignmentError, match="known"):
        a.load_v5_manifest(
            tmp_path / "bad.json",
            bundle["pre_ingest"],
            bundle["privacy"],
            bundle["owner"],
            bundle["confirmation"],
            bundle["shards"],
        )

    def rejected(name: str, mutate: object, match: str) -> None:
        value = json.loads(json.dumps(bundle["manifest_value"]))
        cast_mutate = mutate
        cast_mutate(value)
        path = _write(
            tmp_path / f"{name}.json",
            _seal(
                {key: item for key, item in value.items() if key != "manifest_sha256"},
                "manifest_sha256",
            ),
        )
        with pytest.raises(a.AlignmentError, match=match):
            a.load_v5_manifest(
                path,
                bundle["pre_ingest"],
                bundle["privacy"],
                bundle["owner"],
                bundle["confirmation"],
                bundle["shards"],
            )

    rejected(
        "same-component",
        lambda value: [row.update(component_hash="a" * 64) for row in value["sessions"]],
        "differs from pre-ingest",
    )
    rejected(
        "plain-component",
        lambda value: value["sessions"][0].update(component_hash="component-0"),
        "invalid manifest session",
    )
    rejected(
        "split",
        lambda value: (
            value["sessions"][0].update(split="dev"),
            value["sessions"][8].update(split="train"),
        ),
        "split allocation",
    )
    rejected(
        "transform",
        lambda value: value.update(privacy_transform_sha256="a" * 64),
        "owner/privacy evidence binding",
    )
    rejected(
        "attestation",
        lambda value: value.update(owner_attestation_sha256="a" * 64),
        "owner/privacy evidence binding",
    )
    rejected(
        "context",
        lambda value: value.update(privacy_context_sha256="a" * 64),
        "owner/privacy evidence binding",
    )
    rejected(
        "cohort-manifest",
        lambda value: value.update(component_cohort_sha256="a" * 64),
        "owner/privacy evidence binding",
    )
    rejected(
        "review",
        lambda value: value["sessions"][0].update(privacy_review_sha256="a" * 64),
        "reviewed privacy context",
    )
    rejected(
        "pre-hash",
        lambda value: value.update(pre_ingest_sha256="a" * 64),
        "pre-ingest hash mismatch",
    )
    rejected(
        "pre-component",
        lambda value: value["sessions"][0].update(component_hash="a" * 64),
        "differs from pre-ingest",
    )
    rejected("extra", lambda value: value.update(extra=False), "fields are not exact")
    with pytest.raises(a.AlignmentError, match="reviewed privacy context"):
        a.write_npz_shards(
            [
                {
                    "session_hash": bundle["sessions"][0],
                    "timestamp_ms": 0,
                    "pts": 0,
                    "time_base": (1, 1000),
                    "split": "train",
                    "source": "target",
                    "frame": np.zeros((4, 4, 3), np.uint8),
                }
            ],
            tmp_path / "bare",
        )
    with pytest.raises(a.AlignmentError, match="forbidden"):
        a.write_npz_shards(
            [
                {
                    "session_hash": bundle["sessions"][0],
                    "timestamp_ms": 0,
                    "pts": 0,
                    "time_base": (1, 1000),
                    "split": "train",
                    "source": "target",
                    "frame": np.zeros((4, 4, 3), np.uint8),
                    "privacy_mask": np.zeros((4, 4), np.bool_),
                }
            ],
            tmp_path / "masked",
            pre_ingest_path=bundle["pre_ingest"],
            privacy_context_path=bundle["privacy"],
            owner_attestation_path=bundle["owner"],
            owner_component_confirmation_path=bundle["confirmation"],
        )
    owner_bad = dict(bundle["owner_value"], recording_owner=False)
    owner_bad = _seal(
        {key: value for key, value in owner_bad.items() if key != "owner_attestation_sha256"},
        "owner_attestation_sha256",
    )
    with pytest.raises(a.AlignmentError, match="does not authorize"):
        a.load_privacy_context(bundle["privacy"], _write(tmp_path / "owner-bad.json", owner_bad))
    privacy_bad_value = json.loads(json.dumps(bundle["privacy_value"]))
    privacy_bad_value["reviews"][0]["component_hash"] = privacy_bad_value["reviews"][1][
        "component_hash"
    ]
    privacy_bad_value["reviews"][0] = _seal(
        {
            key: value
            for key, value in privacy_bad_value["reviews"][0].items()
            if key != "privacy_review_sha256"
        },
        "privacy_review_sha256",
    )
    privacy_bad = _write(
        tmp_path / "privacy-component-bad.json",
        _seal(
            {
                key: value
                for key, value in privacy_bad_value.items()
                if key != "privacy_context_sha256"
            },
            "privacy_context_sha256",
        ),
    )
    with pytest.raises(a.AlignmentError, match="privacy context component differs"):
        a.load_v5_manifest(
            bundle["manifest"],
            bundle["pre_ingest"],
            privacy_bad,
            bundle["owner"],
            bundle["confirmation"],
            bundle["shards"],
        )
    with pytest.raises(a.AlignmentError, match="privacy context component differs"):
        a.write_npz_shards(
            [
                {
                    "session_hash": bundle["sessions"][0],
                    "timestamp_ms": 0,
                    "pts": 0,
                    "time_base": (1, 1000),
                    "rotation_degrees": 0,
                    "split": "train",
                    "source": "target",
                    "frame": np.zeros((4, 4, 3), np.uint8),
                }
            ],
            tmp_path / "privacy-component-writer",
            pre_ingest_path=bundle["pre_ingest"],
            privacy_context_path=privacy_bad,
            owner_attestation_path=bundle["owner"],
            owner_component_confirmation_path=bundle["confirmation"],
        )
    with np.load(bundle["shards"][0], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["privacy_review_sha256"][0] = "a" * 64
    tampered_npz = tmp_path / "tampered.npz"
    np.savez(tampered_npz, **arrays)
    with pytest.raises(a.AlignmentError, match="alignment hash mismatch"):
        a.load_npz_shards([tampered_npz])
    with np.load(bundle["shards"][0], allow_pickle=False) as archive:
        rotation_arrays = {key: archive[key].copy() for key in archive.files}
    rotation_arrays["rotation_degrees"][0] = 180
    tampered_rotation_npz = tmp_path / "tampered-rotation.npz"
    np.savez(tampered_rotation_npz, **rotation_arrays)
    with pytest.raises(a.AlignmentError, match="alignment hash mismatch"):
        a.load_npz_shards([tampered_rotation_npz])
    (tmp_path / "link.json").symlink_to(bundle["manifest"])
    with pytest.raises(a.AlignmentError, match="non-symlink"):
        a.load_v5_manifest(
            tmp_path / "link.json",
            bundle["pre_ingest"],
            bundle["privacy"],
            bundle["owner"],
            bundle["confirmation"],
            bundle["shards"],
        )
    for relation in ("near_duplicate", "uncertain"):
        blocked, _ = _pre_ingest(tmp_path, bundle["sessions"], relation=relation)
        with pytest.raises(a.AlignmentError, match="pre-ingest evidence is invalid"):
            a.load_v5_manifest(
                bundle["manifest"],
                blocked,
                bundle["privacy"],
                bundle["owner"],
                bundle["confirmation"],
                bundle["shards"],
            )


def test_manifest_training_snapshot_is_detached_from_original_shards(tmp_path: Path) -> None:
    bundle = _artifacts(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = a.load_v5_manifest(
        bundle["manifest"],
        bundle["pre_ingest"],
        bundle["privacy"],
        bundle["owner"],
        bundle["confirmation"],
        bundle["shards"],
        verify_rows=False,
        snapshot_dir=snapshot,
    )
    assert not manifest.snapshot_paths
    assert next(
        a.iter_v5_shards(
            manifest, "train", verify_rows=False, verify_hash=False, prefetch_shards=2
        )
    )
    original = bundle["shards"][0]
    assert manifest.snapshot_paths[original].parent == snapshot
    original.write_bytes(b"replaced-after-binding")
    assert next(a.iter_v5_shards(manifest, "train", verify_rows=False, verify_hash=False))


def test_training_manifest_defers_shard_bytes_until_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _artifacts(tmp_path)
    read = a._read_regular

    def reject_npz(path: Path, suffix: str | None = None) -> bytes:
        if suffix == ".npz":
            pytest.fail("training manifest must defer shard bytes")
        return read(path, suffix)

    monkeypatch.setattr(a, "_read_regular", reject_npz)
    manifest = a.load_v5_manifest(
        bundle["manifest"],
        bundle["pre_ingest"],
        bundle["privacy"],
        bundle["owner"],
        bundle["confirmation"],
        bundle["shards"],
        verify_rows=False,
        verify_shards=False,
    )
    assert manifest.split_row_counts["train"] > 0


def test_lazy_training_binding_rejects_original_shard_change_before_first_use(
    tmp_path: Path,
) -> None:
    bundle = _artifacts(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = a.load_v5_manifest(
        bundle["manifest"],
        bundle["pre_ingest"],
        bundle["privacy"],
        bundle["owner"],
        bundle["confirmation"],
        bundle["shards"],
        verify_rows=False,
        verify_shards=False,
        snapshot_dir=snapshot,
    )
    bundle["shards"][0].write_bytes(b"replaced-before-first-use")
    with pytest.raises(a.AlignmentError, match="changed after manifest verification"):
        next(a.iter_v5_shards(manifest, "train", verify_rows=False, verify_hash=False))


def test_path_pseudo_mean_teacher_requires_model_evidence_and_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _artifacts(tmp_path)
    pseudo = _mean_teacher(bundle, tmp_path, monkeypatch)
    assert not pseudo.exists()


def test_source_producer_is_independent_and_baseline_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    rows = a._source_rows(8)
    state = {"weight": torch.zeros(1)}
    summary = {
        "embedding_variance": 1.0,
        "effective_rank": 1.0,
        "top_eigen_share": 0.5,
        "black_constant_distance": 1.0,
    }
    monkeypatch.setattr(a, "_source_rows", lambda _episodes: rows)
    monkeypatch.setattr(
        a,
        "_fit_source_seed",
        lambda _data, seed, _device: (
            state,
            {"seed": seed, "best_epoch": 1, "validation_cross_entropy": float(seed)},
        ),
    )
    monkeypatch.setattr(a, "_source_network", lambda _state, _device: object())
    monkeypatch.setattr(a, "_embedding_summary", lambda _model, _frames: summary)
    result = a.produce_v5_source(output_dir=root / "source", device="cpu")
    source = root / "source"
    target_root = tmp_path / "target"
    target_root.mkdir()
    bundle = _artifacts(target_root)
    monkeypatch.setattr(
        a,
        "load_v5_manifest",
        lambda *_args: pytest.fail("producer source must not reload target manifest"),
    )
    data = a.load_v5_source_dataset(
        source / "source.json",
        source / "source.npz",
        bundle["manifest"],
        bundle["pre_ingest"],
        bundle["privacy"],
        bundle["owner"],
        bundle["confirmation"],
        bundle["shards"],
        bundle["config"],
        source / "source-seed-0.safetensors",
    )
    assert result["disposition"] == "SOURCE_ONLY_NON_PROMOTING" and len(data.frames) == len(rows)
    value = json.loads((source / "source.json").read_text())
    value["producer_sha256"] = "a" * 64
    _write(
        source / "tampered.json",
        _seal(
            {key: item for key, item in value.items() if key != "source_metadata_sha256"},
            "source_metadata_sha256",
        ),
    )
    with pytest.raises(a.AlignmentError, match="provenance"):
        a.load_v5_source_dataset(
            source / "tampered.json",
            source / "source.npz",
            bundle["manifest"],
            bundle["pre_ingest"],
            bundle["privacy"],
            bundle["owner"],
            bundle["confirmation"],
            bundle["shards"],
            bundle["config"],
            source / "source-seed-0.safetensors",
        )
