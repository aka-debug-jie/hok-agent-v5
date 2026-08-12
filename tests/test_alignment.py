# ruff: noqa: E501
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import save_file
from torchvision.models import resnet18

from hok_agent.alignment import (
    ACTION_TYPES,
    AlignmentError,
    AuditLabel,
    AuditPrediction,
    CandidatePrediction,
    CausalSourceTeacher,
    SessionRecord,
    SourceDataset,
    SourceRegression,
    UnlabeledTargetDataset,
    audit_evidence_hash,
    filter_pseudo_labels,
    launch_offline_audit_ui,
    load_accepted_pseudo_artifact,
    load_npz_shards,
    load_release,
    offline_audit_release_gate,
    run_mean_teacher_round,
    source_render_128_rgb,
    source_renderer_hash,
    split_session_manifest,
    train_shallow_simsiam,
    write_accepted_pseudo_artifact,
    write_npz_shards,
)


class Action:
    def __init__(self, kind: str, target: str = "none", direction: str = "none") -> None:
        self.action_type, self.target, self.direction = kind, target, direction


ACTIONS = {
    "wait": Action("wait"),
    "forward": Action("move", direction="forward"),
    "backward": Action("move", direction="backward"),
    "attack_hero": Action("attack", "enemy_hero"),
    "attack_tower": Action("attack", "enemy_tower"),
    "attack_crystal": Action("attack", "enemy_crystal"),
}


@pytest.mark.parametrize(
    ("legal", "self_hp", "opponent_hp", "expected"),
    [
        (("wait",), 6, 6, "wait"),
        (("wait", "forward"), 6, 6, "forward"),
        (("wait", "backward"), 1, 6, "backward"),
        (("wait", "attack_hero"), 6, 1, "attack_hero"),
        (("wait", "attack_tower", "attack_hero"), 6, 6, "attack_tower"),
        (("wait", "attack_crystal", "attack_tower"), 6, 6, "attack_crystal"),
        (("wait", "attack_hero"), 1, 6, "wait"),
    ],
)
def test_teacher_exact_six_classes_and_tick_independence(
    legal: tuple[str, ...], self_hp: int, opponent_hp: int, expected: str
) -> None:
    obs = {"self_health": self_hp, "opponent_health": opponent_hp, "tick": 1, "private": 4}
    selected = CausalSourceTeacher().select(obs, tuple(ACTIONS[key] for key in legal))
    changed = CausalSourceTeacher().select(dict(obs, tick=999_999, private=-3), tuple(ACTIONS[key] for key in legal))
    assert ACTION_TYPES == ("wait", "forward", "backward", "attack_hero", "attack_tower", "attack_crystal")
    assert selected is changed or (selected.action_type, selected.target, selected.direction) == (changed.action_type, changed.target, changed.direction)
    assert selected is ACTIONS[expected]


def test_renderer_independent_hash_rgb_and_no_tick_bar() -> None:
    obs = {"side": "blue", "self_position": 2, "opponent_position": 8, "self_health": 5, "opponent_health": 3, "own_tower_health": 4, "enemy_tower_health": 2, "own_crystal_health": 6, "enemy_crystal_health": 4}
    frame = source_render_128_rgb(dict(obs, tick=1))
    assert frame.shape == (128, 128, 3) and frame.dtype == np.uint8
    assert np.array_equal(frame, source_render_128_rgb(dict(obs, tick=999)))
    before = source_renderer_hash()
    assert "offline_audit_release_gate" not in inspect.getsource(source_renderer_hash)
    assert before == source_renderer_hash() and frame.std() > 0


def test_manifest_components_and_strict_npz(tmp_path: Path) -> None:
    records = [SessionRecord(f"s{i}", "linked" if i in (0, 1) else f"f{i}", "s0" if i == 2 else None, ("s3",) if i == 2 else ()) for i in range(12)]
    manifest = split_session_manifest(records)
    split = {row.session_id: row.split for row in manifest}
    assert split["s0"] == split["s1"] == split["s2"] == split["s3"]
    assert {name: sum(row.split == name for row in manifest) for name in ("train", "dev", "test")} == {"train": 8, "dev": 2, "test": 2}
    frame = np.full((20, 30, 3), 100, dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.bool_)
    mask[:2] = True
    paths = write_npz_shards([{"session_id": "secret", "timestamp_ms": 10, "pts": 3, "time_base": (1, 1000), "split": "train", "source": "target", "frame": frame, "privacy_mask": mask}], tmp_path)
    payload = load_npz_shards(paths)[0]
    assert tuple(sorted(payload)) == tuple(sorted(("frames", "session_hash", "timestamp_ms", "pts", "time_base", "frame_hash", "alignment_hash", "split", "source")))
    assert payload["frames"].shape == (1, 128, 128, 3)
    with pytest.raises(AlignmentError):
        write_npz_shards([{**{"session_id": "x", "timestamp_ms": 0, "pts": 0, "time_base": (1, 1), "split": "train", "source": "target", "frame": frame, "privacy_mask": mask}, "label": 2}], tmp_path / "bad")


def _predictions(groups: int) -> list[CandidatePrediction]:
    probs = (0.996, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008)
    return [CandidatePrediction("s", group * 500, model, view, probs) for group in range(groups) for model in ("source", "student") for view in ("t-100", "t", "t+100")]


def test_pseudo_contract_exact_views_vectors_thresholds_and_release_floor() -> None:
    accepted, report = filter_pseudo_labels(_predictions(200))
    assert len(accepted) == 200 and report.release_eligible
    with pytest.raises(AlignmentError, match="cannot be relaxed"):
        filter_pseudo_labels(_predictions(1), min_confidence=0.99)
    malformed = _predictions(1)[:-1]
    assert filter_pseudo_labels(malformed)[1].rejected_by_reason == {"exact_views": 1}


def _source_checkpoint(path: Path) -> str:
    model = resnet18(weights=None, num_classes=6)
    save_file(model.state_dict(), path)
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_resnet18_simsiam_and_one_mean_teacher_cpu_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    source = SourceDataset(rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8), np.arange(4, dtype=np.int64))
    target = UnlabeledTargetDataset(rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8))
    source_path = tmp_path / "source.safetensors"
    result = train_shallow_simsiam(source, target, source_checkpoint=source_path, source_checkpoint_sha256=_source_checkpoint(source_path), output_checkpoint=tmp_path / "adapted.safetensors", epochs=1, batch_size=4)
    assert not result.promoting and result.metrics["cpu_smoke"] is True
    assert set(result.metrics) >= {"embedding_variance", "effective_rank", "covariance_spectrum", "black_constant_distance"}
    pseudo = write_accepted_pseudo_artifact(target.frames, source.labels, "a" * 64, tmp_path / "pseudo.npz")
    pseudo = load_accepted_pseudo_artifact(pseudo.artifact_path, pseudo.artifact_sha256)
    ledger = tmp_path / "round.json"
    report = run_mean_teacher_round(source, pseudo, adapted_checkpoint=result.checkpoint, adapted_checkpoint_sha256=result.checkpoint_sha256, round_ledger=ledger, epochs=1, batch_size=4)
    assert report["weights"] == {"source_ce": 1.0, "pseudo_ce": 1.0, "kl": 0.5} and not report["promoting"]
    with pytest.raises(AlignmentError, match="already exists"):
        run_mean_teacher_round(source, pseudo, adapted_checkpoint=result.checkpoint, adapted_checkpoint_sha256=result.checkpoint_sha256, round_ledger=ledger, epochs=1, batch_size=4)


def test_blind_ui_signature_and_release_remains_disabled(tmp_path: Path) -> None:
    assert "prediction" not in inspect.signature(launch_offline_audit_ui).parameters
    assert launch_offline_audit_ui([], reviewer="r", output_path=tmp_path / "audit.jsonl", enable_gui=False)["blind"] is True
    labels: list[AuditLabel] = []
    predictions: list[AuditPrediction] = []
    for index in range(300):
        action = ACTION_TYPES[index % 6]
        for reviewer in ("r1", "r2"):
            labels.append(AuditLabel(f"c{index:03d}", reviewer, action, index < 285))
        predictions.append(AuditPrediction(f"c{index:03d}", action, index < 285, 0.999, (ACTION_TYPES[(index + 1) % 6], ACTION_TYPES[(index + 2) % 6])))
    regression = SourceRegression(0.90, 0.89, (0.9,) * 6, (0.86,) * 6)
    evidence = audit_evidence_hash(labels, predictions, regression)
    gate = offline_audit_release_gate(labels, predictions, regression, evidence_hash=evidence, model_hash="b" * 64, alignment_hash="d" * 64, config_hash="c" * 64, release_path=tmp_path / "release.json")
    assert not gate.passed and gate.allowed_classes == () and gate.release_path is None
    assert not (tmp_path / "release.json").exists()


def test_release_loader_rejects_missing_framework_release(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_release(tmp_path / "release.json")
