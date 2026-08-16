from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from hok_agent import t8 as t8_module
from hok_agent import t8_shadow as shadow_module
from hok_agent.t8_shadow import T8ShadowError, _accepted_probe_evidence, _action, _decode_candidate


def test_t8_shadow_factor_record_uses_only_frozen_vocabularies() -> None:
    row = _action(np.asarray([1, 2, 3, 0, 2], dtype=np.int64))
    assert row == {
        "movement": "north",
        "ability": "skill1",
        "aim": "east",
        "target": "none",
        "hold_ms": 250,
    }


def test_t8_probe_rejects_factor_collision_and_unproven_evidence(tmp_path) -> None:
    assert _decode_candidate({"movement": "north", "ability": "skill1"}) is None
    model, training, offline, shadow = (
        tmp_path / "model",
        tmp_path / "training.json",
        tmp_path / "offline.json",
        tmp_path / "shadow.json",
    )
    model.write_bytes(b"not-a-model")
    training.write_text("{}", encoding="utf-8")
    offline.write_text("{}", encoding="utf-8")
    shadow.write_text("{}", encoding="utf-8")
    with pytest.raises(T8ShadowError, match="did not admit"):
        _accepted_probe_evidence(training, offline, shadow, model, "a" * 64)


def test_v26_shadow_accepts_only_hash_bound_offline_evidence(tmp_path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"frozen-v26-model")
    layout_sha256 = "a" * 64
    split: dict[str, object] = {
        "schema_version": t8_module.V25_SPLIT_SCHEMA,
        "layout_sha256": layout_sha256,
        "gate_decision_threshold": t8_module.V26_GATE_DECISION_THRESHOLD,
    }
    split["split_sha256"] = hashlib.sha256(t8_module._canonical(split)).hexdigest()
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    offline = {
        "schema_version": t8_module.V26_EVALUATION_SCHEMA,
        "status": "SEALED_OFFLINE_EVALUATION_PASSED",
        "strict_passed": True,
        "shadow_allowed": True,
        "test_accessed": True,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "split_sha256": split["split_sha256"],
        "switch_rate": {"predicted": 0.32},
    }
    offline_path = tmp_path / "offline.json"
    offline_path.write_text(json.dumps(offline), encoding="utf-8")
    assert shadow_module._accepted_v26_shadow_evidence(
        offline_path, split_path, model, layout_sha256
    ) == pytest.approx(0.32)
    offline["model_sha256"] = "b" * 64
    offline_path.write_text(json.dumps(offline), encoding="utf-8")
    with pytest.raises(T8ShadowError, match="did not admit"):
        shadow_module._accepted_v26_shadow_evidence(
            offline_path, split_path, model, layout_sha256
        )


def test_v26_probe_admission_binds_selection_offline_shadow_and_layout(tmp_path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"frozen-v26-model")
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    layout_sha256 = "a" * 64
    split: dict[str, object] = {
        "schema_version": t8_module.V25_SPLIT_SCHEMA,
        "layout_sha256": layout_sha256,
        "gate_decision_threshold": t8_module.V26_GATE_DECISION_THRESHOLD,
    }
    split["split_sha256"] = hashlib.sha256(t8_module._canonical(split)).hexdigest()
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    offline = {
        "schema_version": t8_module.V26_EVALUATION_SCHEMA,
        "status": "SEALED_OFFLINE_EVALUATION_PASSED",
        "strict_passed": True,
        "shadow_allowed": True,
        "test_accessed": True,
        "model_sha256": model_sha256,
        "split_sha256": split["split_sha256"],
        "switch_rate": {"predicted": 0.32},
    }
    offline_path = tmp_path / "offline.json"
    offline_path.write_text(json.dumps(offline), encoding="utf-8")
    selection: dict[str, object] = {
        "schema_version": t8_module.V26_SELECTION_SCHEMA,
        "status": "THREE_SEED_MODEL_SELECTED",
        "selected_seed": 1,
        "selected_model": "seed-1/model-seed-1.safetensors",
        "selected_model_sha256": model_sha256,
        "split_sha256": split["split_sha256"],
    }
    selection["selection_sha256"] = hashlib.sha256(
        t8_module._canonical(selection)
    ).hexdigest()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    shadow = {
        "schema_version": shadow_module.V26_REPLAY_SHADOW_SCHEMA,
        "status": "PASSED",
        "strict_passed": True,
        "source_session": "session-011",
        "model_sha256": model_sha256,
        "layout_sha256": layout_sha256,
        "split_sha256": split["split_sha256"],
        "input_commands_sent": 0,
        "control_output": False,
    }
    shadow_path = tmp_path / "shadow.json"
    shadow_path.write_text(json.dumps(shadow), encoding="utf-8")
    shadow_module._accepted_v26_probe_evidence(
        selection_path=selection_path,
        offline_report=offline_path,
        shadow_summary=shadow_path,
        split_path=split_path,
        model_path=model,
        layout_sha256=layout_sha256,
    )
    shadow["input_commands_sent"] = 1
    shadow_path.write_text(json.dumps(shadow), encoding="utf-8")
    with pytest.raises(T8ShadowError, match="did not admit"):
        shadow_module._accepted_v26_probe_evidence(
            selection_path=selection_path,
            offline_report=offline_path,
            shadow_summary=shadow_path,
            split_path=split_path,
            model_path=model,
            layout_sha256=layout_sha256,
        )


def test_v26_probe_action_surface_is_three_taps_only() -> None:
    assert [shadow_module._v26_probe_action(label).ability for label in (1, 2, 3)] == [
        "basic_attack",
        "skill1",
        "skill2",
    ]
    for label in (0, 4):
        with pytest.raises(T8ShadowError, match="not an allowed combat action"):
            shadow_module._v26_probe_action(label)


def test_v26_probe_scene_gate_rejects_wait_unstable_and_uncertain_candidates() -> None:
    assert shadow_module._v26_scene_candidate_admitted(1, True, 0.45, 0.80)
    assert not shadow_module._v26_scene_candidate_admitted(0, True, 1.0, 0.0)
    assert not shadow_module._v26_scene_candidate_admitted(1, False, 1.0, 0.0)
    assert not shadow_module._v26_scene_candidate_admitted(1, True, 0.44, 0.0)
    assert not shadow_module._v26_scene_candidate_admitted(1, True, 1.0, 0.81)
