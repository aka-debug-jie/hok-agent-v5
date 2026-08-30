from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from hok_agent.global_agent import GlobalArena
from hok_agent.global_policy import GlobalMacroPolicy
from hok_agent.human_ifo import (
    HumanIfoError,
    _all_unlabeled_video_rows,
    _configure_auxiliary_trainable,
    _contract_payload,
    _curriculum_samples,
    _load_curriculum_contract,
    _load_rebind_contract,
    _observable_factor_labels,
    _observable_probe_data,
    _select_unsupervised_cohort,
    _sim_episode_groups,
    cohort_template,
    load_human_cohort,
    write_cohort_template,
)


def _row(index: int, split: str = "train") -> dict[str, object]:
    return {
        "session_hash": f"{index:064x}",
        "split": split,
        "hero_id": "hero-a",
        "role_id": "mid",
        "mode_id": "training",
        "human_controlled": True,
        "complete_match": True,
        "hud_stable": True,
        "overlay_free": True,
        "rois_usable": True,
    }


def test_gate_a_contract_is_self_hashed_and_closed() -> None:
    contract, digest = _contract_payload()
    assert len(digest) == 64
    assert contract["video_test_allowed"] is False
    assert contract["human_dev_training_allowed"] is False
    assert contract["device_input_allowed"] is False


def test_template_is_local_safe_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "human_ifo_cohort.local.json"
    result = write_cohort_template(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["template_written"] is True
    assert payload == cohort_template()
    assert "source_path" not in output.read_text(encoding="utf-8")
    with pytest.raises(HumanIfoError, match="overwrite"):
        write_cohort_template(output)


def test_cohort_requires_exact_count_one_identity_and_qualifications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(index) for index in range(20)] + [_row(100 + index, "dev") for index in range(5)]
    path = tmp_path / "human_ifo_cohort.local.json"
    path.write_text(
        json.dumps({"schema_version": "hok-agent-human-ifo-cohort-v1", "sessions": rows})
    )
    known = {str(row["session_hash"]): str(row["split"]) for row in rows}
    monkeypatch.setattr("hok_agent.human_ifo._manifest_sessions", lambda _root: known)
    selected, digest = load_human_cohort(path, tmp_path)
    assert len(selected) == 25
    assert len(digest) == 64

    rows[-1]["hero_id"] = "hero-b"
    path.write_text(
        json.dumps({"schema_version": "hok-agent-human-ifo-cohort-v1", "sessions": rows})
    )
    with pytest.raises(HumanIfoError, match="one hero"):
        load_human_cohort(path, tmp_path)


@pytest.mark.parametrize(
    "field", ["human_controlled", "complete_match", "hud_stable", "overlay_free", "rois_usable"]
)
def test_cohort_rejects_any_missing_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    rows = [_row(index) for index in range(20)] + [_row(100 + index, "dev") for index in range(5)]
    rows[0][field] = False
    path = tmp_path / "human_ifo_cohort.local.json"
    path.write_text(
        json.dumps({"schema_version": "hok-agent-human-ifo-cohort-v1", "sessions": rows})
    )
    monkeypatch.setattr(
        "hok_agent.human_ifo._manifest_sessions",
        lambda _root: {str(row["session_hash"]): str(row["split"]) for row in rows},
    )
    with pytest.raises(HumanIfoError, match="qualification"):
        load_human_cohort(path, tmp_path)


def test_cohort_rejects_manifest_split_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(index) for index in range(20)] + [_row(100 + index, "dev") for index in range(5)]
    path = tmp_path / "human_ifo_cohort.local.json"
    path.write_text(
        json.dumps({"schema_version": "hok-agent-human-ifo-cohort-v1", "sessions": rows})
    )
    known = {str(row["session_hash"]): str(row["split"]) for row in rows}
    known[str(rows[0]["session_hash"])] = "dev"
    monkeypatch.setattr("hok_agent.human_ifo._manifest_sessions", lambda _root: known)
    with pytest.raises(HumanIfoError, match="split"):
        load_human_cohort(path, tmp_path)


def test_unsupervised_proposal_selection_is_deterministic() -> None:
    rows = [
        {
            "session_hash": f"{index:064x}",
            "split": "train" if index < 20 else "dev",
            "technical_qc_passed": True,
            "signature": [float(index), 1.0],
        }
        for index in range(25)
    ]
    selected = _select_unsupervised_cohort(rows)
    assert len(selected) == 25
    assert sum(row["split"] == "train" for row in selected) == 20
    assert sum(row["split"] == "dev" for row in selected) == 5


def test_broad_unlabeled_rows_use_only_train_and_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    def selected(_root: Path, split: str) -> tuple[list[dict[str, object]], int]:
        count = 103 if split == "train" else 23
        return (
            [
                {"session_hashes": [f"{index + (0 if split == 'train' else 1000):064x}"]}
                for index in range(count)
            ],
            count,
        )

    monkeypatch.setattr("hok_agent.human_ifo._selected_video_shards", selected)
    rows, digest = _all_unlabeled_video_rows(Path("unused"))
    assert Counter(str(row["split"]) for row in rows) == Counter(train=103, dev=23)
    assert len(digest) == 64


def test_sim_temporal_groups_follow_episode_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hok_agent.human_ifo.load_global_manifest",
        lambda _root: {
            "episodes": [
                {"split": "train", "rows": 18},
                {"split": "dev", "rows": 30},
                {"split": "train", "rows": 19},
            ]
        },
    )
    assert _sim_episode_groups(Path("unused"), "train", 7).tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_encoder_rebind_contract_uses_no_human_labels_or_input() -> None:
    payload, digest = _load_rebind_contract(Path("configs/human_ifo_encoder_rebind_v1.json"))
    assert len(digest) == 64
    assert payload["human_labels_allowed"] is False
    assert payload["device_input_allowed"] is False


def test_challenge_curriculum_has_disjoint_parameterized_rgb() -> None:
    contract, digest = _load_curriculum_contract(
        Path("configs/global_challenge_curriculum_v1.json")
    )
    assert len(digest) == 64
    assert contract["human_video_used"] is False
    train = _curriculum_samples("train", 8)
    holdout = _curriculum_samples("holdout", 4)
    train_hashes = {sample[0].tobytes() for sample in train}
    holdout_hashes = {sample[0].tobytes() for sample in holdout}
    assert len(train) == 48
    assert len(holdout) == 24
    assert train_hashes.isdisjoint(holdout_hashes)


def test_observable_factor_probe_uses_disjoint_state_grids() -> None:
    train_views, train_labels = _observable_probe_data("train")
    dev_views, dev_labels = _observable_probe_data("dev")
    assert len(train_views[0]) == 240
    assert len(dev_views[0]) == 288
    assert (
        set(train_labels)
        == set(dev_labels)
        == {
            "health_bucket",
            "at_own_base",
            "enemy_distance_bucket",
            "push_condition",
            "ordinary_lane_advance",
        }
    )
    assert not torch.equal(train_views[0][:1], dev_views[0][:1])


def test_observable_factor_labels_are_current_observation_only() -> None:
    arena = GlobalArena()
    arena.reset(0)
    observation = arena.observe("blue")
    assert _observable_factor_labels(observation) == _observable_factor_labels(dict(observation))


def test_auxiliary_training_unfreezes_only_layer4_and_project() -> None:
    model = GlobalMacroPolicy("tcn")
    names = _configure_auxiliary_trainable(model)
    assert names
    assert all(name.startswith(("main.layer4.", "project.")) for name in names)
