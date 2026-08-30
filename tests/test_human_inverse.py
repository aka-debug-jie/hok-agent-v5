from __future__ import annotations

from pathlib import Path

import torch

from hok_agent.human_inverse import (
    _apply_segment_hold,
    _load_style_contract,
    _pairs,
    load_broad_acceptance,
    load_gate_b_acceptance,
)


def test_broad_acceptance_is_self_hashed_and_simulator_only() -> None:
    payload, digest = load_broad_acceptance(Path("configs/human_ifo_broad_acceptance_v1.json"))
    assert len(digest) == 64
    assert payload["gate_b_simulator_only_allowed"] is True
    assert payload["gate_c_allowed"] is False
    assert payload["device_input_allowed"] is False


def test_inverse_pairs_never_cross_episode_boundaries() -> None:
    values = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    labels = torch.arange(6)
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    inputs, intent, zone = _pairs(values, labels, labels, groups, 2, state_only=False)
    assert inputs.shape == (2, 6)
    assert intent.tolist() == [0, 3]
    assert zone.tolist() == [0, 3]


def test_gate_b_engineering_acceptance_still_blocks_human_bc() -> None:
    payload, digest = load_gate_b_acceptance(Path("configs/human_ifo_gate_b_acceptance_v1.json"))
    assert len(digest) == 64
    assert payload["gate_c_pseudolabel_audit_allowed"] is True
    assert payload["human_bc_allowed"] is False


def test_segment_hold_requires_four_stable_samples() -> None:
    rows = [{"session_hash": "a", "intent": 1, "zone": 2, "confidence": 0.9} for _ in range(3)]
    assert _apply_segment_hold(rows) == [False, False, False]
    rows.append({"session_hash": "a", "intent": 1, "zone": 2, "confidence": 0.9})
    assert _apply_segment_hold(rows) == [True, True, True, True]


def test_transition_style_contract_keeps_phone_closed() -> None:
    payload, digest = _load_style_contract(Path("configs/human_ifo_transition_style_v1.json"))
    assert len(digest) == 64
    assert payload["human_transition_reward_weight"] == 1.0
    assert payload["mobile_input_allowed"] is False
