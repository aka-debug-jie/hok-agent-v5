from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from hok_agent import panel_gating_policy as policy

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "game_rules" / "r1_panel_gating_contract_v1.json"


def test_r1_contract_loads_and_hashes() -> None:
    contract, sha = policy.load_r1_panel_gating_contract(CONTRACT)
    assert len(sha) == 64
    assert contract["milestone"] == "R1"
    assert contract["feedback_verification_class"] == "owner_authorized_bar"
    assert set(cast(list, cast(dict, contract["policy_input"])["excluded"])) == set(
        policy.R1_REQUIRED_EXCLUSIONS
    )
    assert cast(dict, contract["claim_boundary"])["promotion_allowed"] is False
    assert cast(dict, contract["claim_boundary"])["device_input_added"] is False


def test_r1_contract_keeps_the_current_panel_state_out_of_the_input() -> None:
    contract, _ = policy.load_r1_panel_gating_contract(CONTRACT)
    excluded = cast(dict, contract["policy_input"])["excluded"]
    assert "the current measured panel state" in excluded
    assert "action records" in excluded
    assert "reward components" in excluded


def test_r1_contract_rejects_a_relaxed_or_undeclared_bar(tmp_path: Path) -> None:
    tampered = tmp_path / "r1.json"
    mutations = (
        lambda value: value["pre_declared_requirements"].__setitem__(
            "minimum_candidate_accuracy_gain", 0.0
        ),
        lambda value: value["pre_declared_requirements"].__setitem__(
            "maximum_candidate_wrong_phase_dispatch_count", 3
        ),
        lambda value: value["claim_boundary"].__setitem__("promotion_allowed", True),
        lambda value: value["claim_boundary"].__setitem__(
            "independent_feedback_verification_claimed", True
        ),
        lambda value: value["policy_input"].__setitem__("excluded", ["device coordinates"]),
        lambda value: value["baseline"].__setitem__("frozen", False),
        lambda value: value.__setitem__("feedback_verification_class", "pre_registered"),
        lambda value: value.__setitem__("status", "draft"),
    )
    for mutate in mutations:
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        mutate(value)
        tampered.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(policy.PanelGatingError):
            policy.load_r1_panel_gating_contract(tampered)


def test_panel_input_window_is_the_last_k_views_only() -> None:
    views = [f"frame-{index}" for index in range(10)]
    assert policy.panel_input_window(views, 4) == ["frame-6", "frame-7", "frame-8", "frame-9"]
    with pytest.raises(policy.PanelGatingError):
        policy.panel_input_window(views, 1)
    with pytest.raises(policy.PanelGatingError):
        policy.panel_input_window(["only-one"], 4)


def test_policy_module_keeps_torch_out_of_module_level_imports() -> None:
    """The contract loader must stay dependency-free; torch is imported only when training runs."""
    import ast

    source = (ROOT / "src" / "hok_agent" / "panel_gating_policy.py").read_text(encoding="utf-8")
    module_level: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            module_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level.add(node.module.split(".")[0])
    assert "torch" not in module_level
    assert "torchvision" not in module_level
