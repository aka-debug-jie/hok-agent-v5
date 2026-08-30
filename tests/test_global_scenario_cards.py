from __future__ import annotations

import json
from pathlib import Path

import pytest

from hok_agent.global_scenario_cards import GlobalScenarioCardError, verify_scenario_card_contract


def test_scenario_card_contract_is_fixed_and_zero_label() -> None:
    result = verify_scenario_card_contract()
    assert result["status"] == "PASSED"
    assert result["card_count"] == 5
    assert result["episodes_required"] == 100
    assert result["human_frame_labels_used"] is False
    assert result["runtime_app_state_input"] is False
    assert result["device_input_allowed"] is False


def test_scenario_card_contract_rejects_scope_drift(tmp_path: Path) -> None:
    source = Path("configs/global_agent_scenario_cards_v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["sample_hz"] = 10
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GlobalScenarioCardError, match="hash mismatch"):
        verify_scenario_card_contract(path)
