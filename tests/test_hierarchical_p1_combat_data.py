from pathlib import Path

from hok_agent.hierarchical_p1_combat_data import load_contract, macro_f1


def test_p1_combat_data_contract_and_metric() -> None:
    config, payload, digest = load_contract(Path("configs/hierarchical_p1_combat_data.json"))
    assert macro_f1([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == 1.0
    assert config.sample_period_ms == 200 and len(digest) == 64
    assert payload["labels"]["meaning"] == "executed_button_not_tactical_choice"
    assert payload["labels"]["hero_identity_artifact_verified"] is False
    assert payload["selection"]["test_allowed"] is False
