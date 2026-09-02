from pathlib import Path

from hok_agent.hierarchical_e1d_test import load_test_contract


def test_one_shot_test_contract_is_frozen_before_test_open() -> None:
    config, payload, digest = load_test_contract(Path("configs/hierarchical_event_e1d_test.json"))
    assert config.sessions_expected == 23
    assert config.probability_threshold == 0.5
    assert len(digest) == 64
    assert payload["evaluation"]["retrain_or_threshold_tuning_after_test"] is False
    source = Path("src/hok_agent/hierarchical_e1d_test.py").read_text()
    assert "optimizer" not in source and ".backward(" not in source
