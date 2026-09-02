from pathlib import Path

from hok_agent.hierarchical_e1d_test import (
    load_test_contract,
    record_consumed_runtime_failure,
)


def test_one_shot_test_contract_is_frozen_before_test_open() -> None:
    config, payload, digest = load_test_contract(Path("configs/hierarchical_event_e1d_test.json"))
    assert config.sessions_expected == 23
    assert config.probability_threshold == 0.5
    assert len(digest) == 64
    assert payload["evaluation"]["retrain_or_threshold_tuning_after_test"] is False
    source = Path("src/hok_agent/hierarchical_e1d_test.py").read_text()
    assert "optimizer" not in source and ".backward(" not in source


def test_consumed_runtime_failure_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    bundle = {"bundle_sha256": "55a679883119cdbf1a6a7703f945d61ce33408bad84013362e66355e83345c79"}
    monkeypatch.setattr(
        "hok_agent.hierarchical_e1d_test.verify_checkpoint_bundle",
        lambda _bundle, _contract: bundle,
    )
    payload = record_consumed_runtime_failure(
        Path("configs/hierarchical_event_e1d_test.json"),
        Path("configs/hierarchical_event_e1d_checkpoint.json"),
        tmp_path / "bundle",
        tmp_path / "failed",
        "TEST_SESSION_NO_PRE_RESULT_SEQUENCE",
    )
    assert payload["rerun_allowed"] is False
    assert payload["integration_allowed"] is False
    assert payload["reward_allowed"] is False
