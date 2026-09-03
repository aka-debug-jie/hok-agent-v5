from pathlib import Path

from hok_agent.hierarchical_e1e_unused import load_contract


def test_e1e_contract_cannot_replace_test_or_integrate() -> None:
    payload, digest = load_contract(Path("configs/hierarchical_event_e1e_unused.json"))
    assert len(digest) == 64
    assert payload["selection"]["source_splits"] == ["train", "dev"]
    assert payload["selection"]["test_allowed"] is False
    assert payload["claim_boundary"]["formal_test_replacement"] is False
    assert payload["claim_boundary"]["integration_allowed"] is False
