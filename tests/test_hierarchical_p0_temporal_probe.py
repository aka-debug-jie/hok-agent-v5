from pathlib import Path

import torch

from hok_agent.hierarchical_p0_temporal_probe import TemporalHead, load_contract


def test_p0_temporal_probe_contract_and_head_shapes() -> None:
    config, payload, digest = load_contract(Path("configs/hierarchical_p0_temporal_probe.json"))
    features = torch.zeros((2, 16, 512))
    assert TemporalHead(64, "temporal")(features).shape == (2, 2)
    assert TemporalHead(64, "last_frame")(features).shape == (2, 2)
    assert config.seed == 0 and len(digest) == 64
    assert payload["claim_boundary"]["test_allowed"] is False
