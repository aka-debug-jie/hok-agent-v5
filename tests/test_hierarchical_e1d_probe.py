from pathlib import Path

import torch

from hok_agent.hierarchical_e1d_probe import TerminalProbe, load_contract


def test_e1d_probe_contract_and_shapes() -> None:
    config, payload, digest = load_contract(Path("configs/hierarchical_event_e1d_probe.json"))
    clips = torch.zeros((2, 16, 3, 32, 32))
    assert TerminalProbe("temporal")(clips).shape == (2, 2)
    assert TerminalProbe("last_frame")(clips).shape == (2, 2)
    assert config.seed == 0 and len(digest) == 64
    assert payload["claim_boundary"]["reward_allowed"] is False
