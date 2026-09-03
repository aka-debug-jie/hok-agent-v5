from pathlib import Path

import torch

from hok_agent.hierarchical_p0 import FrozenResNet, load_contract


def test_p0_contract_and_frozen_encoder_shape() -> None:
    config, payload, digest = load_contract(Path("configs/hierarchical_p0_adapter_value.json"))
    model = FrozenResNet(None)
    output = model(torch.zeros((2, 3, 128, 128)))
    assert output.shape == (2, 512)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert config.seed == 0 and len(digest) == 64
    assert payload["claim_boundary"]["test_allowed"] is False
