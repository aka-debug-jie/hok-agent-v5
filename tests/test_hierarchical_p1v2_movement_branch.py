from pathlib import Path

import torch

from hok_agent.hierarchical_p0_ssl import TemporalSSL
from hok_agent.hierarchical_p1v2_movement_branch import (
    MovementBranch,
    load_contract,
    overfit_batch_order,
)


def test_p1v2_movement_branch_contract_shape_and_freeze() -> None:
    config, payload, digest = load_contract(
        Path("configs/hierarchical_p1v2_movement_branch.json")
    )
    state = {
        key: value
        for key, value in TemporalSSL().state_dict().items()
        if not key.startswith("classifier.")
    }
    model = MovementBranch(state)
    assert model(torch.zeros((2, 16, 3, 64, 64))).shape == (2, 8)
    assert not any(parameter.requires_grad for parameter in model.encoder.layer2.parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.layer3.parameters())
    assert config.epochs == 20 and len(digest) == 64
    assert payload["claim_boundary"]["capability"] == "local_visible_target_approach_only"
    assert payload["claim_boundary"]["video_test_allowed"] is False
    assert sorted(overfit_batch_order(32, 4, 0).tolist()) == list(range(32))
    assert payload["repair_history"]["model_data_optimizer_epoch_or_gate_changed"] is False
