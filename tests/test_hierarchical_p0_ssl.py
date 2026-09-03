from pathlib import Path

import torch

from hok_agent.hierarchical_p0_ssl import TemporalSSL, load_contract


def test_p0_ssl_contract_and_model_shape() -> None:
    config, payload, digest = load_contract(Path("configs/hierarchical_p0_temporal_ssl.json"))
    logits, state, projection = TemporalSSL()(torch.zeros((2, 16, 3, 64, 64)))
    assert logits.shape == (2, 2)
    assert state.shape == (2, 128)
    assert projection.shape == (2, 64)
    assert config.seed == 0 and len(digest) == 64
    assert payload["repair_history"] == {
        "repairs_allowed": 1,
        "repairs_used": 1,
        "repair": "set_cublas_workspace_before_first_training_update",
        "model_data_or_gate_changed": False,
    }
    assert payload["claim_boundary"]["old_adapter_loaded"] is False
    assert payload["claim_boundary"]["video_test_allowed"] is False
    assert payload["claim_boundary"]["p0_initialization_allowed_only_if_all_gates_pass"] is True
