from pathlib import Path

from hok_agent.hierarchical_p1_macro_data import load_contract, macro_f1


def test_p1_macro_data_contract_and_metric() -> None:
    config, payload, digest = load_contract(Path("configs/hierarchical_p1_macro_data.json"))
    assert macro_f1([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    assert config.window_frames == 16 and len(digest) == 64
    assert payload["selection"]["router_owned_intents"] == ["DISENGAGE", "RECALL"]
    assert payload["selection"]["test_allowed"] is False
    assert payload["labels"]["real_video_semantics_verified"] is False
    assert payload["claim_boundary"]["movement_head_training_allowed"] is False
