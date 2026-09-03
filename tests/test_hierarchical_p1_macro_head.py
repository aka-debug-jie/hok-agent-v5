from pathlib import Path

import torch

from hok_agent.hierarchical_p1_macro_head import (
    MacroHead,
    compose_policy_canvas,
    load_contract,
)


def test_p1_macro_head_contract_shapes_and_canvas() -> None:
    config, payload, digest = load_contract(Path("configs/hierarchical_p1_macro_head.json"))
    main = torch.zeros((2, 16, 3, 128, 128))
    minimap = torch.ones((2, 16, 3, 64, 64))
    hud = torch.full((2, 16, 3, 32, 128), 0.5)
    canvas = compose_policy_canvas(main, minimap, hud)
    assert canvas.shape == main.shape
    assert torch.allclose(canvas[:, :, :, :48, :48], torch.ones((2, 16, 3, 48, 48)))
    assert torch.allclose(
        canvas[:, :, :, 104:, 16:112], torch.full((2, 16, 3, 24, 96), 0.5)
    )
    assert MacroHead()(torch.zeros((2, 128))).shape == (2, 3)
    assert config.seed == 0 and len(digest) == 64
    assert payload["claim_boundary"]["p0_representation_frozen"] is True
    assert payload["claim_boundary"]["video_test_allowed"] is False
