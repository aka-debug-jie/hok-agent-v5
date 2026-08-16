from __future__ import annotations

import torch

from hok_agent import v6_zero


def test_public_coach_always_abstains_without_a_v5_base() -> None:
    rgb = torch.zeros((1, 8, 3, 64, 64), dtype=torch.float32)
    output = v6_zero.TemporalZeroCoach()(rgb, torch.arange(8))
    assert output["advisory_action"] == [v6_zero.ABSTAIN]
    assert output["control_output"] is False
    assert output["metrics"] == {
        "v6_zero_release_binding_passed": False,
        "reason": "V5_ZERO_LABEL_BASE_NOT_AVAILABLE",
    }


def test_public_coach_malformed_input_is_still_fail_closed() -> None:
    output = v6_zero.TemporalZeroCoach()(torch.zeros((1, 4, 3, 64, 64)))
    assert output["advisory_action"] == [v6_zero.ABSTAIN]
    assert output["control_output"] is False


def test_internal_rgb_model_is_causal_and_resets_on_timestamp_regression() -> None:
    model = v6_zero.TemporalZeroModel()
    rgb = torch.rand((1, 8, 3, 64, 64), dtype=torch.float32)
    output = model(rgb, torch.arange(8) * 100)
    assert output["frame_logits"].shape == (1, 8, len(v6_zero.ACTION_NAMES))
    assert output["frame_ood"].shape == (1, 8)
    assert output["tracking_stability"].shape == (1,)
    assert output["possible_timestamps"] is True
    reset = model(rgb, torch.tensor([1000, 900, 950, 1000, 1050, 1100, 1150, 1200]))
    assert reset["possible_timestamps"] is False
    assert reset["reset_count"] >= 1


def test_cpu_smoke_has_no_release_binding_or_advice() -> None:
    result = v6_zero.cpu_smoke()
    assert result == {
        "status": "NON_PROMOTING_FRAMEWORK_SMOKE",
        "advisory": [v6_zero.ABSTAIN],
        "release_binding_passed": False,
    }
