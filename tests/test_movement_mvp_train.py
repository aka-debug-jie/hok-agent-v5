from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hok_agent.movement_mvp_train import TaskSpecificMovement, run_overfit32, train_step

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "movement_mvp.json"


def test_shared_train_step_updates_parameters() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 9))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    clips = torch.randn((8, 1, 3, 2, 2))
    labels = torch.arange(8) % 9
    before = model[-1].weight.detach().clone()
    loss, gradient_norm = train_step(model, clips, labels, optimizer)
    assert loss > 0.0
    assert gradient_norm > 0.0
    assert math.isfinite(loss) and math.isfinite(gradient_norm)
    assert not torch.equal(before, model[-1].weight)


def test_task_specific_model_consumes_sixteen_rgb_frames() -> None:
    model = TaskSpecificMovement()
    assert model(torch.zeros((2, 16, 3, 128, 128))).shape == (2, 9)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_default_contract_closes_fifth_diagnostic(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="attempt limit is exhausted"):
        run_overfit32(
            CONFIG,
            tmp_path / "unused.npz",
            None,
            tmp_path / "blocked",
            device_name="cpu",
        )


def test_failed_overfit_records_first_update_and_checkpoint(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["stage_b"]["maximum_updates"] = 1
    raw["stage_b"]["diagnostic_attempts_used"] = 0
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    labels = np.asarray([0] * 8 + [label for label in range(1, 9) for _ in range(3)])
    dataset = tmp_path / "overfit32.npz"
    np.savez_compressed(
        dataset,
        rgb_sequence=np.zeros((32, 16, 128, 128, 3), dtype=np.uint8),
        label=labels,
    )
    output = tmp_path / "failed"
    report = run_overfit32(config, dataset, None, output, device_name="cpu")
    assert report["status"] == "FAILED"
    assert report["full_training_called"] is False
    assert report["diagnostic_checkpoint_only"] is True
    assert report["diagnostic_checkpoint_reusable_for_formal_training"] is False
    assert report["first_update_parameter_changed"] is True
    assert report["first_update_finite"] is True
    assert float(report["first_update_loss"]) > 0.0
    assert float(report["first_update_gradient_norm"]) > 0.0
    assert (output / "diagnostic-last.safetensors").is_file()
    with pytest.raises(ValueError, match="explicitly requires"):
        run_overfit32(
            config,
            dataset,
            None,
            tmp_path / "p0-explicit",
            device_name="cpu",
            architecture="p0-branch",
        )
