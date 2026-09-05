from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hok_agent.movement_mvp import (
    MOVEMENT_ACTIONS,
    mark_visible_target,
    materialize_stage_c_trajectories,
    rule_movement_in_range,
    stage_c_arena,
    stage_c_scenarios,
    to_arena_action,
)
from hok_agent.movement_mvp_train import (
    RelationalMovement,
    TaskSpecificMovement,
    TrajectoryWindowDataset,
    _rollout,
    _window,
    evaluate_stage_c_dev,
    localized_train_step,
    run_overfit32,
    stage_c_dev_gates,
    train_step,
)
from hok_agent.rich_arena import wait_action
from hok_agent.rich_renderer import render

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "movement_mvp.json"
CONFIG_V2 = ROOT / "configs" / "movement_mvp_stage_c_v2.json"


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


def test_relational_model_consumes_rgb_without_coordinate_labels() -> None:
    model = RelationalMovement()
    clips = torch.zeros((2, 16, 3, 128, 128))
    clips[0, :, 0, 16:32, 16:32] = 1.0
    clips[1, :, 1, 96:112, 96:112] = 1.0
    assert model(clips).shape == (2, 9)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    before = model.attention.weight.detach().clone()
    loss, gradient = train_step(model, clips, torch.tensor([1, 2]), optimizer)
    assert math.isfinite(loss) and math.isfinite(gradient)
    assert gradient > 0.0
    assert not torch.equal(before, model.attention.weight)


def test_localized_train_step_updates_attention_from_automatic_targets() -> None:
    torch.manual_seed(0)
    model = RelationalMovement()
    clips = torch.zeros((2, 16, 3, 128, 128))
    clips[0, :, 0, 16:32, 16:32] = 1.0
    clips[1, :, 1, 96:112, 96:112] = 1.0
    coordinates = torch.zeros((2, 16, 2, 2))
    coordinates[0, :, 0] = torch.tensor([24.0, 24.0])
    coordinates[0, :, 1] = torch.tensor([104.0, 104.0])
    coordinates[1, :, 0] = torch.tensor([104.0, 104.0])
    coordinates[1, :, 1] = torch.tensor([24.0, 24.0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    before = model.attention.weight.detach().clone()
    losses = localized_train_step(
        model,
        clips,
        torch.tensor([1, 2]),
        coordinates,
        optimizer,
        action_weight=1.0,
        localization_weight=1.0,
        grid_size=16,
    )
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] > 0.0
    assert not torch.equal(before, model.attention.weight)


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


def test_stage_c_windows_never_cross_episode_and_rule_baselines_reach(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    materialize_stage_c_trajectories(CONFIG, root)
    dataset = TrajectoryWindowDataset(root, "train")
    clip, label = dataset[0]
    assert clip.shape == (16, 128, 128, 3)
    assert label.ndim == 0
    assert torch.equal(clip[0], clip[-1])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scenario = next(row for row in manifest["episodes"] if row["split"] == "dev")
    for policy in ("teacher", "geometry"):
        result = _rollout(scenario, policy, 128, 16, 7001)
        assert result["status"] == "success"


def test_recovery_actions_create_history_but_never_supervision(tmp_path: Path) -> None:
    root = tmp_path / "recovery"
    report = materialize_stage_c_trajectories(CONFIG_V2, root)
    assert report["teacher_successes"] == {"train": 64, "dev": 24}
    assert report["recovery_episodes"] == 16
    manifest = json.loads((root / "manifest.json").read_text())
    scenarios = manifest["episodes"]
    assert [s["scenario_id"] for s in scenarios if s["split"] == "dev"] == [
        s["scenario_id"] for s in stage_c_scenarios() if s["split"] == "dev"
    ]
    scenario = next(row for row in scenarios if row.get("recovery_cycles"))
    with np.load(root / "episodes" / scenario["basename"], allow_pickle=False) as data:
        frames, labels, ends = data["frames"], data["labels"], data["window_end"]
    assert len(frames) - len(ends) == 16
    assert ends[0] == 4
    arena = stage_c_arena(scenario, 128, navigation_only=True)
    for step, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        obs = arena.observe("blue")
        assert np.array_equal(
            frame, mark_visible_target(render(obs, scenario["render_seed"]), "opponent_hero")
        )
        action = MOVEMENT_ACTIONS[int(label)]
        if step in ends:
            pos = obs["self_position"]
            assert action == rule_movement_in_range((pos["x"], pos["y"]), scenario["goal"])
        arena.step(to_arena_action(action), wait_action())
    dataset = TrajectoryWindowDataset(root, "train")
    assert torch.equal(dataset[0][0], _window(list(frames[:5]), 16)[0])
    assert int(dataset[0][1]) == int(labels[4])
    assert np.array_equal(labels[-3:], [0, 0, 0])


def test_navigation_success_requires_three_consecutive_stops() -> None:
    class ScriptedPolicy(nn.Module):
        def __init__(self, actions: list[str]) -> None:
            super().__init__()
            self.actions = iter(actions)

        def forward(self, clips: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros((1, 9), device=clips.device)
            logits[0, MOVEMENT_ACTIONS.index(next(self.actions))] = 1.0
            return logits

    scenario = {"start": [4, 3], "goal": [6, 3], "render_seed": 0, "scenario_id": "stops"}
    result = _rollout(
        scenario,
        "learned",
        6,
        16,
        0,
        ScriptedPolicy(["E", "STOP", "E", "STOP", "STOP", "STOP"]),
        torch.device("cpu"),
        stop_confirmation_steps=3,
        record_trace=True,
        navigation_only=True,
    )
    assert result["status"] == "success"
    assert result["steps"] == 6
    assert result["stop_streak"] == 3
    assert result["collisions"] == 1
    result = _rollout(
        scenario,
        "learned",
        3,
        16,
        0,
        ScriptedPolicy(["STOP"] * 3),
        torch.device("cpu"),
        stop_confirmation_steps=3,
        navigation_only=True,
    )
    assert result["status"] == "timeout"
    assert result["steps"] == 3


def test_v2_comparison_is_feasible_and_does_not_reclassify_v1() -> None:
    v1 = json.loads(CONFIG.read_text())["stage_c"]
    v2 = json.loads(CONFIG_V2.read_text())["stage_c"]
    baselines = {
        "teacher": {"successes": 24},
        "random": {"successes": 18, "mean_steps": 62.0},
        "fixed_east": {"successes": 2, "mean_steps": 118.0},
    }
    perfect = {
        "successes": 24,
        "collision_fraction": 0.0,
        "oscillation_fraction": 0.0,
        "mean_steps": 6.0,
    }
    assert not all(stage_c_dev_gates(v1, baselines, perfect).values())
    assert all(stage_c_dev_gates(v2, baselines, perfect).values())
    assert not all(stage_c_dev_gates(v2, baselines, {**perfect, "mean_steps": 100.0}).values())
    assert not all(stage_c_dev_gates(v2, baselines, {**perfect, "successes": 20}).values())


def test_mismatched_checkpoint_is_reference_only_never_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hok_agent import movement_mvp_train

    root = tmp_path / "dataset"
    materialize_stage_c_trajectories(CONFIG_V2, root)
    checkpoint = tmp_path / "old.safetensors"
    movement_mvp_train.save_file(
        TaskSpecificMovement().state_dict(), checkpoint, metadata={"config_sha256": "old"}
    )

    def fixed_result(
        scenario: dict[str, object], policy: str, *args: object, **kwargs: object
    ) -> dict[str, object]:
        good = policy in ("teacher", "learned")
        return {
            "scenario_id": scenario["scenario_id"],
            "status": "success" if good else "timeout",
            "steps": 6 if good else 128,
            "collisions": 0,
            "movement_requests": 1,
            "oscillations": 0,
            "comparable_pairs": 1,
        }

    monkeypatch.setattr(movement_mvp_train, "_rollout", fixed_result)
    with pytest.raises(ValueError, match="checkpoint contract binding"):
        evaluate_stage_c_dev(CONFIG_V2, root, [checkpoint], tmp_path / "blocked", device_name="cpu")
    report = evaluate_stage_c_dev(
        CONFIG_V2,
        root,
        [checkpoint],
        tmp_path / "reference",
        device_name="cpu",
        reference_only=True,
    )
    assert all(report["gate_results"].values())
    assert report["reference_only"] is True
    assert report["passed"] is False
