from __future__ import annotations

import hashlib
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
    TaskSpecificLastFrame,
    TaskSpecificMovement,
    TrajectoryWindowDataset,
    _action_step,
    _localization_step,
    _rollout,
    _window,
    evaluate_stage_c_dev,
    localized_train_step,
    run_change_policy_pilot,
    run_joystick_grouped_pilot,
    run_joystick_overfit32,
    run_overfit32,
    run_real_counterfactual_grouped_eval,
    stage_c_dev_gates,
    train_step,
)
from hok_agent.movement_real_rgb import _object_sha256
from hok_agent.rich_arena import wait_action
from hok_agent.rich_renderer import render

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "movement_mvp.json"
CONFIG_V2 = ROOT / "configs" / "movement_mvp_stage_c_v2.json"
JOYSTICK_CONFIG = ROOT / "configs" / "joystick_overfit32.json"
JOYSTICK_PILOT_CONFIG = ROOT / "configs" / "joystick_grouped_pilot.json"
JOYSTICK_SCALE21_CONFIG = ROOT / "configs" / "joystick_scale21_pilot.json"
JOYSTICK_CONTINUATION_CONFIG = ROOT / "configs" / "joystick_continuation_pilot.json"
CHANGE_CONFIG = ROOT / "configs" / "movement_change_pilot_v1.json"
CHANGE_V2_CONFIG = ROOT / "configs" / "movement_change_position_v2_pilot.json"
REAL_COUNTERFACTUAL_TRAIN = (
    ROOT / "configs" / "movement_real_counterfactual_overfit32_train_v1.json"
)
REAL_GROUPED_EVAL = ROOT / "configs" / "movement_real_counterfactual_grouped_eval_v1.json"


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


def test_joystick_overfit_rejects_unbound_dataset_before_output(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "joystick-overfit32.npz").write_bytes(b"changed")
    (dataset / "manifest.json").write_text("{}")
    (dataset / "conclusion.json").write_text(json.dumps({
        "diagnostic_overfit_allowed": True, "formal_training_allowed": False,
    }))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="contract differs"):
        run_joystick_overfit32(JOYSTICK_CONFIG, dataset, output, device_name="cpu")
    assert not output.exists()


def test_joystick_grouped_pilot_rejects_unbound_dataset_before_output(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "joystick-grouped-pilot.npz").write_bytes(b"changed")
    (dataset / "manifest.json").write_text("{}")
    (dataset / "conclusion.json").write_text(json.dumps({
        "pilot_training_allowed": True, "checkpoint_promotion_allowed": False,
    }))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="contract differs"):
        run_joystick_grouped_pilot(
            JOYSTICK_PILOT_CONFIG, dataset, output, device_name="cpu"
        )
    assert not output.exists()


def test_joystick_scale21_pilot_rejects_old_grouped_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "joystick-scale21-grouped.npz").write_bytes(b"old grouped data")
    (dataset / "manifest.json").write_text("{}")
    (dataset / "conclusion.json").write_text(json.dumps({
        "pilot_training_allowed": True, "checkpoint_promotion_allowed": False,
    }))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="contract differs"):
        run_joystick_grouped_pilot(
            JOYSTICK_SCALE21_CONFIG, dataset, output, device_name="cpu"
        )
    assert not output.exists()


def test_joystick_continuation_pilot_rejects_nine_class_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "joystick-continuation-grouped.npz").write_bytes(b"nine-class data")
    (dataset / "manifest.json").write_text("{}")
    (dataset / "conclusion.json").write_text(json.dumps({
        "continuation_training_allowed": True, "checkpoint_promotion_allowed": False,
    }))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="contract differs"):
        run_joystick_grouped_pilot(
            JOYSTICK_CONTINUATION_CONFIG, dataset, output, device_name="cpu"
        )
    assert not output.exists()


def test_task_specific_model_consumes_sixteen_rgb_frames() -> None:
    model = TaskSpecificMovement()
    assert model(torch.zeros((2, 16, 3, 128, 128))).shape == (2, 9)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000
    continuation = TaskSpecificMovement(8)
    assert continuation(torch.zeros((2, 16, 3, 128, 128))).shape == (2, 8)
    assert sum(parameter.numel() for parameter in continuation.parameters()) == 686_152
    last_frame = TaskSpecificLastFrame(8)
    assert last_frame(torch.zeros((2, 16, 3, 128, 128))).shape == (2, 8)
    assert sum(parameter.numel() for parameter in last_frame.parameters()) == 587_080


def test_change_policy_pilot_rejects_unbound_dataset_before_output(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "change-events.npz").write_bytes(b"changed")
    (dataset / "manifest.json").write_text("{}")
    (dataset / "conclusion.json").write_text(json.dumps({
        "pilot_training_allowed": True, "checkpoint_promotion_allowed": False,
    }))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="contract differs"):
        run_change_policy_pilot(CHANGE_CONFIG, dataset, output, device_name="cpu")
    assert not output.exists()


def test_change_position_v2_pilot_rejects_v1_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "change-events-position-v2.npz").write_bytes(b"v1 data")
    (dataset / "manifest.json").write_text("{}")
    (dataset / "conclusion.json").write_text(json.dumps({
        "pilot_training_allowed": True, "checkpoint_promotion_allowed": False,
    }))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="contract differs"):
        run_change_policy_pilot(CHANGE_V2_CONFIG, dataset, output, device_name="cpu")
    assert not output.exists()


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


def test_two_stage_training_freezes_localizer_before_action_update() -> None:
    torch.manual_seed(0)
    model = RelationalMovement()
    clips = torch.zeros((2, 16, 3, 128, 128))
    clips[0, :, 0, 16:32, 16:32] = 1.0
    clips[1, :, 1, 96:112, 96:112] = 1.0
    coordinates = torch.zeros((2, 16, 2, 2))
    coordinates[:, :, 0] = torch.tensor([24.0, 24.0])
    coordinates[:, :, 1] = torch.tensor([104.0, 104.0])
    localizer = [*model.spatial.parameters(), *model.attention.parameters()]
    optimizer = torch.optim.AdamW(localizer, lr=0.001, weight_decay=0.0)
    before = model.attention.weight.detach().clone()
    loss, gradient = _localization_step(model, clips, coordinates, optimizer, 16)
    assert math.isfinite(loss) and gradient > 0.0
    assert not torch.equal(before, model.attention.weight)
    for parameter in localizer:
        parameter.requires_grad = False
    frozen = model.attention.weight.detach().clone()
    project = model.project[0].weight.detach().clone()
    action_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.001,
        weight_decay=0.0,
    )
    loss, gradient = _action_step(model, clips, torch.tensor([1, 2]), action_optimizer)
    assert math.isfinite(loss) and gradient > 0.0
    assert torch.equal(frozen, model.attention.weight)
    assert not torch.equal(project, model.project[0].weight)


def test_default_contract_closes_fifth_diagnostic(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="attempt limit is exhausted"):
        run_overfit32(
            CONFIG,
            tmp_path / "unused.npz",
            None,
            tmp_path / "blocked",
            device_name="cpu",
        )


def test_real_counterfactual_training_contract_rejects_dataset_drift(
    tmp_path: Path,
) -> None:
    raw = json.loads(REAL_COUNTERFACTUAL_TRAIN.read_text(encoding="utf-8"))
    raw["overfit_dataset_sha256"] = "0" * 64
    raw.pop("contract_sha256")
    raw["contract_sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    config = tmp_path / "contract.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"drift")
    with pytest.raises(ValueError, match="training contract differs"):
        run_overfit32(config, dataset, None, tmp_path / "output", device_name="cpu")


def test_real_grouped_eval_isolates_groups_and_persists_no_checkpoints(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    directory = session_root / "teacher-session-002"
    (directory / "shards").mkdir(parents=True)
    frames = np.zeros((192, 128, 128, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        y = 40 + index // 5
        frame[y : y + 8, 52:60] = (20, 180, 40)
        frame[y + 1 : y + 7, 60:66] = (200, 40, 30)
    shard = directory / "shards" / "observations-0000.npz"
    np.savez_compressed(
        shard,
        minimap_rgb=frames,
        scheduled_elapsed_ms=np.arange(len(frames), dtype=np.int64) * 200,
    )
    summary = {
        "status": "PASSED",
        "derived_roi_rgb_persisted": True,
        "raw_frames_persisted": False,
        "observation_shards": [
            {
                "path": shard.name,
                "rows": len(frames),
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    summary["summary_sha256"] = _object_sha256(summary)
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    localization = {
        "status": "PLAYER_CUE_PARTIAL_SESSION002_ONLY",
        "semantic_identity_scope": "session002_partial_action_response_supported",
    }
    localization["report_sha256"] = _object_sha256(localization)
    localization_path = tmp_path / "localization.json"
    localization_path.write_text(json.dumps(localization), encoding="utf-8")

    old_clips = np.arange(32, dtype=np.uint8)[:, None, None, None]
    old_clips[29:] = old_clips[:3]
    old_dataset_path = tmp_path / "old.npz"
    np.savez_compressed(
        old_dataset_path,
        rgb_sequence=old_clips,
        end_timestamp_ms=np.resize(np.arange(5, dtype=np.int64), 32),
    )
    old_dataset_sha = hashlib.sha256(old_dataset_path.read_bytes()).hexdigest()
    old_report = {"status": "PASSED", "dataset_sha256": old_dataset_sha}
    old_report["report_sha256"] = _object_sha256(old_report)
    old_report_path = tmp_path / "old-report.json"
    old_report_path.write_text(json.dumps(old_report), encoding="utf-8")

    contract = json.loads(REAL_GROUPED_EVAL.read_text(encoding="utf-8"))
    contract["localization_report_file_sha256"] = hashlib.sha256(
        localization_path.read_bytes()
    ).hexdigest()
    contract["localization_report_sha256"] = localization["report_sha256"]
    contract["old_data_report_file_sha256"] = hashlib.sha256(
        old_report_path.read_bytes()
    ).hexdigest()
    contract["old_data_report_sha256"] = old_report["report_sha256"]
    contract["old_dataset_sha256"] = old_dataset_sha
    contract["source_session"] = {
        "basename": "teacher-session-002",
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    }
    contract["training"]["updates"] = 1
    contract.pop("contract_sha256")
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "grouped-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    output = tmp_path / "grouped-output"
    report = run_real_counterfactual_grouped_eval(
        contract_path,
        localization_path,
        session_root,
        old_dataset_path,
        old_report_path,
        output,
        device_name="cpu",
    )
    assert report["status"] == "REAL_COUNTERFACTUAL_MODEL_SHORTCUT_OR_NO_GENERALIZATION"
    assert report["model_runs"] == 15
    assert report["checkpoints_persisted"] == 0
    assert len(report["generated_data"]["source_windows"]) == 5
    assert len(report["generated_data"]["samples"]) == 45
    assert report["generated_data"]["duplicates_by_variant"] == {
        "full": 0,
        "goal_only": 0,
        "player_masked": 0,
    }
    assert set(report["generated_data"]["variant_sha256"]) == {
        "full",
        "goal_only",
        "player_masked",
    }
    assert all(row["train_samples"] == 36 for row in report["folds"])
    assert all(row["dev_samples"] == 9 for row in report["folds"])
    assert all(row["source_group_leakage"] is False for row in report["folds"])
    assert report["old_dataset_duplicate_audit"]["duplicate_clips"] == 3
    assert {path.name for path in output.iterdir()} == {"report.json"}


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


def test_native_relation_variants_are_deterministic_and_source_preserving(tmp_path: Path) -> None:
    from hok_agent.movement_mvp_train import _native_relation_variants

    source = np.zeros((2, 16, 256, 256, 3), dtype=np.uint8)
    yy, xx = np.indices((256, 256))
    ring = (np.hypot(yy - 100, xx - 110) >= 9) & (np.hypot(yy - 100, xx - 110) <= 11)
    for group in source:
        for frame in group:
            frame[ring] = (20, 220, 30)
    targets = np.asarray(
        [
            [100, 110],
            [76, 110],
            [124, 110],
            [100, 86],
            [100, 134],
            [76, 86],
            [76, 134],
            [124, 86],
            [124, 134],
        ],
        dtype=np.int16,
    )
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        source_clips=source,
        split=np.asarray(["train", "dev"]),
        anchor_yx=np.asarray([[100, 110], [100, 110]], dtype=np.int16),
        sample_group_index=np.repeat(np.arange(2), 9),
        sample_label=np.tile(np.arange(9), 2),
        sample_target_yx=np.tile(targets, (2, 1)),
    )
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()
    variants, labels, splits, metadata = _native_relation_variants(dataset)
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == before
    assert {name: value.shape for name, value in variants.items()} == {
        "full": (18, 16, 128, 128, 3),
        "anchor_masked": (18, 16, 128, 128, 3),
        "goal_only": (18, 16, 128, 128, 3),
    }
    assert labels.tolist() == list(range(9)) * 2
    assert splits.tolist() == ["train"] * 9 + ["dev"] * 9
    assert len(set(metadata["variant_sha256"].values())) == 3
    repaired, repaired_labels, repaired_splits, repaired_metadata = _native_relation_variants(
        dataset, render_after_resize=True
    )
    assert repaired_labels.tolist() == labels.tolist()
    assert repaired_splits.tolist() == splits.tolist()
    assert repaired_metadata["render_order"] == "resize_128_then_mark_radius7"
    assert (
        np.any(repaired["full"][1] != repaired["full"][0], axis=-1).sum()
        > np.any(variants["full"][1] != variants["full"][0], axis=-1).sum()
    )
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == before


def test_native_relation_runner_passes_controls_and_stops_after_overfit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hok_agent import movement_mvp_train as module

    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(dataset, placeholder=np.asarray([1]))
    data_report = {
        "schema_version": "native-weak-anchor-counterfactual-dataset-v1",
        "status": "WEAK_ANCHOR_COUNTERFACTUAL_DATASET_READY",
        "relation_diagnostic_training_allowed": True,
        "movement_policy_training_allowed": False,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    data_report["report_sha256"] = _object_sha256(data_report)
    report_path = tmp_path / "data-report.json"
    report_path.write_text(json.dumps(data_report))
    variants = {
        name: np.zeros((18, 16, 128, 128, 3), dtype=np.uint8)
        for name in ("full", "anchor_masked", "goal_only")
    }
    labels = np.tile(np.arange(9), 2)
    splits = np.asarray(["train"] * 9 + ["dev"] * 9)
    monkeypatch.setattr(
        module, "_native_relation_variants", lambda _path, **_kwargs: (variants, labels, splits, {})
    )
    calls: list[int] = []

    def trained(
        *_args: object, **_kwargs: object
    ) -> tuple[TaskSpecificMovement, dict[str, object]]:
        index = len(calls)
        calls.append(index)
        accuracy = [1.0, 0.9, 0.5, 0.4][index]
        return TaskSpecificMovement(), {
            "accuracy": accuracy,
            "macro_f1": accuracy,
            "recall": {action: accuracy for action in MOVEMENT_ACTIONS},
            "cross_entropy": 0.01,
            "first_update_loss": 2.0,
            "first_update_gradient_norm": 1.0,
            "first_update_parameter_changed": True,
            "final_update_loss": 0.01,
        }

    monkeypatch.setattr(module, "_train_native_relation_variant", trained)
    report = module.run_native_anchor_relation_diagnostic(
        dataset, report_path, tmp_path / "passed", device_name="cpu"
    )
    assert report["status"] == "WEAK_ANCHOR_RELATION_SIGNAL_PASSED"
    assert report["model_runs"] == 4 and all(report["checks"].values())
    assert report["checkpoint_saved"] is report["movement_policy_training_allowed"] is False
    assert list((tmp_path / "passed").iterdir()) == [tmp_path / "passed" / "report.json"]

    monkeypatch.setattr(
        module,
        "_train_native_relation_variant",
        lambda *_args, **_kwargs: (
            TaskSpecificMovement(),
            {"accuracy": 0.5, "cross_entropy": 1.0},
        ),
    )
    failed = module.run_native_anchor_relation_diagnostic(
        dataset, report_path, tmp_path / "failed", device_name="cpu"
    )
    assert failed["status"] == "WEAK_ANCHOR_RELATION_DIAGNOSTIC_FAILED"
    assert failed["model_runs"] == 1 and failed["results"] == {}
    assert failed["checks"]["overfit36"] is False

    calls.clear()
    monkeypatch.setattr(module, "_train_native_relation_variant", trained)
    repaired = module.run_native_anchor_relation_diagnostic(
        dataset,
        report_path,
        tmp_path / "repaired",
        device_name="cpu",
        repair_report_path=tmp_path / "failed" / "report.json",
    )
    assert repaired["status"] == "WEAK_ANCHOR_RELATION_SIGNAL_PASSED"
    assert repaired["repair_of_report_sha256"] == failed["report_sha256"]
    assert repaired["model_runs"] == 4
