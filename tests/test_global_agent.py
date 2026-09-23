from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from hok_agent import global_policy as gp
from hok_agent.global_agent import (
    ENABLED_INTENTS,
    ENABLED_ZONES,
    GlobalRuleTeacher,
    MacroIntent,
    ProgressWatchdog,
    TargetZone,
    TargetZoneNavigator,
    evaluate_teacher,
    load_global_config,
)
from hok_agent.global_policy import (
    GlobalMacroPolicy,
    GlobalPolicyError,
    GlobalWindowDataset,
    _adapter_promotion_allowed,
    _challenge_arena,
    _holdout_eligible,
    _holdout_order,
    _is_static_window,
    _selected_video_shards,
    load_global_manifest,
    materialize_global_dataset,
    real_video_views,
)
from hok_agent.rich_arena import RichPixelArena, move_action, wait_action


def test_global_contract_keeps_full_vocab_but_enables_middle_lane_slice() -> None:
    assert len(MacroIntent) == 9
    assert len(TargetZone) == 10
    assert set(ENABLED_INTENTS) == {
        MacroIntent.FARM_LANE,
        MacroIntent.PUSH_STRUCTURE,
        MacroIntent.ENGAGE,
        MacroIntent.DISENGAGE,
        MacroIntent.RECALL,
    }
    assert set(ENABLED_ZONES) == {
        TargetZone.OWN_BASE,
        TargetZone.MID_LANE,
        TargetZone.ENEMY_BASE,
        TargetZone.HOLD_CURRENT_ZONE,
    }
    config, digest = load_global_config()
    assert config["device_input_allowed"] is False
    assert config["video_test_allowed"] is False
    assert len(digest) == 64


def test_rule_teacher_is_a_function_of_current_observation() -> None:
    arena = RichPixelArena()
    arena.reset(17)
    observation = arena.observe("blue")
    legal = arena.legal_actions("blue")
    first = GlobalRuleTeacher().decide("blue", legal, deepcopy(observation))
    second = GlobalRuleTeacher().decide("blue", legal, deepcopy(observation))
    assert first == second


def test_target_zone_navigator_uses_middle_waypoints() -> None:
    arena = RichPixelArena()
    observation = arena.observe("blue")
    legal = arena.legal_actions("blue")
    navigator = TargetZoneNavigator()
    assert navigator.action("blue", TargetZone.MID_LANE, observation, legal) == move_action("east")
    assert (
        navigator.action("blue", TargetZone.HOLD_CURRENT_ZONE, observation, legal)
        == wait_action()
    )


def test_progress_watchdog_stops_after_three_recoveries() -> None:
    observation = RichPixelArena().observe("blue")
    watchdog = ProgressWatchdog(max_no_progress_ticks=1, max_recoveries=3)
    assert not watchdog.observe(observation)
    assert not watchdog.observe(observation)
    assert not watchdog.observe(observation)
    assert watchdog.observe(observation)


def test_stage_1a_seed_17_reaches_crystal_terminal() -> None:
    result = evaluate_teacher(1, 17)
    assert result["status"] == "PASSED"
    assert result["non_timeout_terminals"] == 1
    report = result["reports"][0]
    assert report["crystal_destroyed"] is True
    assert report["invalid_actions"] == 0


def test_stage_1b_twenty_seed_gate_passes() -> None:
    result = evaluate_teacher(20, 17)
    assert result["status"] == "PASSED"
    assert result["non_timeout_terminals"] == 19
    assert result["tower_progress_episodes"] == 20
    assert result["death_recovery_rate"] == 1.0
    assert result["stuck_time_ratio"] < 0.10


def test_small_global_dataset_is_episode_split_and_rgb_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    output = tmp_path / "dataset"
    result = materialize_global_dataset(
        output, train_seeds=(1000, 1001), dev_seeds=(2000,), enforce=False
    )
    assert result["status"] == "PASSED"
    manifest = load_global_manifest(output)
    assert manifest["test_present"] is False
    assert manifest["source_paths_persisted"] is False
    assert {row["split"] for row in manifest["episodes"]} == {"train", "dev"}
    dataset = GlobalWindowDataset(output, "dev")
    main, minimap, hud, intent, zone, scene, tick = dataset[0]
    assert main.shape == (16, 3, 128, 128)
    assert minimap.shape == (16, 3, 64, 64)
    assert hud.shape == (16, 3, 32, 128)
    assert all(value.ndim == 0 for value in (intent, zone, scene, tick))


def test_global_manifest_tamper_and_test_split_fail_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    output = tmp_path / "dataset"
    materialize_global_dataset(output, train_seeds=(1000,), dev_seeds=(2000,), enforce=False)
    path = output / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["test_present"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GlobalPolicyError, match="hash mismatch"):
        load_global_manifest(output)


def test_global_policy_accepts_rgb_views_only() -> None:
    model = GlobalMacroPolicy("tcn")
    with torch.no_grad():
        intent, zone, scene = model(
            torch.zeros(1, 16, 3, 128, 128),
            torch.zeros(1, 16, 3, 64, 64),
            torch.zeros(1, 16, 3, 32, 128),
        )
    assert intent.shape == (1, 5)
    assert zone.shape == (1, 4)
    assert scene.shape == (1, 5)


def test_video_test_access_is_rejected_before_manifest_open(tmp_path) -> None:
    with pytest.raises(GlobalPolicyError, match="video-test access is prohibited"):
        _selected_video_shards(tmp_path / "missing", "test")


def test_real_video_views_and_static_negative_are_deterministic() -> None:
    frame = torch.arange(128 * 128 * 3, dtype=torch.int64).remainder(255).byte().numpy()
    frame = frame.reshape(128, 128, 3)
    main, minimap, hud = real_video_views(frame)
    assert main.shape == (128, 128, 3)
    assert minimap.shape == (64, 64, 3)
    assert hud.shape == (32, 128, 3)
    assert _is_static_window([frame] * 16)


def test_adapter_promotion_never_trades_terminal_for_consistency() -> None:
    assert _adapter_promotion_allowed(9, 9)
    assert not _adapter_promotion_allowed(9, 8)


def test_holdout_selector_uses_lexicographic_episode_metrics() -> None:
    base = {
        "non_timeout_terminals": 17,
        "tower_progress_episodes": 20,
        "mean_stuck_time_ratio": 0.05,
        "mean_fallback_rate": 0.06,
        "safety_violations": 0,
        "invalid_actions": 0,
    }
    dagger = {**base, "non_timeout_terminals": 18, "mean_fallback_rate": 0.05}
    assert _holdout_eligible(base)
    assert max(
        (("adapted", base), ("dagger", dagger)), key=lambda item: _holdout_order(*item)
    )[0] == "dagger"


def test_challenge_teacher_has_all_six_expected_semantics() -> None:
    teacher = GlobalRuleTeacher()
    expected = {
        "low_health_far_from_base": (MacroIntent.DISENGAGE, TargetZone.OWN_BASE),
        "low_health_at_base": (MacroIntent.RECALL, TargetZone.OWN_BASE),
        "wave_in_tower_range": (MacroIntent.PUSH_STRUCTURE, TargetZone.ENEMY_BASE),
        "enemy_hero_contact": (MacroIntent.ENGAGE, TargetZone.HOLD_CURRENT_ZONE),
        "ordinary_lane_advance": (MacroIntent.FARM_LANE, TargetZone.MID_LANE),
        "tower_destroyed_crystal_range": (MacroIntent.PUSH_STRUCTURE, TargetZone.ENEMY_BASE),
    }
    for name, value in expected.items():
        arena, intent, zone = _challenge_arena(name)
        assert (intent, zone) == value
        decision = teacher.decide("blue", arena.legal_actions("blue"), arena.observe("blue"))
        assert (decision.command.intent, decision.command.target_zone) == value


def test_public_offline_evidence_is_path_free_and_keeps_shadow_closed() -> None:
    path = Path(__file__).resolve().parents[1] / "docs/GLOBAL_AGENT_V1_OFFLINE_EVIDENCE.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["promoted_checkpoint_sha256"] == evidence["dagger"]["checkpoint_sha256"]
    assert evidence["adapter"]["promotion_allowed"] is False
    assert evidence["challenge_pack"]["student_passed"] is False
    assert evidence["mobile_shadow_authorized"] is False
    assert "/" not in path.read_text(encoding="utf-8")


# --- F3 distillation smoke: one gradient step, and the load path ---------------------------------
# The pack's minimal gradient/load smoke before any small-data pilot. It changes only the `main`
# view, which is what F3 allows, and it checks what the frozen loader does with such a student.
# This lives here because tests/test_global_agent.py is the allowlisted focused test for the line.

_MAIN_PARAMETERS = 11_168_832


def _distill_shard(path: Path, frames: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        main_rgb=rng.integers(0, 255, (frames, 16, 16, 3), dtype=np.uint8),
        minimap_rgb=rng.integers(0, 255, (frames, 4, 4, 3), dtype=np.uint8),
        hud_rgb=rng.integers(0, 255, (frames, 2, 4, 3), dtype=np.uint8),
        intent=np.arange(frames, dtype=np.int64) % 3,
        zone=np.arange(frames, dtype=np.int64) % 2,
        scene=np.zeros(frames, dtype=np.int64),
        tick=np.arange(frames, dtype=np.int64),
    )
    return gp._sha(path.read_bytes())


def _distill_dataset(
    root: Path, episodes: int, splits: tuple[str, ...] = ("dev",)
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in splits:
        for index in range(episodes):
            name = f"episode-{split}-{index}.npz"
            rows.append(
                {
                    "split": split,
                    "shard": name,
                    "shard_sha256": _distill_shard(root / name, gp.WINDOW_FRAMES, len(rows)),
                    "episode": f"ep-{split}-{index}",
                }
            )
    payload: dict[str, object] = {"test_present": False, "episodes": rows}
    payload["manifest_sha256"] = gp._sha(gp._canonical(payload).encode())
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _teacher_and_student() -> tuple[gp.GlobalMacroPolicy, gp.GlobalMacroPolicy]:
    teacher = gp.GlobalMacroPolicy("tcn")
    student = gp.GlobalMacroPolicy("tcn", "compact")
    return teacher, student


def test_a_smaller_main_keeps_the_interface_and_shrinks_the_model() -> None:
    teacher, student = _teacher_and_student()
    assert gp._parameter_groups(teacher)["main"] == _MAIN_PARAMETERS
    groups = gp._parameter_groups(student)
    assert groups["main"] < _MAIN_PARAMETERS / 10
    for name in ("minimap", "hud", "project", "temporal", "pool", "intent", "zone", "scene"):
        assert sum(p.numel() for p in getattr(student, name).parameters()) == sum(
            p.numel() for p in getattr(teacher, name).parameters()
        )


def test_one_distillation_step_touches_only_the_main_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    root = _distill_dataset(tmp_path / "dataset", episodes=2)
    dataset = gp.GlobalWindowDataset(root, "dev")
    batch = gp._speed_batch(dataset, list(range(2)))
    teacher, student = _teacher_and_student()
    teacher.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in student.main.parameters():
        parameter.requires_grad_(True)
    before = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if not name.startswith("main.")
    }

    optimizer = torch.optim.SGD(student.main.parameters(), lr=0.01)
    with torch.no_grad():
        target = teacher(*batch)
    logits = student(*batch)
    loss = sum(
        torch.nn.functional.mse_loss(student_logits, target_logits)
        for student_logits, target_logits in zip(logits, target, strict=True)
    )
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(
        p.grad is not None and bool(p.grad.abs().sum() > 0) for p in student.main.parameters()
    )
    for name, parameter in student.named_parameters():
        if name.startswith("main."):
            continue
        assert parameter.grad is None, name
        assert torch.equal(parameter.detach(), before[name]), name


def test_a_declared_variant_round_trips_through_the_frozen_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    _teacher, student = _teacher_and_student()
    checkpoint = tmp_path / "student.safetensors"
    gp._save_model(checkpoint, student, "tcn", "0" * 64)

    loaded, metadata = gp.load_global_model(checkpoint, torch.device("cpu"))
    assert metadata[gp.MAIN_ARCHITECTURE_METADATA_KEY] == "compact"
    assert loaded.main_architecture == "compact"
    assert gp._parameter_groups(loaded) == gp._parameter_groups(student)


def test_a_metadata_without_the_key_reads_as_the_historical_resnet18() -> None:
    assert gp.architecture_from_metadata({}) == gp.DEFAULT_MAIN_ARCHITECTURE
    assert gp.architecture_from_metadata({"variant": "tcn"}) == "resnet18"
    declared = {gp.MAIN_ARCHITECTURE_METADATA_KEY: "compact"}
    assert gp.architecture_from_metadata(declared) == "compact"
    with pytest.raises(gp.GlobalPolicyError):
        gp.architecture_from_metadata({gp.MAIN_ARCHITECTURE_METADATA_KEY: "tinyv2"})


def test_an_unknown_architecture_is_refused_at_construction() -> None:
    with pytest.raises(gp.GlobalPolicyError):
        gp.GlobalMacroPolicy("tcn", "tinyv2")


def test_distill_trains_only_main_and_copies_the_frozen_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    root = _distill_dataset(tmp_path / "dataset", episodes=2, splits=("train", "dev"))
    teacher = gp.GlobalMacroPolicy("tcn")
    teacher_path = tmp_path / "teacher.safetensors"
    gp._save_model(teacher_path, teacher, "tcn", "0" * 64)

    report = gp.distill_global_main(
        root,
        tmp_path / "run",
        teacher_checkpoint=teacher_path,
        main_architecture="compact",
        epochs=1,
        maximum_steps=2,
        batch_size=1,
        seed=0,
    )

    assert report["schema_version"] == gp.GLOBAL_DISTILL_SCHEMA
    assert report["steps"] == 2
    assert report["student_architecture"] == "compact"
    assert report["teacher_architecture"] == "resnet18"
    assert report["frozen_parameters_unchanged"] is True
    assert report["promotion_allowed"] is False
    assert report["parameters"]["total"] < report["parameters"]["main"] * 100
    loaded, metadata = gp.load_global_model(Path(report["student_checkpoint"]), torch.device("cpu"))
    assert loaded.main_architecture == "compact"
    assert metadata[gp.MAIN_ARCHITECTURE_METADATA_KEY] == "compact"


def test_distill_refuses_an_unknown_architecture_and_a_bad_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    root = _distill_dataset(tmp_path / "dataset", episodes=2, splits=("train", "dev"))
    teacher_path = tmp_path / "teacher.safetensors"
    gp._save_model(teacher_path, gp.GlobalMacroPolicy("tcn"), "tcn", "0" * 64)
    with pytest.raises(gp.GlobalPolicyError):
        gp.distill_global_main(
            root, tmp_path / "run-a", teacher_checkpoint=teacher_path, main_architecture="tinyv2"
        )
    with pytest.raises(gp.GlobalPolicyError):
        gp.distill_global_main(
            root, tmp_path / "run-b", teacher_checkpoint=teacher_path, maximum_steps=0
        )


def test_distill_uses_auxiliary_windows_and_still_leaves_frozen_weights_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    root = _distill_dataset(tmp_path / "dataset", episodes=4, splits=("train", "dev"))
    teacher_path = tmp_path / "teacher.safetensors"
    gp._save_model(teacher_path, gp.GlobalMacroPolicy("tcn"), "tcn", "0" * 64)
    dataset = gp.GlobalWindowDataset(root, "dev")
    samples = [dataset[i] for i in range(4)]

    def as_windows(value):  # (T, C, H, W) -> (N, T, H, W, C), the archive convention
        return value.permute(0, 2, 3, 1).numpy()

    auxiliary = tmp_path / "boundary-windows.npz"
    np.savez(
        auxiliary,
        main_rgb=np.stack([as_windows(s[0]) for s in samples]),
        minimap_rgb=np.stack([as_windows(s[1]) for s in samples]),
        hud_rgb=np.stack([as_windows(s[2]) for s in samples]),
        intent=np.zeros(4, dtype=np.int16),
        zone=np.zeros(4, dtype=np.int16),
        scene=np.zeros(4, dtype=np.int16),
        tick=np.arange(4, dtype=np.int32),
    )

    report = gp.distill_global_main(
        root,
        tmp_path / "run",
        teacher_checkpoint=teacher_path,
        main_architecture="resnet18_shallow",
        epochs=1,
        maximum_steps=2,
        batch_size=1,
        seed=1,
        auxiliary_windows=auxiliary,
    )
    assert report["steps"] == 2
    assert report["frozen_parameters_unchanged"] is True
    assert report["student_architecture"] == "resnet18_shallow"


def test_distill_refuses_a_symlinked_auxiliary_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    root = _distill_dataset(tmp_path / "dataset", episodes=2, splits=("train", "dev"))
    teacher_path = tmp_path / "teacher.safetensors"
    gp._save_model(teacher_path, gp.GlobalMacroPolicy("tcn"), "tcn", "0" * 64)
    real = tmp_path / "real.npz"
    np.savez(
        real,
        main_rgb=np.zeros((2, gp.WINDOW_FRAMES, 16, 16, 3), dtype=np.uint8),
        minimap_rgb=np.zeros((2, gp.WINDOW_FRAMES, 4, 4, 3), dtype=np.uint8),
        hud_rgb=np.zeros((2, gp.WINDOW_FRAMES, 2, 4, 3), dtype=np.uint8),
        intent=np.zeros(2, dtype=np.int16),
        zone=np.zeros(2, dtype=np.int16),
        scene=np.zeros(2, dtype=np.int16),
        tick=np.arange(2, dtype=np.int32),
    )
    link = tmp_path / "link.npz"
    link.symlink_to(real)
    with pytest.raises(gp.GlobalPolicyError):
        gp.distill_global_main(
            root,
            tmp_path / "run",
            teacher_checkpoint=teacher_path,
            maximum_steps=1,
            batch_size=1,
            auxiliary_windows=link,
        )


def test_truth_source_acceptance_contract_is_declared_but_not_implemented() -> None:
    """The truth-source acceptance is a proposal that reuses the R0 gates and builds nothing."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "game_rules" / "truth_source_acceptance_v1.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PROPOSED_NOT_IMPLEMENTED"
    assert payload["facts"]["minimum_viable"] == ["F1", "F2"]
    assert payload["device_input_allowed"] is False
    assert payload["training_allowed"] is False
    assert payload["reward_allowed"] is False
    assert payload["implementation_gate"]["may_start_only_after"].startswith(
        "the owner's source declaration"
    )


def test_truth_source_acceptance_reuses_the_r0_gate_values() -> None:
    """The proposed gates must match the R0 contract exactly, not redefine them."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r0 = json.loads(
        (root / "game_rules" / "r0_feedback_contract_v1.json").read_text(encoding="utf-8")
    )
    proposal = json.loads(
        (root / "game_rules" / "truth_source_acceptance_v1.json").read_text(encoding="utf-8")
    )
    assert proposal["gates"] == r0["gates"]
    assert proposal["duplication_guard"] == r0["independence"]["duplication_guard"]
