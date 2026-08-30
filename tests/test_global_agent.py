from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

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
    assert result["non_timeout_terminals"] >= 16
    assert result["tower_progress_episodes"] >= 14
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
