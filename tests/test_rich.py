# ruff: noqa: E501, E701, E702
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hok_agent.rich_arena import (
    ACTION_TYPES,
    DIRECTIONS,
    MACROS,
    RICH_IDENTITY,
    SKILLS,
    TARGETS,
    FactorizedAction,
    Minion,
    ReplayError,
    RichNullPolicy,
    RichPixelArena,
    RichRandomPolicy,
    RichTeacherPolicy,
    action_factor_index,
    action_from_factor_index,
    canonical_actions,
    make_rich_policy,
    observation_hash,
    record_rich_trace,
    verify_rich_trace,
)
from hok_agent.rich_pixel import (
    RichPixelActor,
    _decode_legal_template,
    _ego_action,
    _formal_failure_report,
    _publish_failed_report,
    _teacher_episode,
    collect_rich_data,
    load_dataset,
    load_model,
    save_model,
    write_dataset,
)
from hok_agent.rich_renderer import render


def test_rich_constants_vocab_and_factor_api() -> None:
    arena = RichPixelArena(); assert arena.config.identity == RICH_IDENTITY; assert (arena.config.width, arena.config.height, arena.config.max_ticks) == (15, 7, 96); assert tuple(map(len, (MACROS, ACTION_TYPES, DIRECTIONS, TARGETS, SKILLS))) == (6, 4, 9, 5, 5)
    for sample in (0, 1, 17, 999, 1000):
        action = action_from_factor_index(sample); assert action_factor_index(action) == sample
    with pytest.raises(ValueError):
        action_from_factor_index(-1)
    action = next(a for a in canonical_actions() if a.macro == 'move'); assert action_factor_index(action) >= 0
def test_illegal_action_raises_and_state_is_atomic() -> None:
    arena = RichPixelArena(); response = arena.reset(11); before = observation_hash(response['observation']); legal_red = arena.legal_actions('red'); assert legal_red; illegal = FactorizedAction('attack', 'attack', target='enemy_crystal', direction='none', skill='basic')
    with pytest.raises(ValueError, match='illegal blue'):
        arena.step(illegal, legal_red[0])
    assert observation_hash(arena.public_state()) == before; assert arena.state.tick == 0
def test_same_target_cancel_and_swap_allowed() -> None:
    arena = RichPixelArena(); arena.reset(0); arena.state.blue.x, arena.state.blue.y = (7, 3); arena.state.red.x, arena.state.red.y = (9, 3); blue_east = next(action for action in arena.legal_actions('blue') if action.action_type == 'move' and action.direction == 'east'); red_west = next(action for action in arena.legal_actions('red') if action.action_type == 'move' and action.direction == 'west'); arena.step(blue_east, red_west); assert (arena.state.blue.x, arena.state.blue.y) == (7, 3); assert (arena.state.red.x, arena.state.red.y) == (9, 3); arena.state.blue.x, arena.state.blue.y = (7, 3); arena.state.red.x, arena.state.red.y = (8, 3); arena.state.blue.respawn = 0; arena.state.red.respawn = 0; arena.step(blue_east, red_west); assert (arena.state.blue.x, arena.state.blue.y) == (8, 3); assert (arena.state.red.x, arena.state.red.y) == (7, 3)
def test_skill_cd_and_recovery() -> None:
    arena = RichPixelArena(); arena.reset(7); dash = next(action for action in arena.legal_actions('blue') if action.action_type == 'skill' and action.skill == 'skill1'); arena.step(dash, arena.legal_actions('red')[0]); assert all(not (action.action_type == 'skill' and action.skill == 'skill1') for action in arena.legal_actions('blue'))
    for _ in range(4):
        arena.step(arena.legal_actions('blue')[0], arena.legal_actions('red')[0])
    assert any(action.action_type == 'skill' and action.skill == 'skill1' for action in arena.legal_actions('blue'))
def test_minion_spawn_every_six_ticks_and_tower_prefers_minion() -> None:
    arena = RichPixelArena(); arena.reset(0); wait_blue = arena.legal_actions('blue')[0]; wait_red = arena.legal_actions('red')[0]
    for _ in range(6):
        arena.step(wait_blue, wait_red)
    assert len(arena.state.blue_minions) == 2; assert len(arena.state.red_minions) == 2; arena.state.red.x, arena.state.red.y = (10, 3); arena.state.blue_minions = [Minion(10, 3, 3, 'blue')]; arena.state.blue.health = 10; before_blue_health = arena.state.blue.health; arena.step(wait_blue, wait_red); assert arena.state.blue.health == before_blue_health; assert arena.state.blue_minions and arena.state.blue_minions[0].health == 1
def test_minion_same_target_intents_cancel_symmetrically() -> None:
    arena = RichPixelArena(); arena.reset(0); arena.state.blue_minions = [Minion(6, 2, 3, 'blue')]; arena.state.red_minions = [Minion(8, 2, 3, 'red')]; arena.step(arena.legal_actions('blue')[0], arena.legal_actions('red')[0]); assert (arena.state.blue_minions[0].x, arena.state.red_minions[0].x) == (6, 8)
def test_renderer_determinism_and_mutation() -> None:
    arena = RichPixelArena(); arena.reset(3); obs = arena.observe('blue'); frame_a = render(obs, render_seed=19); frame_b = render(obs, render_seed=19); frame_c = render(obs, render_seed=20); assert frame_a.shape == (128, 128, 3); assert np.array_equal(frame_a, frame_b); assert not np.array_equal(frame_a, frame_c); modified = dict(obs); modified['self_health'] = modified.get('self_health', 0) - 1; frame_modified = render(modified, render_seed=19); assert not np.array_equal(frame_a, frame_modified)
def test_red_view_is_180_self_perspective() -> None:
    arena = RichPixelArena(); arena.reset(12); assert np.array_equal(render(arena.observe('blue'), 3), render(arena.observe('red'), 3))
def test_teacher_trajectories_are_seeded_and_complete() -> None:
    rows = [_teacher_episode(seed, side) for seed in (0, 71) for side in ('blue', 'red')]; assert len({row[0] for row in rows}) == 4; assert all(row[3] for row in rows)
def test_teacher_has_no_hidden_episode_mode_or_semantic_label_conflict() -> None:
    with pytest.raises(TypeError):
        RichTeacherPolicy(tower_drill=True)
    data = collect_rich_data(range(8), variants=2, enforce=False); assert len(data.episodes) == 16
def test_baseline_policies_and_replay_tamper() -> None:
    arena = RichPixelArena(); arena.reset(12); legal = arena.legal_actions('blue'); teacher = RichTeacherPolicy(); null = RichNullPolicy(); random_a = make_rich_policy('random', 22, 'blue'); random_b = make_rich_policy('random', 22, 'blue'); assert isinstance(teacher.select('blue', tuple(legal), 0), FactorizedAction); assert teacher.select('blue', tuple(legal), 1).macro in {'hold', 'move', 'attack', 'dash', 'projectile', 'targeted'}; assert null.select('blue', tuple(legal)) == next(a for a in legal if a.action_type == 'wait'); assert isinstance(random_a, RichRandomPolicy); assert isinstance(random_b, RichRandomPolicy); arena_a = RichPixelArena(); arena_a.reset(22); arena_b = RichPixelArena(); arena_b.reset(22); assert random_a.select('blue', arena_a.legal_actions('blue')) == random_b.select('blue', arena_b.legal_actions('blue')); path = Path('/tmp/rich_trace_test.jsonl'); traced = record_rich_trace(path, 'teacher', 'null', 12); assert traced['ticks'] > 0; verified = verify_rich_trace(path); assert verified['verified'] is True; lines = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]; lines[-1]['terminal'] = 'tampered'; path.write_text('\n'.join(json.dumps(line, sort_keys=True, separators=(',', ':')) for line in lines) + '\n', encoding='utf-8')
    with pytest.raises(ReplayError):
        verify_rich_trace(path)
def test_rgb_actor_decode_dataset_and_metadata(tmp_path: Path) -> None:
    actor = RichPixelActor().eval()
    with torch.no_grad():
        logits = actor(torch.zeros((1, 3, 128, 128)))
    assert [head.shape[1] for head in logits] == [6, 4, 9, 5, 5]; arena = RichPixelArena(); legal = arena.legal_actions('blue'); executed, _ = _decode_legal_template(logits, legal); assert executed in legal; data = collect_rich_data(range(1), variants=2, enforce=False); assert set(data.variants.tolist()) == {0, 1}; dataset, model = (tmp_path / 'data.npz', tmp_path / 'model.safetensors'); write_dataset(dataset, data); assert not {'legal', 'reward', 'state'} & set(load_dataset(dataset)); save_model(model, actor, data.config, 2); loaded, seed = load_model(model, data.config); assert seed == 2 and type(loaded) is RichPixelActor
def test_red_self_view_uses_ego_direction_labels() -> None:
    west = next(action for action in canonical_actions() if action.action_type == 'move' and action.direction == 'west'); red_ego = _ego_action(west, 'red'); assert red_ego.direction == 'east'; assert _ego_action(red_ego, 'red') == west
def test_teacher_vs_null_is_blue_red_frame_and_action_symmetric() -> None:
    blue_arena, red_arena = (RichPixelArena(), RichPixelArena()); blue_arena.reset(0); red_arena.reset(0); blue_teacher, red_teacher = (RichTeacherPolicy(), RichTeacherPolicy()); null = RichNullPolicy()
    for tick in range(blue_arena.config.max_ticks):
        blue_observation = blue_arena.observe('blue'); red_observation = red_arena.observe('red'); assert np.array_equal(render(blue_observation, 0), render(red_observation, 0)); blue = blue_teacher.select('blue', blue_arena.legal_actions('blue'), tick, blue_observation); red = red_teacher.select('red', red_arena.legal_actions('red'), tick, red_observation); assert blue == _ego_action(red, 'red'); blue_arena.step(blue, null.select('red', blue_arena.legal_actions('red'))); red_arena.step(null.select('blue', red_arena.legal_actions('blue')), red); assert blue_arena.state.terminal == red_arena.state.terminal
        if blue_arena.state.terminal:
            break
    assert blue_arena.state.outcome == 'blue_win_crystal_destroyed'; assert red_arena.state.outcome == 'red_win_crystal_destroyed'
def test_seeded_random_is_ego_symmetric() -> None:
    blue_arena, red_arena = (RichPixelArena(), RichPixelArena()); blue_arena.reset(19); red_arena.reset(19); blue_random = RichRandomPolicy(19, 'blue'); red_random = RichRandomPolicy(19, 'red')
    for _ in range(32):
        blue = blue_random.select('blue', blue_arena.legal_actions('blue')); red = red_random.select('red', red_arena.legal_actions('red')); assert blue == _ego_action(red, 'red'); blue_arena.step(blue, RichNullPolicy().select('red', blue_arena.legal_actions('red'))); red_arena.step(RichNullPolicy().select('blue', red_arena.legal_actions('blue')), red); assert np.array_equal(render(blue_arena.observe('blue'), 3), render(red_arena.observe('red'), 3))
        if blue_arena.state.terminal:
            break
def test_failed_formal_report_retains_only_nonpromoting_diagnostics(tmp_path: Path) -> None:
    data = collect_rich_data(range(1), variants=2, enforce=False); runs = [{'seed': seed, 'test_passed': True} for seed in (0, 1, 2)]; controls: dict[str, object] = {'passed': False, 'evaluated_seed': 0, 'joint_accuracy_drop': {'black': 0.5, 'mismatched': 0.1}, 'checks': {'black_joint_accuracy_drop_over_0_20': True, 'mismatch_joint_accuracy_drop_over_0_20': False}, 'failed_checks': ['mismatch_joint_accuracy_drop_over_0_20']}; closed = [{'training_seed': seed, 'passed': seed != 1, 'null_completion': 1.0, 'random_completion': 0.9, 'blue_red_completion_gap': 0.1 if seed == 1 else 0.0, 'raw_illegal_rate': 0.0, 'correction_rate': 0.0, 'executed_illegal': 0, 'checks': {'blue_red_gap_at_most_0_05': seed != 1}, 'failed_checks': [] if seed != 1 else ['blue_red_gap_at_most_0_05']} for seed in (0, 1, 2)]; report = _formal_failure_report(data, runs, controls, closed, {'verified': True}, 0, {'python': 'test', 'torch': 'test', 'device': 'test'}); requested = tmp_path / 'rich-v7-v1'; failed = _publish_failed_report(requested, report); assert not requested.exists(); assert failed.parent == requested.parent; assert failed.name.startswith('rich-v7-v1.failed-'); assert {path.name for path in failed.iterdir()} == {'report.json'}; loaded = json.loads((failed / 'report.json').read_text(encoding='utf-8')); assert loaded['status'] == 'FAILED'; assert loaded['promotion_eligible'] is False; assert loaded['models_retained'] is False; assert loaded['files'] == {}; assert [row['training_seed'] for row in loaded['closed_loop']] == [0, 1, 2]; assert loaded['selected_evaluation_seed'] == loaded['controls']['evaluated_seed'] == 0; assert loaded['failed_checks'] == ['controls.mismatch_joint_accuracy_drop_over_0_20', 'closed_loop.seed_1.blue_red_gap_at_most_0_05']
