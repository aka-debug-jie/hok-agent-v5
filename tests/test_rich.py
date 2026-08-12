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
    _teacher_episode,
    collect_rich_data,
    load_dataset,
    load_model,
    save_model,
    write_dataset,
)
from hok_agent.rich_renderer import render


def test_rich_constants_vocab_and_factor_api() -> None:
    arena = RichPixelArena()
    assert arena.config.identity == RICH_IDENTITY
    assert (arena.config.width, arena.config.height, arena.config.max_ticks) == (15, 7, 96)
    assert tuple(map(len, (MACROS, ACTION_TYPES, DIRECTIONS, TARGETS, SKILLS))) == (6, 4, 9, 5, 5)
    for sample in (0, 1, 17, 999, 1000):
        action = action_from_factor_index(sample)
        assert action_factor_index(action) == sample
    with pytest.raises(ValueError):
        action_from_factor_index(-1)
    action = next(a for a in canonical_actions() if a.macro == "move")
    assert action_factor_index(action) >= 0


def test_illegal_action_raises_and_state_is_atomic() -> None:
    arena = RichPixelArena()
    response = arena.reset(11)
    before = observation_hash(response["observation"])
    legal_red = arena.legal_actions("red")
    assert legal_red
    illegal = FactorizedAction(
        "attack", "attack", target="enemy_crystal", direction="none", skill="basic"
    )
    with pytest.raises(ValueError, match="illegal blue"):
        arena.step(illegal, legal_red[0])
    assert observation_hash(arena.public_state()) == before
    assert arena.state.tick == 0


def test_same_target_cancel_and_swap_allowed() -> None:
    arena = RichPixelArena()
    arena.reset(0)
    arena.state.blue.x, arena.state.blue.y = 7, 3
    arena.state.red.x, arena.state.red.y = 9, 3

    blue_east = next(
        action
        for action in arena.legal_actions("blue")
        if action.action_type == "move" and action.direction == "east"
    )
    red_west = next(
        action
        for action in arena.legal_actions("red")
        if action.action_type == "move" and action.direction == "west"
    )
    arena.step(blue_east, red_west)
    assert (arena.state.blue.x, arena.state.blue.y) == (7, 3)
    assert (arena.state.red.x, arena.state.red.y) == (9, 3)

    arena.state.blue.x, arena.state.blue.y = 7, 3
    arena.state.red.x, arena.state.red.y = 8, 3
    arena.state.blue.respawn = 0
    arena.state.red.respawn = 0
    arena.step(blue_east, red_west)
    assert (arena.state.blue.x, arena.state.blue.y) == (8, 3)
    assert (arena.state.red.x, arena.state.red.y) == (7, 3)


def test_skill_cd_and_recovery() -> None:
    arena = RichPixelArena()
    arena.reset(7)
    dash = next(
        action
        for action in arena.legal_actions("blue")
        if action.action_type == "skill" and action.skill == "skill1"
    )
    arena.step(dash, arena.legal_actions("red")[0])
    assert all(
        not (action.action_type == "skill" and action.skill == "skill1")
        for action in arena.legal_actions("blue")
    )
    for _ in range(4):
        arena.step(arena.legal_actions("blue")[0], arena.legal_actions("red")[0])
    assert any(
        action.action_type == "skill" and action.skill == "skill1"
        for action in arena.legal_actions("blue")
    )


def test_minion_spawn_every_six_ticks_and_tower_prefers_minion() -> None:
    arena = RichPixelArena()
    arena.reset(0)
    wait_blue = arena.legal_actions("blue")[0]
    wait_red = arena.legal_actions("red")[0]
    for _ in range(6):
        arena.step(wait_blue, wait_red)
    assert len(arena.state.blue_minions) == 2
    assert len(arena.state.red_minions) == 2

    arena.state.red.x, arena.state.red.y = 10, 3
    arena.state.blue_minions = [Minion(10, 3, 3, "blue")]
    arena.state.blue.health = 10
    before_blue_health = arena.state.blue.health
    arena.step(wait_blue, wait_red)
    assert arena.state.blue.health == before_blue_health
    assert arena.state.blue_minions and arena.state.blue_minions[0].health == 1


def test_renderer_determinism_and_mutation() -> None:
    arena = RichPixelArena()
    arena.reset(3)
    obs = arena.observe("blue")
    frame_a = render(obs, render_seed=19)
    frame_b = render(obs, render_seed=19)
    frame_c = render(obs, render_seed=20)
    assert frame_a.shape == (128, 128, 3)
    assert np.array_equal(frame_a, frame_b)
    assert not np.array_equal(frame_a, frame_c)

    modified = dict(obs)
    modified["self_health"] = modified.get("self_health", 0) - 1
    frame_modified = render(modified, render_seed=19)
    assert not np.array_equal(frame_a, frame_modified)


def test_red_view_is_180_self_perspective() -> None:
    arena = RichPixelArena()
    arena.reset(12)
    assert np.array_equal(render(arena.observe("blue"), 3), render(arena.observe("red"), 3))


def test_teacher_trajectories_are_seeded_complete_and_cover_tower_attack() -> None:
    rows = [_teacher_episode(seed, side) for seed in (0, 71) for side in ("blue", "red")]
    assert len({row[0] for row in rows}) == 4
    assert all(row[3] for row in rows)
    assert all(any(step.template == 3 for step in row[1]) for row in rows)


def test_baseline_policies_and_replay_tamper() -> None:
    arena = RichPixelArena()
    arena.reset(12)
    legal = arena.legal_actions("blue")

    teacher = RichTeacherPolicy()
    null = RichNullPolicy()
    random_a = make_rich_policy("random", 22, "blue")
    random_b = make_rich_policy("random", 22, "blue")

    assert isinstance(teacher.select("blue", tuple(legal), 0), FactorizedAction)
    assert teacher.select("blue", tuple(legal), 1).macro in {
        "hold",
        "move",
        "attack",
        "dash",
        "projectile",
        "targeted",
    }
    assert null.select("blue", tuple(legal)) == next(a for a in legal if a.action_type == "wait")
    assert isinstance(random_a, RichRandomPolicy)
    assert isinstance(random_b, RichRandomPolicy)
    arena_a = RichPixelArena()
    arena_a.reset(22)
    arena_b = RichPixelArena()
    arena_b.reset(22)
    assert random_a.select("blue", arena_a.legal_actions("blue")) == random_b.select(
        "blue", arena_b.legal_actions("blue")
    )

    path = Path("/tmp/rich_trace_test.jsonl")
    traced = record_rich_trace(path, "teacher", "null", 12)
    assert traced["ticks"] > 0
    verified = verify_rich_trace(path)
    assert verified["verified"] is True

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[-1]["terminal"] = "tampered"
    path.write_text(
        "\n".join(json.dumps(line, sort_keys=True, separators=(",", ":")) for line in lines) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ReplayError):
        verify_rich_trace(path)


def test_rgb_actor_decode_dataset_and_metadata(tmp_path: Path) -> None:
    actor = RichPixelActor().eval()
    with torch.no_grad():
        logits = actor(torch.zeros((1, 3, 128, 128)))
    assert [head.shape[1] for head in logits] == [6, 4, 9, 5, 5]
    arena = RichPixelArena()
    legal = arena.legal_actions("blue")
    executed, _ = _decode_legal_template(logits, legal)
    assert executed in legal
    data = collect_rich_data(range(1), variants=2, enforce=False)
    assert set(data.variants.tolist()) == {0, 1}
    dataset, model = tmp_path / "data.npz", tmp_path / "model.safetensors"
    write_dataset(dataset, data)
    assert not {"legal", "reward", "state"} & set(load_dataset(dataset))
    save_model(model, actor, data.config, 2)
    loaded, seed = load_model(model, data.config)
    assert seed == 2 and type(loaded) is RichPixelActor


def test_red_self_view_uses_ego_direction_labels() -> None:
    west = next(
        action
        for action in canonical_actions()
        if action.action_type == "move" and action.direction == "west"
    )
    red_ego = _ego_action(west, "red")
    assert red_ego.direction == "east"
    assert _ego_action(red_ego, "red") == west
