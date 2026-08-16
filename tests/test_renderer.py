# ruff: noqa: E501, E702
from __future__ import annotations

from random import Random

import numpy as np

from hok_agent.arena import (
    FactorizedAction,
    PixelArena,
    attack_action,
    move_action,
    wait_action,
)
from hok_agent.policies import TacticalTeacher
from hok_agent.renderer import render


def _classify(action: FactorizedAction) -> str:
    if action.action_type == "wait":
        return "wait"
    if action.action_type == "move" and action.direction == "forward":
        return "forward"
    if action.action_type == "move" and action.direction == "backward":
        return "backward"
    if action.action_type == "attack" and action.target == "enemy_hero":
        return "enemy_hero"
    if action.action_type == "attack" and action.target == "enemy_crystal":
        return "enemy_crystal"
    if action.action_type == "attack" and action.target == "enemy_tower":
        return "enemy_tower"
    return action.action_type


def test_renderer_deterministic_and_dtype_shape() -> None:
    arena = PixelArena()
    arena.reset(11)
    obs = arena.observe("blue")

    frame_one = render(obs, 123)
    frame_two = render(obs, 123)
    frame_three = render(obs, 124)

    assert frame_one.shape == (128, 128, 3)
    assert frame_one.dtype == np.uint8
    assert np.array_equal(frame_one, frame_two)
    assert not np.array_equal(frame_one, frame_three)


def test_renderer_key_fields_change_pixels() -> None:
    arena = PixelArena()
    arena.reset(22)
    obs = arena.observe("blue")

    base = render(obs, 777)
    hp = dict(obs); hp["self_health"] = hp["self_health"] - 1
    hp_frame = render(hp, 777)

    pos = dict(obs); pos["self_position"] = pos["self_position"] + 1
    pos_frame = render(pos, 777)

    tick = dict(obs); tick["tick"] = tick["tick"] + 1
    tick_frame = render(tick, 777)

    assert not np.array_equal(base, hp_frame)
    assert not np.array_equal(base, pos_frame)
    assert not np.array_equal(base, tick_frame)


def test_renderer_red_blue_self_perspective() -> None:
    blue_obs = {"side":"blue","tick":5,"max_ticks":32,"self_position":2,"opponent_position":8,"self_health":6,"opponent_health":6,"own_tower_health":4,"enemy_tower_health":4,"own_crystal_health":6,"enemy_crystal_health":6}
    red_obs = {"side":"red","tick":5,"max_ticks":32,"self_position":8,"opponent_position":2,"self_health":6,"opponent_health":6,"own_tower_health":4,"enemy_tower_health":4,"own_crystal_health":6,"enemy_crystal_health":6}
    blue_frame = render(blue_obs, 99)
    red_frame = render(red_obs, 99)
    assert np.array_equal(blue_frame, red_frame)


def test_tactical_teacher_coverage_rollout_terminal_and_crystal_completion() -> None:
    outcomes: list[str] = []
    action_classes: set[str] = set()

    for _seed in range(256):
        arena = PixelArena()
        arena.reset(_seed)
        blue = TacticalTeacher()
        red = TacticalTeacher()
        done = False; steps = 0

        while not done:
            blue_obs = arena.observe("blue")
            red_obs = arena.observe("red")
            blue_action = blue.select("blue", arena.legal_actions("blue"), _to_tick(blue_obs)); red_action = red.select("red", arena.legal_actions("red"), _to_tick(red_obs))
            action_classes.update({_classify(blue_action), _classify(red_action)})
            result = arena.step(blue_action, red_action); done = bool(result["terminal"])
            steps += 1
            if steps > 64:
                break

        outcomes.append(arena.state.outcome)

    assert len(outcomes) == 256
    assert all(outcome != "ongoing" for outcome in outcomes)
    for required in {"wait", "backward", "enemy_tower", "enemy_crystal", "forward"}:
        assert required in action_classes
    completed = sum("crystal" in outcome for outcome in outcomes)
    assert completed >= 243


def test_tactical_teacher_different_legal_inputs_cover_every_class() -> None:
    policy = TacticalTeacher()
    base = (wait_action(), move_action("forward"), move_action("backward"), attack_action("enemy_hero"), attack_action("enemy_tower"), attack_action("enemy_crystal"))
    legal_cases = [(0, (base[0], base[1]), "wait"), (1, (base[0], base[1], base[3]), "enemy_hero"), (3, (base[0], base[1], base[2], base[4]), "backward"), (2, (base[0], base[1], base[5]), "enemy_crystal"), (4, (base[0], base[1], base[4]), "enemy_tower"), (5, (base[0], base[1]), "forward")]

    policy_hits = set()

    for tick, legal, expected in legal_cases:
        result = policy.select("blue", legal, tick)
        policy_hits.add(_classify(result)); assert _classify(result) == expected

    random = Random(11)
    for _ in range(250):
        tick, legal, _ = random.choice(legal_cases); policy_hits.add(_classify(policy.select("blue", legal, tick)))

    assert policy_hits == {
        "wait",
        "enemy_hero",
        "backward",
        "enemy_crystal",
        "enemy_tower",
        "forward",
    }


def _to_tick(observation: dict[str, object]) -> int:
    return int(observation["tick"])
