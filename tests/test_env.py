# ruff: noqa: E501, E702
from __future__ import annotations

import pytest

from hok_agent.arena import PixelArena, attack_action, observation_hash, wait_action


def test_environment_is_deterministic_and_legal_actions_are_separate() -> None:
    first = PixelArena(); second = PixelArena(); left = first.reset(7); right = second.reset(7)
    assert left == right
    observation = left["observation"]; assert isinstance(observation, dict)
    assert "legal_actions" not in observation["blue"]
    action = first.legal_actions("blue")[1]
    assert first.step(action, wait_action()) == second.step(action, wait_action())


def test_illegal_action_is_rejected_before_state_changes() -> None:
    arena = PixelArena(); response = arena.reset(11)
    before = observation_hash(response["observation"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="illegal blue"):
        arena.step(attack_action("enemy_crystal"), wait_action())
    assert observation_hash(arena.public_state()) == before


def test_scripted_legal_path_reaches_both_objectives() -> None:
    arena = PixelArena(); arena.reset(3); targets: set[str] = set()
    for _ in range(arena.config.max_ticks):
        legal = arena.legal_actions("blue")
        action = next(
            (candidate for target in ("enemy_crystal", "enemy_tower") for candidate in legal if
             candidate.target == target),
            next(candidate for candidate in legal if candidate.action_type == "move"),
        )
        targets.add(action.target); result = arena.step(action, wait_action())
        if result["terminal"]:
            break
    assert targets >= {"enemy_tower", "enemy_crystal"}
    assert result["outcome"] == "blue_win_crystal_destroyed"
