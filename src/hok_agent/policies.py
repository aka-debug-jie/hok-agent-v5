from __future__ import annotations

import random
from collections.abc import Callable

from hok_agent.arena import FactorizedAction, Side


class Policy:
    def select(self, side: Side, legal: tuple[FactorizedAction, ...],
               tick: int | None = None) -> FactorizedAction:
        raise NotImplementedError


class NullPolicy(Policy):
    def select(self, side: Side, legal: tuple[FactorizedAction, ...],
               tick: int | None = None) -> FactorizedAction:
        del side, tick
        return next(action for action in legal if action.action_type == "wait")


class RandomPolicy(Policy):
    def __init__(self, seed: int, side: Side) -> None:
        self.rng = random.Random(seed * 2 + (0 if side == "blue" else 1))

    def select(self, side: Side, legal: tuple[FactorizedAction, ...],
               tick: int | None = None) -> FactorizedAction:
        del side, tick; return self.rng.choice(legal)  # noqa: E702


class ScriptedPolicy(Policy):
    def select(self, side: Side, legal: tuple[FactorizedAction, ...],
               tick: int | None = None) -> FactorizedAction:
        del side; del tick  # noqa: E702
        for target in ("enemy_crystal", "enemy_tower"):
            for action in legal:
                if action.action_type == "attack" and action.target == target:
                    return action
        for action in legal:
            if action.action_type == "move" and action.direction == "forward":
                return action
        return next(action for action in legal if action.action_type == "wait")


class TacticalTeacher(Policy):
    def __init__(self, start_tick: int = 0) -> None:
        self._tick = start_tick

    def select(self, side: Side, legal: tuple[FactorizedAction, ...],
               tick: int | None = None) -> FactorizedAction:
        del side
        if tick is None:
            tick = self._tick
            self._tick += 1

        def first(predicate: Callable[[FactorizedAction], bool]) -> FactorizedAction | None:
            for action in legal:
                if predicate(action):
                    return action
            return None

        wait = first(lambda action: action.action_type == "wait")
        if wait is None: raise ValueError("illegal legal set: no wait action")  # noqa: E701

        if tick % 17 == 0: return wait  # noqa: E701

        attack_hero = first(
            lambda action: action.action_type == "attack" and action.target == "enemy_hero"
        )
        if tick % 7 == 1 and attack_hero is not None: return attack_hero  # noqa: E701

        move_backward = first(
            lambda action: action.action_type == "move" and action.direction == "backward"
        )
        if tick % 11 == 3 and move_backward is not None: return move_backward  # noqa: E701

        attack_crystal = first(
            lambda action: action.action_type == "attack" and action.target == "enemy_crystal"
        )
        if attack_crystal is not None: return attack_crystal  # noqa: E701

        attack_tower = first(
            lambda action: action.action_type == "attack" and action.target == "enemy_tower"
        )
        if attack_tower is not None: return attack_tower  # noqa: E701

        move_forward = first(
            lambda action: action.action_type == "move" and action.direction == "forward"
        )
        if move_forward is not None: return move_forward  # noqa: E701

        return wait


def make_policy(name: str, seed: int, side: Side) -> Policy:
    if name == "null": return NullPolicy()  # noqa: E701
    if name == "random": return RandomPolicy(seed, side)  # noqa: E701
    if name == "scripted": return ScriptedPolicy()  # noqa: E701
    if name == "tactical": return TacticalTeacher()  # noqa: E701
    raise ValueError(f"unknown policy: {name}")
