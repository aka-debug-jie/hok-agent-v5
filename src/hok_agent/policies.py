from __future__ import annotations

import random

from hok_agent.arena import FactorizedAction, Side


class Policy:
    def select(self, side: Side, legal: tuple[FactorizedAction, ...]) -> FactorizedAction:
        raise NotImplementedError


class NullPolicy(Policy):
    def select(self, side: Side, legal: tuple[FactorizedAction, ...]) -> FactorizedAction:
        del side
        return next(action for action in legal if action.action_type == "wait")


class RandomPolicy(Policy):
    def __init__(self, seed: int, side: Side) -> None:
        self.rng = random.Random(seed * 2 + (0 if side == "blue" else 1))

    def select(self, side: Side, legal: tuple[FactorizedAction, ...]) -> FactorizedAction:
        del side
        return self.rng.choice(legal)


class ScriptedPolicy(Policy):
    def select(self, side: Side, legal: tuple[FactorizedAction, ...]) -> FactorizedAction:
        del side
        for target in ("enemy_crystal", "enemy_tower"):
            for action in legal:
                if action.action_type == "attack" and action.target == target:
                    return action
        for action in legal:
            if action.action_type == "move" and action.direction == "forward":
                return action
        return next(action for action in legal if action.action_type == "wait")


def make_policy(name: str, seed: int, side: Side) -> Policy:
    if name == "null":
        return NullPolicy()
    if name == "random":
        return RandomPolicy(seed, side)
    if name == "scripted":
        return ScriptedPolicy()
    raise ValueError(f"unknown policy: {name}")
