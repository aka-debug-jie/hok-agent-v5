from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

Side = Literal["blue", "red"]
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/pixelarena_1v1_v1.json"


@dataclass(frozen=True)
class ArenaConfig:
    identity: str; ruleset: str  # noqa: E702
    lane_min: int; lane_max: int  # noqa: E702
    blue_start: int; red_start: int; blue_tower: int; red_tower: int  # noqa: E702
    blue_crystal: int; red_crystal: int  # noqa: E702
    hero_health: int; tower_health: int; crystal_health: int  # noqa: E702
    basic_damage: int; attack_range: int; max_ticks: int  # noqa: E702

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> ArenaConfig:
        raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        return cls(**raw)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FactorizedAction:
    macro: str; action_type: str  # noqa: E702
    target: str = "none"; direction: str = "none"; skill: str = "none"  # noqa: E702
    upgrade: str = "none"; auxiliary: int = 0  # noqa: E702

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> FactorizedAction:
        return cls(
            macro=str(raw["macro"]), action_type=str(raw["action_type"]),
            target=str(raw["target"]), direction=str(raw["direction"]),
            skill=str(raw["skill"]), upgrade=str(raw["upgrade"]),
            auxiliary=int(cast(int, raw["auxiliary"])),
        )


def wait_action() -> FactorizedAction: return FactorizedAction("hold", "wait")  # noqa: E704


def move_action(direction: str) -> FactorizedAction:
    return FactorizedAction("advance", "move", direction=direction)


def attack_action(target: str) -> FactorizedAction:
    macro = "siege" if target in {"enemy_tower", "enemy_crystal"} else "engage"
    return FactorizedAction(macro, "attack", target=target, skill="basic")


@dataclass
class ArenaState:
    tick: int; blue_position: int; red_position: int  # noqa: E702
    blue_health: int; red_health: int  # noqa: E702
    blue_tower_health: int; red_tower_health: int  # noqa: E702
    blue_crystal_health: int; red_crystal_health: int  # noqa: E702
    terminal: bool = False; outcome: str = "ongoing"  # noqa: E702


def observation_hash(observation: dict[str, object]) -> str:
    payload = json.dumps(observation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class PixelArena:
    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        self.config = ArenaConfig.load(config_path); self.seed = 0  # noqa: E702
        self.state = self._initial_state()

    def _initial_state(self) -> ArenaState:
        c = self.config
        return ArenaState(
            0, c.blue_start, c.red_start, c.hero_health, c.hero_health,
            c.tower_health, c.tower_health, c.crystal_health, c.crystal_health,
        )

    def health(self) -> dict[str, object]:
        return {
            "identity": self.config.identity,
            "ruleset": self.config.ruleset,
            "config_hash": self.config.digest,
            "claim_scope": "pixelarena_engineering",
            "hok_capability_claim": False, "gamecore_equivalence_claim": False,
            "capabilities": {"network": False, "device": False, "external_client": False},
        }

    def reset(self, seed: int) -> dict[str, object]:
        self.seed = seed; self.state = self._initial_state()  # noqa: E702
        return self._response([])

    def observe(self, side: Side) -> dict[str, object]:
        s = self.state
        own, enemy = ("blue", "red") if side == "blue" else ("red", "blue")
        return {
            "tick": s.tick,
            "max_ticks": self.config.max_ticks,
            "side": side,
            "self_position": getattr(s, f"{own}_position"),
            "opponent_position": getattr(s, f"{enemy}_position"),
            "self_health": getattr(s, f"{own}_health"),
            "opponent_health": getattr(s, f"{enemy}_health"),
            "own_tower_health": getattr(s, f"{own}_tower_health"),
            "enemy_tower_health": getattr(s, f"{enemy}_tower_health"),
            "own_crystal_health": getattr(s, f"{own}_crystal_health"),
            "enemy_crystal_health": getattr(s, f"{enemy}_crystal_health"),
            "terminal": s.terminal,
            "outcome": s.outcome,
        }

    def public_state(self) -> dict[str, object]:
        return {"blue": self.observe("blue"), "red": self.observe("red")}

    def legal_actions(self, side: Side) -> tuple[FactorizedAction, ...]:
        if self.state.terminal:
            return ()
        c, s = self.config, self.state
        position = s.blue_position if side == "blue" else s.red_position
        enemy_position = s.red_position if side == "blue" else s.blue_position
        enemy_tower_position = c.red_tower if side == "blue" else c.blue_tower
        enemy_crystal_position = c.red_crystal if side == "blue" else c.blue_crystal
        enemy_tower_health = s.red_tower_health if side == "blue" else s.blue_tower_health
        enemy_crystal_health = s.red_crystal_health if side == "blue" else s.blue_crystal_health
        actions = [wait_action()]
        forward = position + (1 if side == "blue" else -1)
        backward = position + (-1 if side == "blue" else 1)
        if c.lane_min <= forward <= c.lane_max:
            actions.append(move_action("forward"))
        if c.lane_min <= backward <= c.lane_max:
            actions.append(move_action("backward"))
        if abs(position - enemy_position) <= c.attack_range:
            actions.append(attack_action("enemy_hero"))
        if enemy_tower_health > 0 and abs(position - enemy_tower_position) <= c.attack_range:
            actions.append(attack_action("enemy_tower"))
        if (enemy_tower_health <= 0 and enemy_crystal_health > 0
                and abs(position - enemy_crystal_position) <= c.attack_range):
            actions.append(attack_action("enemy_crystal"))
        return tuple(actions)

    def step(self, blue_action: FactorizedAction,
             red_action: FactorizedAction) -> dict[str, object]:
        if self.state.terminal:
            raise ValueError("episode already ended")
        if blue_action not in self.legal_actions("blue"):
            raise ValueError("illegal blue action")
        if red_action not in self.legal_actions("red"):
            raise ValueError("illegal red action")
        events: list[str] = []
        self._move("blue", blue_action, events); self._move("red", red_action, events)  # noqa: E702
        self._attack("blue", blue_action, events); self._attack("red", red_action, events)  # noqa: E702
        self.state.tick += 1
        self._finish(events)
        return self._response(events)

    def _move(self, side: Side, action: FactorizedAction, events: list[str]) -> None:
        if action.action_type != "move":
            return
        delta = 1 if action.direction == "forward" else -1
        if side == "red":
            delta *= -1
        key = f"{side}_position"
        position = cast(int, getattr(self.state, key)) + delta
        setattr(self.state, key, position); events.append(f"{side}:move:{position}")  # noqa: E702

    def _attack(self, side: Side, action: FactorizedAction, events: list[str]) -> None:
        if action.action_type != "attack":
            return
        enemy: Side = "red" if side == "blue" else "blue"
        suffix = {
            "enemy_hero": "health",
            "enemy_tower": "tower_health",
            "enemy_crystal": "crystal_health",
        }[action.target]
        key = f"{enemy}_{suffix}"
        remaining = max(0, cast(int, getattr(self.state, key)) - self.config.basic_damage)
        setattr(self.state, key, remaining)
        events.append(f"{side}:attack:{enemy}_{suffix}:{self.config.basic_damage}")
        if suffix == "health" and remaining == 0:
            setattr(self.state, key, self.config.hero_health)
            setattr(self.state, f"{enemy}_position", getattr(self.config, f"{enemy}_start"))
            events.append(f"{side}:knockout:{enemy}")

    def _finish(self, events: list[str]) -> None:
        s = self.state
        if s.blue_crystal_health == 0 and s.red_crystal_health == 0:
            s.terminal, s.outcome = True, "draw_mutual_crystal"
        elif s.red_crystal_health == 0:
            s.terminal, s.outcome = True, "blue_win_crystal_destroyed"
        elif s.blue_crystal_health == 0:
            s.terminal, s.outcome = True, "red_win_crystal_destroyed"
        elif s.tick >= self.config.max_ticks:
            s.terminal, s.outcome = True, "draw_tick_limit"
        if s.terminal:
            events.append(f"terminal:{s.outcome}")

    def _response(self, events: list[str]) -> dict[str, object]:
        return {
            "observation": self.public_state(),
            "legal_actions": {
                side: [action.to_dict() for action in self.legal_actions(side)]
                for side in ("blue", "red")
            },
            "events": events, "terminal": self.state.terminal, "outcome": self.state.outcome,
        }
