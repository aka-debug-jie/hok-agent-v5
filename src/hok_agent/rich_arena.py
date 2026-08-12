from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path
from random import Random
from typing import Final, Literal, cast

Side = Literal["blue", "red"]

RICH_IDENTITY = "pixelarena-rich-1v1-v2"
RICH_RULESET = "abstract-lane-rich-1v1-v2"
BOARD_WIDTH, BOARD_HEIGHT, LANE_Y = 15, 7, (2, 3, 4)
HERO_HP, HERO_RESPAWN_TICKS = 10, 4
TOWER_HP, TOWER_RANGE, TOWER_DAMAGE = 12, 3, 2
CRYSTAL_HP = 16
BASIC_RANGE, BASIC_DAMAGE = 1, 2
SKILL1_RANGE, SKILL1_DAMAGE, SKILL1_COOLDOWN = 2, 0, 4
SKILL2_RANGE, SKILL2_DAMAGE, SKILL2_COOLDOWN = 4, 3, 3
SKILL3_RANGE, SKILL3_DAMAGE, SKILL3_COOLDOWN = 3, 4, 6
MINION_HP, MINION_DAMAGE, MINION_RANGE = 3, 1, 2
MINION_BATCH, MINION_SPAWN_EVERY_TICKS = 2, 6
MAX_TICKS = 96
BLUE_START, RED_START = (2, 3), (12, 3)
BLUE_TOWER, RED_TOWER = (1, 3), (13, 3)
BLUE_CRYSTAL, RED_CRYSTAL = (0, 3), (14, 3)
BLUE_MINION_SPAWN, RED_MINION_SPAWN = (4, 3), (10, 3)
MINION_SPAWN_LANES: Final = (2, 4)

MACROS: Final = ("hold", "move", "attack", "dash", "projectile", "targeted")
ACTION_TYPES: Final = ("wait", "move", "attack", "skill")
DIRECTIONS: Final = (
    "none",
    "north",
    "south",
    "west",
    "east",
    "northwest",
    "northeast",
    "southwest",
    "southeast",
)
TARGETS: Final = ("none", "enemy_hero", "enemy_tower", "enemy_crystal", "enemy_minion")
SKILLS: Final = ("none", "basic", "skill1", "skill2", "skill3")
CANONICAL_ACTION_TEMPLATES: Final = (
    ("hold", "wait", "none", "none", "none"),
    ("move", "move", "none", "{direction}", "none"),
    ("attack", "attack", "enemy_hero", "none", "basic"),
    ("attack", "attack", "enemy_tower", "none", "basic"),
    ("attack", "attack", "enemy_crystal", "none", "basic"),
    ("attack", "attack", "enemy_minion", "none", "basic"),
    ("dash", "skill", "none", "{direction}", "skill1"),
    ("projectile", "skill", "none", "{direction}", "skill2"),
    ("targeted", "skill", "enemy_hero", "none", "skill3"),
)
MACRO_INDEX = {value: index for index, value in enumerate(MACROS)}
ACTION_TYPE_INDEX = {value: index for index, value in enumerate(ACTION_TYPES)}
DIRECTION_INDEX = {value: index for index, value in enumerate(DIRECTIONS)}
TARGET_INDEX = {value: index for index, value in enumerate(TARGETS)}
SKILL_INDEX = {value: index for index, value in enumerate(SKILLS)}
ACTION_FACTOR_SPACE = len(MACROS) * len(ACTION_TYPES) * len(DIRECTIONS) * len(TARGETS) * len(SKILLS)
_VECTOR: Final = {
    "none": (0, 0),
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
    "northwest": (-1, -1),
    "northeast": (1, -1),
    "southwest": (-1, 1),
    "southeast": (1, 1),
}


def _hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def observation_hash(observation: dict[str, object]) -> str:
    return _hash(observation)


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _valid(position: tuple[int, int]) -> bool:
    return 0 <= position[0] < BOARD_WIDTH and position[1] in LANE_Y


def _opposite(side: Side) -> Side:
    return "red" if side == "blue" else "blue"


@dataclass(frozen=True)
class FactorizedAction:
    macro: str
    action_type: str
    target: str = "none"
    direction: str = "none"
    skill: str = "none"
    upgrade: str = "none"
    auxiliary: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> FactorizedAction:
        if set(raw) != {
            "macro",
            "action_type",
            "target",
            "direction",
            "skill",
            "upgrade",
            "auxiliary",
        }:
            raise ValueError("invalid action wire fields")
        return cls(
            str(raw["macro"]),
            str(raw["action_type"]),
            str(raw["target"]),
            str(raw["direction"]),
            str(raw["skill"]),
            str(raw["upgrade"]),
            int(cast(int, raw["auxiliary"])),
        )


def wait_action() -> FactorizedAction:
    return FactorizedAction("hold", "wait")


def move_action(direction: str) -> FactorizedAction:
    return FactorizedAction("move", "move", direction=direction)


def attack_action(target: str) -> FactorizedAction:
    return FactorizedAction("attack", "attack", target=target, skill="basic")


def dash_action(direction: str) -> FactorizedAction:
    return FactorizedAction("dash", "skill", direction=direction, skill="skill1")


def skill2_action(direction: str) -> FactorizedAction:
    return FactorizedAction("projectile", "skill", direction=direction, skill="skill2")


def skill3_action() -> FactorizedAction:
    return FactorizedAction("targeted", "skill", target="enemy_hero", skill="skill3")


def canonical_actions() -> tuple[FactorizedAction, ...]:
    result: list[FactorizedAction] = []
    for macro, kind, target, direction, skill in CANONICAL_ACTION_TEMPLATES:
        directions = DIRECTIONS[1:] if direction == "{direction}" else (direction,)
        result.extend(FactorizedAction(macro, kind, target, item, skill) for item in directions)
    return tuple(result)


def ego_action(action: FactorizedAction, side: Side) -> FactorizedAction:
    if side == "blue" or action.direction == "none":
        return action
    opposite = {
        "north": "south",
        "south": "north",
        "west": "east",
        "east": "west",
        "northwest": "southeast",
        "northeast": "southwest",
        "southwest": "northeast",
        "southeast": "northwest",
    }
    return FactorizedAction(
        action.macro,
        action.action_type,
        action.target,
        opposite[action.direction],
        action.skill,
    )


def action_factor_index(action: FactorizedAction) -> int:
    if action.upgrade != "none" or action.auxiliary != 0:
        raise ValueError("wire requires upgrade=none and auxiliary=0")
    try:
        values = (
            MACRO_INDEX[action.macro],
            ACTION_TYPE_INDEX[action.action_type],
            DIRECTION_INDEX[action.direction],
            TARGET_INDEX[action.target],
            SKILL_INDEX[action.skill],
        )
    except KeyError as exc:
        raise ValueError(f"unknown action factor: {exc.args[0]}") from exc
    macro, kind, direction, target, skill = values
    return (
        ((macro * len(ACTION_TYPES) + kind) * len(DIRECTIONS) + direction) * len(TARGETS) + target
    ) * len(SKILLS) + skill


def action_from_factor_index(index: int) -> FactorizedAction:
    if not 0 <= index < ACTION_FACTOR_SPACE:
        raise ValueError(f"action index out of range: {index}")
    values: list[int] = []
    for width in (len(SKILLS), len(TARGETS), len(DIRECTIONS), len(ACTION_TYPES)):
        values.append(index % width)
        index //= width
    skill, target, direction, kind = values
    return FactorizedAction(
        MACROS[index], ACTION_TYPES[kind], TARGETS[target], DIRECTIONS[direction], SKILLS[skill]
    )


@dataclass(frozen=True)
class ArenaConfig:
    identity: str = RICH_IDENTITY
    ruleset: str = RICH_RULESET
    width: int = BOARD_WIDTH
    height: int = BOARD_HEIGHT
    hero_health: int = HERO_HP
    hero_respawn_ticks: int = HERO_RESPAWN_TICKS
    tower_health: int = TOWER_HP
    tower_range: int = TOWER_RANGE
    tower_damage: int = TOWER_DAMAGE
    crystal_health: int = CRYSTAL_HP
    basic_range: int = BASIC_RANGE
    basic_damage: int = BASIC_DAMAGE
    skill1_range: int = SKILL1_RANGE
    skill1_damage: int = SKILL1_DAMAGE
    skill1_cooldown: int = SKILL1_COOLDOWN
    skill2_range: int = SKILL2_RANGE
    skill2_damage: int = SKILL2_DAMAGE
    skill2_cooldown: int = SKILL2_COOLDOWN
    skill3_range: int = SKILL3_RANGE
    skill3_damage: int = SKILL3_DAMAGE
    skill3_cooldown: int = SKILL3_COOLDOWN
    minion_health: int = MINION_HP
    minion_damage: int = MINION_DAMAGE
    minion_range: int = MINION_RANGE
    minion_spawn_every_ticks: int = MINION_SPAWN_EVERY_TICKS
    minion_batch: int = MINION_BATCH
    minion_movement: str = "simultaneous-intents-same-target-cancel-v1"
    max_ticks: int = MAX_TICKS
    blue_start: tuple[int, int] = BLUE_START
    red_start: tuple[int, int] = RED_START
    blue_tower: tuple[int, int] = BLUE_TOWER
    red_tower: tuple[int, int] = RED_TOWER
    blue_crystal: tuple[int, int] = BLUE_CRYSTAL
    red_crystal: tuple[int, int] = RED_CRYSTAL
    blue_minion_spawn: tuple[int, int] = BLUE_MINION_SPAWN
    red_minion_spawn: tuple[int, int] = RED_MINION_SPAWN

    @property
    def digest(self) -> str:
        return _hash(
            {
                **asdict(self),
                "lane_y": LANE_Y,
                "templates": CANONICAL_ACTION_TEMPLATES,
                "vocab": [MACROS, ACTION_TYPES, DIRECTIONS, TARGETS, SKILLS],
            }
        )


@dataclass
class _Hero:
    x: int
    y: int
    health: int
    respawn: int = 0
    cooldowns: dict[str, int] = field(
        default_factory=lambda: {"skill1": 0, "skill2": 0, "skill3": 0}
    )


@dataclass
class Minion:
    x: int
    y: int
    health: int
    side: Side


@dataclass
class _State:
    tick: int
    blue: _Hero
    red: _Hero
    blue_tower_health: int
    red_tower_health: int
    blue_crystal_health: int
    red_crystal_health: int
    blue_minions: list[Minion]
    red_minions: list[Minion]
    terminal: bool = False
    outcome: str = "ongoing"


class RichNullPolicy:
    def select(
        self,
        side: Side,
        legal: tuple[FactorizedAction, ...],
        tick: int | None = None,
        observation: dict[str, object] | None = None,
    ) -> FactorizedAction:
        del side, tick, observation
        return next(action for action in legal if action == wait_action())


class RichRandomPolicy:
    def __init__(self, seed: int, side: Side) -> None:
        del side
        self._rng = Random(seed)

    def select(
        self,
        side: Side,
        legal: tuple[FactorizedAction, ...],
        tick: int | None = None,
        observation: dict[str, object] | None = None,
    ) -> FactorizedAction:
        del tick, observation
        ordered = sorted(legal, key=lambda action: action_factor_index(ego_action(action, side)))
        return self._rng.choice(ordered)


class RichTeacherPolicy:
    """Deterministic causal policy: it receives only side, tick and the current legal domain."""

    def __init__(self, start_tick: int = 0) -> None:
        self._tick = start_tick

    def select(
        self,
        side: Side,
        legal: tuple[FactorizedAction, ...],
        tick: int | None = None,
        observation: dict[str, object] | None = None,
    ) -> FactorizedAction:
        if tick is None:
            tick, self._tick = self._tick, self._tick + 1
        forward = "east" if side == "blue" else "west"
        for direct in (
            attack_action("enemy_crystal"),
            attack_action("enemy_tower"),
            attack_action("enemy_hero"),
        ):
            if direct in legal:
                return direct
        if observation is not None:
            own = cast(dict[str, int], observation["self_position"])
            tower_x = 13 if side == "blue" else 1
            enemy_minions = cast(
                list[dict[str, int]], observation[f"{_opposite(side)}_minions"]
            )
            if abs(own["x"] - tower_x) > 4:
                for item in sorted(
                    enemy_minions,
                    key=lambda value: (
                        abs(value["x"] - own["x"]) + abs(value["y"] - own["y"]),
                        value["x"] if side == "blue" else BOARD_WIDTH - 1 - value["x"],
                        value["y"] if side == "blue" else BOARD_HEIGHT - 1 - value["y"],
                    ),
                ):
                    dx, dy = item["x"] - own["x"], item["y"] - own["y"]
                    if max(abs(dx), abs(dy)) > 4 or dx == 0 and dy == 0:
                        continue
                    x_name = "east" if dx > 0 else "west" if dx < 0 else ""
                    y_name = "south" if dy > 0 else "north" if dy < 0 else ""
                    projectile = skill2_action(f"{y_name}{x_name}")
                    if projectile in legal:
                        return projectile
        if observation is not None and cast(int, observation["enemy_tower_health"]) > 0:
            own = cast(dict[str, int], observation["self_position"])
            opponent = cast(dict[str, int], observation["opponent_position"])
            tower_x = 13 if side == "blue" else 1
            own_minions = cast(list[dict[str, int]], observation[f"{side}_minions"])
            retreat = dash_action("west" if side == "blue" else "east")
            tanking_soon = any(
                abs(item["x"] - tower_x) + abs(item["y"] - 3) <= 4 for item in own_minions
            )
            advance = move_action(forward)
            tower_distance = abs(own["x"] - tower_x)
            defender_far = abs(opponent["x"] - tower_x) > 3
            if (
                cast(int, observation["enemy_tower_health"]) <= 4
                and (tanking_soon or defender_far)
                and tower_distance > 1
                and advance in legal
            ):
                return advance
            if tower_distance <= 4 and not tanking_soon:
                if tower_distance <= TOWER_RANGE and retreat in legal:
                    return retreat
                targeted = skill3_action()
                if targeted in legal:
                    return targeted
                projectile = skill2_action(forward)
                if projectile in legal:
                    return projectile
                minion_attack = attack_action("enemy_minion")
                if minion_attack in legal:
                    return minion_attack
                return wait_action()
        if observation is not None and cast(int, observation["enemy_tower_health"]) == 0:
            for objective in (
                attack_action("enemy_crystal"),
                dash_action(forward),
                move_action(forward),
            ):
                if objective in legal:
                    return objective
        priorities: tuple[Callable[[FactorizedAction], bool], ...] = (
            lambda a: a == attack_action("enemy_crystal"),
            lambda a: a == attack_action("enemy_tower"),
            lambda a: a == attack_action("enemy_minion"),
            lambda a: a == skill3_action(),
            lambda a: a == skill2_action(forward),
            lambda a: a == attack_action("enemy_hero"),
            lambda a: a == dash_action(forward),
            lambda a: a == move_action(forward),
        )
        for predicate in priorities:
            match = next((action for action in legal if predicate(action)), None)
            if match is not None:
                return match
        return wait_action()

def make_rich_policy(
    name: str, seed: int, side: Side
) -> RichNullPolicy | RichRandomPolicy | RichTeacherPolicy:
    if name == "null":
        return RichNullPolicy()
    if name == "random":
        return RichRandomPolicy(seed, side)
    if name == "teacher":
        return RichTeacherPolicy()
    raise ValueError(f"unknown policy: {name}")


Damage = tuple[str, Side, int]


class RichPixelArena:
    def __init__(self, config: ArenaConfig | None = None) -> None:
        self.config = config or ArenaConfig()
        self.seed = 0
        self.state = self._new_state()

    def _new_state(self) -> _State:
        c = self.config
        return _State(
            0,
            _Hero(*c.blue_start, c.hero_health),
            _Hero(*c.red_start, c.hero_health),
            c.tower_health,
            c.tower_health,
            c.crystal_health,
            c.crystal_health,
            [],
            [],
        )

    def reset(self, seed: int = 0) -> dict[str, object]:
        self.seed, self.state = seed, self._new_state()
        return self._response([])

    def health(self) -> dict[str, object]:
        return {
            "identity": self.config.identity,
            "ruleset": self.config.ruleset,
            "config_hash": self.config.digest,
            "claim_scope": "pixelarena_engineering",
            "hok_capability_claim": False,
            "gamecore_equivalence_claim": False,
            "capabilities": {"network": False, "device": False, "external_client": False},
        }

    def _hero(self, side: Side) -> _Hero:
        return self.state.blue if side == "blue" else self.state.red

    def _pos(self, side: Side) -> tuple[int, int]:
        hero = self._hero(side)
        return hero.x, hero.y

    def _tower_pos(self, side: Side) -> tuple[int, int]:
        return self.config.blue_tower if side == "blue" else self.config.red_tower

    def _crystal_pos(self, side: Side) -> tuple[int, int]:
        return self.config.blue_crystal if side == "blue" else self.config.red_crystal

    def _tower_hp(self, side: Side) -> int:
        return self.state.blue_tower_health if side == "blue" else self.state.red_tower_health

    def _crystal_hp(self, side: Side) -> int:
        return self.state.blue_crystal_health if side == "blue" else self.state.red_crystal_health

    def _set_hp(self, kind: str, side: Side, value: int) -> None:
        name = f"{side}_{kind}_health"
        setattr(self.state, name, max(0, value))

    def _minions(self, side: Side) -> list[Minion]:
        return self.state.blue_minions if side == "blue" else self.state.red_minions

    def observe(self, side: Side) -> dict[str, object]:
        enemy = _opposite(side)
        own, other = self._hero(side), self._hero(enemy)
        return {
            "tick": self.state.tick,
            "max_ticks": self.config.max_ticks,
            "side": side,
            "self_position": {"x": own.x, "y": own.y},
            "opponent_position": {"x": other.x, "y": other.y},
            "self_health": own.health,
            "opponent_health": other.health,
            "self_tower_health": self._tower_hp(side),
            "enemy_tower_health": self._tower_hp(enemy),
            "self_crystal_health": self._crystal_hp(side),
            "enemy_crystal_health": self._crystal_hp(enemy),
            "self_respawn": own.respawn,
            "opponent_respawn": other.respawn,
            "self_cooldowns": dict(own.cooldowns),
            "blue_minions": [
                {"x": item.x, "y": item.y, "health": item.health}
                for item in self.state.blue_minions
            ],
            "red_minions": [
                {"x": item.x, "y": item.y, "health": item.health} for item in self.state.red_minions
            ],
            "terminal": self.state.terminal,
            "outcome": self.state.outcome,
        }

    def public_state(self) -> dict[str, object]:
        return {"blue": self.observe("blue"), "red": self.observe("red")}

    def legal_actions(self, side: Side) -> tuple[FactorizedAction, ...]:
        if self.state.terminal:
            return ()
        hero = self._hero(side)
        if hero.health <= 0 or hero.respawn:
            return (wait_action(),)
        enemy, origin = _opposite(side), self._pos(side)
        actions = [wait_action()]
        for direction in DIRECTIONS[1:]:
            dx, dy = _VECTOR[direction]
            if _valid((origin[0] + dx, origin[1] + dy)):
                actions.append(move_action(direction))
            if hero.cooldowns["skill1"] == 0 and _valid((origin[0] + 2 * dx, origin[1] + 2 * dy)):
                actions.append(dash_action(direction))
            if hero.cooldowns["skill2"] == 0:
                actions.append(skill2_action(direction))
        if (
            hero.cooldowns["skill3"] == 0
            and self._hero(enemy).health > 0
            and _dist(origin, self._pos(enemy)) <= self.config.skill3_range
        ):
            actions.append(skill3_action())
        if (
            _dist(origin, self._pos(enemy)) <= self.config.basic_range
            and self._hero(enemy).health > 0
        ):
            actions.append(attack_action("enemy_hero"))
        if (
            self._tower_hp(enemy) > 0
            and _dist(origin, self._tower_pos(enemy)) <= self.config.basic_range
        ):
            actions.append(attack_action("enemy_tower"))
        if (
            self._tower_hp(enemy) == 0
            and _dist(origin, self._crystal_pos(enemy)) <= self.config.basic_range
        ):
            actions.append(attack_action("enemy_crystal"))
        if any(_dist(origin, (m.x, m.y)) <= self.config.basic_range for m in self._minions(enemy)):
            actions.append(attack_action("enemy_minion"))
        return tuple(actions)

    def _spawn(self, events: list[str]) -> None:
        if (self.state.tick + 1) % self.config.minion_spawn_every_ticks:
            return
        for side, point in (
            ("blue", self.config.blue_minion_spawn),
            ("red", self.config.red_minion_spawn),
        ):
            for y in MINION_SPAWN_LANES:
                self._minions(cast(Side, side)).append(
                    Minion(point[0], y, self.config.minion_health, cast(Side, side))
                )
            events.append(f"spawn:{side}:2")

    def step(
        self, blue_action: FactorizedAction, red_action: FactorizedAction
    ) -> dict[str, object]:
        if self.state.terminal:
            raise ValueError("episode already ended")
        before = deepcopy(self.state)
        events: list[str] = []
        self._spawn(events)
        if blue_action not in self.legal_actions("blue"):
            self.state = before
            raise ValueError("illegal blue action")
        if red_action not in self.legal_actions("red"):
            self.state = before
            raise ValueError("illegal red action")
        old_respawn = {side: self._hero(side).respawn for side in ("blue", "red")}
        self._move_heroes(blue_action, red_action, events)
        self._move_minions()
        damage: defaultdict[Damage, int] = defaultdict(int)
        self._hero_combat("blue", blue_action, damage, events)
        self._hero_combat("red", red_action, damage, events)
        self._minion_combat(damage, events)
        self._tower_combat(damage, events)
        knocked_out = self._commit(damage, events)
        self._cooldowns("blue", blue_action)
        self._cooldowns("red", red_action)
        self._finish_tick(old_respawn, knocked_out, events)
        return self._response(events)

    def _destination(self, side: Side, action: FactorizedAction) -> tuple[int, int] | None:
        stride = 2 if action.skill == "skill1" else 1
        if action.action_type != "move" and action.skill != "skill1":
            return None
        dx, dy = _VECTOR[action.direction]
        point = (self._pos(side)[0] + stride * dx, self._pos(side)[1] + stride * dy)
        return point if _valid(point) else None

    def _move_heroes(
        self, blue: FactorizedAction, red: FactorizedAction, events: list[str]
    ) -> None:
        b0, r0 = self._pos("blue"), self._pos("red")
        bd, rd = self._destination("blue", blue), self._destination("red", red)
        if bd == r0 and rd == b0:
            self.state.blue.x, self.state.blue.y = r0
            self.state.red.x, self.state.red.y = b0
            events.append("move:swap")
        elif bd is not None and bd == rd:
            events.append("move:cancel")
        else:
            if bd is not None and (bd != r0 or self.state.red.health == 0):
                self.state.blue.x, self.state.blue.y = bd
            if rd is not None and (rd != b0 or self.state.blue.health == 0):
                self.state.red.x, self.state.red.y = rd

    def _move_minions(self) -> None:
        intents: list[tuple[Minion, Side, tuple[int, int]]] = []
        destination_sides: defaultdict[tuple[int, int], set[Side]] = defaultdict(set)
        for side in ("blue", "red"):
            owner: Side = "blue" if side == "blue" else "red"
            direction = 1 if side == "blue" else -1
            enemy = _opposite(owner)
            occupied = {(m.x, m.y) for m in self._minions(enemy)} | {self._pos(enemy)}
            for minion in self._minions(owner):
                point = (minion.x + direction, minion.y)
                if _valid(point) and point not in occupied:
                    intents.append((minion, owner, point))
                    destination_sides[point].add(owner)
        for minion, _side, point in intents:
            if len(destination_sides[point]) == 1:
                minion.x, minion.y = point

    def _line_target(self, side: Side, direction: str) -> Damage | None:
        enemy, (dx, dy), origin = _opposite(side), _VECTOR[direction], self._pos(side)
        for distance in range(1, self.config.skill2_range + 1):
            point = (origin[0] + distance * dx, origin[1] + distance * dy)
            if not _valid(point):
                break
            for index, minion in enumerate(self._minions(enemy)):
                if (minion.x, minion.y) == point:
                    return "minion", enemy, index
            if self._hero(enemy).health > 0 and self._pos(enemy) == point:
                return "hero", enemy, -1
            if self._tower_hp(enemy) > 0 and self._tower_pos(enemy) == point:
                return "tower", enemy, -1
            if self._tower_hp(enemy) == 0 and self._crystal_pos(enemy) == point:
                return "crystal", enemy, -1
        return None

    def _minion_target_key(
        self, owner: Side, origin: tuple[int, int], index: int
    ) -> tuple[int, int, int, int]:
        minion = self._minions(_opposite(owner))[index]
        return (
            _dist(origin, (minion.x, minion.y)),
            minion.x if owner == "blue" else BOARD_WIDTH - 1 - minion.x,
            minion.y if owner == "blue" else BOARD_HEIGHT - 1 - minion.y,
            index,
        )

    def _hero_combat(
        self,
        side: Side,
        action: FactorizedAction,
        damage: defaultdict[Damage, int],
        events: list[str],
    ) -> None:
        enemy, origin = _opposite(side), self._pos(side)
        target: Damage | None = None
        amount = 0
        if action.action_type == "attack":
            amount = self.config.basic_damage
            if action.target == "enemy_hero":
                target = ("hero", enemy, -1)
            elif action.target == "enemy_tower":
                target = ("tower", enemy, -1)
            elif action.target == "enemy_crystal":
                target = ("crystal", enemy, -1)
            else:
                candidates = range(len(self._minions(enemy)))
                target = (
                    "minion",
                    enemy,
                    min(candidates, key=lambda index: self._minion_target_key(side, origin, index)),
                )
        elif action.skill == "skill2":
            target, amount = self._line_target(side, action.direction), self.config.skill2_damage
        elif action.skill == "skill3":
            target, amount = ("hero", enemy, -1), self.config.skill3_damage
        if target is not None:
            damage[target] += amount
            events.append(f"attack:hero:{target[0]}:{side}")

    def _minion_combat(self, damage: defaultdict[Damage, int], events: list[str]) -> None:
        for side in ("blue", "red"):
            owner: Side = "blue" if side == "blue" else "red"
            enemy = _opposite(owner)
            for index, minion in enumerate(self._minions(owner)):
                origin = (minion.x, minion.y)
                nearby = [
                    i
                    for i, m in enumerate(self._minions(enemy))
                    if _dist(origin, (m.x, m.y)) <= self.config.minion_range
                ]
                if nearby:
                    target: Damage = (
                        "minion",
                        enemy,
                        min(
                            nearby,
                            key=lambda target_index: self._minion_target_key(
                                owner, origin, target_index
                            ),
                        ),
                    )
                elif self._hero(enemy).health > 0 and _dist(origin, self._pos(enemy)) <= 1:
                    target = ("hero", enemy, -1)
                elif self._tower_hp(enemy) > 0 and _dist(origin, self._tower_pos(enemy)) <= 1:
                    target = ("tower", enemy, -1)
                elif self._tower_hp(enemy) == 0 and _dist(origin, self._crystal_pos(enemy)) <= 1:
                    target = ("crystal", enemy, -1)
                else:
                    continue
                damage[target] += self.config.minion_damage
                events.append(f"attack:minion:{index}:{target[0]}:{side}")

    def _tower_combat(self, damage: defaultdict[Damage, int], events: list[str]) -> None:
        for side in ("blue", "red"):
            owner: Side = "blue" if side == "blue" else "red"
            enemy = _opposite(owner)
            if self._tower_hp(owner) <= 0:
                continue
            origin = self._tower_pos(owner)
            nearby = [
                i
                for i, m in enumerate(self._minions(enemy))
                if _dist(origin, (m.x, m.y)) <= self.config.tower_range
            ]
            if nearby:
                target = (
                    "minion",
                    enemy,
                    min(
                        nearby,
                        key=lambda target_index: self._minion_target_key(
                            owner, origin, target_index
                        ),
                    ),
                )
            elif (
                self._hero(enemy).health > 0
                and _dist(origin, self._pos(enemy)) <= self.config.tower_range
            ):
                target = ("hero", enemy, -1)
            else:
                continue
            damage[target] += self.config.tower_damage
            events.append(f"attack:tower:{target[0]}:{side}")

    def _commit(self, damage: defaultdict[Damage, int], events: list[str]) -> set[Side]:
        knocked_out: set[Side] = set()
        for (kind, side, index), amount in sorted(damage.items()):
            if kind == "hero":
                hero = self._hero(side)
                hero.health = max(0, hero.health - amount)
                if hero.health == 0 and hero.respawn == 0:
                    hero.respawn = self.config.hero_respawn_ticks
                    knocked_out.add(side)
                    events.append(f"ko:{side}")
            elif kind in {"tower", "crystal"}:
                current = self._tower_hp(side) if kind == "tower" else self._crystal_hp(side)
                self._set_hp(kind, side, current - amount)
            else:
                minions = self._minions(side)
                if 0 <= index < len(minions):
                    minions[index].health = max(0, minions[index].health - amount)
        self.state.blue_minions = [m for m in self.state.blue_minions if m.health]
        self.state.red_minions = [m for m in self.state.red_minions if m.health]
        return knocked_out

    def _cooldowns(self, side: Side, action: FactorizedAction) -> None:
        hero = self._hero(side)
        for skill in hero.cooldowns:
            hero.cooldowns[skill] = max(0, hero.cooldowns[skill] - 1)
        if action.skill in hero.cooldowns:
            hero.cooldowns[action.skill] = getattr(self.config, f"{action.skill}_cooldown")

    def _finish_tick(self, old: dict[str, int], knocked_out: set[Side], events: list[str]) -> None:
        for raw_side in ("blue", "red"):
            side: Side = "blue" if raw_side == "blue" else "red"
            hero = self._hero(side)
            if old[side] > 0 and side not in knocked_out:
                hero.respawn = max(0, hero.respawn - 1)
                if hero.respawn == 0:
                    start = self.config.blue_start if side == "blue" else self.config.red_start
                    hero.x, hero.y, hero.health = start[0], start[1], self.config.hero_health
                    events.append(f"respawn:{side}")
        self.state.tick += 1
        blue_dead, red_dead = (
            self.state.blue_crystal_health <= 0,
            self.state.red_crystal_health <= 0,
        )
        if blue_dead or red_dead:
            self.state.terminal = True
            self.state.outcome = (
                "draw_mutual_crystal"
                if blue_dead and red_dead
                else "red_win_crystal_destroyed"
                if blue_dead
                else "blue_win_crystal_destroyed"
            )
        elif self.state.tick >= self.config.max_ticks:
            self.state.terminal, self.state.outcome = True, "draw_tick_limit"

    def _response(self, events: list[str]) -> dict[str, object]:
        return {
            "observation": self.public_state(),
            "legal_actions": {
                side: [a.to_dict() for a in self.legal_actions(side)] for side in ("blue", "red")
            },
            "events": events,
            "terminal": self.state.terminal,
            "outcome": self.state.outcome,
        }


class ReplayError(RuntimeError):
    pass


_HEADER = {
    "kind",
    "identity",
    "config_hash",
    "seed",
    "blue_policy",
    "red_policy",
    "initial_state_hash",
}
_ROW = {
    "tick",
    "blue_action",
    "red_action",
    "action_hash",
    "events",
    "event_hash",
    "state_hash",
    "terminal",
    "outcome",
}


def record_rich_trace(
    path: Path, blue_policy: str, red_policy: str, seed: int
) -> dict[str, object]:
    arena = RichPixelArena()
    response = arena.reset(seed)
    blue, red = (
        make_rich_policy(blue_policy, seed, "blue"),
        make_rich_policy(red_policy, seed, "red"),
    )
    documents: list[dict[str, object]] = [
        {
            "kind": "pixelarena_rich_trace_v2",
            "identity": RICH_IDENTITY,
            "config_hash": arena.config.digest,
            "seed": seed,
            "blue_policy": blue_policy,
            "red_policy": red_policy,
            "initial_state_hash": observation_hash(
                cast(dict[str, object], response["observation"])
            ),
        }
    ]
    while not bool(response["terminal"]):
        b, r = (
            blue.select(
                "blue", arena.legal_actions("blue"), arena.state.tick, arena.observe("blue")
            ),
            red.select("red", arena.legal_actions("red"), arena.state.tick, arena.observe("red")),
        )
        response = arena.step(b, r)
        actions = {"blue": b.to_dict(), "red": r.to_dict()}
        events = cast(list[str], response["events"])
        documents.append(
            {
                "tick": arena.state.tick,
                "blue_action": actions["blue"],
                "red_action": actions["red"],
                "action_hash": _hash(actions),
                "events": events,
                "event_hash": _hash(events),
                "state_hash": observation_hash(arena.public_state()),
                "terminal": arena.state.terminal,
                "outcome": arena.state.outcome,
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in documents),
        encoding="utf-8",
    )
    return {"path": str(path), "ticks": arena.state.tick, "outcome": arena.state.outcome}


def _verify_local(path: str) -> dict[str, object]:
    try:
        documents = [
            cast(dict[str, object], json.loads(line))
            for line in Path(path).read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError("invalid trace encoding") from exc
    if len(documents) < 2 or set(documents[0]) != _HEADER:
        raise ReplayError("invalid trace header")
    header = documents[0]
    if (
        header["kind"] != "pixelarena_rich_trace_v2"
        or header["identity"] != RICH_IDENTITY
        or type(header["seed"]) is not int
    ):
        raise ReplayError("trace identity or seed mismatch")
    arena = RichPixelArena()
    arena.reset(header["seed"])
    if header["config_hash"] != arena.config.digest or header[
        "initial_state_hash"
    ] != observation_hash(arena.public_state()):
        raise ReplayError("config or initial state mismatch")
    for tick, row in enumerate(documents[1:], 1):
        if set(row) != _ROW or row["tick"] != tick:
            raise ReplayError(f"invalid row at tick {tick}")
        try:
            blue = FactorizedAction.from_dict(cast(dict[str, object], row["blue_action"]))
            red = FactorizedAction.from_dict(cast(dict[str, object], row["red_action"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayError(f"invalid action at tick {tick}") from exc
        actions = {"blue": blue.to_dict(), "red": red.to_dict()}
        if row["action_hash"] != _hash(actions) or row["event_hash"] != _hash(row["events"]):
            raise ReplayError(f"action or event hash mismatch at tick {tick}")
        try:
            response = arena.step(blue, red)
        except ValueError as exc:
            raise ReplayError(f"illegal replay action at tick {tick}") from exc
        if (
            row["events"] != response["events"]
            or row["state_hash"] != observation_hash(arena.public_state())
            or row["terminal"] != arena.state.terminal
            or row["outcome"] != arena.state.outcome
        ):
            raise ReplayError(f"replay mismatch at tick {tick}")
    if not arena.state.terminal:
        raise ReplayError("trace does not terminate")
    return {
        "verified": True,
        "ticks": arena.state.tick,
        "outcome": arena.state.outcome,
        "terminal": True,
        "process": "spawn",
    }


def _verify_worker(path: str, connection: Connection) -> None:
    try:
        connection.send((True, _verify_local(path)))
    except Exception as exc:
        connection.send((False, str(exc)))
    finally:
        connection.close()


def verify_rich_trace(path: Path) -> dict[str, object]:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_verify_worker, args=(str(path), sender))
    process.start()
    sender.close()
    if not receiver.poll(30):
        process.terminate()
        process.join()
        raise ReplayError("spawned replay timed out")
    ok, payload = cast(tuple[bool, object], receiver.recv())
    process.join()
    if process.exitcode != 0 or not ok:
        raise ReplayError(str(payload))
    return cast(dict[str, object], payload)
