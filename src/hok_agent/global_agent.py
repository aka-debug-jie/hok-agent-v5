from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from hok_agent.rich_arena import (
    FactorizedAction,
    RichPixelArena,
    RichRandomPolicy,
    RichTeacherPolicy,
    Side,
    attack_action,
    dash_action,
    move_action,
    skill2_action,
    skill3_action,
    wait_action,
)


class MacroIntent(StrEnum):
    FARM_LANE = "FARM_LANE"
    FARM_JUNGLE = "FARM_JUNGLE"
    CLEAR_WAVE = "CLEAR_WAVE"
    PUSH_STRUCTURE = "PUSH_STRUCTURE"
    DEFEND = "DEFEND"
    CONTEST_OBJECTIVE = "CONTEST_OBJECTIVE"
    ENGAGE = "ENGAGE"
    DISENGAGE = "DISENGAGE"
    RECALL = "RECALL"


class TargetZone(StrEnum):
    OWN_BASE = "OWN_BASE"
    TOP_LANE = "TOP_LANE"
    MID_LANE = "MID_LANE"
    BOTTOM_LANE = "BOTTOM_LANE"
    OWN_JUNGLE = "OWN_JUNGLE"
    TOP_RIVER_OBJECTIVE = "TOP_RIVER_OBJECTIVE"
    BOTTOM_RIVER_OBJECTIVE = "BOTTOM_RIVER_OBJECTIVE"
    ENEMY_JUNGLE = "ENEMY_JUNGLE"
    ENEMY_BASE = "ENEMY_BASE"
    HOLD_CURRENT_ZONE = "HOLD_CURRENT_ZONE"


class FailureCode(StrEnum):
    NAV_STUCK = "NAV_STUCK"
    NO_WAVE_PROGRESS = "NO_WAVE_PROGRESS"
    COMBAT_LOOP = "COMBAT_LOOP"
    DEATH_RECOVERY_FAILED = "DEATH_RECOVERY_FAILED"
    NO_TOWER_DAMAGE = "NO_TOWER_DAMAGE"
    CRYSTAL_NOT_REACHED = "CRYSTAL_NOT_REACHED"
    SAFETY_STOP = "SAFETY_STOP"
    TIMEOUT = "TIMEOUT"


ENABLED_INTENTS = (
    MacroIntent.FARM_LANE,
    MacroIntent.PUSH_STRUCTURE,
    MacroIntent.ENGAGE,
    MacroIntent.DISENGAGE,
    MacroIntent.RECALL,
)
ENABLED_ZONES = (
    TargetZone.OWN_BASE,
    TargetZone.MID_LANE,
    TargetZone.ENEMY_BASE,
    TargetZone.HOLD_CURRENT_ZONE,
)
MID_WAYPOINTS = (
    "OWN_BASE",
    "MID_ENTRY",
    "MID_CENTER",
    "ENEMY_MID_TOWER",
    "ENEMY_HIGH_GROUND",
    "ENEMY_BASE",
)
GLOBAL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/global_agent_v1.json"


def load_global_config(path: Path = GLOBAL_CONFIG_PATH) -> tuple[dict[str, object], str]:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if (
        raw.get("schema_version") != "hok-agent-global-agent-config-v1"
        or raw.get("enabled_intents") != [value.value for value in ENABLED_INTENTS]
        or raw.get("enabled_target_zones") != [value.value for value in ENABLED_ZONES]
        or raw.get("middle_waypoints") != list(MID_WAYPOINTS)
        or raw.get("holdout_seeds") != [4000, 4019]
        or raw.get("adapter_terminal_promotion_rule")
        != "must_equal_or_exceed_baseline"
        or raw.get("video_test_allowed") is not False
        or raw.get("mobile_capture_allowed") is not False
        or raw.get("device_input_allowed") is not False
    ):
        raise ValueError("Global Agent config contract mismatch")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return raw, hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class MacroCommand:
    intent: MacroIntent
    target_zone: TargetZone
    confidence: float = 1.0
    valid_for_ms: int = 1_500

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent.value,
            "target_zone": self.target_zone.value,
            "confidence": self.confidence,
            "valid_for_ms": self.valid_for_ms,
        }


@dataclass(frozen=True)
class TeacherDecision:
    command: MacroCommand
    action: FactorizedAction
    scene_id: str


@dataclass(frozen=True)
class TraceRow:
    tick: int
    observation: dict[str, object]
    command: MacroCommand
    action: FactorizedAction
    scene_id: str
    student_visited: bool = False
    teacher_fallback: bool = False


@dataclass(frozen=True)
class EpisodeReport:
    seed: int
    ticks: int
    outcome: str
    non_timeout_terminal: bool
    crystal_destroyed: bool
    manual_interventions: int
    safety_violations: int
    invalid_actions: int
    tower_damage: int
    tower_progress: bool
    deaths: int
    death_recovery_opportunities: int
    death_recoveries: int
    stuck_ticks: int
    active_ticks: int
    stuck_time_ratio: float
    failure_code: str | None
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GlobalRuleTeacher:
    """Map the existing deterministic low-level teacher to observable macro labels."""

    def __init__(self) -> None:
        self._policy = RichTeacherPolicy()

    def decide(
        self,
        side: Side,
        legal: tuple[FactorizedAction, ...],
        observation: dict[str, object],
    ) -> TeacherDecision:
        own = cast(dict[str, int], observation["self_position"])
        oriented_x = own["x"] if side == "blue" else 14 - own["x"]
        if int(cast(int, observation["self_health"])) <= 2:
            if oriented_x > 2:
                backward = move_action("west" if side == "blue" else "east")
                action = backward if backward in legal else wait_action()
                return TeacherDecision(
                    MacroCommand(MacroIntent.DISENGAGE, TargetZone.OWN_BASE),
                    action,
                    "RETURN_DEFEND",
                )
            return TeacherDecision(
                MacroCommand(MacroIntent.RECALL, TargetZone.OWN_BASE),
                wait_action(),
                "RETURN_DEFEND",
            )
        action = self._policy.select(
            side, legal, int(cast(int, observation["tick"])), observation
        )
        intent, zone, scene = self._label(action, observation)
        return TeacherDecision(MacroCommand(intent, zone), action, scene)

    @staticmethod
    def _label(
        action: FactorizedAction, observation: dict[str, object]
    ) -> tuple[MacroIntent, TargetZone, str]:
        if action.target in {"enemy_tower", "enemy_crystal"}:
            return MacroIntent.PUSH_STRUCTURE, TargetZone.ENEMY_BASE, "PUSH_STRUCTURE"
        if action.target == "enemy_hero" or action.skill in {"skill2", "skill3"}:
            return MacroIntent.ENGAGE, TargetZone.HOLD_CURRENT_ZONE, "COMBAT"
        if action.target == "enemy_minion":
            return MacroIntent.FARM_LANE, TargetZone.MID_LANE, "LANE_FARM"
        own = cast(dict[str, int], observation["self_position"])
        enemy_tower = int(cast(int, observation["enemy_tower_health"]))
        if action.action_type == "move" or action.skill == "skill1":
            if enemy_tower == 0 or own["x"] >= 11:
                return MacroIntent.PUSH_STRUCTURE, TargetZone.ENEMY_BASE, "PUSH_STRUCTURE"
            return MacroIntent.FARM_LANE, TargetZone.MID_LANE, "NAVIGATION"
        return MacroIntent.FARM_LANE, TargetZone.HOLD_CURRENT_ZONE, "LANE_FARM"


class TargetZoneNavigator:
    """Convert a semantic zone into one legal eight-direction waypoint step."""

    _BLUE_WAYPOINT_X = (2, 5, 7, 12, 13, 14)

    def action(
        self,
        side: Side,
        zone: TargetZone,
        observation: dict[str, object],
        legal: tuple[FactorizedAction, ...],
    ) -> FactorizedAction:
        own = cast(dict[str, int], observation["self_position"])
        oriented_x = own["x"] if side == "blue" else 14 - own["x"]
        if zone == TargetZone.HOLD_CURRENT_ZONE:
            return wait_action()
        if zone == TargetZone.OWN_BASE:
            target_x = self._BLUE_WAYPOINT_X[0]
        elif zone == TargetZone.MID_LANE:
            target_x = next((x for x in self._BLUE_WAYPOINT_X[1:4] if x > oriented_x), 12)
        else:
            target_x = next((x for x in self._BLUE_WAYPOINT_X[3:] if x > oriented_x), 14)
        if oriented_x == target_x:
            return wait_action()
        forward = "east" if side == "blue" else "west"
        backward = "west" if side == "blue" else "east"
        candidate = move_action(forward if oriented_x < target_x else backward)
        return candidate if candidate in legal else wait_action()


class GlobalCommandRouter:
    def __init__(self) -> None:
        self._navigator = TargetZoneNavigator()

    def action(
        self,
        side: Side,
        command: MacroCommand,
        observation: dict[str, object],
        legal: tuple[FactorizedAction, ...],
    ) -> FactorizedAction:
        if command.intent in {MacroIntent.RECALL, MacroIntent.DISENGAGE}:
            return self._navigator.action(side, TargetZone.OWN_BASE, observation, legal)
        priorities: tuple[FactorizedAction, ...]
        if command.intent == MacroIntent.PUSH_STRUCTURE:
            forward = "east" if side == "blue" else "west"
            priorities = (
                attack_action("enemy_crystal"),
                attack_action("enemy_tower"),
                skill3_action(),
                skill2_action(forward),
                attack_action("enemy_minion"),
                attack_action("enemy_hero"),
                dash_action(forward),
            )
        elif command.intent == MacroIntent.ENGAGE:
            forward = "east" if side == "blue" else "west"
            priorities = (
                skill3_action(),
                skill2_action(forward),
                attack_action("enemy_hero"),
                attack_action("enemy_minion"),
                dash_action(forward),
            )
        else:
            forward = "east" if side == "blue" else "west"
            priorities = (
                attack_action("enemy_minion"),
                skill2_action(forward),
                attack_action("enemy_hero"),
                skill3_action(),
                dash_action(forward),
            )
        for action in priorities:
            if action in legal:
                return action
        return self._navigator.action(side, command.target_zone, observation, legal)


class ProgressWatchdog:
    def __init__(self, max_no_progress_ticks: int = 6, max_recoveries: int = 3) -> None:
        self.max_no_progress_ticks = max_no_progress_ticks
        self.max_recoveries = max_recoveries
        self._signature: tuple[object, ...] | None = None
        self.no_progress_ticks = 0
        self.recoveries = 0
        self.stuck_ticks = 0

    def observe(self, observation: dict[str, object]) -> bool:
        own = cast(dict[str, int], observation["self_position"])
        signature = (
            own["x"],
            own["y"],
            observation["self_health"],
            observation["opponent_health"],
            observation["enemy_tower_health"],
            observation["enemy_crystal_health"],
            observation["self_respawn"],
            tuple(
                (item["x"], item["y"], item["health"])
                for item in cast(list[dict[str, int]], observation["red_minions"])
            ),
        )
        if self._signature == signature and int(cast(int, observation["self_respawn"])) == 0:
            self.no_progress_ticks += 1
            self.stuck_ticks += 1
        else:
            self.no_progress_ticks = 0
        self._signature = signature
        if self.no_progress_ticks < self.max_no_progress_ticks:
            return False
        self.no_progress_ticks = 0
        self.recoveries += 1
        return self.recoveries >= self.max_recoveries


class GlobalArena(RichPixelArena):
    """Middle-lane orchestration over Rich PixelArena without a second physics engine."""

    def apply_recall(self, side: Side) -> None:
        hero = self.state.blue if side == "blue" else self.state.red
        start = self.config.blue_start if side == "blue" else self.config.red_start
        if (hero.x, hero.y) == start and hero.respawn == 0:
            hero.health = self.config.hero_health


def _failure_code(arena: RichPixelArena, watchdog_failed: bool) -> FailureCode | None:
    if watchdog_failed:
        return FailureCode.NAV_STUCK
    if arena.state.outcome != "draw_tick_limit":
        return None
    if arena.state.red_tower_health == arena.config.tower_health:
        return FailureCode.NO_TOWER_DAMAGE
    if arena.state.red_tower_health > 0:
        return FailureCode.NO_WAVE_PROGRESS
    if arena.state.red_crystal_health == arena.config.crystal_health:
        return FailureCode.CRYSTAL_NOT_REACHED
    return FailureCode.TIMEOUT


def run_teacher_episode(
    seed: int, include_trace: bool = False
) -> tuple[EpisodeReport, list[TraceRow]]:
    arena = GlobalArena()
    arena.reset(seed)
    teacher = GlobalRuleTeacher()
    opponent = RichRandomPolicy(seed, "red")
    watchdog = ProgressWatchdog()
    trace: list[TraceRow] = []
    invalid_actions = deaths = recoveries = active_ticks = 0
    previous_respawn = 0
    watchdog_failed = False
    while not arena.state.terminal:
        observation = arena.observe("blue")
        legal = arena.legal_actions("blue")
        decision = teacher.decide("blue", legal, observation)
        if include_trace:
            trace.append(
                TraceRow(
                    arena.state.tick,
                    observation,
                    decision.command,
                    decision.action,
                    decision.scene_id,
                )
            )
        if decision.action not in legal:
            invalid_actions += 1
            action = wait_action()
        else:
            action = decision.action
        if decision.command.intent == MacroIntent.RECALL:
            arena.apply_recall("blue")
        other = opponent.select("red", arena.legal_actions("red"), arena.state.tick)
        arena.step(action, other)
        current = arena.observe("blue")
        respawn = int(cast(int, current["self_respawn"]))
        deaths += int(previous_respawn == 0 and respawn > 0)
        recoveries += int(previous_respawn > 0 and respawn == 0)
        previous_respawn = respawn
        active_ticks += int(respawn == 0)
        if watchdog.observe(current):
            watchdog_failed = True
            break
    failure = _failure_code(arena, watchdog_failed)
    crystal_destroyed = arena.state.outcome == "blue_win_crystal_destroyed"
    ratio = watchdog.stuck_ticks / max(1, active_ticks)
    recovery_opportunities = deaths - int(previous_respawn > 0 and arena.state.terminal)
    report = EpisodeReport(
        seed=seed,
        ticks=arena.state.tick,
        outcome=arena.state.outcome if not watchdog_failed else "safe_stop_nav_stuck",
        non_timeout_terminal=crystal_destroyed,
        crystal_destroyed=crystal_destroyed,
        manual_interventions=0,
        safety_violations=0,
        invalid_actions=invalid_actions,
        tower_damage=arena.config.tower_health - arena.state.red_tower_health,
        tower_progress=arena.state.red_tower_health < arena.config.tower_health,
        deaths=deaths,
        death_recovery_opportunities=recovery_opportunities,
        death_recoveries=recoveries,
        stuck_ticks=watchdog.stuck_ticks,
        active_ticks=active_ticks,
        stuck_time_ratio=ratio,
        failure_code=failure.value if failure is not None else None,
        passed=crystal_destroyed and invalid_actions == 0 and failure is None,
    )
    return report, trace


def evaluate_teacher(episodes: int, seed: int) -> dict[str, object]:
    _config, config_sha256 = load_global_config()
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rows = [run_teacher_episode(seed + index)[0] for index in range(episodes)]
    normal = sum(row.non_timeout_terminal for row in rows)
    tower = sum(row.tower_progress for row in rows)
    recovery_opportunities = sum(row.death_recovery_opportunities for row in rows)
    recoveries = sum(row.death_recoveries for row in rows)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-global-evaluation-v1",
        "policy": "rule_teacher",
        "episodes": episodes,
        "seed_start": seed,
        "config_sha256": config_sha256,
        "non_timeout_terminals": normal,
        "tower_progress_episodes": tower,
        "safety_violations": sum(row.safety_violations for row in rows),
        "invalid_actions": sum(row.invalid_actions for row in rows),
        "death_recovery_rate": (
            1.0 if recovery_opportunities == 0 else recoveries / recovery_opportunities
        ),
        "stuck_time_ratio": sum(row.stuck_ticks for row in rows)
        / max(1, sum(row.active_ticks for row in rows)),
        "reports": [row.to_dict() for row in rows],
    }
    if episodes == 1:
        passed = bool(rows[0].passed)
        stage = "1A"
    else:
        passed = (
            episodes == 20
            and normal >= 16
            and tower >= 14
            and payload["safety_violations"] == 0
            and payload["invalid_actions"] == 0
            and cast(float, payload["death_recovery_rate"]) == 1.0
            and cast(float, payload["stuck_time_ratio"]) < 0.10
        )
        stage = "1B"
    payload["stage"] = stage
    payload["status"] = "PASSED" if passed else "FAILED"
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["report_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload
