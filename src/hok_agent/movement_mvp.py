from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np

from hok_agent.frame_bus import FramePacket, LatestFrameBus, RgbView, ViewName
from hok_agent.rich_arena import (
    ArenaConfig,
    FactorizedAction,
    RichPixelArena,
    move_action,
    wait_action,
)
from hok_agent.rich_renderer import render
from hok_agent.transition_store import (
    ExecutedActionRecord,
    HierarchicalTransitionRecord,
    MovementAction,
    PolicyProposalRecord,
    ProposalBundleRecord,
    ReplayRecord,
    RewardComponentsRecord,
    RewardRecord,
    UnifiedTransitionStore,
)

StageAMovement = Literal["STOP", "N", "S", "W", "E", "NW", "NE", "SW", "SE"]

MOVEMENT_ACTIONS: Final[tuple[StageAMovement, ...]] = (
    "STOP",
    "N",
    "S",
    "W",
    "E",
    "NW",
    "NE",
    "SW",
    "SE",
)
_RICH_DIRECTION: Final = {
    "N": "north",
    "S": "south",
    "W": "west",
    "E": "east",
    "NW": "northwest",
    "NE": "northeast",
    "SW": "southwest",
    "SE": "southeast",
}
_SIGN_TO_ACTION: Final[dict[tuple[int, int], StageAMovement]] = {
    (0, 0): "STOP",
    (0, -1): "N",
    (0, 1): "S",
    (-1, 0): "W",
    (1, 0): "E",
    (-1, -1): "NW",
    (1, -1): "NE",
    (-1, 1): "SW",
    (1, 1): "SE",
}
_ZERO_HASH = hashlib.sha256(b"movement-mvp-zero-reward-event-engine-v0").hexdigest()
_TARGET_COLORS: Final = {
    "opponent_hero": np.array((225, 70, 65), dtype=np.uint8),
    "enemy_tower": np.array((220, 120, 65), dtype=np.uint8),
}
_VECTORS: Final = {
    "N": (0, -1),
    "S": (0, 1),
    "W": (-1, 0),
    "E": (1, 0),
    "NW": (-1, -1),
    "NE": (1, -1),
    "SW": (-1, 1),
    "SE": (1, 1),
}


@dataclass(frozen=True, slots=True)
class NavigationConfig:
    contract_version: str
    hero: str
    role: str
    side: Literal["blue"]
    lane: Literal["bottom"]
    start: tuple[int, int]
    goal: tuple[int, int]
    step_duration_ms: int
    max_steps: int
    seed: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "contract_version": self.contract_version,
                    "hero": self.hero,
                    "role": self.role,
                    "side": self.side,
                    "lane": self.lane,
                    "start": self.start,
                    "goal": self.goal,
                    "step_duration_ms": self.step_duration_ms,
                    "max_steps": self.max_steps,
                    "seed": self.seed,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def load_navigation_config(path: Path) -> NavigationConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = NavigationConfig(
        contract_version=str(raw["contract_version"]),
        hero=str(raw["hero"]),
        role=str(raw["role"]),
        side=cast(Literal["blue"], raw["side"]),
        lane=cast(Literal["bottom"], raw["lane"]),
        start=cast(tuple[int, int], tuple(map(int, raw["start"]))),
        goal=cast(tuple[int, int], tuple(map(int, raw["goal"]))),
        step_duration_ms=int(raw["step_duration_ms"]),
        max_steps=int(raw["max_steps"]),
        seed=int(raw["seed"]),
    )
    if (config.hero, config.role, config.side, config.lane) != (
        "houyi",
        "marksman",
        "blue",
        "bottom",
    ):
        raise ValueError("stage A is fixed to blue-side Houyi marksman bottom lane")
    if config.start[1] != 4 or config.goal[1] != 4:
        raise ValueError("stage A start and goal must remain on the bottom lane")
    if config.step_duration_ms != 100 or config.max_steps <= 0:
        raise ValueError("stage A requires 100 ms steps and a positive step limit")
    return config


def rule_movement(position: tuple[int, int], goal: tuple[int, int]) -> StageAMovement:
    dx = (goal[0] > position[0]) - (goal[0] < position[0])
    dy = (goal[1] > position[1]) - (goal[1] < position[1])
    return _SIGN_TO_ACTION[(dx, dy)]


def rule_movement_in_range(
    position: tuple[int, int], goal: tuple[int, int], stop_radius: int = 1
) -> StageAMovement:
    delta_x, delta_y = goal[0] - position[0], goal[1] - position[1]
    if abs(delta_x) + abs(delta_y) <= stop_radius:
        return "STOP"
    if abs(delta_x) == 1 and abs(delta_y) == 1:
        return "E" if delta_x > 0 else "W"
    return rule_movement(position, goal)


def mark_visible_target(rgb: np.ndarray, target_category: str) -> np.ndarray:
    color = _TARGET_COLORS[target_category]
    ys, xs = np.nonzero(np.all(rgb == color, axis=2))
    if not len(xs):
        raise ValueError(f"visible target not found: {target_category}")
    center_x, center_y = int(round(float(xs.mean()))), int(round(float(ys.mean())))
    marked = rgb.copy()
    marker = np.array((245, 225, 45), dtype=np.uint8)
    radius = 7
    for y in range(max(0, center_y - radius), min(marked.shape[0], center_y + radius + 1)):
        for x in range(max(0, center_x - radius), min(marked.shape[1], center_x + radius + 1)):
            distance = (x - center_x) ** 2 + (y - center_y) ** 2
            if (radius - 1) ** 2 <= distance <= radius**2:
                marked[y, x] = marker
    return marked


def rgb_geometry_movement(marked_rgb: np.ndarray) -> StageAMovement:
    own_y, own_x = np.nonzero(np.all(marked_rgb == (55, 195, 235), axis=2))
    target_y, target_x = np.nonzero(np.all(marked_rgb == (245, 225, 45), axis=2))
    if not len(own_x) or not len(target_x):
        raise ValueError("RGB geometry baseline requires visible self and target")
    dx = int(round((float(target_x.mean()) - float(own_x.mean())) / 8.0))
    dy = int(round((float(target_y.mean()) - float(own_y.mean())) / 13.0))
    return rule_movement_in_range((0, 0), (dx, dy))


def _warmup_pair(current: tuple[int, int], goal: tuple[int, int]) -> tuple[str, str]:
    pairs = (("east", "west"), ("west", "east"), ("north", "south"), ("south", "north"))
    vector = {
        "north": (0, -1),
        "south": (0, 1),
        "west": (-1, 0),
        "east": (1, 0),
    }
    for outgoing, incoming in pairs:
        dx, dy = vector[outgoing]
        destination = (current[0] + dx, current[1] + dy)
        if 0 <= destination[0] < 15 and destination[1] in (2, 3, 4) and destination != goal:
            return outgoing, incoming
    raise ValueError("no warmup movement is available")


def _causal_clip(
    current: tuple[int, int], goal: tuple[int, int], render_seed: int
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    arena = RichPixelArena(ArenaConfig(max_ticks=32, blue_start=current, red_start=goal))
    arena.reset(render_seed)
    outgoing, incoming = _warmup_pair(current, goal)
    frames: list[np.ndarray] = []

    def observe(count: int) -> None:
        for _ in range(count):
            raw = render(arena.observe("blue"), render_seed)
            frames.append(mark_visible_target(raw, "opponent_hero"))

    observe(4)
    arena.step(move_action(outgoing), wait_action())
    observe(4)
    arena.step(move_action(incoming), wait_action())
    observe(4)
    arena.step(move_action(outgoing), wait_action())
    observe(2)
    arena.step(move_action(incoming), wait_action())
    observe(2)
    before_raw = cast(dict[str, int], arena.observe("blue")["self_position"])
    before = (before_raw["x"], before_raw["y"])
    action = rule_movement_in_range(before, goal)
    arena.step(to_arena_action(action), wait_action())
    after_raw = cast(dict[str, int], arena.observe("blue")["self_position"])
    after = (after_raw["x"], after_raw["y"])
    return np.stack(frames), before, after


def materialize_overfit32(config_path: Path, output_dir: Path) -> dict[str, object]:
    config = load_navigation_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "overfit32.npz"
    report_path = output_dir / "overfit32.json"
    if dataset_path.exists() or report_path.exists():
        raise ValueError("stage B overfit32 output already exists")

    direction_positions = {
        "N": ((7, 4), (7, 2)),
        "S": ((7, 2), (7, 4)),
        "W": ((8, 3), (6, 3)),
        "E": ((6, 3), (8, 3)),
        "NW": ((8, 4), (6, 2)),
        "NE": ((6, 4), (8, 2)),
        "SW": ((8, 2), (6, 4)),
        "SE": ((6, 2), (8, 4)),
    }
    stop_goals = ((7, 2), (8, 3), (7, 4), (6, 3)) * 2
    samples: list[tuple[StageAMovement, tuple[int, int], tuple[int, int], int]] = []
    for replicate in range(3):
        for action in MOVEMENT_ACTIONS[1:]:
            current, goal = direction_positions[action]
            samples.append((action, current, goal, 100 + replicate))
    samples.extend(("STOP", (7, 3), goal, 200 + index) for index, goal in enumerate(stop_goals))

    clips, labels, episode_ids = [], [], []
    moved_history = 0
    for index, (expected, current, goal, render_seed) in enumerate(samples):
        clip, before, after = _causal_clip(current, goal, render_seed)
        actual = rule_movement_in_range(before, goal)
        if actual != expected:
            raise ValueError("stage B teacher label differs")
        changed = before != after
        if changed != (expected != "STOP"):
            raise ValueError("stage B executed action causality differs")
        clips.append(clip)
        labels.append(MOVEMENT_ACTIONS.index(expected))
        episode_ids.append(f"stage-b-overfit-{index:02d}")
        moved_history += int(any(not np.array_equal(clip[i], clip[i + 1]) for i in range(15)))

    np.savez_compressed(
        dataset_path,
        rgb_sequence=np.stack(clips).astype(np.uint8),
        label=np.asarray(labels, dtype=np.int64),
        episode_id=np.asarray(episode_ids),
        action=np.asarray([MOVEMENT_ACTIONS[label] for label in labels]),
    )
    counts = {action: labels.count(index) for index, action in enumerate(MOVEMENT_ACTIONS)}
    report: dict[str, object] = {
        "status": "PASSED",
        "schema_version": "movement-mvp-stage-b-overfit32-v0",
        "config_sha256": config.sha256,
        "samples": len(samples),
        "frames_per_sample": 16,
        "class_counts": counts,
        "unique_episode_ids": len(set(episode_ids)),
        "windows_with_state_change": moved_history,
        "model_inputs": ["goal_marked_rgb_sequence"],
        "teacher_fields_in_model_input": False,
        "action_executed_after_window": True,
        "input_commands_sent": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_c_scenarios() -> tuple[dict[str, object], ...]:
    positions = tuple((x, y) for x in range(2, 13) for y in (2, 3, 4))
    buckets: dict[StageAMovement, list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(
        list
    )
    for start in positions:
        for goal in positions:
            action = rule_movement_in_range(start, goal)
            if action != "STOP":
                buckets[action].append((start, goal))
    rows: list[dict[str, object]] = []
    for action in MOVEMENT_ACTIONS[1:]:
        pairs = sorted(
            buckets[action],
            key=lambda pair: _canonical_sha256([action, pair[0], pair[1]]),
        )
        for ordinal, (start, goal) in enumerate(pairs[:11]):
            split = "train" if ordinal < 8 else "dev"
            identity = _canonical_sha256(["movement-stage-c", action, start, goal])
            rows.append(
                {
                    "scenario_id": identity[:16],
                    "split": split,
                    "initial_action": action,
                    "start": list(start),
                    "goal": list(goal),
                    "render_seed": int(identity[:8], 16),
                }
            )
    return tuple(rows)


def _materialize_teacher_episode(
    scenario: dict[str, object], maximum_steps: int, step_duration_ms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    start = cast(tuple[int, int], tuple(cast(list[int], scenario["start"])))
    goal = cast(tuple[int, int], tuple(cast(list[int], scenario["goal"])))
    render_seed = int(cast(int, scenario["render_seed"]))
    arena = RichPixelArena(
        ArenaConfig(max_ticks=maximum_steps + 1, blue_start=start, red_start=goal)
    )
    arena.reset(render_seed)
    frames: list[np.ndarray] = []
    labels: list[int] = []
    timestamps_ms: list[int] = []
    success = False
    for step in range(maximum_steps):
        observation = arena.observe("blue")
        raw = render(observation, render_seed)
        frames.append(mark_visible_target(raw, "opponent_hero"))
        timestamps_ms.append(step * step_duration_ms)
        position_raw = cast(dict[str, int], observation["self_position"])
        position = (position_raw["x"], position_raw["y"])
        action = rule_movement_in_range(position, goal)
        labels.append(MOVEMENT_ACTIONS.index(action))
        arena.step(to_arena_action(action), wait_action())
        if action == "STOP":
            success = True
            break
    return (
        np.stack(frames).astype(np.uint8),
        np.asarray(labels, dtype=np.int64),
        np.asarray(timestamps_ms, dtype=np.int64),
        success,
    )


def materialize_stage_c_trajectories(config_path: Path, output_dir: Path) -> dict[str, object]:
    raw = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
    stage = cast(dict[str, object], raw["stage_c"])
    maximum_steps = int(cast(int, stage["maximum_episode_steps"]))
    step_duration_ms = int(cast(int, stage["step_duration_ms"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.json"
    episodes_dir = output_dir / "episodes"
    if manifest_path.exists() or report_path.exists() or episodes_dir.exists():
        raise ValueError("stage C trajectory output already exists")
    episodes_dir.mkdir()

    episodes: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    teacher_successes: Counter[str] = Counter()
    for scenario in stage_c_scenarios():
        split = str(scenario["split"])
        ordinal = counts[split]
        basename = f"{split}-{ordinal:03d}.npz"
        frames, labels, timestamps_ms, success = _materialize_teacher_episode(
            scenario, maximum_steps, step_duration_ms
        )
        np.savez_compressed(
            episodes_dir / basename,
            frames=frames,
            labels=labels,
            frame_timestamps_ms=timestamps_ms,
            window_end=np.arange(len(labels), dtype=np.int64),
        )
        episodes.append(
            {
                **scenario,
                "basename": basename,
                "steps": len(labels),
                "teacher_success": success,
                "artifact_sha256": _file_sha256(episodes_dir / basename),
            }
        )
        counts[split] += 1
        teacher_successes[split] += int(success)

    expected = {
        "train": int(cast(int, stage["train_episodes"])),
        "dev": int(cast(int, stage["dev_episodes"])),
    }
    initial_support = {
        split: Counter(str(row["initial_action"]) for row in episodes if row["split"] == split)
        for split in ("train", "dev")
    }
    scenario_overlap = len(
        {str(row["scenario_id"]) for row in episodes if row["split"] == "train"}
        & {str(row["scenario_id"]) for row in episodes if row["split"] == "dev"}
    )
    passed = (
        dict(counts) == expected
        and teacher_successes["train"] == expected["train"]
        and teacher_successes["dev"] >= int(cast(int, stage["minimum_teacher_successes"]))
        and scenario_overlap == 0
        and all(initial_support["train"][action] == 8 for action in MOVEMENT_ACTIONS[1:])
        and all(initial_support["dev"][action] == 3 for action in MOVEMENT_ACTIONS[1:])
    )
    manifest_without_hash: dict[str, object] = {
        "schema_version": "movement-mvp-stage-c-trajectories-v0",
        "step_duration_ms": step_duration_ms,
        "sequence_frames": int(cast(int, stage["sequence_frames"])),
        "episodes": episodes,
    }
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": _canonical_sha256(manifest_without_hash),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    label_counts = {
        split: {action: 0 for action in MOVEMENT_ACTIONS} for split in ("train", "dev")
    }
    for row in episodes:
        split = str(row["split"])
        with np.load(episodes_dir / str(row["basename"]), allow_pickle=False) as data:
            for label in data["labels"].tolist():
                label_counts[split][MOVEMENT_ACTIONS[int(label)]] += 1
    report: dict[str, object] = {
        "status": "PASSED" if passed else "FAILED",
        "schema_version": "movement-mvp-stage-c-trajectory-report-v0",
        "manifest_sha256": manifest["manifest_sha256"],
        "episode_counts": dict(counts),
        "teacher_successes": dict(teacher_successes),
        "initial_direction_support": {
            split: dict(initial_support[split]) for split in ("train", "dev")
        },
        "label_counts": label_counts,
        "scenario_overlap": scenario_overlap,
        "frames_stored_once_per_episode": True,
        "windows_stored_as_indices": True,
        "actor_input": "goal_marked_rgb_only",
        "simulator_only": True,
        "input_commands_sent": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError("stage C trajectory gate failed")
    return report


def to_arena_action(action: StageAMovement) -> FactorizedAction:
    return wait_action() if action == "STOP" else move_action(_RICH_DIRECTION[action])


def _packet(
    output_dir: Path,
    episode_id: str,
    step_id: int,
    timestamp_ns: int,
    observation: dict[str, object],
    render_seed: int,
) -> FramePacket:
    main = render(observation, render_seed)
    minimap = np.ascontiguousarray(main[::2, ::2])
    hud = np.ascontiguousarray(main[-16:])
    basename = f"{episode_id}-{step_id:03d}.npz"
    np.savez_compressed(output_dir / basename, main=main, minimap=minimap, hud=hud)
    views = tuple(
        RgbView(cast(ViewName, name), value.tobytes(), cast(tuple[int, int, int], value.shape))
        for name, value in (("main", main), ("minimap", minimap), ("hud", hud))
    )
    return FramePacket(
        observation_id=f"{episode_id}-obs-{step_id:03d}",
        capture_start_ns=timestamp_ns,
        capture_end_ns=timestamp_ns + 1_000_000,
        capture_source_class="pixelarena",
        frame_bundle_ref=basename,
        views=views,
    )


def _proposal(observation_id: str, value: str, start_ns: int) -> PolicyProposalRecord:
    return {
        "observation_id": observation_id,
        "applied_observation_id": observation_id,
        "carried_forward": False,
        "value": value,
        "confidence": 1.0,
        "decision_start_ns": start_ns,
        "decision_end_ns": start_ns + 1_000_000,
        "valid_until_ns": start_ns + 20_000_000,
        "policy_bundle_version": "movement-mvp-stage-a-rule-v0",
    }


def _movement_command(previous: StageAMovement, current: StageAMovement) -> str:
    if current == "STOP":
        return "UP" if previous != "STOP" else "NOOP"
    if previous == "STOP":
        return "DOWN"
    return "KEEP" if previous == current else "MOVE"


def _transition(
    config: NavigationConfig,
    episode_id: str,
    step_id: int,
    observation: FramePacket,
    next_observation: FramePacket,
    action: StageAMovement,
    previous_action: StageAMovement,
    *,
    success: bool,
    timeout: bool,
) -> HierarchicalTransitionRecord:
    decision_start = observation.capture_end_ns + 1_000_000
    dispatch_start = decision_start + 2_000_000
    dispatch_ack = dispatch_start + 1_000_000
    settle_end = dispatch_ack + 1_000_000
    proposals: ProposalBundleRecord = {
        "macro": _proposal(observation.observation_id, "HOLD", decision_start),
        "movement": _proposal(observation.observation_id, action, decision_start),
        "combat": _proposal(observation.observation_id, "WAIT", decision_start),
    }
    executed: ExecutedActionRecord = {
        "requested_movement": cast(MovementAction, action),
        "applied_movement": cast(MovementAction, action),
        "requested_combat": "WAIT",
        "applied_combat": "WAIT",
        "movement_command": cast(
            Literal["DOWN", "MOVE", "UP", "KEEP", "NOOP"],
            _movement_command(previous_action, action),
        ),
        "combat_command": "NOOP",
        "dispatch_start_ns": dispatch_start,
        "dispatch_ack_ns": dispatch_ack,
        "first_attempt_status": "acknowledged" if action != "STOP" else "not_attempted",
        "retry_status": "not_attempted",
        "retry_count": 0,
        "final_status": "acknowledged" if action != "STOP" else "noop",
    }
    components: RewardComponentsRecord = {
        "terminal": 0.0,
        "death": 0.0,
        "self_hp_delta": 0.0,
        "tower_damage": 0.0,
    }
    reward: RewardRecord = {
        "reward_version": "zero-reward-v0",
        "components": components,
        "total": 0.0,
        "event_ids": [],
    }
    replay: ReplayRecord = {"source": "controller", "failure_tags": [], "priority": 1.0}
    done = success or timeout
    return {
        "schema_version": "hok-agent-hierarchical-transition-v0",
        "episode_id": episode_id,
        "step_id": step_id,
        "policy_bundle_version": "movement-mvp-stage-a-rule-v0",
        "policy_bundle_sha256": config.sha256,
        "event_engine_version": "disabled-zero-reward-v0",
        "event_engine_sha256": _ZERO_HASH,
        "observation": observation.to_record(),
        "proposals": proposals,
        "executed_action": executed,
        "settle_end_ns": settle_end,
        "next_observation": next_observation.to_record(),
        "events": [],
        "reward": reward,
        "done": done,
        "terminal_reason": (
            "NAVIGATION_GOAL_REACHED" if success else "TIMEOUT" if timeout else "NOT_DONE"
        ),
        "episode_end_kind": "TERMINATED" if success else "TRUNCATED" if timeout else "NOT_DONE",
        "causal_order_valid": True,
        "training_eligible": True,
        "replay": replay,
    }


def run_stage_a(config_path: Path, output_dir: Path) -> dict[str, object]:
    config = load_navigation_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "replay.sqlite3"
    summary_path = output_dir / "summary.json"
    if database.exists() or summary_path.exists():
        raise ValueError("stage A output already exists")

    arena = RichPixelArena(
        ArenaConfig(
            max_ticks=max(config.max_steps + 1, 32),
            blue_start=config.start,
            red_start=(12, 2),
        )
    )
    arena.reset(config.seed)
    episode_id = f"movement-stage-a-seed-{config.seed}"
    bus = LatestFrameBus()
    step_ns = config.step_duration_ms * 1_000_000
    observation = _packet(
        output_dir, episode_id, 0, 0, arena.observe("blue"), config.seed
    )
    bus.publish(observation)
    previous_action: StageAMovement = "STOP"
    positions = [config.start]
    actions: list[StageAMovement] = []

    with UnifiedTransitionStore(database) as store:
        for step_id in range(config.max_steps):
            current = arena.observe("blue")
            position_raw = cast(dict[str, int], current["self_position"])
            position = (position_raw["x"], position_raw["y"])
            action = rule_movement(position, config.goal)
            arena.step(to_arena_action(action), wait_action())
            next_position_raw = cast(dict[str, int], arena.observe("blue")["self_position"])
            next_position = (next_position_raw["x"], next_position_raw["y"])
            success = action == "STOP" and position == config.goal and next_position == position
            timeout = step_id + 1 == config.max_steps and not success
            next_observation = _packet(
                output_dir,
                episode_id,
                step_id + 1,
                (step_id + 1) * step_ns,
                arena.observe("blue"),
                config.seed + step_id + 1,
            )
            bus.publish(next_observation)
            stored = store.append(
                _transition(
                    config,
                    episode_id,
                    step_id,
                    observation,
                    next_observation,
                    action,
                    previous_action,
                    success=success,
                    timeout=timeout,
                )
            )
            if not stored.validation.valid:
                raise ValueError(f"stage A transition invalid: {stored.validation.errors}")
            actions.append(action)
            positions.append(next_position)
            observation = next_observation
            previous_action = action
            if success or timeout:
                break
        rows = store.load_episode(episode_id)

    final = rows[-1]
    summary: dict[str, object] = {
        "status": "PASSED" if final["terminal_reason"] == "NAVIGATION_GOAL_REACHED" else "FAILED",
        "contract_version": config.contract_version,
        "config_sha256": config.sha256,
        "hero": config.hero,
        "role": config.role,
        "side": config.side,
        "lane": config.lane,
        "step_duration_ms": config.step_duration_ms,
        "episode_id": episode_id,
        "steps": len(rows),
        "actions": actions,
        "positions": positions,
        "position_changed": any(a != b for a, b in zip(positions, positions[1:], strict=True)),
        "terminal_reason": final["terminal_reason"],
        "episode_end_kind": final["episode_end_kind"],
        "terminal_transition_committed": final["done"] and len(rows) == len(actions),
        "reward_total": sum(float(row["reward"]["total"]) for row in rows),
        "input_commands_sent": 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
