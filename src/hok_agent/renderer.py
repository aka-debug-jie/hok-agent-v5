from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random
from typing import Final

import numpy as np

from hok_agent.arena import DEFAULT_CONFIG, ArenaConfig


def _render_defaults() -> ArenaConfig:
    return ArenaConfig.load(DEFAULT_CONFIG)


_RENDERER_CONFIG: Final = _render_defaults()
Color = tuple[int, int, int]
RENDERER_ID: Final = "hok-agent-v3-pure-numpy-rgb-128"
_RENDERER_SPEC: Final = {
    "id": RENDERER_ID,
    "size": [128, 128, 3],
    "dtype": "uint8",
    "channels": ["r", "g", "b"],
    "palette": "seeded_nonsemantic",
}
RENDERER_HASH: Final = hashlib.sha256(
    json.dumps(
        {
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "spec": _RENDERER_SPEC,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def renderer_identity() -> str:
    return RENDERER_ID


def renderer_hash() -> str:
    return RENDERER_HASH


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return minimum if value < minimum else maximum if value > maximum else value


def _to_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def _jitter_color(rng: Random, base: Color, spread: int) -> Color:
    return (
        _clamp(base[0] + rng.randint(-spread, spread), 0, 255),
        _clamp(base[1] + rng.randint(-spread, spread), 0, 255),
        _clamp(base[2] + rng.randint(-spread, spread), 0, 255),
    )


def _draw_rect(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Color,
) -> None:
    x0 = _clamp(x, 0, canvas.shape[1])
    x1 = _clamp(x + w, 0, canvas.shape[1])
    y0 = _clamp(y, 0, canvas.shape[0])
    y1 = _clamp(y + h, 0, canvas.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    canvas[y0:y1, x0:x1] = color


def _draw_bar(canvas: np.ndarray, x: int, y: int, value: int, color: Color) -> None:
    max_units = 10
    units = _clamp(value, 0, max_units)
    _draw_rect(canvas, x, y, 2 * units, 3, color)


def _draw_progress(canvas: np.ndarray, ratio: float, y: int, color: Color) -> None:
    ratio = max(0.0, min(1.0, ratio))
    total = canvas.shape[1] - 16
    width = int(round(ratio * total))
    _draw_rect(canvas, 8, y, total, 2, (color[0] // 2, color[1] // 2, color[2] // 2))
    _draw_rect(canvas, 8, y, width, 2, color)


def _lane_x(position: int, lane_min: int, lane_max: int) -> int:
    clamped = _clamp(position, lane_min, lane_max) - lane_min
    span = max(1, lane_max - lane_min)
    return int(round(clamped * (127 / span)))


def render(observation: dict[str, object], render_seed: int = 0) -> np.ndarray:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    c = _RENDERER_CONFIG

    seed = int(render_seed)
    rng = Random(seed)

    side = str(observation.get("side", "blue"))
    tick = _to_int(observation.get("tick", 0))
    max_ticks = _to_int(observation.get("max_ticks", c.max_ticks), max(c.max_ticks, 1))
    self_position = _to_int(observation.get("self_position", c.blue_start))
    opponent_position = _to_int(observation.get("opponent_position", c.red_start))
    self_health = _to_int(observation.get("self_health", c.hero_health))
    opponent_health = _to_int(observation.get("opponent_health", c.hero_health))
    self_tower_health = _to_int(observation.get("own_tower_health", c.tower_health))
    opponent_tower_health = _to_int(observation.get("enemy_tower_health", c.tower_health))
    self_crystal_health = _to_int(observation.get("own_crystal_health", c.crystal_health))
    opponent_crystal_health = _to_int(observation.get("enemy_crystal_health", c.crystal_health))

    lane_min = c.lane_min
    lane_max = c.lane_max
    blue_tower = c.blue_tower
    red_tower = c.red_tower
    blue_crystal = c.blue_crystal
    red_crystal = c.red_crystal

    if side == "blue":
        self_tower = blue_tower
        enemy_tower = red_tower
        self_crystal = blue_crystal
        enemy_crystal = red_crystal
    else:
        self_tower = red_tower
        enemy_tower = blue_tower
        self_crystal = red_crystal
        enemy_crystal = blue_crystal

    lane_width = rng.randint(2, 4)
    vertical_jitter = rng.randint(-2, 2)
    terrain_color = _jitter_color(rng, (60, 75, 55), 10)
    bg_color = _jitter_color(rng, (8, 12, 16), 5)
    self_color = _jitter_color(rng, (40, 205, 235), 20)
    enemy_color = _jitter_color(rng, (235, 55, 45), 20)
    tower_color = _jitter_color(rng, (230, 180, 45), 20)
    crystal_color = _jitter_color(rng, (190, 60, 220), 20)
    health_colors = (
        self_color,
        enemy_color,
        tower_color,
        tower_color,
        crystal_color,
        crystal_color,
    )

    frame[:] = bg_color

    lane_span = max(1, lane_max - lane_min)
    lane_y = 60 + vertical_jitter

    def view_x(position: int) -> int:
        normalized = lane_min + lane_max - position if side == "red" else position
        return _lane_x(normalized, lane_min, lane_max)

    for lane in range(lane_span + 1):
        x = view_x(lane_min + lane)
        _draw_rect(frame, x, lane_y - 4, lane_width, 30, terrain_color)

    self_x = view_x(self_position)
    opp_x = view_x(opponent_position)
    self_tower_x = view_x(self_tower)
    enemy_tower_x = view_x(enemy_tower)
    self_crystal_x = view_x(self_crystal)
    enemy_crystal_x = view_x(enemy_crystal)

    _draw_rect(frame, self_x - 4, lane_y - 12, 7, 7, self_color)
    _draw_rect(
        frame,
        opp_x - 4,
        lane_y - 12 + 4,
        7,
        7,
        enemy_color,
    )
    _draw_rect(frame, self_tower_x - 4, lane_y - 26, 7, 7, tower_color)
    _draw_rect(frame, enemy_tower_x - 4, lane_y - 26 + 2, 7, 7, tower_color)
    _draw_rect(
        frame,
        self_crystal_x - 4,
        lane_y + 10,
        7,
        7,
        crystal_color,
    )
    _draw_rect(
        frame,
        enemy_crystal_x - 4,
        lane_y + 12,
        7,
        7,
        crystal_color,
    )

    _draw_bar(frame, 8, 4, self_health, health_colors[0])
    _draw_bar(frame, 8, 10, opponent_health, health_colors[1])
    _draw_bar(frame, 8, 16, self_tower_health, health_colors[2])
    _draw_bar(frame, 8, 22, opponent_tower_health, health_colors[3])
    _draw_bar(frame, 8, 28, self_crystal_health, health_colors[4])
    _draw_bar(frame, 8, 34, opponent_crystal_health, health_colors[5])

    _draw_rect(
        frame,
        8,
        lane_y + 22,
        112,
        2,
        (terrain_color[0] // 2, terrain_color[1] // 2, terrain_color[2] // 2),
    )
    _draw_progress(frame, tick / max_ticks if max_ticks else 0.0, 118, terrain_color)

    return frame.astype(np.uint8)
