from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random
from typing import Final, cast

import numpy as np

from hok_agent.rich_arena import BOARD_HEIGHT, BOARD_WIDTH, LANE_Y, RICH_IDENTITY

Color = tuple[int, int, int]
RENDERER_ID: Final = "hok-agent-v7-rich-rgb-128-v2"
RENDERER_SPEC: Final = {
    "identity": RICH_IDENTITY,
    "size": [128, 128, 3],
    "dtype": "uint8",
    "perspective": "self-side; red rotated 180 degrees",
    "visible": ["two heroes", "all minions", "four building HP", "three cooldowns", "respawn"],
}
RENDERER_HASH: Final = hashlib.sha256(
    json.dumps(
        {
            "id": RENDERER_ID,
            "spec": RENDERER_SPEC,
            "source": Path(__file__).read_text(encoding="utf-8"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def renderer_identity() -> str:
    return RENDERER_ID


def renderer_hash() -> str:
    return RENDERER_HASH


def _rect(frame: np.ndarray, x: int, y: int, width: int, height: int, color: Color) -> None:
    x0, y0, x1, y1 = max(0, x), max(0, y), min(128, x + width), min(128, y + height)
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = color


def _xy(x: int, y: int, side: str) -> tuple[int, int]:
    if side == "red":
        x, y = BOARD_WIDTH - 1 - x, BOARD_HEIGHT - 1 - y
    return 7 + round(x * 114 / (BOARD_WIDTH - 1)), 31 + round(y * 76 / (BOARD_HEIGHT - 1))


def _position(value: object) -> tuple[int, int]:
    raw = cast(dict[str, object], value) if isinstance(value, dict) else {}
    return _integer(raw.get("x", 0)), _integer(raw.get("y", 0))


def _integer(value: object) -> int:
    return cast(int, value)


def _entity(
    frame: np.ndarray, point: tuple[int, int], color: Color, size: int, health: int, maximum: int
) -> None:
    x, y = point
    _rect(frame, x - size // 2, y - size // 2, size, size, color)
    _bar(frame, x - size // 2, y - size // 2 - 3, size, health, maximum, (80, 230, 100))


def _bar(
    frame: np.ndarray, x: int, y: int, width: int, value: int, maximum: int, color: Color
) -> None:
    _rect(frame, x, y, width, 2, (25, 25, 28))
    _rect(frame, x, y, round(width * max(0, min(value, maximum)) / maximum), 2, color)


def render(observation: dict[str, object], render_seed: int = 0) -> np.ndarray:
    """Render one public observation; no legal domain or structured truth is accepted."""
    side, rng = str(observation.get("side", "blue")), Random(render_seed)
    frame = np.empty((128, 128, 3), dtype=np.uint8)
    frame[:] = (12 + rng.randrange(9), 18 + rng.randrange(9), 22 + rng.randrange(9))
    lane = (38 + rng.randrange(8), 49 + rng.randrange(8), 30 + rng.randrange(8))
    for y in LANE_Y:
        _, py = _xy(0, y, side)
        _rect(frame, 5, py - 1, 118, 3, lane)

    own_color, enemy_color = (55, 195, 235), (225, 70, 65)
    own_hp, enemy_hp = (
        _integer(observation.get("self_health", 0)),
        _integer(observation.get("opponent_health", 0)),
    )
    _entity(
        frame, _xy(*_position(observation.get("self_position")), side), own_color, 9, own_hp, 10
    )
    _entity(
        frame,
        _xy(*_position(observation.get("opponent_position")), side),
        enemy_color,
        9,
        enemy_hp,
        10,
    )

    structure = (
        (
            (1, 3),
            _integer(
                observation.get("self_tower_health" if side == "blue" else "enemy_tower_health", 0)
            ),
            12,
            (70, 130, 220) if side == "blue" else (220, 120, 65),
            8,
        ),
        (
            (13, 3),
            _integer(
                observation.get("enemy_tower_health" if side == "blue" else "self_tower_health", 0)
            ),
            12,
            (220, 120, 65) if side == "blue" else (70, 130, 220),
            8,
        ),
        (
            (0, 3),
            _integer(
                observation.get(
                    "self_crystal_health" if side == "blue" else "enemy_crystal_health", 0
                )
            ),
            16,
            (110, 90, 220) if side == "blue" else (215, 80, 155),
            7,
        ),
        (
            (14, 3),
            _integer(
                observation.get(
                    "enemy_crystal_health" if side == "blue" else "self_crystal_health", 0
                )
            ),
            16,
            (215, 80, 155) if side == "blue" else (110, 90, 220),
            7,
        ),
    )
    for position, health, maximum, color, size in structure:
        _entity(frame, _xy(*position, side), color, size, health, maximum)

    for team in ("blue", "red"):
        is_self = team == side
        color = own_color if is_self else enemy_color
        for raw in cast(list[object], observation.get(f"{team}_minions", [])):
            item = cast(dict[str, object], raw)
            health = _integer(item.get("health", 0))
            if health > 0:
                _entity(frame, _xy(*_position(item), side), color, 5, health, 3)

    cooldowns = cast(dict[str, object], observation.get("self_cooldowns", {}))
    for index, (skill, maximum, color) in enumerate(
        (
            ("skill1", 4, (80, 180, 255)),
            ("skill2", 3, (95, 235, 165)),
            ("skill3", 6, (255, 170, 90)),
        )
    ):
        value = _integer(cooldowns.get(skill, 0))
        _bar(frame, 7 + index * 39, 7, 34, maximum - value, maximum, color)
    _bar(
        frame,
        7,
        13,
        112,
        4 - _integer(observation.get("self_respawn", 0)),
        4,
        (230, 200, 120),
    )
    _bar(
        frame,
        7,
        18,
        112,
        _integer(observation.get("tick", 0)),
        max(1, _integer(observation.get("max_ticks", 96))),
        (130, 140, 155),
    )
    return frame
