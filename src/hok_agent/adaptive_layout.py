"""Icon-independent device geometry and hero skill behavior profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np

from hok_agent.mobile_testbed import ABILITIES, Layout

ADAPTIVE_LAYOUT_SCHEMA: Final = "hok-agent-adaptive-layout-v1"
HERO_PROFILE_SCHEMA: Final = "hok-agent-hero-ability-profile-v1"
ABILITY_MODES: Final = (
    "tap",
    "directional_drag",
    "charge_release",
    "targeted_tap",
    "disabled",
)
GROUP_NAMES: Final = ("joystick", "combat", "minimap", "purchase")


class AdaptiveLayoutError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


@dataclass(frozen=True)
class ContentBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def normalized_to_pixel(self, point: tuple[float, float]) -> tuple[float, float]:
        return (self.x0 + point[0] * self.width, self.y0 + point[1] * self.height)

    def pixel_to_normalized(self, point: tuple[float, float]) -> tuple[float, float]:
        return ((point[0] - self.x0) / self.width, (point[1] - self.y0) / self.height)


@dataclass(frozen=True)
class GroupTransform:
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    confidence: float = 1.0
    reprojection_error_fraction: float = 0.0

    def apply(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            self.offset_x + self.scale_x * point[0],
            self.offset_y + self.scale_y * point[1],
        )


@dataclass(frozen=True)
class AdaptiveLayout:
    width: int
    height: int
    content_box: ContentBox
    reference_layout_sha256: str
    build_identity_sha256: str
    groups: dict[str, GroupTransform]
    layout_sha256: str


@dataclass(frozen=True)
class AbilityBehavior:
    mode: str
    aim_radius: float | None
    hold_ms: int | None
    minimum_interval_ms: int
    cooldown_observation_required: bool


@dataclass(frozen=True)
class HeroProfile:
    hero_id: str
    abilities: dict[str, AbilityBehavior]
    profile_sha256: str

    def behavior(self, ability: str) -> AbilityBehavior:
        if ability not in ABILITIES[1:]:
            raise AdaptiveLayoutError("hero profile ability is invalid")
        return self.abilities[ability]


def detect_content_box(frame: np.ndarray) -> ContentBox:
    """Find the non-black game content rectangle; fail closed for tiny candidates."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise AdaptiveLayoutError("content-box input must be HxWx3 uint8 RGB")
    active = frame.max(axis=2) > 8
    rows = np.flatnonzero(active.any(axis=1))
    columns = np.flatnonzero(active.any(axis=0))
    if not len(rows) or not len(columns):
        raise AdaptiveLayoutError("content-box detector found no active pixels")
    box = ContentBox(int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1)
    if box.width < frame.shape[1] * 0.5 or box.height < frame.shape[0] * 0.5:
        raise AdaptiveLayoutError("content-box detector found an implausibly small region")
    return box


def project_reference_point(
    reference_point: tuple[float, float],
    group: GroupTransform,
) -> tuple[float, float]:
    projected = group.apply(reference_point)
    if not 0.0 <= projected[0] <= 1.0 or not 0.0 <= projected[1] <= 1.0:
        raise AdaptiveLayoutError("adaptive group transform projects outside the content box")
    return projected


def adapt_layout(reference: Layout, profile: AdaptiveLayout) -> Layout:
    joystick = project_reference_point(reference.joystick_center, profile.groups["joystick"])
    buttons = {
        ability: None
        if point is None
        else project_reference_point(point, profile.groups["combat"])
        for ability, point in reference.buttons.items()
    }
    radius_scale = min(
        profile.groups["joystick"].scale_x,
        profile.groups["joystick"].scale_y,
    )
    aim_scale = min(
        profile.groups["combat"].scale_x,
        profile.groups["combat"].scale_y,
    )
    return Layout(
        profile.content_box.width,
        profile.content_box.height,
        joystick,
        reference.joystick_radius * radius_scale,
        reference.forward_vector,
        reference.move_hold_ms,
        reference.skill_hold_ms,
        reference.aim_radius * aim_scale,
        buttons,
    )


def _pair(value: object, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, (int, float)) for item in value)
    ):
        raise AdaptiveLayoutError(f"{name} is invalid")
    return (float(value[0]), float(value[1]))


def _group(value: object, name: str) -> GroupTransform:
    if not isinstance(value, dict):
        raise AdaptiveLayoutError(f"adaptive group {name} is invalid")
    keys = {
        "scale_x",
        "scale_y",
        "offset_x",
        "offset_y",
        "confidence",
        "reprojection_error_fraction",
    }
    if set(value) != keys:
        raise AdaptiveLayoutError(f"adaptive group {name} fields differ")
    numbers = {key: value[key] for key in keys}
    if not all(isinstance(item, (int, float)) for item in numbers.values()):
        raise AdaptiveLayoutError(f"adaptive group {name} values are invalid")
    result = GroupTransform(**{key: float(item) for key, item in numbers.items()})
    if (
        not 0.5 <= result.scale_x <= 1.5
        or not 0.5 <= result.scale_y <= 1.5
        or not -0.2 <= result.offset_x <= 0.2
        or not -0.2 <= result.offset_y <= 0.2
        or not 0.9 <= result.confidence <= 1.0
        or not 0.0 <= result.reprojection_error_fraction <= 0.015
    ):
        raise AdaptiveLayoutError(f"adaptive group {name} confidence or transform is unsafe")
    return result


def load_adaptive_layout(path: Path) -> AdaptiveLayout:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveLayoutError("adaptive layout is unavailable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != ADAPTIVE_LAYOUT_SCHEMA:
        raise AdaptiveLayoutError("adaptive layout schema is invalid")
    if value.get("calibration_status") != "ADAPTED":
        raise AdaptiveLayoutError("adaptive layout is not calibrated")
    if value.get("layout_sha256") != _self_hash(value, "layout_sha256"):
        raise AdaptiveLayoutError("adaptive layout hash differs")
    width, height = value.get("width"), value.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise AdaptiveLayoutError("adaptive layout screen is invalid")
    raw_box = value.get("content_box_xyxy")
    if (
        not isinstance(raw_box, list)
        or len(raw_box) != 4
        or not all(isinstance(item, int) for item in raw_box)
    ):
        raise AdaptiveLayoutError("adaptive layout content box is invalid")
    x0, y0, x1, y1 = cast(list[int], raw_box)
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise AdaptiveLayoutError("adaptive layout content box is outside the screen")
    reference_sha = value.get("reference_layout_sha256")
    identity_sha = value.get("build_identity_sha256")
    if not (
        isinstance(reference_sha, str)
        and len(reference_sha) == 64
        and isinstance(identity_sha, str)
        and len(identity_sha) == 64
    ):
        raise AdaptiveLayoutError("adaptive layout binding hashes are invalid")
    raw_groups = value.get("groups")
    if not isinstance(raw_groups, dict) or set(raw_groups) != set(GROUP_NAMES):
        raise AdaptiveLayoutError("adaptive layout groups differ")
    return AdaptiveLayout(
        width,
        height,
        ContentBox(x0, y0, x1, y1),
        reference_sha,
        identity_sha,
        {name: _group(raw_groups[name], name) for name in GROUP_NAMES},
        cast(str, value["layout_sha256"]),
    )


def load_hero_profile(path: Path) -> HeroProfile:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveLayoutError("hero profile is unavailable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != HERO_PROFILE_SCHEMA:
        raise AdaptiveLayoutError("hero profile schema is invalid")
    if value.get("profile_status") != "CONFIGURED":
        raise AdaptiveLayoutError("hero profile is not configured")
    if value.get("profile_sha256") != _self_hash(value, "profile_sha256"):
        raise AdaptiveLayoutError("hero profile hash differs")
    hero_id = value.get("hero_id")
    raw_abilities = value.get("abilities")
    if not isinstance(hero_id, str) or not hero_id or not isinstance(raw_abilities, dict):
        raise AdaptiveLayoutError("hero profile fields are invalid")
    if set(raw_abilities) != set(ABILITIES[1:]):
        raise AdaptiveLayoutError("hero profile ability slots differ")
    abilities: dict[str, AbilityBehavior] = {}
    for ability in ABILITIES[1:]:
        raw = raw_abilities[ability]
        if not isinstance(raw, dict):
            raise AdaptiveLayoutError("hero ability behavior is invalid")
        required = {
            "mode",
            "aim_radius",
            "hold_ms",
            "minimum_interval_ms",
            "cooldown_observation_required",
        }
        if set(raw) != required:
            raise AdaptiveLayoutError("hero ability behavior fields differ")
        mode = raw["mode"]
        radius, hold, interval, cooldown = (
            raw["aim_radius"],
            raw["hold_ms"],
            raw["minimum_interval_ms"],
            raw["cooldown_observation_required"],
        )
        if (
            not isinstance(mode, str)
            or mode not in ABILITY_MODES
            or radius is not None and not isinstance(radius, (int, float))
            or hold is not None and not isinstance(hold, int)
            or not isinstance(interval, int)
            or not isinstance(cooldown, bool)
        ):
            raise AdaptiveLayoutError("hero ability behavior values are invalid")
        if (
            interval < 0
            or hold is not None and not 1 <= hold <= 1500
            or radius is not None and not 0.0 < float(radius) <= 0.3
            or mode == "directional_drag" and radius is None
            or mode == "charge_release" and hold is None
            or mode == "disabled" and (radius is not None or hold is not None or cooldown)
        ):
            raise AdaptiveLayoutError("hero ability behavior is unsafe")
        abilities[ability] = AbilityBehavior(
            mode,
            None if radius is None else float(radius),
            hold,
            interval,
            cooldown,
        )
    return HeroProfile(hero_id, abilities, cast(str, value["profile_sha256"]))


def execution_mode(profile: HeroProfile | None, ability: str) -> str:
    if ability == "basic_attack":
        return "tap"
    if profile is None:
        return "disabled"
    return profile.behavior(ability).mode
