from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent import adaptive_layout
from hok_agent.mobile_testbed import Layout


def _hashed(value: dict[str, object], field: str) -> dict[str, object]:
    value[field] = hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != field},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return value


def _groups() -> dict[str, dict[str, float]]:
    return {
        name: {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "confidence": 0.95,
            "reprojection_error_fraction": 0.01,
        }
        for name in adaptive_layout.GROUP_NAMES
    }


def test_content_box_detects_letterboxed_game_area() -> None:
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    frame[10:110, 20:220] = 32
    assert adaptive_layout.detect_content_box(frame) == adaptive_layout.ContentBox(20, 10, 220, 110)


def test_adaptive_layout_projects_groups_without_icon_content(tmp_path: Path) -> None:
    groups = _groups()
    groups["combat"]["scale_x"] = 0.9
    groups["combat"]["offset_x"] = 0.05
    value: dict[str, object] = {
        "schema_version": adaptive_layout.ADAPTIVE_LAYOUT_SCHEMA,
        "calibration_status": "ADAPTED",
        "width": 2400,
        "height": 1080,
        "content_box_xyxy": [100, 40, 2300, 1040],
        "reference_layout_sha256": "a" * 64,
        "build_identity_sha256": "b" * 64,
        "groups": groups,
    }
    path = tmp_path / "adaptive.local.json"
    path.write_text(json.dumps(_hashed(value, "layout_sha256")), encoding="utf-8")
    profile = adaptive_layout.load_adaptive_layout(path)
    reference = Layout(
        1600,
        720,
        (0.2, 0.8),
        0.1,
        (0.0, -1.0),
        800,
        300,
        0.08,
        {"basic_attack": (0.8, 0.8), "skill1": (0.7, 0.7), "skill2": None, "skill3": None},
    )
    result = adaptive_layout.adapt_layout(reference, profile)
    assert result.width == 2200
    assert result.height == 1000
    assert result.joystick_center == (0.2, 0.8)
    assert result.buttons["basic_attack"] == pytest.approx((0.77, 0.8))


def test_hero_profile_requires_explicit_behavior_and_unknown_disables(tmp_path: Path) -> None:
    value: dict[str, object] = {
        "schema_version": adaptive_layout.HERO_PROFILE_SCHEMA,
        "profile_status": "CONFIGURED",
        "hero_id": "hero-a",
        "abilities": {
            "basic_attack": {
                "mode": "tap",
                "aim_radius": None,
                "hold_ms": None,
                "minimum_interval_ms": 500,
                "cooldown_observation_required": False,
            },
            "skill1": {
                "mode": "directional_drag",
                "aim_radius": 0.08,
                "hold_ms": None,
                "minimum_interval_ms": 600,
                "cooldown_observation_required": True,
            },
            "skill2": {
                "mode": "charge_release",
                "aim_radius": None,
                "hold_ms": 400,
                "minimum_interval_ms": 900,
                "cooldown_observation_required": True,
            },
            "skill3": {
                "mode": "disabled",
                "aim_radius": None,
                "hold_ms": None,
                "minimum_interval_ms": 0,
                "cooldown_observation_required": False,
            },
        },
    }
    path = tmp_path / "hero.local.json"
    path.write_text(json.dumps(_hashed(value, "profile_sha256")), encoding="utf-8")
    profile = adaptive_layout.load_hero_profile(path)
    assert adaptive_layout.execution_mode(profile, "skill1") == "directional_drag"
    assert adaptive_layout.execution_mode(profile, "skill3") == "disabled"
    assert adaptive_layout.execution_mode(None, "skill2") == "disabled"

    root = Path(__file__).resolve().parents[1]
    with pytest.raises(adaptive_layout.AdaptiveLayoutError, match="not configured"):
        adaptive_layout.load_hero_profile(root / "configs/hero_profile.example.json")
