from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent import cli, global_shadow


def test_shadow_contract_is_explicitly_read_only() -> None:
    config, config_sha256, authorization_sha256 = global_shadow._load_contracts()
    assert config["read_only_shadow_allowed"] is True
    assert config["control_output"] is False
    assert config["device_input_allowed"] is False
    assert len(config_sha256) == len(authorization_sha256) == 64


def test_shadow_candidates_are_descriptive_not_input_operations() -> None:
    assert global_shadow._candidate_modes("ENGAGE", "HOLD_CURRENT_ZONE", False) == (
        "hold",
        "hero_combat",
        "none",
    )
    assert global_shadow._candidate_modes("PUSH_STRUCTURE", "ENEMY_BASE", False) == (
        "advance",
        "structure",
        "none",
    )
    assert global_shadow._candidate_modes("ENGAGE", "ENEMY_BASE", True) == (
        "hold",
        "no_combat",
        "none",
    )


def test_shadow_rejects_unadmitted_duration_before_device_access(tmp_path: Path) -> None:
    with pytest.raises(global_shadow.GlobalShadowError, match="duration"):
        global_shadow.run_global_shadow(
            serial="ABC123",
            video_node=Path("/dev/videoN"),
            checkpoint=tmp_path / "missing.safetensors",
            output_dir=tmp_path / "output",
            run_seconds=1,
            device_name="cpu",
        )


def test_shadow_cli_has_no_enable_input_flag() -> None:
    parser = cli._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "global-agent-shadow",
                "--serial",
                "ABC123",
                "--video-node",
                "/dev/videoN",
                "--checkpoint",
                "model.safetensors",
                "--output-dir",
                "out",
                "--run-seconds",
                "60",
                "--enable-input",
            ]
        )


def test_shadow_source_has_no_sender_or_control_transport() -> None:
    source = inspect.getsource(global_shadow)
    for forbidden in ("AdbInputPipe", "ScrcpyControlSession", "input touchscreen", "touch("):
        assert forbidden not in source


def test_local_observation_views_bind_geometry_without_persisting_boxes(tmp_path: Path) -> None:
    config = {
        "schema_version": "hok-agent-mobile-observation-rois-v1",
        "screen": {"width": 8, "height": 6},
        "main_view": {"observation_only": True, "pixel_box_xyxy": [0, 0, 4, 3]},
        "minimap": {"observation_only": True, "pixel_box_xyxy": [4, 0, 8, 3]},
        "hud": {"observation_only": True, "pixel_box_xyxy": [0, 3, 8, 6]},
    }
    path = tmp_path / "rois.local.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    boxes, digest = global_shadow._load_local_observation_rois(path)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    frame[:3, :4] = 10
    frame[:3, 4:] = 40
    frame[3:] = 90
    main, minimap, hud = global_shadow._local_observation_views(frame, boxes)
    assert (main.shape, minimap.shape, hud.shape) == ((128, 128, 3), (64, 64, 3), (32, 128, 3))
    assert (int(main[0, 0, 0]), int(minimap[0, 0, 0]), int(hud[0, 0, 0])) == (10, 40, 90)
    assert len(digest) == 64
