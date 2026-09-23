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


def test_candidate_shadow_has_its_own_read_only_authorization() -> None:
    """The candidate keeps every read-only flag and is explicitly non-promoting."""
    authorization_sha256 = global_shadow._load_candidate_authorization()
    assert len(authorization_sha256) == 64
    payload = json.loads(
        global_shadow.CANDIDATE_AUTHORIZATION_PATH.read_text(encoding="utf-8")
    )
    assert payload["read_only_shadow_allowed"] is True
    assert payload["control_output"] is False
    assert payload["device_input_allowed"] is False
    assert payload["online_learning_allowed"] is False
    assert payload["promotion_allowed"] is False
    assert payload["candidate_is_non_inferior"] is True
    assert payload["candidate_checkpoint_sha256"] == global_shadow.CANDIDATE_CHECKPOINT_SHA256
    assert payload["baseline_checkpoint_sha256"] == global_shadow.PROMOTED_CHECKPOINT_SHA256
    assert payload["candidate_checkpoint_sha256"] != payload["baseline_checkpoint_sha256"]


def test_candidate_and_promoted_shadow_cannot_be_confused() -> None:
    """The two variants carry disjoint schema names and disjoint checkpoint identities."""
    assert global_shadow.CANDIDATE_SCHEMA != global_shadow.SCHEMA
    assert global_shadow.CANDIDATE_CHECKPOINT_SHA256 != global_shadow.PROMOTED_CHECKPOINT_SHA256
    assert global_shadow.CANDIDATE_AUTHORIZATION_PATH != global_shadow.AUTHORIZATION_PATH


def test_candidate_shadow_authorization_rejects_a_tampered_promotion_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate authorization that claims promotion is refused, not silently accepted."""
    payload = json.loads(
        global_shadow.CANDIDATE_AUTHORIZATION_PATH.read_text(encoding="utf-8")
    )
    payload["promotion_allowed"] = True
    tampered = tmp_path / "candidate.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(global_shadow, "CANDIDATE_AUTHORIZATION_PATH", tampered)
    with pytest.raises(global_shadow.GlobalShadowError, match="candidate Shadow authorization"):
        global_shadow._load_candidate_authorization()


def test_promoted_shadow_contract_is_unchanged_by_the_candidate_addition() -> None:
    """The promoted authorization still loads and still pins the promoted checkpoint."""
    _config, _config_sha, authorization_sha256 = global_shadow._load_contracts()
    assert len(authorization_sha256) == 64
    assert global_shadow.PROMOTED_CHECKPOINT_SHA256 == (
        "c033264f83d1c667c4dff5f93dec02183535386cfcde1a527a60303647a3e39e"
    )


def test_shadow_cli_candidate_flag_grants_no_input(tmp_path: Path) -> None:
    args = cli._parser().parse_args(
        [
            "global-agent-shadow",
            "--serial",
            "X",
            "--video-node",
            str(tmp_path / "v"),
            "--checkpoint",
            str(tmp_path / "c.safetensors"),
            "--output-dir",
            str(tmp_path / "o"),
            "--run-seconds",
            "60",
            "--candidate",
        ]
    )
    assert args.candidate is True
    assert not hasattr(args, "enable_input")
