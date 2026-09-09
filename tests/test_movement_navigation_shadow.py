from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hok_agent.cli import main
from hok_agent.movement_navigation_shadow import run_partial_navigation_shadow


def _object_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    session_root = tmp_path / "sessions"
    directory = session_root / "teacher-session-002"
    shard_dir = directory / "shards"
    shard_dir.mkdir(parents=True)
    frames = np.zeros((12, 128, 128, 3), dtype=np.uint8)
    frames[:, 6:14, 116:124] = (20, 180, 40)
    frames[:, 7:13, 111:117] = (200, 40, 30)
    for index, y, x in ((0, 90, 52), (1, 112, 90), (2, 90, 52), (5, 90, 52), (6, 90, 52)):
        frames[index, y : y + 8, x : x + 8] = (20, 180, 40)
        frames[index, y + 1 : y + 7, x + 8 : x + 14] = (200, 40, 30)
    shard_path = shard_dir / "observations-0000.npz"
    np.savez_compressed(
        shard_path,
        minimap_rgb=frames,
        scheduled_elapsed_ms=np.arange(len(frames), dtype=np.int64) * 200,
    )
    source_summary: dict[str, object] = {
        "status": "PASSED",
        "derived_roi_rgb_persisted": True,
        "raw_frames_persisted": False,
        "observation_shards": [
            {
                "path": shard_path.name,
                "rows": len(frames),
                "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
            }
        ],
    }
    source_summary["summary_sha256"] = _object_sha256(source_summary)
    source_summary_path = directory / "summary.json"
    source_summary_path.write_text(json.dumps(source_summary), encoding="utf-8")

    prior: dict[str, object] = {
        "schema_version": "movement-real-navigation-demo-report-v1",
        "status": "DATA_SOURCE_LIMITED",
        "contract_sha256": "a" * 64,
        "visual_demonstrator_complete": True,
        "localization_improved": False,
        "sessions": [
            {
                "session": "teacher-session-002",
                "samples_total": 12,
                "direct_observations": 5,
            }
        ],
    }
    prior["summary_sha256"] = _object_sha256(prior)
    prior_path = tmp_path / "n1-summary.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")

    contract: dict[str, object] = {
        "schema_version": "movement-partial-navigation-shadow-v1",
        "prior_summary_file_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "prior_summary_sha256": prior["summary_sha256"],
        "prior_contract_sha256": prior["contract_sha256"],
        "session": {
            "basename": "teacher-session-002",
            "summary_sha256": hashlib.sha256(source_summary_path.read_bytes()).hexdigest(),
        },
        "color": {
            "green_minimum": 85,
            "green_red_margin": 18,
            "green_blue_margin": 10,
            "red_minimum": 105,
            "red_green_margin": 28,
            "red_blue_margin": 18,
        },
        "components": {
            "green_size": [20, 140],
            "green_extent": [7, 24],
            "red_size": [20, 240],
            "red_extent": [5, 24],
            "maximum_pair_l1_distance": 7.0,
            "reset_after_missing_frames": 10,
        },
        "excluded_ui_xyxy": [112, 0, 128, 16],
        "frame_period_ms": 200,
        "expected_frames": 12,
        "goal_xy_relative": [0.78, 0.78],
        "stop_radius_pixels": 8.0,
        "expected": {
            "direct_observations": 5,
            "unknown_observations": 7,
            "valid_runs": 2,
            "longest_valid_run_frames": 3,
            "proposal_counts": {"E": 4, "N": 1},
            "command_counts": {"DOWN": 2, "KEEP": 1, "MOVE": 2, "NOOP": 5, "UP": 2},
        },
        "training_allowed": False,
        "test_allowed": False,
        "device_input_allowed": False,
    }
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "shadow-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return session_root, prior_path, contract_path


def test_partial_navigation_shadow_resets_immediately_on_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session_root, prior, contract = _inputs(tmp_path)
    output = tmp_path / "shadow"
    forbidden = ("hok_agent.mobile_testbed", "hok_agent.movement_mvp_train")
    for module in forbidden:
        sys.modules.pop(module, None)
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "real-navigation-shadow",
                "--config",
                str(contract),
                "--prior-report",
                str(prior),
                "--session-root",
                str(session_root),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PARTIAL_OFFLINE_SHADOW_COMPLETE_DATA_SOURCE_LIMITED"
    assert report["frames"] == 12
    assert report["direct_observations"] == 5
    assert report["unknown_observations"] == 7
    assert report["valid_runs"] == 2
    assert report["longest_valid_run_frames"] == 3
    assert report["executed_actions"] == report["transitions_written"] == 0
    assert report["input_commands_sent"] == report["gpu_seconds"] == 0
    rows = [json.loads(line) for line in (output / "shadow.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    assert rows[1]["movement_command"] == rows[2]["movement_command"] == "MOVE"
    assert rows[3]["proposal"] is None and rows[3]["movement_command"] == "UP"
    assert rows[4]["proposal"] is None and rows[4]["movement_command"] == "NOOP"
    assert rows[5]["movement_command"] == "DOWN"
    assert all(row["executed_action"] is None for row in rows)
    assert all(row["transition_written"] is False for row in rows)
    assert {path.name for path in output.iterdir()} == {"shadow.jsonl", "summary.json"}
    assert not tuple(output.glob("*.sqlite3")) and not tuple(output.glob("*.npz"))
    assert all(module not in sys.modules for module in forbidden)
    with pytest.raises(ValueError, match="output already exists"):
        run_partial_navigation_shadow(contract, prior, session_root, output)


def test_partial_navigation_shadow_rejects_prior_tampering(tmp_path: Path) -> None:
    session_root, prior, contract = _inputs(tmp_path)
    prior.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="self hash differs"):
        run_partial_navigation_shadow(
            contract, prior, session_root, tmp_path / "tampered-output"
        )
