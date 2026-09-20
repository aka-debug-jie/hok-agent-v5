from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hok_agent.cli import main
from hok_agent.movement_navigation_shadow import (
    _response_candidate_pairs,
    run_action_response_identity_audit,
    run_active_probe_forensics,
    run_partial_navigation_shadow,
)


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


def _response_audit_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    session_root = tmp_path / "response-sessions"
    declarations: list[dict[str, object]] = []
    for session in ("teacher-session-002", "teacher-session-003", "teacher-session-005"):
        directory = session_root / session
        shard_dir = directory / "shards"
        shard_dir.mkdir(parents=True)
        frames = np.zeros((12, 128, 128, 3), dtype=np.uint8)
        frames[:, 6:14, 116:124] = (20, 180, 40)
        frames[:, 7:13, 111:117] = (200, 40, 30)
        movement = np.zeros(12, dtype=np.int8)
        sent = np.zeros(12, dtype=np.uint8)
        if session == "teacher-session-002":
            for index, y, x in ((0, 40, 52), (5, 42, 52), (6, 60, 52), (11, 60, 54)):
                frames[index, y : y + 8, x : x + 8] = (20, 180, 40)
                frames[index, y + 1 : y + 7, x + 8 : x + 14] = (200, 40, 30)
            movement[0], movement[6] = 5, 3
            sent[0], sent[6] = 1, 1
        else:
            movement[0], sent[0] = 5, 1
        shard_path = shard_dir / "observations-0000.npz"
        np.savez_compressed(
            shard_path,
            minimap_rgb=frames,
            scheduled_elapsed_ms=np.arange(12, dtype=np.int64) * 200,
            movement_id=movement,
            movement_input_sent=sent,
        )
        summary: dict[str, object] = {
            "status": "PASSED",
            "derived_roi_rgb_persisted": True,
            "raw_frames_persisted": False,
            "observation_shards": [
                {
                    "path": shard_path.name,
                    "rows": 12,
                    "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                }
            ],
        }
        summary["summary_sha256"] = _object_sha256(summary)
        summary_path = directory / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        declarations.append(
            {
                "basename": session,
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            }
        )
    prior: dict[str, object] = {
        "status": "DATA_SOURCE_LIMITED",
        "visual_demonstrator_complete": True,
    }
    prior["summary_sha256"] = _object_sha256(prior)
    prior_path = tmp_path / "prior-summary.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    contract: dict[str, object] = {
        "schema_version": "movement-action-response-identity-audit-v1",
        "prior_summary_file_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "prior_summary_sha256": prior["summary_sha256"],
        "sessions": declarations,
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
        "expected_frames_per_session": 12,
        "response_lag_ms": 1000,
        "movement_names": [
            "wait",
            "north",
            "north_east",
            "east",
            "south_east",
            "south",
            "south_west",
            "west",
            "north_west",
        ],
        "minimum_projection_pixels": 1.0,
        "gates": {
            "minimum_session002_interior_pairs": 2,
            "minimum_session002_directions": 2,
            "minimum_session002_responsive_fraction": 1.0,
            "minimum_session002_median_projection_pixels": 1.0,
            "maximum_fixed_ui_responsive_fraction": 0.0,
            "maximum_fixed_ui_median_displacement_pixels": 0.5,
        },
        "training_allowed": False,
        "test_allowed": False,
        "device_input_allowed": False,
    }
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "response-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return session_root, prior_path, contract_path


def test_response_candidate_generation_is_action_independent() -> None:
    rows = _response_candidate_pairs(
        [(40.0, 55.0, 1.0), (10.0, 116.0, 1.0)],
        [(42.0, 55.0, 1.0), (10.0, 116.0, 1.0)],
        (112, 0, 128, 16),
    )
    assert [row["candidate_group"] for row in rows] == ["interior", "fixed_ui"]
    assert "action" not in rows[0]


def test_action_response_audit_separates_moving_candidate_from_fixed_ui(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session_root, prior, contract = _response_audit_inputs(tmp_path)
    output = tmp_path / "response-audit"
    forbidden = ("hok_agent.mobile_testbed", "hok_agent.movement_mvp_train")
    for module in forbidden:
        sys.modules.pop(module, None)
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "action-response-identity-audit",
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
    assert report["status"] == "ACTION_RESPONSE_SEPARATES_FIXED_UI_SESSION002_ONLY"
    assert report["action_events_total"] == 4
    assert report["candidate_pairs"] == 6
    assert report["session_metrics"]["teacher-session-002:interior"]["responsive_pairs"] == 2
    assert report["fixed_ui_aggregate"]["responsive_pairs"] == 0
    assert report["other_sessions_interior_pairs"] == 0
    assert report["candidate_generation_uses_action"] is False
    assert report["semantic_player_identity_verified"] is False
    assert report["multi_session_identity_verified"] is False
    assert report["device_input_commands_sent"] == report["gpu_seconds"] == 0
    assert {path.name for path in output.iterdir()} == {"pairs.jsonl", "report.json"}
    assert all(module not in sys.modules for module in forbidden)
    with pytest.raises(ValueError, match="output already exists"):
        run_action_response_identity_audit(contract, prior, session_root, output)


def _draw_probe_player(frame: np.ndarray, y: int, x: int) -> None:
    frame[y : y + 8, x : x + 8] = (20, 180, 40)
    frame[y + 1 : y + 7, x + 8 : x + 14] = (200, 40, 30)


def _write_probe_session(
    session_root: Path,
    name: str,
    contract: dict[str, object],
    *,
    contaminate: bool,
) -> None:
    directory = session_root / name
    shard_dir = directory / "shards"
    shard_dir.mkdir(parents=True)
    frames = np.zeros((30, 128, 128, 3), dtype=np.uint8)
    for frame in frames:
        frame[6:14, 116:124] = (20, 180, 40)
        frame[7:13, 111:117] = (200, 40, 30)
    for index in range(30):
        if index <= 5:
            _draw_probe_player(frames[index], 60, 60)
        elif index <= 15:
            _draw_probe_player(frames[index], 57, 60)
        else:
            _draw_probe_player(frames[index], 57, 63)
    times = np.arange(30, dtype=np.int64) * 200
    movement = np.zeros(30, dtype=np.int8)
    sent = np.zeros(30, dtype=np.uint8)
    for start, direction_id in ((5, 1), (15, 3), (25, 5)):
        movement[start : start + 3] = direction_id
        sent[start : start + 3] = 1
    if contaminate:
        sent[20] = 1
        movement[20] = 0
    shard_path = shard_dir / "observations-0000.npz"
    np.savez_compressed(
        shard_path,
        minimap_rgb=frames,
        scheduled_elapsed_ms=times,
        frame_elapsed_ms=times,
        screen_valid=np.ones(30, dtype=np.uint8),
        operation_allowed=np.ones(30, dtype=np.uint8),
        movement_id=movement,
        movement_input_sent=sent,
    )
    events: list[dict[str, object]] = [
        {
            "kind": "control",
            "window_start_ms": 200,
            "window_end_ms": 800,
            "contaminated": False,
            "hard_stop": False,
        },
        {
            "kind": "pulse",
            "direction": "north",
            "press_scheduled_ms": 1000,
            "press_ack_ms": 1000,
            "release_scheduled_ms": 1500,
            "release_ack_ms": 1500,
            "contaminated": False,
            "hard_stop": False,
        },
        {
            "kind": "pulse",
            "direction": "east",
            "press_scheduled_ms": 3000,
            "press_ack_ms": 3000,
            "release_scheduled_ms": 3500,
            "release_ack_ms": 3500,
            "contaminated": False,
            "hard_stop": False,
        },
        {
            "kind": "pulse",
            "direction": "south",
            "press_scheduled_ms": 5000,
            "press_ack_ms": 5000,
            "release_scheduled_ms": 5500,
            "release_ack_ms": 5500,
            "contaminated": False,
            "hard_stop": False,
        },
    ]
    with (directory / "pulses.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    summary: dict[str, object] = {
        "schema_version": contract["session_schema_version"],
        "contract_sha256": contract["contract_sha256"],
        "derived_roi_rgb_persisted": True,
        "raw_frames_persisted": False,
        "pulses_dispatched": 3,
        "control_windows_dispatched": 1,
        "input_commands_sent": 6,
        "observation_shards": [
            {
                "path": shard_path.name,
                "rows": 30,
                "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
            }
        ],
    }
    summary["summary_sha256"] = _object_sha256(summary)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _probe_inputs(
    tmp_path: Path,
    *,
    contaminate: bool = False,
    extra_contract: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    session_root = tmp_path / "probe-sessions"
    contract: dict[str, object] = {
        "schema_version": "movement-active-probe-contract-v1",
        "session_schema_version": "hok-agent-mobile-active-probe-session-v1",
        "directions": [
            "north",
            "north_east",
            "east",
            "south_east",
            "south",
            "south_west",
            "west",
            "north_west",
        ],
        "movement_names": [
            "wait",
            "north",
            "north_east",
            "east",
            "south_east",
            "south",
            "south_west",
            "west",
            "north_west",
        ],
        "frame_period_ms": 200,
        "observation_ms": 1000,
        "hold_ms": 500,
        "inter_pulse_gap_ms": 400,
        "sessions_required": 2,
        "measurement": {"maximum_gap_to_analysis_frame_ms": 600},
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
        "gates_per_session": {
            "minimum_dispatched_pulses": 3,
            "minimum_paired_fraction": 0.6,
            "minimum_balanced_directions": 2,
            "minimum_direction_correct_fraction": 0.5,
            "minimum_projection_pixels": 1.0,
            "minimum_pulse_median_minus_control_p95_pixels": 0.0,
            "maximum_fixed_ui_responsive_fraction": 0.05,
            "minimum_analysis_coverage_fraction": 0.5,
            "maximum_unknown_streak_seconds": 1.0,
            "minimum_valid_run_seconds": 1.0,
            "maximum_identity_switch_events": 0,
        },
        "training_allowed": False,
        "test_allowed": False,
    }
    if extra_contract is not None:
        contract.update(extra_contract)
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "probe-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    for name in ("probe-session-001", "probe-session-002"):
        _write_probe_session(session_root, name, contract, contaminate=contaminate)
    return contract_path, session_root


def test_active_probe_audit_verifies_identity_and_control(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, session_root = _probe_inputs(tmp_path)
    output = tmp_path / "probe-audit"
    forbidden = ("hok_agent.mobile_testbed", "hok_agent.movement_mvp_train")
    for module in forbidden:
        sys.modules.pop(module, None)
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "active-probe-audit",
                "--config",
                str(contract),
                "--session-root",
                str(session_root),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ACTIVE_PROBE_IDENTITY_AND_CONTROL_VERIFIED"
    assert report["sessions_found"] == 2
    assert report["sessions_passed"] == ["probe-session-001", "probe-session-002"]
    assert report["identity_control_verified"] is True
    metrics = report["session_metrics"]["probe-session-001"]
    assert metrics["fates"] == {"outside_analysis_window": 1, "paired": 2}
    assert metrics["paired_fraction"] == pytest.approx(2 / 3)
    assert metrics["direction_correct_fraction"] == 1.0
    assert metrics["analysis_coverage"] == 1.0
    assert metrics["identity_switch_events"] == 0
    assert metrics["checks"] == {key: True for key in metrics["checks"]}
    assert report["device_input_commands_sent"] == 12
    assert report["navigation_available"] is False
    assert report["training_called"] is False
    assert {path.name for path in output.iterdir()} == {"pulses.jsonl", "report.json"}
    rows = [
        json.loads(line) for line in (output / "pulses.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 8
    assert sum(row["fate"] == "paired" and row["kind"] == "pulse" for row in rows) == 4
    assert all(module not in sys.modules for module in forbidden)


def test_active_probe_audit_counts_contaminated_denominator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, session_root = _probe_inputs(tmp_path, contaminate=True)
    output = tmp_path / "probe-audit-contaminated"
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "active-probe-audit",
                "--config",
                str(contract),
                "--session-root",
                str(session_root),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ACTIVE_PROBE_GATES_FAILED"
    assert report["identity_control_verified"] is False
    metrics = report["session_metrics"]["probe-session-001"]
    assert metrics["fates"]["contaminated"] == 1
    assert metrics["fates"]["paired"] == 1
    assert metrics["paired_fraction"] == pytest.approx(1 / 3)
    assert metrics["checks"]["paired_fraction"] is False


def test_active_probe_forensics_is_read_only_and_measures_windows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, session_root = _probe_inputs(tmp_path)
    output = tmp_path / "probe-forensics"
    forbidden = ("hok_agent.mobile_testbed", "hok_agent.movement_mvp_train")
    for module in forbidden:
        sys.modules.pop(module, None)
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "active-probe-forensics",
                "--config",
                str(contract),
                "--session-root",
                str(session_root),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "movement-active-probe-forensics-report-v1"
    assert report["sessions_found"] == 2
    assert report["device_input_commands_sent"] == 0
    assert report["training_called"] is False
    assert report["test_frames_read"] == 0
    assert report["report_sha256"]
    assert {path.name for path in output.iterdir()} == {"report.json"}
    metrics = report["session_metrics"]["probe-session-001"]
    assert metrics["frames"] == 30
    assert metrics["pulses"] == 3
    assert metrics["control_windows"] == 1
    assert metrics["per_direction"]["north"]["pulses"] == 1
    assert metrics["per_direction"]["north"]["hold_projection_median"] == pytest.approx(3.0)
    assert metrics["green_only_candidates_per_frame"]["many"] >= 1
    assert report["pooled"]["hold_projection"]["n"] == 6
    indicators = report["pooled"]["indicators"]
    assert indicators["pulse_hold_signal_over_matched_idle_auc"] is not None
    assert indicators["pulse_hold_nonzero_fraction"] >= 0.5
    assert all(module not in sys.modules for module in forbidden)


def test_active_probe_forensics_fails_closed_on_existing_output(tmp_path: Path) -> None:
    contract, session_root = _probe_inputs(tmp_path)
    output = tmp_path / "probe-forensics-existing"
    output.mkdir()
    with pytest.raises(ValueError, match="output already exists"):
        run_active_probe_forensics(contract, session_root, output)


def test_active_probe_audit_enforces_declared_guards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    guards: dict[str, object] = {
        "free_movement_region": {
            "minimum_y": 0,
            "maximum_y": 100,
            "minimum_x": 0,
            "maximum_x": 128,
            "maximum_consecutive_violation_frames": 5,
        },
        "maximum_frame_gap_ms": 1000,
        "maximum_press_start_drift_ms": 300,
    }
    contract, session_root = _probe_inputs(tmp_path, extra_contract=guards)
    output = tmp_path / "probe-audit-guards"
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "active-probe-audit",
                "--config",
                str(contract),
                "--session-root",
                str(session_root),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ACTIVE_PROBE_IDENTITY_AND_CONTROL_VERIFIED"
    metrics = report["session_metrics"]["probe-session-001"]
    assert metrics["checks"]["free_movement_region"] is True
    assert metrics["checks"]["capture_stall"] is True
    assert metrics["checks"]["press_start_drift"] is True
    assert metrics["region_maximum_streak"] == 0
    assert metrics["region_violation_frames"] == 0


def test_active_probe_audit_rejects_region_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    guards: dict[str, object] = {
        "free_movement_region": {
            "minimum_y": 0,
            "maximum_y": 50,
            "minimum_x": 0,
            "maximum_x": 128,
            "maximum_consecutive_violation_frames": 5,
        },
    }
    contract, session_root = _probe_inputs(tmp_path, extra_contract=guards)
    output = tmp_path / "probe-audit-region"
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "active-probe-audit",
                "--config",
                str(contract),
                "--session-root",
                str(session_root),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ACTIVE_PROBE_GATES_FAILED"
    metrics = report["session_metrics"]["probe-session-001"]
    assert metrics["checks"]["free_movement_region"] is False
    assert metrics["region_maximum_streak"] > 5
    assert "capture_stall" not in metrics["checks"]
