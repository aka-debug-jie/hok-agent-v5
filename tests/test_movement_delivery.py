from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from hok_agent.cli import main
from hok_agent.hierarchical_e1 import (
    AuditSession,
    audit_health_sessions,
    replay_health_event_transitions,
)
from hok_agent.movement_delivery import (
    _object_sha256,
    create_hierarchical_rule_package,
    create_offline_cycle_package,
    create_r0_package,
    verify_hierarchical_rule_package,
    verify_offline_cycle_package,
    verify_r0_package,
)
from hok_agent.movement_mvp import run_rule_batch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "movement_mvp.json"
HEALTH_CONFIG = ROOT / "configs" / "hierarchical_event_e1_health.json"


@pytest.fixture(scope="module")
def delivery_evidence(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("movement-delivery")
    interrupted, control, package = root / "interrupted", root / "control", root / "package"
    run_rule_batch(CONFIG, interrupted, 10, step_budget=4)
    run_rule_batch(CONFIG, interrupted, 10, resume=True)
    run_rule_batch(CONFIG, control, 10)
    create_r0_package(interrupted, control, package)
    return interrupted, control, package


def _health_frame(run_length: int) -> np.ndarray:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    if run_length:
        frame[30, 58 : 58 + run_length] = (40, 180, 50)
    return frame


def _health_session(path: Path, runs: list[int], hard_stops: list[int]) -> None:
    shard_dir = path / "shards"
    shard_dir.mkdir(parents=True)
    frames = np.stack([_health_frame(run) for run in runs])
    shard = shard_dir / "observations-0000.npz"
    np.savez_compressed(
        shard,
        main_rgb=frames,
        minimap_rgb=frames,
        hud_rgb=frames,
        scheduled_elapsed_ms=np.arange(len(frames), dtype=np.int64) * 200,
        hard_stop=np.asarray(hard_stops, dtype=np.uint8),
    )
    summary = {
        "status": "PASSED",
        "derived_roi_rgb_persisted": True,
        "raw_frames_persisted": False,
        "observation_shards": [
            {
                "path": shard.name,
                "rows": len(frames),
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (path / "summary.json").write_text(json.dumps(summary))


@pytest.fixture(scope="module")
def cycle_evidence(
    tmp_path_factory: pytest.TempPathFactory,
    delivery_evidence: tuple[Path, Path, Path],
) -> tuple[Path, Path, list[Path], Path]:
    root = tmp_path_factory.mktemp("cycle-delivery")
    session_paths = [root / name for name in ("train-a", "train-b", "dev", "challenge")]
    _health_session(session_paths[0], [12] * 10, [0] * 10)
    _health_session(session_paths[1], [12] * 10, [0] * 10)
    _health_session(
        session_paths[2],
        [12, 12, 12, 0, 0, 0, 0, 12, 12, 12],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    )
    _health_session(session_paths[3], [12] * 10, [0] * 10)
    health_audit = root / "health-audit"
    audit_health_sessions(
        HEALTH_CONFIG,
        (
            AuditSession("train-a", "train_live", session_paths[0]),
            AuditSession("train-b", "train_live", session_paths[1]),
            AuditSession("dev", "dev_death", session_paths[2]),
            AuditSession("challenge", "challenge_false_positive", session_paths[3]),
        ),
        health_audit,
    )
    event_run = root / "event-run"
    replay_health_event_transitions(
        HEALTH_CONFIG, health_audit / "report.json", session_paths[2], event_run
    )
    statuses = (
        "WEAK_ANCHOR_RELATION_DIAGNOSTIC_FAILED",
        "DEATH_BANNER_CONSENSUS_DATA_INSUFFICIENT",
        "NATIVE_DEATH_CUE_PREFLIGHT_INSUFFICIENT",
        "NATIVE_DEATH_CUE_PREFLIGHT_DOMAIN_MISMATCH",
    )
    failures: list[Path] = []
    for index, status in enumerate(statuses):
        payload = {
            "status": status,
            "reward_allowed": False,
            "training_allowed": False,
        }
        if index < 3:
            payload["report_sha256"] = _object_sha256(payload)
        path = root / f"failure-{index}.json"
        path.write_text(json.dumps(payload))
        failures.append(path)
    package = root / "package"
    create_offline_cycle_package(delivery_evidence[2], event_run, failures, package)
    return delivery_evidence[2], event_run, failures, package


def _rewrite_self_bound(path: Path, field: str, updates: dict[str, object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    payload.pop(field)
    payload[field] = _object_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _hierarchical_reports(root: Path) -> list[Path]:
    definitions = (
        ("joystick-persistence-baseline-audit-v1", "DETERMINISTIC_DIRECTION_PERSISTENCE_EXACT"),
        ("movement-change-only-contract-report-v1", "CHANGE_ONLY_MOVEMENT_CONTRACT_PASSED"),
        ("movement-change-pilot-report-v1", "PASSED"),
        ("movement-change-replay-report-v1", "FAILED"),
        ("movement-change-position-v2-pilot-report-v1", "FAILED"),
        ("movement-change-geometry-replay-report-v1", "PASSED"),
    )
    reports: list[Path] = []
    for index, (schema, status) in enumerate(definitions):
        payload: dict[str, object] = {
            "schema_version": schema,
            "status": status,
            "input_commands_sent": 0,
            "video_test_opened": False,
            "device_input_allowed": False,
            "real_rgb_generalization_verified": False,
        }
        if schema == "movement-change-geometry-replay-report-v1":
            payload.update(
                {
                    "dataset_results": {
                        "train": {"correct": 256},
                        "dev": {"correct": 96},
                    },
                    "episodes_passed": 10,
                    "geometry_changes": 40,
                    "executor_keep_steps": 40,
                    "router_stop_steps": 40,
                    "arena_steps": 120,
                    "model_runs": 0,
                }
            )
        payload["report_sha256"] = _object_sha256(payload)
        path = root / f"hierarchical-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        reports.append(path)
    return reports


def test_r0_package_create_and_verify(
    delivery_evidence: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    interrupted, control, package = delivery_evidence
    result = verify_r0_package(package)
    assert result["status"] == "PASSED"
    assert result["delivery_grade"] == "R0_RULE_OFFLINE"
    assert result["transitions"] == 90
    assert result["terminal_transitions"] == 10
    assert result["frame_bundles"] == 100
    assert result["reward_total"] == 0.0
    assert result["input_commands_sent"] == 0
    assert result["promoted_checkpoint"] is None
    assert not tuple(package.glob("replay.sqlite3-*"))
    assert not tuple(package.rglob("*.safetensors"))
    with pytest.raises(ValueError, match="already exists"):
        create_r0_package(interrupted, control, package)
    with pytest.raises(ValueError, match="accepted pair"):
        create_r0_package(control, interrupted, tmp_path / "reversed")

    fresh = tmp_path / "fresh"
    created = create_r0_package(interrupted, control, fresh)
    assert created["transition_content_sha256"] == result["transition_content_sha256"]


@pytest.mark.parametrize(
    ("relative", "content"),
    (
        ("summary.json", b"{}"),
        ("manifest.json", b"{}"),
        ("frames/movement-rule-seed-0-episode-000-000.npz", b"corrupt"),
        ("replay.sqlite3", b"corrupt"),
    ),
)
def test_r0_package_rejects_tampered_files(
    delivery_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
    relative: str,
    content: bytes,
) -> None:
    package = delivery_evidence[2]
    tampered = tmp_path / relative.replace("/", "-")
    shutil.copytree(package, tampered)
    (tampered / relative).write_bytes(content)
    with pytest.raises(ValueError):
        verify_r0_package(tampered)


@pytest.mark.parametrize(
    "updates",
    (
        {"completed_episodes": 9},
        {"reward_total": 1.0},
        {"input_commands_sent": 1},
    ),
)
def test_r0_package_rejects_ineligible_source(
    delivery_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    source, control, _package = delivery_evidence
    changed = tmp_path / next(iter(updates))
    shutil.copytree(source, changed)
    _rewrite_self_bound(changed / "batch-summary.json", "summary_sha256", updates)
    with pytest.raises(ValueError, match="admission fields"):
        create_r0_package(changed, control, tmp_path / f"output-{next(iter(updates))}")


def test_package_cli_is_lazy_and_verify_only_is_read_only(
    delivery_evidence: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    package = delivery_evidence[2]
    forbidden = (
        "hok_agent.mobile_testbed",
        "hok_agent.shadow",
        "hok_agent.movement_mvp_train",
    )
    for name in forbidden:
        sys.modules.pop(name, None)
    before = (package / "manifest.json").read_bytes()
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "package",
                "--verify-only",
                "--output-dir",
                str(package),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (package / "manifest.json").read_bytes() == before
    assert all(name not in sys.modules for name in forbidden)


def test_offline_cycle_package_create_verify_and_boundaries(
    cycle_evidence: tuple[Path, Path, list[Path], Path], tmp_path: Path
) -> None:
    r0, event, failures, package = cycle_evidence
    result = verify_offline_cycle_package(package)
    assert result["status"] == "PASSED"
    assert result["delivery_grade"] == "R1_ENGINEERING_OFFLINE_ZERO_REWARD"
    assert result["r0_transitions"] == 90
    assert result["event_transitions"] == 9
    assert result["frames"] == 110
    assert result["event_counts"] == {"DEATH": 1, "RESPAWN": 1, "SELF_HP_DELTA": 0}
    assert result["reward_total"] == 0.0
    assert result["training_eligible_event_transitions"] == 0
    assert result["failure_reports"] == 4
    assert result["promoted_checkpoint"] is None
    assert not tuple(package.rglob("*.safetensors"))
    assert not tuple(package.rglob("replay.sqlite3-*"))
    with pytest.raises(ValueError, match="already exists"):
        create_offline_cycle_package(r0, event, failures, package)
    fresh = tmp_path / "fresh"
    created = create_offline_cycle_package(r0, event, failures, fresh)
    assert created["event_counts"] == result["event_counts"]


@pytest.mark.parametrize(
    "relative",
    (
        "summary.json",
        "manifest.json",
        "event/report.json",
        "event/replay.sqlite3",
        "event/frames/frame-000000.npz",
        "evidence/native-death-preflight-qa.json",
        "r0/summary.json",
    ),
)
def test_offline_cycle_package_rejects_tampering(
    cycle_evidence: tuple[Path, Path, list[Path], Path],
    tmp_path: Path,
    relative: str,
) -> None:
    package = cycle_evidence[3]
    tampered = tmp_path / relative.replace("/", "-")
    shutil.copytree(package, tampered)
    (tampered / relative).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify_offline_cycle_package(tampered)


def test_cycle_package_cli_verify_only_is_read_only(
    cycle_evidence: tuple[Path, Path, list[Path], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    package = cycle_evidence[3]
    before = (package / "manifest.json").read_bytes()
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "package-cycle",
                "--verify-only",
                "--output-dir",
                str(package),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["delivery_grade"] == "R1_ENGINEERING_OFFLINE_ZERO_REWARD"
    assert (package / "manifest.json").read_bytes() == before
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "package-cycle",
                "--verify-only",
                "--source-run",
                "unexpected",
                "--output-dir",
                str(package),
            ]
        )
        == 2
    )
    assert "accepts only" in capsys.readouterr().err


def test_hierarchical_rule_package_create_verify_and_boundaries(
    cycle_evidence: tuple[Path, Path, list[Path], Path], tmp_path: Path
) -> None:
    r1 = cycle_evidence[3]
    reports = _hierarchical_reports(tmp_path)
    package = tmp_path / "hierarchical-package"
    result = create_hierarchical_rule_package(r1, reports, package)
    assert result == verify_hierarchical_rule_package(package)
    assert result["delivery_grade"] == "R1_HIERARCHICAL_RULE_OFFLINE"
    assert result["dataset_correct"] == 352
    assert result["episodes_passed"] == 10
    assert result["geometry_changes"] == 40
    assert result["executor_keep_steps"] == 40
    assert result["router_stop_steps"] == 40
    assert result["arena_steps"] == 120
    assert result["model_runs"] == 0
    assert result["input_commands_sent"] == 0
    assert result["promoted_checkpoint"] is None
    assert result["evidence_reports"] == 6
    assert not tuple(package.rglob("*.safetensors"))
    assert not tuple(package.rglob("replay.sqlite3-*"))
    with pytest.raises(ValueError, match="already exists"):
        create_hierarchical_rule_package(r1, reports, package)


@pytest.mark.parametrize(
    "relative",
    (
        "summary.json",
        "manifest.json",
        "movement-evidence/geometry-replay-passed.json",
        "movement-evidence/neural-position-v2-failed.json",
        "r1/summary.json",
    ),
)
def test_hierarchical_rule_package_rejects_tampering(
    cycle_evidence: tuple[Path, Path, list[Path], Path],
    tmp_path: Path,
    relative: str,
) -> None:
    package = tmp_path / "source-package"
    create_hierarchical_rule_package(
        cycle_evidence[3], _hierarchical_reports(tmp_path), package
    )
    tampered = tmp_path / relative.replace("/", "-")
    shutil.copytree(package, tampered)
    (tampered / relative).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify_hierarchical_rule_package(tampered)


def test_hierarchical_rule_package_cli_verify_only_is_read_only(
    cycle_evidence: tuple[Path, Path, list[Path], Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = tmp_path / "hierarchical-package"
    create_hierarchical_rule_package(
        cycle_evidence[3], _hierarchical_reports(tmp_path), package
    )
    before = (package / "manifest.json").read_bytes()
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "package-hierarchical-rule",
                "--verify-only",
                "--output-dir",
                str(package),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["delivery_grade"] == "R1_HIERARCHICAL_RULE_OFFLINE"
    assert (package / "manifest.json").read_bytes() == before
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "package-hierarchical-rule",
                "--verify-only",
                "--source-run",
                "unexpected",
                "--output-dir",
                str(package),
            ]
        )
        == 2
    )
    assert "accepts only" in capsys.readouterr().err
