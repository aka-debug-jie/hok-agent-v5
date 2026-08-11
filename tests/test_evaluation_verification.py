from __future__ import annotations

import json
from pathlib import Path

from hok_agent.artifacts.hashing import sha256_file, sha256_json
from hok_agent.artifacts.verification import verify_artifact, write_hashed_json
from hok_agent.contracts.types import EvaluationReport, RunManifest

ROOT = Path(__file__).resolve().parents[1]
RUN_MANIFEST_SCHEMA = ROOT / "schemas" / "run_manifest.schema.json"
EVALUATION_REPORT_SCHEMA = ROOT / "schemas" / "evaluation_report.schema.json"
HASH = "sha256:" + ("0" * 64)


def _run_manifest_base(tmp_path: Path, *, artifact_path: str = "artifacts/episodes.jsonl") -> tuple[Path, Path]:
    config_path = tmp_path / "configs" / "run_smoke.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("kind: mock\n", encoding="utf-8")

    episodes_path = tmp_path / artifact_path
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text("episode row\n", encoding="utf-8")

    manifest = RunManifest(
        run_id="m0-mock-smoke",
        created_at_utc="2026-01-01T00:00:00Z",
        stage="M0",
        status="COMPLETED",
        formal=False,
        git_commit="abc1234",
        git_dirty=False,
        environment={
            "kind": "mock",
            "service_version": "mock-service-0.1.0",
            "identity_hash": HASH,
        },
        config_path="configs/run_smoke.yaml",
        config_hash=HASH,
        algorithm_name="deterministic_mock_smoke_policy",
        algorithm_version="1",
        seed_registry_hash=HASH,
        safety={
            "real_client_read_only": True,
            "commercial_client_actions_used": False,
            "privileged_state_enters_actor": False,
            "license_or_secret_written_to_git": False,
        },
        artifacts=(
            {
                "path": artifact_path,
                "sha256": sha256_file(episodes_path),
                "role": "episode_rows",
            },
        ),
    )

    manifest_path = tmp_path / "artifacts" / "run_manifest.json"
    write_hashed_json(manifest_path, manifest.to_dict(), self_hash_field="manifest_hash")
    return manifest_path, episodes_path


def test_verify_artifact_rejects_run_manifest_artifact_hash_mismatch(tmp_path: Path) -> None:
    manifest_path, _ = _run_manifest_base(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = document["artifacts"][0]
    artifact["sha256"] = "sha256:" + "1" * 64
    document["manifest_hash"] = sha256_json(document, exclude_key="manifest_hash")
    manifest_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = verify_artifact(manifest_path, RUN_MANIFEST_SCHEMA)

    assert not result.passed
    assert any("artifact sha256 mismatch" in message for message in result.errors)


def test_verify_artifact_rejects_run_manifest_path_traversal(tmp_path: Path) -> None:
    manifest_path, _ = _run_manifest_base(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["artifacts"][0]["path"] = "../outside/episodes.jsonl"
    document["manifest_hash"] = sha256_json(document, exclude_key="manifest_hash")
    manifest_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "outside" / "episodes.jsonl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("episode row\n", encoding="utf-8")

    result = verify_artifact(manifest_path, RUN_MANIFEST_SCHEMA, repo_root=tmp_path)

    assert not result.passed
    assert any("not safe" in message for message in result.errors)


def test_verify_artifact_rejects_mock_report_gate_pass_mismatch(tmp_path: Path) -> None:
    report = EvaluationReport(
        report_id="report-1",
        created_at_utc="2026-01-01T00:00:00Z",
        suite_id="m0-mock-smoke",
        suite_version=1,
        suite_config_hash=HASH,
        checkpoint_path="builtin://deterministic_mock_smoke_policy",
        checkpoint_hash=HASH,
        training_run_id="M0-NO-TRAINING",
        environment={"kind": "mock", "identity_hash": HASH},
        episode_count=1,
        episode_completed=1,
        details_artifact="artifacts/episodes.jsonl",
        metrics={"environment_kind": "mock"},
        engineering={
            "protocol_errors": 0,
            "action_decode_errors": 0,
            "illegal_executed_actions": 0,
            "replay_errors": 0,
            "nonfinite_events": 0,
        },
        gate_checks={"mock_only": True},
        disposition="DIAGNOSTIC_ONLY",
    )
    report_path = tmp_path / "evaluation_report.json"
    write_hashed_json(
        report_path,
        report.to_dict(),
        self_hash_field="report_hash",
        schema_path=EVALUATION_REPORT_SCHEMA,
    )

    document = json.loads(report_path.read_text(encoding="utf-8"))
    document["gate"]["checks"]["mock_only"] = False
    document["report_hash"] = sha256_json(document, exclude_key="report_hash")
    report_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = verify_artifact(report_path, EVALUATION_REPORT_SCHEMA)

    assert not result.passed
    assert any("gate.pass must equal all checks are true" in message for message in result.errors)


def test_verify_artifact_rejects_mock_report_disposition_non_diagnostic(tmp_path: Path) -> None:
    report = EvaluationReport(
        report_id="report-2",
        created_at_utc="2026-01-01T00:00:00Z",
        suite_id="m0-mock-smoke",
        suite_version=1,
        suite_config_hash=HASH,
        checkpoint_path="builtin://deterministic_mock_smoke_policy",
        checkpoint_hash=HASH,
        training_run_id="M0-NO-TRAINING",
        environment={"kind": "mock", "identity_hash": HASH},
        episode_count=1,
        episode_completed=1,
        details_artifact="artifacts/episodes.jsonl",
        metrics={"environment_kind": "mock"},
        engineering={
            "protocol_errors": 0,
            "action_decode_errors": 0,
            "illegal_executed_actions": 0,
            "replay_errors": 0,
            "nonfinite_events": 0,
        },
        gate_checks={"mock_only": True},
        disposition="PROMOTED_ACTIVE",
    )
    report_path = tmp_path / "evaluation_report.json"
    write_hashed_json(report_path, report.to_dict(), self_hash_field="report_hash")

    result = verify_artifact(report_path, EVALUATION_REPORT_SCHEMA)

    assert not result.passed
    assert any("DIAGNOSTIC_ONLY" in message for message in result.errors)
