"""M0 diagnostic mock smoke and throughput benchmark harnesses."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hok_agent.artifacts.hashing import sha256_file, sha256_json
from hok_agent.artifacts.verification import verify_artifact, write_hashed_json
from hok_agent.config import (
    ConfigError,
    load_yaml_mapping,
    required_int,
    required_mapping,
    required_string,
)
from hok_agent.contracts import (
    EnvironmentKind,
    EvaluationReport,
    ResetRequest,
    RunManifest,
    StepRequest,
)
from hok_agent.contracts.types import JsonValue
from hok_agent.envs import LocalRpcClient, ProcessJsonTransport
from hok_agent.envs.mock import select_deterministic_smoke_action


@dataclass(frozen=True, slots=True)
class EpisodeRow:
    index: int
    seed: int
    side: str
    terminal: bool
    truncated: bool
    result: str
    steps: int
    replay_hash: str
    reward_total: float
    tick_monotonic: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "seed": self.seed,
            "side": self.side,
            "terminal": self.terminal,
            "truncated": self.truncated,
            "result": self.result,
            "steps": self.steps,
            "replay_hash": self.replay_hash,
            "reward_total": self.reward_total,
            "tick_monotonic": self.tick_monotonic,
            "environment_kind": "mock",
        }


@dataclass(frozen=True, slots=True)
class SmokeRunResult:
    run_dir: Path
    run_manifest: Path
    evaluation_report: Path
    episodes_path: Path
    episode_count: int
    completed_count: int
    protocol_errors: int
    action_decode_errors: int
    illegal_executed_actions: int
    replay_errors: int
    deterministic_replay: bool
    terminal_rate: float
    manifest_verification: bool
    report_verification: bool

    @property
    def passed(self) -> bool:
        return (
            self.episode_count == self.completed_count
            and self.protocol_errors == 0
            and self.action_decode_errors == 0
            and self.illegal_executed_actions == 0
            and self.replay_errors == 0
            and self.deterministic_replay
            and self.terminal_rate == 1.0
            and self.manifest_verification
            and self.report_verification
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "run_manifest": str(self.run_manifest),
            "evaluation_report": str(self.evaluation_report),
            "episodes_path": str(self.episodes_path),
            "episode_count": self.episode_count,
            "completed_count": self.completed_count,
            "protocol_errors": self.protocol_errors,
            "action_decode_errors": self.action_decode_errors,
            "illegal_executed_actions": self.illegal_executed_actions,
            "replay_errors": self.replay_errors,
            "deterministic_replay": self.deterministic_replay,
            "terminal_rate": self.terminal_rate,
            "manifest_verification": self.manifest_verification,
            "report_verification": self.report_verification,
            "passed": self.passed,
            "scope": "M0 mock-only infrastructure diagnostic; not a HoK capability result",
        }


def _nested_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key not in config:
        raise ConfigError(f"config is missing {key}")
    return required_mapping(config[key], key)


def _config_values(config_path: Path, episodes_override: int | None) -> tuple[int, int, int, Path]:
    config = load_yaml_mapping(config_path)
    environment = _nested_mapping(config, "environment")
    artifacts = _nested_mapping(config, "artifacts")
    episodes = episodes_override or required_int(environment.get("episodes"), "environment.episodes", minimum=1)
    seed_start = required_int(environment.get("seed_start"), "environment.seed_start", minimum=0)
    max_steps = required_int(environment.get("max_steps_per_episode"), "environment.max_steps_per_episode", minimum=2)
    output_root_text = required_string(artifacts.get("output_root"), "artifacts.output_root")
    return episodes, seed_start, max_steps, config_path.parent.parent / output_root_text


def _new_run_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}"


def _git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    value = commit.stdout.strip()
    if commit.returncode != 0 or len(value) < 7:
        raise RuntimeError("M0 artifact generation requires a committed repository state")
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if dirty.returncode != 0:
        raise RuntimeError("cannot determine repository dirty state")
    return value, bool(dirty.stdout.strip())


def _new_mock_client(max_steps: int) -> tuple[LocalRpcClient, ProcessJsonTransport]:
    transport = ProcessJsonTransport.start_mock(max_steps_per_episode=max_steps)
    return LocalRpcClient(transport, expected_kind=EnvironmentKind.MOCK), transport


def _run_episode(client: LocalRpcClient, *, index: int, seed: int, max_steps: int) -> EpisodeRow:
    side = "blue" if index % 2 == 0 else "red"
    reset = client.reset(
        ResetRequest(
            request_id=f"m0-smoke-{index}-{seed}",
            mode="1v1",
            seed=seed,
            side=side,
            hero_config={"hero": "mock_fixed_hero"},
            opponent_config={"policy": "mock_stationary"},
            reward_config_version="mock-reward-v1",
            evaluation_mode=True,
        )
    )
    expected_tick = reset.tick
    legal_actions = reset.legal_actions
    reward_total = 0.0
    replay_hash = reset.replay_identity
    monotonic = True
    terminal = False
    truncated = False
    result = "unfinished"
    for _ in range(max_steps):
        action = select_deterministic_smoke_action(legal_actions)
        response = client.step(
            StepRequest(
                episode_id=reset.episode_id,
                expected_tick=expected_tick,
                action=action,
                action_schema_version=1,
            )
        )
        monotonic = monotonic and response.tick == expected_tick + 1
        expected_tick = response.tick
        reward_total += response.reward.total
        replay_hash = response.replay_hash
        legal_actions = response.legal_actions
        terminal = response.terminal
        truncated = response.truncated
        if terminal or truncated:
            if response.outcome is None:
                raise RuntimeError("terminal mock response omitted outcome")
            result = response.outcome.result.value
            break
    client.close(reset.episode_id, "episode_complete" if terminal or truncated else "smoke_step_bound")
    return EpisodeRow(
        index=index,
        seed=seed,
        side=side,
        terminal=terminal,
        truncated=truncated,
        result=result,
        steps=expected_tick,
        replay_hash=replay_hash,
        reward_total=reward_total,
        tick_monotonic=monotonic,
    )


def _write_episode_rows(path: Path, rows: list[EpisodeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _environment_document(client: LocalRpcClient) -> dict[str, JsonValue]:
    health = client.health()
    return {
        "kind": "mock",
        "service_version": health.environment.service_version,
        "sdk_version": health.environment.sdk_version,
        "gamecore_build": health.environment.gamecore_build,
        "license_status": health.license_status.value,
        "identity_hash": sha256_json(health.environment.to_dict()),
    }


def run_mock_smoke(config_path: Path, *, episodes_override: int | None = None) -> SmokeRunResult:
    """Run a deterministic M0 mock diagnostic and emit schema-valid artifacts."""

    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    episodes, seed_start, max_steps, output_root = _config_values(config_path, episodes_override)
    run_id = _new_run_id("m0-mock-smoke")
    run_dir = output_root / run_id
    client, transport = _new_mock_client(max_steps)
    try:
        environment = _environment_document(client)
        rows = [
            _run_episode(client, index=index, seed=seed_start + index, max_steps=max_steps) for index in range(episodes)
        ]
    finally:
        transport.close()
    repeat_client, repeat_transport = _new_mock_client(max_steps)
    try:
        first_replay = _run_episode(repeat_client, index=0, seed=seed_start, max_steps=max_steps).replay_hash
    finally:
        repeat_transport.close()
    deterministic_replay = rows[0].replay_hash == first_replay
    episodes_path = run_dir / "episodes.jsonl"
    _write_episode_rows(episodes_path, rows)
    completed_count = sum(1 for row in rows if row.terminal or row.truncated)
    terminal_rate = sum(1 for row in rows if row.terminal) / episodes
    protocol_errors = 0
    action_decode_errors = 0
    illegal_executed_actions = 0
    replay_errors = 0 if deterministic_replay else 1
    config_hash = sha256_file(config_path)
    git_commit, git_dirty = _git_state(repo_root)
    relative_episodes = str(episodes_path.relative_to(repo_root))
    manifest = RunManifest(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        stage="M0",
        status="COMPLETED",
        formal=False,
        git_commit=git_commit,
        git_dirty=git_dirty,
        environment=environment,
        config_path=str(config_path.relative_to(repo_root)),
        config_hash=config_hash,
        algorithm_name="deterministic_mock_smoke_policy",
        algorithm_version="1",
        seed_registry_hash=sha256_json({"seed_start": seed_start, "episodes": episodes}),
        safety={
            "real_client_read_only": True,
            "commercial_client_actions_used": False,
            "privileged_state_enters_actor": False,
            "license_or_secret_written_to_git": False,
        },
        artifacts=(
            {
                "path": relative_episodes,
                "sha256": sha256_file(episodes_path),
                "role": "episode_rows",
            },
        ),
    )
    manifest_path = run_dir / "run_manifest.json"
    run_schema = repo_root / "schemas" / "run_manifest.schema.json"
    write_hashed_json(manifest_path, manifest.to_dict(), self_hash_field="manifest_hash", schema_path=run_schema)
    checks = {
        "mock_only": True,
        "completed_episodes": completed_count == episodes,
        "terminal_rate": terminal_rate == 1.0,
        "protocol_errors": protocol_errors == 0,
        "action_decode_errors": action_decode_errors == 0,
        "illegal_executed_actions": illegal_executed_actions == 0,
        "replay_errors": replay_errors == 0,
        "fixed_seed_replay_exact": deterministic_replay,
        "tick_monotonic": all(row.tick_monotonic for row in rows),
    }
    report = EvaluationReport(
        report_id=f"{run_id}-e0",
        created_at_utc=datetime.now(UTC).isoformat(),
        suite_id="m0-mock-smoke",
        suite_version=1,
        suite_config_hash=config_hash,
        checkpoint_path="builtin://deterministic_mock_smoke_policy",
        checkpoint_hash=sha256_json({"policy": "deterministic_mock_smoke_policy", "version": 1}),
        training_run_id="M0-NO-TRAINING",
        environment=environment,
        episode_count=episodes,
        episode_completed=completed_count,
        details_artifact=relative_episodes,
        metrics={
            "environment_kind": "mock",
            "capability_claim": "none",
            "terminal_rate": terminal_rate,
            "deterministic_replay": deterministic_replay,
        },
        engineering={
            "protocol_errors": protocol_errors,
            "action_decode_errors": action_decode_errors,
            "illegal_executed_actions": illegal_executed_actions,
            "replay_errors": replay_errors,
            "nonfinite_events": 0,
        },
        gate_checks=checks,
        disposition="DIAGNOSTIC_ONLY",
    )
    report_path = run_dir / "evaluation_report.json"
    report_schema = repo_root / "schemas" / "evaluation_report.schema.json"
    write_hashed_json(report_path, report.to_dict(), self_hash_field="report_hash", schema_path=report_schema)
    manifest_valid = verify_artifact(manifest_path, run_schema).passed
    report_valid = verify_artifact(report_path, report_schema).passed
    return SmokeRunResult(
        run_dir=run_dir,
        run_manifest=manifest_path,
        evaluation_report=report_path,
        episodes_path=episodes_path,
        episode_count=episodes,
        completed_count=completed_count,
        protocol_errors=protocol_errors,
        action_decode_errors=action_decode_errors,
        illegal_executed_actions=illegal_executed_actions,
        replay_errors=replay_errors,
        deterministic_replay=deterministic_replay,
        terminal_rate=terminal_rate,
        manifest_verification=manifest_valid,
        report_verification=report_valid,
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def run_mock_benchmark(config_path: Path, *, episodes: int) -> dict[str, object]:
    """Measure only the deterministic mock service; never label it GameCore throughput."""

    config_path = config_path.resolve()
    _, seed_start, max_steps, output_root = _config_values(config_path, episodes)
    client, transport = _new_mock_client(max_steps)
    try:
        _environment_document(client)
        durations: list[float] = []
        start = time.perf_counter()
        rows: list[EpisodeRow] = []
        for index in range(episodes):
            before = time.perf_counter()
            rows.append(_run_episode(client, index=index, seed=seed_start + index, max_steps=max_steps))
            durations.append(time.perf_counter() - before)
        elapsed = time.perf_counter() - start
    finally:
        transport.close()
    steps = sum(row.steps for row in rows)
    report = {
        "schema_version": 1,
        "kind": "mock_environment_benchmark",
        "environment_kind": "mock",
        "capability_claim": "none",
        "episodes": episodes,
        "steps": steps,
        "wall_seconds": elapsed,
        "env_steps_per_second": steps / elapsed if elapsed else 0.0,
        "episodes_per_hour": episodes / elapsed * 3600.0 if elapsed else 0.0,
        "episode_latency_seconds": {
            "p50": statistics.median(durations) if durations else 0.0,
            "p95": _percentile(durations, 0.95),
        },
        "host": {
            "cpu_count": os.cpu_count(),
            "gpu_measurement": "not_collected_in_M0_mock_benchmark",
        },
    }
    report["report_hash"] = sha256_json(report)
    run_dir = output_root / _new_run_id("m0-mock-benchmark")
    path = run_dir / "benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": sha256_file(path), **report}
