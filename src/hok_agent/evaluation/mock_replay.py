"""Public-only transition recording and fresh-process replay for the deterministic mock."""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hok_agent.artifacts.hashing import sha256_file, sha256_json
from hok_agent.artifacts.verification import verify_artifact
from hok_agent.contracts import (
    EnvironmentKind,
    EpisodeOutcome,
    FactorizedAction,
    MockPublicOutcome,
    MockPublicProfile,
    MockPublicReplay,
    MockPublicTransition,
    PublicObservation,
    ResetRequest,
    StepRequest,
)
from hok_agent.envs import LocalRpcClient, ProcessJsonTransport
from hok_agent.envs.mock import select_deterministic_smoke_action
from hok_agent.envs.rpc import RpcError

_PROFILE_VERSION = "mock-public-profile-v1"
_MODE = "1v1"
_SIDES = frozenset({"blue", "red"})


class MockReplayExecutionError(ValueError):
    """Raised when a trace cannot be safely recorded or re-executed."""


@dataclass(frozen=True, slots=True)
class MockReplayRecordResult:
    path: Path
    artifact_hash: str
    transition_count: int
    service_pid: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "artifact_hash": self.artifact_hash,
            "transition_count": self.transition_count,
            "service_pid": self.service_pid,
            "environment_kind": "mock",
            "formal": False,
            "capability_claim": "none",
        }


@dataclass(frozen=True, slots=True)
class MockReplayVerificationResult:
    path: Path
    artifact_hash: str
    transition_count: int
    service_pid: int
    input_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "artifact_hash": self.artifact_hash,
            "transition_count": self.transition_count,
            "service_pid": self.service_pid,
            "input_sha256": self.input_sha256,
            "environment_kind": "mock",
            "formal": False,
            "capability_claim": "none",
            "fresh_process_exact": True,
        }


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[3] / "schemas" / "mock_public_replay.schema.json"


def _validate_inputs(*, seed: int, side: str, max_steps_per_episode: int) -> None:
    if seed < 0:
        raise MockReplayExecutionError("seed must be non-negative")
    if side not in _SIDES:
        raise MockReplayExecutionError("side must be blue or red")
    if max_steps_per_episode < 2:
        raise MockReplayExecutionError("max_steps_per_episode must be at least 2")


def _fixed_reset_request(seed: int, side: str) -> ResetRequest:
    return ResetRequest(
        request_id=f"mock-public-trace-{seed}-{side}",
        mode=_MODE,
        seed=seed,
        side=side,
        hero_config={"hero": "mock_fixed_hero"},
        opponent_config={"policy": "mock_stationary"},
        reward_config_version="mock-reward-v1",
        evaluation_mode=True,
    )


def _identity_hashes(client: LocalRpcClient) -> tuple[str, str, str]:
    health = client.health()
    return (
        sha256_json(health.environment.to_dict()),
        sha256_json({"protocol_version": health.protocol_version}),
        health.environment.mapping_hash,
    )


def _public_observation_hash(observation: PublicObservation) -> str:
    return sha256_json(observation.to_dict())


def _public_outcome(outcome: EpisodeOutcome | None) -> MockPublicOutcome | None:
    if outcome is None:
        return None
    return MockPublicOutcome(result=outcome.result.value)


def _require_public_target(action: FactorizedAction, observation: PublicObservation) -> None:
    if action.target_key is None:
        return
    public_keys = {entity.public_key for entity in observation.visible_entities}
    if action.target_key not in public_keys:
        raise MockReplayExecutionError("action target_key is not present in the fresh public observation")


def _transition(
    *,
    sequence: int,
    previous_row_hash: str | None,
    input_tick: int,
    pre_observation: PublicObservation,
    action: FactorizedAction,
    output_tick: int,
    post_observation: PublicObservation,
    terminal: bool,
    truncated: bool,
    outcome: MockPublicOutcome | None,
) -> MockPublicTransition:
    document: dict[str, object] = {
        "sequence": sequence,
        "previous_row_hash": previous_row_hash,
        "input_tick": input_tick,
        "pre_public_observation_hash": _public_observation_hash(pre_observation),
        "action": action.to_dict(),
        "output_tick": output_tick,
        "post_public_observation_hash": _public_observation_hash(post_observation),
        "terminal": terminal,
        "truncated": truncated,
        "outcome": None if outcome is None else outcome.to_dict(),
    }
    document["row_hash"] = sha256_json(document)
    return MockPublicTransition.from_dict(document)


def _build_replay(
    *,
    service_identity_hash: str,
    protocol_identity_hash: str,
    mapping_identity_hash: str,
    max_steps_per_episode: int,
    seed: int,
    side: str,
    episode_id: str,
    transitions: tuple[MockPublicTransition, ...],
) -> MockPublicReplay:
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "mock_public_transition_replay",
        "environment_kind": "mock",
        "formal": False,
        "capability_claim": "none",
        "service_identity_hash": service_identity_hash,
        "protocol_identity_hash": protocol_identity_hash,
        "mapping_identity_hash": mapping_identity_hash,
        "mock_profile": MockPublicProfile(
            version=_PROFILE_VERSION,
            max_steps_per_episode=max_steps_per_episode,
        ).to_dict(),
        "seed": seed,
        "side": side,
        "mode": _MODE,
        "episode_id": episode_id,
        "transition_count": len(transitions),
        "transitions": [transition.to_dict() for transition in transitions],
        "chain_head": transitions[-1].row_hash,
    }
    document["artifact_hash"] = sha256_json(document)
    return MockPublicReplay.from_dict(document)


def _write_exclusive_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise MockReplayExecutionError(f"refusing to overwrite existing trace: {path}") from error


def _read_replay(path: Path) -> MockPublicReplay:
    if not path.is_file():
        raise MockReplayExecutionError(f"trace does not exist: {path}")
    verification = verify_artifact(path, _schema_path())
    if not verification.passed:
        raise MockReplayExecutionError("trace schema/self-hash verification failed: " + "; ".join(verification.errors))
    try:
        raw_document = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise MockReplayExecutionError(f"cannot read trace: {error}") from error
    return MockPublicReplay.from_dict(raw_document)


def _close_quietly(client: LocalRpcClient | None, episode_id: str | None, stop_reason: str) -> None:
    if client is None or episode_id is None:
        return
    with suppress(RpcError):
        client.close(episode_id, stop_reason)


def record_mock_public_replay(
    output_path: Path,
    *,
    seed: int = 101,
    side: str = "blue",
    max_steps_per_episode: int = 12,
) -> MockReplayRecordResult:
    """Record a complete diagnostic trace through a newly spawned mock service."""

    _validate_inputs(seed=seed, side=side, max_steps_per_episode=max_steps_per_episode)
    if output_path.exists():
        raise MockReplayExecutionError(f"refusing to overwrite existing trace: {output_path}")

    transport = ProcessJsonTransport.start_mock(max_steps_per_episode=max_steps_per_episode)
    service_pid = transport.service_pid
    if service_pid is None:
        transport.close()
        raise MockReplayExecutionError("spawned mock service has no process identifier")
    client: LocalRpcClient | None = None
    episode_id: str | None = None
    try:
        client = LocalRpcClient(transport, expected_kind=EnvironmentKind.MOCK)
        service_identity_hash, protocol_identity_hash, mapping_identity_hash = _identity_hashes(client)
        reset = client.reset(_fixed_reset_request(seed, side))
        episode_id = reset.episode_id
        expected_episode_id = f"mock-{seed}-{side}-1"
        if episode_id != expected_episode_id:
            raise MockReplayExecutionError("mock episode_id does not match fixed public profile")

        observation = reset.observation
        legal_actions = reset.legal_actions
        tick = reset.tick
        previous_row_hash: str | None = None
        transitions: list[MockPublicTransition] = []
        for sequence in range(max_steps_per_episode):
            action = select_deterministic_smoke_action(legal_actions)
            _require_public_target(action, observation)
            response = client.step(
                StepRequest(
                    episode_id=episode_id,
                    expected_tick=tick,
                    action=action,
                    action_schema_version=1,
                )
            )
            transition = _transition(
                sequence=sequence,
                previous_row_hash=previous_row_hash,
                input_tick=tick,
                pre_observation=observation,
                action=action,
                output_tick=response.tick,
                post_observation=response.observation,
                terminal=response.terminal,
                truncated=response.truncated,
                outcome=_public_outcome(response.outcome),
            )
            transitions.append(transition)
            previous_row_hash = transition.row_hash
            if response.terminal or response.truncated:
                break
            observation = response.observation
            legal_actions = response.legal_actions
            tick = response.tick
        else:
            raise MockReplayExecutionError("mock trace did not reach terminal or truncated state")

        replay = _build_replay(
            service_identity_hash=service_identity_hash,
            protocol_identity_hash=protocol_identity_hash,
            mapping_identity_hash=mapping_identity_hash,
            max_steps_per_episode=max_steps_per_episode,
            seed=seed,
            side=side,
            episode_id=episode_id,
            transitions=tuple(transitions),
        )
        _write_exclusive_json(output_path, replay.to_dict())
        verification = verify_artifact(output_path, _schema_path())
        if not verification.passed:
            raise MockReplayExecutionError("recorded trace failed schema/self-hash verification")
        return MockReplayRecordResult(
            path=output_path,
            artifact_hash=replay.artifact_hash,
            transition_count=replay.transition_count,
            service_pid=service_pid,
        )
    finally:
        _close_quietly(client, episode_id, "mock_public_trace_recorded")
        transport.close()


def _require_equal(field: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise MockReplayExecutionError(f"fresh replay mismatch for {field}")


def verify_mock_public_replay(path: Path) -> MockReplayVerificationResult:
    """Fail closed unless a trace exactly replays in a fresh owned mock process."""

    input_sha256 = sha256_file(path)
    replay = _read_replay(path)
    max_steps_per_episode = replay.mock_profile.max_steps_per_episode
    transport = ProcessJsonTransport.start_mock(max_steps_per_episode=max_steps_per_episode)
    service_pid = transport.service_pid
    if service_pid is None:
        transport.close()
        raise MockReplayExecutionError("spawned mock service has no process identifier")
    client: LocalRpcClient | None = None
    episode_id: str | None = None
    try:
        client = LocalRpcClient(transport, expected_kind=EnvironmentKind.MOCK)
        _require_equal("service_identity_hash", _identity_hashes(client)[0], replay.service_identity_hash)
        _require_equal("protocol_identity_hash", _identity_hashes(client)[1], replay.protocol_identity_hash)
        _require_equal("mapping_identity_hash", _identity_hashes(client)[2], replay.mapping_identity_hash)

        reset = client.reset(_fixed_reset_request(replay.seed, replay.side))
        episode_id = reset.episode_id
        _require_equal("episode_id", episode_id, replay.episode_id)
        observation = reset.observation
        legal_actions = reset.legal_actions
        tick = reset.tick

        for index, transition in enumerate(replay.transitions):
            _require_equal("transition.sequence", index, transition.sequence)
            _require_equal("transition.input_tick", tick, transition.input_tick)
            _require_equal(
                "transition.pre_public_observation_hash",
                _public_observation_hash(observation),
                transition.pre_public_observation_hash,
            )
            _require_public_target(transition.action, observation)
            if not legal_actions.contains(transition.action):
                raise MockReplayExecutionError("recorded action is not in the fresh legal action set")
            response = client.step(
                StepRequest(
                    episode_id=episode_id,
                    expected_tick=tick,
                    action=transition.action,
                    action_schema_version=1,
                )
            )
            _require_equal("transition.output_tick", response.tick, transition.output_tick)
            _require_equal(
                "transition.post_public_observation_hash",
                _public_observation_hash(response.observation),
                transition.post_public_observation_hash,
            )
            _require_equal("transition.terminal", response.terminal, transition.terminal)
            _require_equal("transition.truncated", response.truncated, transition.truncated)
            _require_equal("transition.outcome", _public_outcome(response.outcome), transition.outcome)
            if response.terminal or response.truncated:
                if index != replay.transition_count - 1:
                    raise MockReplayExecutionError("trace contains transitions after a terminal state")
                break
            observation = response.observation
            legal_actions = response.legal_actions
            tick = response.tick
        else:
            raise MockReplayExecutionError("fresh replay did not reach terminal or truncated state")
    finally:
        _close_quietly(client, episode_id, "mock_public_trace_verified")
        transport.close()

    if sha256_file(path) != input_sha256:
        raise MockReplayExecutionError("verifier modified the input trace")
    return MockReplayVerificationResult(
        path=path,
        artifact_hash=replay.artifact_hash,
        transition_count=replay.transition_count,
        service_pid=service_pid,
        input_sha256=input_sha256,
    )
