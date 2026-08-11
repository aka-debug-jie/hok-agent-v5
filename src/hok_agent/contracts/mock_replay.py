"""Public-only mock replay transition contracts and validators."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from hok_agent.artifacts.hashing import sha256_json
from hok_agent.contracts.types import CONTRACT_VERSION, ContractValidationError, EnvironmentKind, FactorizedAction

ARTIFACT_KIND = "mock_public_transition_replay"
CAPABILITY_CLAIM = "none"
PROFILE_VERSION = "mock-public-profile-v1"
SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class MockReplayValidationError(ContractValidationError):
    """Specialized exception name for mock public replay validation."""


DENYLIST_KEYS = frozenset(
    {
        "reward",
        "legal",
        "teacher",
        "truth",
        "privileged",
        "entity_id",
        "replay_hash",
        "replay_identity",
        "abs",
        "upstream",
        "raw_observations",
        "reward_components",
        "legal_actions",
        "legal_mask",
        "teacher_action",
        "truth_state",
        "privileged_state",
        "entity_ids",
        "upstream_identity",
    }
)


def _is_forbidden_key(normalized: str) -> bool:
    return any(
        normalized == key or normalized.startswith(f"{key}_")
        for key in DENYLIST_KEYS
    )


def _normalize_key(value: str) -> str:
    return value.casefold().replace("-", "_").replace(" ", "_")


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MockReplayValidationError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _require_keys(value: Mapping[str, object], allowed: set[str], required: set[str], field: str) -> None:
    unexpected = set(value).difference(allowed)
    missing = required.difference(value)
    if unexpected:
        raise MockReplayValidationError(f"{field} contains unexpected keys: {sorted(unexpected)}")
    if missing:
        raise MockReplayValidationError(f"{field} is missing keys: {sorted(missing)}")


def _string(value: object, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise MockReplayValidationError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MockReplayValidationError(f"{field} must be an integer")
    if value < minimum:
        raise MockReplayValidationError(f"{field} must be >= {minimum}")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise MockReplayValidationError(f"{field} must be a boolean")
    return value


def _nullable_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    text = _string(value, field)
    if not re.fullmatch(SHA256_PATTERN, text):
        raise MockReplayValidationError(f"{field} must be a sha256:<64 lowercase hex> string")
    return text


def _sha256(value: object, field: str) -> str:
    text = _nullable_sha256(value, field)
    if text is None:
        raise MockReplayValidationError(f"{field} must be a sha256:<64 lowercase hex> string")
    return text


def _require_public_keys_only(value: object, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MockReplayValidationError(f"{field} contains a non-string key")
            normalized = _normalize_key(key)
            if _is_forbidden_key(normalized):
                raise MockReplayValidationError(f"{field}.{key} is not a public-only key")
            _require_public_keys_only(item, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_public_keys_only(item, f"{field}[{index}]")


def _transition_hash_payload(transition: Mapping[str, object]) -> str:
    payload = dict(transition)
    payload.pop("row_hash", None)
    return sha256_json(payload)


def _compute_artifact_hash(document: Mapping[str, object]) -> str:
    return sha256_json(document)


def _assert_row_chain(transitions: tuple[MockPublicTransition, ...]) -> None:
    previous: str | None = None
    for expected_sequence, transition in enumerate(transitions):
        if transition.sequence != expected_sequence:
            raise MockReplayValidationError(
                f"transition sequence must be contiguous from 0, expected {expected_sequence}"
            )
        if transition.previous_row_hash != previous:
            raise MockReplayValidationError("transition.previous_row_hash mismatch")
        transition_hash = _transition_hash_payload(transition.to_dict())
        if transition.row_hash != transition_hash:
            raise MockReplayValidationError("transition row_hash mismatch")
        previous = transition.row_hash


@dataclass(frozen=True, slots=True)
class MockPublicOutcome:
    """Public-only outcome payload."""

    result: str

    def __post_init__(self) -> None:
        if self.result not in {"win", "loss", "draw"}:
            raise MockReplayValidationError("outcome.result must be win, loss, or draw")

    def to_dict(self) -> dict[str, object]:
        return {"result": self.result}

    @classmethod
    def from_dict(cls, value: object) -> MockPublicOutcome:
        mapped = _require_mapping(value, "outcome")
        allowed = {"result"}
        _require_keys(mapped, allowed, allowed, "outcome")
        _require_public_keys_only(mapped, "outcome")
        return cls(result=_string(mapped["result"], "outcome.result"))


@dataclass(frozen=True, slots=True)
class MockPublicProfile:
    """Mock profile identity carried by the top-level artifact."""

    version: str
    max_steps_per_episode: int

    def __post_init__(self) -> None:
        if self.version != PROFILE_VERSION:
            raise MockReplayValidationError("mock_profile.version must be mock-public-profile-v1")
        if self.max_steps_per_episode < 2:
            raise MockReplayValidationError("mock_profile.max_steps_per_episode must be >= 2")

    def to_dict(self) -> dict[str, object]:
        return {"version": self.version, "max_steps_per_episode": self.max_steps_per_episode}

    @classmethod
    def from_dict(cls, value: object) -> MockPublicProfile:
        mapped = _require_mapping(value, "mock_profile")
        allowed = {"version", "max_steps_per_episode"}
        _require_keys(mapped, allowed, allowed, "mock_profile")
        _require_public_keys_only(mapped, "mock_profile")
        return cls(
            version=_string(mapped["version"], "mock_profile.version"),
            max_steps_per_episode=_integer(
                mapped["max_steps_per_episode"],
                "mock_profile.max_steps_per_episode",
                minimum=2,
            ),
        )


@dataclass(frozen=True, slots=True)
class MockPublicTransition:
    sequence: int
    previous_row_hash: str | None
    input_tick: int
    pre_public_observation_hash: str
    action: FactorizedAction
    output_tick: int
    post_public_observation_hash: str
    terminal: bool
    truncated: bool
    outcome: MockPublicOutcome | None
    row_hash: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise MockReplayValidationError("transition.sequence must be non-negative")
        if self.input_tick < 0 or self.output_tick < 0:
            raise MockReplayValidationError("transition.tick must be >= 0")
        if self.output_tick != self.input_tick + 1:
            raise MockReplayValidationError("transition.output_tick must equal input_tick + 1")
        if self.previous_row_hash is not None:
            _sha256(self.previous_row_hash, "transition.previous_row_hash")
        _sha256(self.pre_public_observation_hash, "transition.pre_public_observation_hash")
        _sha256(self.post_public_observation_hash, "transition.post_public_observation_hash")
        _sha256(self.row_hash, "transition.row_hash")
        if self.terminal and self.truncated:
            raise MockReplayValidationError("transition cannot be terminal and truncated simultaneously")
        if self.terminal or self.truncated:
            if self.outcome is None:
                raise MockReplayValidationError("terminal/truncated transition requires outcome")
        elif self.outcome is not None:
            raise MockReplayValidationError("non-terminal transition cannot include outcome")

    def to_dict(self, include_row_hash: bool = True) -> dict[str, object]:
        data: dict[str, object] = {
            "sequence": self.sequence,
            "previous_row_hash": self.previous_row_hash,
            "input_tick": self.input_tick,
            "pre_public_observation_hash": self.pre_public_observation_hash,
            "action": self.action.to_dict(),
            "output_tick": self.output_tick,
            "post_public_observation_hash": self.post_public_observation_hash,
            "terminal": self.terminal,
            "truncated": self.truncated,
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
            "row_hash": self.row_hash,
        }
        if include_row_hash:
            return data
        data.pop("row_hash")
        return data

    @classmethod
    def from_dict(cls, value: object) -> MockPublicTransition:
        mapped = _require_mapping(value, "transition")
        allowed = {
            "sequence",
            "previous_row_hash",
            "input_tick",
            "pre_public_observation_hash",
            "action",
            "output_tick",
            "post_public_observation_hash",
            "terminal",
            "truncated",
            "outcome",
            "row_hash",
        }
        _require_keys(mapped, allowed, allowed, "transition")
        _require_public_keys_only(mapped, "transition")

        outcome = mapped["outcome"]
        parsed_outcome = None if outcome is None else MockPublicOutcome.from_dict(outcome)

        transition = cls(
            sequence=_integer(mapped["sequence"], "transition.sequence", minimum=0),
            previous_row_hash=_nullable_sha256(mapped["previous_row_hash"], "transition.previous_row_hash"),
            input_tick=_integer(mapped["input_tick"], "transition.input_tick", minimum=0),
            pre_public_observation_hash=_sha256(
                mapped["pre_public_observation_hash"], "transition.pre_public_observation_hash"
            ),
            action=FactorizedAction.from_dict(mapped["action"]),
            output_tick=_integer(mapped["output_tick"], "transition.output_tick", minimum=0),
            post_public_observation_hash=_sha256(
                mapped["post_public_observation_hash"], "transition.post_public_observation_hash"
            ),
            terminal=_boolean(mapped["terminal"], "transition.terminal"),
            truncated=_boolean(mapped["truncated"], "transition.truncated"),
            outcome=parsed_outcome,
            row_hash=_sha256(mapped["row_hash"], "transition.row_hash"),
        )

        expected = _transition_hash_payload(transition.to_dict(include_row_hash=False))
        if transition.row_hash != expected:
            raise MockReplayValidationError("transition row_hash mismatch")
        return transition


@dataclass(frozen=True, slots=True)
class MockPublicReplay:
    schema_version: int
    artifact_kind: str
    environment_kind: EnvironmentKind
    formal: bool
    capability_claim: str
    service_identity_hash: str
    protocol_identity_hash: str
    mapping_identity_hash: str
    mock_profile: MockPublicProfile
    seed: int
    side: str
    mode: str
    episode_id: str
    transition_count: int
    transitions: tuple[MockPublicTransition, ...]
    chain_head: str | None
    artifact_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise MockReplayValidationError("unsupported schema_version")
        if self.artifact_kind != ARTIFACT_KIND:
            raise MockReplayValidationError(f"artifact_kind must be {ARTIFACT_KIND}")
        if self.environment_kind is not EnvironmentKind.MOCK:
            raise MockReplayValidationError("environment_kind must be mock")
        if self.formal is not False:
            raise MockReplayValidationError("mock replay must set formal=False")
        if self.capability_claim != CAPABILITY_CLAIM:
            raise MockReplayValidationError("capability_claim must be none")
        _sha256(self.service_identity_hash, "service_identity_hash")
        _sha256(self.protocol_identity_hash, "protocol_identity_hash")
        _sha256(self.mapping_identity_hash, "mapping_identity_hash")
        _sha256(self.artifact_hash, "artifact_hash")
        if self.seed < 0:
            raise MockReplayValidationError("seed must be >= 0")
        if self.side not in {"blue", "red"}:
            raise MockReplayValidationError("side must be blue or red")
        if self.mode != "1v1":
            raise MockReplayValidationError("mode must be 1v1")
        expected_episode_id = f"mock-{self.seed}-{self.side}-1"
        if self.episode_id != expected_episode_id:
            raise MockReplayValidationError("episode_id does not match the fixed mock profile")
        if self.transition_count != len(self.transitions):
            raise MockReplayValidationError("transition_count must match transitions length")
        if self.transition_count == 0:
            raise MockReplayValidationError("complete mock replay requires at least one transition")
        if self.chain_head is None:
            raise MockReplayValidationError("chain_head is required when transitions exist")
        if self.chain_head != self.transitions[-1].row_hash:
            raise MockReplayValidationError("chain_head must equal last transition row_hash")
        if self.chain_head is not None:
            _sha256(self.chain_head, "chain_head")
        if any(transition.terminal or transition.truncated for transition in self.transitions[:-1]):
            raise MockReplayValidationError("only the last transition may be terminal or truncated")
        if self.transitions and not (self.transitions[-1].terminal or self.transitions[-1].truncated):
            raise MockReplayValidationError("last transition must be terminal or truncated")

    def verify(self) -> None:
        _assert_row_chain(self.transitions)
        if self.artifact_hash != _compute_artifact_hash(self.to_dict(include_artifact_hash=False)):
            raise MockReplayValidationError("artifact_hash mismatch")

    def to_dict(self, *, include_artifact_hash: bool = True) -> dict[str, object]:
        data = {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "environment_kind": self.environment_kind.value,
            "formal": self.formal,
            "capability_claim": self.capability_claim,
            "service_identity_hash": self.service_identity_hash,
            "protocol_identity_hash": self.protocol_identity_hash,
            "mapping_identity_hash": self.mapping_identity_hash,
            "mock_profile": self.mock_profile.to_dict(),
            "seed": self.seed,
            "side": self.side,
            "mode": self.mode,
            "episode_id": self.episode_id,
            "transition_count": self.transition_count,
            "transitions": [transition.to_dict() for transition in self.transitions],
            "chain_head": self.chain_head,
            "artifact_hash": self.artifact_hash,
        }
        if include_artifact_hash:
            return data
        data.pop("artifact_hash")
        return data

    @classmethod
    def from_dict(cls, value: object) -> MockPublicReplay:
        mapped = _require_mapping(value, "mock_public_replay")
        allowed = {
            "schema_version",
            "artifact_kind",
            "environment_kind",
            "formal",
            "capability_claim",
            "service_identity_hash",
            "protocol_identity_hash",
            "mapping_identity_hash",
            "mock_profile",
            "seed",
            "side",
            "mode",
            "episode_id",
            "transition_count",
            "transitions",
            "chain_head",
            "artifact_hash",
        }
        _require_keys(mapped, allowed, allowed, "mock_public_replay")
        _require_public_keys_only(mapped, "mock_public_replay")

        transitions_value = mapped["transitions"]
        if not isinstance(transitions_value, list):
            raise MockReplayValidationError("transitions must be an array")

        transitions = tuple(MockPublicTransition.from_dict(item) for item in transitions_value)
        try:
            environment_kind = EnvironmentKind(
                _string(mapped["environment_kind"], "mock_public_replay.environment_kind")
            )
        except ValueError as error:
            raise MockReplayValidationError(str(error)) from error

        artifact = cls(
            schema_version=_integer(mapped["schema_version"], "mock_public_replay.schema_version"),
            artifact_kind=_string(mapped["artifact_kind"], "mock_public_replay.artifact_kind"),
            environment_kind=environment_kind,
            formal=_boolean(mapped["formal"], "mock_public_replay.formal"),
            capability_claim=_string(
                mapped["capability_claim"], "mock_public_replay.capability_claim"
            ),
            service_identity_hash=_sha256(
                mapped["service_identity_hash"],
                "mock_public_replay.service_identity_hash",
            ),
            protocol_identity_hash=_sha256(
                mapped["protocol_identity_hash"],
                "mock_public_replay.protocol_identity_hash",
            ),
            mapping_identity_hash=_sha256(mapped["mapping_identity_hash"], "mock_public_replay.mapping_identity_hash"),
            mock_profile=MockPublicProfile.from_dict(mapped["mock_profile"]),
            seed=_integer(mapped["seed"], "mock_public_replay.seed", minimum=0),
            side=_string(mapped["side"], "mock_public_replay.side"),
            mode=_string(mapped["mode"], "mock_public_replay.mode"),
            episode_id=_string(mapped["episode_id"], "mock_public_replay.episode_id"),
            transition_count=_integer(mapped["transition_count"], "mock_public_replay.transition_count", minimum=0),
            transitions=transitions,
            chain_head=_nullable_sha256(mapped["chain_head"], "mock_public_replay.chain_head"),
            artifact_hash=_sha256(mapped["artifact_hash"], "mock_public_replay.artifact_hash"),
        )
        artifact.verify()
        return artifact


MockPublicTransitionReplay = MockPublicReplay


def build_mock_public_replay(value: Mapping[str, Any]) -> MockPublicReplay:
    """Build a validated public replay from an already-materialized mapping."""
    return MockPublicReplay.from_dict(value)


def parse_mock_public_transition_replay(value: object) -> MockPublicReplay:
    """Backward-compatible entrypoint for contract parsing."""
    return MockPublicReplay.from_dict(value)
