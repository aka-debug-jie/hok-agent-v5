"""Strict, versioned JSON contracts for the learner/service boundary.

The contracts intentionally keep legal actions out of :class:`PublicObservation`.
They are carried beside an observation and are only consumed at sampling, execution,
or loss-audit time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

CONTRACT_VERSION = 1
SHA256_PREFIX_LENGTH = len("sha256:") + 64


class ContractValidationError(ValueError):
    """Raised when a boundary payload is not a valid V5 contract."""


class EnvironmentKind(StrEnum):
    HOK_GAMECORE = "hok_gamecore"
    MOCK = "mock"
    PIXELARENA = "pixelarena"


class LicenseStatus(StrEnum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ActionType(StrEnum):
    WAIT = "wait"
    MOVE = "move"
    ATTACK = "attack"
    SKILL = "skill"
    UPGRADE = "upgrade"


class TargetType(StrEnum):
    NONE = "none"
    SELF = "self"
    ENEMY = "enemy"
    STRUCTURE = "structure"


class OutcomeResult(StrEnum):
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _require_keys(value: Mapping[str, object], allowed: set[str], required: set[str], field: str) -> None:
    unexpected = set(value).difference(allowed)
    missing = required.difference(value)
    if unexpected:
        raise ContractValidationError(f"{field} contains unexpected keys: {sorted(unexpected)}")
    if missing:
        raise ContractValidationError(f"{field} is missing keys: {sorted(missing)}")


def _string(value: object, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ContractValidationError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractValidationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{field} must be >= {minimum}")
    return value


def _number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractValidationError(f"{field} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ContractValidationError(f"{field} must be finite")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{field} must be a boolean")
    return value


def _nullable_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _nullable_integer(value: object, field: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, field, minimum=minimum)


def _nullable_number(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field)


def _sha256(value: object, field: str) -> str:
    text = _string(value, field)
    if len(text) != SHA256_PREFIX_LENGTH or not text.startswith("sha256:"):
        raise ContractValidationError(f"{field} must be a sha256:<64 lowercase hex> value")
    digest = text.removeprefix("sha256:")
    if any(character not in "0123456789abcdef" for character in digest):
        raise ContractValidationError(f"{field} must be a sha256:<64 lowercase hex> value")
    return text


def _json_value(value: object, field: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ContractValidationError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_json_value(item, field) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{field} contains a non-string key")
            result[key] = _json_value(item, f"{field}.{key}")
        return result
    raise ContractValidationError(f"{field} is not JSON serializable")


def _json_mapping(value: object, field: str) -> dict[str, JsonValue]:
    mapped = _require_mapping(value, field)
    return {key: _json_value(item, f"{field}.{key}") for key, item in mapped.items()}


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{field} must be an array")
    result = tuple(_string(item, f"{field}[]") for item in value)
    if len(set(result)) != len(result):
        raise ContractValidationError(f"{field} must not contain duplicates")
    return result


ACTOR_DENYLIST = frozenset(
    {
        "truth",
        "teacher",
        "reward",
        "legal",
        "legal_action",
        "legal_actions",
        "legal_mask",
        "privileged",
        "entity_id",
        "opponent_internal",
        "future",
    }
)


def assert_actor_mapping_safe(value: Mapping[str, JsonValue], field: str) -> None:
    """Reject training-only/privileged fields from any actor-facing mapping."""

    for key, nested_value in value.items():
        normalized = key.casefold().replace("-", "_").replace(" ", "_")
        if normalized in ACTOR_DENYLIST or normalized.startswith("legal_"):
            raise ContractValidationError(f"{field}.{key} is not permitted in Actor input")
        if isinstance(nested_value, dict):
            assert_actor_mapping_safe(nested_value, f"{field}.{key}")
        if isinstance(nested_value, list):
            for index, item in enumerate(nested_value):
                if isinstance(item, dict):
                    assert_actor_mapping_safe(item, f"{field}.{key}[{index}]")


@dataclass(frozen=True, slots=True)
class FactorizedAction:
    """A V5 factorized action; it is never a K96 candidate index."""

    schema_version: int
    macro_intent: str
    action_type: ActionType
    target_type: TargetType
    target_key: str | None
    direction: int | None
    skill_slot: int | None
    item_slot: int | None
    auxiliary_parameter: float | None

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise ContractValidationError("unsupported action schema version")
        if not self.macro_intent:
            raise ContractValidationError("macro_intent must be non-empty")
        if self.direction is not None and not 0 <= self.direction < 8:
            raise ContractValidationError("direction must be in [0, 7]")
        if self.skill_slot is not None and self.skill_slot < 0:
            raise ContractValidationError("skill_slot must be non-negative")
        if self.item_slot is not None and self.item_slot < 0:
            raise ContractValidationError("item_slot must be non-negative")
        if self.auxiliary_parameter is not None and not isfinite(self.auxiliary_parameter):
            raise ContractValidationError("auxiliary_parameter must be finite")
        if self.action_type is ActionType.MOVE and self.direction is None:
            raise ContractValidationError("move action requires direction")
        if self.action_type is ActionType.SKILL and self.skill_slot is None:
            raise ContractValidationError("skill action requires skill_slot")
        if self.action_type is ActionType.UPGRADE and self.item_slot is None:
            raise ContractValidationError("upgrade action requires item_slot")
        requires_target = self.action_type in {ActionType.ATTACK, ActionType.SKILL}
        if requires_target and (self.target_type is TargetType.NONE or self.target_key is None):
            raise ContractValidationError("attack and skill actions require a public target")
        if not requires_target and self.target_type is TargetType.NONE and self.target_key is not None:
            raise ContractValidationError("no-target action must not include target_key")

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "macro_intent": self.macro_intent,
            "action_type": self.action_type.value,
            "target_type": self.target_type.value,
            "target_key": self.target_key,
            "direction": self.direction,
            "skill_slot": self.skill_slot,
            "item_slot": self.item_slot,
            "auxiliary_parameter": self.auxiliary_parameter,
        }

    @classmethod
    def from_dict(cls, value: object) -> FactorizedAction:
        mapped = _require_mapping(value, "factorized_action")
        allowed = {
            "schema_version",
            "macro_intent",
            "action_type",
            "target_type",
            "target_key",
            "direction",
            "skill_slot",
            "item_slot",
            "auxiliary_parameter",
        }
        _require_keys(mapped, allowed, allowed, "factorized_action")
        try:
            action_type = ActionType(_string(mapped["action_type"], "factorized_action.action_type"))
            target_type = TargetType(_string(mapped["target_type"], "factorized_action.target_type"))
        except ValueError as error:
            raise ContractValidationError(str(error)) from error
        return cls(
            schema_version=_integer(mapped["schema_version"], "factorized_action.schema_version"),
            macro_intent=_string(mapped["macro_intent"], "factorized_action.macro_intent"),
            action_type=action_type,
            target_type=target_type,
            target_key=_nullable_string(mapped["target_key"], "factorized_action.target_key"),
            direction=_nullable_integer(mapped["direction"], "factorized_action.direction"),
            skill_slot=_nullable_integer(mapped["skill_slot"], "factorized_action.skill_slot"),
            item_slot=_nullable_integer(mapped["item_slot"], "factorized_action.item_slot"),
            auxiliary_parameter=_nullable_number(
                mapped["auxiliary_parameter"], "factorized_action.auxiliary_parameter"
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicEntity:
    public_key: str
    category: str
    team_relation: str
    x: float
    y: float
    health: float
    visible: bool

    def __post_init__(self) -> None:
        if not self.public_key or not self.category or not self.team_relation:
            raise ContractValidationError("public entity identity fields must be non-empty")
        if not all(isfinite(value) for value in (self.x, self.y, self.health)):
            raise ContractValidationError("public entity numbers must be finite")

    def to_dict(self) -> JsonObject:
        return {
            "public_key": self.public_key,
            "category": self.category,
            "team_relation": self.team_relation,
            "x": self.x,
            "y": self.y,
            "health": self.health,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, value: object) -> PublicEntity:
        mapped = _require_mapping(value, "public_entity")
        allowed = {"public_key", "category", "team_relation", "x", "y", "health", "visible"}
        _require_keys(mapped, allowed, allowed, "public_entity")
        return cls(
            public_key=_string(mapped["public_key"], "public_entity.public_key"),
            category=_string(mapped["category"], "public_entity.category"),
            team_relation=_string(mapped["team_relation"], "public_entity.team_relation"),
            x=_number(mapped["x"], "public_entity.x"),
            y=_number(mapped["y"], "public_entity.y"),
            health=_number(mapped["health"], "public_entity.health"),
            visible=_boolean(mapped["visible"], "public_entity.visible"),
        )


@dataclass(frozen=True, slots=True)
class PublicObservation:
    schema_version: int
    tick: int
    self_state: Mapping[str, float]
    visible_entities: tuple[PublicEntity, ...]
    global_features: Mapping[str, float]
    previous_action: FactorizedAction | None

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise ContractValidationError("unsupported observation schema version")
        if self.tick < 0:
            raise ContractValidationError("observation tick must be non-negative")
        for field, values in (
            ("self_state", self.self_state),
            ("global_features", self.global_features),
        ):
            converted = {key: _json_value(value, f"{field}.{key}") for key, value in values.items()}
            assert_actor_mapping_safe(converted, field)
            if any(not isfinite(float(number)) for number in values.values()):
                raise ContractValidationError(f"{field} values must be finite")
        public_keys = [entity.public_key for entity in self.visible_entities]
        if len(public_keys) != len(set(public_keys)):
            raise ContractValidationError("visible entity public keys must be unique")

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "tick": self.tick,
            "self_state": dict(self.self_state),
            "visible_entities": [entity.to_dict() for entity in self.visible_entities],
            "global_features": dict(self.global_features),
            "previous_action": None if self.previous_action is None else self.previous_action.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> PublicObservation:
        mapped = _require_mapping(value, "public_observation")
        allowed = {
            "schema_version",
            "tick",
            "self_state",
            "visible_entities",
            "global_features",
            "previous_action",
        }
        _require_keys(mapped, allowed, allowed, "public_observation")
        entities_value = mapped["visible_entities"]
        if not isinstance(entities_value, list):
            raise ContractValidationError("public_observation.visible_entities must be an array")
        previous_value = mapped["previous_action"]
        return cls(
            schema_version=_integer(mapped["schema_version"], "public_observation.schema_version"),
            tick=_integer(mapped["tick"], "public_observation.tick", minimum=0),
            self_state=_numeric_mapping(mapped["self_state"], "public_observation.self_state"),
            visible_entities=tuple(PublicEntity.from_dict(item) for item in entities_value),
            global_features=_numeric_mapping(mapped["global_features"], "public_observation.global_features"),
            previous_action=None if previous_value is None else FactorizedAction.from_dict(previous_value),
        )


def _numeric_mapping(value: object, field: str) -> dict[str, float]:
    mapped = _require_mapping(value, field)
    return {key: _number(item, f"{field}.{key}") for key, item in mapped.items()}


@dataclass(frozen=True, slots=True)
class LegalActionSet:
    schema_version: int
    actions: tuple[FactorizedAction, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise ContractValidationError("unsupported legal action schema version")
        if len(set(self.actions)) != len(self.actions):
            raise ContractValidationError("legal action list must not contain duplicates")

    def contains(self, action: FactorizedAction) -> bool:
        return action in self.actions

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(cls, value: object) -> LegalActionSet:
        mapped = _require_mapping(value, "legal_actions")
        allowed = {"schema_version", "actions"}
        _require_keys(mapped, allowed, allowed, "legal_actions")
        actions = mapped["actions"]
        if not isinstance(actions, list):
            raise ContractValidationError("legal_actions.actions must be an array")
        return cls(
            schema_version=_integer(mapped["schema_version"], "legal_actions.schema_version"),
            actions=tuple(FactorizedAction.from_dict(action) for action in actions),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    schema_version: int
    environment_kind: EnvironmentKind
    service_version: str
    sdk_version: str | None
    gamecore_build: str | None
    mapping_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise ContractValidationError("unsupported environment identity schema version")
        if not self.service_version:
            raise ContractValidationError("service_version must be non-empty")
        _sha256(self.mapping_hash, "environment_identity.mapping_hash")

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "environment_kind": self.environment_kind.value,
            "service_version": self.service_version,
            "sdk_version": self.sdk_version,
            "gamecore_build": self.gamecore_build,
            "mapping_hash": self.mapping_hash,
        }

    @classmethod
    def from_dict(cls, value: object) -> EnvironmentIdentity:
        mapped = _require_mapping(value, "environment_identity")
        allowed = {
            "schema_version",
            "environment_kind",
            "service_version",
            "sdk_version",
            "gamecore_build",
            "mapping_hash",
        }
        _require_keys(mapped, allowed, allowed, "environment_identity")
        try:
            kind = EnvironmentKind(_string(mapped["environment_kind"], "environment_identity.kind"))
        except ValueError as error:
            raise ContractValidationError(str(error)) from error
        return cls(
            schema_version=_integer(mapped["schema_version"], "environment_identity.schema_version"),
            environment_kind=kind,
            service_version=_string(mapped["service_version"], "environment_identity.service_version"),
            sdk_version=_nullable_string(mapped["sdk_version"], "environment_identity.sdk_version"),
            gamecore_build=_nullable_string(mapped["gamecore_build"], "environment_identity.gamecore_build"),
            mapping_hash=_sha256(mapped["mapping_hash"], "environment_identity.mapping_hash"),
        )


@dataclass(frozen=True, slots=True)
class HealthResponse:
    protocol_version: int
    environment: EnvironmentIdentity
    license_status: LicenseStatus
    supported_modes: tuple[str, ...]
    supported_heroes: tuple[str, ...]
    ready: bool

    def __post_init__(self) -> None:
        if self.protocol_version != CONTRACT_VERSION:
            raise ContractValidationError("unsupported protocol version")

    def to_dict(self) -> JsonObject:
        return {
            "protocol_version": self.protocol_version,
            "environment": self.environment.to_dict(),
            "license_status": self.license_status.value,
            "supported_modes": list(self.supported_modes),
            "supported_heroes": list(self.supported_heroes),
            "ready": self.ready,
        }

    @classmethod
    def from_dict(cls, value: object) -> HealthResponse:
        mapped = _require_mapping(value, "health_response")
        allowed = {
            "protocol_version",
            "environment",
            "license_status",
            "supported_modes",
            "supported_heroes",
            "ready",
        }
        _require_keys(mapped, allowed, allowed, "health_response")
        try:
            license_status = LicenseStatus(_string(mapped["license_status"], "health_response.license_status"))
        except ValueError as error:
            raise ContractValidationError(str(error)) from error
        return cls(
            protocol_version=_integer(mapped["protocol_version"], "health_response.protocol_version"),
            environment=EnvironmentIdentity.from_dict(mapped["environment"]),
            license_status=license_status,
            supported_modes=_string_tuple(mapped["supported_modes"], "health_response.supported_modes"),
            supported_heroes=_string_tuple(mapped["supported_heroes"], "health_response.supported_heroes"),
            ready=_boolean(mapped["ready"], "health_response.ready"),
        )


@dataclass(frozen=True, slots=True)
class ResetRequest:
    request_id: str
    mode: str
    seed: int
    side: str
    hero_config: Mapping[str, JsonValue]
    opponent_config: Mapping[str, JsonValue]
    reward_config_version: str
    evaluation_mode: bool

    def __post_init__(self) -> None:
        if not self.request_id or not self.mode or not self.side or not self.reward_config_version:
            raise ContractValidationError("reset request identity fields must be non-empty")
        if self.seed < 0:
            raise ContractValidationError("reset seed must be non-negative")

    def to_dict(self) -> JsonObject:
        return {
            "request_id": self.request_id,
            "mode": self.mode,
            "seed": self.seed,
            "side": self.side,
            "hero_config": dict(self.hero_config),
            "opponent_config": dict(self.opponent_config),
            "reward_config_version": self.reward_config_version,
            "evaluation_mode": self.evaluation_mode,
        }

    @classmethod
    def from_dict(cls, value: object) -> ResetRequest:
        mapped = _require_mapping(value, "reset_request")
        allowed = {
            "request_id",
            "mode",
            "seed",
            "side",
            "hero_config",
            "opponent_config",
            "reward_config_version",
            "evaluation_mode",
        }
        _require_keys(mapped, allowed, allowed, "reset_request")
        return cls(
            request_id=_string(mapped["request_id"], "reset_request.request_id"),
            mode=_string(mapped["mode"], "reset_request.mode"),
            seed=_integer(mapped["seed"], "reset_request.seed", minimum=0),
            side=_string(mapped["side"], "reset_request.side"),
            hero_config=_json_mapping(mapped["hero_config"], "reset_request.hero_config"),
            opponent_config=_json_mapping(mapped["opponent_config"], "reset_request.opponent_config"),
            reward_config_version=_string(mapped["reward_config_version"], "reset_request.reward_config_version"),
            evaluation_mode=_boolean(mapped["evaluation_mode"], "reset_request.evaluation_mode"),
        )


@dataclass(frozen=True, slots=True)
class RewardVector:
    components: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.components:
            raise ContractValidationError("reward components must not be empty")
        if any(not key or not isfinite(value) for key, value in self.components.items()):
            raise ContractValidationError("reward component names and values must be finite")

    @property
    def total(self) -> float:
        return sum(self.components.values())

    def to_dict(self) -> JsonObject:
        return {"components": dict(self.components), "total": self.total}

    @classmethod
    def from_dict(cls, value: object) -> RewardVector:
        mapped = _require_mapping(value, "reward_vector")
        allowed = {"components", "total"}
        _require_keys(mapped, allowed, allowed, "reward_vector")
        components = _numeric_mapping(mapped["components"], "reward_vector.components")
        total = _number(mapped["total"], "reward_vector.total")
        result = cls(components=components)
        if abs(result.total - total) > 1e-12:
            raise ContractValidationError("reward_vector.total must equal component sum")
        return result


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    result: OutcomeResult
    reason: str
    terminal: bool
    truncated: bool

    def __post_init__(self) -> None:
        if not self.reason:
            raise ContractValidationError("episode outcome reason must be non-empty")
        if not (self.terminal or self.truncated):
            raise ContractValidationError("episode outcome requires terminal or truncated")

    def to_dict(self) -> JsonObject:
        return {
            "result": self.result.value,
            "reason": self.reason,
            "terminal": self.terminal,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, value: object) -> EpisodeOutcome:
        mapped = _require_mapping(value, "episode_outcome")
        allowed = {"result", "reason", "terminal", "truncated"}
        _require_keys(mapped, allowed, allowed, "episode_outcome")
        try:
            result = OutcomeResult(_string(mapped["result"], "episode_outcome.result"))
        except ValueError as error:
            raise ContractValidationError(str(error)) from error
        return cls(
            result=result,
            reason=_string(mapped["reason"], "episode_outcome.reason"),
            terminal=_boolean(mapped["terminal"], "episode_outcome.terminal"),
            truncated=_boolean(mapped["truncated"], "episode_outcome.truncated"),
        )


@dataclass(frozen=True, slots=True)
class ResetResponse:
    request_id: str
    episode_id: str
    tick: int
    observation: PublicObservation
    legal_actions: LegalActionSet
    reward_component_names: tuple[str, ...]
    environment: EnvironmentIdentity
    replay_identity: str

    def __post_init__(self) -> None:
        if not self.request_id or not self.episode_id or self.tick != self.observation.tick:
            raise ContractValidationError("reset response episode/request/tick mismatch")
        if not self.legal_actions.actions:
            raise ContractValidationError("reset response legal actions must not be empty")
        _sha256(self.replay_identity, "reset_response.replay_identity")

    def to_dict(self) -> JsonObject:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "tick": self.tick,
            "observation": self.observation.to_dict(),
            "legal_actions": self.legal_actions.to_dict(),
            "reward_component_names": list(self.reward_component_names),
            "environment": self.environment.to_dict(),
            "replay_identity": self.replay_identity,
        }

    @classmethod
    def from_dict(cls, value: object) -> ResetResponse:
        mapped = _require_mapping(value, "reset_response")
        allowed = {
            "request_id",
            "episode_id",
            "tick",
            "observation",
            "legal_actions",
            "reward_component_names",
            "environment",
            "replay_identity",
        }
        _require_keys(mapped, allowed, allowed, "reset_response")
        return cls(
            request_id=_string(mapped["request_id"], "reset_response.request_id"),
            episode_id=_string(mapped["episode_id"], "reset_response.episode_id"),
            tick=_integer(mapped["tick"], "reset_response.tick", minimum=0),
            observation=PublicObservation.from_dict(mapped["observation"]),
            legal_actions=LegalActionSet.from_dict(mapped["legal_actions"]),
            reward_component_names=_string_tuple(
                mapped["reward_component_names"], "reset_response.reward_component_names"
            ),
            environment=EnvironmentIdentity.from_dict(mapped["environment"]),
            replay_identity=_sha256(mapped["replay_identity"], "reset_response.replay_identity"),
        )


@dataclass(frozen=True, slots=True)
class StepRequest:
    episode_id: str
    expected_tick: int
    action: FactorizedAction
    action_schema_version: int

    def __post_init__(self) -> None:
        if not self.episode_id or self.expected_tick < 0:
            raise ContractValidationError("invalid step request episode_id or expected_tick")
        if self.action_schema_version != CONTRACT_VERSION:
            raise ContractValidationError("unsupported step action schema version")
        if self.action.schema_version != self.action_schema_version:
            raise ContractValidationError("step action schema version mismatch")

    def to_dict(self) -> JsonObject:
        return {
            "episode_id": self.episode_id,
            "expected_tick": self.expected_tick,
            "action": self.action.to_dict(),
            "action_schema_version": self.action_schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> StepRequest:
        mapped = _require_mapping(value, "step_request")
        allowed = {"episode_id", "expected_tick", "action", "action_schema_version"}
        _require_keys(mapped, allowed, allowed, "step_request")
        return cls(
            episode_id=_string(mapped["episode_id"], "step_request.episode_id"),
            expected_tick=_integer(mapped["expected_tick"], "step_request.expected_tick", minimum=0),
            action=FactorizedAction.from_dict(mapped["action"]),
            action_schema_version=_integer(mapped["action_schema_version"], "step_request.action_schema_version"),
        )


@dataclass(frozen=True, slots=True)
class StepResponse:
    observation: PublicObservation
    reward: RewardVector
    episode_id: str
    terminal: bool
    truncated: bool
    outcome: EpisodeOutcome | None
    legal_actions: LegalActionSet
    tick: int
    replay_hash: str
    error_code: str | None

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ContractValidationError("step response episode_id must be non-empty")
        if self.tick != self.observation.tick:
            raise ContractValidationError("step response tick must match observation tick")
        if self.terminal or self.truncated:
            if self.outcome is None:
                raise ContractValidationError("terminal step requires outcome")
        elif self.outcome is not None:
            raise ContractValidationError("nonterminal step must not include outcome")
        if not (self.terminal or self.truncated) and not self.legal_actions.actions:
            raise ContractValidationError("nonterminal step legal actions must not be empty")
        _sha256(self.replay_hash, "step_response.replay_hash")

    def to_dict(self) -> JsonObject:
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward.to_dict(),
            "episode_id": self.episode_id,
            "terminal": self.terminal,
            "truncated": self.truncated,
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
            "legal_actions": self.legal_actions.to_dict(),
            "tick": self.tick,
            "replay_hash": self.replay_hash,
            "error_code": self.error_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> StepResponse:
        mapped = _require_mapping(value, "step_response")
        allowed = {
            "observation",
            "reward",
            "episode_id",
            "terminal",
            "truncated",
            "outcome",
            "legal_actions",
            "tick",
            "replay_hash",
            "error_code",
        }
        _require_keys(mapped, allowed, allowed, "step_response")
        outcome_value = mapped["outcome"]
        return cls(
            observation=PublicObservation.from_dict(mapped["observation"]),
            reward=RewardVector.from_dict(mapped["reward"]),
            episode_id=_string(mapped["episode_id"], "step_response.episode_id"),
            terminal=_boolean(mapped["terminal"], "step_response.terminal"),
            truncated=_boolean(mapped["truncated"], "step_response.truncated"),
            outcome=None if outcome_value is None else EpisodeOutcome.from_dict(outcome_value),
            legal_actions=LegalActionSet.from_dict(mapped["legal_actions"]),
            tick=_integer(mapped["tick"], "step_response.tick", minimum=0),
            replay_hash=_sha256(mapped["replay_hash"], "step_response.replay_hash"),
            error_code=_nullable_string(mapped["error_code"], "step_response.error_code"),
        )


@dataclass(frozen=True, slots=True)
class CloseResponse:
    episode_id: str
    stop_reason: str
    already_closed: bool

    def __post_init__(self) -> None:
        if not self.episode_id or not self.stop_reason:
            raise ContractValidationError("close response fields must be non-empty")

    def to_dict(self) -> JsonObject:
        return {
            "episode_id": self.episode_id,
            "stop_reason": self.stop_reason,
            "already_closed": self.already_closed,
        }

    @classmethod
    def from_dict(cls, value: object) -> CloseResponse:
        mapped = _require_mapping(value, "close_response")
        allowed = {"episode_id", "stop_reason", "already_closed"}
        _require_keys(mapped, allowed, allowed, "close_response")
        return cls(
            episode_id=_string(mapped["episode_id"], "close_response.episode_id"),
            stop_reason=_string(mapped["stop_reason"], "close_response.stop_reason"),
            already_closed=_boolean(mapped["already_closed"], "close_response.already_closed"),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Schema-compatible typed representation of a run manifest."""

    run_id: str
    created_at_utc: str
    stage: str
    status: str
    formal: bool
    git_commit: str
    git_dirty: bool
    environment: Mapping[str, JsonValue]
    config_path: str
    config_hash: str
    algorithm_name: str
    algorithm_version: str
    seed_registry_hash: str
    safety: Mapping[str, bool]
    artifacts: tuple[Mapping[str, str], ...]

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": CONTRACT_VERSION,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "stage": self.stage,
            "status": self.status,
            "formal": self.formal,
            "git": {"commit": self.git_commit, "dirty": self.git_dirty},
            "environment": dict(self.environment),
            "config": {"path": self.config_path, "hash": self.config_hash},
            "algorithm": {"name": self.algorithm_name, "version": self.algorithm_version},
            "seeds": {"registry_hash": self.seed_registry_hash},
            "safety": dict(self.safety),
            "artifacts": [dict(artifact) for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Schema-compatible typed representation of an E0 diagnostic report."""

    report_id: str
    created_at_utc: str
    suite_id: str
    suite_version: int
    suite_config_hash: str
    checkpoint_path: str
    checkpoint_hash: str
    training_run_id: str
    environment: Mapping[str, JsonValue]
    episode_count: int
    episode_completed: int
    details_artifact: str
    metrics: Mapping[str, JsonValue]
    engineering: Mapping[str, int]
    gate_checks: Mapping[str, bool]
    disposition: str

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": CONTRACT_VERSION,
            "report_id": self.report_id,
            "created_at_utc": self.created_at_utc,
            "suite": {
                "id": self.suite_id,
                "version": self.suite_version,
                "config_hash": self.suite_config_hash,
            },
            "checkpoint": {
                "path": self.checkpoint_path,
                "sha256": self.checkpoint_hash,
                "training_run_id": self.training_run_id,
                "frozen_before_eval": True,
            },
            "environment": dict(self.environment),
            "episodes": {
                "count": self.episode_count,
                "completed": self.episode_completed,
                "details_artifact": self.details_artifact,
            },
            "metrics": dict(self.metrics),
            "engineering": dict(self.engineering),
            "gate": {"pass": all(self.gate_checks.values()), "checks": dict(self.gate_checks)},
            "disposition": self.disposition,
            "safety": {
                "commercial_client_actions_used": False,
                "privileged_state_enters_actor": False,
            },
        }
