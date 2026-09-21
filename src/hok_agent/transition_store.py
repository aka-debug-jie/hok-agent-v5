from __future__ import annotations

import copy
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from hok_agent.frame_bus import FramePacketRecord
from hok_agent.visual_events import VisualEventRecord

TRANSITION_SCHEMA = "hok-agent-hierarchical-transition-v0"

MacroAction = Literal["FARM_LANE", "PUSH_STRUCTURE", "ENGAGE", "DISENGAGE", "RECALL", "HOLD"]
MovementAction = Literal["NONE", "STOP", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
CombatAction = Literal["WAIT", "BASIC_ATTACK", "SKILL1", "SKILL2", "SKILL3"]
AttemptStatus = Literal["not_attempted", "acknowledged", "rejected", "failed"]
FinalActionStatus = Literal["acknowledged", "rejected", "failed", "noop"]
TerminalReason = Literal[
    "NOT_DONE",
    "WIN",
    "LOSS",
    "SAFETY_STOP",
    "CAPTURE_FAILURE",
    "ACTION_FAILURE",
    "TIMEOUT",
    "NAVIGATION_GOAL_REACHED",
    "VIDEO_EOF",
    "UNKNOWN",
]
EpisodeEndKind = Literal["NOT_DONE", "TERMINATED", "TRUNCATED", "ERROR"]
ReplaySource = Literal["demo", "sim", "controller", "online", "offline_video"]
NavigationDecisionOwner = Literal[
    "geometry_rule", "deterministic_executor", "deterministic_router"
]

_MACRO_ACTIONS = {"FARM_LANE", "PUSH_STRUCTURE", "ENGAGE", "DISENGAGE", "RECALL", "HOLD"}
_MOVEMENT_ACTIONS = {"NONE", "STOP", "N", "NE", "E", "SE", "S", "SW", "W", "NW"}
_COMBAT_ACTIONS = {"WAIT", "BASIC_ATTACK", "SKILL1", "SKILL2", "SKILL3"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class PolicyProposalRecord(TypedDict):
    observation_id: str
    applied_observation_id: str
    carried_forward: bool
    value: str
    confidence: float
    decision_start_ns: int
    decision_end_ns: int
    valid_until_ns: int
    policy_bundle_version: str


class ProposalBundleRecord(TypedDict):
    macro: PolicyProposalRecord
    movement: PolicyProposalRecord
    combat: PolicyProposalRecord


class ExecutedActionRecord(TypedDict):
    requested_movement: MovementAction
    applied_movement: MovementAction
    requested_combat: CombatAction
    applied_combat: CombatAction
    movement_command: Literal["DOWN", "MOVE", "UP", "KEEP", "NOOP"]
    combat_command: Literal["TAP", "NOOP"]
    dispatch_start_ns: int
    dispatch_ack_ns: int
    first_attempt_status: AttemptStatus
    retry_status: AttemptStatus
    retry_count: int
    final_status: FinalActionStatus


class RewardComponentsRecord(TypedDict):
    terminal: float
    death: float
    self_hp_delta: float
    tower_damage: float


class RewardRecord(TypedDict):
    reward_version: str
    components: RewardComponentsRecord
    total: float
    event_ids: list[str]


class ReplayRecord(TypedDict):
    source: ReplaySource
    failure_tags: list[str]
    priority: float


class NavigationContextRecord(TypedDict):
    goal_version: int
    next_goal_version: int
    goal_xy: list[int]
    next_goal_xy: list[int]
    position_before: list[int]
    position_after: list[int]
    decision_owner: NavigationDecisionOwner
    decision_reason: str
    goal_source: Literal["simulator_config"]
    simulation_time_ms: int


class HierarchicalTransitionRecord(TypedDict):
    schema_version: str
    episode_id: str
    step_id: int
    policy_bundle_version: str
    policy_bundle_sha256: str
    event_engine_version: str
    event_engine_sha256: str
    observation: FramePacketRecord
    proposals: ProposalBundleRecord
    executed_action: ExecutedActionRecord
    settle_end_ns: int
    next_observation: FramePacketRecord
    events: list[VisualEventRecord]
    reward: RewardRecord
    done: bool
    terminal_reason: TerminalReason
    episode_end_kind: NotRequired[EpisodeEndKind]
    causal_order_valid: bool
    training_eligible: bool
    ineligibility_reasons: NotRequired[list[str]]
    replay: ReplayRecord
    navigation_context: NotRequired[NavigationContextRecord]


@dataclass(frozen=True, slots=True)
class TransitionValidation:
    causal_order_valid: bool
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class StoredTransition:
    payload: HierarchicalTransitionRecord
    validation: TransitionValidation


def _validate_frame(
    frame: FramePacketRecord,
    label: str,
    errors: list[str],
    causal_errors: list[str],
) -> None:
    if frame["capture_start_ns"] > frame["capture_end_ns"]:
        message = f"{label}_capture_time_invalid"
        errors.append(message)
        causal_errors.append(message)
    if frame["source_locator_persisted"] is not False:
        errors.append(f"{label}_source_locator_persisted")
    if set(frame["view_sha256"]) < {"main", "minimap", "hud"}:
        errors.append(f"{label}_required_views_missing")
    if any(_SHA256.fullmatch(value) is None for value in frame["view_sha256"].values()):
        errors.append(f"{label}_view_hash_invalid")


def validate_transition(row: HierarchicalTransitionRecord) -> TransitionValidation:
    errors: list[str] = []
    causal_errors: list[str] = []
    observation = row["observation"]
    next_observation = row["next_observation"]

    if row["schema_version"] != TRANSITION_SCHEMA:
        errors.append("schema_version_invalid")
    if not row["episode_id"] or row["step_id"] < 0:
        errors.append("transition_identity_invalid")
    if _SHA256.fullmatch(row["policy_bundle_sha256"]) is None:
        errors.append("policy_bundle_hash_invalid")
    if _SHA256.fullmatch(row["event_engine_sha256"]) is None:
        errors.append("event_engine_hash_invalid")

    _validate_frame(observation, "observation", errors, causal_errors)
    _validate_frame(next_observation, "next_observation", errors, causal_errors)
    if observation["observation_id"] == next_observation["observation_id"]:
        errors.append("next_observation_id_not_new")

    proposal_values = {
        "macro": _MACRO_ACTIONS,
        "movement": _MOVEMENT_ACTIONS,
        "combat": _COMBAT_ACTIONS,
    }
    decision_ends: list[int] = []
    for head, allowed in proposal_values.items():
        proposal = row["proposals"][cast(Literal["macro", "movement", "combat"], head)]
        if proposal["applied_observation_id"] != observation["observation_id"]:
            errors.append(f"{head}_applied_observation_id_mismatch")
        if proposal["carried_forward"]:
            if proposal["observation_id"] == observation["observation_id"]:
                errors.append(f"{head}_carried_source_not_previous")
            if proposal["decision_end_ns"] > observation["capture_start_ns"]:
                message = f"{head}_carried_decision_not_previous"
                errors.append(message)
                causal_errors.append(message)
        elif proposal["observation_id"] != observation["observation_id"]:
            errors.append(f"{head}_source_observation_id_mismatch")
        if proposal["policy_bundle_version"] != row["policy_bundle_version"]:
            errors.append(f"{head}_bundle_version_mismatch")
        if proposal["value"] not in allowed:
            errors.append(f"{head}_value_invalid")
        if not 0.0 <= proposal["confidence"] <= 1.0:
            errors.append(f"{head}_confidence_invalid")
        if (
            not proposal["carried_forward"]
            and proposal["decision_start_ns"] < observation["capture_end_ns"]
        ):
            message = f"{head}_decision_before_capture_end"
            errors.append(message)
            causal_errors.append(message)
        if proposal["decision_end_ns"] < proposal["decision_start_ns"]:
            message = f"{head}_decision_time_invalid"
            errors.append(message)
            causal_errors.append(message)
        if proposal["valid_until_ns"] < proposal["decision_end_ns"]:
            errors.append(f"{head}_expired_before_decision_end")
        decision_ends.append(proposal["decision_end_ns"])

    executed = row["executed_action"]
    if executed["requested_movement"] not in _MOVEMENT_ACTIONS:
        errors.append("requested_movement_invalid")
    if executed["applied_movement"] not in _MOVEMENT_ACTIONS:
        errors.append("applied_movement_invalid")
    if executed["requested_combat"] not in _COMBAT_ACTIONS:
        errors.append("requested_combat_invalid")
    if executed["applied_combat"] not in _COMBAT_ACTIONS:
        errors.append("applied_combat_invalid")
    navigation = row.get("navigation_context")
    if navigation is not None:
        if (
            navigation["goal_version"] < 0
            or navigation["next_goal_version"] not in {
                navigation["goal_version"],
                navigation["goal_version"] + 1,
            }
            or len(navigation["goal_xy"]) != 2
            or len(navigation["next_goal_xy"]) != 2
            or len(navigation["position_before"]) != 2
            or len(navigation["position_after"]) != 2
            or navigation["goal_source"] != "simulator_config"
            or navigation["simulation_time_ms"] != row["step_id"] * 100
        ):
            errors.append("navigation_context_invalid")
        if (
            navigation["decision_owner"] == "deterministic_executor"
            and executed["movement_command"] != "KEEP"
        ):
            errors.append("navigation_executor_without_keep")
        if (
            navigation["decision_owner"] == "deterministic_router"
            and executed["applied_movement"] != "STOP"
        ):
            errors.append("navigation_router_without_stop")
    if executed["dispatch_start_ns"] < max(decision_ends):
        message = "dispatch_before_decision_end"
        errors.append(message)
        causal_errors.append(message)
    for head in proposal_values:
        proposal = row["proposals"][cast(Literal["macro", "movement", "combat"], head)]
        if proposal["valid_until_ns"] < executed["dispatch_start_ns"]:
            errors.append(f"{head}_proposal_stale")
    if executed["dispatch_ack_ns"] < executed["dispatch_start_ns"]:
        message = "dispatch_ack_before_start"
        errors.append(message)
        causal_errors.append(message)
    if row["settle_end_ns"] < executed["dispatch_ack_ns"]:
        message = "settle_before_dispatch_ack"
        errors.append(message)
        causal_errors.append(message)
    if next_observation["capture_start_ns"] < row["settle_end_ns"]:
        message = "next_capture_before_settle"
        errors.append(message)
        causal_errors.append(message)

    if executed["retry_count"] not in {0, 1}:
        errors.append("retry_count_invalid")
    if executed["retry_count"] == 0 and executed["retry_status"] != "not_attempted":
        errors.append("retry_status_without_retry")
    if executed["retry_count"] == 1 and executed["retry_status"] == "not_attempted":
        errors.append("retry_status_missing")

    event_ids = [event["event_id"] for event in row["events"]]
    dedup_keys = [event["dedup_key"] for event in row["events"]]
    if len(event_ids) != len(set(event_ids)):
        errors.append("duplicate_event_id")
    if len(dedup_keys) != len(set(dedup_keys)):
        errors.append("duplicate_event_dedup_key")
    if not set(row["reward"]["event_ids"]).issubset(event_ids):
        errors.append("reward_references_unknown_event")
    components = row["reward"]["components"]
    reward_total = sum(
        (
            components["terminal"],
            components["death"],
            components["self_hp_delta"],
            components["tower_damage"],
        )
    )
    if not math.isclose(row["reward"]["total"], reward_total, abs_tol=1e-9):
        errors.append("reward_total_mismatch")

    if row["done"] and row["terminal_reason"] == "NOT_DONE":
        errors.append("done_without_terminal_reason")
    if not row["done"] and row["terminal_reason"] != "NOT_DONE":
        errors.append("terminal_reason_without_done")
    event_types = {event["event_type"] for event in row["events"]}
    if row["terminal_reason"] == "WIN" and "WIN" not in event_types:
        errors.append("win_terminal_event_missing")
    if row["terminal_reason"] == "LOSS" and "LOSS" not in event_types:
        errors.append("loss_terminal_event_missing")
    end_kind = row.get("episode_end_kind")
    if end_kind is not None:
        expected_end_kind: EpisodeEndKind
        if row["terminal_reason"] == "NOT_DONE":
            expected_end_kind = "NOT_DONE"
        elif row["terminal_reason"] in {"TIMEOUT", "VIDEO_EOF"}:
            expected_end_kind = "TRUNCATED"
        elif row["terminal_reason"] in {
            "SAFETY_STOP",
            "CAPTURE_FAILURE",
            "ACTION_FAILURE",
        }:
            expected_end_kind = "ERROR"
        else:
            expected_end_kind = "TERMINATED"
        if end_kind != expected_end_kind:
            errors.append("episode_end_kind_mismatch")

    causal_order_valid = not causal_errors
    if row["causal_order_valid"] != causal_order_valid:
        errors.append("causal_order_flag_mismatch")
    if row["training_eligible"] and errors:
        errors.append("training_eligible_with_validation_errors")
    return TransitionValidation(causal_order_valid, tuple(errors))


def _normalized_for_storage(
    row: HierarchicalTransitionRecord,
    validation: TransitionValidation,
) -> HierarchicalTransitionRecord:
    payload = copy.deepcopy(row)
    payload["causal_order_valid"] = validation.causal_order_valid
    if validation.errors:
        payload["training_eligible"] = False
        reasons = list(payload.get("ineligibility_reasons", []))
        payload["ineligibility_reasons"] = list(dict.fromkeys([*reasons, *validation.errors]))
    return payload


class UnifiedTransitionStore:
    """Transactional metadata store; RGB frame bundles remain external."""

    def __init__(self, path: Path) -> None:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("TransitionStore path must be a regular file")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection: sqlite3.Connection | None = sqlite3.connect(path)
        connection = self._require_connection()
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transitions (
                episode_id TEXT NOT NULL,
                step_id INTEGER NOT NULL,
                observation_id TEXT NOT NULL,
                next_observation_id TEXT NOT NULL,
                done INTEGER NOT NULL,
                training_eligible INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (episode_id, step_id)
            )
            """
        )
        connection.commit()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("TransitionStore is closed")
        return self._connection

    def _sequence_errors(self, row: HierarchicalTransitionRecord) -> tuple[str, ...]:
        connection = self._require_connection()
        previous = connection.execute(
            """
            SELECT step_id, next_observation_id, done, training_eligible, payload_json
            FROM transitions
            WHERE episode_id = ?
            ORDER BY step_id DESC
            LIMIT 1
            """,
            (row["episode_id"],),
        ).fetchone()
        errors: list[str] = []
        if previous is None:
            if row["step_id"] != 0:
                errors.append("episode_does_not_start_at_step_zero")
        else:
            (
                previous_step,
                previous_next_observation,
                previous_done,
                previous_training_eligible,
                previous_payload_json,
            ) = previous
            if row["step_id"] != int(previous_step) + 1:
                errors.append("episode_step_not_contiguous")
            if row["observation"]["observation_id"] != str(previous_next_observation):
                errors.append("episode_observation_chain_broken")
            if bool(previous_done):
                errors.append("episode_continues_after_terminal")
            previous_payload = cast(dict[str, object], json.loads(str(previous_payload_json)))
            if (
                not bool(previous_training_eligible)
                and previous_payload.get("causal_order_valid") is not True
            ):
                errors.append("previous_transition_ineligible")

        for head in ("macro", "movement", "combat"):
            proposal = row["proposals"][head]
            if not proposal["carried_forward"]:
                continue
            found = connection.execute(
                """
                SELECT 1
                FROM transitions
                WHERE episode_id = ?
                  AND step_id < ?
                  AND (observation_id = ? OR next_observation_id = ?)
                LIMIT 1
                """,
                (
                    row["episode_id"],
                    row["step_id"],
                    proposal["observation_id"],
                    proposal["observation_id"],
                ),
            ).fetchone()
            if found is None:
                errors.append(f"{head}_carried_source_not_in_episode")
        return tuple(errors)

    def append(self, row: HierarchicalTransitionRecord) -> StoredTransition:
        validation = validate_transition(row)
        sequence_errors = self._sequence_errors(row)
        if sequence_errors:
            errors = [*validation.errors, *sequence_errors]
            if (
                row["training_eligible"]
                and "training_eligible_with_validation_errors" not in errors
            ):
                errors.append("training_eligible_with_validation_errors")
            validation = TransitionValidation(False, tuple(errors))
        payload = _normalized_for_storage(row, validation)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        connection = self._require_connection()
        with connection:
            connection.execute(
                """
                INSERT INTO transitions (
                    episode_id, step_id, observation_id, next_observation_id,
                    done, training_eligible, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["episode_id"],
                    payload["step_id"],
                    payload["observation"]["observation_id"],
                    payload["next_observation"]["observation_id"],
                    int(payload["done"]),
                    int(payload["training_eligible"]),
                    encoded,
                ),
            )
        return StoredTransition(payload, validation)

    def load_episode(self, episode_id: str) -> tuple[HierarchicalTransitionRecord, ...]:
        cursor = self._require_connection().execute(
            "SELECT payload_json FROM transitions WHERE episode_id = ? ORDER BY step_id",
            (episode_id,),
        )
        return tuple(
            cast(HierarchicalTransitionRecord, json.loads(encoded)) for (encoded,) in cursor
        )

    def episode_ids(self) -> tuple[str, ...]:
        cursor = self._require_connection().execute(
            "SELECT DISTINCT episode_id FROM transitions ORDER BY episode_id"
        )
        return tuple(str(row[0]) for row in cursor)

    def integrity(self) -> str:
        row = self._require_connection().execute("PRAGMA integrity_check").fetchone()
        return "unknown" if row is None else str(row[0])

    def count(self, *, training_only: bool = False) -> int:
        query = "SELECT COUNT(*) FROM transitions"
        if training_only:
            query += " WHERE training_eligible = 1"
        value = self._require_connection().execute(query).fetchone()
        if value is None:
            raise RuntimeError("TransitionStore count query failed")
        return int(value[0])

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> UnifiedTransitionStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
