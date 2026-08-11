"""Deterministic M0 test double, not a simulation of Honor of Kings capability."""

from __future__ import annotations

from dataclasses import dataclass, field

from hok_agent.artifacts.hashing import sha256_json
from hok_agent.contracts import (
    CloseResponse,
    EnvironmentIdentity,
    EnvironmentKind,
    EpisodeOutcome,
    FactorizedAction,
    HealthResponse,
    LegalActionSet,
    LicenseStatus,
    PublicEntity,
    PublicObservation,
    ResetRequest,
    ResetResponse,
    RewardVector,
    StepRequest,
    StepResponse,
)
from hok_agent.contracts.types import ActionType, JsonValue, OutcomeResult, TargetType
from hok_agent.envs.base import ProtocolViolation

REWARD_COMPONENT_NAMES = (
    "win_crystal",
    "structure",
    "economy",
    "survival",
    "time_step",
    "upstream_raw",
)


@dataclass(slots=True)
class _MockSession:
    episode_id: str
    seed: int
    side: str
    max_steps: int
    tick: int = 0
    enemy_health: float = 0.0
    player_health: float = 12.0
    position: float = 0.0
    previous_action: FactorizedAction | None = None
    closed: bool = False
    action_history: list[dict[str, JsonValue]] = field(default_factory=list)


class MockEnvironment:
    """A fully deterministic, explicitly labelled ``environment_kind=mock`` service."""

    def __init__(self, *, max_steps_per_episode: int = 16, inject_error_on_step: int | None = None) -> None:
        if max_steps_per_episode < 2:
            raise ValueError("max_steps_per_episode must be at least 2")
        self._max_steps_per_episode = max_steps_per_episode
        self._inject_error_on_step = inject_error_on_step
        self._episode_counter = 0
        self._sessions: dict[str, _MockSession] = {}
        self._identity = EnvironmentIdentity(
            schema_version=1,
            environment_kind=EnvironmentKind.MOCK,
            service_version="mock-service-0.1.0",
            sdk_version=None,
            gamecore_build=None,
            mapping_hash=sha256_json(
                {
                    "adapter": "deterministic_mock",
                    "action_schema_version": 1,
                    "observation_schema_version": 1,
                }
            ),
        )

    def health(self) -> HealthResponse:
        return HealthResponse(
            protocol_version=1,
            environment=self._identity,
            license_status=LicenseStatus.NOT_APPLICABLE,
            supported_modes=("1v1",),
            supported_heroes=("mock_fixed_hero",),
            ready=True,
        )

    def reset(self, request: ResetRequest) -> ResetResponse:
        if request.mode != "1v1":
            raise ProtocolViolation("UNSUPPORTED_MODE", "mock service supports only 1v1")
        self._episode_counter += 1
        episode_id = f"mock-{request.seed}-{request.side}-{self._episode_counter}"
        session = _MockSession(
            episode_id=episode_id,
            seed=request.seed,
            side=request.side,
            max_steps=self._max_steps_per_episode,
            enemy_health=float(4 + request.seed % 3),
            position=float(request.seed % 5),
        )
        self._sessions[episode_id] = session
        return ResetResponse(
            episode_id=episode_id,
            tick=session.tick,
            observation=self._observation(session),
            legal_actions=self._legal_actions(session),
            reward_component_names=REWARD_COMPONENT_NAMES,
            environment=self._identity,
            replay_identity=self._reset_replay_identity(request),
        )

    def step(self, request: StepRequest) -> StepResponse:
        session = self._session_for(request.episode_id)
        if session.closed:
            raise ProtocolViolation("SESSION_CLOSED", "cannot step a closed mock episode")
        if request.expected_tick != session.tick:
            raise ProtocolViolation("STALE_TICK", "expected_tick does not match current mock tick")
        legal_actions = self._legal_actions(session)
        if not legal_actions.contains(request.action):
            raise ProtocolViolation("ILLEGAL_ACTION", "mock service rejected an illegal factorized action")
        if self._inject_error_on_step is not None and session.tick == self._inject_error_on_step:
            raise ProtocolViolation("INJECTED_ERROR", "injected mock protocol error")

        rewards = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        rewards["time_step"] = -0.01
        if request.action.action_type is ActionType.MOVE:
            assert request.action.direction is not None
            session.position += -1.0 if request.action.direction % 2 else 1.0
            rewards["economy"] = 0.02
        elif request.action.action_type is ActionType.ATTACK:
            session.enemy_health -= 2.0
            rewards["structure"] = 0.5
        elif request.action.action_type is ActionType.SKILL:
            session.enemy_health -= 3.0
            rewards["structure"] = 0.8
        else:
            rewards["survival"] = 0.01

        session.tick += 1
        if session.enemy_health > 0.0:
            session.player_health -= 0.25
            rewards["survival"] += 0.05
        session.previous_action = request.action
        session.action_history.append(request.action.to_dict())

        terminal = session.enemy_health <= 0.0 or session.player_health <= 0.0
        truncated = not terminal and session.tick >= session.max_steps
        outcome = self._outcome(session, terminal=terminal, truncated=truncated)
        if outcome is not None and outcome.result is OutcomeResult.WIN:
            rewards["win_crystal"] = 10.0
            rewards["upstream_raw"] = 1.0
        elif outcome is not None and outcome.result is OutcomeResult.LOSS:
            rewards["upstream_raw"] = -1.0

        return StepResponse(
            observation=self._observation(session),
            reward=RewardVector(components=rewards),
            terminal=terminal,
            truncated=truncated,
            outcome=outcome,
            legal_actions=LegalActionSet(schema_version=1, actions=())
            if terminal or truncated
            else self._legal_actions(session),
            tick=session.tick,
            replay_hash=self._replay_hash(session),
            error_code=None,
        )

    def close(self, episode_id: str, stop_reason: str) -> CloseResponse:
        session = self._session_for(episode_id)
        already_closed = session.closed
        session.closed = True
        return CloseResponse(
            episode_id=episode_id,
            stop_reason=stop_reason if stop_reason else "caller_requested",
            already_closed=already_closed,
        )

    def _session_for(self, episode_id: str) -> _MockSession:
        session = self._sessions.get(episode_id)
        if session is None:
            raise ProtocolViolation("UNKNOWN_EPISODE", "episode_id is not known by mock service")
        return session

    def _observation(self, session: _MockSession) -> PublicObservation:
        return PublicObservation(
            schema_version=1,
            tick=session.tick,
            self_state={"health": session.player_health, "position": session.position},
            visible_entities=(
                PublicEntity(
                    public_key="enemy_visible_0",
                    category="enemy_structure",
                    team_relation="enemy",
                    x=1.0,
                    y=0.0,
                    health=max(session.enemy_health, 0.0),
                    visible=True,
                ),
            ),
            global_features={"arena_phase": float(session.tick % 2), "visible_enemy_count": 1.0},
            previous_action=session.previous_action,
        )

    def _legal_actions(self, session: _MockSession) -> LegalActionSet:
        if session.closed or session.enemy_health <= 0.0 or session.player_health <= 0.0:
            return LegalActionSet(schema_version=1, actions=())
        direction = (session.seed + session.tick) % 8
        return LegalActionSet(
            schema_version=1,
            actions=(
                FactorizedAction(
                    schema_version=1,
                    macro_intent="hold",
                    action_type=ActionType.WAIT,
                    target_type=TargetType.NONE,
                    target_key=None,
                    direction=None,
                    skill_slot=None,
                    item_slot=None,
                    auxiliary_parameter=None,
                ),
                FactorizedAction(
                    schema_version=1,
                    macro_intent="advance",
                    action_type=ActionType.MOVE,
                    target_type=TargetType.NONE,
                    target_key=None,
                    direction=direction,
                    skill_slot=None,
                    item_slot=None,
                    auxiliary_parameter=None,
                ),
                FactorizedAction(
                    schema_version=1,
                    macro_intent="engage",
                    action_type=ActionType.ATTACK,
                    target_type=TargetType.ENEMY,
                    target_key="enemy_visible_0",
                    direction=None,
                    skill_slot=None,
                    item_slot=None,
                    auxiliary_parameter=None,
                ),
                FactorizedAction(
                    schema_version=1,
                    macro_intent="engage",
                    action_type=ActionType.SKILL,
                    target_type=TargetType.ENEMY,
                    target_key="enemy_visible_0",
                    direction=None,
                    skill_slot=0,
                    item_slot=None,
                    auxiliary_parameter=1.0,
                ),
            ),
        )

    def _reset_replay_identity(self, request: ResetRequest) -> str:
        return sha256_json(
            {
                "environment_mapping_hash": self._identity.mapping_hash,
                "mode": request.mode,
                "seed": request.seed,
                "side": request.side,
                "hero_config": dict(request.hero_config),
                "opponent_config": dict(request.opponent_config),
                "reward_config_version": request.reward_config_version,
            }
        )

    def _replay_hash(self, session: _MockSession) -> str:
        return sha256_json(
            {
                "episode_seed": session.seed,
                "side": session.side,
                "tick": session.tick,
                "enemy_health": session.enemy_health,
                "player_health": session.player_health,
                "position": session.position,
                "actions": session.action_history,
            }
        )

    @staticmethod
    def _outcome(session: _MockSession, *, terminal: bool, truncated: bool) -> EpisodeOutcome | None:
        if session.enemy_health <= 0.0:
            return EpisodeOutcome(
                result=OutcomeResult.WIN,
                reason="mock_enemy_structure_depleted",
                terminal=True,
                truncated=False,
            )
        if session.player_health <= 0.0:
            return EpisodeOutcome(
                result=OutcomeResult.LOSS,
                reason="mock_player_health_depleted",
                terminal=True,
                truncated=False,
            )
        if truncated:
            return EpisodeOutcome(
                result=OutcomeResult.DRAW,
                reason="mock_step_limit",
                terminal=False,
                truncated=True,
            )
        if terminal:
            raise AssertionError("mock terminal state must have a terminal outcome")
        return None


def select_deterministic_smoke_action(legal_actions: LegalActionSet) -> FactorizedAction:
    """Fixed non-learning policy used only to exercise the M0 service path."""

    for action in legal_actions.actions:
        if action.action_type is ActionType.SKILL:
            return action
    if not legal_actions.actions:
        raise ProtocolViolation("EMPTY_LEGAL_ACTIONS", "cannot sample from an empty legal action set")
    return legal_actions.actions[0]
