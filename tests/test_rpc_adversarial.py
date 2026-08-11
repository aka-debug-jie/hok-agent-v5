from __future__ import annotations

import pytest

from hok_agent.contracts import EnvironmentKind, ResetRequest, ResetResponse, StepRequest, StepResponse
from hok_agent.envs import InProcessJsonTransport, LocalRpcClient, LocalRpcServer, MockEnvironment
from hok_agent.envs.mock import select_deterministic_smoke_action
from hok_agent.envs.rpc import RpcError


class _BadResetRequestIdMock(MockEnvironment):
    def __init__(self, *, max_steps_per_episode: int = 16, inject_error_on_step: int | None = None) -> None:
        super().__init__(
            max_steps_per_episode=max_steps_per_episode, inject_error_on_step=inject_error_on_step
        )
        self._has_issued_reset = False

    def reset(self, request: ResetRequest) -> ResetResponse:
        response = super().reset(request)
        if not self._has_issued_reset:
            self._has_issued_reset = True
            return response
        return ResetResponse(
            request_id=f"spoofed-{request.request_id}",
            episode_id=f"spoofed-{response.episode_id}",
            tick=response.tick,
            observation=response.observation,
            legal_actions=response.legal_actions,
            reward_component_names=response.reward_component_names,
            environment=response.environment,
            replay_identity=response.replay_identity,
        )


def _build_client(service: MockEnvironment) -> LocalRpcClient:
    return LocalRpcClient(
        InProcessJsonTransport(LocalRpcServer(service, expected_kind=EnvironmentKind.MOCK)),
        expected_kind=EnvironmentKind.MOCK,
    )


def _build_request(seed: int) -> ResetRequest:
    return ResetRequest(
        request_id=f"adversarial-{seed}",
        mode="1v1",
        seed=seed,
        side="blue",
        hero_config={},
        opponent_config={},
        reward_config_version="mock-reward-v1",
        evaluation_mode=True,
    )


def test_reset_response_request_id_mismatch_closes_all_known_sessions() -> None:
    client = _build_client(_BadResetRequestIdMock(max_steps_per_episode=8))
    first_reset = client.reset(_build_request(1))
    assert first_reset.episode_id is not None

    with pytest.raises(RpcError) as caught:
        client.reset(_build_request(2))
    assert caught.value.code == "REQUEST_ID_MISMATCH"
    with pytest.raises(RpcError) as poisoned_health:
        client.health()
    assert poisoned_health.value.code == "CLIENT_POISONED"
    with pytest.raises(RpcError) as poisoned_step:
        client.step(
            StepRequest(
                episode_id=first_reset.episode_id,
                expected_tick=first_reset.tick,
                action=select_deterministic_smoke_action(first_reset.legal_actions),
                action_schema_version=1,
            )
        )
    assert poisoned_step.value.code == "CLIENT_POISONED"
    with pytest.raises(RpcError) as poisoned:
        client.reset(_build_request(3))
    assert poisoned.value.code == "CLIENT_POISONED"
    assert client.close(first_reset.episode_id, "after_reset_mismatch").already_closed is True


class _BadStepEpisodeMock(MockEnvironment):
    def step(self, request: StepRequest) -> StepResponse:
        response = super().step(request)
        return StepResponse(
            observation=response.observation,
            reward=response.reward,
            episode_id=f"spoofed-{request.episode_id}",
            terminal=response.terminal,
            truncated=response.truncated,
            outcome=response.outcome,
            legal_actions=response.legal_actions,
            tick=response.tick,
            replay_hash=response.replay_hash,
            error_code=response.error_code,
        )


def test_step_response_episode_id_mismatch_fail_closed() -> None:
    client = _build_client(_BadStepEpisodeMock(max_steps_per_episode=8))
    reset = client.reset(_build_request(3))
    assert reset.request_id == "adversarial-3"
    request = StepRequest(
        episode_id=reset.episode_id,
        expected_tick=reset.tick,
        action=select_deterministic_smoke_action(reset.legal_actions),
        action_schema_version=1,
    )

    with pytest.raises(RpcError) as caught:
        client.step(request)
    assert caught.value.code == "EPISODE_ID_MISMATCH"
    with pytest.raises(RpcError) as poisoned_health:
        client.health()
    assert poisoned_health.value.code == "CLIENT_POISONED"
    with pytest.raises(RpcError) as poisoned_step:
        client.step(request)
    assert poisoned_step.value.code == "CLIENT_POISONED"
    with pytest.raises(RpcError) as poisoned:
        client.reset(_build_request(4))
    assert poisoned.value.code == "CLIENT_POISONED"
    assert client.close(reset.episode_id, "after_step_mismatch").already_closed is True
