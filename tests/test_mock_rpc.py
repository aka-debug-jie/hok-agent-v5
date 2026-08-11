from __future__ import annotations

import pytest

from hok_agent.contracts import EnvironmentKind, ResetRequest, StepRequest
from hok_agent.envs import InProcessJsonTransport, LocalRpcClient, LocalRpcServer, MockEnvironment
from hok_agent.envs.mock import select_deterministic_smoke_action
from hok_agent.envs.rpc import RpcError, ServiceIdentityError, require_safe_service_identity


def _client() -> LocalRpcClient:
    return LocalRpcClient(InProcessJsonTransport(LocalRpcServer(MockEnvironment(max_steps_per_episode=12))))


def _play(seed: int) -> str:
    client = _client()
    reset = client.reset(
        ResetRequest(
            request_id=f"request-{seed}",
            mode="1v1",
            seed=seed,
            side="blue",
            hero_config={"hero": "mock_fixed_hero"},
            opponent_config={"policy": "mock_stationary"},
            reward_config_version="mock-reward-v1",
            evaluation_mode=True,
        )
    )
    assert len(reset.legal_actions.actions) >= 3
    tick = reset.tick
    legal_actions = reset.legal_actions
    while True:
        response = client.step(
            StepRequest(
                episode_id=reset.episode_id,
                expected_tick=tick,
                action=select_deterministic_smoke_action(legal_actions),
                action_schema_version=1,
            )
        )
        assert response.tick == tick + 1
        tick = response.tick
        legal_actions = response.legal_actions
        if response.terminal or response.truncated:
            assert response.outcome is not None
            break
    assert client.close(reset.episode_id, "test_complete").already_closed is False
    assert client.close(reset.episode_id, "test_complete").already_closed is True
    return response.replay_hash


def test_mock_fixed_seed_replay_is_exact() -> None:
    assert _play(101) == _play(101)


def test_rpc_rejects_stale_tick_and_identity_mismatch() -> None:
    client = _client()
    health = client.health()
    require_safe_service_identity(health, EnvironmentKind.MOCK)
    with pytest.raises(ServiceIdentityError):
        require_safe_service_identity(health, EnvironmentKind.PIXELARENA)
    reset = client.reset(
        ResetRequest(
            request_id="stale",
            mode="1v1",
            seed=1,
            side="red",
            hero_config={},
            opponent_config={},
            reward_config_version="mock-reward-v1",
            evaluation_mode=True,
        )
    )
    with pytest.raises(RpcError) as caught:
        client.step(
            StepRequest(
                episode_id=reset.episode_id,
                expected_tick=reset.tick + 1,
                action=select_deterministic_smoke_action(reset.legal_actions),
                action_schema_version=1,
            )
        )
    assert caught.value.code == "STALE_TICK"
