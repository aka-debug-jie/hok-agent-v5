from __future__ import annotations

import os

from hok_agent.contracts import EnvironmentKind, ResetRequest, StepRequest
from hok_agent.envs import LocalRpcClient, ProcessJsonTransport
from hok_agent.envs.mock import select_deterministic_smoke_action


def test_mock_process_transport_is_separate_from_learner() -> None:
    transport = ProcessJsonTransport.start_mock(max_steps_per_episode=8)
    try:
        assert transport.service_pid is not None
        assert transport.service_pid != os.getpid()
        client = LocalRpcClient(transport, expected_kind=EnvironmentKind.MOCK)
        assert client.health().environment.environment_kind is EnvironmentKind.MOCK
        reset = client.reset(
            ResetRequest(
                request_id="process-boundary",
                mode="1v1",
                seed=7,
                side="blue",
                hero_config={},
                opponent_config={},
                reward_config_version="mock-reward-v1",
                evaluation_mode=True,
            )
        )
        response = client.step(
            StepRequest(
                episode_id=reset.episode_id,
                expected_tick=reset.tick,
                action=select_deterministic_smoke_action(reset.legal_actions),
                action_schema_version=1,
            )
        )
        assert response.tick == reset.tick + 1
        assert client.close(reset.episode_id, "process_test").already_closed is False
        assert transport.is_alive
    finally:
        transport.close()
    assert not transport.is_alive
