from __future__ import annotations

import json
from pathlib import Path

import pytest

from hok_agent.artifacts.hashing import sha256_json
from hok_agent.config import load_yaml_mapping, required_mapping
from hok_agent.contracts import (
    EnvironmentIdentity,
    EnvironmentKind,
    HealthResponse,
    LicenseStatus,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from hok_agent.control_plane import ControlledOperation, ExternalAccessDenied, ExternalAccessGate
from hok_agent.envs import InProcessJsonTransport, LocalRpcClient, LocalRpcServer, MockEnvironment
from hok_agent.envs.rpc import PROTOCOL_VERSION, ServiceIdentityError

ROOT = Path(__file__).resolve().parents[1]


def test_program_config_keeps_external_access_locked() -> None:
    config = load_yaml_mapping(ROOT / "configs" / "program_v1.yaml")
    program = required_mapping(config["program"], "program")
    environments = required_mapping(config["environments"], "environments")
    strategy_authority = required_mapping(environments["strategy_authority"], "strategy_authority")
    locks = required_mapping(config["locks"], "locks")

    assert program["current_milestone"] == "M1"
    assert program["current_task"] == "TASK-005"
    assert program["current_status"] == "WAITING_EXTERNAL"
    assert strategy_authority["kind"] == "hok_gamecore"
    assert strategy_authority["connection_status"] == "waiting_external"
    assert strategy_authority["external_authorization_attested"] is False
    assert strategy_authority["external_authorization_evidence_ref"] is None
    assert locks["gamecore_transport"] is True
    assert locks["formal_evaluation"] is True
    assert locks["behavior_cloning"] is True
    assert locks["promotion"] is True


def _confirmed_control_plane(*, locked: frozenset[ControlledOperation] = frozenset()) -> dict[str, object]:
    locks = {operation.value: operation in locked for operation in ControlledOperation}
    return {
        "program": {"current_status": "EXTERNAL_ACCESS_CONFIRMED"},
        "environments": {
            "strategy_authority": {
                "kind": "hok_gamecore",
                "connection_status": "ready",
                "external_authorization_attested": True,
                "external_authorization_evidence_ref": "external-record:authorization-001",
            }
        },
        "locks": locks,
    }


def test_current_control_plane_rejects_all_sensitive_operations() -> None:
    gate = ExternalAccessGate.from_yaml(ROOT / "configs" / "program_v1.yaml")

    for operation in ControlledOperation:
        with pytest.raises(ExternalAccessDenied) as caught:
            gate.require(operation, runtime_license_status=LicenseStatus.VALID)
        assert caught.value.code == "WAITING_EXTERNAL"


def test_control_plane_requires_explicit_unlock_after_external_confirmation() -> None:
    gate = ExternalAccessGate.from_mapping(
        _confirmed_control_plane(locked=frozenset({ControlledOperation.PROMOTION}))
    )

    gate.require(ControlledOperation.GAMECORE_TRANSPORT, runtime_license_status=LicenseStatus.VALID)
    gate.require(ControlledOperation.FORMAL_EVALUATION, runtime_license_status=LicenseStatus.VALID)
    with pytest.raises(ExternalAccessDenied) as caught:
        gate.require(ControlledOperation.PROMOTION, runtime_license_status=LicenseStatus.VALID)
    assert caught.value.code == "OPERATION_LOCKED"


class _LicensedGameCoreLikeMock(MockEnvironment):
    """In-memory test double only; it never contacts a real GameCore service."""

    def __init__(self, *, max_steps_per_episode: int) -> None:
        super().__init__(max_steps_per_episode=max_steps_per_episode)
        self.health_calls = 0
        self.reset_calls = 0
        self.step_calls = 0

    def health(self) -> HealthResponse:
        self.health_calls += 1
        return HealthResponse(
            protocol_version=1,
            environment=EnvironmentIdentity(
                schema_version=1,
                environment_kind=EnvironmentKind.HOK_GAMECORE,
                service_version="test-double-service",
                sdk_version="test-double-sdk",
                gamecore_build="test-double-build",
                mapping_hash=sha256_json({"test_double": "licensed_gamecore_identity"}),
            ),
            license_status=LicenseStatus.VALID,
            supported_modes=("1v1",),
            supported_heroes=("test_double",),
            ready=True,
        )

    def reset(self, request: ResetRequest) -> ResetResponse:
        self.reset_calls += 1
        return super().reset(request)

    def step(self, request: StepRequest) -> StepResponse:
        self.step_calls += 1
        return super().step(request)


def test_current_control_plane_blocks_a_runtime_valid_gamecore_test_double() -> None:
    gate = ExternalAccessGate.from_yaml(ROOT / "configs" / "program_v1.yaml")
    service = _LicensedGameCoreLikeMock(max_steps_per_episode=8)
    transport = InProcessJsonTransport(
        LocalRpcServer(service, expected_kind=EnvironmentKind.HOK_GAMECORE)
    )

    with pytest.raises(ServiceIdentityError) as caught:
        LocalRpcClient(
            transport,
            expected_kind=EnvironmentKind.HOK_GAMECORE,
            external_access_gate=gate,
        )

    assert caught.value.code == "WAITING_EXTERNAL"
    assert service.health_calls == 0
    assert service.reset_calls == 0
    assert service.step_calls == 0


def test_server_refuses_gamecore_action_when_current_control_plane_is_waiting() -> None:
    gate = ExternalAccessGate.from_yaml(ROOT / "configs" / "program_v1.yaml")
    service = _LicensedGameCoreLikeMock(max_steps_per_episode=8)
    server = LocalRpcServer(
        service,
        expected_kind=EnvironmentKind.HOK_GAMECORE,
        external_access_gate=gate,
    )
    request = ResetRequest(
        request_id="server-gate",
        mode="1v1",
        seed=1,
        side="blue",
        hero_config={},
        opponent_config={},
        reward_config_version="test-double-reward-v1",
        evaluation_mode=True,
    )

    for method, payload in (("health", {}), ("reset", request.to_dict()), ("step", {})):
        response = json.loads(
            server.handle(
                json.dumps(
                    {"protocol_version": PROTOCOL_VERSION, "method": method, "payload": payload},
                    sort_keys=True,
                )
            )
        )
        assert response["ok"] is False
        assert response["error"]["code"] == "WAITING_EXTERNAL"
    assert service.health_calls == 0
    assert service.reset_calls == 0
    assert service.step_calls == 0
