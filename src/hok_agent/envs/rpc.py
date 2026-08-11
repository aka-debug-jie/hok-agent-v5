"""Versioned JSON RPC stub with an explicit transport boundary."""

from __future__ import annotations

import json
from typing import Protocol, cast

from hok_agent.contracts import (
    CloseResponse,
    EnvironmentKind,
    HealthResponse,
    LicenseStatus,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from hok_agent.contracts.types import ContractValidationError, JsonObject
from hok_agent.envs.base import EnvironmentService, ProtocolViolation

PROTOCOL_VERSION = 1


class RpcError(RuntimeError):
    """Returned by a service when a transport-valid request fails closed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ServiceIdentityError(RpcError):
    """A service did not satisfy the execution safety identity requirements."""


class JsonTransport(Protocol):
    def request(self, payload: str) -> str: ...


class LocalRpcServer:
    """In-process server stub; network transports can reuse this exact dispatcher."""

    def __init__(self, service: EnvironmentService) -> None:
        self._service = service

    def handle(self, raw_request: str) -> str:
        try:
            request = self._decode_envelope(raw_request)
            result = self._dispatch(request["method"], request["payload"])
            return self._encode({"protocol_version": PROTOCOL_VERSION, "ok": True, "result": result})
        except (ContractValidationError, ProtocolViolation, RpcError, ValueError) as error:
            code = error.code if isinstance(error, (ProtocolViolation, RpcError)) else "INVALID_REQUEST"
            return self._encode(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "ok": False,
                    "error": {"code": code, "message": str(error)},
                }
            )

    @staticmethod
    def _encode(value: object) -> str:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _decode_envelope(raw_request: str) -> dict[str, object]:
        try:
            parsed = cast(object, json.loads(raw_request))
        except json.JSONDecodeError as error:
            raise RpcError("MALFORMED_JSON", "request is not valid JSON") from error
        if not isinstance(parsed, dict):
            raise RpcError("INVALID_ENVELOPE", "request envelope must be an object")
        envelope = cast(dict[str, object], parsed)
        if set(envelope) != {"protocol_version", "method", "payload"}:
            raise RpcError("INVALID_ENVELOPE", "request envelope keys do not match protocol")
        if envelope["protocol_version"] != PROTOCOL_VERSION:
            raise RpcError("VERSION_MISMATCH", "request protocol version is unsupported")
        if not isinstance(envelope["method"], str):
            raise RpcError("INVALID_ENVELOPE", "request method must be a string")
        if not isinstance(envelope["payload"], dict):
            raise RpcError("INVALID_ENVELOPE", "request payload must be an object")
        return envelope

    def _dispatch(self, method: object, payload: object) -> JsonObject:
        if not isinstance(method, str) or not isinstance(payload, dict):
            raise RpcError("INVALID_ENVELOPE", "method and payload are invalid")
        if method == "health":
            if payload:
                raise RpcError("INVALID_REQUEST", "health request payload must be empty")
            return self._service.health().to_dict()
        if method == "reset":
            return self._service.reset(ResetRequest.from_dict(payload)).to_dict()
        if method == "step":
            return self._service.step(StepRequest.from_dict(payload)).to_dict()
        if method == "close":
            if set(payload) != {"episode_id", "stop_reason"}:
                raise RpcError("INVALID_REQUEST", "close request keys do not match protocol")
            episode_id = payload["episode_id"]
            stop_reason = payload["stop_reason"]
            if not isinstance(episode_id, str) or not isinstance(stop_reason, str):
                raise RpcError("INVALID_REQUEST", "close request values must be strings")
            return self._service.close(episode_id, stop_reason).to_dict()
        raise RpcError("UNKNOWN_METHOD", f"unsupported method {method}")


class InProcessJsonTransport:
    """Test-only transport; it serializes every call exactly as an external client would."""

    def __init__(self, server: LocalRpcServer) -> None:
        self._server = server

    def request(self, payload: str) -> str:
        return self._server.handle(payload)


class LocalRpcClient:
    """Learner-facing client with no direct reference to service state."""

    def __init__(self, transport: JsonTransport) -> None:
        self._transport = transport

    def health(self) -> HealthResponse:
        return HealthResponse.from_dict(self._call("health", {}))

    def reset(self, request: ResetRequest) -> ResetResponse:
        return ResetResponse.from_dict(self._call("reset", request.to_dict()))

    def step(self, request: StepRequest) -> StepResponse:
        return StepResponse.from_dict(self._call("step", request.to_dict()))

    def close(self, episode_id: str, stop_reason: str) -> CloseResponse:
        return CloseResponse.from_dict(self._call("close", {"episode_id": episode_id, "stop_reason": stop_reason}))

    def _call(self, method: str, payload: JsonObject) -> object:
        raw_response = self._transport.request(
            json.dumps(
                {"protocol_version": PROTOCOL_VERSION, "method": method, "payload": payload},
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        try:
            response = cast(object, json.loads(raw_response))
        except json.JSONDecodeError as error:
            raise RpcError("MALFORMED_RESPONSE", "service returned malformed JSON") from error
        if not isinstance(response, dict):
            raise RpcError("INVALID_RESPONSE", "service response must be an object")
        envelope = cast(dict[str, object], response)
        if envelope.get("protocol_version") != PROTOCOL_VERSION:
            raise RpcError("VERSION_MISMATCH", "service response protocol version is unsupported")
        if envelope.get("ok") is True:
            if set(envelope) != {"protocol_version", "ok", "result"}:
                raise RpcError("INVALID_RESPONSE", "success envelope keys do not match protocol")
            return envelope["result"]
        error_payload = envelope.get("error")
        if not isinstance(error_payload, dict):
            raise RpcError("INVALID_RESPONSE", "failed response lacks error object")
        code = error_payload.get("code")
        message = error_payload.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            raise RpcError("INVALID_RESPONSE", "failed response error fields are invalid")
        raise RpcError(code, message)


def require_safe_service_identity(health: HealthResponse, expected_kind: EnvironmentKind) -> None:
    """Fail closed before any action reaches a service.

    M0 accepts the mock service. A future GameCore connection additionally requires
    a valid license status from the authorized service itself.
    """

    if not health.ready:
        raise ServiceIdentityError("SERVICE_NOT_READY", "environment service is not ready")
    if health.environment.environment_kind is not expected_kind:
        raise ServiceIdentityError("IDENTITY_MISMATCH", "environment kind does not match requested target")
    if expected_kind is EnvironmentKind.HOK_GAMECORE and health.license_status is not LicenseStatus.VALID:
        raise ServiceIdentityError("LICENSE_NOT_VALID", "authorized GameCore service license is not valid")
