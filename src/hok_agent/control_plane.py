"""Fail-closed gate for operations that would use an authorized GameCore service.

This module deliberately separates a service's runtime license metadata from the
external authorization record.  A caller must satisfy both before it can use a
GameCore transport, formal evaluation, or promotion path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from hok_agent.config import ConfigError, load_yaml_mapping, required_mapping, required_string
from hok_agent.contracts import LicenseStatus


class ControlledOperation(StrEnum):
    """Operations that remain unavailable until external access is confirmed."""

    GAMECORE_TRANSPORT = "gamecore_transport"
    FORMAL_EVALUATION = "formal_evaluation"
    PROMOTION = "promotion"


class ExternalAccessDenied(RuntimeError):
    """A control-plane gate rejected a sensitive operation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ExternalAccessGate:
    """Non-secret authorization state loaded from the project control-plane config.

    ``external_authorization_evidence_ref`` is an opaque reference to evidence
    stored outside Git.  It must never contain a license, binary, key, account,
    or credential value.
    """

    current_status: str
    connection_status: str
    external_authorization_attested: bool
    external_authorization_evidence_ref: str | None
    locked_operations: frozenset[ControlledOperation]

    @classmethod
    def from_yaml(cls, path: Path) -> ExternalAccessGate:
        return cls.from_mapping(load_yaml_mapping(path))

    @classmethod
    def from_mapping(cls, document: dict[str, Any]) -> ExternalAccessGate:
        program = required_mapping(document.get("program"), "program")
        environments = required_mapping(document.get("environments"), "environments")
        authority = required_mapping(environments.get("strategy_authority"), "strategy_authority")
        locks = required_mapping(document.get("locks"), "locks")

        if authority.get("kind") != "hok_gamecore":
            raise ConfigError("strategy_authority.kind must be hok_gamecore")

        attested = authority.get("external_authorization_attested")
        if not isinstance(attested, bool):
            raise ConfigError("strategy_authority.external_authorization_attested must be a boolean")

        evidence_ref = authority.get("external_authorization_evidence_ref")
        if evidence_ref is not None and (not isinstance(evidence_ref, str) or not evidence_ref):
            raise ConfigError(
                "strategy_authority.external_authorization_evidence_ref must be null or a non-empty string"
            )

        locked: set[ControlledOperation] = set()
        for operation in ControlledOperation:
            value = locks.get(operation.value)
            if not isinstance(value, bool):
                raise ConfigError(f"locks.{operation.value} must be a boolean")
            if value:
                locked.add(operation)

        return cls(
            current_status=required_string(program.get("current_status"), "program.current_status"),
            connection_status=required_string(
                authority.get("connection_status"), "strategy_authority.connection_status"
            ),
            external_authorization_attested=attested,
            external_authorization_evidence_ref=evidence_ref,
            locked_operations=frozenset(locked),
        )

    def require(self, operation: ControlledOperation, *, runtime_license_status: LicenseStatus) -> None:
        """Allow an operation only after every independent safety condition holds."""

        self.require_local(operation)
        self.require_runtime_license(runtime_license_status)

    def require_local(self, operation: ControlledOperation) -> None:
        """Check only local, non-service state before opening a transport or probing health."""

        if self.current_status != "EXTERNAL_ACCESS_CONFIRMED" or self.connection_status != "ready":
            raise ExternalAccessDenied(
                "WAITING_EXTERNAL",
                "external GameCore access is not confirmed by the project control plane",
            )
        if not self.external_authorization_attested:
            raise ExternalAccessDenied(
                "AUTHORIZATION_NOT_ATTESTED",
                "external GameCore authorization has not been attested",
            )
        if self.external_authorization_evidence_ref is None:
            raise ExternalAccessDenied(
                "AUTHORIZATION_EVIDENCE_MISSING",
                "external GameCore authorization has no non-secret evidence reference",
            )
        if operation in self.locked_operations:
            raise ExternalAccessDenied(
                "OPERATION_LOCKED",
                f"{operation.value} remains locked by the project control plane",
            )

    @staticmethod
    def require_runtime_license(runtime_license_status: LicenseStatus) -> None:
        """Validate service metadata only after a locally authorized health response exists."""

        if runtime_license_status is not LicenseStatus.VALID:
            raise ExternalAccessDenied(
                "LICENSE_NOT_VALID",
                "authorized GameCore service runtime license is not valid",
            )
