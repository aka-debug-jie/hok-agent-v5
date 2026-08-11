"""Fail-closed gate for the inactive optional GameCore calibration track.

This module deliberately separates a service's runtime license metadata from the
external authorization record.  A caller must satisfy both before it can use a
GameCore transport, external evaluation, or external promotion path.  It does not
gate or authorize the project-owned PixelArena route.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from hok_agent.config import ConfigError, load_yaml_mapping, required_mapping, required_string
from hok_agent.contracts import LicenseStatus


class ControlledOperation(StrEnum):
    """Optional GameCore operations unavailable until external access is confirmed."""

    GAMECORE_TRANSPORT = "gamecore_transport"
    GAMECORE_EVALUATION = "gamecore_evaluation"
    GAMECORE_PROMOTION = "gamecore_promotion"


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

    enabled: bool
    external_status: str
    connection_status: str
    external_authorization_attested: bool
    external_authorization_evidence_ref: str | None
    locked_operations: frozenset[ControlledOperation]

    @classmethod
    def from_yaml(cls, path: Path) -> ExternalAccessGate:
        return cls.from_mapping(load_yaml_mapping(path))

    @classmethod
    def from_mapping(cls, document: dict[str, Any]) -> ExternalAccessGate:
        environments = required_mapping(document.get("environments"), "environments")
        optional_authority = required_mapping(
            environments.get("optional_external_calibration"),
            "optional_external_calibration",
        )
        locks = required_mapping(document.get("locks"), "locks")

        if optional_authority.get("kind") != "hok_gamecore":
            raise ConfigError("optional_external_calibration.kind must be hok_gamecore")

        enabled = optional_authority.get("enabled")
        if not isinstance(enabled, bool):
            raise ConfigError("optional_external_calibration.enabled must be a boolean")

        attested = optional_authority.get("external_authorization_attested")
        if not isinstance(attested, bool):
            raise ConfigError(
                "optional_external_calibration.external_authorization_attested must be a boolean"
            )

        evidence_ref = optional_authority.get("external_authorization_evidence_ref")
        if evidence_ref is not None and (not isinstance(evidence_ref, str) or not evidence_ref):
            raise ConfigError(
                "optional_external_calibration.external_authorization_evidence_ref must be null "
                "or a non-empty string"
            )

        locked: set[ControlledOperation] = set()
        for operation in ControlledOperation:
            value = locks.get(operation.value)
            if not isinstance(value, bool):
                raise ConfigError(f"locks.{operation.value} must be a boolean")
            if value:
                locked.add(operation)

        return cls(
            enabled=enabled,
            external_status=required_string(optional_authority.get("status"), "optional_external_calibration.status"),
            connection_status=required_string(
                optional_authority.get("connection_status"),
                "optional_external_calibration.connection_status",
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

        if self.external_status != "EXTERNAL_ACCESS_CONFIRMED" or self.connection_status != "ready":
            raise ExternalAccessDenied(
                "WAITING_EXTERNAL",
                "external GameCore access is not confirmed by the project control plane",
            )
        if not self.enabled:
            raise ExternalAccessDenied(
                "OPTIONAL_TRACK_DISABLED",
                "optional GameCore calibration track is disabled",
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
