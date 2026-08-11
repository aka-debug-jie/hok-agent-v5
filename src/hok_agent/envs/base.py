"""Abstract boundary for environment services.

The learner must only see this interface; a legacy upstream SDK belongs behind an
independent service process and never in the learner environment.
"""

from __future__ import annotations

from typing import Protocol

from hok_agent.contracts import (
    CloseResponse,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)


class ProtocolViolation(RuntimeError):
    """A fail-closed service contract error with a machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class EnvironmentService(Protocol):
    """The entire allowed V5 environment-service API at M0."""

    def health(self) -> HealthResponse: ...

    def reset(self, request: ResetRequest) -> ResetResponse: ...

    def step(self, request: StepRequest) -> StepResponse: ...

    def close(self, episode_id: str, stop_reason: str) -> CloseResponse: ...
