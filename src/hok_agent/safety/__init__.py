"""Fail-closed safety gates for the V5 codebase."""

from hok_agent.safety.scan import SafetyViolation, scan_repository

__all__ = ["SafetyViolation", "scan_repository"]
