"""Small, typed loader for versioned YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


class ConfigError(ValueError):
    """Raised for a missing or invalid project configuration file."""


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot load YAML config {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigError(f"YAML config {path} must contain an object at its root")
    return cast(dict[str, Any], loaded)


def required_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def required_int(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a non-empty string")
    return value
