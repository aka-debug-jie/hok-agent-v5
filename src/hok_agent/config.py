"""Legacy-style versioned YAML loading and bounded configuration validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class ConfigError(ValueError):
    """Raised for a missing or invalid project configuration file."""


CONFIG_VERSION = 1


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot load YAML config {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigError(f"YAML config {path} must contain an object at its root")
    mapping = cast(dict[str, Any], loaded)
    version = mapping.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigError(f"YAML config {path} must declare integer root version {CONFIG_VERSION}")
    if version != CONFIG_VERSION:
        raise ConfigError(f"YAML config {path} has unsupported version {version}; expected {CONFIG_VERSION}")
    return mapping


def stable_file_hash(path: Path) -> str:
    """Return the short content hash used by configuration validation output."""

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def validate_config_tree(config_dir: Path) -> list[str]:
    """Validate V5's versioned YAML tree and JSON Schema syntax without execution."""

    if not config_dir.is_dir():
        raise ConfigError(f"config directory does not exist: {config_dir}")
    yaml_paths = sorted(config_dir.rglob("*.yaml"))
    if not yaml_paths:
        raise ConfigError(f"no YAML configs found in {config_dir}")

    messages: list[str] = []
    for path in yaml_paths:
        load_yaml_mapping(path)
        messages.append(f"YAML OK: {path.relative_to(config_dir)} [{stable_file_hash(path)}]")

    schema_dir = config_dir.parent / "schemas"
    for path in sorted(schema_dir.glob("*.schema.json")):
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(schema, dict):
                raise ConfigError(f"JSON schema {path} must contain an object at its root")
            Draft202012Validator.check_schema(schema)
        except (OSError, json.JSONDecodeError, SchemaError) as error:
            raise ConfigError(f"invalid JSON schema {path}: {error}") from error
        messages.append(f"Schema OK: {path.name} [{stable_file_hash(path)}]")
    return messages


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
