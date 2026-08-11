from __future__ import annotations

from pathlib import Path

import pytest

from hok_agent.cli import main
from hok_agent.config import ConfigError, load_yaml_mapping, validate_config_tree

ROOT = Path(__file__).resolve().parents[1]


def test_versioned_yaml_loader_rejects_missing_root_version(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("runtime_inputs: {}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="integer root version 1"):
        load_yaml_mapping(path)


def test_versioned_yaml_loader_rejects_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "unknown-version.yaml"
    path.write_text("version: 2\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unsupported version 2; expected 1"):
        load_yaml_mapping(path)


def test_validate_config_tree_accepts_tracked_versioned_configs() -> None:
    messages = validate_config_tree(ROOT / "configs")

    assert any(message.startswith("YAML OK: program_v1.yaml") for message in messages)
    assert any(message.startswith("YAML OK: runtime_inputs_v1.yaml") for message in messages)
    assert any(message.startswith("Schema OK: run_manifest.schema.json") for message in messages)


def test_validate_config_cli_uses_the_project_config_tree(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate-config", "--config-dir", str(ROOT / "configs")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "YAML OK: program_v1.yaml" in captured.out
