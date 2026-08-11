from __future__ import annotations

import json
from pathlib import Path

import pytest

from hok_agent.config import ConfigError
from hok_agent.contracts import LicenseStatus
from hok_agent.control_plane import ControlledOperation, ExternalAccessDenied, ExternalAccessGate
from hok_agent.preflight import collect_preflight, load_runtime_input_config


def test_preflight_without_external_inputs_is_waiting_external(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.delenv("HOK_GAMECORE_PATH", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("HOK_LICENSE_PATH", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("GAMECORE_LICENSE_PATH", raising=False)  # type: ignore[attr-defined]
    report = collect_preflight(tmp_path)
    assert report["status"] == "WAITING_EXTERNAL"
    scope = report["scope"]
    assert isinstance(scope, dict)
    assert scope["license_contents_read"] is False


def test_runtime_input_config_reuses_versioned_yaml_without_recording_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "external" / "hok_env"
    checkout.mkdir(parents=True)
    gamecore = tmp_path / "external" / "gamecore"
    gamecore.mkdir(parents=True)
    license_file = tmp_path / "external" / "license.dat"
    license_file.write_text("not-read", encoding="utf-8")
    runtime_config = tmp_path / "runtime_inputs.yaml"
    runtime_config.write_text(
        "\n".join(
            [
                "version: 1",
                "runtime_inputs:",
                "  upstream_checkout_env: V5_TEST_HOK_CHECKOUT",
                "  gamecore_path_env: V5_TEST_GAMECORE_PATH",
                "  license_path_env: V5_TEST_LICENSE_PATH",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("V5_TEST_HOK_CHECKOUT", str(checkout))
    monkeypatch.setenv("V5_TEST_GAMECORE_PATH", str(gamecore))
    monkeypatch.setenv("V5_TEST_LICENSE_PATH", str(license_file))

    report = collect_preflight(tmp_path, runtime_config=runtime_config)

    assert report["status"] == "PREFLIGHT_INPUTS_PRESENT_UNVERIFIED"
    runtime_inputs = report["runtime_inputs"]
    assert isinstance(runtime_inputs, dict)
    assert runtime_inputs["config_loaded"] is True
    assert runtime_inputs["environment_variable_names"] == {
        "upstream_checkout": "V5_TEST_HOK_CHECKOUT",
        "gamecore_path": "V5_TEST_GAMECORE_PATH",
        "license_path": "V5_TEST_LICENSE_PATH",
    }
    report_text = json.dumps(report)
    assert str(checkout) not in report_text
    assert str(gamecore) not in report_text
    assert str(license_file) not in report_text

    gate = ExternalAccessGate.from_yaml(Path(__file__).parents[1] / "configs" / "program_v1.yaml")
    with pytest.raises(ExternalAccessDenied) as caught:
        gate.require(ControlledOperation.GAMECORE_TRANSPORT, runtime_license_status=LicenseStatus.VALID)
    assert caught.value.code == "WAITING_EXTERNAL"


def test_runtime_input_config_rejects_authorization_fields(tmp_path: Path) -> None:
    runtime_config = tmp_path / "invalid_runtime_inputs.yaml"
    runtime_config.write_text(
        "\n".join(
            [
                "version: 1",
                "runtime_inputs:",
                "  upstream_checkout_env: HOK_ENV_CHECKOUT",
                "  gamecore_path_env: HOK_GAMECORE_PATH",
                "  license_path_env: HOK_LICENSE_PATH",
                "  external_authorization_attested: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="runtime_inputs must contain exactly"):
        load_runtime_input_config(runtime_config)


def test_runtime_input_config_rejects_unknown_version(tmp_path: Path) -> None:
    runtime_config = tmp_path / "unknown-version-runtime-inputs.yaml"
    runtime_config.write_text(
        "\n".join(
            [
                "version: 2",
                "runtime_inputs:",
                "  upstream_checkout_env: HOK_ENV_CHECKOUT",
                "  gamecore_path_env: HOK_GAMECORE_PATH",
                "  license_path_env: HOK_LICENSE_PATH",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unsupported version 2; expected 1"):
        load_runtime_input_config(runtime_config)
