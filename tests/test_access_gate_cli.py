from __future__ import annotations

from pathlib import Path

import pytest
from pytest import CaptureFixture

from hok_agent.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("operation", ["gamecore_transport", "formal_evaluation", "promotion"])
def test_access_gate_cli_refuses_runtime_valid_gamecore_while_waiting_external(
    capsys: CaptureFixture[str], operation: str
) -> None:
    exit_code = main(
        [
            "access-gate",
            "--config",
            str(ROOT / "configs" / "program_v1.yaml"),
            "--operation",
            operation,
            "--runtime-license-status",
            "valid",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "WAITING_EXTERNAL" in captured.err
