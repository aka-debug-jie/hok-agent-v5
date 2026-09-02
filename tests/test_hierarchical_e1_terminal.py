from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hok_agent.hierarchical_e1_terminal import (
    OcrToken,
    TerminalAuditError,
    TerminalLabel,
    _select_sessions,
    classify_ocr_tokens,
    load_terminal_contract,
    load_terminal_report,
)

CONTRACT = Path("configs/hierarchical_event_e1_terminal.json")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_terminal_ocr_classifier_uses_only_allowlisted_confident_tokens() -> None:
    config, _payload, _sha = load_terminal_contract(CONTRACT)
    win = classify_ocr_tokens((OcrToken("胜利", 0.95), OcrToken("player-name", 1.0)), config)
    loss = classify_ocr_tokens((OcrToken("DEFEAT", 0.9),), config)
    result = classify_ocr_tokens(
        (OcrToken("数据表现", 0.99), OcrToken("本局奖励", 0.98)),
        config,
    )
    ignored = classify_ocr_tokens((OcrToken("胜利", 0.2),), config)
    conflict = classify_ocr_tokens((OcrToken("胜利", 0.9), OcrToken("失败", 0.9)), config)
    assert win.label == TerminalLabel.WIN and "player-name" not in win.allowlisted_hits
    assert loss.label == TerminalLabel.LOSS
    assert result.label == TerminalLabel.GAME_END_UNKNOWN
    assert ignored.label == TerminalLabel.IN_PROGRESS
    assert conflict.label == TerminalLabel.CONFLICT


def test_terminal_selection_is_deterministic_and_never_selects_test() -> None:
    splits = {
        **{f"train-{index}": "train" for index in range(10)},
        **{f"dev-{index}": "dev" for index in range(6)},
        **{f"test-{index}": "test" for index in range(6)},
    }
    first = _select_sessions(splits, "0" * 64, 8, 4)
    second = _select_sessions(splits, "0" * 64, 8, 4)
    assert first == second
    assert len(first["train"]) == 8 and len(first["dev"]) == 4
    assert all(not session.startswith("test-") for values in first.values() for session in values)


def test_terminal_report_is_self_verifying_and_non_promoting(tmp_path: Path) -> None:
    _config, _contract, contract_sha = load_terminal_contract(CONTRACT)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-hierarchical-event-e1-terminal-audit-v1",
        "contract_sha256": contract_sha,
        "status": "E1B_WEAK_LABEL_COVERAGE_FAILED",
        "reward_allowed": False,
        "promotion_allowed": False,
        "video_test_opened": False,
    }
    payload["report_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    report = tmp_path / "report.json"
    report.write_bytes(_canonical(payload) + b"\n")
    assert load_terminal_report(report, CONTRACT) == payload
    report.write_text(report.read_text().replace("FAILED", "PASSED"))
    with pytest.raises(TerminalAuditError, match="report is invalid"):
        load_terminal_report(report, CONTRACT)

