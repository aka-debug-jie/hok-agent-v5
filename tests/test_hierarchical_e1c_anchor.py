from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hok_agent.hierarchical_e1_terminal import OcrToken, TerminalLabel
from hok_agent.hierarchical_e1c_anchor import (
    E1cAnchorError,
    classify_anchor,
    load_anchor_contract,
    load_anchor_report,
)

CONTRACT = Path("configs/hierarchical_event_e1c_anchor.json")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_e1c_anchor_requires_allowlisted_result_evidence() -> None:
    config, _payload, _sha = load_anchor_contract(CONTRACT)
    result = classify_anchor(
        (OcrToken("数据表现", 0.99), OcrToken("本局奖励", 0.99)),
        config,
    )
    win = classify_anchor((OcrToken("胜利", 0.95),), config)
    unrelated = classify_anchor((OcrToken("player-name", 1.0),), config)
    assert result.anchored is True and result.outcome == TerminalLabel.GAME_END_UNKNOWN
    assert win.anchored is True and win.outcome == TerminalLabel.WIN
    assert unrelated.anchored is False and unrelated.allowlisted_hits == ()


def test_e1c_report_is_self_verifying_and_keeps_time_out_of_labels(tmp_path: Path) -> None:
    _config, _contract, contract_sha = load_anchor_contract(CONTRACT)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-hierarchical-event-e1c-anchor-report-v1",
        "contract_sha256": contract_sha,
        "status": "E1C_RESULT_PAGE_ANCHOR_PREFLIGHT_FAILED",
        "video_end_used_as_label": False,
        "anchor_frame_allowed_in_future_model_input": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "video_test_opened": False,
    }
    payload["report_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    report = tmp_path / "report.json"
    report.write_bytes(_canonical(payload) + b"\n")
    assert load_anchor_report(report, CONTRACT) == payload
    report.write_text(report.read_text().replace("FAILED", "PASSED"))
    with pytest.raises(E1cAnchorError, match="report is invalid"):
        load_anchor_report(report, CONTRACT)

