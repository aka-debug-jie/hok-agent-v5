from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.hierarchical_e1c_clip import (
    E1cClipError,
    load_clip_contract,
    load_clip_report,
    select_window_indices,
)

CONTRACT = Path("configs/hierarchical_event_e1c_clip.json")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_e1c_clip_selects_unique_nearest_frames_without_anchor() -> None:
    config, _payload, _sha = load_clip_contract(CONTRACT)
    timestamps = np.arange(0, 40_001, 200, dtype=np.int64)
    indices = select_window_indices(timestamps, 40_000, 7, 3, 16, 500)
    assert len(indices) == 16 and len(set(indices.tolist())) == 16
    assert timestamps[indices].max() <= 37_000
    assert config.windows_seconds_before_anchor[-1] == (7, 3)


def test_e1c_clip_rejects_sparse_timestamps() -> None:
    timestamps = np.asarray([0, 10_000, 20_000, 30_000, 40_000], dtype=np.int64)
    with pytest.raises(E1cClipError, match="continuous target frames"):
        select_window_indices(timestamps, 40_000, 7, 3, 16, 500)


def test_e1c_clip_report_is_self_verifying_and_non_promoting(tmp_path: Path) -> None:
    _config, _payload, contract_sha = load_clip_contract(CONTRACT)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-hierarchical-event-e1c-clip-report-v1",
        "contract_sha256": contract_sha,
        "status": "E1C_CLIP_MATERIALIZATION_FAILED",
        "model_input_fields": ["rgb_sequence"],
        "video_test_opened": False,
        "reward_allowed": False,
        "promotion_allowed": False,
    }
    payload["report_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    report = tmp_path / "report.json"
    report.write_bytes(_canonical(payload) + b"\n")
    assert load_clip_report(report, CONTRACT) == payload
    report.write_text(report.read_text().replace("FAILED", "PASSED"))
    with pytest.raises(E1cClipError, match="report is invalid"):
        load_clip_report(report, CONTRACT)

