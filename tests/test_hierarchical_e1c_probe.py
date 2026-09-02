from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from hok_agent.hierarchical_e1c_probe import (
    load_probe_contract,
    load_probe_report,
    ordinal_time_only_score,
)


def test_ordinal_time_only_baseline_exposes_triplet_confound(tmp_path: Path) -> None:
    shard = tmp_path / "split.npz"
    np.savez_compressed(
        shard,
        rgb_sequence=np.zeros((6, 2, 2, 2, 3), dtype=np.uint8),
        label=np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int8),
        session_id=np.asarray(["a", "a", "a", "b", "b", "b"]),
    )
    report = ordinal_time_only_score(shard)
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["labels_visible_to_predictor"] is False


def test_probe_contract_blocks_training_until_time_preflight_passes() -> None:
    payload, digest = load_probe_contract(Path("configs/hierarchical_event_e1c_probe.json"))
    assert len(digest) == 64
    assert payload["claim_boundary"]["visual_training_allowed"] is False


def test_probe_report_is_self_verifying(tmp_path: Path) -> None:
    contract, contract_sha = load_probe_contract(Path("configs/hierarchical_event_e1c_probe.json"))
    payload: dict[str, object] = {
        "schema_version": "hok-agent-hierarchical-event-e1c-probe-report-v1",
        "contract_sha256": contract_sha,
        "status": "E1C_PROBE_BLOCKED_TIME_CONFOUND",
        "visual_training_started": False,
        "reward_allowed": False,
        "video_test_opened": False,
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["report_sha256"] = hashlib.sha256(data).hexdigest()
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    assert contract["status"] == "frozen_time_confound_preflight"
    assert load_probe_report(report, Path("configs/hierarchical_event_e1c_probe.json")) == payload
