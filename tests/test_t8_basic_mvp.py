from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.t8_basic_mvp import (
    T8BasicMVPError,
    _enemy_probability,
    _sample_indices,
    _screen_valid,
    verify_t8_basic_mvp_contract,
    verify_t8_basic_mvp_shadow_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t8_basic_mvp_v1.json"
SHADOW_CONTRACT = ROOT / "configs/t8_basic_mvp_shadow_v1.json"


def test_basic_mvp_contract_is_single_action_and_non_promoting() -> None:
    result = verify_t8_basic_mvp_contract(CONTRACT)
    assert result["status"] == "PASSED"
    assert result["semantic_accuracy_verified"] is False
    assert result["video_test_accessed"] is False
    assert result["control_output"] is False
    assert result["device_input_allowed"] is False


def test_basic_mvp_rehashed_rate_drift_fails(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["maximum_candidates_per_minute"] = 11
    value["contract_sha256"] = hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "contract_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(T8BasicMVPError, match="frozen contract values differ"):
        verify_t8_basic_mvp_contract(path)


def test_basic_mvp_shadow_is_five_minutes_and_zero_control() -> None:
    base = verify_t8_basic_mvp_contract(CONTRACT)
    shadow = verify_t8_basic_mvp_shadow_contract(SHADOW_CONTRACT, str(base["contract_sha256"]))
    assert shadow["run_seconds"] == 300.0
    assert shadow["minimum_candidates"] == 1
    assert shadow["input_commands_sent"] == 0
    assert shadow["control_output"] is False
    assert shadow["device_input_allowed"] is False


def test_basic_mvp_screen_and_enemy_rules_are_deterministic() -> None:
    black = np.zeros((128, 128, 3), dtype=np.uint8)
    gray = np.full((128, 128, 3), 127, dtype=np.uint8)
    gameplay = gray.copy()
    gameplay[40:46, 40:80] = (220, 30, 25)
    assert not _screen_valid(black, 8.0, 5.0)
    assert not _screen_valid(gray, 8.0, 5.0)
    assert _screen_valid(gameplay, 8.0, 5.0)
    first = _enemy_probability(np.stack((black, gameplay)))
    second = _enemy_probability(np.stack((black, gameplay)))
    assert np.array_equal(first, second)
    assert first[0] < 0.2
    assert first[1] > 0.8


def test_basic_mvp_sampling_is_monotonic_and_fixed_rate() -> None:
    timestamps = np.arange(0, 2001, 100, dtype=np.int64)
    indices = _sample_indices(timestamps, 5)
    assert indices.tolist() == list(range(0, 21, 2))
