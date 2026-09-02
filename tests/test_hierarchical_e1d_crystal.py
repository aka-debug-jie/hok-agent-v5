from __future__ import annotations

from pathlib import Path

import numpy as np

from hok_agent.hierarchical_e1d_crystal import detect_transition, load_contract


def test_crystal_consensus_detects_flash_and_persistent_change() -> None:
    config, _payload, _sha = load_contract(Path("configs/hierarchical_event_e1d_crystal.json"))
    frames = np.zeros((30, 128, 128, 3), dtype=np.uint8)
    frames[:15, 20:108, 30:98] = 100
    frames[15] = 255
    frames[16:, 20:108, 30:98] = 20
    times = np.arange(30, dtype=np.int64) * 200
    result = detect_transition(frames, times, config)
    assert result["accepted"] is True
    assert result["white_fraction"] > 0.55
