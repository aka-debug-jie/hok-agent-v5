from pathlib import Path

import numpy as np

from hok_agent.hierarchical_e1d_clip import load_contract, select_matched_negative


def test_event_centered_clip_selects_non_flash_negative() -> None:
    config, payload, _sha = load_contract(Path("configs/hierarchical_event_e1d_clip.json"))
    frames = np.full((100, 128, 128, 3), 80, dtype=np.uint8)
    frames[80] = 255
    frames[81:] = 20
    negative = select_matched_negative(frames, 80, config)
    assert negative + config.after < 80 - config.exclusion_radius
    assert payload["model_input"]["timestamp_allowed"] is False
