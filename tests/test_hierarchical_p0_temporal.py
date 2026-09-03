from pathlib import Path

import numpy as np

from hok_agent.hierarchical_p0_temporal import load_contract, middle_shuffle


def test_middle_shuffle_preserves_endpoints_and_frame_multiset() -> None:
    config, payload, digest = load_contract(Path("configs/hierarchical_p0_temporal_data.json"))
    frames = np.arange(16 * 2, dtype=np.uint8).reshape(16, 1, 2, 1)
    shuffled = middle_shuffle(frames, 17)
    assert np.array_equal(shuffled[0], frames[0])
    assert np.array_equal(shuffled[-1], frames[-1])
    assert sorted(shuffled[:, 0, 0, 0].tolist()) == sorted(frames[:, 0, 0, 0].tolist())
    assert not np.array_equal(shuffled[1:-1], frames[1:-1])
    assert config.sequence_frames == 16 and len(digest) == 64
    assert payload["selection"]["test_allowed"] is False
