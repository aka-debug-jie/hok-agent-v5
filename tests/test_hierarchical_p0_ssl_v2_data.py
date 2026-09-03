from pathlib import Path

import numpy as np

from hok_agent.hierarchical_p0_ssl_v2_data import (
    load_contract,
    middle_permutation,
    selected_train_sessions,
)


def test_v2_data_contract_and_middle_permutation() -> None:
    config, payload, digest = load_contract(
        Path("configs/hierarchical_p0_temporal_ssl_v2_data.json")
    )
    order = middle_permutation(16, 7)
    assert order[0] == 0 and order[-1] == 15
    assert sorted(order.tolist()) == list(range(16))
    assert not np.array_equal(order[1:-1], np.arange(1, 15))
    assert config.required_sessions == 103 and config.windows_per_session == 16
    assert len(digest) == 64
    assert payload["selection"]["video_dev_allowed"] is False
    assert payload["selection"]["video_test_allowed"] is False


def test_v2_session_selection_is_train_only() -> None:
    manifest: dict[str, object] = {
        "sessions": [
            {"session_hash": "train-b", "split": "train"},
            {"session_hash": "dev", "split": "dev"},
            {"session_hash": "train-a", "split": "train"},
            {"session_hash": "test", "split": "test"},
        ]
    }
    assert selected_train_sessions(manifest) == ("train-a", "train-b")
