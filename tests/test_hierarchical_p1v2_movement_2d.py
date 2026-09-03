from pathlib import Path

from hok_agent.hierarchical_p1v2_movement_2d import (
    load_contract,
    world_delta,
)


def test_p1v2_movement_2d_contract_and_ego_direction() -> None:
    config, payload, digest = load_contract(
        Path("configs/hierarchical_p1v2_movement_2d_source.json")
    )
    assert world_delta("north_east", "blue") == (1, -1)
    assert world_delta("north_east", "red") == (-1, 1)
    assert config.sequence_frames == 16 and len(digest) == 64
    assert payload["claim_boundary"]["capability"] == "local_visible_target_approach_only"
    assert payload["claim_boundary"]["lane_strategy_verified"] is False
    assert payload["curriculum"]["test_allowed"] is False
