from pathlib import Path

from hok_agent.hierarchical_p1v2_movement_data import load_contract, normalize_move


def test_p1v2_movement_contract_and_move_only_normalization() -> None:
    config, payload, digest = load_contract(
        Path("configs/hierarchical_p1v2_movement_data.json")
    )
    assert normalize_move({"action_type": "move", "direction": "northeast"}) == "north_east"
    assert normalize_move({"action_type": "skill", "direction": "northeast"}) is None
    assert config.required_direction_count == 8 and len(digest) == 64
    assert payload["selection"]["skill_aim_as_movement_allowed"] is False
    assert payload["selection"]["wait_as_direction_allowed"] is False
    assert payload["selection"]["test_allowed"] is False
