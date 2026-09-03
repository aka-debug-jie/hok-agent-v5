from pathlib import Path

from hok_agent.hierarchical_p1v2 import DIRECTIONS, BranchSchedule, load_architecture


def test_p1v2_task_specific_branches_and_router_boundary() -> None:
    schedule, payload, digest = load_architecture(
        Path("configs/hierarchical_p1v2_architecture.json")
    )
    assert schedule == BranchSchedule(2, 10, 10)
    assert tuple(payload["branches"]["movement"]["outputs"]) == DIRECTIONS
    assert payload["branches"]["macro"]["training_independent"] is True
    assert payload["branches"]["movement"]["training_independent"] is True
    assert payload["branches"]["combat"]["trainable"] is False
    assert payload["router"]["movement_and_combat_may_execute_concurrently"] is True
    assert payload["claim_boundary"]["branch_training_allowed"] is False
    assert len(digest) == 64
