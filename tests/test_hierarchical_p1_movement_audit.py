from pathlib import Path

from hok_agent.hierarchical_p1_movement_audit import (
    display_rotation,
    load_contract,
    stable_directions,
)


def test_p1_movement_audit_contract_and_stability() -> None:
    config, payload, digest = load_contract(
        Path("configs/hierarchical_p1_movement_teacher_audit.json")
    )
    values = ["north", "north", "north", "east", None, "east", "east", "east"]
    assert stable_directions(values, 3) == [None, None, "north", None, None, None, None, "east"]
    assert config.segment_fractions == (0.25, 0.5, 0.75)
    assert len(digest) == 64
    assert payload["labels"]["wait_label_allowed"] is False
    assert payload["labels"]["semantic_accuracy_verified"] is False
    assert payload["sampling"]["video_test_allowed"] is False
    assert payload["repair_history"]["sampling_teacher_or_gate_changed"] is False


def test_display_matrix_rotation_is_applied_before_roi_crop() -> None:
    clockwise_90 = bytes.fromhex(
        "000000000000ffff00000000000001000000000000000000000000000000000000000040"
    )
    assert display_rotation(clockwise_90) == 90
