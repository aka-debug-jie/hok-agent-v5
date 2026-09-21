from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from hok_agent import navigation_feedback_audit as audit
from hok_agent.mobile_testbed import _goal_navigation_contract

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK = ROOT / "game_rules" / "r0_feedback_contract_v1.json"
NAVIGATION = ROOT / "configs" / "movement_goal_navigation_store_v1.json"


def test_feedback_contract_loads_and_hashes() -> None:
    contract, sha = audit.load_r0_feedback_contract(FEEDBACK)
    assert len(sha) == 64
    names = {cast(dict, item)["name"] for item in cast(list, contract["signals"])}
    assert names == set(audit.R0_REQUIRED_SIGNALS)
    assert cast(dict, contract["claim_boundary"])["reward_allowed"] is False


def test_feedback_contract_rejects_tampering(tmp_path: Path) -> None:
    value = json.loads(FEEDBACK.read_text(encoding="utf-8"))
    value["claim_boundary"]["reward_allowed"] = True
    tampered = tmp_path / "r0.json"
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(audit.FeedbackAuditError):
        audit.load_r0_feedback_contract(tampered)

    value = json.loads(FEEDBACK.read_text(encoding="utf-8"))
    value["gates"].pop("minimum_motion_agreement")
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(audit.FeedbackAuditError):
        audit.load_r0_feedback_contract(tampered)

    value = json.loads(FEEDBACK.read_text(encoding="utf-8"))
    value["signals"] = [item for item in value["signals"] if item["name"] != "hero_motion"]
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(audit.FeedbackAuditError):
        audit.load_r0_feedback_contract(tampered)


def test_expected_dispatch_matches_the_store_vocabulary() -> None:
    assert audit._expected_dispatch("DOWN", "E") == "east"
    assert audit._expected_dispatch("MOVE", "NW") == "north_west"
    assert audit._expected_dispatch("UP", "STOP") == "wait"
    assert audit._expected_dispatch("NOOP", "STOP") is None
    with pytest.raises(audit.FeedbackAuditError):
        audit._expected_dispatch("MOVE", "STOP")


def _marker_frame(center_y: int, center_x: int, value: int = 200) -> np.ndarray:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[center_y - 5 : center_y + 5, center_x - 5 : center_x + 5] = (0, value, 0)
    return frame


def test_independent_position_finds_the_interior_marker() -> None:
    contract, _ = _goal_navigation_contract(NAVIGATION)
    boxes = audit._fixed_boxes(contract)
    position = audit._independent_position(_marker_frame(60, 70), contract, boxes)
    assert position is not None
    assert abs(position[0] - 59.5) < 1.0
    assert abs(position[1] - 69.5) < 1.0
    blank = np.zeros((128, 128, 3), dtype=np.uint8)
    assert audit._independent_position(blank, contract, boxes) is None


def test_independent_position_ignores_the_declared_ui_boxes() -> None:
    contract, _ = _goal_navigation_contract(NAVIGATION)
    boxes = audit._fixed_boxes(contract)
    assert boxes
    corner = _marker_frame(8, 120)
    assert audit._independent_position(corner, contract, boxes) is None


def test_motion_centroid_tracks_the_changed_region() -> None:
    contract, _ = _goal_navigation_contract(NAVIGATION)
    boxes = audit._fixed_boxes(contract)
    first = np.zeros((128, 128, 3), dtype=np.uint8)
    second = _marker_frame(50, 60)
    motion = audit._motion_centroid(first, second, boxes)
    assert motion is not None
    assert abs(motion[0] - 49.5) < 2.0
    assert abs(motion[1] - 59.5) < 2.0
    assert audit._motion_centroid(first, first, boxes) is None


def test_fixed_boxes_reads_both_declared_shapes() -> None:
    contract, _ = _goal_navigation_contract(NAVIGATION)
    boxes = audit._fixed_boxes(contract)
    assert audit._numbers(contract["excluded_ui_xyxy"], label="x") in boxes
    assert len(boxes) >= 2
    with pytest.raises(audit.FeedbackAuditError):
        audit._numbers([1, 2, 3], label="bad")


RESPONSE = ROOT / "game_rules" / "r0_response_task_contract_v1.json"


def test_response_contract_loads_and_hashes() -> None:
    contract, sha = audit.load_response_task_contract(RESPONSE)
    assert len(sha) == 64
    names = {cast(dict, item)["name"] for item in cast(list, contract["signals"])}
    assert names == set(audit.RESPONSE_REQUIRED_SIGNALS)
    assert cast(dict, contract["independence"])["duplication_guard"]


def test_response_contract_rejects_tampering(tmp_path: Path) -> None:
    value = json.loads(RESPONSE.read_text(encoding="utf-8"))
    value["gates"].pop("minimum_positive_response_fraction")
    tampered = tmp_path / "r0r.json"
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(audit.FeedbackAuditError):
        audit.load_response_task_contract(tampered)

    value = json.loads(RESPONSE.read_text(encoding="utf-8"))
    value["independence"].pop("duplication_guard")
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(audit.FeedbackAuditError):
        audit.load_response_task_contract(tampered)


def test_direction_vectors_are_unit_and_axis_correct() -> None:
    vectors = audit.DIRECTION_VECTORS
    assert vectors["north"] == (-1.0, 0.0)
    assert vectors["east"] == (0.0, 1.0)
    for name, vector in vectors.items():
        norm = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
        assert abs(norm - 1.0) < 1e-9, name


def test_patch_displacement_measures_a_shifted_patch() -> None:
    current = np.zeros((128, 128, 3), dtype=np.uint8)
    current[50:70, 50:70] = 200
    current[55:65, 55:65] = 40
    following = np.zeros((128, 128, 3), dtype=np.uint8)
    following[54:74, 47:67] = 200
    following[59:69, 52:62] = 40
    measured = audit._patch_displacement(current, following, (60.0, 60.0))
    assert measured is not None
    assert measured[0] > 0.9
    assert abs(measured[1] - 4.0) <= 1.0
    assert abs(measured[2] + 3.0) <= 1.0


def test_patch_displacement_rejects_an_edge_patch() -> None:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    assert audit._patch_displacement(frame, frame, (4.0, 4.0)) is None
    assert audit._patch_displacement(frame, frame, (60.0, 60.0)) is None
