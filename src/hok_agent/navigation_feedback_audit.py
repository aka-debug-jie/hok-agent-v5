"""R0 feedback audit: check the device navigation feedback against independent references.

R0 has to answer one question before any training budget is spent: is the feedback that a policy
would be trained against reliable, and is it checked by something other than the same detector
that produced it? This module reads a recorded store-bound device episode set, recomputes the
position with a second derivation and a motion feature from the persisted minimap views, and
cross-checks the recorded actions and the terminal label against their own provenance.

Honest scope, carried into the report:

- Two derivations of the same minimap RGB are implementation-independent, not semantically
  independent. There is no independent map reference on this route, so the report never claims
  semantic position accuracy.
- The abort condition (death or ended screen) reads an ROI that is not persisted in the derived
  views, so it cannot be re-derived from the episode evidence. It is reported as unverifiable
  rather than checked, and it must not become an R1 reward component.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np

from hok_agent.mobile_navigation_store import JOYSTICK_TO_STORE_DIRECTION
from hok_agent.mobile_testbed import (
    _goal_navigation_components,
    _goal_navigation_contract,
    _new_large_output,
)
from hok_agent.transition_store import HierarchicalTransitionRecord, UnifiedTransitionStore

R0_FEEDBACK_SCHEMA = "hok-agent-r0-feedback-contract-v1"
NAVIGATION_FEEDBACK_AUDIT_SCHEMA = "hok-agent-navigation-feedback-audit-v1"
R0_REQUIRED_SIGNALS = (
    "hero_position",
    "hero_motion",
    "action_dispatch",
    "terminal",
    "death_or_ended",
)
R0_REQUIRED_GATES = (
    "minimum_position_coverage",
    "maximum_reference_l1_median_px",
    "maximum_reference_l1_p95_px",
    "minimum_reference_agreement",
    "minimum_motion_agreement",
    "minimum_action_agreement",
    "minimum_terminal_agreement",
    "maximum_exact_agreement_fraction",
)
EXACT_AGREEMENT_EPSILON = 1e-6
MOTION_CHANGE_THRESHOLD = 30
REFERENCE_TOLERANCE_PX = 3.0
MOTION_TOLERANCE_PX = 6.0
INTERIOR_SLICE = (22, 106)


class FeedbackAuditError(ValueError):
    """Raised when the R0 feedback contract or its audit inputs are invalid."""


def load_r0_feedback_contract(path: Path) -> tuple[dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    signals = payload.get("signals")
    gates = payload.get("gates")
    names = {
        cast(dict[str, object], item)["name"] for item in cast(list[object], signals or [])
    }
    if (
        payload.get("schema_version") != R0_FEEDBACK_SCHEMA
        or not isinstance(signals, list)
        or not isinstance(gates, dict)
        or names != set(R0_REQUIRED_SIGNALS)
        or any(key not in gates for key in R0_REQUIRED_GATES)
        or cast(dict[str, object], payload.get("claim_boundary", {})).get("reward_allowed")
        is not False
    ):
        raise FeedbackAuditError("r0 feedback contract differs")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return payload, digest


def _numbers(value: object, *, label: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list):
        raise FeedbackAuditError(f"{label} is not xyxy")
    entries = [int(cast(float, entry)) for entry in cast(list[object], value)]
    if len(entries) != 4:
        raise FeedbackAuditError(f"{label} is not xyxy")
    return (entries[0], entries[1], entries[2], entries[3])


def _fixed_boxes(contract: dict[str, object]) -> list[tuple[int, int, int, int]]:
    """The frozen cue declares excluded_ui_xyxy as one box and the extension as a list of boxes."""
    boxes = [_numbers(contract.get("excluded_ui_xyxy"), label="excluded_ui_xyxy")]
    extension = contract.get("hero_cue_extension")
    if isinstance(extension, dict):
        extra = extension.get("fixed_ui_boxes_xyxy")
        if isinstance(extra, list):
            boxes.extend(
                _numbers(item, label="fixed_ui_boxes_xyxy")
                for item in cast(list[object], extra)
            )
    return boxes


def _blank_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    result = frame.copy()
    for x0, y0, x1, y1 in boxes:
        result[y0:y1, x0:x1] = 0
    return result


def _green_mask(frame: np.ndarray, contract: dict[str, object]) -> np.ndarray:
    color = cast(dict[str, object], contract["color"])
    rgb = frame.astype(np.int16)
    return (
        (rgb[..., 1] > int(cast(int, color["green_minimum"])))
        & (rgb[..., 1] - rgb[..., 0] > int(cast(int, color["green_red_margin"])))
        & (rgb[..., 1] - rgb[..., 2] > int(cast(int, color["green_blue_margin"])))
    )


def _independent_position(
    minimap: np.ndarray, contract: dict[str, object], boxes: list[tuple[int, int, int, int]]
) -> tuple[float, float] | None:
    """Second derivation: largest interior green component, no temporal gating or red pairing."""
    interior = _blank_boxes(_green_mask(minimap, contract), boxes)[
        INTERIOR_SLICE[0] : INTERIOR_SLICE[1], INTERIOR_SLICE[0] : INTERIOR_SLICE[1]
    ]
    candidates = [
        (size, mean_y, mean_x)
        for size, mean_y, mean_x, height, width in _goal_navigation_components(interior)
        if 20 <= size <= 140 and 7 <= height <= 24 and 7 <= width <= 24
    ]
    if not candidates:
        return None
    _size, mean_y, mean_x = max(candidates)
    return (mean_y + INTERIOR_SLICE[0], mean_x + INTERIOR_SLICE[0])


def _motion_centroid(
    current: np.ndarray, following: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> tuple[float, float] | None:
    """Independent feature: largest inter-observation change region, colour-agnostic."""
    change = np.abs(following.astype(np.int16) - current.astype(np.int16)).max(axis=2)
    mask = _blank_boxes((change > MOTION_CHANGE_THRESHOLD).astype(np.uint8), boxes)
    components = _goal_navigation_components(mask.astype(bool))
    if not components:
        return None
    _size, mean_y, mean_x, _height, _width = max(components)
    return (mean_y, mean_x)


def _minimap(frame_root: Path, ref: str) -> np.ndarray:
    with np.load(frame_root / ref, allow_pickle=False) as saved:
        return np.ascontiguousarray(saved["minimap"])


def _expected_dispatch(command: str, applied: str) -> str | None:
    if command in {"DOWN", "MOVE"}:
        for name, token in JOYSTICK_TO_STORE_DIRECTION.items():
            if token == applied and name != "wait":
                return name
        raise FeedbackAuditError("applied movement has no joystick direction")
    if command == "UP":
        return "wait"
    return None


def _step_rows(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    paths = sorted(run_root.glob("episode-*/steps.jsonl"))
    if (run_root / "steps.jsonl").exists():
        paths.append(run_root / "steps.jsonl")
    for path in paths:
        episode_id = _episode_id_for(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = cast(dict[str, object], json.loads(line))
            if episode_id is not None and "episode_id" not in row:
                row["episode_id"] = episode_id
            rows.append(row)
    return rows


def _episode_id_for(steps_path: Path) -> str | None:
    summary = steps_path.parent / "summary.json"
    if not summary.exists():
        return None
    payload = cast(dict[str, object], json.loads(summary.read_text(encoding="utf-8")))
    episode_id = payload.get("episode_id")
    return episode_id if isinstance(episode_id, str) else None


def run_navigation_feedback_audit(
    *,
    feedback_contract_path: Path,
    navigation_contract_path: Path,
    store_path: Path,
    frame_root: Path,
    output_dir: Path,
    qa_samples: int = 12,
) -> dict[str, object]:
    feedback, feedback_sha = load_r0_feedback_contract(feedback_contract_path)
    navigation, navigation_sha = _goal_navigation_contract(navigation_contract_path)
    gates = cast(dict[str, object], feedback["gates"])
    boxes = _fixed_boxes(navigation)
    targets = [
        (
            float(cast(float, cast(list[object], item)[0])),
            float(cast(float, cast(list[object], item)[1])),
        )
        for item in cast(list[object], navigation["targets_minimap_xy"])
    ]
    tolerance = float(cast(float, navigation["arrival_tolerance_pixels"]))
    output = _new_large_output(output_dir)
    with UnifiedTransitionStore(store_path) as store:
        episode_ids = store.episode_ids()
        episodes: dict[str, tuple[HierarchicalTransitionRecord, ...]] = {
            episode_id: store.load_episode(episode_id) for episode_id in episode_ids
        }
        integrity = store.integrity()

    derivations: list[dict[str, object]] = []
    qa_frames: list[np.ndarray] = []
    qa_markers: list[list[tuple[float, float, str]]] = []
    qa_labels: list[str] = []
    for episode_id in episode_ids:
        rows = episodes[episode_id]
        for index, record in enumerate(rows):
            observation = _minimap(frame_root, record["observation"]["frame_bundle_ref"])
            following = _minimap(frame_root, record["next_observation"]["frame_bundle_ref"])
            independent = _independent_position(observation, navigation, boxes)
            motion = _motion_centroid(observation, following, boxes)
            derivations.append(
                {
                    "episode_id": episode_id,
                    "step_id": int(record["step_id"]),
                    "independent_position": None
                    if independent is None
                    else [independent[0], independent[1]],
                    "motion_position": None if motion is None else [motion[0], motion[1]],
                    "done": bool(record["done"]),
                }
            )
            if independent is not None and len(qa_frames) < qa_samples:
                markers: list[tuple[float, float, str]] = [(independent[0], independent[1], "b")]
                if motion is not None:
                    markers.append((motion[0], motion[1], "motion"))
                qa_frames.append(observation)
                qa_markers.append(markers)
                qa_labels.append(f"{episode_id[-2:]}#{index}")

    lookup = {
        (cast(str, item["episode_id"]), cast(int, item["step_id"])): item
        for item in derivations
    }
    steps = _step_rows(store_path.parent)
    agreement: list[float] = []
    without_reference = 0
    for step_row in steps:
        position = step_row.get("position")
        if position is None:
            continue
        candidate = lookup.get(
            (cast(str, step_row.get("episode_id")), cast(int, step_row.get("step_id", -1)))
        )
        if candidate is None or candidate["independent_position"] is None:
            without_reference += 1
            continue
        recorded_xy = cast(list[float], position)
        independent_xy = cast(list[float], candidate["independent_position"])
        agreement.append(
            abs(float(recorded_xy[0]) - independent_xy[0])
            + abs(float(recorded_xy[1]) - independent_xy[1])
        )
    motion_defined = sum(1 for item in derivations if item["motion_position"] is not None)
    motion_agreed = sum(
        1
        for item in derivations
        if item["motion_position"] is not None
        and item["independent_position"] is not None
        and (
            abs(
                float(cast(list[float], item["motion_position"])[0])
                - float(cast(list[float], item["independent_position"])[0])
            )
            + abs(
                float(cast(list[float], item["motion_position"])[1])
                - float(cast(list[float], item["independent_position"])[1])
            )
        )
        <= MOTION_TOLERANCE_PX
    )

    action_checked = 0
    action_agreed = 0
    action_mismatches: list[str] = []
    for step_row in steps:
        command = str(step_row.get("movement_command"))
        if command in {"KEEP", "NOOP"}:
            continue
        expected = _expected_dispatch(command, str(step_row.get("applied_movement")))
        action_checked += 1
        if step_row.get("dispatched_direction") == expected:
            action_agreed += 1
        else:
            action_mismatches.append(
                f"{step_row.get('episode_id')}#{step_row.get('step_id')}:{command}:"
                f"{step_row.get('dispatched_direction')}!={expected}"
            )

    per_episode: list[dict[str, object]] = []
    terminal_checked = 0
    terminal_agreed = 0
    terminal_mismatches: list[str] = []
    for episode_id in episode_ids:
        rows = episodes[episode_id]
        last = lookup.get((episode_id, int(rows[-1]["step_id"])))
        independent_last = None if last is None else last["independent_position"]
        arrived = independent_last is not None and (
            abs(float(cast(list[float], independent_last)[0]) - targets[-1][0])
            + abs(float(cast(list[float], independent_last)[1]) - targets[-1][1])
            <= 2 * tolerance
        )
        stored = rows[-1]["terminal_reason"] == "NAVIGATION_GOAL_REACHED"
        terminal_checked += 1
        if arrived == stored:
            terminal_agreed += 1
        else:
            terminal_mismatches.append(episode_id)
        lost = sum(
            1
            for item in derivations
            if item["episode_id"] == episode_id and item["independent_position"] is None
        )
        per_episode.append(
            {
                "episode_id": episode_id,
                "steps": len(rows),
                "position_lost_steps": lost,
                "store_terminal_reason": rows[-1]["terminal_reason"],
                "independent_arrival": bool(arrived),
            }
        )

    coverage = len(agreement) / len(steps) if steps else 0.0
    reference_agreement = (
        sum(1 for value in agreement if value <= REFERENCE_TOLERANCE_PX) / len(agreement)
        if agreement
        else 0.0
    )
    exact_fraction = (
        sum(1 for value in agreement if value <= EXACT_AGREEMENT_EPSILON) / len(agreement)
        if agreement
        else 0.0
    )
    median = float(np.median(np.asarray(agreement, dtype=np.float64))) if agreement else None
    p95 = (
        float(np.quantile(np.asarray(agreement, dtype=np.float64), 0.95)) if agreement else None
    )
    motion_agreement = motion_agreed / motion_defined if motion_defined else 0.0
    action_agreement = action_agreed / action_checked if action_checked else 0.0
    terminal_agreement = terminal_agreed / terminal_checked if terminal_checked else 0.0
    checks = (
        ("minimum_position_coverage", coverage, "minimum_position_coverage"),
        ("minimum_reference_agreement", reference_agreement, "minimum_reference_agreement"),
        ("minimum_motion_agreement", motion_agreement, "minimum_motion_agreement"),
        ("minimum_action_agreement", action_agreement, "minimum_action_agreement"),
        ("minimum_terminal_agreement", terminal_agreement, "minimum_terminal_agreement"),
    )
    failures = [
        name
        for name, value, gate_name in checks
        if value < float(cast(float, gates[gate_name]))
    ]
    if median is not None and median > float(cast(float, gates["maximum_reference_l1_median_px"])):
        failures.append("maximum_reference_l1_median_px")
    if p95 is not None and p95 > float(cast(float, gates["maximum_reference_l1_p95_px"])):
        failures.append("maximum_reference_l1_p95_px")
    if exact_fraction > float(cast(float, gates["maximum_exact_agreement_fraction"])):
        failures.append("maximum_exact_agreement_fraction")

    report: dict[str, object] = {
        "schema_version": NAVIGATION_FEEDBACK_AUDIT_SCHEMA,
        "status": "PASSED" if not failures and integrity == "ok" else "FAILED",
        "r0_feedback_contract_sha256": feedback_sha,
        "navigation_contract_sha256": navigation_sha,
        "store_path": store_path.name,
        "store_integrity": integrity,
        "episodes": len(episode_ids),
        "denominators": {
            "episodes": len(episode_ids),
            "steps": len(steps),
            "steps_without_reference": without_reference,
            "position_lost_steps": sum(
                cast(int, item["position_lost_steps"]) for item in per_episode
            ),
        },
        "position": {
            "coverage": coverage,
            "reference_agreement": reference_agreement,
            "l1_median_px": median,
            "l1_p95_px": p95,
            "compared_steps": len(agreement),
            "tolerance_px": REFERENCE_TOLERANCE_PX,
            "exact_agreement_fraction": exact_fraction,
            "reference_is_independent": exact_fraction
            <= float(cast(float, gates["maximum_exact_agreement_fraction"])),
        },
        "motion": {
            "defined_steps": motion_defined,
            "agreement": motion_agreement,
            "tolerance_px": MOTION_TOLERANCE_PX,
        },
        "action": {
            "checked_steps": action_checked,
            "agreement": action_agreement,
            "mismatches": action_mismatches,
        },
        "terminal": {
            "checked_episodes": terminal_checked,
            "agreement": terminal_agreement,
            "mismatches": terminal_mismatches,
        },
        "per_episode": per_episode,
        "gate_failures": failures,
        "unverified_remainder": [
            "no independent map reference exists on this route, so position coverage and agreement "
            "are implementation-independent checks on the same minimap RGB, not semantic accuracy",
            "the second position derivation is treated as a duplicate when it agrees with the "
            "recorded signal exactly on most steps; an exact match is evidence of duplication, not "
            "of accuracy",
            "the death_or_ended abort condition reads an ROI that is not persisted in the derived "
            "views and therefore cannot be re-derived here; it is not checked and must not become "
            "an R1 reward component",
            "the terminal label's only reference is its own rule outcome",
            "no visual event is claimed for this route and the transition event vocabulary has no "
            "navigation type",
        ],
        "claim_boundary": {
            "semantic_position_accuracy_verified": False,
            "independent_map_reference_available": False,
            "reward_allowed": False,
            "training_allowed": not failures
            and integrity == "ok"
            and exact_fraction
            <= float(cast(float, gates["maximum_exact_agreement_fraction"])),
        },
    }
    if qa_frames:
        from hok_agent.movement_real_rgb import write_marker_contact_sheet

        qa_dir = output / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        write_marker_contact_sheet(
            qa_dir / "feedback-qa.png", np.stack(qa_frames), qa_markers, qa_labels
        )
        report["qa_sheet"] = "qa/feedback-qa.png"
    (output / "step-derivations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in derivations), encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
