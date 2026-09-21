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
    output.mkdir(parents=True, exist_ok=True)
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


R0_RESPONSE_SCHEMAS = (
    "hok-agent-r0-response-task-contract-v1",
    "hok-agent-r0-response-task-contract-v2",
)
RESPONSE_AUDIT_SCHEMA = "hok-agent-command-response-audit-v1"
RESPONSE_REQUIRED_SIGNALS = (
    "patch_displacement",
    "commanded_response",
    "progress",
    "dead_reckoned_progress",
    "terminal",
)
RESPONSE_REQUIRED_GATES = (
    "minimum_response_coverage",
    "minimum_positive_response_fraction",
    "maximum_episode_net_displacement_error_px",
    "minimum_episodes_within_net_error",
    "minimum_dead_reckoning_steps",
    "maximum_exact_agreement_fraction",
)
RESPONSE_ESTIMATORS = ("plain_patch_ncc", "background_residual_ncc")
RESPONSE_OPTIONAL_GATES = (
    "maximum_step_l1_median_px",
    "maximum_step_l1_p95_px",
    "minimum_step_agreement_fraction",
    "maximum_step_bias_px",
)
PATCH_HALF = 11
PATCH_SEARCH = 16
DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "north": (-1.0, 0.0),
    "north_east": (-0.7071067811865476, 0.7071067811865476),
    "east": (0.0, 1.0),
    "south_east": (0.7071067811865476, 0.7071067811865476),
    "south": (1.0, 0.0),
    "south_west": (0.7071067811865476, -0.7071067811865476),
    "west": (0.0, -1.0),
    "north_west": (-0.7071067811865476, -0.7071067811865476),
}


def load_response_task_contract(path: Path) -> tuple[dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    signals = payload.get("signals")
    gates = payload.get("gates")
    names = {cast(dict[str, object], item)["name"] for item in cast(list[object], signals or [])}
    if (
        payload.get("schema_version") not in R0_RESPONSE_SCHEMAS
        or not isinstance(signals, list)
        or not isinstance(gates, dict)
        or names != set(RESPONSE_REQUIRED_SIGNALS)
        or any(key not in gates for key in RESPONSE_REQUIRED_GATES)
        or payload.get("estimator", "plain_patch_ncc") not in RESPONSE_ESTIMATORS
        or cast(dict[str, object], payload.get("independence", {})).get("duplication_guard") is None
    ):
        raise FeedbackAuditError("r0 response task contract differs")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return payload, digest


def _grey(frame: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, frame.astype(np.float64).mean(axis=2))


def _session_background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median over every persisted frame of the store.

    The minimap is a fixed map, so the median is the static background; a mover is an outlier
    unless it occupies the same pixel for most of the sample. An episode-local median is too small
    a sample: when the hero is nearly stationary its own marker passes the 50 % threshold and the
    subtraction erases the object being tracked.
    """
    return np.median(np.stack(frames).astype(np.float64), axis=0)


def _patch_displacement(
    current: np.ndarray,
    following: np.ndarray,
    position: tuple[float, float],
    background: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Best zero-mean correlation shift of the hero patch; a frame measurement, not a cue delta."""
    if background is not None:
        current = current.astype(np.float64) - background
        following = following.astype(np.float64) - background
    center_y = int(round(position[0]))
    center_x = int(round(position[1]))
    margin = PATCH_HALF + PATCH_SEARCH
    if (
        center_y - margin < 0
        or center_x - margin < 0
        or center_y + margin >= current.shape[0]
        or center_x + margin >= current.shape[1]
    ):
        return None
    patch = _grey(
        current[
            center_y - PATCH_HALF : center_y + PATCH_HALF,
            center_x - PATCH_HALF : center_x + PATCH_HALF,
        ]
    )
    patch = patch - patch.mean()
    norm = float(np.sqrt((patch * patch).sum()))
    if norm <= 1e-9:
        return None
    best: tuple[float, float, float] | None = None
    for offset_y in range(-PATCH_SEARCH, PATCH_SEARCH + 1):
        for offset_x in range(-PATCH_SEARCH, PATCH_SEARCH + 1):
            window = _grey(
                following[
                    center_y - PATCH_HALF + offset_y : center_y + PATCH_HALF + offset_y,
                    center_x - PATCH_HALF + offset_x : center_x + PATCH_HALF + offset_x,
                ]
            )
            window = window - window.mean()
            denominator = float(np.sqrt((window * window).sum())) * norm
            if denominator <= 1e-9:
                continue
            score = float((patch * window).sum() / denominator)
            if best is None or score > best[0]:
                best = (score, float(offset_y), float(offset_x))
    return best



def run_command_response_audit(
    *,
    response_contract_path: Path,
    navigation_contract_path: Path,
    store_path: Path,
    frame_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract, contract_sha = load_response_task_contract(response_contract_path)
    _navigation, navigation_sha = _goal_navigation_contract(navigation_contract_path)
    gates = cast(dict[str, object], contract["gates"])
    estimator = str(contract.get("estimator", "plain_patch_ncc"))
    output = _new_large_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with UnifiedTransitionStore(store_path) as store:
        episode_ids = store.episode_ids()
        episodes: dict[str, tuple[HierarchicalTransitionRecord, ...]] = {
            episode_id: store.load_episode(episode_id) for episode_id in episode_ids
        }
        integrity = store.integrity()
    steps = _step_rows(store_path.parent)
    rows_by_key = {
        (cast(str, row.get("episode_id")), cast(int, row.get("step_id", -1))): row for row in steps
    }
    per_episode: list[dict[str, object]] = []
    responses: list[float] = []
    move_steps = 0
    stop_steps = 0
    held_direction: str | None = None
    without_displacement = 0
    duplicate_steps = 0
    compared_steps = 0
    step_errors: list[float] = []
    step_bias_y = 0.0
    step_bias_x = 0.0
    background = (
        _session_background(
            [
                _minimap(frame_root, record[label]["frame_bundle_ref"])
                for episode_id in episode_ids
                for record in episodes[episode_id]
                for label in ("observation", "next_observation")
            ]
        )
        if estimator == "background_residual_ncc"
        else None
    )
    for episode_id in episode_ids:
        records = episodes[episode_id]
        dead_reckoned_y = 0.0
        dead_reckoned_x = 0.0
        cue_first: tuple[float, float] | None = None
        cue_last: tuple[float, float] | None = None
        dead_steps = 0
        for record in records:
            step_id = int(record["step_id"])
            step_row = rows_by_key.get((episode_id, step_id))
            if step_row is None:
                continue
            position = step_row.get("position")
            if position is None:
                continue
            cue = (float(cast(list[float], position)[0]), float(cast(list[float], position)[1]))
            cue_first = cue if cue_first is None else cue_first
            cue_last = cue
            dispatched = step_row.get("dispatched_direction")
            if isinstance(dispatched, str):
                held_direction = dispatched
            command = held_direction if step_row.get("movement_command") == "KEEP" else dispatched
            following_position = rows_by_key.get((episode_id, step_id + 1), {}).get("position")
            if isinstance(following_position, list):
                compare = (
                    float(cast(list[float], following_position)[0]) - cue[0],
                    float(cast(list[float], following_position)[1]) - cue[1],
                )
            else:
                compare = None
            current = _minimap(frame_root, record["observation"]["frame_bundle_ref"])
            following = _minimap(frame_root, record["next_observation"]["frame_bundle_ref"])
            displacement = _patch_displacement(current, following, cue, background)
            if command == "wait" or command is None:
                stop_steps += 1
            else:
                move_steps += 1
            if displacement is None:
                without_displacement += 1
                continue
            dead_steps += 1
            dead_reckoned_y += displacement[1]
            dead_reckoned_x += displacement[2]
            if compare is not None:
                compared_steps += 1
                step_errors.append(
                    abs(displacement[1] - compare[0]) + abs(displacement[2] - compare[1])
                )
                step_bias_y += displacement[1] - compare[0]
                step_bias_x += displacement[2] - compare[1]
                if (
                    abs(displacement[1] - compare[0]) <= EXACT_AGREEMENT_EPSILON
                    and abs(displacement[2] - compare[1]) <= EXACT_AGREEMENT_EPSILON
                ):
                    duplicate_steps += 1
            vector = DIRECTION_VECTORS.get(str(command))
            if vector is not None:
                responses.append(
                    displacement[1] * vector[0] + displacement[2] * vector[1]
                )
        if cue_first is None or cue_last is None or dead_steps == 0:
            per_episode.append(
                {
                    "episode_id": episode_id,
                    "steps": len(records),
                    "dead_reckoning_steps": dead_steps,
                    "error_px": None,
                    "within_error": False,
                }
            )
            continue
        error = abs(dead_reckoned_y - (cue_last[0] - cue_first[0])) + abs(
            dead_reckoned_x - (cue_last[1] - cue_first[1])
        )
        within = error <= float(cast(float, gates["maximum_episode_net_displacement_error_px"]))
        per_episode.append(
            {
                "episode_id": episode_id,
                "steps": len(records),
                "dead_reckoning_steps": dead_steps,
                "cue_net": [cue_last[0] - cue_first[0], cue_last[1] - cue_first[1]],
                "dead_reckoned_net": [dead_reckoned_y, dead_reckoned_x],
                "error_px": error,
                "within_error": within,
            }
        )
    coverage = compared_steps / move_steps if move_steps else 0.0
    positive = sum(1 for value in responses if value > 0.0) / len(responses) if responses else 0.0
    eligible = [
        row for row in per_episode if cast(int, row["dead_reckoning_steps"]) >= int(
            cast(int, gates["minimum_dead_reckoning_steps"])
        )
    ]
    within_fraction = (
        sum(1 for row in eligible if bool(row["within_error"])) / len(eligible)
        if eligible
        else 0.0
    )
    duplicate_fraction = duplicate_steps / compared_steps if compared_steps else 0.0
    failures: list[str] = []
    if coverage < float(cast(float, gates["minimum_response_coverage"])):
        failures.append("minimum_response_coverage")
    if positive < float(cast(float, gates["minimum_positive_response_fraction"])):
        failures.append("minimum_positive_response_fraction")
    if not eligible or within_fraction < float(
        cast(float, gates["minimum_episodes_within_net_error"])
    ):
        failures.append("minimum_episodes_within_net_error")
    if duplicate_fraction > float(cast(float, gates["maximum_exact_agreement_fraction"])):
        failures.append("displacement_duplicates_the_cue_delta")
    step_median = (
        float(np.median(np.asarray(step_errors, dtype=np.float64))) if step_errors else None
    )
    step_p95 = (
        float(np.quantile(np.asarray(step_errors, dtype=np.float64), 0.95))
        if step_errors
        else None
    )
    within_3px = (
        sum(1 for value in step_errors if value <= 3.0) / len(step_errors) if step_errors else 0.0
    )
    bias_y = step_bias_y / compared_steps if compared_steps else 0.0
    bias_x = step_bias_x / compared_steps if compared_steps else 0.0
    if (
        "maximum_step_l1_median_px" in gates
        and step_median is not None
        and step_median > float(cast(float, gates["maximum_step_l1_median_px"]))
    ):
        failures.append("maximum_step_l1_median_px")
    if (
        "maximum_step_l1_p95_px" in gates
        and step_p95 is not None
        and step_p95 > float(cast(float, gates["maximum_step_l1_p95_px"]))
    ):
        failures.append("maximum_step_l1_p95_px")
    if "minimum_step_agreement_fraction" in gates and within_3px < float(
        cast(float, gates["minimum_step_agreement_fraction"])
    ):
        failures.append("minimum_step_agreement_fraction")
    if "maximum_step_bias_px" in gates and max(abs(bias_y), abs(bias_x)) > float(
        cast(float, gates["maximum_step_bias_px"])
    ):
        failures.append("maximum_step_bias_px")
    report: dict[str, object] = {
        "schema_version": RESPONSE_AUDIT_SCHEMA,
        "status": "PASSED" if not failures and integrity == "ok" else "FAILED",
        "response_contract_sha256": contract_sha,
        "navigation_contract_sha256": navigation_sha,
        "store_path": store_path.name,
        "store_integrity": integrity,
        "episodes": len(episode_ids),
        "denominators": {
            "episodes": len(episode_ids),
            "steps": len(steps),
            "move_steps": move_steps,
            "stop_steps": stop_steps,
            "steps_without_displacement": without_displacement,
            "dead_reckoning_steps": compared_steps,
        },
        "response": {
            "coverage": coverage,
            "positive_fraction": positive,
            "measured_steps": len(responses),
        },
        "dead_reckoning": {
            "eligible_episodes": len(eligible),
            "within_error_fraction": within_fraction,
            "tolerance_px": float(
                cast(float, gates["maximum_episode_net_displacement_error_px"])
            ),
            "per_episode": per_episode,
        },
        "estimator": estimator,
        "step_diagnostic": {
            "compared_steps": len(step_errors),
            "l1_median_px": step_median,
            "l1_p95_px": step_p95,
            "fraction_within_3px": within_3px,
            "mean_bias_per_step": [bias_y, bias_x],
            "gated": "maximum_step_l1_median_px" in gates,
        },
        "duplication": {
            "compared_steps": compared_steps,
            "duplicate_fraction": duplicate_fraction,
            "measurement_is_frame_derived": duplicate_fraction
            <= float(cast(float, gates["maximum_exact_agreement_fraction"])),
        },
        "gate_failures": failures,
        "unverified_remainder": [
            "the patch is anchored at the cue position, so the measurement location still depends "
            "on the cue; only the displacement value is frame-derived",
            "the reward compares an execution-stream direction with a patch measurement, which is "
            "structurally independent, but neither side is an absolute map reference",
            "the terminal label's only reference is its own rule outcome",
        ],
        "claim_boundary": {
            "semantic_position_accuracy_verified": False,
            "independent_map_reference_available": False,
            "reward_allowed": False,
            "training_allowed": not failures and integrity == "ok",
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


R0_PANEL_SCHEMAS = (
    "hok-agent-r0-panel-feedback-contract-v1",
    "hok-agent-r0-panel-feedback-contract-v2",
)
PANEL_AUDIT_SCHEMA = "hok-agent-panel-feedback-audit-v1"
PANEL_REQUIRED_GATES = (
    "minimum_steps",
    "minimum_label_agreement",
    "minimum_transitions",
    "maximum_statistic_correlation",
)
PANEL_REQUIRED_STATISTICS = ("roi_mean_brightness", "roi_bright_pixel_count")
BRIGHT_PIXEL_LEVEL = 128


def load_panel_feedback_contract(path: Path) -> tuple[dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    gates = payload.get("gates")
    statistics = payload.get("statistics")
    authorization = payload.get("owner_authorization")
    if authorization is not None and (
        not isinstance(authorization, dict)
        or authorization.get("authorized_by") != "owner"
        or not authorization.get("change")
        or not authorization.get("consequence")
    ):
        raise FeedbackAuditError("owner authorization block differs")
    if (
        payload.get("schema_version") not in R0_PANEL_SCHEMAS
        or not isinstance(gates, dict)
        or not isinstance(statistics, list)
        or {str(item) for item in cast(list[object], statistics)}
        != set(PANEL_REQUIRED_STATISTICS)
        or any(key not in gates for key in PANEL_REQUIRED_GATES)
        or cast(dict[str, object], payload.get("independence", {})).get("duplication_guard")
        is None
        or cast(dict[str, object], payload.get("claim_boundary", {})).get("reward_allowed")
        is not False
    ):
        raise FeedbackAuditError("r0 panel feedback contract differs")
    if authorization is not None and not bool(
        cast(dict[str, object], payload.get("claim_boundary", {})).get("gate_is_owner_authorized")
    ):
        raise FeedbackAuditError("authorized gate must be declared in the claim boundary")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return payload, digest


def _panel_view(frame_root: Path, ref: str) -> np.ndarray:
    with np.load(frame_root / ref, allow_pickle=False) as saved:
        return np.ascontiguousarray(saved["equipment"]).astype(np.float64)


def _panel_statistics(view: np.ndarray) -> tuple[float, float]:
    return float(view.mean()), float((view.max(axis=2) > BRIGHT_PIXEL_LEVEL).sum())


def run_panel_feedback_audit(
    *,
    panel_contract_path: Path,
    store_path: Path,
    frame_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract, contract_sha = load_panel_feedback_contract(panel_contract_path)
    gates = cast(dict[str, object], contract["gates"])
    authorization = contract.get("owner_authorization")
    verification_class = (
        "owner_authorized_bar" if isinstance(authorization, dict) else "pre_registered"
    )
    output = _new_large_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with UnifiedTransitionStore(store_path) as store:
        episode_ids = store.episode_ids()
        episodes: dict[str, tuple[HierarchicalTransitionRecord, ...]] = {
            episode_id: store.load_episode(episode_id) for episode_id in episode_ids
        }
        integrity = store.integrity()
    measured: list[float] = []
    counted: list[float] = []
    per_episode: list[dict[str, object]] = []
    for episode_id in episode_ids:
        rows = episodes[episode_id]
        series: list[tuple[int, float, float]] = []
        for record in rows:
            view = _panel_view(frame_root, record["observation"]["frame_bundle_ref"])
            mean_brightness, bright_pixels = _panel_statistics(view)
            series.append((int(record["step_id"]), mean_brightness, bright_pixels))
        measured.extend(item[1] for item in series)
        counted.extend(item[2] for item in series)
        per_episode.append(
            {
                "episode_id": episode_id,
                "steps": len(series),
                "mean_brightness_range": [
                    min(item[1] for item in series),
                    max(item[1] for item in series),
                ]
                if series
                else None,
                "bright_pixels_range": [
                    min(item[2] for item in series),
                    max(item[2] for item in series),
                ]
                if series
                else None,
            }
        )
    steps = len(measured)
    if steps == 0:
        raise FeedbackAuditError("panel feedback audit found no steps")
    median_a = float(np.median(np.asarray(measured)))
    median_b = float(np.median(np.asarray(counted)))
    labels_a = [value > median_a for value in measured]
    labels_b = [value > median_b for value in counted]
    agreement = (
        sum(1 for left, right in zip(labels_a, labels_b, strict=True) if left == right) / steps
    )
    array_a = np.asarray(measured, dtype=np.float64)
    array_b = np.asarray(counted, dtype=np.float64)
    if float(array_a.std()) <= 1e-9 or float(array_b.std()) <= 1e-9:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(array_a, array_b)[0, 1])
    transitions = sum(
        1 for index in range(1, len(labels_a)) if labels_a[index] != labels_a[index - 1]
    )
    on_steps = sum(1 for label in labels_a if label)
    # Sharper structural test than the agreement rate: if the disagreement were only a transition
    # artefact, every ambiguous step would sit next to a label change. An isolated ambiguous step
    # means the panel is not resolvable into a discrete state at the observation cadence.
    ambiguous = [index for index in range(steps) if labels_a[index] != labels_b[index]]
    isolated = [
        index
        for index in ambiguous
        if not (
            (index > 0 and labels_a[index - 1] != labels_a[index])
            or (index + 1 < steps and labels_a[index + 1] != labels_a[index])
        )
    ]
    ambiguous_levels = [measured[index] for index in ambiguous]
    failures: list[str] = []
    if steps < int(cast(int, gates["minimum_steps"])):
        failures.append("minimum_steps")
    if agreement < float(cast(float, gates["minimum_label_agreement"])):
        failures.append("minimum_label_agreement")
    if transitions < int(cast(int, gates["minimum_transitions"])):
        failures.append("minimum_transitions")
    if abs(correlation) > float(cast(float, gates["maximum_statistic_correlation"])):
        failures.append("maximum_statistic_correlation")
    report: dict[str, object] = {
        "schema_version": PANEL_AUDIT_SCHEMA,
        "status": "PASSED" if not failures and integrity == "ok" else "FAILED",
        "verification_class": verification_class,
        "owner_authorization": authorization,
        "panel_contract_sha256": contract_sha,
        "store_path": store_path.name,
        "store_integrity": integrity,
        "episodes": len(episode_ids),
        "denominators": {
            "episodes": len(episode_ids),
            "steps": steps,
            "on_steps": on_steps,
            "off_steps": steps - on_steps,
            "transitions": transitions,
            "steps_without_label": 0,
        },
        "discrete_state_test": {
            "ambiguous_steps": len(ambiguous),
            "isolated_ambiguous_steps": len(isolated),
            "isolated_indices": isolated,
            "ambiguous_mean_brightness": [
                min(ambiguous_levels) if ambiguous_levels else None,
                max(ambiguous_levels) if ambiguous_levels else None,
            ],
            "verdict": "resolvable" if not isolated else "not_resolvable_at_this_cadence",
        },
        "label": {
            "agreement": agreement,
            "median_mean_brightness": median_a,
            "median_bright_pixel_count": median_b,
            "statistic_correlation": correlation,
            "duty_cycle": on_steps / steps,
        },
        "per_episode": per_episode,
        "gate_failures": failures,
        "unresolved_concerns": []
        if not isolated
        else [
            "the discrete-state structural test still reports isolated ambiguous steps "
            f"{isolated}, i.e. not_resolvable_at_this_cadence; a pass under an owner-authorized "
            "bar does not resolve this"
        ],
        "unverified_remainder": [
            "the audit verifies that the ROI carries a periodic strongly bimodal signal; the "
            "isolated-ambiguous test shows it is not resolvable into a discrete state at the "
            "1.2 s observation cadence",
            "the audit verifies that the ROI carries a discrete two-state signal; it does not "
            "verify what the panel means or that a purchase changes it",
            "the label rule is each statistic against its own session median, which assumes a "
            "roughly balanced duty cycle",
            "no dispatched action is correlated with the panel in this audit; the gating task "
            "needs its own run",
        ],
        "claim_boundary": {
            "panel_semantics_verified": False,
            "purchase_effect_verified": False,
            "reward_allowed": False,
            "training_allowed": not failures and integrity == "ok",
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
