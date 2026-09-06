from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np

REPORT_SCHEMA = "movement-real-rgb-observability-report-v1"

# One bounded appearance candidate, independent of the frozen v2 detector.
PLAYER_TRACKING_SETTINGS = {
    "patch_radius": 7,
    "center_search_radius": 3,
    "minimum_correlation": 0.70,
    "minimum_distinct_peak_margin": 0.05,
    "maximum_step_pixels": 8.0,
    "confirmation_frames": 2,
    "template_frames": 16,
}

NATIVE_PLAYER_SOURCES = {
    "0667d97cdb3024f7c5d39d6e797bfe1d527cbd99c07e8e2d2e31cd6eb0ed0993": "train",
    "c1121610049b451fb1e3f8d3c3695da15ff04f5391898d7b9a0de867b685c009": "dev",
}


def green_ring_candidates(frame: np.ndarray) -> list[tuple[int, int]]:
    """Uncalibrated visual cue on a 256px native-cropped map, not player identity."""
    if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
        raise ValueError("ring cue requires a 256x256 uint8 RGB map")
    rgb = frame.astype(np.int16)
    red, green, blue = (rgb[..., i] for i in range(3))
    mask = (green >= 150) & (green - red >= 30) & (green - blue >= 20)
    padded = np.pad(mask, 1)
    thick = np.zeros_like(mask)
    core = np.zeros(mask.shape, dtype=np.int16)
    for dy in range(3):
        for dx in range(3):
            shifted = padded[dy : dy + 256, dx : dx + 256]
            thick |= shifted
            core += shifted
    scores = np.zeros(mask.shape, dtype=np.int16)
    for radius in range(8, 15):
        hits = np.zeros(mask.shape, dtype=np.int16)
        border = np.pad(thick, 15)
        for angle in np.arange(8) * math.pi / 4:
            dy, dx = round(radius * math.sin(angle)), round(radius * math.cos(angle))
            hits += border[15 + dy : 271 + dy, 15 + dx : 271 + dx]
        scores = np.maximum(scores, hits)
    scores[core >= 4] = 0  # green filled blobs are not hollow portrait borders
    scores[:15] = scores[-15:] = 0
    scores[:, :15] = scores[:, -15:] = 0
    scores[:40, 224:] = 0  # team-strip UI outside the map, not a candidate player
    ys, xs = np.where(scores >= 7)
    ordered = sorted(zip(ys.tolist(), xs.tolist(), strict=True), key=lambda p: (-int(scores[p]), p))
    peaks: list[tuple[int, int]] = []
    for point in ordered:
        if all(math.dist(point, other) > 20 for other in peaks):
            peaks.append(point)
    return peaks


def green_ring_track(frames: np.ndarray) -> list[tuple[int, int] | None]:
    """Only a unique observed ring can be confirmed; no extrapolation or identity claim."""
    return _confirm_ring_candidates([green_ring_candidates(frame) for frame in frames])[0]


def _confirm_ring_candidates(
    rows: list[list[tuple[int, int]]],
) -> tuple[list[tuple[int, int] | None], list[str]]:
    previous: tuple[int, int] | None = None
    positions: list[tuple[int, int] | None] = []
    reasons: list[str] = []
    for row in rows:
        candidates = row
        if previous is not None:
            nearby = [point for point in candidates if math.dist(previous, point) <= 12]
            if nearby:
                candidates = nearby
        selected = candidates[0] if len(candidates) == 1 else None
        reason = (
            "no_ring_evidence"
            if not candidates
            else "ambiguous_candidates"
            if selected is None
            else "awaiting_confirmation"
            if previous is None
            else "discontinuous_candidate"
            if math.dist(previous, selected) > 12
            else "confirmed_visual_cue"
        )
        positions.append(selected if reason == "confirmed_visual_cue" else None)
        reasons.append(reason)
        previous = selected
    return positions, reasons


def _ring_diagnostic_region(point: tuple[int, int]) -> str:
    """Provisional QA envelopes, not semantic map boundaries or runtime masks."""
    y, x = point
    if 16 <= x < 224 and 0 <= y < 205:
        return "interior"
    if 4 <= x < 236 and 0 <= y < 225:
        return "edge_margin"
    return "context"


def audit_native_player_background(source_run: Path, output_dir: Path) -> dict[str, object]:
    """Compare spatial pruning using only the two cached windows; never modify RGB."""
    if output_dir.exists():
        raise ValueError("background audit output already exists")
    source = _load_bound_json(source_run / "report.json", "report_sha256")
    if source.get("status") != "NATIVE_LANDSCAPE_WINDOWS_MATERIALIZED_QA_ONLY":
        raise ValueError("background audit requires native pilot evidence")
    rows = cast(list[dict[str, object]], source["sessions"])
    if {str(row["session_hash"]): row["split"] for row in rows} != NATIVE_PLAYER_SOURCES:
        raise ValueError("background audit source sessions differ")
    sessions: list[dict[str, object]] = []
    for row in rows:
        identity = str(row["session_hash"])
        name = identity[:8] + "-native-window.npz"
        artifact = next(
            item
            for item in cast(list[dict[str, object]], row["artifacts"])
            if item["basename"] == name
        )
        path = source_run / name
        if path.is_symlink() or _file_sha256(path) != artifact["sha256"]:
            raise ValueError("background audit cached window hash differs")
        with np.load(path, allow_pickle=False) as arrays:
            frames = arrays["minimap_rgb"]
            candidates = [green_ring_candidates(frame) for frame in frames]
        variants: dict[str, object] = {}
        for variant in ("unfiltered", "interior_plus_edge", "interior_only_diagnostic"):
            kept = [
                [
                    point
                    for point in points
                    if variant == "unfiltered"
                    or _ring_diagnostic_region(point) == "interior"
                    or (
                        variant == "interior_plus_edge"
                        and _ring_diagnostic_region(point) == "edge_margin"
                    )
                ]
                for points in candidates
            ]
            positions, reasons = _confirm_ring_candidates(kept)
            if (
                variant == "unfiltered"
                and [list(p) if p is not None else None for p in positions]
                != row["confirmed_green_ring_yx"]
            ):
                raise ValueError("background audit no longer reproduces the frozen tracker")
            variants[variant] = {
                "candidate_count": sum(map(len, kept)),
                "confirmed_frames": sum(p is not None for p in positions),
                "positions_yx": positions,
                "reason_counts": dict(Counter(reasons)),
                "reason_by_frame": reasons,
                "frames_emptied_by_envelope": [
                    i for i, points in enumerate(kept) if not points and candidates[i]
                ],
            }
        sessions.append(
            {
                "session_hash": identity,
                "split": row["split"],
                "frames": len(candidates),
                "cached_window_sha256": artifact["sha256"],
                "candidate_regions": dict(
                    Counter(_ring_diagnostic_region(p) for points in candidates for p in points)
                ),
                "candidates_yx": candidates,
                "variants": variants,
            }
        )
    report: dict[str, object] = {
        "status": "NATIVE_BACKGROUND_ABLATION_DIAGNOSTIC_ONLY",
        "source_report_sha256": source["report_sha256"],
        "implementation_sha256": _file_sha256(Path(__file__)),
        "sessions": sessions,
        "diagnostic_interior_xyxy": [16, 0, 224, 205],
        "diagnostic_expanded_xyxy": [4, 0, 236, 225],
        "envelope_note": (
            "provisional rectangles from existing QA; not exact map segmentation, "
            "identity labels, or runtime masks"
        ),
        "candidate_thresholds_changed": False,
        "rgb_modified": False,
        "video_frames_decoded": 0,
        "training_allowed": False,
        "runtime_filter_promoted": False,
        "controlled_player_identity_verified": False,
        "input_commands_sent": 0,
        "model_runs": 0,
        "gpu_seconds": 0,
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_bytes(_canonical(report) + b"\n")
    return report


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound_json(path: Path, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required JSON is not a regular file: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path.name}")
    result = cast(dict[str, object], payload)
    supplied = str(result.pop(field, ""))
    calculated = _object_sha256(result)
    result[field] = supplied
    if supplied != calculated:
        raise ValueError(f"{path.name} self hash differs")
    return result


def _content_bounds(frames: np.ndarray, config: dict[str, object]) -> tuple[int, int, int, int]:
    pixel_minimum = float(cast(float, config["pixel_mean_minimum"]))
    support = float(cast(float, config["row_column_support_minimum"]))
    bounds: list[tuple[int, int, int, int]] = []
    for frame in frames:
        mask = frame.astype(np.float32).mean(axis=2) > pixel_minimum
        rows = np.flatnonzero(mask.mean(axis=1) >= support)
        columns = np.flatnonzero(mask.mean(axis=0) >= support)
        if len(rows) and len(columns):
            bounds.append((int(columns[0]), int(rows[0]), int(columns[-1] + 1), int(rows[-1] + 1)))
    if not bounds:
        raise ValueError("selected session has no visible content")
    return cast(
        tuple[int, int, int, int],
        tuple(int(round(float(np.median([row[index] for row in bounds])))) for index in range(4)),
    )


def _canonical_content(
    frames: np.ndarray, config: dict[str, object]
) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
    bounds = _content_bounds(frames, config)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    orientation = "stored"
    canonical = frames
    aspect = float(cast(float, config["landscape_aspect_minimum"]))
    if height > width * aspect:
        canonical = np.rot90(frames, 1, axes=(1, 2)).copy()
        orientation = "counter_clockwise_90"
        bounds = _content_bounds(canonical, config)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if (
        width < height * aspect
        or width < int(cast(int, config["minimum_width"]))
        or height < int(cast(int, config["minimum_height"]))
    ):
        raise ValueError("selected session content orientation is ambiguous")
    return canonical, orientation, bounds


def _normalize_content(frame: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    rows = np.linspace(y0, y1 - 1, 128).astype(np.int64)
    columns = np.linspace(x0, x1 - 1, 128).astype(np.int64)
    return np.ascontiguousarray(frame[rows[:, None], columns[None, :], :])


def _detect(frame: np.ndarray, contract: dict[str, object]) -> dict[str, object]:
    roi_raw = cast(list[int], contract["minimap_roi_xyxy"])
    x0, y0, x1, y1 = map(int, roi_raw)
    roi = frame[y0:y1, x0:x1].astype(np.int16)
    red, green, blue = (roi[..., index] for index in range(3))
    player = cast(dict[str, object], contract["player_color"])
    target = cast(dict[str, object], contract["target_color"])
    player_mask = (
        (green > int(cast(int, player["green_minimum"])))
        & (green - red > int(cast(int, player["green_red_margin"])))
        & (green - blue > int(cast(int, player["green_blue_margin"])))
    )
    target_mask = (
        (red > int(cast(int, target["red_minimum"])))
        & (red - green > int(cast(int, target["red_green_margin"])))
        & (red - blue > int(cast(int, target["red_blue_margin"])))
    )
    player_y, player_x = np.where(player_mask)
    target_y, target_x = np.where(target_mask)
    player_visible = (
        int(cast(int, player["support_minimum"]))
        <= len(player_y)
        <= int(cast(int, player["support_maximum"]))
    )
    target_visible = bool(len(target_y))
    player_yx: tuple[float, float] | None = None
    target_yx: tuple[int, int] | None = None
    pair_visible = False
    if player_visible:
        player_yx = (round(float(player_y.mean()), 4), round(float(player_x.mean()), 4))
    if player_yx is not None and target_visible:
        candidates = np.stack((target_y, target_x), axis=1)
        center = np.asarray(player_yx)
        squared = np.square(candidates - center).sum(axis=1)
        selected = candidates[int(np.argmin(squared))]
        distance = float(np.sqrt(float(np.min(squared))))
        if distance >= float(cast(float, target["minimum_distance_pixels"])):
            target_yx = (int(selected[0]), int(selected[1]))
            pair_visible = True
    return {
        "player_visible": player_visible,
        "target_visible": target_visible,
        "pair_visible": pair_visible,
        "player_yx": list(player_yx) if player_yx is not None else None,
        "target_yx": list(target_yx) if target_yx is not None else None,
    }


def _selected_frames(
    target_root: Path,
    shard_rows: list[dict[str, object]],
    session_hash: str,
    split: str,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], set[int]]:
    frames: dict[int, np.ndarray] = {}
    timestamps: dict[int, int] = {}
    opened: list[dict[str, object]] = []
    rotations: set[int] = set()
    offset = 0
    selected_set = set(indices)
    for row in shard_rows:
        count = int(cast(int, row["row_count"]))
        local = sorted(index - offset for index in selected_set if offset <= index < offset + count)
        if not local:
            offset += count
            continue
        if row.get("split") != split or row.get("source") != "target":
            raise ValueError("selected target shard role differs")
        path = target_root / "shards" / str(row["path"])
        if path.is_symlink() or not path.is_file() or _file_sha256(path) != row["sha256"]:
            raise ValueError("selected target shard binding differs")
        with np.load(path, allow_pickle=False) as shard:
            required = {
                "frames",
                "session_hash",
                "timestamp_ms",
                "rotation_degrees",
                "frame_hash",
                "split",
            }
            if not required.issubset(shard.files) or len(shard["frames"]) != count:
                raise ValueError("selected target shard fields differ")
            for local_index in local:
                frame = shard["frames"][local_index]
                if (
                    frame.shape != (128, 128, 3)
                    or frame.dtype != np.uint8
                    or str(shard["session_hash"][local_index]) != session_hash
                    or str(shard["split"][local_index]) != split
                    or hashlib.sha256(frame.tobytes()).hexdigest()
                    != str(shard["frame_hash"][local_index])
                ):
                    raise ValueError("selected target frame binding differs")
                frames[offset + local_index] = frame.copy()
                timestamps[offset + local_index] = int(shard["timestamp_ms"][local_index])
                rotations.add(int(shard["rotation_degrees"][local_index]))
            opened.append(
                {
                    "basename": path.name,
                    "sha256": row["sha256"],
                    "declared_split": row["split"],
                }
            )
        offset += count
    if set(frames) != selected_set:
        raise ValueError("selected frame indices were not fully resolved")
    return (
        np.stack([frames[index] for index in indices]),
        np.asarray([timestamps[index] for index in indices], dtype=np.int64),
        opened,
        rotations,
    )


def run_real_rgb_preflight(
    contract_path: Path, target_root: Path, output_dir: Path
) -> dict[str, object]:
    contract = _load_bound_json(contract_path, "contract_sha256")
    if target_root.is_symlink() or not target_root.is_dir():
        raise ValueError("target root is not a regular directory")
    manifest = _load_bound_json(target_root / "manifest.json", "manifest_sha256")
    if (
        contract.get("schema_version") != "movement-real-rgb-observability-contract-v1"
        or contract.get("target_manifest_sha256") != manifest.get("manifest_sha256")
        or contract.get("test_allowed") is not False
        or contract.get("training_allowed") is not False
        or contract.get("device_input_allowed") is not False
    ):
        raise ValueError("real RGB observability contract binding differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("real RGB preflight output already exists")
    session_declarations = cast(list[dict[str, object]], contract["sessions"])
    manifest_sessions = {
        str(row["session_hash"]): str(row["split"])
        for row in cast(list[dict[str, object]], manifest["sessions"])
    }
    manifest_shards = cast(list[dict[str, object]], manifest["shards"])
    by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest_shards:
        for identity in cast(list[str], row["session_hashes"]):
            by_session[str(identity)].append(row)
    allowed_splits = set(cast(list[str], contract["allowed_splits"]))
    segment_fractions = cast(list[float], contract["segment_start_fractions"])
    segment_frames = int(cast(int, contract["frames_per_segment"]))
    all_frame_results: list[dict[str, object]] = []
    session_results: list[dict[str, object]] = []
    opened_shards: dict[str, dict[str, object]] = {}
    observed_splits: set[str] = set()
    observed_rotations: set[int] = set()
    jump_count = jump_denominator = 0
    for declaration in session_declarations:
        identity = str(declaration["session_hash"])
        split = str(declaration["split"])
        if split not in allowed_splits or manifest_sessions.get(identity) != split:
            raise ValueError("selected session split differs")
        rows = by_session[identity]
        total = sum(int(cast(int, row["row_count"])) for row in rows)
        starts = [round((total - segment_frames) * float(value)) for value in segment_fractions]
        indices = [index for start in starts for index in range(start, start + segment_frames)]
        frames, timestamps, opened, rotations = _selected_frames(
            target_root, rows, identity, split, indices
        )
        if not rotations <= {0, 90, 180, 270}:
            raise ValueError("selected target rotation differs")
        observed_rotations.update(rotations)
        for row in opened:
            opened_shards[str(row["basename"])] = row
            observed_splits.add(str(row["declared_split"]))
        content = cast(dict[str, object], contract["content_box"])
        canonical, orientation, bounds = _canonical_content(frames, content)
        normalized = np.stack([_normalize_content(frame, bounds) for frame in canonical])
        detections = [_detect(frame, contract) for frame in normalized]
        for segment_index, _start in enumerate(starts):
            segment_detections = detections[
                segment_index * segment_frames : (segment_index + 1) * segment_frames
            ]
            segment_times = timestamps[
                segment_index * segment_frames : (segment_index + 1) * segment_frames
            ]
            if not np.all(np.diff(segment_times) == int(cast(int, contract["frame_period_ms"]))):
                raise ValueError("selected segment sampling period differs")
            previous: dict[str, object] | None = None
            for timestamp, detection in zip(
                segment_times.tolist(), segment_detections, strict=True
            ):
                jumped = False
                if previous is not None and previous["pair_visible"] and detection["pair_visible"]:
                    jump_denominator += 1
                    temporal = cast(dict[str, object], contract["temporal"])
                    current_player = np.asarray(detection["player_yx"], dtype=np.float32)
                    previous_player = np.asarray(previous["player_yx"], dtype=np.float32)
                    current_target = np.asarray(detection["target_yx"], dtype=np.float32)
                    previous_target = np.asarray(previous["target_yx"], dtype=np.float32)
                    jumped = bool(
                        np.linalg.norm(current_player - previous_player)
                        > float(cast(float, temporal["player_jump_pixels"]))
                        or np.linalg.norm(current_target - previous_target)
                        > float(cast(float, temporal["target_jump_pixels"]))
                    )
                    jump_count += int(jumped)
                frame_result = {
                    "session_hash": identity,
                    "split": split,
                    "segment_index": segment_index,
                    "timestamp_ms": timestamp,
                    **detection,
                    "marker_jump": jumped,
                }
                all_frame_results.append(frame_result)
                previous = detection
        session_frames = [row for row in all_frame_results if row["session_hash"] == identity]
        pair_count = sum(bool(row["pair_visible"]) for row in session_frames)
        session_results.append(
            {
                "session_hash": identity,
                "split": split,
                "declared_content_geometry": declaration["content_geometry"],
                "source_rows": total,
                "segments": len(starts),
                "sampled_frames": len(session_frames),
                "stored_rotation_degrees": sorted(rotations),
                "detected_orientation": orientation,
                "content_box_xyxy": list(bounds),
                "player_visible_fraction": sum(
                    bool(row["player_visible"]) for row in session_frames
                )
                / len(session_frames),
                "target_visible_fraction": sum(
                    bool(row["target_visible"]) for row in session_frames
                )
                / len(session_frames),
                "pair_coverage": pair_count / len(session_frames),
                "unknown_fraction": 1.0 - pair_count / len(session_frames),
            }
        )
    pair_count = sum(bool(row["pair_visible"]) for row in all_frame_results)
    overall_coverage = pair_count / len(all_frame_results)
    marker_jump_fraction = jump_count / jump_denominator if jump_denominator else 0.0
    gates = cast(dict[str, object], contract["gates"])
    checks = {
        "content_boxes": len(session_results) >= int(cast(int, gates["required_content_boxes"])),
        "overall_pair_coverage": overall_coverage
        >= float(cast(float, gates["minimum_overall_pair_coverage"])),
        "per_session_pair_coverage": all(
            float(cast(float, row["pair_coverage"]))
            >= float(cast(float, gates["minimum_per_session_pair_coverage"]))
            for row in session_results
        ),
        "marker_jump_fraction": marker_jump_fraction
        <= float(cast(float, gates["maximum_marker_jump_fraction"])),
        "test_isolation": observed_splits <= {"train", "dev"},
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "TARGET_CONDITION_CANDIDATE_SUPPORTED" if passed else "TARGET_CONDITION_NOT_OBSERVABLE"
        ),
        "contract_sha256": contract["contract_sha256"],
        "target_manifest_sha256": manifest["manifest_sha256"],
        "selected_sessions": len(session_results),
        "selected_segments": len(session_results) * len(segment_fractions),
        "sampled_frames": len(all_frame_results),
        "session_results": session_results,
        "frame_results": all_frame_results,
        "overall_pair_coverage": overall_coverage,
        "overall_unknown_fraction": 1.0 - overall_coverage,
        "marker_jump_count": jump_count,
        "marker_jump_denominator": jump_denominator,
        "marker_jump_fraction": marker_jump_fraction,
        "checks": checks,
        "opened_shards": sorted(opened_shards.values(), key=lambda row: str(row["basename"])),
        "opened_splits": sorted(observed_splits),
        "test_frames_read": 0,
        "stored_rotation_values": sorted(observed_rotations),
        "rotation_diversity_verified": len(observed_rotations) > 1,
        "semantic_accuracy_verified": False,
        "promotion_allowed": False,
        "r2_allowed": False,
        "human_labels_consumed": False,
        "raw_rgb_persisted": False,
        "training_called": False,
        "device_input_commands_sent": 0,
        "next_action": (
            "manual_visual_confirmation_before_new_r2_contract"
            if passed
            else "repair_content_geometry_or_minimap_detector_in_new_contract"
        ),
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        path = staging / "report.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return report


def _resize_minimap(frame: np.ndarray, crop: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = map(int, crop)
    if not (0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 128):
        raise ValueError("goal canvas minimap crop differs")
    selected = frame[y0:y1, x0:x1]
    rows = np.linspace(0, len(selected) - 1, 128).astype(np.int64)
    columns = np.linspace(0, selected.shape[1] - 1, 128).astype(np.int64)
    return np.ascontiguousarray(selected[rows[:, None], columns[None, :], :])


def _mark_goal(minimap: np.ndarray, goal_xy: list[float], marker: dict[str, object]) -> np.ndarray:
    center_x = round(float(goal_xy[0]) * 127)
    center_y = round(float(goal_xy[1]) * 127)
    radius = int(cast(int, marker["radius"]))
    thickness = int(cast(int, marker["thickness"]))
    if (
        radius <= 0
        or thickness <= 0
        or thickness > radius
        or not radius <= center_x < 128 - radius
        or not radius <= center_y < 128 - radius
    ):
        raise ValueError("goal canvas marker geometry differs")
    color = np.asarray(cast(list[int], marker["rgb"]), dtype=np.uint8)
    marked = minimap.copy()
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            squared = (x - center_x) ** 2 + (y - center_y) ** 2
            if (radius - thickness) ** 2 <= squared <= radius**2:
                marked[y, x] = color
    return marked


def run_real_rgb_goal_canvas(
    contract_path: Path,
    prior_report_path: Path,
    target_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract = _load_bound_json(contract_path, "contract_sha256")
    prior = _load_bound_json(prior_report_path, "report_sha256")
    if target_root.is_symlink() or not target_root.is_dir():
        raise ValueError("target root is not a regular directory")
    manifest = _load_bound_json(target_root / "manifest.json", "manifest_sha256")
    if (
        contract.get("schema_version") != "movement-real-rgb-goal-canvas-contract-v2"
        or prior.get("status") != "TARGET_CONDITION_NOT_OBSERVABLE"
        or contract.get("prior_contract_sha256") != prior.get("contract_sha256")
        or contract.get("prior_report_sha256") != prior.get("report_sha256")
        or contract.get("target_manifest_sha256") != manifest.get("manifest_sha256")
        or contract.get("test_allowed") is not False
        or contract.get("training_allowed") is not False
        or contract.get("r2_allowed") is not False
        or contract.get("device_input_allowed") is not False
    ):
        raise ValueError("real RGB goal canvas contract binding differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("real RGB goal canvas output already exists")

    manifest_sessions = {
        str(row["session_hash"]): str(row["split"])
        for row in cast(list[dict[str, object]], manifest["sessions"])
    }
    by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cast(list[dict[str, object]], manifest["shards"]):
        for identity in cast(list[str], row["session_hashes"]):
            by_session[str(identity)].append(row)
    declarations = cast(list[dict[str, object]], contract["sessions"])
    fractions = cast(list[float], contract["segment_start_fractions"])
    segment_frames = int(cast(int, contract["frames_per_segment"]))
    crop = cast(list[int], contract["minimap_crop_xyxy"])
    goals = cast(dict[str, list[float]], contract["goals"])
    marker = cast(dict[str, object], contract["marker"])
    frame_rows: list[dict[str, object]] = []
    session_rows: list[dict[str, object]] = []
    opened_shards: dict[str, dict[str, object]] = {}
    observed_splits: set[str] = set()
    observed_rotations: set[int] = set()
    for declaration in declarations:
        identity, split = str(declaration["session_hash"]), str(declaration["split"])
        if split not in {"train", "dev"} or manifest_sessions.get(identity) != split:
            raise ValueError("goal canvas selected session split differs")
        shards = by_session[identity]
        total = sum(int(cast(int, row["row_count"])) for row in shards)
        starts = [round((total - segment_frames) * float(value)) for value in fractions]
        indices = [index for start in starts for index in range(start, start + segment_frames)]
        frames, timestamps, opened, rotations = _selected_frames(
            target_root, shards, identity, split, indices
        )
        for row in opened:
            opened_shards[str(row["basename"])] = row
            observed_splits.add(str(row["declared_split"]))
        observed_rotations.update(rotations)
        canonical, orientation, bounds = _canonical_content(
            frames, cast(dict[str, object], contract["content_box"])
        )
        nonblack_values: list[float] = []
        changed_values: list[bool] = []
        repeated_values: list[bool] = []
        for index, (frame, timestamp) in enumerate(
            zip(canonical, timestamps.tolist(), strict=True)
        ):
            if index % segment_frames and timestamp - timestamps[index - 1] != int(
                cast(int, contract["frame_period_ms"])
            ):
                raise ValueError("goal canvas segment sampling period differs")
            normalized = _normalize_content(frame, bounds)
            minimap = _resize_minimap(normalized, crop)
            primary = _mark_goal(minimap, goals["blue_marksman_bottom"], marker)
            counterfactual = _mark_goal(minimap, goals["counterfactual_top_left"], marker)
            repeated = _mark_goal(minimap, goals["blue_marksman_bottom"], marker)
            nonblack = float(np.mean(minimap.astype(np.float32).mean(axis=2) > 5.0))
            changed = bool(
                hashlib.sha256(primary.tobytes()).digest()
                != hashlib.sha256(counterfactual.tobytes()).digest()
            )
            deterministic = bool(np.array_equal(primary, repeated))
            nonblack_values.append(nonblack)
            changed_values.append(changed)
            repeated_values.append(deterministic)
            frame_rows.append(
                {
                    "session_hash": identity,
                    "split": split,
                    "segment_index": index // segment_frames,
                    "timestamp_ms": timestamp,
                    "minimap_sha256": hashlib.sha256(minimap.tobytes()).hexdigest(),
                    "primary_goal_sha256": hashlib.sha256(primary.tobytes()).hexdigest(),
                    "counterfactual_goal_sha256": hashlib.sha256(
                        counterfactual.tobytes()
                    ).hexdigest(),
                    "nonblack_crop_fraction": nonblack,
                    "counterfactual_changed": changed,
                    "deterministic_repeat": deterministic,
                }
            )
        session_rows.append(
            {
                "session_hash": identity,
                "split": split,
                "sampled_frames": len(frames),
                "detected_orientation": orientation,
                "content_box_xyxy": list(bounds),
                "stored_rotation_degrees": sorted(rotations),
                "minimum_nonblack_crop_fraction": min(nonblack_values),
                "counterfactual_change_fraction": sum(changed_values) / len(changed_values),
                "deterministic_repeat_fraction": sum(repeated_values) / len(repeated_values),
            }
        )
    gates = cast(dict[str, object], contract["gates"])
    checks = {
        "content_boxes": len(session_rows) >= int(cast(int, gates["required_content_boxes"])),
        "nonblack_minimap_crop": all(
            float(cast(float, row["minimum_nonblack_crop_fraction"]))
            >= float(cast(float, gates["minimum_nonblack_crop_fraction"]))
            for row in session_rows
        ),
        "counterfactual_goal_changes_input": all(
            float(cast(float, row["counterfactual_change_fraction"]))
            >= float(cast(float, gates["required_counterfactual_change_fraction"]))
            for row in session_rows
        ),
        "deterministic_repeat": all(
            float(cast(float, row["deterministic_repeat_fraction"]))
            >= float(cast(float, gates["required_deterministic_repeat_fraction"]))
            for row in session_rows
        ),
        "test_isolation": observed_splits <= {"train", "dev"},
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": "movement-real-rgb-goal-canvas-report-v2",
        "status": (
            "GOAL_CANVAS_GENERATION_PASSED_SELF_LOCALIZATION_UNRESOLVED"
            if passed
            else "GOAL_CANVAS_GENERATION_FAILED"
        ),
        "contract_sha256": contract["contract_sha256"],
        "prior_report_sha256": prior["report_sha256"],
        "target_manifest_sha256": manifest["manifest_sha256"],
        "actor_input": contract["actor_input"],
        "macro_goal_source": "fixed_role_side_lane_contract",
        "selected_sessions": len(session_rows),
        "selected_segments": len(session_rows) * len(fractions),
        "sampled_frames": len(frame_rows),
        "session_results": session_rows,
        "frame_results": frame_rows,
        "checks": checks,
        "opened_shards": sorted(opened_shards.values(), key=lambda row: str(row["basename"])),
        "opened_splits": sorted(observed_splits),
        "stored_rotation_values": sorted(observed_rotations),
        "test_frames_read": 0,
        "raw_rgb_persisted": False,
        "human_labels_consumed": False,
        "target_detection_required": False,
        "player_localization_required_for_canvas": False,
        "player_localization_verified": False,
        "movement_policy_value_verified": False,
        "semantic_lane_coordinate_verified": False,
        "training_called": False,
        "promotion_allowed": False,
        "r2_allowed": False,
        "device_input_commands_sent": 0,
        "next_action": (
            "train_simulator_goal_canvas_only_after_separate_learnability_contract"
            if passed
            else "repair_goal_canvas_geometry_in_new_contract"
        ),
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        path = staging / "report.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return report


def _mask_components(mask: np.ndarray) -> list[tuple[int, float, float, int, int]]:
    if mask.shape != (128, 128) or mask.dtype != np.bool_:
        raise ValueError("player cue color mask differs")
    seen = np.zeros(mask.shape, dtype=np.bool_)
    components: list[tuple[int, float, float, int, int]] = []
    for raw_y, raw_x in zip(*np.where(mask), strict=True):
        y, x = int(raw_y), int(raw_x)
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        points: list[tuple[int, int]] = []
        while stack:
            current_y, current_x = stack.pop()
            points.append((current_y, current_x))
            for delta_y, delta_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_y, next_x = current_y + delta_y, current_x + delta_x
                if (
                    0 <= next_y < 128
                    and 0 <= next_x < 128
                    and mask[next_y, next_x]
                    and not seen[next_y, next_x]
                ):
                    seen[next_y, next_x] = True
                    stack.append((next_y, next_x))
        if len(points) < 3:
            continue
        values = np.asarray(points, dtype=np.int16)
        components.append(
            (
                len(points),
                float(values[:, 0].mean()),
                float(values[:, 1].mean()),
                int(np.ptp(values[:, 0]) + 1),
                int(np.ptp(values[:, 1]) + 1),
            )
        )
    return components


def _player_candidates(
    frame: np.ndarray, contract: dict[str, object]
) -> list[tuple[float, float, float]]:
    rgb = frame.astype(np.int16)
    red, green, blue = (rgb[..., index] for index in range(3))
    color = cast(dict[str, object], contract["color"])
    green_mask = (
        (green > int(cast(int, color["green_minimum"])))
        & (green - red > int(cast(int, color["green_red_margin"])))
        & (green - blue > int(cast(int, color["green_blue_margin"])))
    )
    red_mask = (
        (red > int(cast(int, color["red_minimum"])))
        & (red - green > int(cast(int, color["red_green_margin"])))
        & (red - blue > int(cast(int, color["red_blue_margin"])))
    )
    config = cast(dict[str, object], contract["components"])
    green_size = cast(list[int], config["green_size"])
    green_extent = cast(list[int], config["green_extent"])
    red_size = cast(list[int], config["red_size"])
    red_extent = cast(list[int], config["red_extent"])
    greens = [
        item
        for item in _mask_components(green_mask)
        if green_size[0] <= item[0] <= green_size[1]
        and green_extent[0] <= item[3] <= green_extent[1]
        and green_extent[0] <= item[4] <= green_extent[1]
        and 5 < item[1] < 123
        and 5 < item[2] < 123
    ]
    reds = [
        item
        for item in _mask_components(red_mask)
        if red_size[0] <= item[0] <= red_size[1]
        and red_extent[0] <= item[3] <= red_extent[1]
        and red_extent[0] <= item[4] <= red_extent[1]
        and 4 < item[1] < 124
        and 4 < item[2] < 124
    ]
    maximum = float(cast(float, config["maximum_pair_l1_distance"]))
    candidates = []
    for green_item in greens:
        distances = [
            abs(green_item[1] - red_item[1]) + abs(green_item[2] - red_item[2]) for red_item in reds
        ]
        if distances and min(distances) <= maximum:
            candidates.append((green_item[1], green_item[2], min(distances)))
    return candidates


def run_real_player_cue_preflight(
    contract_path: Path, session_root: Path, output_dir: Path
) -> dict[str, object]:
    contract = _load_bound_json(contract_path, "contract_sha256")
    if (
        contract.get("schema_version") != "movement-real-player-cue-contract-v1"
        or contract.get("training_allowed") is not False
        or contract.get("test_allowed") is not False
        or contract.get("r2_allowed") is not False
        or contract.get("device_input_allowed") is not False
    ):
        raise ValueError("real player cue contract differs")
    if session_root.is_symlink() or not session_root.is_dir():
        raise ValueError("real player cue session root differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("real player cue output already exists")
    config = cast(dict[str, object], contract["components"])
    gates = cast(dict[str, object], contract["gates"])
    session_results: list[dict[str, object]] = []
    opened_shards: list[dict[str, object]] = []
    for declaration in cast(list[dict[str, object]], contract["sessions"]):
        basename = str(declaration["basename"])
        if Path(basename).name != basename:
            raise ValueError("real player cue session basename differs")
        directory = session_root / basename
        summary_path = directory / "summary.json"
        if _file_sha256(summary_path) != declaration["summary_sha256"]:
            raise ValueError("real player cue summary file differs")
        summary = _load_bound_json(summary_path, "summary_sha256")
        if (
            summary.get("status") != "PASSED"
            or summary.get("derived_roi_rgb_persisted") is not True
            or summary.get("raw_frames_persisted") is not False
        ):
            raise ValueError("real player cue source summary differs")
        detections = single_candidates = missing_streak = maximum_missing = 0
        previous: tuple[float, float] | None = None
        jumps: list[float] = []
        frames_seen = 0
        for raw_row in cast(list[dict[str, object]], summary["observation_shards"]):
            shard_basename = str(raw_row["path"])
            path = directory / "shards" / shard_basename
            if (
                Path(shard_basename).name != shard_basename
                or path.is_symlink()
                or not path.is_file()
                or _file_sha256(path) != raw_row["sha256"]
            ):
                raise ValueError("real player cue shard differs")
            with np.load(path, allow_pickle=False) as shard:
                frames = shard["minimap_rgb"]
                timestamps = shard["scheduled_elapsed_ms"]
                if (
                    frames.dtype != np.uint8
                    or frames.shape[1:] != (128, 128, 3)
                    or len(frames) != raw_row["rows"]
                    or timestamps.shape != (len(frames),)
                ):
                    raise ValueError("real player cue shard arrays differ")
                for frame in frames:
                    frames_seen += 1
                    candidates = _player_candidates(frame, contract)
                    if not candidates:
                        missing_streak += 1
                        maximum_missing = max(maximum_missing, missing_streak)
                        if missing_streak > int(cast(int, config["reset_after_missing_frames"])):
                            previous = None
                        continue
                    detections += 1
                    single_candidates += int(len(candidates) == 1)
                    if previous is None:
                        selected = min(candidates, key=lambda item: item[2])
                    else:
                        selected = min(
                            candidates,
                            key=lambda item: float(
                                np.linalg.norm(np.asarray(item[:2]) - np.asarray(previous))
                            ),
                        )
                        jumps.append(
                            float(np.linalg.norm(np.asarray(selected[:2]) - np.asarray(previous)))
                        )
                    previous = (selected[0], selected[1])
                    missing_streak = 0
            opened_shards.append(
                {"session": basename, "basename": shard_basename, "sha256": raw_row["sha256"]}
            )
        coverage = detections / frames_seen
        single_fraction = single_candidates / detections if detections else 0.0
        jump_p95 = float(np.percentile(jumps, 95)) if jumps else 128.0
        session_results.append(
            {
                "session": basename,
                "frames": frames_seen,
                "detections": detections,
                "coverage": coverage,
                "single_candidate_fraction": single_fraction,
                "player_jump_p95": jump_p95,
                "maximum_missing_streak": maximum_missing,
            }
        )
    checks = {
        "coverage": all(
            float(cast(float, row["coverage"]))
            >= float(cast(float, gates["minimum_coverage_per_session"]))
            for row in session_results
        ),
        "single_candidate": all(
            float(cast(float, row["single_candidate_fraction"]))
            >= float(cast(float, gates["minimum_single_candidate_fraction"]))
            for row in session_results
        ),
        "player_jump": all(
            float(cast(float, row["player_jump_p95"]))
            <= float(cast(float, gates["maximum_player_jump_p95"]))
            for row in session_results
        ),
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": "movement-real-player-cue-report-v1",
        "status": "REAL_PLAYER_CUE_PASSED" if passed else "REAL_PLAYER_CUE_FAILED",
        "contract_sha256": contract["contract_sha256"],
        "sessions": session_results,
        "checks": checks,
        "opened_shards": opened_shards,
        "source": contract["input"],
        "semantic_identity_verified": False,
        "direction_accuracy_verified": False,
        "human_labels_consumed": False,
        "raw_rgb_persisted": False,
        "new_recording_used": False,
        "training_called": False,
        "test_frames_read": 0,
        "r2_allowed": False,
        "device_input_commands_sent": 0,
        "next_action": (
            "bind_player_cue_to_goal_canvas_before_policy_training"
            if passed
            else "stop_real_player_cue_lineage_without_threshold_tuning"
        ),
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return report


def _goal_direction(
    player_yx: tuple[float, float], goal_yx: tuple[float, float], stop_radius: float
) -> str:
    delta_y = goal_yx[0] - player_yx[0]
    delta_x = goal_yx[1] - player_yx[1]
    if math.hypot(delta_y, delta_x) <= stop_radius:
        return "STOP"
    directions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    sector = int(round(math.atan2(delta_x, -delta_y) / (math.pi / 4.0))) % 8
    return directions[sector]


def run_real_player_goal_continuity(
    contract_path: Path,
    player_report_path: Path,
    goal_report_path: Path,
    session_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract = _load_bound_json(contract_path, "contract_sha256")
    if (
        contract.get("schema_version") != "movement-real-player-goal-continuity-contract-v1"
        or contract.get("training_allowed") is not False
        or contract.get("test_allowed") is not False
        or contract.get("r2_allowed") is not False
        or contract.get("device_input_allowed") is not False
    ):
        raise ValueError("player goal continuity contract differs")
    player_report = _load_bound_json(player_report_path, "report_sha256")
    goal_report = _load_bound_json(goal_report_path, "report_sha256")
    lineage = cast(dict[str, object], contract["lineage"])
    if (
        _file_sha256(player_report_path) != lineage["player_report_file_sha256"]
        or player_report.get("report_sha256") != lineage["player_report_sha256"]
        or player_report.get("status") != "REAL_PLAYER_CUE_PASSED"
        or _file_sha256(goal_report_path) != lineage["goal_report_file_sha256"]
        or goal_report.get("report_sha256") != lineage["goal_report_sha256"]
        or goal_report.get("status") != "GOAL_CANVAS_GENERATION_PASSED_SELF_LOCALIZATION_UNRESOLVED"
    ):
        raise ValueError("player goal continuity lineage differs")
    if session_root.is_symlink() or not session_root.is_dir():
        raise ValueError("player goal continuity session root differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("player goal continuity output already exists")

    player_config = cast(dict[str, object], contract["player_cue"])
    goal_config = cast(dict[str, object], contract["goal_canvas"])
    gates = cast(dict[str, object], contract["gates"])
    goal_xy = cast(list[float], goal_config["goal_xy_relative"])
    goal_yx = (round(goal_xy[1] * 127), round(goal_xy[0] * 127))
    marker = cast(dict[str, object], goal_config["marker"])
    cue_contract: dict[str, object] = {
        "color": contract["color"],
        "components": player_config,
    }
    confirmation_frames = int(cast(int, contract["confirmation_frames"]))
    stop_radius = float(cast(float, contract["stop_radius_pixels"]))
    session_results: list[dict[str, object]] = []
    opened_shards: list[dict[str, object]] = []

    for declaration in cast(list[dict[str, object]], contract["sessions"]):
        basename = str(declaration["basename"])
        if Path(basename).name != basename:
            raise ValueError("player goal continuity session basename differs")
        directory = session_root / basename
        summary_path = directory / "summary.json"
        if _file_sha256(summary_path) != declaration["summary_sha256"]:
            raise ValueError("player goal continuity summary file differs")
        summary = _load_bound_json(summary_path, "summary_sha256")
        if (
            summary.get("status") != "PASSED"
            or summary.get("derived_roi_rgb_persisted") is not True
            or summary.get("raw_frames_persisted") is not False
        ):
            raise ValueError("player goal continuity source summary differs")

        frames_seen = detections = stable_frames = 0
        raw_changes = raw_adjacencies = stable_switches = 0
        canvas_changed = canvas_deterministic = 0
        missing_streak = 0
        previous_player: tuple[float, float] | None = None
        previous_raw: str | None = None
        stable_direction: str | None = None
        pending_direction: str | None = None
        pending_count = 0
        direction_counts: dict[str, int] = defaultdict(int)
        stable_direction_counts: dict[str, int] = defaultdict(int)
        first_timestamp: int | None = None
        last_timestamp: int | None = None

        for raw_row in cast(list[dict[str, object]], summary["observation_shards"]):
            shard_basename = str(raw_row["path"])
            path = directory / "shards" / shard_basename
            if (
                Path(shard_basename).name != shard_basename
                or path.is_symlink()
                or not path.is_file()
                or _file_sha256(path) != raw_row["sha256"]
            ):
                raise ValueError("player goal continuity shard differs")
            with np.load(path, allow_pickle=False) as shard:
                frames = shard["minimap_rgb"]
                timestamps = shard["scheduled_elapsed_ms"]
                if (
                    frames.dtype != np.uint8
                    or frames.shape[1:] != (128, 128, 3)
                    or len(frames) != raw_row["rows"]
                    or timestamps.shape != (len(frames),)
                ):
                    raise ValueError("player goal continuity shard arrays differ")
                for frame, timestamp_value in zip(frames, timestamps, strict=True):
                    frames_seen += 1
                    timestamp = int(timestamp_value)
                    first_timestamp = timestamp if first_timestamp is None else first_timestamp
                    last_timestamp = timestamp
                    marked = _mark_goal(frame, goal_xy, marker)
                    repeated = _mark_goal(frame, goal_xy, marker)
                    canvas_changed += int(not np.array_equal(marked, frame))
                    canvas_deterministic += int(np.array_equal(marked, repeated))
                    candidates = _player_candidates(frame, cue_contract)
                    if not candidates:
                        missing_streak += 1
                        previous_raw = None
                        pending_direction = None
                        pending_count = 0
                        if missing_streak > int(
                            cast(int, player_config["reset_after_missing_frames"])
                        ):
                            previous_player = None
                            stable_direction = None
                        continue
                    detections += 1
                    if previous_player is None:
                        selected = min(candidates, key=lambda item: item[2])
                    else:
                        selected = min(
                            candidates,
                            key=lambda item: float(
                                np.linalg.norm(np.asarray(item[:2]) - np.asarray(previous_player))
                            ),
                        )
                    previous_player = (selected[0], selected[1])
                    missing_streak = 0
                    direction = _goal_direction(previous_player, goal_yx, stop_radius)
                    direction_counts[direction] += 1
                    if previous_raw is not None:
                        raw_adjacencies += 1
                        raw_changes += int(direction != previous_raw)
                    previous_raw = direction
                    if pending_direction == direction:
                        pending_count += 1
                    else:
                        pending_direction = direction
                        pending_count = 1
                    if pending_count >= confirmation_frames:
                        if stable_direction is not None and stable_direction != direction:
                            stable_switches += 1
                        stable_direction = direction
                    if stable_direction is not None:
                        stable_frames += 1
                        stable_direction_counts[stable_direction] += 1
            opened_shards.append(
                {"session": basename, "basename": shard_basename, "sha256": raw_row["sha256"]}
            )

        duration_minutes = (
            (last_timestamp - first_timestamp) / 60_000.0
            if first_timestamp is not None
            and last_timestamp is not None
            and last_timestamp > first_timestamp
            else 0.0
        )
        session_results.append(
            {
                "session": basename,
                "frames": frames_seen,
                "detections": detections,
                "raw_direction_coverage": detections / frames_seen,
                "stable_direction_coverage": stable_frames / frames_seen,
                "raw_direction_change_fraction": raw_changes / raw_adjacencies
                if raw_adjacencies
                else 0.0,
                "stable_switches": stable_switches,
                "stable_switches_per_minute": stable_switches / duration_minutes
                if duration_minutes
                else 0.0,
                "raw_direction_counts": dict(sorted(direction_counts.items())),
                "stable_direction_counts": dict(sorted(stable_direction_counts.items())),
                "canvas_changed_fraction": canvas_changed / frames_seen,
                "canvas_deterministic_fraction": canvas_deterministic / frames_seen,
            }
        )

    checks = {
        "raw_direction_coverage": all(
            float(cast(float, row["raw_direction_coverage"]))
            >= float(cast(float, gates["minimum_raw_direction_coverage_per_session"]))
            for row in session_results
        ),
        "stable_direction_coverage": all(
            float(cast(float, row["stable_direction_coverage"]))
            >= float(cast(float, gates["minimum_stable_direction_coverage_per_session"]))
            for row in session_results
        ),
        "raw_direction_continuity": all(
            float(cast(float, row["raw_direction_change_fraction"]))
            <= float(cast(float, gates["maximum_raw_direction_change_fraction"]))
            for row in session_results
        ),
        "stable_direction_continuity": all(
            float(cast(float, row["stable_switches_per_minute"]))
            <= float(cast(float, gates["maximum_stable_switches_per_minute"]))
            for row in session_results
        ),
        "goal_canvas_changed": all(
            float(cast(float, row["canvas_changed_fraction"])) == 1.0 for row in session_results
        ),
        "goal_canvas_deterministic": all(
            float(cast(float, row["canvas_deterministic_fraction"])) == 1.0
            for row in session_results
        ),
    }
    passed = all(checks.values())
    observed_directions = sorted(
        {
            direction
            for row in session_results
            for direction in cast(dict[str, int], row["raw_direction_counts"])
        }
    )
    report: dict[str, object] = {
        "schema_version": "movement-real-player-goal-continuity-report-v1",
        "status": "REAL_PLAYER_GOAL_CONTINUITY_PASSED"
        if passed
        else "REAL_PLAYER_GOAL_CONTINUITY_FAILED",
        "contract_sha256": contract["contract_sha256"],
        "player_report_sha256": player_report["report_sha256"],
        "goal_report_sha256": goal_report["report_sha256"],
        "goal_xy_relative": goal_xy,
        "goal_yx_pixels": list(goal_yx),
        "direction_order": contract["direction_order"],
        "confirmation_frames": confirmation_frames,
        "sessions": session_results,
        "checks": checks,
        "observed_direction_union": observed_directions,
        "all_nine_directions_observed": len(observed_directions) == 9,
        "opened_shards": opened_shards,
        "semantic_player_identity_verified": False,
        "semantic_lane_coordinate_verified": False,
        "direction_accuracy_verified": False,
        "continuity_only": True,
        "policy_training_allowed": False,
        "human_labels_consumed": False,
        "raw_rgb_persisted": False,
        "training_called": False,
        "test_frames_read": 0,
        "r2_allowed": False,
        "device_input_commands_sent": 0,
        "next_action": "freeze_read_only_composition_evidence"
        if passed
        else "stop_without_threshold_tuning",
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return report


def _counterfactual_goal(
    player_yx: tuple[float, float], action: str, distance: int, marker_radius: int
) -> tuple[int, int] | None:
    offsets = {
        "STOP": (0, 0),
        "N": (-distance, 0),
        "S": (distance, 0),
        "W": (0, -distance),
        "E": (0, distance),
        "NW": (-distance, -distance),
        "NE": (-distance, distance),
        "SW": (distance, -distance),
        "SE": (distance, distance),
    }
    delta_y, delta_x = offsets[action]
    goal_y = round(player_yx[0]) + delta_y
    goal_x = round(player_yx[1]) + delta_x
    if not (
        marker_radius <= goal_y < 128 - marker_radius
        and marker_radius <= goal_x < 128 - marker_radius
    ):
        return None
    return goal_y, goal_x


def materialize_real_counterfactual_overfit32(
    contract_path: Path,
    continuity_report_path: Path,
    session_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract = _load_bound_json(contract_path, "contract_sha256")
    report = _load_bound_json(continuity_report_path, "report_sha256")
    if (
        contract.get("schema_version") != "movement-real-counterfactual-overfit32-data-contract-v1"
        or contract.get("test_allowed") is not False
        or contract.get("training_allowed") is not False
        or contract.get("r2_allowed") is not False
        or contract.get("device_input_allowed") is not False
        or report.get("status") != "REAL_PLAYER_GOAL_CONTINUITY_PASSED"
        or report.get("policy_training_allowed") is not False
        or _file_sha256(continuity_report_path) != contract.get("continuity_report_file_sha256")
        or report.get("report_sha256") != contract.get("continuity_report_sha256")
    ):
        raise ValueError("real counterfactual data contract differs")
    if session_root.is_symlink() or not session_root.is_dir():
        raise ValueError("real counterfactual session root differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("real counterfactual output already exists")

    actions = ("STOP", "N", "S", "W", "E", "NW", "NE", "SW", "SE")
    expected_counts = {"STOP": 8, **{action: 3 for action in actions[1:]}}
    if (
        contract.get("action_order") != list(actions)
        or contract.get("class_counts") != expected_counts
    ):
        raise ValueError("real counterfactual action contract differs")
    sequence_frames = int(cast(int, contract["sequence_frames"]))
    frame_period_ms = int(cast(int, contract["frame_period_ms"]))
    goal_distance = int(cast(int, contract["goal_distance_pixels"]))
    stop_radius = float(cast(float, contract["stop_radius_pixels"]))
    marker = cast(dict[str, object], contract["marker"])
    marker_radius = int(cast(int, marker["radius"]))
    player_config = cast(dict[str, object], contract["player_cue"])
    cue_contract: dict[str, object] = {
        "color": contract["color"],
        "components": player_config,
    }
    sessions: list[tuple[str, np.ndarray, np.ndarray, list[tuple[float, float] | None]]] = []
    opened_shards: list[dict[str, object]] = []
    for declaration in cast(list[dict[str, object]], contract["sessions"]):
        basename = str(declaration["basename"])
        directory = session_root / basename
        summary_path = directory / "summary.json"
        if (
            Path(basename).name != basename
            or _file_sha256(summary_path) != declaration["summary_sha256"]
        ):
            raise ValueError("real counterfactual summary differs")
        summary = _load_bound_json(summary_path, "summary_sha256")
        if (
            summary.get("status") != "PASSED"
            or summary.get("derived_roi_rgb_persisted") is not True
            or summary.get("raw_frames_persisted") is not False
        ):
            raise ValueError("real counterfactual source summary differs")
        frame_parts: list[np.ndarray] = []
        timestamp_parts: list[np.ndarray] = []
        for raw_row in cast(list[dict[str, object]], summary["observation_shards"]):
            shard_basename = str(raw_row["path"])
            path = directory / "shards" / shard_basename
            if (
                Path(shard_basename).name != shard_basename
                or path.is_symlink()
                or not path.is_file()
                or _file_sha256(path) != raw_row["sha256"]
            ):
                raise ValueError("real counterfactual shard differs")
            with np.load(path, allow_pickle=False) as shard:
                frames = shard["minimap_rgb"]
                timestamps = shard["scheduled_elapsed_ms"]
                if (
                    frames.dtype != np.uint8
                    or frames.shape[1:] != (128, 128, 3)
                    or len(frames) != raw_row["rows"]
                    or timestamps.shape != (len(frames),)
                ):
                    raise ValueError("real counterfactual shard arrays differ")
                frame_parts.append(frames.copy())
                timestamp_parts.append(timestamps.copy())
            opened_shards.append(
                {"session": basename, "basename": shard_basename, "sha256": raw_row["sha256"]}
            )
        all_frames = np.concatenate(frame_parts)
        all_timestamps = np.concatenate(timestamp_parts)
        positions: list[tuple[float, float] | None] = []
        previous: tuple[float, float] | None = None
        missing_streak = 0
        for frame in all_frames:
            candidates = _player_candidates(frame, cue_contract)
            if not candidates:
                positions.append(None)
                missing_streak += 1
                if missing_streak > int(cast(int, player_config["reset_after_missing_frames"])):
                    previous = None
                continue
            selected = (
                min(candidates, key=lambda item: item[2])
                if previous is None
                else min(
                    candidates,
                    key=lambda item: float(
                        np.linalg.norm(np.asarray(item[:2]) - np.asarray(previous))
                    ),
                )
            )
            previous = (selected[0], selected[1])
            positions.append(previous)
            missing_streak = 0
        sessions.append((basename, all_frames, all_timestamps, positions))

    sample_actions = [action for action in actions for _ in range(expected_counts[action])]
    required_source_windows = int(cast(int, contract["unique_source_windows"]))
    used_indices: dict[str, set[int]] = {
        basename: set() for basename, _frames, _times, _positions in sessions
    }
    source_windows: list[tuple[str, int, int, np.ndarray, np.ndarray, tuple[float, float]]] = []
    for basename, frames, timestamps, positions in sessions:
        for end in range(sequence_frames - 1, len(frames)):
            start = end - sequence_frames + 1
            if any(index in used_indices[basename] for index in range(start, end + 1)):
                continue
            window_positions = positions[start : end + 1]
            if any(position is None for position in window_positions):
                continue
            window_times = timestamps[start : end + 1]
            if not np.all(np.diff(window_times) == frame_period_ms):
                continue
            player = cast(tuple[float, float], window_positions[-1])
            goals = [
                _counterfactual_goal(player, action, goal_distance, marker_radius)
                for action in actions
            ]
            if any(goal is None for goal in goals) or any(
                _goal_direction(player, cast(tuple[int, int], goal), stop_radius) != action
                for action, goal in zip(actions, goals, strict=True)
            ):
                continue
            source_windows.append(
                (basename, start, end, frames[start : end + 1], window_times, player)
            )
            used_indices[basename].update(range(start, end + 1))
            if len(source_windows) == required_source_windows:
                break
        if len(source_windows) == required_source_windows:
            break
    if len(source_windows) != required_source_windows:
        raise ValueError("real counterfactual source-window support differs")

    clips: list[np.ndarray] = []
    labels: list[int] = []
    session_ids: list[str] = []
    end_timestamps: list[int] = []
    goals_xy: list[tuple[int, int]] = []
    players_yx: list[tuple[float, float]] = []
    selected_windows: list[dict[str, object]] = []
    for sample_index, action in enumerate(sample_actions):
        source_window_id = sample_index % len(source_windows)
        basename, start, end, source_clip, window_times, player = source_windows[source_window_id]
        goal_yx = _counterfactual_goal(player, action, goal_distance, marker_radius)
        if goal_yx is None:
            raise ValueError("real counterfactual selected geometry differs")
        goal_y, goal_x = goal_yx
        selected_windows.append(
            {
                "sample": sample_index,
                "source_window": source_window_id,
                "session": basename,
                "start_index": start,
                "end_index": end,
                "end_timestamp_ms": int(window_times[-1]),
                "action": action,
            }
        )
        goal_xy_relative = [goal_x / 127.0, goal_y / 127.0]
        clips.append(
            np.stack([_mark_goal(frame, goal_xy_relative, marker) for frame in source_clip])
        )
        labels.append(actions.index(action))
        session_ids.append(basename)
        end_timestamps.append(int(window_times[-1]))
        goals_xy.append((goal_x, goal_y))
        players_yx.append(player)

    base_player = source_windows[0][-1]
    counterfactual_actions = {
        action
        for action in actions
        if (goal := _counterfactual_goal(base_player, action, goal_distance, marker_radius))
        is not None
        and _goal_direction(base_player, goal, stop_radius) == action
    }
    if len(counterfactual_actions) != len(actions):
        raise ValueError("real counterfactual nine-way geometry differs")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        dataset_path = staging / "overfit32.npz"
        with dataset_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                rgb_sequence=np.stack(clips).astype(np.uint8),
                label=np.asarray(labels, dtype=np.int64),
                action=np.asarray(sample_actions),
                session_id=np.asarray(session_ids),
                end_timestamp_ms=np.asarray(end_timestamps, dtype=np.int64),
                goal_xy=np.asarray(goals_xy, dtype=np.float32),
                player_yx=np.asarray(players_yx, dtype=np.float32),
                contract_sha256=np.asarray([contract["contract_sha256"]]),
                continuity_report_sha256=np.asarray([report["report_sha256"]]),
            )
            handle.flush()
            os.fsync(handle.fileno())
        actual_counts = {action: sample_actions.count(action) for action in actions}
        data_report: dict[str, object] = {
            "schema_version": "movement-real-counterfactual-overfit32-data-report-v1",
            "status": "PASSED",
            "contract_sha256": contract["contract_sha256"],
            "continuity_report_sha256": report["report_sha256"],
            "dataset_sha256": _file_sha256(dataset_path),
            "samples": len(clips),
            "frames_per_sample": sequence_frames,
            "derived_rgb_frames": len(clips) * sequence_frames,
            "class_counts": actual_counts,
            "source_session_counts": dict(sorted(Counter(session_ids).items())),
            "unique_source_windows": len(source_windows),
            "source_windows_nonoverlapping": True,
            "samples_reuse_source_windows_with_different_goals": True,
            "cross_session_windows": 0,
            "counterfactual_classes_verified": len(counterfactual_actions),
            "selected_windows": selected_windows,
            "structured_coordinates_in_model_input": False,
            "direction_arrow_in_model_input": False,
            "labels_are_executed_actions": False,
            "labels_are_geometric_counterfactuals": True,
            "semantic_player_identity_verified": False,
            "direction_accuracy_verified": False,
            "human_labels_consumed": False,
            "raw_fullscreen_rgb_persisted": False,
            "test_frames_read": 0,
            "training_called": False,
            "formal_training_allowed": False,
            "r2_allowed": False,
            "device_input_commands_sent": 0,
            "opened_shards": opened_shards,
        }
        data_report["report_sha256"] = _object_sha256(data_report)
        (staging / "report.json").write_text(
            json.dumps(data_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return data_report


def _filtered_player_positions(
    frames: np.ndarray,
    cue_contract: dict[str, object],
    exclusion_xyxy: tuple[int, int, int, int],
) -> tuple[list[list[tuple[float, float, float]]], list[tuple[float, float] | None]]:
    x0, y0, x1, y1 = exclusion_xyxy
    candidates_by_frame: list[list[tuple[float, float, float]]] = []
    positions: list[tuple[float, float] | None] = []
    previous: tuple[float, float] | None = None
    missing = 0
    reset_after = int(
        cast(
            int,
            cast(dict[str, object], cue_contract["components"])["reset_after_missing_frames"],
        )
    )
    for frame in frames:
        raw = _player_candidates(frame, cue_contract)
        candidates_by_frame.append(raw)
        candidates = [item for item in raw if not (x0 <= item[1] < x1 and y0 <= item[0] < y1)]
        if not candidates:
            positions.append(None)
            missing += 1
            if missing > reset_after:
                previous = None
            continue
        selected = (
            min(candidates, key=lambda item: item[2])
            if previous is None
            else min(
                candidates,
                key=lambda item: float(np.linalg.norm(np.asarray(item[:2]) - np.asarray(previous))),
            )
        )
        previous = (selected[0], selected[1])
        positions.append(previous)
        missing = 0
    return candidates_by_frame, positions


def _write_localization_contact_sheet(
    path: Path,
    frames: np.ndarray,
    indices: list[int],
    candidates: list[list[tuple[float, float, float]]],
    positions: list[tuple[float, float] | None],
    exclusion_xyxy: tuple[int, int, int, int],
) -> None:
    from PIL import Image, ImageDraw

    columns = 4
    scale = 2
    rows = math.ceil(len(indices) / columns)
    canvas = Image.new("RGB", (columns * 128 * scale, rows * 128 * scale))
    for ordinal, index in enumerate(indices):
        image = Image.fromarray(frames[index].copy())
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = exclusion_xyxy
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 225, 0), width=1)
        for y, x, _distance in candidates[index]:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 0, 255), width=2)
        if positions[index] is not None:
            y, x = cast(tuple[float, float], positions[index])
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(0, 255, 255), width=2)
        draw.text((2, 2), str(index), fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
        image = image.resize((128 * scale, 128 * scale), Image.Resampling.NEAREST)
        canvas.paste(
            image,
            ((ordinal % columns) * 128 * scale, (ordinal // columns) * 128 * scale),
        )
    canvas.save(path, format="PNG", optimize=False)


def run_real_player_localization_audit_v2(
    contract_path: Path,
    prior_report_path: Path,
    session_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract = _load_bound_json(contract_path, "contract_sha256")
    prior = _load_bound_json(prior_report_path, "report_sha256")
    if (
        contract.get("schema_version") != "movement-real-player-localization-audit-v2"
        or contract.get("test_allowed") is not False
        or contract.get("training_allowed") is not False
        or contract.get("r2_allowed") is not False
        or contract.get("device_input_allowed") is not False
        or prior.get("status") != "REAL_PLAYER_CUE_PASSED"
        or _file_sha256(prior_report_path) != contract.get("prior_report_file_sha256")
        or prior.get("report_sha256") != contract.get("prior_report_sha256")
    ):
        raise ValueError("player localization audit v2 contract differs")
    if session_root.is_symlink() or not session_root.is_dir():
        raise ValueError("player localization audit v2 session root differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("player localization audit v2 output already exists")

    exclusion = tuple(map(int, cast(list[int], contract["excluded_ui_xyxy"])))
    if exclusion != (112, 0, 128, 16):
        raise ValueError("player localization audit v2 exclusion differs")
    movement_names = (
        "wait",
        "north",
        "north_east",
        "east",
        "south_east",
        "south",
        "south_west",
        "west",
        "north_west",
    )
    movement_vectors = {
        "north": (-1.0, 0.0),
        "north_east": (-1.0, 1.0),
        "east": (0.0, 1.0),
        "south_east": (1.0, 1.0),
        "south": (1.0, 0.0),
        "south_west": (1.0, -1.0),
        "west": (0.0, -1.0),
        "north_west": (-1.0, -1.0),
    }
    frame_period_ms = int(cast(int, contract["frame_period_ms"]))
    response_lag_ms = int(cast(int, contract["response_lag_ms"]))
    if response_lag_ms % frame_period_ms:
        raise ValueError("player localization response lag differs")
    response_lag_frames = response_lag_ms // frame_period_ms
    cue_contract: dict[str, object] = {
        "color": contract["color"],
        "components": contract["components"],
    }
    gates = cast(dict[str, object], contract["gates"])
    visual_qa = cast(dict[str, str], contract["developer_visual_qa"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    session_results: list[dict[str, object]] = []
    opened_shards: list[dict[str, object]] = []
    contact_sheets: list[dict[str, object]] = []
    try:
        for declaration in cast(list[dict[str, object]], contract["sessions"]):
            basename = str(declaration["basename"])
            directory = session_root / basename
            summary_path = directory / "summary.json"
            if (
                Path(basename).name != basename
                or _file_sha256(summary_path) != declaration["summary_sha256"]
            ):
                raise ValueError("player localization audit v2 summary differs")
            summary = _load_bound_json(summary_path, "summary_sha256")
            if (
                summary.get("status") != "PASSED"
                or summary.get("derived_roi_rgb_persisted") is not True
                or summary.get("raw_frames_persisted") is not False
            ):
                raise ValueError("player localization audit v2 source summary differs")
            frame_parts: list[np.ndarray] = []
            timestamp_parts: list[np.ndarray] = []
            movement_parts: list[np.ndarray] = []
            sent_parts: list[np.ndarray] = []
            for raw_row in cast(list[dict[str, object]], summary["observation_shards"]):
                shard_name = str(raw_row["path"])
                shard_path = directory / "shards" / shard_name
                if (
                    Path(shard_name).name != shard_name
                    or shard_path.is_symlink()
                    or _file_sha256(shard_path) != raw_row["sha256"]
                ):
                    raise ValueError("player localization audit v2 shard differs")
                with np.load(shard_path, allow_pickle=False) as shard:
                    required = {
                        "minimap_rgb",
                        "scheduled_elapsed_ms",
                        "movement_id",
                        "movement_input_sent",
                    }
                    if not required.issubset(shard.files):
                        raise ValueError("player localization audit v2 shard fields differ")
                    frame_parts.append(shard["minimap_rgb"].copy())
                    timestamp_parts.append(shard["scheduled_elapsed_ms"].copy())
                    movement_parts.append(shard["movement_id"].copy())
                    sent_parts.append(shard["movement_input_sent"].copy())
                opened_shards.append(
                    {"session": basename, "basename": shard_name, "sha256": raw_row["sha256"]}
                )
            frames = np.concatenate(frame_parts)
            timestamps = np.concatenate(timestamp_parts)
            movement_ids = np.concatenate(movement_parts).astype(np.int64)
            movement_sent = np.concatenate(sent_parts).astype(bool)
            if (
                frames.dtype != np.uint8
                or frames.shape[1:] != (128, 128, 3)
                or timestamps.shape != (len(frames),)
                or movement_ids.shape != (len(frames),)
                or movement_sent.shape != (len(frames),)
                or not np.all(np.diff(timestamps) == frame_period_ms)
            ):
                raise ValueError("player localization audit v2 arrays differ")
            candidates, positions = _filtered_player_positions(frames, cue_contract, exclusion)
            raw_detected = sum(bool(row) for row in candidates)
            filtered_detected = sum(position is not None for position in positions)
            single = sum(
                sum(
                    not (
                        exclusion[0] <= item[1] < exclusion[2]
                        and exclusion[1] <= item[0] < exclusion[3]
                    )
                    for item in row
                )
                == 1
                for row in candidates
            )
            rejected_ui = sum(
                any(
                    exclusion[0] <= item[1] < exclusion[2]
                    and exclusion[1] <= item[0] < exclusion[3]
                    for item in row
                )
                for row in candidates
            )
            valid_indices = [
                index for index, position in enumerate(positions) if position is not None
            ]
            jumps = [
                math.dist(
                    cast(tuple[float, float], positions[left]),
                    cast(tuple[float, float], positions[right]),
                )
                for left, right in zip(valid_indices, valid_indices[1:], strict=False)
                if right == left + 1
            ]
            maximum_missing = 0
            missing = 0
            heat = np.zeros((8, 8), dtype=np.int64)
            for position in positions:
                if position is None:
                    missing += 1
                    maximum_missing = max(maximum_missing, missing)
                    continue
                missing = 0
                y, x = position
                heat[min(int(y) // 16, 7), min(int(x) // 16, 7)] += 1
            projections: list[float] = []
            response_rows: list[dict[str, object]] = []
            for index in range(len(frames) - response_lag_frames):
                action = movement_names[int(movement_ids[index])]
                start = positions[index]
                end = positions[index + response_lag_frames]
                if not movement_sent[index] or action == "wait" or start is None or end is None:
                    continue
                vector_y, vector_x = movement_vectors[action]
                norm = math.hypot(vector_y, vector_x)
                delta_y, delta_x = end[0] - start[0], end[1] - start[1]
                projection = (delta_y * vector_y + delta_x * vector_x) / norm
                projections.append(projection)
                response_rows.append(
                    {
                        "frame_index": index,
                        "action": action,
                        "projection_pixels": round(projection, 6),
                    }
                )
            qa_count = int(cast(int, contract["qa_frames_per_supported_session"]))
            if basename == "teacher-session-002" and valid_indices:
                qa_indices = [
                    valid_indices[round(index * (len(valid_indices) - 1) / (qa_count - 1))]
                    for index in range(qa_count)
                ]
            else:
                qa_indices = [
                    round(index * (len(frames) - 1) / (qa_count - 1)) for index in range(qa_count)
                ]
            contact_name = f"{basename}-contact.png"
            _write_localization_contact_sheet(
                staging / contact_name,
                frames,
                qa_indices,
                candidates,
                positions,
                exclusion,
            )
            contact_sheets.append(
                {
                    "session": basename,
                    "path": contact_name,
                    "sha256": _file_sha256(staging / contact_name),
                    "frame_indices": qa_indices,
                }
            )
            positive_fraction = (
                sum(value > 0.0 for value in projections) / len(projections) if projections else 0.0
            )
            median_projection = float(np.median(projections)) if projections else 0.0
            session_results.append(
                {
                    "session": basename,
                    "frames": len(frames),
                    "raw_candidate_coverage": raw_detected / len(frames),
                    "filtered_candidate_coverage": filtered_detected / len(frames),
                    "filtered_single_candidate_fraction": single / filtered_detected
                    if filtered_detected
                    else 0.0,
                    "rejected_ui_candidate_coverage": rejected_ui / len(frames),
                    "filtered_jump_p95": float(np.percentile(jumps, 95)) if jumps else None,
                    "maximum_filtered_missing_streak": maximum_missing,
                    "candidate_heatmap_8x8": heat.tolist(),
                    "response_events": len(projections),
                    "positive_projection_fraction": positive_fraction,
                    "median_projection_pixels": median_projection,
                    "response_rows": response_rows,
                    "developer_visual_qa": visual_qa[basename],
                }
            )
        by_session = {str(row["session"]): row for row in session_results}
        supported = by_session["teacher-session-002"]
        unsupported = [by_session[name] for name in ("teacher-session-003", "teacher-session-005")]
        checks = {
            "session002_filtered_coverage": float(
                cast(float, supported["filtered_candidate_coverage"])
            )
            >= float(cast(float, gates["minimum_session002_filtered_coverage"])),
            "session002_response_events": int(cast(int, supported["response_events"]))
            >= int(cast(int, gates["minimum_response_events"])),
            "session002_response_alignment": float(
                cast(float, supported["positive_projection_fraction"])
            )
            >= float(cast(float, gates["minimum_positive_projection_fraction"])),
            "session002_response_distance": float(
                cast(float, supported["median_projection_pixels"])
            )
            >= float(cast(float, gates["minimum_median_projection_pixels"])),
            "session002_visual_qa": supported["developer_visual_qa"]
            == "consistent_with_controlled_player",
            "session003_005_unsupported": all(
                float(cast(float, row["filtered_candidate_coverage"]))
                <= float(cast(float, gates["maximum_unsupported_session_coverage"]))
                for row in unsupported
            ),
            "fixed_ui_confusion_reproduced": all(
                float(cast(float, row["rejected_ui_candidate_coverage"]))
                >= float(cast(float, gates["minimum_fixed_ui_coverage"]))
                for row in unsupported
            ),
        }
        passed = all(checks.values())
        audit: dict[str, object] = {
            "schema_version": "movement-real-player-localization-audit-report-v2",
            "status": "PLAYER_CUE_PARTIAL_SESSION002_ONLY" if passed else "PLAYER_CUE_V2_FAILED",
            "contract_sha256": contract["contract_sha256"],
            "prior_report_sha256": prior["report_sha256"],
            "excluded_ui_xyxy": list(exclusion),
            "sessions": session_results,
            "checks": checks,
            "contact_sheets": contact_sheets,
            "opened_shards": opened_shards,
            "old_player_cue_report_promoting": False,
            "old_continuity_report_promoting": False,
            "old_overfit32_report_promoting": False,
            "semantic_identity_scope": "session002_partial_action_response_supported"
            if passed
            else "unverified",
            "general_three_session_localization_verified": False,
            "human_training_labels_created": False,
            "training_called": False,
            "test_frames_read": 0,
            "r2_allowed": False,
            "device_input_commands_sent": 0,
        }
        audit["report_sha256"] = _object_sha256(audit)
        (staging / "report.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return audit


def _mask_player_patch(
    frame: np.ndarray, player_yx: tuple[float, float], radius: int
) -> np.ndarray:
    center_y, center_x = map(round, player_yx)
    y0, y1 = max(0, center_y - radius), min(128, center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(128, center_x + radius + 1)
    outer = radius + 4
    outer_y0, outer_y1 = max(0, center_y - outer), min(128, center_y + outer + 1)
    outer_x0, outer_x1 = max(0, center_x - outer), min(128, center_x + outer + 1)
    surround = frame[outer_y0:outer_y1, outer_x0:outer_x1].copy()
    inner_y0, inner_y1 = y0 - outer_y0, y1 - outer_y0
    inner_x0, inner_x1 = x0 - outer_x0, x1 - outer_x0
    keep = np.ones(surround.shape[:2], dtype=bool)
    keep[inner_y0:inner_y1, inner_x0:inner_x1] = False
    fill = np.median(surround[keep], axis=0).astype(np.uint8)
    masked = frame.copy()
    masked[y0:y1, x0:x1] = fill
    return masked


def prepare_real_counterfactual_grouped_data(
    contract: dict[str, object], session_root: Path
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, object]]:
    actions = ("STOP", "N", "S", "W", "E", "NW", "NE", "SW", "SE")
    if contract.get("action_order") != list(actions):
        raise ValueError("real grouped action order differs")
    declaration = cast(dict[str, object], contract["source_session"])
    basename = str(declaration["basename"])
    directory = session_root / basename
    summary_path = directory / "summary.json"
    if (
        basename != "teacher-session-002"
        or _file_sha256(summary_path) != declaration["summary_sha256"]
    ):
        raise ValueError("real grouped source summary differs")
    summary = _load_bound_json(summary_path, "summary_sha256")
    if (
        summary.get("status") != "PASSED"
        or summary.get("derived_roi_rgb_persisted") is not True
        or summary.get("raw_frames_persisted") is not False
    ):
        raise ValueError("real grouped source summary content differs")
    frames_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    opened_shards: list[dict[str, object]] = []
    for raw_row in cast(list[dict[str, object]], summary["observation_shards"]):
        name = str(raw_row["path"])
        path = directory / "shards" / name
        if Path(name).name != name or path.is_symlink() or _file_sha256(path) != raw_row["sha256"]:
            raise ValueError("real grouped source shard differs")
        with np.load(path, allow_pickle=False) as shard:
            frames_parts.append(shard["minimap_rgb"].copy())
            timestamp_parts.append(shard["scheduled_elapsed_ms"].copy())
        opened_shards.append({"basename": name, "sha256": raw_row["sha256"]})
    frames = np.concatenate(frames_parts)
    timestamps = np.concatenate(timestamp_parts)
    sequence_frames = int(cast(int, contract["sequence_frames"]))
    frame_period_ms = int(cast(int, contract["frame_period_ms"]))
    if (
        frames.dtype != np.uint8
        or frames.shape[1:] != (128, 128, 3)
        or timestamps.shape != (len(frames),)
        or not np.all(np.diff(timestamps) == frame_period_ms)
    ):
        raise ValueError("real grouped source arrays differ")
    cue_contract: dict[str, object] = {
        "color": contract["color"],
        "components": contract["components"],
    }
    exclusion = cast(
        tuple[int, int, int, int],
        tuple(map(int, cast(list[int], contract["excluded_ui_xyxy"]))),
    )
    _candidates, positions = _filtered_player_positions(frames, cue_contract, exclusion)
    goal_distance = int(cast(int, contract["goal_distance_pixels"]))
    stop_radius = float(cast(float, contract["stop_radius_pixels"]))
    marker = cast(dict[str, object], contract["marker"])
    marker_radius = int(cast(int, marker["radius"]))
    group_count = int(cast(int, contract["source_group_count"]))
    used: set[int] = set()
    source_windows: list[tuple[int, int, np.ndarray, list[tuple[float, float]]]] = []
    for end in range(sequence_frames - 1, len(frames)):
        start = end - sequence_frames + 1
        if any(index in used for index in range(start, end + 1)):
            continue
        window_positions_raw = positions[start : end + 1]
        if any(position is None for position in window_positions_raw):
            continue
        window_positions = [cast(tuple[float, float], item) for item in window_positions_raw]
        if not np.all(np.diff(timestamps[start : end + 1]) == frame_period_ms):
            continue
        player = window_positions[-1]
        goals = [
            _counterfactual_goal(player, action, goal_distance, marker_radius) for action in actions
        ]
        if any(goal is None for goal in goals) or any(
            _goal_direction(player, cast(tuple[int, int], goal), stop_radius) != action
            for action, goal in zip(actions, goals, strict=True)
        ):
            continue
        source_windows.append((start, end, frames[start : end + 1], window_positions))
        used.update(range(start, end + 1))
        if len(source_windows) == group_count:
            break
    if len(source_windows) != group_count:
        raise ValueError("real grouped source-window support differs")

    mask_radius = int(cast(int, contract["player_mask_radius_pixels"]))
    neutral = np.asarray(cast(list[int], contract["goal_only_rgb"]), dtype=np.uint8)
    variant_rows: dict[str, list[np.ndarray]] = {
        "full": [],
        "player_masked": [],
        "goal_only": [],
    }
    labels: list[int] = []
    groups: list[int] = []
    sample_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for group, (start, end, source_clip, player_positions) in enumerate(source_windows):
        group_rows.append(
            {
                "group": group,
                "start_index": start,
                "end_index": end,
                "end_timestamp_ms": int(timestamps[end]),
                "source_rgb_sha256": hashlib.sha256(source_clip.tobytes()).hexdigest(),
            }
        )
        player = player_positions[-1]
        masked_source = np.stack(
            [
                _mask_player_patch(frame, position, mask_radius)
                for frame, position in zip(source_clip, player_positions, strict=True)
            ]
        )
        goal_only_source = np.empty_like(source_clip)
        goal_only_source[:] = neutral
        for label, action in enumerate(actions):
            goal_yx = _counterfactual_goal(player, action, goal_distance, marker_radius)
            if goal_yx is None:
                raise ValueError("real grouped goal geometry differs")
            goal_y, goal_x = goal_yx
            goal_xy = [goal_x / 127.0, goal_y / 127.0]
            generated = {
                "full": np.stack([_mark_goal(frame, goal_xy, marker) for frame in source_clip]),
                "player_masked": np.stack(
                    [_mark_goal(frame, goal_xy, marker) for frame in masked_source]
                ),
                "goal_only": np.stack(
                    [_mark_goal(frame, goal_xy, marker) for frame in goal_only_source]
                ),
            }
            hashes: dict[str, str] = {}
            for variant, clip in generated.items():
                variant_rows[variant].append(clip)
                hashes[variant] = hashlib.sha256(clip.tobytes()).hexdigest()
            labels.append(label)
            groups.append(group)
            sample_rows.append(
                {"group": group, "action": action, "goal_xy": [goal_x, goal_y], "hashes": hashes}
            )
    variants = {name: np.stack(rows).astype(np.uint8) for name, rows in variant_rows.items()}
    labels_array = np.asarray(labels, dtype=np.int64)
    groups_array = np.asarray(groups, dtype=np.int64)
    duplicates = {
        name: len(clips) - len({hashlib.sha256(clip.tobytes()).hexdigest() for clip in clips})
        for name, clips in variants.items()
    }
    metadata: dict[str, object] = {
        "source_session": basename,
        "source_windows": group_rows,
        "samples": sample_rows,
        "duplicates_by_variant": duplicates,
        "variant_sha256": {
            name: hashlib.sha256(clips.tobytes()).hexdigest() for name, clips in variants.items()
        },
        "opened_shards": opened_shards,
    }
    return variants, labels_array, groups_array, metadata


def _appearance_peaks(frame: np.ndarray, template: np.ndarray) -> list[tuple[float, float, float]]:
    """RGB-only local template matching; score is correlation, not probability."""
    rgb = frame.astype(np.int16)
    red, green, blue = (rgb[..., index] for index in range(3))
    components = _mask_components((green > 85) & (green - red > 18) & (green - blue > 10))
    radius = int(PLAYER_TRACKING_SETTINGS["patch_radius"])
    search = int(PLAYER_TRACKING_SETTINGS["center_search_radius"])
    centers: set[tuple[int, int]] = set()
    for size, y, x, height, width in components:
        if not (20 <= size <= 140 and 7 <= height <= 24 and 7 <= width <= 24):
            continue
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                cy, cx = round(y) + dy, round(x) + dx
                if (
                    radius <= cy < 128 - radius
                    and radius <= cx < 128 - radius
                    and not (112 <= cx < 128 and 0 <= cy < 16)
                ):
                    centers.add((cy, cx))
    if not centers:
        return []
    ordered = sorted(centers)
    patches = (
        np.stack(
            [frame[y - radius : y + radius + 1, x - radius : x + radius + 1] for y, x in ordered]
        )
        .astype(np.float32)
        .reshape(len(ordered), -1)
    )
    reference = template.astype(np.float32).ravel()
    reference -= reference.mean()
    patches -= patches.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(patches, axis=1) * np.linalg.norm(reference)
    scores = np.sum(patches * reference, axis=1) / np.maximum(denominator, 1e-8)
    peaks: list[tuple[float, float, float]] = []
    for index in np.argsort(-scores, kind="stable"):
        y, x = ordered[index]
        if all(math.dist((y, x), peak[:2]) > 10 for peak in peaks):
            peaks.append((float(y), float(x), float(scores[index])))
    return peaks


def _track_appearance(
    frames: np.ndarray, template: np.ndarray
) -> tuple[list[tuple[float, float] | None], list[str], list[float]]:
    previous: tuple[float, float] | None = None
    confirmations = 0
    positions: list[tuple[float, float] | None] = []
    reasons: list[str] = []
    scores: list[float] = []
    for frame in frames:
        peaks = _appearance_peaks(frame, template)
        scores.append(peaks[0][2] if peaks else 0.0)
        reason = "no_candidate"
        selected: tuple[float, float] | None = None
        if peaks:
            y, x, score = peaks[0]
            if score < PLAYER_TRACKING_SETTINGS["minimum_correlation"]:
                reason = "appearance_mismatch"
            elif (
                len(peaks) > 1
                and score - peaks[1][2] < PLAYER_TRACKING_SETTINGS["minimum_distinct_peak_margin"]
            ):
                reason = "ambiguous"
            else:
                candidate = (y, x)
                if (
                    previous is None
                    or math.dist(previous, candidate)
                    > PLAYER_TRACKING_SETTINGS["maximum_step_pixels"]
                ):
                    confirmations = 1
                else:
                    confirmations += 1
                previous = candidate
                reason = "confirming"
                if confirmations >= PLAYER_TRACKING_SETTINGS["confirmation_frames"]:
                    selected = candidate
                    reason = "tracked"
        if reason not in {"tracked", "confirming"}:
            previous = None
            confirmations = 0
        positions.append(selected)
        reasons.append(reason)
    return positions, reasons, scores


def _write_tracking_qa(
    path: Path,
    main: np.ndarray,
    frames: np.ndarray,
    positions: list[tuple[float, float] | None],
    indices: list[int],
) -> None:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (768, math.ceil(len(indices) / 3) * 148))
    for ordinal, index in enumerate(indices):
        tile = Image.new("RGB", (256, 148))
        tile.paste(Image.fromarray(main[index]), (0, 20))
        tile.paste(Image.fromarray(frames[index]), (128, 20))
        draw = ImageDraw.Draw(tile)
        draw.text((0, 2), f"{index}: {'tracked' if positions[index] else 'unknown'}")
        if positions[index] is not None:
            y, x = cast(tuple[float, float], positions[index])
            draw.ellipse((128 + x - 9, 20 + y - 9, 128 + x + 9, 20 + y + 9), outline="magenta")
        canvas.paste(tile, (ordinal % 3 * 256, ordinal // 3 * 148))
    canvas.save(path, format="PNG")


def run_real_player_tracking_audit(
    contract_path: Path,
    prior_report_path: Path,
    session_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """One offline appearance candidate. Never promotes or supplies action labels."""
    contract = _load_bound_json(contract_path, "contract_sha256")
    prior = _load_bound_json(prior_report_path, "report_sha256")
    if (
        contract.get("schema_version") != "movement-real-player-localization-audit-v2"
        or prior.get("contract_sha256") != contract["contract_sha256"]
        or prior.get("status") != "PLAYER_CUE_PARTIAL_SESSION002_ONLY"
    ):
        raise ValueError("tracking requires the completed localization v2 source binding")
    if output_dir.exists():
        raise ValueError("tracking output already exists")
    sessions: list[dict[str, object]] = []
    template: np.ndarray | None = None
    template_indices: list[int] = []
    opened: list[dict[str, object]] = []
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    for declaration in cast(list[dict[str, object]], contract["sessions"]):
        name = str(declaration["basename"])
        if name not in {"teacher-session-002", "teacher-session-003", "teacher-session-005"}:
            raise ValueError("tracking session outside existing offline scope")
        directory = session_root / name
        summary_path = directory / "summary.json"
        if _file_sha256(summary_path) != declaration["summary_sha256"]:
            raise ValueError("tracking summary binding differs")
        summary = _load_bound_json(summary_path, "summary_sha256")
        fields = ("main_rgb", "minimap_rgb", "movement_id", "movement_input_sent")
        parts: dict[str, list[np.ndarray]] = {key: [] for key in fields}
        for shard_row in cast(list[dict[str, object]], summary["observation_shards"]):
            basename = str(shard_row["path"])
            shard_path = directory / "shards" / basename
            if (
                Path(basename).name != basename
                or shard_path.is_symlink()
                or _file_sha256(shard_path) != shard_row["sha256"]
            ):
                raise ValueError("tracking shard binding differs")
            with np.load(shard_path, allow_pickle=False) as shard:
                for key in fields:
                    parts[key].append(shard[key].copy())
            opened.append({"session": name, "basename": basename, "sha256": shard_row["sha256"]})
        arrays = {key: np.concatenate(value) for key, value in parts.items()}
        frames = arrays["minimap_rgb"]
        if template is None:
            if name != "teacher-session-002":
                raise ValueError("appearance template must originate in session002")
            _, old = _filtered_player_positions(frames, contract, (112, 0, 128, 16))
            radius = int(PLAYER_TRACKING_SETTINGS["patch_radius"])
            template_indices = [
                index
                for index, position in enumerate(old)
                if position is not None and all(16 <= value < 112 for value in position)
            ][: int(PLAYER_TRACKING_SETTINGS["template_frames"])]
            if len(template_indices) != PLAYER_TRACKING_SETTINGS["template_frames"]:
                raise ValueError("insufficient source template evidence")
            crops = []
            for index in template_indices:
                y, x = map(round, cast(tuple[float, float], old[index]))
                crops.append(
                    frames[index, y - radius : y + radius + 1, x - radius : x + radius + 1]
                )
            template = np.median(np.stack(crops), axis=0).astype(np.uint8)
            from PIL import Image

            Image.fromarray(template).save(staging / "appearance-template.png")
        positions, reasons, scores = _track_appearance(frames, template)
        valid = [i for i, p in enumerate(positions) if p is not None]
        jumps = [
            math.dist(
                cast(tuple[float, float], positions[i - 1]),
                cast(tuple[float, float], positions[i]),
            )
            for i in valid
            if i and positions[i - 1] is not None
        ]
        spans: list[dict[str, int]] = []
        start: int | None = None
        for i in range(len(positions) + 1):
            if i < len(positions) and positions[i] is None:
                if start is None:
                    start = i
            elif start is not None:
                spans.append({"start": start, "frames": i - start})
                start = None
        # Historical compass order, not the Movement head order. Actions never enter tracking.
        vectors = ((0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
        responses: list[float] = []
        for i in range(len(frames) - 5):
            action = int(arrays["movement_id"][i])
            if not arrays["movement_input_sent"][i] or action == 0:
                continue
            if any(p is None for p in positions[i : i + 6]):
                continue
            if any(
                arrays["movement_input_sent"][j] and arrays["movement_id"][j] != action
                for j in range(i + 1, i + 6)
            ):
                continue
            delta = np.asarray(positions[i + 5]) - np.asarray(positions[i])
            vector = np.asarray(vectors[action])
            responses.append(float(delta @ vector / np.linalg.norm(vector)))
        indices = list(map(int, np.linspace(0, len(frames) - 1, 12)))
        if valid:
            indices.extend(valid[int(i)] for i in np.linspace(0, len(valid) - 1, 6))
        indices = sorted(set(indices))
        qa_name = f"{name}-paired-qa.png"
        _write_tracking_qa(staging / qa_name, arrays["main_rgb"], frames, positions, indices)
        sessions.append(
            {
                "session": name,
                "frames": len(frames),
                "tracked_frames": len(valid),
                "coverage": len(valid) / len(frames),
                "unknown_reasons": dict(Counter(reasons)),
                "unknown_spans": spans,
                "maximum_unknown_frames": max((s["frames"] for s in spans), default=0),
                "adjacent_jump_p95": float(np.percentile(jumps, 95)) if jumps else None,
                "acquisitions": sum(i == 0 or positions[i - 1] is None for i in valid),
                "response_events": len(responses),
                "positive_response_fraction": sum(p > 0 for p in responses) / len(responses)
                if responses
                else None,
                "median_projection_pixels": float(np.median(responses)) if responses else None,
                "positions_yx": positions,
                "correlation_scores": scores,
                "qa": {
                    "basename": qa_name,
                    "indices": indices,
                    "sha256": _file_sha256(staging / qa_name),
                },
            }
        )
    report: dict[str, object] = {
        "schema_version": "movement-real-player-appearance-tracking-v1",
        "status": "PLAYER_APPEARANCE_TRACKING_DIAGNOSTIC_ONLY",
        "source_contract_sha256": contract["contract_sha256"],
        "prior_report_sha256": prior["report_sha256"],
        "implementation_sha256": _file_sha256(Path(__file__)),
        "settings": PLAYER_TRACKING_SETTINGS,
        "template_session": "teacher-session-002",
        "template_frame_indices": template_indices,
        "template_sha256": _file_sha256(staging / "appearance-template.png"),
        "sessions": sessions,
        "opened_shards": opened,
        "response_note": (
            "scheduled 1000ms endpoints, all six positions observed, "
            "no intervening different dispatched action; not ground truth"
        ),
        "semantic_accuracy_verified": False,
        "false_lock_rate": None,
        "reacquisition_latency_verified": False,
        "training_allowed": False,
        "r2_allowed": False,
        "test_opened": False,
        "device_input_allowed": False,
        "input_commands_sent": 0,
        "model_runs": 0,
        "gpu_seconds": 0,
        "next_step": "inspect paired QA; no automatic navigation integration or training",
    }
    report["report_sha256"] = _object_sha256(report)
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return report


def _native_landscape_window(path: Path) -> dict[str, np.ndarray]:
    """A 3-second development window; crop before resizing, never decode audio."""
    import av

    from hok_agent import pre_ingest

    descriptor, opened = pre_ingest._open_regular(path)
    minimaps: list[np.ndarray] = []
    main_views: list[np.ndarray] = []
    times: list[int] = []
    source_hashes: list[str] = []
    with os.fdopen(descriptor, "rb") as handle, av.open(handle, mode="r") as container:
        stream = container.streams.video[0]
        if stream.duration is None or stream.time_base is None:
            raise ValueError("native window requires video duration and time base")
        if stream.width <= stream.height or pre_ingest._rotation(stream) != 0:
            raise ValueError("native pilot accepts the two landscape sources only")
        start = int(stream.duration * 0.2)
        start_us = round(start * stream.time_base * 1_000_000)
        container.seek(start, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("native frame has no timestamp")
            timestamp_us = round(frame.pts * stream.time_base * 1_000_000)
            if timestamp_us < start_us + len(times) * 200_000:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            # Common landscape inspection geometry, including margins beyond the map border.
            map_crop = rgb[: round(height * 0.4), round(width * 0.025) : round(width * 0.215)]
            main_crop = rgb[
                round(height * 0.15) : round(height * 0.85), round(width * 0.3) : round(width * 0.7)
            ]
            for crop, destination in ((map_crop, minimaps), (main_crop, main_views)):
                yy = np.linspace(0, crop.shape[0] - 1, 256).astype(np.int64)
                xx = np.linspace(0, crop.shape[1] - 1, 256).astype(np.int64)
                destination.append(crop[yy[:, None], xx[None, :]].copy())
            times.append(timestamp_us)
            source_hashes.append(hashlib.sha256(map_crop.tobytes()).hexdigest())
            if len(times) == 16:
                break
        pre_ingest._assert_unchanged(handle.fileno(), opened)
    if len(times) != 16 or not np.all(np.diff(times) > 0):
        raise ValueError("native window incomplete or non-monotonic")
    return {
        "minimap_rgb": np.stack(minimaps),
        "main_rgb": np.stack(main_views),
        "timestamp_us": np.asarray(times, dtype=np.int64),
        "native_crop_sha256": np.asarray(source_hashes, dtype="U64"),
    }


def run_native_player_pilot(
    source_root: Path,
    cohort_dir: Path,
    pre_ingest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Materialize only the two identity-bound landscape development windows."""
    from PIL import Image, ImageDraw

    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    if output_dir.exists():
        raise ValueError("native pilot output already exists")
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    if any(cohort.session_splits.get(key) != split for key, split in NATIVE_PLAYER_SOURCES.items()):
        raise ValueError("native player source split binding differs")
    sources: dict[str, Path] = {}
    for path in pre_ingest._scan(source_root):
        identity = pre_ingest._sha(
            pre_ingest._canonical(
                [
                    "candidate-v2-file-atomic",
                    path.relative_to(source_root).as_posix(),
                    pre_ingest._stat_signature(path.stat()),
                ]
            )
        )
        if identity in NATIVE_PLAYER_SOURCES:
            sources[identity] = path
    if set(sources) != set(NATIVE_PLAYER_SOURCES):
        raise ValueError("native player source identities unavailable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    rows: list[dict[str, object]] = []
    for identity, path in sorted(sources.items()):
        if pre_ingest._candidate(path, source_root).candidate_id != identity:
            raise ValueError("native source identity changed")
        arrays = _native_landscape_window(path)
        positions = green_ring_track(arrays["minimap_rgb"])
        basename = identity[:8] + "-native-window.npz"
        np.savez_compressed(
            staging / basename,
            minimap_rgb=arrays["minimap_rgb"],
            main_rgb=arrays["main_rgb"],
            timestamp_us=arrays["timestamp_us"],
            native_crop_sha256=arrays["native_crop_sha256"],
        )
        sheet = Image.new("RGB", (1024, 4 * 276))
        for index in range(16):
            x, y = index % 4 * 256, index // 4 * 276
            sheet.paste(Image.fromarray(arrays["minimap_rgb"][index]), (x, y + 20))
            ImageDraw.Draw(sheet).text((x + 2, y + 2), str(index))
            if positions[index] is not None:
                py, px = cast(tuple[int, int], positions[index])
                ImageDraw.Draw(sheet).ellipse(
                    (x + px - 16, y + py + 4, x + px + 16, y + py + 36), outline="magenta", width=2
                )
        qa_name = identity[:8] + "-minimap-qa.png"
        sheet.save(staging / qa_name)
        main_name = identity[:8] + "-main-qa.png"
        for index in range(16):
            x, y = index % 4 * 256, index // 4 * 276
            sheet.paste(Image.fromarray(arrays["main_rgb"][index]), (x, y + 20))
        sheet.save(staging / main_name)
        rows.append(
            {
                "session_hash": identity,
                "split": NATIVE_PLAYER_SOURCES[identity],
                "frames": 16,
                "window_start_us": int(arrays["timestamp_us"][0]),
                "window_end_us": int(arrays["timestamp_us"][-1]),
                "maximum_sample_gap_us": int(np.diff(arrays["timestamp_us"]).max()),
                "green_ring_candidate_counts": [
                    len(green_ring_candidates(f)) for f in arrays["minimap_rgb"]
                ],
                "confirmed_green_ring_yx": positions,
                "confirmed_frames": sum(p is not None for p in positions),
                "artifacts": [
                    {"basename": name, "sha256": _file_sha256(staging / name)}
                    for name in (basename, qa_name, main_name)
                ],
            }
        )
    report: dict[str, object] = {
        "status": "NATIVE_LANDSCAPE_WINDOWS_MATERIALIZED_QA_ONLY",
        "cohort_sha256": cohort.cohort_sha256,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "sessions": rows,
        "start_fraction": 0.2,
        "sampling_period_ms": 200,
        "minimap_source_roi_xyxy_fraction": [0.025, 0.0, 0.215, 0.4],
        "main_source_roi_xyxy_fraction": [0.3, 0.15, 0.7, 0.85],
        "derived_rgb_size": 256,
        "crop_before_resize": True,
        "controlled_player_identity_verified": False,
        "cue_scope": "green ring with local continuity; not calibrated player identity",
        "training_allowed": False,
        "action_labels_created": False,
        "test_frames_decoded": 0,
        "raw_full_frames_persisted": False,
        "raw_source_locators_persisted": False,
        "model_runs": 0,
        "gpu_seconds": 0,
        "input_commands_sent": 0,
    }
    report["report_sha256"] = _object_sha256(report)
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return report
