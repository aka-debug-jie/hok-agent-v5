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

FLOW_SETTINGS = {
    "window": 15, "pyramid_level": 2, "max_corners": 20,
    "corner_quality": 0.01, "corner_min_distance": 2, "patch_radius": 12,
    "minimum_points": 4, "fb_error_pixels": 1.0, "step_pixels": 8.0,
    "rejoin_pixels": 3.0, "maximum_gap_ms": 1000,
}


def _flow_step(
    previous: np.ndarray, current: np.ndarray, points: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    import cv2

    forward, valid, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, points.copy(), winSize=(15, 15), maxLevel=2
    )
    if forward is None:
        return None, None, "forward_failed"
    backward, reverse_valid, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward, forward.copy(), winSize=(15, 15), maxLevel=2
    )
    if backward is None:
        return None, None, "backward_failed"
    delta = (forward - points).reshape(-1, 2)
    keep = (
        (valid.ravel() == 1) & (reverse_valid.ravel() == 1)
        & np.isfinite(delta).all(axis=1)
        & (np.linalg.norm((backward - points).reshape(-1, 2), axis=1) <= 1)
        & (np.linalg.norm(delta, axis=1) <= 8)
    )
    if int(keep.sum()) < 4:
        return None, None, "insufficient_consistent_points"
    return forward[keep], np.median(delta[keep], axis=0)[::-1], "ok"


def bridge_player_gaps(
    frames: np.ndarray, positions: list[tuple[float, float] | None], period_ms: int,
) -> tuple[list[tuple[float, float] | None], list[dict[str, object]]]:
    """Retrospective RGB-only interpolation; actions cannot enter this interface."""
    import cv2

    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    result = list(positions)
    gaps: list[dict[str, object]] = []
    for start in range(len(positions) - 1):
        origin = positions[start]
        if origin is None or positions[start + 1] is not None:
            continue
        end = start + 1
        while end < len(positions) and positions[end] is None:
            end += 1
        reason = "unclosed_gap" if end == len(positions) else "gap_too_long"
        accepted = False
        if end < len(positions) and (end - start) * period_ms <= 1000:
            y, x = origin
            mask = np.zeros_like(gray[start])
            cv2.circle(mask, (round(x), round(y)), 12, 255, -1)
            mask[:16] = mask[-16:] = 0
            mask[:, :16] = mask[:, -16:] = 0
            points = cv2.goodFeaturesToTrack(
                gray[start], maxCorners=20, qualityLevel=0.01, minDistance=2, mask=mask
            )
            proposed: list[tuple[float, float]] = []
            center = np.asarray(origin, dtype=np.float64)
            reason = "insufficient_initial_points"
            if points is not None and len(points) >= 4:
                for index in range(start + 1, end + 1):
                    points, delta, reason = _flow_step(gray[index - 1], gray[index], points)
                    if points is None or delta is None:
                        break
                    center += delta
                    if not all(16 <= value < 112 for value in center):
                        reason = "outside_interior"
                        break
                    proposed.append((float(center[0]), float(center[1])))
                if len(proposed) == end - start:
                    endpoint = cast(tuple[float, float], positions[end])
                    accepted = math.dist(proposed[-1], endpoint) <= 3
                    reason = "accepted" if accepted else "rejoin_mismatch"
                    if accepted:
                        result[start + 1 : end] = proposed[:-1]
        gaps.append({"start": start, "end": end, "accepted": accepted, "reason": reason})
    return result, gaps


def _flow_track_stats(positions: list[tuple[float, float] | None]) -> dict[str, int]:
    lengths: list[int] = []
    run = 0
    for point in [*positions, None]:
        if point is None:
            lengths.append(run)
            run = 0
        else:
            run += 1
    return {
        "valid_frames": sum(lengths), "longest_run": max(lengths),
        "continuous_frames": sum(n for n in lengths if n >= 2),
        "nonoverlapping_windows16": sum(n // 16 for n in lengths),
    }


def _flow_response(
    positions: list[tuple[float, float] | None], actions: np.ndarray, sent: np.ndarray,
    lag: int,
) -> dict[str, object]:
    vectors = ((0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1))
    projections: list[float] = []
    for i in range(len(positions) - lag):
        if not sent[i] or actions[i] == 0 or any(p is None for p in positions[i:i + lag + 1]):
            continue
        if any(sent[j] and actions[j] != actions[i] for j in range(i + 1, i + lag + 1)):
            continue
        vector = np.asarray(vectors[int(actions[i])])
        delta = np.asarray(positions[i + lag]) - np.asarray(positions[i])
        projections.append(float(delta @ vector / np.linalg.norm(vector)))
    total = int(np.count_nonzero(sent))
    fraction = sum(p > 0 for p in projections) / len(projections) if projections else 0.0
    median = float(np.median(projections)) if projections else None
    return {
        "all_sent_events": total, "valid_events": len(projections),
        "valid_fraction_of_all_sent": len(projections) / total if total else 0.0,
        "positive_fraction": fraction, "median_projection_pixels": median,
        "passed": len(projections) >= 10 and fraction >= 0.75
        and median is not None and median >= 1,
    }


def run_player_flow_audit(
    contract_path: Path, prior_report_path: Path, session_root: Path, output_dir: Path,
) -> dict[str, object]:
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    contract = _load_bound_json(contract_path, "contract_sha256")
    prior = _load_bound_json(prior_report_path, "report_sha256")
    if (prior.get("contract_sha256") != contract["contract_sha256"]
            or prior.get("status") != "PLAYER_CUE_PARTIAL_SESSION002_ONLY"):
        raise ValueError("flow requires frozen localization v2 evidence")
    if output_dir.exists():
        raise ValueError("flow output already exists")
    period = int(cast(int, contract["frame_period_ms"]))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".player-flow-", dir=output_dir.parent))
    sessions: list[dict[str, object]] = []
    for declaration in cast(list[dict[str, object]], contract["sessions"]):
        name = str(declaration["basename"])
        directory = session_root / name
        if _file_sha256(directory / "summary.json") != declaration["summary_sha256"]:
            raise ValueError("flow source summary hash differs")
        summary = _load_bound_json(directory / "summary.json", "summary_sha256")
        fields = ("minimap_rgb", "scheduled_elapsed_ms", "movement_id", "movement_input_sent")
        parts: dict[str, list[np.ndarray]] = {key: [] for key in fields}
        for row in cast(list[dict[str, object]], summary["observation_shards"]):
            basename = str(row["path"])
            path = directory / "shards" / basename
            if Path(basename).name != basename or path.is_symlink():
                raise ValueError("invalid flow shard path")
            if _file_sha256(path) != row["sha256"]:
                raise ValueError("flow shard hash differs")
            with np.load(path, allow_pickle=False) as shard:
                for key in fields:
                    parts[key].append(shard[key].copy())
        arrays = {key: np.concatenate(value) for key, value in parts.items()}
        if not np.all(np.diff(arrays["scheduled_elapsed_ms"]) == period):
            raise ValueError("flow source sampling differs")
        frames = arrays["minimap_rgb"]
        _, detected = _filtered_player_positions(frames, contract, (112, 0, 128, 16))
        direct = [p if p is not None and all(16 <= v < 112 for v in p) else None
                  for p in detected]
        bridged, gaps = bridge_player_gaps(frames, direct, period)
        before, after = _flow_track_stats(direct), _flow_track_stats(bridged)
        accepted = [g for g in gaps if g["accepted"]]
        qa_indices: list[int] = []
        selected = [accepted[int(i)] for i in np.linspace(0, len(accepted) - 1,
                    min(12, len(accepted)))] if accepted else []
        for gap in selected:
            qa_indices.extend(range(int(cast(int, gap["start"])), int(cast(int, gap["end"])) + 1))
        qa_name = f"{name}-gaps.png"
        if qa_indices:
            _write_tracking_qa(staging / qa_name, frames, frames, bridged, qa_indices)
        gain = ((after["continuous_frames"] - before["continuous_frames"])
                / before["continuous_frames"] if before["continuous_frames"] else 0.0)
        response = _flow_response(bridged, arrays["movement_id"],
                                  arrays["movement_input_sent"], 1000 // period)
        sessions.append({
            "session": name, "frames": len(frames), "before": before, "after": after,
            "original_v2_detected_frames": sum(p is not None for p in detected),
            "positions_yx": bridged,
            "position_kind": ["direct" if a is not None else
                              "retrospective_bridge" if b is not None else "unknown"
                              for a, b in zip(direct, bridged, strict=True)],
            "direct_coverage": before["valid_frames"] / len(frames),
            "bridged_coverage": after["valid_frames"] / len(frames),
            "continuous_frame_gain": gain, "accepted_gaps": len(accepted), "gaps": gaps,
            "failure_counts": dict(Counter(str(g["reason"]) for g in gaps if not g["accepted"])),
            "response": response, "qa_clips": len(selected),
            "qa_file": qa_name if qa_indices else None,
            "qa_sha256": _file_sha256(staging / qa_name) if qa_indices else None,
            "quantitative_passed": gain >= 0.5 and len(accepted) >= 10 and response["passed"],
        })
    eligible = sum(cast(dict[str, int], s["after"])["nonoverlapping_windows16"] >= 20
                   for s in sessions)
    report: dict[str, object] = {
        "schema_version": "player-flow-gap-audit-v1", "settings": FLOW_SETTINGS,
        "opencv_version": cv2.__version__, "contract_sha256": contract["contract_sha256"],
        "prior_report_file_sha256": _file_sha256(prior_report_path),
        "source_sha256": _file_sha256(Path(__file__)), "sessions": sessions,
        "status": "QUANTITATIVE_FAILED" if not any(s["quantitative_passed"] for s in sessions)
        else "QA_REQUIRED", "sessions_with_20_windows": eligible,
        "cross_session_planning_allowed": False, "qa_status": "PENDING",
        "training_allowed": False, "real_time_localization": False,
        "gpu_seconds": 0, "input_commands_sent": 0, "test_frames_opened": 0,
        "window_policy": "nonoverlapping within contiguous runs; retrospective labels only",
        "timing_basis": "scheduled_elapsed_ms; not measured action or capture latency",
        "gap_policy": "rejoin deadline measured from last direct detection",
    }
    report["report_sha256"] = _object_sha256(report)
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    os.rename(staging, output_dir)
    return report

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

NATIVE_ANCHOR_AUDIT_SOURCES = {
    "0667d97cdb3024f7c5d39d6e797bfe1d527cbd99c07e8e2d2e31cd6eb0ed0993": "train",
    "06c5a8e67a7a19a59e4b13bdae9dafeaa2b07a0168e41406d58e5209ed0e2b4a": "train",
    "08565f01b75fb400394b35d33cd2bab8a94298f082e127115edc27f0b073b46b": "train",
    "0a84c341d16222bb3424e95cbb5d51797eecbea8b528417cd02d569ae9a360cc": "train",
    "0c60062fdde6ed50e0bf2fe231fda206535b90911d2f74e6f0f171a54e486c6c": "train",
    "0e34a785656d464bd946559e9a7ac9602ef2ffcd0e7a9c6d8ba265f37261de24": "train",
    "109a2a343dd37d406f987cc22a4c4c876e59a4e93e9f4127bf5ba06107bd91d7": "train",
    "12214351b55ac24beffe2c52b77464e009120a3a3cc69fc7cb17adcfe1f87e39": "train",
    "c1121610049b451fb1e3f8d3c3695da15ff04f5391898d7b9a0de867b685c009": "dev",
    "c154c394fd9b570544feb68c3b669e4cdb1a3f95146d66a44e6751424936946e": "dev",
    "c5b6e1aeb6d509734afcd5830c2ebe399cf49fa9be35fd7a4ade0ad1a2352f4c": "dev",
    "c84d549a0e1c149f294883b897830229f68bda9f36011be788034aba7e68df88": "dev",
}
NATIVE_ANCHOR_AUDIT_FRACTIONS = (0.1, 0.3, 0.6)
NATIVE_ANCHOR_REPAIR_SOURCES = {
    "cdcad0621f8abec83ccd66c5feed2fb14347b2ad82e2aae46c862b48129b609e": "dev",
    "d13bfda28eb4057a05e62528074b0c540f6c7aada944d67c2d44f1ea3802d72d": "dev",
    "d41ffac336abbb96f5b2c9dc96662f53ebc324c2dcd90449779f7bf032f75cba": "dev",
    "d81cc35a4e461a3052d59e344f79b1cc9adbf8d07c9b9343d18a8f13ab2adab5": "dev",
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


def native_map_point_to_source_xy(
    point_yx: tuple[int, int],
    source_wh: tuple[int, int],
) -> tuple[int, int]:
    """Return the actual source pixel sampled at this map cell, not a world coordinate."""
    y, x = point_yx
    if not (0 <= y < 256 and 0 <= x < 256):
        raise ValueError("native map coordinate outside 256x256")
    width, height = source_wh
    columns = np.linspace(round(width * 0.025), round(width * 0.215) - 1, 256).astype(np.int64)
    rows = np.linspace(0, round(height * 0.4) - 1, 256).astype(np.int64)
    return int(columns[x]), int(rows[y])


def native_coordinate_diagnostic(frames: np.ndarray) -> dict[str, object]:
    """Fixed translation checks on cached RGB; no parameters are selected or changed."""
    baseline = green_ring_track(frames)
    valid = [i for i, p in enumerate(baseline) if p is not None]
    jumps = [
        math.dist(cast(tuple[int, int], baseline[i - 1]), cast(tuple[int, int], baseline[i]))
        for i in valid
        if i and baseline[i - 1] is not None
    ]
    translations: list[dict[str, object]] = []
    for dy, dx in ((-4, 0), (4, 0), (0, -4), (0, 4)):
        shifted = np.roll(frames, (dy, dx), axis=(1, 2))
        if dy > 0:
            shifted[:, :dy] = 0
        elif dy < 0:
            shifted[:, dy:] = 0
        if dx > 0:
            shifted[:, :, :dx] = 0
        elif dx < 0:
            shifted[:, :, dx:] = 0
        predicted = green_ring_track(shifted)
        errors = [
            math.dist((a[0] + dy, a[1] + dx), b)
            for a, b in zip(baseline, predicted, strict=True)
            if a is not None and b is not None
        ]
        translations.append(
            {
                "shift_yx": [dy, dx],
                "compared_frames": len(errors),
                "lost_confirmations": sum(
                    a is not None and b is None for a, b in zip(baseline, predicted, strict=True)
                ),
                "new_confirmations": sum(
                    a is None and b is not None for a, b in zip(baseline, predicted, strict=True)
                ),
                "maximum_equivariance_error_pixels": max(errors) if errors else None,
            }
        )
    return {
        "positions_yx": baseline,
        "confirmed_frames": len(valid),
        "maximum_adjacent_jump_pixels": max(jumps) if jumps else None,
        "median_adjacent_jump_pixels": float(np.median(jumps)) if jumps else None,
        "translations": translations,
        "scope": (
            "geometric consistency only; zero error does not establish identity "
            "or localization accuracy"
        ),
    }


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


def mark_pixel_goal(
    image: np.ndarray,
    goal_yx: tuple[int, int],
    *,
    radius: int = 7,
    thickness: int = 2,
) -> np.ndarray:
    """Draw the fixed yellow hollow goal in pixel coordinates on square RGB."""
    if (
        image.ndim != 3
        or image.shape[0] != image.shape[1]
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or radius <= 0
        or not 0 < thickness <= radius
    ):
        raise ValueError("pixel goal image or marker differs")
    y, x = goal_yx
    size = image.shape[0]
    if not radius <= y < size - radius or not radius <= x < size - radius:
        raise ValueError("pixel goal outside canvas")
    marked = image.copy()
    yy, xx = np.ogrid[:size, :size]
    squared = (yy - y) ** 2 + (xx - x) ** 2
    ring = ((radius - thickness) ** 2 <= squared) & (squared <= radius**2)
    marked[ring] = (245, 225, 45)
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
    player_yx: tuple[float, float],
    action: str,
    distance: int,
    marker_radius: int,
    *,
    canvas_size: int = 128,
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
        marker_radius <= goal_y < canvas_size - marker_radius
        and marker_radius <= goal_x < canvas_size - marker_radius
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
    if frame.ndim != 3 or frame.shape[0] != frame.shape[1] or frame.shape[2] != 3:
        raise ValueError("player mask requires square RGB")
    size = frame.shape[0]
    center_y, center_x = map(round, player_yx)
    y0, y1 = max(0, center_y - radius), min(size, center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(size, center_x + radius + 1)
    outer = radius + 4
    outer_y0, outer_y1 = max(0, center_y - outer), min(size, center_y + outer + 1)
    outer_x0, outer_x1 = max(0, center_x - outer), min(size, center_x + outer + 1)
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


JOYSTICK_COVERAGE_FRACTIONS = (0.05, 0.15, 0.2, 0.25, 0.35, 0.4,
                              0.45, 0.5, 0.65, 0.7, 0.8, 0.9)
JOYSTICK_TRANSFER_SOURCES = (
    "06c5a8e67a7a19a59e4b13bdae9dafeaa2b07a0168e41406d58e5209ed0e2b4a",
    "08565f01b75fb400394b35d33cd2bab8a94298f082e127115edc27f0b073b46b",
    "0a84c341d16222bb3424e95cbb5d51797eecbea8b528417cd02d569ae9a360cc",
    "0c60062fdde6ed50e0bf2fe231fda206535b90911d2f74e6f0f171a54e486c6c",
)
JOYSTICK_TRANSFER_FRACTIONS = (0.2, 0.5, 0.8)
JOYSTICK_FINAL_TRANSFER_SOURCES = (
    "0e34a785656d464bd946559e9a7ac9602ef2ffcd0e7a9c6d8ba265f37261de24",
    "109a2a343dd37d406f987cc22a4c4c876e59a4e93e9f4127bf5ba06107bd91d7",
    "12214351b55ac24beffe2c52b77464e009120a3a3cc69fc7cb17adcfe1f87e39",
)
JOYSTICK_FINAL_TRANSFER_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
JOYSTICK_V3_FINGERPRINT = "8063ac69f592096358c8ba9ea9bcf04f260b295e5e5ab61d51a60ef4d2a5dbb9"


def joystick_extractor_fingerprint() -> str:
    import ast

    names = {"_joystick_signal", "_joystick_match", "_joystick_geometric_base",
             "extract_joystick_sequence"}
    tree = ast.parse(Path(__file__).read_text())
    return _object_sha256({
        "functions": [ast.dump(n, include_attributes=False) for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name in names],
        "settings": JOYSTICK_EXTRACT_SETTINGS, "scales": JOYSTICK_SCALES,
        "marker_boxes": JOYSTICK_MARKER_BOXES,
    })


def joystick_coverage_summary(windows: list[dict[str, object]]) -> dict[str, object]:
    actions = ("STOP", "N", "S", "W", "E", "NW", "NE", "SW", "SE")
    counts = dict.fromkeys(actions, 0)
    supports = dict.fromkeys(actions, 0)
    runs = dict.fromkeys(actions, 0)
    known = total = 0
    for window in windows:
        seen: set[str] = set()
        previous = "unknown"
        for row in cast(list[dict[str, object]], window["predictions"]):
            action = str(row["candidate_action"])
            total += 1
            if action in counts:
                counts[action] += 1
                known += 1
                seen.add(action)
                if previous != action:
                    runs[action] += 1
            previous = action
        for action in seen:
            supports[action] += 1
    return {"total_frames": total, "known_frames": known, "unknown_frames": total - known,
            "coverage": known / total if total else 0,
            "class_frames": counts, "class_windows": supports, "class_runs": runs,
            "missing_classes": [a for a in actions if not counts[a]],
            "all_nine_classes_observed": all(counts.values()),
            "training_allowed": False,
            "interpretation": "correlated UI candidates, not independent examples or accuracy"}


def joystick_training_eligibility(
    windows: list[dict[str, object]], *, minimum_run_frames: int = 2,
    release_lookback_ms: int = 500,
) -> dict[str, object]:
    """Count causal candidates without materializing RGB policy samples."""
    directions = ("N", "S", "W", "E", "NW", "NE", "SW", "SE")
    stable_frames = dict.fromkeys(directions, 0)
    stable_runs = dict.fromkeys(directions, 0)
    releases: list[dict[str, object]] = []
    for window in windows:
        predictions = cast(list[dict[str, object]], window["predictions"])
        actions = [str(row["candidate_action"]) for row in predictions]
        times = list(map(int, cast(list[int], window["timestamp_us"])))
        start = 0
        while start < len(actions):
            end = start + 1
            while end < len(actions) and actions[end] == actions[start]:
                end += 1
            action = actions[start]
            if action in stable_frames and end - start >= minimum_run_frames:
                stable_frames[action] += end - start
                stable_runs[action] += 1
            start = end
        for index, action in enumerate(actions):
            if action != "STOP" or (index and actions[index - 1] == "STOP"):
                continue
            previous = next(
                (earlier for earlier in range(index - 1, -1, -1)
                 if times[index] - times[earlier] <= release_lookback_ms * 1000
                 and actions[earlier] in directions),
                None,
            )
            if previous is not None:
                releases.append({
                    "fraction": window["fraction"], "frame_index": index,
                    "label_timestamp_us": times[index],
                    "input_end_timestamp_us": times[index - 1],
                    "actual_input_label_gap_us": times[index] - times[index - 1],
                    "previous_direction": actions[previous],
                    "previous_direction_age_ms": (times[index] - times[previous]) / 1000,
                })
    all_directions = all(stable_runs.values())
    enough_stop = len(releases) >= 8
    return {
        "stable_direction_frames": stable_frames, "stable_direction_runs": stable_runs,
        "minimum_run_frames": minimum_run_frames, "release_stop_events": releases,
        "release_stop_count": len(releases), "release_lookback_ms": release_lookback_ms,
        "all_directions_have_stable_run": all_directions,
        "minimum_release_stop_events": 8, "stop_support_passed": enough_stop,
        "single_session_only": True, "cross_session_support": False,
        "causal_policy": (
            "Actor RGB ends at the previous sampled PTS; joystick target is the current PTS. "
            "Stable direction labels require the current and next UI candidates to agree."
        ),
        "retrospective_confirmation": True,
        "sample_materialization_allowed": all_directions and enough_stop and False,
        "training_allowed": False,
    }


def joystick_continuation_support(
    sessions: list[dict[str, object]], dev_sources: set[str],
) -> dict[str, object]:
    directions = JOYSTICK_ACTIONS[1:]
    split_rows: dict[str, object] = {}
    for split in ("train", "dev"):
        onset = dict.fromkeys(directions, 0)
        one_history = dict.fromkeys(directions, 0)
        two_history = dict.fromkeys(directions, 0)
        sources: dict[str, set[str]] = {action: set() for action in directions}
        stop_runs = stop_one = stop_two = releases = 0
        split_source_ids: set[str] = set()
        for session in sessions:
            source_id = str(session["session"])
            if (source_id in dev_sources) != (split == "dev"):
                continue
            split_source_ids.add(source_id)
            windows = cast(list[dict[str, object]], session["windows"])
            for window in windows:
                actions = [str(row["candidate_action"])
                           for row in cast(list[dict[str, object]], window["predictions"])]
                start = 0
                while start < len(actions):
                    end = start + 1
                    while end < len(actions) and actions[end] == actions[start]:
                        end += 1
                    action, length = actions[start], end - start
                    if action in onset and length >= 2:
                        onset[action] += 1
                        one_history[action] += length - 1
                        if length >= 3:
                            two_history[action] += length - 2
                            sources[action].add(source_id)
                    elif action == "STOP":
                        stop_runs += 1
                        stop_one += max(0, length - 1)
                        stop_two += max(0, length - 2)
                    start = end
            releases += cast(int, joystick_training_eligibility(windows)["release_stop_count"])
        source_support = {action: len(value) for action, value in sources.items()}
        split_rows[split] = {
            "sources": len(split_source_ids), "stable_run_onsets": onset,
            "one_prior_same_direction_frames": one_history,
            "two_prior_same_direction_frames": two_history,
            "two_prior_direction_source_support": source_support,
            "direction_continuation_samples": sum(two_history.values()),
            "stop": {"runs": stop_runs, "one_prior_center_frames": stop_one,
                     "two_prior_center_frames": stop_two, "release_events": releases,
                     "learning_eligible": False,
                     "reason": "centered UI scene semantics remain unresolved"},
        }
    train = cast(dict[str, object], split_rows["train"])
    dev = cast(dict[str, object], split_rows["dev"])
    train_counts = cast(dict[str, int], train["two_prior_same_direction_frames"])
    dev_counts = cast(dict[str, int], dev["two_prior_same_direction_frames"])
    train_sources = cast(dict[str, int], train["two_prior_direction_source_support"])
    dev_sources_count = cast(dict[str, int], dev["two_prior_direction_source_support"])
    gates = {
        "train_each_direction_at_least_5": min(train_counts.values()) >= 5,
        "train_each_direction_at_least_3_sources": min(train_sources.values()) >= 3,
        "dev_each_direction_at_least_1": min(dev_counts.values()) >= 1,
        "dev_each_direction_at_least_1_source": min(dev_sources_count.values()) >= 1,
        "source_split_is_16_5": train["sources"] == 16 and dev["sources"] == 5,
    }
    return {
        "definition": (
            "direction target at the third or later equal candidate in a run; the preceding "
            "two sampled frames carry the same joystick direction"
        ),
        "actor_input": "16 RGB frames ending at the preceding PTS; joystick pixels excluded",
        "retrospective_ui_requirement": True, "splits": split_rows, "gates": gates,
        "direction_continuation_allowed": all(gates.values()),
        "learned_stop_allowed": False, "training_allowed": False,
    }


def run_joystick_continuation_audit(
    source_run: Path, failed_pilot_report: Path, output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise ValueError("continuation audit output exists")
    conclusion_path = source_run / "conclusion.json"
    conclusion = cast(dict[str, object], json.loads(conclusion_path.read_text()))
    failed = _load_bound_json(failed_pilot_report, "report_sha256")
    if (conclusion.get("status") != "JOYSTICK_SCALE24_LANDSCAPE_SUBSET_PASSED"
            or failed.get("schema_version") != "joystick-scale21-pilot-report-v1"
            or failed.get("passed") is not False
            or failed.get("next_stage_allowed") is not False):
        raise ValueError("continuation audit requires frozen scale evidence and failed pilot")
    sessions, report_hashes = _joystick_scale21_sessions(source_run.parent, source_run)
    support = joystick_continuation_support(sessions, JOYSTICK_SCALE21_DEV_SOURCES)
    report: dict[str, object] = {
        "schema_version": "joystick-continuation-support-audit-v1",
        "source_report_file_sha256": report_hashes,
        "scale_conclusion_file_sha256": _file_sha256(conclusion_path),
        "failed_pilot_report_file_sha256": _file_sha256(failed_pilot_report),
        "support": support,
        "status": (
            "JOYSTICK_DIRECTION_CONTINUATION_SUPPORT_PASSED"
            if support["direction_continuation_allowed"]
            else "JOYSTICK_DIRECTION_CONTINUATION_SUPPORT_FAILED"
        ),
        "dataset_materialization_allowed": support["direction_continuation_allowed"],
        "learned_action_space": list(JOYSTICK_ACTIONS[1:]),
        "stop_owner": "deterministic_router", "rgb_decodes": 0, "model_runs": 0,
        "training_allowed": False, "device_input_allowed": False,
        "video_dev_opened": False, "video_test_opened": False,
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-continuation-", dir=output_dir.parent))
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return report


def run_joystick_eligibility(source_run: Path, output_dir: Path) -> dict[str, object]:
    source = _load_bound_json(source_run / "report.json", "report_sha256")
    contract = _load_bound_json(source_run / "contract.json", "contract_sha256")
    if (source.get("schema_version") != "joystick-train-coverage-v1"
            or source.get("contract_sha256") != contract["contract_sha256"]
            or source.get("dev_frames_opened") != 0
            or source.get("test_frames_opened") != 0):
        raise ValueError("eligibility requires frozen train coverage")
    if output_dir.exists():
        raise ValueError("eligibility output exists")
    result = joystick_training_eligibility(cast(list[dict[str, object]], source["windows"]))
    report: dict[str, object] = {
        "schema_version": "joystick-training-eligibility-v1",
        "source_report_sha256": source["report_sha256"],
        "source_report_file_sha256": _file_sha256(source_run / "report.json"),
        "source_contract_sha256": contract["contract_sha256"],
        "eligibility": result,
        "status": (
            "JOYSTICK_CAUSAL_CANDIDATES_INSUFFICIENT_STOP_AND_SESSION_SUPPORT"
            if not result["sample_materialization_allowed"] else "READY_TO_MATERIALIZE"
        ),
        "policy_samples_written": 0, "training_allowed": False,
        "input_commands_sent": 0, "gpu_seconds": 0,
        "raw_video_decodes": 0, "dev_frames_opened": 0, "test_frames_opened": 0,
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-eligibility-", dir=output_dir.parent))
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return report


def joystick_transfer_summary(sessions: list[dict[str, object]]) -> dict[str, object]:
    directions = ("N", "S", "W", "E", "NW", "NE", "SW", "SE")
    direction_sessions = {
        action: sum(
            cast(dict[str, int], cast(dict[str, object], row["eligibility"])
                 ["stable_direction_runs"])[action] > 0
            for row in sessions
        )
        for action in directions
    }
    session_coverage = [cast(float, cast(dict[str, object], row["coverage"])["coverage"])
                        for row in sessions]
    return {
        "sessions": len(sessions), "direction_supporting_sessions": direction_sessions,
        "release_stop_events": sum(
            cast(int, cast(dict[str, object], row["eligibility"])["release_stop_count"])
            for row in sessions
        ),
        "sessions_with_any_candidate": sum(value > 0 for value in session_coverage),
        "mean_candidate_coverage": sum(session_coverage) / len(session_coverage),
        "minimum_candidate_coverage": min(session_coverage),
        "all_directions_have_two_sessions": all(
            value >= 2 for value in direction_sessions.values()
        ),
        "training_allowed": False,
        "interpretation": "fixed-extractor transfer candidates; not semantic accuracy",
    }


def _run_joystick_transfer(
    source_root: Path, cohort_dir: Path, pre_ingest_path: Path,
    extractor_run: Path, output_dir: Path,
    *, source_ids: tuple[str, ...], fractions: tuple[float, ...], schema_version: str,
    selection_contract_sha256: str | None = None,
    selection_preflight: dict[str, object] | None = None,
) -> dict[str, object]:
    import cv2
    from PIL import Image, ImageDraw

    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    cv2.setNumThreads(1)
    if output_dir.exists():
        raise ValueError("transfer output exists")
    frozen = _load_bound_json(extractor_run / "contract.json", "contract_sha256")
    prior = _load_bound_json(extractor_run / "report.json", "report_sha256")
    if (frozen["contract_sha256"] !=
            "f687f5423235e918fe37728793d24f0ea73a33936fc46344429d38ad91ec7c7c"
            or prior.get("contract_sha256") != frozen["contract_sha256"]
            or joystick_extractor_fingerprint() != JOYSTICK_V3_FINGERPRINT):
        raise ValueError("transfer requires frozen v3 extractor")
    template_path = extractor_run / "train-templates.npz"
    if _file_sha256(template_path) != frozen["template_sha256"]:
        raise ValueError("transfer template differs")
    with np.load(template_path, allow_pickle=False) as arrays:
        templates = {key: arrays[key].copy() for key in arrays.files}
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    if any(cohort.session_splits.get(identity) != "train"
           for identity in source_ids):
        raise ValueError("transfer sources must all be train")
    sources: dict[str, Path] = {}
    for path in pre_ingest._scan(source_root):
        identity = pre_ingest._sha(pre_ingest._canonical([
            "candidate-v2-file-atomic", path.relative_to(source_root).as_posix(),
            pre_ingest._stat_signature(path.stat()),
        ]))
        if identity in source_ids:
            sources[identity] = path
    if set(sources) != set(source_ids):
        raise ValueError("transfer sources unavailable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-transfer-", dir=output_dir.parent))
    contract: dict[str, object] = {
        "extractor_contract_sha256": frozen["contract_sha256"],
        "extractor_fingerprint": JOYSTICK_V3_FINGERPRINT,
        "template_sha256": frozen["template_sha256"],
        "sources": source_ids, "split": "train",
        "fractions": fractions, "frames_per_window": 40,
        "sample_period_us": 100000, "training_allowed": False,
        "dev_frames_opened": 0, "test_frames_opened": 0,
        "selection_contract_sha256": selection_contract_sha256,
        "selection_preflight": selection_preflight,
    }
    contract["contract_sha256"] = _object_sha256(contract)
    (staging / "contract.json").write_bytes(_canonical(contract) + b"\n")
    session_rows: list[dict[str, object]] = []
    for identity, path in sorted(sources.items()):
        if pre_ingest._candidate(path, source_root).candidate_id != identity:
            raise ValueError("transfer source identity changed")
        windows: list[dict[str, object]] = []
        qa_items: list[tuple[Image.Image, str]] = []
        previous_end = -1
        for fraction in fractions:
            arrays = _joystick_window(path, fraction)
            frames, times = arrays["rgb"], arrays["timestamp_us"]
            if times[0] <= previous_end:
                raise ValueError("transfer windows overlap")
            previous_end = int(times[-1])
            predictions = extract_joystick_sequence(
                frames, templates, normalize_scale=True, geometric_base=True
            )
            indices = list(map(int, np.linspace(0, len(frames) - 1, 4)))
            for index in indices:
                row = predictions[index]
                image = Image.fromarray(frames[index])
                draw = ImageDraw.Draw(image)
                bx, by = cast(list[float], row["base_xy"])
                kx, ky = cast(list[float], row["knob_xy"])
                draw.ellipse((bx-10, by-10, bx+10, by+10), outline="yellow", width=3)
                draw.ellipse((kx-10, ky-10, kx+10, ky+10), outline="magenta", width=3)
                draw.line((bx, by, kx, ky), fill="yellow", width=3)
                image.thumbnail((320, 210))
                qa_items.append((image, f"f{fraction:.1f}:{index} {row['candidate_action']}"))
            windows.append({
                "fraction": fraction, "timestamp_us": times.tolist(),
                "predictions": predictions,
                "frame_sha256": [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames],
            })
        sheet = Image.new("RGB", (4*320, math.ceil(len(qa_items)/4)*240))
        for ordinal, (image, label) in enumerate(qa_items):
            x, y = ordinal % 4*320, ordinal // 4*240
            sheet.paste(image, (x, y+26))
            ImageDraw.Draw(sheet).text((x, y), label, fill="white")
        qa_name = f"{identity[:8]}-qa.png"
        sheet.save(staging / qa_name)
        session_rows.append({
            "session": identity, "split": "train", "windows": windows,
            "coverage": joystick_coverage_summary(windows),
            "eligibility": joystick_training_eligibility(windows),
            "qa": qa_name, "qa_sha256": _file_sha256(staging / qa_name),
        })
    summary = joystick_transfer_summary(session_rows)
    report: dict[str, object] = {
        "schema_version": schema_version,
        "contract_sha256": contract["contract_sha256"], "sessions": session_rows,
        "summary": summary, "status": "QA_REQUIRED", "training_allowed": False,
        "native_cache_written": False, "gpu_seconds": 0, "input_commands_sent": 0,
        "dev_frames_opened": 0, "test_frames_opened": 0,
    }
    report["report_sha256"] = _object_sha256(report)
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return {k: v for k, v in report.items() if k != "sessions"}


def run_joystick_transfer(
    source_root: Path, cohort_dir: Path, pre_ingest_path: Path,
    extractor_run: Path, output_dir: Path,
) -> dict[str, object]:
    return _run_joystick_transfer(
        source_root, cohort_dir, pre_ingest_path, extractor_run, output_dir,
        source_ids=JOYSTICK_TRANSFER_SOURCES, fractions=JOYSTICK_TRANSFER_FRACTIONS,
        schema_version="joystick-cross-train-transfer-v1",
    )


JOYSTICK_ACTIONS = ("STOP", "N", "S", "W", "E", "NW", "NE", "SW", "SE")
JOYSTICK_CONTINUATION_ACTIONS = JOYSTICK_ACTIONS[1:]
JOYSTICK_PILOT_DEV_SOURCES = {
    "0e34a785656d464bd946559e9a7ac9602ef2ffcd0e7a9c6d8ba265f37261de24",
    "12214351b55ac24beffe2c52b77464e009120a3a3cc69fc7cb17adcfe1f87e39",
}
JOYSTICK_SCALE21_DEV_SOURCES = {
    *JOYSTICK_PILOT_DEV_SOURCES,
    "1720186328476967e418016d404814eab4186d6f806c8ff77eb4cbb149c6dab1",
    "3927aa0891e9712e9122db7dae2832a71332d9744da78b404cbc75bdb97e55d2",
    "493cf58d4f09b0dd0e0e0b25c1a58eaa141874ef4a51b010e6ac3d64a85b1417",
}


def select_joystick_overfit32(
    sessions: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for action in JOYSTICK_ACTIONS[1:]:
        candidates: list[dict[str, object]] = []
        for session in sessions:
            for window in cast(list[dict[str, object]], session["windows"]):
                predictions = cast(list[dict[str, object]], window["predictions"])
                times = list(map(int, cast(list[int], window["timestamp_us"])))
                start = 0
                while start < len(predictions):
                    current = str(predictions[start]["candidate_action"])
                    end = start + 1
                    while (end < len(predictions)
                           and predictions[end]["candidate_action"] == current):
                        end += 1
                    if current == action and end - start >= 2:
                        candidates.append({
                            "source_id": session["session"], "action": action,
                            "label_timestamp_us": times[start],
                            "confirmation_timestamp_us": times[start + 1],
                            "source_fraction": window["fraction"], "source_frame_index": start,
                            "label_rule": "two_equal_direction_candidates",
                        })
                        break
                    start = end
        by_source = {str(row["source_id"]): row for row in candidates}
        if len(by_source) < 3:
            raise ValueError(f"insufficient source support for {action}")
        selected.extend(by_source[key] for key in sorted(by_source)[:3])
    releases: list[dict[str, object]] = []
    for session in sessions:
        result = joystick_training_eligibility(
            cast(list[dict[str, object]], session["windows"])
        )
        for event in cast(list[dict[str, object]], result["release_stop_events"]):
            releases.append({
                "source_id": session["session"], "action": "STOP",
                "label_timestamp_us": event["label_timestamp_us"],
                "confirmation_timestamp_us": None,
                "source_fraction": event["fraction"],
                "source_frame_index": event["frame_index"],
                "previous_direction": event["previous_direction"],
                "previous_direction_age_ms": event["previous_direction_age_ms"],
                "label_rule": "direction_to_center_release_within_500ms",
            })
    releases.sort(
        key=lambda row: (str(row["source_id"]), cast(int, row["label_timestamp_us"]))
    )
    if len(releases) != 8:
        raise ValueError("overfit32 requires exactly eight release STOP events")
    selected.extend(releases)
    counts = Counter(str(row["action"]) for row in selected)
    expected = {"STOP": 8, **dict.fromkeys(JOYSTICK_ACTIONS[1:], 3)}
    if len(selected) != 32 or dict(counts) != expected:
        raise ValueError("overfit32 selection differs")
    if len({(row["source_id"], row["label_timestamp_us"]) for row in selected}) != 32:
        raise ValueError("overfit32 labels are not unique")
    return selected


def select_joystick_grouped_pilot(
    sessions: list[dict[str, object]], *,
    dev_sources: set[str] | None = None,
    expected_samples: tuple[int, int] = (73, 25),
) -> dict[str, list[dict[str, object]]]:
    selected_dev = JOYSTICK_PILOT_DEV_SOURCES if dev_sources is None else dev_sources
    result: dict[str, list[dict[str, object]]] = {"train": [], "dev": []}
    for session in sessions:
        source_id = str(session["session"])
        split = "dev" if source_id in selected_dev else "train"
        windows = cast(list[dict[str, object]], session["windows"])
        for window in windows:
            predictions = cast(list[dict[str, object]], window["predictions"])
            times = list(map(int, cast(list[int], window["timestamp_us"])))
            start = 0
            while start < len(predictions):
                action = str(predictions[start]["candidate_action"])
                end = start + 1
                while (end < len(predictions)
                       and predictions[end]["candidate_action"] == action):
                    end += 1
                if action in JOYSTICK_ACTIONS[1:] and end - start >= 2:
                    result[split].append({
                        "source_id": source_id, "action": action,
                        "label_timestamp_us": times[start],
                        "confirmation_timestamp_us": times[start + 1],
                        "source_fraction": window["fraction"],
                        "source_frame_index": start,
                        "label_rule": "stable_run_onset_two_equal_candidates",
                    })
                start = end
        eligibility = joystick_training_eligibility(windows)
        for event in cast(list[dict[str, object]], eligibility["release_stop_events"]):
            result[split].append({
                "source_id": source_id, "action": "STOP",
                "label_timestamp_us": event["label_timestamp_us"],
                "confirmation_timestamp_us": None,
                "source_fraction": event["fraction"],
                "source_frame_index": event["frame_index"],
                "previous_direction": event["previous_direction"],
                "previous_direction_age_ms": event["previous_direction_age_ms"],
                "label_rule": "direction_to_center_release_within_500ms",
            })
    for split, rows in result.items():
        rows.sort(key=lambda row: (str(row["source_id"]), cast(int, row["label_timestamp_us"])))
        counts = Counter(str(row["action"]) for row in rows)
        if set(counts) != set(JOYSTICK_ACTIONS):
            raise ValueError(f"{split} pilot split lacks a class")
        if len({(row["source_id"], row["label_timestamp_us"]) for row in rows}) != len(rows):
            raise ValueError(f"{split} pilot labels are not unique")
    train_sources = {str(row["source_id"]) for row in result["train"]}
    dev_sources = {str(row["source_id"]) for row in result["dev"]}
    if (train_sources & dev_sources or dev_sources != selected_dev
            or len(train_sources) != len(sessions) - len(selected_dev)
            or len(result["train"]) != expected_samples[0]
            or len(result["dev"]) != expected_samples[1]):
        raise ValueError("pilot source grouping or sample counts differ")
    return result


def select_joystick_continuation(
    sessions: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {"train": [], "dev": []}
    for session in sessions:
        source_id = str(session["session"])
        split = "dev" if source_id in JOYSTICK_SCALE21_DEV_SOURCES else "train"
        for window in cast(list[dict[str, object]], session["windows"]):
            predictions = cast(list[dict[str, object]], window["predictions"])
            times = list(map(int, cast(list[int], window["timestamp_us"])))
            start = 0
            while start < len(predictions):
                action = str(predictions[start]["candidate_action"])
                end = start + 1
                while (end < len(predictions)
                       and predictions[end]["candidate_action"] == action):
                    end += 1
                if action in JOYSTICK_CONTINUATION_ACTIONS and end - start >= 3:
                    for index in range(start + 2, end):
                        result[split].append({
                            "source_id": source_id, "action": action,
                            "label_timestamp_us": times[index],
                            "prior_same_direction_timestamp_us": [
                                times[index - 2], times[index - 1]
                            ],
                            "source_fraction": window["fraction"],
                            "source_frame_index": index,
                            "label_rule": "third_or_later_equal_direction_candidate",
                        })
                start = end
    for split, expected in (("train", 203), ("dev", 80)):
        rows = result[split]
        rows.sort(key=lambda row: (str(row["source_id"]), cast(int, row["label_timestamp_us"])))
        counts = Counter(str(row["action"]) for row in rows)
        if len(rows) != expected or set(counts) != set(JOYSTICK_CONTINUATION_ACTIONS):
            raise ValueError(f"{split} continuation counts differ")
        if len({(row["source_id"], row["label_timestamp_us"]) for row in rows}) != len(rows):
            raise ValueError(f"{split} continuation labels are not unique")
    if ({str(row["source_id"]) for row in result["train"]}
            & {str(row["source_id"]) for row in result["dev"]}):
        raise ValueError("continuation source groups overlap")
    return result


def _masked_actor_frame(rgb: np.ndarray) -> np.ndarray:
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("actor frame must be uint8 RGB")
    masked = rgb.copy()
    masked[round(rgb.shape[0] * 0.45):, :round(rgb.shape[1] * 0.35)] = 0
    resized = cv2.resize(masked, (128, 128), interpolation=cv2.INTER_AREA)
    resized[round(128 * 0.45):, :round(128 * 0.35)] = 0
    return resized


def _actor_window_before_label(path: Path, label_us: int) -> tuple[np.ndarray, np.ndarray]:
    import av

    from hok_agent import pre_ingest

    targets = [label_us - offset * 100_000 for offset in range(16, 0, -1)]
    if targets[0] < 0:
        raise ValueError("label lacks sixteen causal frames")
    descriptor, opened = pre_ingest._open_regular(path)
    frames: list[np.ndarray] = []
    times: list[int] = []
    with os.fdopen(descriptor, "rb") as handle, av.open(handle, mode="r") as container:
        stream = container.streams.video[0]
        if (stream.time_base is None or stream.width <= stream.height
                or pre_ingest._rotation(stream) != 0):
            raise ValueError("actor window requires unrotated landscape video")
        seek_us = max(0, targets[0] - 500_000)
        container.seek(int(seek_us / float(stream.time_base) / 1_000_000),
                       stream=stream, backward=True)
        target_index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("actor frame lacks PTS")
            timestamp = round(frame.pts * stream.time_base * 1_000_000)
            if timestamp >= label_us:
                break
            while target_index < len(targets) and timestamp >= targets[target_index]:
                if times and timestamp == times[-1]:
                    raise ValueError("one decoded frame satisfies multiple actor targets")
                frames.append(_masked_actor_frame(frame.to_ndarray(format="rgb24")))
                times.append(timestamp)
                target_index += 1
            if target_index == len(targets):
                break
        pre_ingest._assert_unchanged(handle.fileno(), opened)
    if (len(frames) != 16 or len(set(times)) != 16 or times[-1] >= label_us
            or not np.all(np.diff(times) > 0)):
        raise ValueError("incomplete causal actor window")
    return np.stack(frames), np.asarray(times, dtype=np.int64)


def _joystick_dataset_sessions(root: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    files = {
        "single_source_coverage": root / "joystick-train-coverage-v1" / "report.json",
        "single_source_eligibility": root / "joystick-training-eligibility-v1" / "report.json",
        "four_source_transfer": root / "joystick-cross-train-transfer-v1" / "report.json",
        "three_source_final_transfer": root / "joystick-final-train-transfer-v1" / "report.json",
    }
    reports = {key: _load_bound_json(path, "report_sha256") for key, path in files.items()}
    source_id = next(key for key, split in NATIVE_PLAYER_SOURCES.items() if split == "train")
    coverage = reports["single_source_coverage"]
    sessions = [{"session": source_id, "windows": coverage["windows"]}]
    sessions.extend(cast(list[dict[str, object]], reports["four_source_transfer"]["sessions"]))
    sessions.extend(
        cast(list[dict[str, object]], reports["three_source_final_transfer"]["sessions"])
    )
    if len(sessions) != 8 or len({row["session"] for row in sessions}) != 8:
        raise ValueError("dataset requires eight unique source sessions")
    return sessions, {key: _file_sha256(path) for key, path in files.items()}


def _joystick_scale21_sessions(
    root: Path, scale_run: Path,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    sessions, hashes = _joystick_dataset_sessions(root)
    scale_path = scale_run / "report.json"
    scale = _load_bound_json(scale_path, "report_sha256")
    if (scale.get("schema_version") != "joystick-scale24-audit-v2-landscape-subset"
            or scale.get("dev_frames_opened") != 0 or scale.get("test_frames_opened") != 0):
        raise ValueError("scale21 dataset requires completed train-only scale audit")
    sessions.extend(cast(list[dict[str, object]], scale["sessions"]))
    hashes["scale24_landscape_subset"] = _file_sha256(scale_path)
    if len(sessions) != 21 or len({row["session"] for row in sessions}) != 21:
        raise ValueError("scale21 dataset requires 21 unique sessions")
    return sessions, hashes


def validate_joystick_overfit32(output_dir: Path) -> dict[str, object]:
    manifest = _load_bound_json(output_dir / "manifest.json", "manifest_sha256")
    dataset_path = output_dir / "joystick-overfit32.npz"
    if dataset_path.is_symlink() or _file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("joystick dataset hash differs")
    with np.load(dataset_path, allow_pickle=False) as data:
        clips, labels, times, targets = (
            data["rgb"], data["label"], data["input_timestamp_us"], data["label_timestamp_us"]
        )
    if (clips.shape != (32, 16, 128, 128, 3) or clips.dtype != np.uint8
            or labels.shape != (32,) or times.shape != (32, 16) or targets.shape != (32,)):
        raise ValueError("joystick dataset arrays differ")
    if not np.all(np.diff(times, axis=1) > 0) or not np.all(times[:, -1] < targets):
        raise ValueError("joystick dataset is not causal")
    if np.any(clips[:, :, round(128 * 0.45):, :round(128 * 0.35)]):
        raise ValueError("joystick pixels remain in Actor input")
    counts = dict(Counter(JOYSTICK_ACTIONS[int(label)] for label in labels))
    if counts != {"STOP": 8, **dict.fromkeys(JOYSTICK_ACTIONS[1:], 3)}:
        raise ValueError("joystick dataset class balance differs")
    return {"status": "JOYSTICK_OVERFIT32_DATASET_VALIDATED", "samples": 32,
            "class_counts": counts, "causal": True, "joystick_pixels_zero": True,
            "training_allowed": False}


def _validate_joystick_grouped_dataset(
    output_dir: Path, *, dataset_name: str, expected_samples: tuple[int, int],
    expected_sources: tuple[int, int], dev_sources: set[str], status: str,
    action_order: tuple[str, ...] = JOYSTICK_ACTIONS,
) -> dict[str, object]:
    manifest = _load_bound_json(output_dir / "manifest.json", "manifest_sha256")
    dataset_path = output_dir / dataset_name
    if dataset_path.is_symlink() or _file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("grouped joystick dataset hash differs")
    summaries: dict[str, object] = {}
    source_sets: dict[str, set[str]] = {}
    all_hashes: list[str] = []
    with np.load(dataset_path, allow_pickle=False) as data:
        for split, expected in zip(("train", "dev"), expected_samples, strict=True):
            clips = data[f"{split}_rgb"]
            labels = data[f"{split}_label"]
            times = data[f"{split}_input_timestamp_us"]
            targets = data[f"{split}_label_timestamp_us"]
            sources = data[f"{split}_source_id"]
            if (clips.shape != (expected, 16, 128, 128, 3) or clips.dtype != np.uint8
                    or labels.shape != (expected,) or times.shape != (expected, 16)
                    or targets.shape != (expected,) or sources.shape != (expected,)):
                raise ValueError(f"{split} grouped arrays differ")
            if not np.all(np.diff(times, axis=1) > 0) or not np.all(times[:, -1] < targets):
                raise ValueError(f"{split} grouped timing is not causal")
            if np.any(clips[:, :, round(128 * 0.45):, :round(128 * 0.35)]):
                raise ValueError(f"{split} grouped Actor input contains joystick pixels")
            counts = dict(Counter(action_order[int(label)] for label in labels))
            if set(counts) != set(action_order):
                raise ValueError(f"{split} grouped labels lack a class")
            source_sets[split] = set(map(str, sources.tolist()))
            all_hashes.extend(hashlib.sha256(clip.tobytes()).hexdigest() for clip in clips)
            summaries[split] = {"samples": expected, "classes": counts,
                                "sources": len(source_sets[split])}
    if (source_sets["train"] & source_sets["dev"]
            or len(source_sets["train"]) != expected_sources[0]
            or source_sets["dev"] != dev_sources
            or len(set(all_hashes)) != sum(expected_samples)):
        raise ValueError("grouped source isolation or clip uniqueness differs")
    return {"status": status, "splits": summaries,
            "source_overlap": 0, "unique_actor_clips": sum(expected_samples), "causal": True,
            "joystick_pixels_zero": True, "pilot_training_allowed": True,
            "formal_training_allowed": False}


def validate_joystick_grouped_pilot(output_dir: Path) -> dict[str, object]:
    return _validate_joystick_grouped_dataset(
        output_dir, dataset_name="joystick-grouped-pilot.npz",
        expected_samples=(73, 25), expected_sources=(6, 2),
        dev_sources=JOYSTICK_PILOT_DEV_SOURCES,
        status="JOYSTICK_GROUPED_PILOT_DATASET_VALIDATED",
    )


def validate_joystick_scale21_dataset(output_dir: Path) -> dict[str, object]:
    return _validate_joystick_grouped_dataset(
        output_dir, dataset_name="joystick-scale21-grouped.npz",
        expected_samples=(201, 48), expected_sources=(16, 5),
        dev_sources=JOYSTICK_SCALE21_DEV_SOURCES,
        status="JOYSTICK_SCALE21_GROUPED_DATASET_VALIDATED",
    )


def validate_joystick_continuation_dataset(output_dir: Path) -> dict[str, object]:
    return _validate_joystick_grouped_dataset(
        output_dir, dataset_name="joystick-continuation-grouped.npz",
        expected_samples=(203, 80), expected_sources=(16, 5),
        dev_sources=JOYSTICK_SCALE21_DEV_SOURCES,
        status="JOYSTICK_CONTINUATION_DATASET_VALIDATED",
        action_order=JOYSTICK_CONTINUATION_ACTIONS,
    )


def run_joystick_materialize_continuation(
    source_root: Path, cohort_dir: Path, pre_ingest_path: Path,
    audit_run: Path, output_dir: Path, *, verify_only: bool = False,
) -> dict[str, object]:
    if verify_only:
        return validate_joystick_continuation_dataset(output_dir)
    if output_dir.exists():
        raise ValueError("continuation dataset output exists")
    audit = _load_bound_json(audit_run / "report.json", "report_sha256")
    if (audit.get("schema_version") != "joystick-continuation-support-audit-v1"
            or audit.get("status") != "JOYSTICK_DIRECTION_CONTINUATION_SUPPORT_PASSED"
            or audit.get("dataset_materialization_allowed") is not True
            or audit.get("learned_action_space") != list(JOYSTICK_CONTINUATION_ACTIONS)
            or audit.get("stop_owner") != "deterministic_router"):
        raise ValueError("continuation audit does not allow materialization")
    scale_run = audit_run.parent / "joystick-scale24-audit-v2-landscape-subset"
    sessions, report_hashes = _joystick_scale21_sessions(audit_run.parent, scale_run)
    if report_hashes != audit["source_report_file_sha256"]:
        raise ValueError("continuation source report bindings differ")
    selected = select_joystick_continuation(sessions)
    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    source_ids = {str(row["source_id"]) for rows in selected.values() for row in rows}
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    if any(cohort.session_splits.get(source_id) != "train" for source_id in source_ids):
        raise ValueError("continuation sources must all be cohort train")
    sources: dict[str, Path] = {}
    for path in pre_ingest._scan(source_root):
        identity = pre_ingest._sha(pre_ingest._canonical([
            "candidate-v2-file-atomic", path.relative_to(source_root).as_posix(),
            pre_ingest._stat_signature(path.stat()),
        ]))
        if identity in source_ids:
            sources[identity] = path
    if set(sources) != source_ids:
        raise ValueError("continuation source videos unavailable")
    arrays: dict[str, np.ndarray] = {}
    manifest_rows: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "dev"):
        clips: list[np.ndarray] = []
        timestamps: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        for index, selection in enumerate(selected[split]):
            source_id = str(selection["source_id"])
            path = sources[source_id]
            if pre_ingest._candidate(path, source_root).candidate_id != source_id:
                raise ValueError("continuation source identity changed")
            label_us = cast(int, selection["label_timestamp_us"])
            clip, times = _actor_window_before_label(path, label_us)
            rows.append({
                **selection, "sample_id": f"{split}-{index:03d}",
                "label_id": JOYSTICK_CONTINUATION_ACTIONS.index(str(selection["action"])),
                "input_start_timestamp_us": int(times[0]),
                "input_end_timestamp_us": int(times[-1]),
                "actual_input_label_gap_us": label_us - int(times[-1]),
                "rgb_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
            })
            clips.append(clip)
            timestamps.append(times)
        arrays[f"{split}_rgb"] = np.stack(clips)
        arrays[f"{split}_label"] = np.asarray(
            [row["label_id"] for row in rows], dtype=np.int64
        )
        arrays[f"{split}_input_timestamp_us"] = np.stack(timestamps)
        arrays[f"{split}_label_timestamp_us"] = np.asarray(
            [row["label_timestamp_us"] for row in rows], dtype=np.int64
        )
        arrays[f"{split}_source_id"] = np.asarray(
            [row["source_id"] for row in rows], dtype="U64"
        )
        manifest_rows[split] = rows
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-continuation-", dir=output_dir.parent))
    dataset_path = staging / "joystick-continuation-grouped.npz"
    np.savez_compressed(
        dataset_path,
        train_rgb=arrays["train_rgb"], train_label=arrays["train_label"],
        train_input_timestamp_us=arrays["train_input_timestamp_us"],
        train_label_timestamp_us=arrays["train_label_timestamp_us"],
        train_source_id=arrays["train_source_id"],
        dev_rgb=arrays["dev_rgb"], dev_label=arrays["dev_label"],
        dev_input_timestamp_us=arrays["dev_input_timestamp_us"],
        dev_label_timestamp_us=arrays["dev_label_timestamp_us"],
        dev_source_id=arrays["dev_source_id"],
    )
    manifest: dict[str, object] = {
        "schema_version": "joystick-continuation-dataset-v1",
        "dataset": dataset_path.name, "dataset_sha256": _file_sha256(dataset_path),
        "action_order": JOYSTICK_CONTINUATION_ACTIONS,
        "train_sources": 16, "dev_sources": 5,
        "dev_source_ids": sorted(JOYSTICK_SCALE21_DEV_SOURCES),
        "samples": manifest_rows, "input_shape": [16, 128, 128, 3],
        "input_dtype": "uint8_rgb", "source_report_file_sha256": report_hashes,
        "continuation_audit_file_sha256": _file_sha256(audit_run / "report.json"),
        "target_rule": "third or later equal direction; two prior same-direction candidates",
        "stop_owner": "deterministic_router", "semantic_accuracy_verified": False,
        "continuation_training_allowed": True, "formal_training_allowed": False,
        "checkpoint_promotion_allowed": False, "video_dev_opened": False,
        "video_test_opened": False, "input_commands_sent": 0, "gpu_seconds": 0,
    }
    manifest["manifest_sha256"] = _object_sha256(manifest)
    (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    staging.rename(output_dir)
    return validate_joystick_continuation_dataset(output_dir)


def evaluate_joystick_persistence(
    samples: dict[str, list[dict[str, object]]], sessions: list[dict[str, object]],
) -> dict[str, object]:
    session_map = {str(row["session"]): row for row in sessions}
    splits: dict[str, object] = {}
    for split in ("train", "dev"):
        correct = 0
        class_counts = dict.fromkeys(JOYSTICK_CONTINUATION_ACTIONS, 0)
        class_correct = dict.fromkeys(JOYSTICK_CONTINUATION_ACTIONS, 0)
        for sample in samples[split]:
            source_id = str(sample["source_id"])
            session = session_map.get(source_id)
            if session is None:
                raise ValueError("persistence sample source is unbound")
            fraction = float(cast(float, sample["source_fraction"]))
            windows = cast(list[dict[str, object]], session["windows"])
            matching = [row for row in windows if float(cast(float, row["fraction"])) == fraction]
            if len(matching) != 1:
                raise ValueError("persistence source window is ambiguous")
            window = matching[0]
            index = cast(int, sample["source_frame_index"])
            predictions = cast(list[dict[str, object]], window["predictions"])
            times = list(map(int, cast(list[int], window["timestamp_us"])))
            if index < 2 or index >= len(predictions):
                raise ValueError("persistence sample index differs")
            expected = str(sample["action"])
            observed = [str(predictions[offset]["candidate_action"])
                        for offset in (index - 2, index - 1, index)]
            prior_times = list(map(
                int, cast(list[int], sample["prior_same_direction_timestamp_us"])
            ))
            if (observed != [expected] * 3
                    or prior_times != [times[index - 2], times[index - 1]]
                    or cast(int, sample["label_timestamp_us"]) != times[index]
                    or cast(int, sample["input_end_timestamp_us"]) >= times[index]
                    or abs(cast(int, sample["input_end_timestamp_us"]) - times[index - 1])
                    > 25_000):
                raise ValueError("persistence causal binding differs")
            baseline = observed[-2]
            class_counts[expected] += 1
            if baseline == expected:
                correct += 1
                class_correct[expected] += 1
        recalls = {
            action: class_correct[action] / class_counts[action] if class_counts[action] else 0.0
            for action in JOYSTICK_CONTINUATION_ACTIONS
        }
        splits[split] = {
            "samples": len(samples[split]), "correct": correct,
            "accuracy": correct / len(samples[split]), "class_support": class_counts,
            "recall": recalls, "macro_f1": 1.0 if correct == len(samples[split]) else None,
        }
    return {"splits": splits, "exact": all(
        cast(dict[str, object], row)["accuracy"] == 1.0 for row in splits.values()
    )}


def run_joystick_persistence_audit(
    dataset_dir: Path, continuation_audit_run: Path, output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise ValueError("persistence audit output exists")
    validate_joystick_continuation_dataset(dataset_dir)
    manifest = _load_bound_json(dataset_dir / "manifest.json", "manifest_sha256")
    audit = _load_bound_json(continuation_audit_run / "report.json", "report_sha256")
    if (audit.get("status") != "JOYSTICK_DIRECTION_CONTINUATION_SUPPORT_PASSED"
            or manifest.get("action_order") != list(JOYSTICK_CONTINUATION_ACTIONS)
            or manifest.get("stop_owner") != "deterministic_router"
            or manifest.get("continuation_audit_file_sha256") !=
            _file_sha256(continuation_audit_run / "report.json")):
        raise ValueError("persistence audit bindings differ")
    scale_run = continuation_audit_run.parent / "joystick-scale24-audit-v2-landscape-subset"
    sessions, report_hashes = _joystick_scale21_sessions(
        continuation_audit_run.parent, scale_run
    )
    if report_hashes != manifest["source_report_file_sha256"]:
        raise ValueError("persistence source reports differ")
    result = evaluate_joystick_persistence(
        cast(dict[str, list[dict[str, object]]], manifest["samples"]), sessions
    )
    from hok_agent.movement_mvp import StageAMovement, _movement_command

    commands = {
        action: _movement_command(cast(StageAMovement, action), cast(StageAMovement, action))
        for action in JOYSTICK_CONTINUATION_ACTIONS
    }
    exact = result["exact"] is True and set(commands.values()) == {"KEEP"}
    report: dict[str, object] = {
        "schema_version": "joystick-persistence-baseline-audit-v1",
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_file_sha256": _file_sha256(dataset_dir / "manifest.json"),
        "continuation_audit_file_sha256": _file_sha256(
            continuation_audit_run / "report.json"
        ),
        "movement_runtime_source_sha256": _file_sha256(Path(__file__).with_name("movement_mvp.py")),
        "baseline": "predict previous executed direction; request same direction",
        "clock_alignment": (
            "joystick sampling PTS and decoded Actor PTS may differ by <=25ms; both precede label"
        ),
        "executor_commands": commands, "result": result,
        "status": (
            "DETERMINISTIC_DIRECTION_PERSISTENCE_EXACT"
            if exact else "DETERMINISTIC_DIRECTION_PERSISTENCE_FAILED"
        ),
        "learned_continuation_recommended": False if exact else None,
        "continuation_checkpoint_allowed": False,
        "learned_direction_change_verified": False,
        "stop_owner": "deterministic_router", "model_runs": 0,
        "rgb_decodes": 0, "gpu_seconds": 0, "input_commands_sent": 0,
        "video_dev_opened": False, "video_test_opened": False,
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-persistence-", dir=output_dir.parent))
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return report


def run_joystick_materialize_pilot(
    source_root: Path, cohort_dir: Path, pre_ingest_path: Path,
    final_run: Path, overfit_report_path: Path, output_dir: Path,
    *, verify_only: bool = False, scale21: bool = False,
) -> dict[str, object]:
    if verify_only:
        return (
            validate_joystick_scale21_dataset(output_dir)
            if scale21 else validate_joystick_grouped_pilot(output_dir)
        )
    if output_dir.exists():
        raise ValueError("grouped joystick output exists")
    overfit = _load_bound_json(overfit_report_path, "report_sha256")
    if scale21:
        if (overfit.get("schema_version") != "joystick-grouped-pilot-report-v1"
                or overfit.get("passed") is not False
                or overfit.get("checkpoint_promotion_allowed") is not False
                or overfit.get("next_stage_allowed") is not False):
            raise ValueError("scale21 dataset requires frozen failed grouped pilot")
    elif (overfit.get("schema_version") != "joystick-overfit32-report-v1"
          or overfit.get("passed") is not True
          or overfit.get("checkpoint_promotion_allowed") is not False
          or overfit.get("generalization_verified") is not False):
        raise ValueError("grouped pilot requires passed diagnostic overfit")
    conclusion_path = final_run / (
        "conclusion.json" if scale21 else "cohort-conclusion.json"
    )
    conclusion = cast(dict[str, object], json.loads(conclusion_path.read_text()))
    if scale21:
        sessions, report_hashes = _joystick_scale21_sessions(final_run.parent, final_run)
        if (conclusion.get("status") != "JOYSTICK_SCALE24_LANDSCAPE_SUBSET_PASSED"
                or conclusion.get("dataset_materialization_allowed") is not True):
            raise ValueError("scale21 conclusion does not permit materialization")
        selected = select_joystick_grouped_pilot(
            sessions, dev_sources=JOYSTICK_SCALE21_DEV_SOURCES,
            expected_samples=(201, 48),
        )
    else:
        sessions, report_hashes = _joystick_dataset_sessions(final_run.parent)
        if (conclusion.get("status") != "JOYSTICK_8_TRAIN_SOURCE_WEAK_LABEL_SUPPORT_PASSED"
                or report_hashes != conclusion["bound_report_file_sha256"]):
            raise ValueError("grouped pilot source bindings differ")
        selected = select_joystick_grouped_pilot(sessions)
    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    source_ids = {str(row["source_id"]) for rows in selected.values() for row in rows}
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    if any(cohort.session_splits.get(source_id) != "train" for source_id in source_ids):
        raise ValueError("grouped pilot may use cohort train sources only")
    sources: dict[str, Path] = {}
    for path in pre_ingest._scan(source_root):
        identity = pre_ingest._sha(pre_ingest._canonical([
            "candidate-v2-file-atomic", path.relative_to(source_root).as_posix(),
            pre_ingest._stat_signature(path.stat()),
        ]))
        if identity in source_ids:
            sources[identity] = path
    if set(sources) != source_ids:
        raise ValueError("grouped source videos unavailable")
    arrays: dict[str, np.ndarray] = {}
    manifest_rows: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "dev"):
        clips: list[np.ndarray] = []
        times: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        for index, selection in enumerate(selected[split]):
            source_id = str(selection["source_id"])
            path = sources[source_id]
            if pre_ingest._candidate(path, source_root).candidate_id != source_id:
                raise ValueError("grouped source identity changed")
            label_us = cast(int, selection["label_timestamp_us"])
            clip, timestamps = _actor_window_before_label(path, label_us)
            rows.append({
                **selection, "sample_id": f"{split}-{index:03d}",
                "label_id": JOYSTICK_ACTIONS.index(str(selection["action"])),
                "input_start_timestamp_us": int(timestamps[0]),
                "input_end_timestamp_us": int(timestamps[-1]),
                "actual_input_label_gap_us": label_us - int(timestamps[-1]),
                "rgb_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
            })
            clips.append(clip)
            times.append(timestamps)
        arrays[f"{split}_rgb"] = np.stack(clips)
        arrays[f"{split}_label"] = np.asarray(
            [row["label_id"] for row in rows], dtype=np.int64
        )
        arrays[f"{split}_input_timestamp_us"] = np.stack(times)
        arrays[f"{split}_label_timestamp_us"] = np.asarray(
            [row["label_timestamp_us"] for row in rows], dtype=np.int64
        )
        arrays[f"{split}_source_id"] = np.asarray(
            [row["source_id"] for row in rows], dtype="U64"
        )
        manifest_rows[split] = rows
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-grouped-pilot-", dir=output_dir.parent))
    dataset_name = "joystick-scale21-grouped.npz" if scale21 else "joystick-grouped-pilot.npz"
    dataset_path = staging / dataset_name
    np.savez_compressed(
        dataset_path,
        train_rgb=arrays["train_rgb"], train_label=arrays["train_label"],
        train_input_timestamp_us=arrays["train_input_timestamp_us"],
        train_label_timestamp_us=arrays["train_label_timestamp_us"],
        train_source_id=arrays["train_source_id"],
        dev_rgb=arrays["dev_rgb"], dev_label=arrays["dev_label"],
        dev_input_timestamp_us=arrays["dev_input_timestamp_us"],
        dev_label_timestamp_us=arrays["dev_label_timestamp_us"],
        dev_source_id=arrays["dev_source_id"],
    )
    manifest: dict[str, object] = {
        "schema_version": (
            "joystick-scale21-grouped-dataset-v1"
            if scale21 else "joystick-grouped-pilot-dataset-v1"
        ),
        "dataset": dataset_path.name, "dataset_sha256": _file_sha256(dataset_path),
        "action_order": JOYSTICK_ACTIONS,
        "train_sources": 16 if scale21 else 6,
        "dev_sources": 5 if scale21 else 2,
        "dev_source_ids": sorted(
            JOYSTICK_SCALE21_DEV_SOURCES if scale21 else JOYSTICK_PILOT_DEV_SOURCES
        ),
        "samples": manifest_rows,
        "input_shape": [16, 128, 128, 3], "input_dtype": "uint8_rgb",
        "source_report_file_sha256": report_hashes,
        "cohort_conclusion_file_sha256": _file_sha256(conclusion_path),
        "overfit_report_file_sha256": _file_sha256(overfit_report_path),
        "split_policy": (
            "fixed source groups; one sample per stable direction run plus release STOP"
        ),
        "semantic_accuracy_verified": False, "pilot_training_allowed": True,
        "formal_training_allowed": False, "checkpoint_promotion_allowed": False,
        "video_dev_opened": False, "video_test_opened": False,
        "input_commands_sent": 0, "gpu_seconds": 0,
    }
    manifest["manifest_sha256"] = _object_sha256(manifest)
    (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    staging.rename(output_dir)
    return (
        validate_joystick_scale21_dataset(output_dir)
        if scale21 else validate_joystick_grouped_pilot(output_dir)
    )


def run_joystick_materialize32(
    source_root: Path, cohort_dir: Path, pre_ingest_path: Path,
    final_run: Path, output_dir: Path, *, verify_only: bool = False,
) -> dict[str, object]:
    if verify_only:
        return validate_joystick_overfit32(output_dir)
    if output_dir.exists():
        raise ValueError("joystick dataset output exists")
    conclusion_path = final_run / "cohort-conclusion.json"
    conclusion = cast(dict[str, object], json.loads(conclusion_path.read_text()))
    if (conclusion.get("status") != "JOYSTICK_8_TRAIN_SOURCE_WEAK_LABEL_SUPPORT_PASSED"
            or conclusion.get("diagnostic_dataset_materialization_allowed") is not True):
        raise ValueError("cohort conclusion does not allow diagnostic materialization")
    sessions, report_hashes = _joystick_dataset_sessions(final_run.parent)
    if report_hashes != conclusion["bound_report_file_sha256"]:
        raise ValueError("cohort report bindings differ")
    selected = select_joystick_overfit32(sessions)
    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    source_ids = {str(row["source_id"]) for row in selected}
    if any(cohort.session_splits.get(source_id) != "train" for source_id in source_ids):
        raise ValueError("dataset sources must all be train")
    sources: dict[str, Path] = {}
    for path in pre_ingest._scan(source_root):
        identity = pre_ingest._sha(pre_ingest._canonical([
            "candidate-v2-file-atomic", path.relative_to(source_root).as_posix(),
            pre_ingest._stat_signature(path.stat()),
        ]))
        if identity in source_ids:
            sources[identity] = path
    if set(sources) != source_ids:
        raise ValueError("dataset source videos unavailable")
    clips: list[np.ndarray] = []
    times: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for index, selection in enumerate(selected):
        source_id = str(selection["source_id"])
        path = sources[source_id]
        if pre_ingest._candidate(path, source_root).candidate_id != source_id:
            raise ValueError("dataset source identity changed")
        label_us = cast(int, selection["label_timestamp_us"])
        clip, timestamps = _actor_window_before_label(path, label_us)
        clip_hash = hashlib.sha256(clip.tobytes()).hexdigest()
        row = {**selection, "sample_id": f"joystick32-{index:02d}",
               "label_id": JOYSTICK_ACTIONS.index(str(selection["action"])),
               "input_start_timestamp_us": int(timestamps[0]),
               "input_end_timestamp_us": int(timestamps[-1]),
               "actual_input_label_gap_us": label_us - int(timestamps[-1]),
               "rgb_sha256": clip_hash}
        clips.append(clip)
        times.append(timestamps)
        rows.append(row)
    if len({row["rgb_sha256"] for row in rows}) != 32:
        raise ValueError("joystick Actor clips are duplicated")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-overfit32-", dir=output_dir.parent))
    dataset_path = staging / "joystick-overfit32.npz"
    np.savez_compressed(
        dataset_path, rgb=np.stack(clips),
        label=np.asarray([row["label_id"] for row in rows], dtype=np.int64),
        input_timestamp_us=np.stack(times),
        label_timestamp_us=np.asarray([row["label_timestamp_us"] for row in rows], dtype=np.int64),
    )
    manifest: dict[str, object] = {
        "schema_version": "joystick-overfit32-dataset-v1",
        "dataset": dataset_path.name, "dataset_sha256": _file_sha256(dataset_path),
        "samples": rows, "action_order": JOYSTICK_ACTIONS,
        "class_counts": {"STOP": 8, **dict.fromkeys(JOYSTICK_ACTIONS[1:], 3)},
        "input_shape": [16, 128, 128, 3], "input_dtype": "uint8_rgb",
        "input_policy": "16 frames before label PTS; lower-left 35% x bottom 55% zeroed",
        "source_report_file_sha256": report_hashes,
        "cohort_conclusion_file_sha256": _file_sha256(conclusion_path),
        "semantic_accuracy_verified": False, "training_allowed": False,
        "dev_frames_opened": 0, "test_frames_opened": 0,
        "input_commands_sent": 0, "gpu_seconds": 0,
    }
    manifest["manifest_sha256"] = _object_sha256(manifest)
    (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    staging.rename(output_dir)
    return validate_joystick_overfit32(output_dir)


def run_joystick_final_transfer(
    source_root: Path, cohort_dir: Path, pre_ingest_path: Path,
    extractor_run: Path, output_dir: Path,
) -> dict[str, object]:
    return _run_joystick_transfer(
        source_root, cohort_dir, pre_ingest_path, extractor_run, output_dir,
        source_ids=JOYSTICK_FINAL_TRANSFER_SOURCES,
        fractions=JOYSTICK_FINAL_TRANSFER_FRACTIONS,
        schema_version="joystick-final-train-transfer-v1",
    )


def run_joystick_scale24_audit(
    config_path: Path, source_root: Path, cohort_dir: Path, pre_ingest_path: Path,
    extractor_run: Path, output_dir: Path,
) -> dict[str, object]:
    config = _load_bound_json(config_path, "contract_sha256")
    sources = tuple(map(str, cast(list[str], config["source_ids"])))
    fractions = tuple(map(float, cast(list[float], config["fractions"])))
    if (
        config.get("schema_version") != "joystick-scale24-audit-contract-v1"
        or config.get("extractor_contract_sha256") !=
        "f687f5423235e918fe37728793d24f0ea73a33936fc46344429d38ad91ec7c7c"
        or len(sources) != 24 or list(sources) != sorted(set(sources))
        or fractions != (0.15, 0.35, 0.55, 0.75, 0.9)
        or config.get("frames_per_window") != 40
        or config.get("sample_period_us") != 100000
        or config.get("extractor_frozen") is not True
        or config.get("model_training_allowed") is not False
        or config.get("video_dev_allowed") is not False
        or config.get("video_test_allowed") is not False
        or config.get("device_input_allowed") is not False
    ):
        raise ValueError("scale24 audit contract differs")
    import av

    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    if any(cohort.session_splits.get(identity) != "train" for identity in sources):
        raise ValueError("scale24 sources must all be train")
    paths: dict[str, Path] = {}
    for path in pre_ingest._scan(source_root):
        identity = pre_ingest._sha(pre_ingest._canonical([
            "candidate-v2-file-atomic", path.relative_to(source_root).as_posix(),
            pre_ingest._stat_signature(path.stat()),
        ]))
        if identity in sources:
            paths[identity] = path
    if set(paths) != set(sources):
        raise ValueError("scale24 preflight sources unavailable")
    geometry: list[dict[str, object]] = []
    compatible: list[str] = []
    for identity in sources:
        descriptor, opened = pre_ingest._open_regular(paths[identity])
        with os.fdopen(descriptor, "rb") as handle, av.open(handle, mode="r") as container:
            stream = container.streams.video[0]
            rotation = pre_ingest._rotation(stream)
            accepted = stream.width > stream.height and rotation == 0
            geometry.append({"source_id": identity, "width": stream.width,
                             "height": stream.height, "rotation": rotation,
                             "landscape_compatible": accepted})
            if accepted:
                compatible.append(identity)
            pre_ingest._assert_unchanged(handle.fileno(), opened)
    if len(compatible) < 12:
        raise ValueError("scale24 has insufficient landscape-compatible sources")
    preflight: dict[str, object] = {
        "rule": "width > height and metadata rotation == 0",
        "requested_sources": len(sources), "compatible_sources": len(compatible),
        "excluded_sources": len(sources) - len(compatible), "replacement_sources": 0,
        "geometry": geometry,
    }
    return _run_joystick_transfer(
        source_root, cohort_dir, pre_ingest_path, extractor_run, output_dir,
        source_ids=tuple(compatible), fractions=fractions,
        schema_version="joystick-scale24-audit-v2-landscape-subset",
        selection_contract_sha256=cast(str, config["contract_sha256"]),
        selection_preflight=preflight,
    )


def run_joystick_coverage(
    source_root: Path, cohort_dir: Path, pre_ingest_path: Path,
    extractor_run: Path, output_dir: Path,
) -> dict[str, object]:
    import cv2
    from PIL import Image, ImageDraw

    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    cv2.setNumThreads(1)
    if output_dir.exists():
        raise ValueError("coverage output exists")
    frozen = _load_bound_json(extractor_run / "contract.json", "contract_sha256")
    prior = _load_bound_json(extractor_run / "report.json", "report_sha256")
    if (frozen["contract_sha256"] !=
            "f687f5423235e918fe37728793d24f0ea73a33936fc46344429d38ad91ec7c7c"
            or prior.get("contract_sha256") != frozen["contract_sha256"]
            or joystick_extractor_fingerprint() != JOYSTICK_V3_FINGERPRINT):
        raise ValueError("coverage requires the frozen v3 extractor")
    template_path = extractor_run / "train-templates.npz"
    if _file_sha256(template_path) != frozen["template_sha256"]:
        raise ValueError("frozen template hash differs")
    with np.load(template_path, allow_pickle=False) as arrays:
        templates = {key: arrays[key].copy() for key in arrays.files}
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    identity = next(k for k, v in NATIVE_PLAYER_SOURCES.items() if v == "train")
    if cohort.session_splits.get(identity) != "train":
        raise ValueError("coverage source must be train")
    selected: Path | None = None
    for path in pre_ingest._scan(source_root):
        key = pre_ingest._sha(pre_ingest._canonical([
            "candidate-v2-file-atomic", path.relative_to(source_root).as_posix(),
            pre_ingest._stat_signature(path.stat()),
        ]))
        if key == identity:
            selected = path
            break
    if selected is None or pre_ingest._candidate(selected, source_root).candidate_id != identity:
        raise ValueError("coverage source unavailable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-coverage-", dir=output_dir.parent))
    contract: dict[str, object] = {
        "extractor_contract_sha256": frozen["contract_sha256"],
        "extractor_fingerprint": JOYSTICK_V3_FINGERPRINT,
        "template_sha256": frozen["template_sha256"],
        "pre_ingest_sha256": _file_sha256(pre_ingest_path),
        "session": identity, "split": "train", "fractions": JOYSTICK_COVERAGE_FRACTIONS,
        "frames_per_window": 40, "sample_period_us": 100000,
        "training_allowed": False, "dev_frames_opened": 0, "test_frames_opened": 0,
    }
    contract["contract_sha256"] = _object_sha256(contract)
    (staging / "contract.json").write_bytes(_canonical(contract) + b"\n")
    windows: list[dict[str, object]] = []
    previous_end = -1
    for fraction in JOYSTICK_COVERAGE_FRACTIONS:
        arrays = _joystick_window(selected, fraction)
        frames, times = arrays["rgb"], arrays["timestamp_us"]
        if times[0] <= previous_end:
            raise ValueError("coverage windows overlap")
        previous_end = int(times[-1])
        predictions = extract_joystick_sequence(frames, templates,
                                                normalize_scale=True, geometric_base=True)
        # Retain QA only, not another full native-RGB cache.
        indices = list(map(int, np.linspace(0, len(frames) - 1, 12)))
        seen: set[str] = set()
        for i, prediction in enumerate(predictions):
            action = str(prediction["candidate_action"])
            if action != "unknown" and action not in seen:
                indices.append(i)
                seen.add(action)
        indices = sorted(set(indices))
        sheet = Image.new("RGB", (4*320, math.ceil(len(indices)/4)*240))
        for ordinal, index in enumerate(indices):
            row = predictions[index]
            image = Image.fromarray(frames[index])
            draw = ImageDraw.Draw(image)
            bx, by = cast(list[float], row["base_xy"])
            kx, ky = cast(list[float], row["knob_xy"])
            draw.ellipse((bx-10, by-10, bx+10, by+10), outline="yellow", width=3)
            draw.ellipse((kx-10, ky-10, kx+10, ky+10), outline="magenta", width=3)
            draw.line((bx, by, kx, ky), fill="yellow", width=3)
            image.thumbnail((320, 210))
            x, y = ordinal % 4*320, ordinal // 4*240
            sheet.paste(image, (x, y+26))
            ImageDraw.Draw(sheet).text((x, y), f"{index} {row['candidate_action']} "
                                      f"{times[index]/1e6:.2f}s", fill="white")
        qa_name = f"f{round(fraction*100):02d}-qa.png"
        sheet.save(staging / qa_name)
        windows.append({"fraction": fraction, "timestamp_us": times.tolist(),
                        "predictions": predictions, "qa": qa_name, "qa_indices": indices,
                        "qa_sha256": _file_sha256(staging / qa_name),
                        "frame_sha256": [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]})
    summary = joystick_coverage_summary(windows)
    report: dict[str, object] = {
        "schema_version": "joystick-train-coverage-v1",
        "contract_sha256": contract["contract_sha256"],
        "status": "QA_REQUIRED", "summary": summary, "windows": windows,
        "gpu_seconds": 0, "input_commands_sent": 0, "dev_frames_opened": 0,
        "test_frames_opened": 0, "training_allowed": False,
        "native_cache_written": False, "qa_policy": "uniform plus first occurrence of each class",
    }
    report["report_sha256"] = _object_sha256(report)
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return {k: v for k, v in report.items() if k != "windows"}


def _native_landscape_window(path: Path, *, start_fraction: float = 0.2) -> dict[str, np.ndarray]:
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
        start = int(stream.duration * start_fraction)
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


def _joystick_window(path: Path, fraction: float) -> dict[str, np.ndarray]:
    """Four seconds of native lower-left RGB, with decoded presentation timestamps."""
    import av

    from hok_agent import pre_ingest

    descriptor, opened = pre_ingest._open_regular(path)
    crops: list[np.ndarray] = []
    times: list[int] = []
    with os.fdopen(descriptor, "rb") as handle, av.open(handle, mode="r") as container:
        stream = container.streams.video[0]
        if (stream.duration is None or stream.time_base is None
                or stream.width <= stream.height or pre_ingest._rotation(stream) != 0):
            raise ValueError("joystick inspection requires unrotated landscape timestamps")
        start = int(stream.duration * fraction)
        start_us = round(start * stream.time_base * 1_000_000)
        container.seek(start, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("joystick frame lacks PTS")
            timestamp = round(frame.pts * stream.time_base * 1_000_000)
            if timestamp < start_us + len(times) * 100_000:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            # Broad lower-left region includes floating as well as fixed joystick bases.
            crops.append(rgb[round(height * 0.45):, :round(width * 0.35)].copy())
            times.append(timestamp)
            if len(times) == 40:
                break
        pre_ingest._assert_unchanged(handle.fileno(), opened)
    if len(times) != 40 or not np.all(np.diff(times) > 0):
        raise ValueError("incomplete joystick inspection window")
    return {"rgb": np.stack(crops), "timestamp_us": np.asarray(times, dtype=np.int64)}


def run_joystick_visibility(
    source_root: Path, cohort_dir: Path, pre_ingest_path: Path, output_dir: Path,
) -> dict[str, object]:
    from PIL import Image, ImageDraw

    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    if output_dir.exists():
        raise ValueError("joystick inspection output exists")
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    if any(cohort.session_splits.get(k) != v for k, v in NATIVE_PLAYER_SOURCES.items()):
        raise ValueError("joystick source split differs")
    sources: dict[str, Path] = {}
    # Resolve identities from stat metadata only; never open unselected/test containers.
    for path in pre_ingest._scan(source_root):
        identity = pre_ingest._sha(pre_ingest._canonical([
            "candidate-v2-file-atomic", path.relative_to(source_root).as_posix(),
            pre_ingest._stat_signature(path.stat()),
        ]))
        if identity in NATIVE_PLAYER_SOURCES:
            sources[identity] = path
    if set(sources) != set(NATIVE_PLAYER_SOURCES):
        raise ValueError("joystick selected sources unavailable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-visibility-", dir=output_dir.parent))
    rows: list[dict[str, object]] = []
    for identity, path in sorted(sources.items()):
        if pre_ingest._candidate(path, source_root).candidate_id != identity:
            raise ValueError("joystick source identity changed")
        for fraction in (0.1, 0.3, 0.6):
            arrays = _joystick_window(path, fraction)
            prefix = f"{identity[:8]}-f{round(fraction * 100):02d}"
            data = staging / f"{prefix}.npz"
            np.savez_compressed(data, rgb=arrays["rgb"], timestamp_us=arrays["timestamp_us"])
            sheet = Image.new("RGB", (4 * 320, 3 * 220))
            for ordinal, index in enumerate(np.linspace(0, 39, 12).astype(int)):
                image = Image.fromarray(arrays["rgb"][index])
                image.thumbnail((320, 195))
                x, y = ordinal % 4 * 320, ordinal // 4 * 220
                sheet.paste(image, (x, y + 22))
                ImageDraw.Draw(sheet).text(
                    (x, y), f"{index}: {arrays['timestamp_us'][index] / 1e6:.3f}s", fill="white"
                )
            qa = staging / f"{prefix}.png"
            sheet.save(qa)
            rows.append({"session": identity, "split": NATIVE_PLAYER_SOURCES[identity],
                         "fraction": fraction, "shape": list(arrays["rgb"].shape),
                         "data": data.name, "data_sha256": _file_sha256(data),
                         "qa": qa.name, "qa_sha256": _file_sha256(qa),
                         "max_pts_gap_us": int(np.diff(arrays["timestamp_us"]).max())})
    report: dict[str, object] = {
        "schema_version": "joystick-visibility-v1", "windows": rows,
        "roi_xyxy_normalized": [0, 0.45, 0.35, 1], "sample_period_us": 100000,
        "source_sha256": _file_sha256(Path(__file__)),
        "pre_ingest_sha256": _file_sha256(pre_ingest_path),
        "status": "VISUAL_INSPECTION_REQUIRED", "sampled_frames": 240,
        "labels_created": 0, "training_allowed": False, "gpu_seconds": 0,
        "test_containers_opened": 0, "input_commands_sent": 0,
        "actor_input_policy": "joystick pixels excluded from any future policy input",
    }
    report["report_sha256"] = _object_sha256(report)
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return report


JOYSTICK_EXTRACT_SETTINGS = {
    "calibration_source": "0667d97c-f30.npz", "calibration_frame": 39,
    "scale": 0.5, "base_radius": 94, "knob_radius": 28,
    "base_min_score": 0.35, "knob_min_score": 0.65,
    "minimum_peak_margin": 0.04, "minimum_contrast_ratio": 0.60,
    "stop_radius_native_pixels": 12, "maximum_displacement_native_pixels": 200,
    "stop_confirmation_frames": 2,
    "base_template_aggregation": "median_aligned_train_patches",
    "calibration_alignment_min_score": 0.15, "calibration_alignment_min_margin": 0.04,
}


def _joystick_signal(rgb: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    reduced = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return reduced - cv2.GaussianBlur(reduced, (0, 0), 2)


def _joystick_match(
    signal: np.ndarray, template: np.ndarray, mask: np.ndarray,
) -> tuple[tuple[float, float], float, float, float]:
    import cv2

    scores = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    best = float(scores[y, x])
    other = scores.copy()
    other[max(0, y - 12): y + 13, max(0, x - 12): x + 13] = -1
    margin = best - float(other.max())
    height, width = template.shape
    patch = signal[y:y + height, x:x + width]
    contrast = float(np.std(patch[mask > 0]) / max(float(np.std(template[mask > 0])), 1e-6))
    return (float(x + width // 2), float(y + height // 2)), best, margin, contrast


JOYSTICK_MARKER_BOXES = ((5, 41, 58, 131), (148, 184, 58, 131),
                         (58, 131, 5, 41), (58, 131, 148, 184))


def _joystick_geometric_base(
    signal: np.ndarray, template: np.ndarray,
) -> tuple[tuple[float, float], float, float, float]:
    """Require three of four independently normalized markers at a shared cross center."""
    import cv2

    height = signal.shape[0] - template.shape[0] + 1
    width = signal.shape[1] - template.shape[1] + 1
    maps = []
    for y0, y1, x0, x1 in JOYSTICK_MARKER_BOXES:
        marker = template[y0:y1, x0:x1]
        response = cv2.matchTemplate(signal, marker, cv2.TM_CCOEFF_NORMED)
        maps.append(np.nan_to_num(response[y0:y0 + height, x0:x0 + width],
                                  nan=-1, posinf=-1, neginf=-1))
    stacked = np.stack(maps)
    # The third-best score requires three independent marker agreements, tolerating one occlusion.
    scores = np.partition(stacked, 1, axis=0)[1]
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    best = float(scores[y, x])
    other = scores.copy()
    other[max(0, y - 12):y + 13, max(0, x - 12):x + 13] = -1
    selected = np.argsort(stacked[:, y, x])[-3:]
    contrasts = []
    for ordinal in selected:
        y0, y1, x0, x1 = JOYSTICK_MARKER_BOXES[int(ordinal)]
        contrast = np.std(signal[y + y0:y + y1, x + x0:x + x1]) / max(
            float(np.std(template[y0:y1, x0:x1])), 1e-6
        )
        contrasts.append(float(contrast))
    return (float(x + 94), float(y + 94)), best, best - float(other.max()), min(contrasts)


def calibrate_joystick_templates(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Train-only development reference with visible centered knob; center found by Hough."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 60,
                               param1=80, param2=40, minRadius=40, maxRadius=85)
    if circles is None or len(circles[0]) != 1:
        raise ValueError("train calibration requires a unique knob circle")
    x, y = np.rint(circles[0, 0, :2] * 0.5).astype(int)
    signal = _joystick_signal(rgb)
    base = signal[y - 94:y + 95, x - 94:x + 95].copy()
    knob = signal[y - 28:y + 29, x - 28:x + 29].copy()
    if base.shape != (189, 189) or knob.shape != (57, 57):
        raise ValueError("calibration templates outside crop")
    mask = np.zeros_like(base, dtype=np.uint8)
    mask[5:41, 58:131] = 1
    mask[148:184, 58:131] = 1
    mask[58:131, 5:41] = 1
    mask[58:131, 148:184] = 1
    yy, xx = np.indices(knob.shape)
    knob_mask = (((yy - 28)**2 + (xx - 28)**2) <= 26**2).astype(np.uint8)
    return {"base": base, "knob": knob, "base_mask": mask, "knob_mask": knob_mask}


JOYSTICK_SCALES = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.25)


def extract_joystick_sequence(
    frames: np.ndarray, templates: dict[str, np.ndarray], *, normalize_scale: bool = False,
    geometric_base: bool = False,
) -> list[dict[str, object]]:
    import cv2

    directions = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
    rows: list[dict[str, object]] = []
    centered = 0
    for frame in frames:
        candidates = []
        for scale in JOYSTICK_SCALES if normalize_scale else (1.0,):
            normalized = cv2.resize(frame, None, fx=1 / scale, fy=1 / scale,
                                    interpolation=cv2.INTER_LINEAR) if scale != 1 else frame
            signal = _joystick_signal(normalized)
            base, bs, bm, bc = (
                _joystick_geometric_base(signal, templates["base"]) if geometric_base else
                _joystick_match(signal, templates["base"], templates["base_mask"])
            )
            knob, ks, km, kc = _joystick_match(signal, templates["knob"], templates["knob_mask"])
            valid = (bs >= 0.35 and ks >= 0.65 and min(bm, km) >= 0.04
                     and min(bc, kc) >= 0.60 and math.dist(base, knob) * 2 <= 200)
            candidates.append(((valid, min(bs / 0.35, ks / 0.65)),
                               scale, base, bs, bm, bc, knob, ks, km, kc))
        _, scale, base, bs, bm, bc, knob, ks, km, kc = max(candidates, key=lambda c: c[0])
        dx, dy = (knob[0] - base[0]) * 2, (knob[1] - base[1]) * 2
        distance = math.hypot(dx, dy)
        reason = "observed"
        if bs < 0.35 or ks < 0.65:
            reason = "low_match"
        elif min(bm, km) < 0.04:
            reason = "ambiguous_match"
        elif min(bc, kc) < 0.60:
            reason = "low_contrast_or_occluded"
        elif distance > 200:
            reason = "displacement_outside_range"
        action = "unknown"
        if reason == "observed":
            if distance <= 12:
                centered += 1
                action = "STOP" if centered >= 2 else "unknown"
                reason = "center_confirmed" if centered >= 2 else "center_confirming"
            else:
                centered = 0
                action = directions[int(math.floor((math.atan2(dy, dx) + math.pi / 8)
                                                   / (math.pi / 4))) % 8]
        else:
            centered = 0
        rows.append({"base_xy": [v * 2 * scale for v in base],
                     "knob_xy": [v * 2 * scale for v in knob],
                     "offset_xy": [dx * scale, dy * scale],
                     "normalized_offset_xy": [dx, dy], "control_scale": scale,
                     "candidate_action": action, "reason": reason,
                     "base_score": bs, "knob_score": ks, "base_margin": bm, "knob_margin": km,
                     "base_contrast_ratio": bc, "knob_contrast_ratio": kc,
                     "confidence": min(max(bs, 0), max(ks, 0)),
                     "confidence_calibrated": False})
    return rows


def joystick_scale_regression(
    frames: np.ndarray, templates: dict[str, np.ndarray],
    *, geometric_base: bool = False,
) -> dict[str, object]:
    """Geometric consistency on train transforms, not semantic label accuracy."""
    import cv2

    results: list[dict[str, object]] = []
    for index in (3, 17, 39):
        original = frames[index]
        reference = extract_joystick_sequence(
            np.stack([original] * 2), templates, geometric_base=geometric_base
        )[-1]
        for scale in (0.75, 1.0, 1.25):
            translation = (24 * scale, 16 * scale)
            matrix = np.asarray(
                [[scale, 0, translation[0]], [0, scale, translation[1]]], np.float32
            )
            canvas = (math.ceil(original.shape[1] * max(1, scale) + 48),
                      math.ceil(original.shape[0] * max(1, scale) + 32))
            transformed = cv2.warpAffine(original, matrix, canvas,
                                         borderValue=(64, 64, 64))
            row = extract_joystick_sequence(np.stack([transformed] * 2), templates,
                                            normalize_scale=True, geometric_base=geometric_base)[-1]
            errors = [math.dist(cast(list[float], row[key]),
                               [v * scale + d for v, d in zip(
                                   cast(list[float], reference[key]), translation, strict=True)])
                      for key in ("base_xy", "knob_xy")]
            same_action = row["candidate_action"] == reference["candidate_action"]
            reference_known = reference["candidate_action"] != "unknown"
            results.append({"train_frame": index, "scale": scale,
                            "reference_action": reference["candidate_action"],
                            "predicted_action": row["candidate_action"], "reason": row["reason"],
                            "base_error_pixels": errors[0], "knob_error_pixels": errors[1],
                            "action_consistent": same_action,
                            "passed": max(errors) <= 8 and reference_known and same_action})
    return {"cases": results, "passed": all(r["passed"] for r in results),
            "meaning": "train-derived geometric consistency, not independent accuracy"}


def run_joystick_extraction(
    source_run: Path, output_dir: Path, *, normalize_scale: bool = False,
    geometric_base: bool = False,
) -> dict[str, object]:
    import cv2
    from PIL import Image, ImageDraw

    cv2.setNumThreads(1)
    normalize_scale = normalize_scale or geometric_base
    source = _load_bound_json(source_run / "report.json", "report_sha256")
    if source.get("schema_version") != "joystick-visibility-v1":
        raise ValueError("extraction requires visibility cache")
    windows = cast(list[dict[str, object]], source["windows"])
    if output_dir.exists():
        raise ValueError("joystick extraction output exists")
    seed = next(w for w in windows if w["data"] == "0667d97c-f30.npz")
    if seed["split"] != "train":
        raise ValueError("calibration must be train")

    def read_window(row: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
        path = source_run / str(row["data"])
        if path.name != row["data"] or path.is_symlink():
            raise ValueError("invalid joystick cache path")
        if _file_sha256(path) != row["data_sha256"]:
            raise ValueError("joystick cache hash differs")
        with np.load(path, allow_pickle=False) as arrays:
            return arrays["rgb"].copy(), arrays["timestamp_us"].copy()

    frames, _ = read_window(seed)
    templates = calibrate_joystick_templates(frames[39])
    patches: list[np.ndarray] = []
    for window in windows:
        if window["split"] != "train":
            continue
        frames, _ = read_window(window)
        for frame in frames:
            signal = _joystick_signal(frame)
            center, score, margin, _ = _joystick_match(
                signal, templates["base"], templates["base_mask"]
            )
            x, y = map(int, center)
            if score >= 0.15 and margin >= 0.04:
                patches.append(signal[y - 94:y + 95, x - 94:x + 95])
    if len(patches) < 16:
        raise ValueError("insufficient aligned train base patches")
    templates["base"] = np.median(np.stack(patches), axis=0).astype(np.float32)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-extraction-", dir=output_dir.parent))
    template_path = staging / "train-templates.npz"
    np.savez_compressed(template_path, base=templates["base"], knob=templates["knob"],
                        base_mask=templates["base_mask"], knob_mask=templates["knob_mask"])
    # Freeze before reading any dev frame; no selection based on dev.
    frozen: dict[str, object] = {"settings": JOYSTICK_EXTRACT_SETTINGS,
                                "scales": JOYSTICK_SCALES if normalize_scale else (1.0,),
                                "normalize_scale": normalize_scale,
                                "geometric_base": geometric_base,
                                "marker_boxes": JOYSTICK_MARKER_BOXES if geometric_base else None,
                                "dev_usage": "previously inspected regression set",
                                "train_calibration_patches": len(patches),
                                "template_sha256": _file_sha256(template_path),
                                "source_report_sha256": source["report_sha256"],
                                "implementation_sha256": _file_sha256(Path(__file__))}
    frozen["contract_sha256"] = _object_sha256(frozen)
    (staging / "contract.json").write_bytes(_canonical(frozen) + b"\n")
    scale_check: dict[str, object] | None = None
    if normalize_scale:
        frames, _ = read_window(seed)
        scale_check = joystick_scale_regression(frames, templates, geometric_base=geometric_base)
        (staging / "scale-regression.json").write_bytes(_canonical(scale_check) + b"\n")
        if not scale_check["passed"]:
            failure: dict[str, object] = {
                "status": "SYNTHETIC_SCALE_REGRESSION_FAILED", "training_allowed": False,
                "dev_frames_opened": 0, "scale_regression": scale_check,
                "contract_sha256": frozen["contract_sha256"],
            }
            failure["report_sha256"] = _object_sha256(failure)
            (staging / "report.json").write_bytes(_canonical(failure) + b"\n")
            staging.rename(output_dir)
            return failure
    results: list[dict[str, object]] = []
    for window in sorted(windows, key=lambda w: (w["split"] != "train", str(w["data"]))):
        if window["split"] not in {"train", "dev"}:
            raise ValueError("test window prohibited")
        frames, times = read_window(window)
        predictions = extract_joystick_sequence(
            frames, templates, normalize_scale=normalize_scale, geometric_base=geometric_base
        )
        sheet = Image.new("RGB", (4 * 320, 3 * 240))
        for ordinal, index in enumerate(np.linspace(0, len(frames) - 1, 12).astype(int)):
            row = predictions[index]
            image = Image.fromarray(frames[index])
            draw = ImageDraw.Draw(image)
            bx, by = cast(list[float], row["base_xy"])
            kx, ky = cast(list[float], row["knob_xy"])
            draw.ellipse((bx - 10, by - 10, bx + 10, by + 10), outline="yellow", width=3)
            draw.ellipse((kx - 10, ky - 10, kx + 10, ky + 10), outline="magenta", width=3)
            draw.line((bx, by, kx, ky), fill="yellow", width=3)
            image.thumbnail((320, 210))
            x, y = ordinal % 4 * 320, ordinal // 4 * 240
            sheet.paste(image, (x, y + 28))
            ImageDraw.Draw(sheet).text((x, y), f"{index} {row['candidate_action']} "
                                      f"{row['reason']}", fill="white")
        qa_name = Path(str(window["data"])).stem + "-extracted.png"
        sheet.save(staging / qa_name)
        results.append({"session": window["session"], "split": window["split"],
                        "data": window["data"], "timestamp_us": times.tolist(),
                        "predictions": predictions, "qa": qa_name,
                        "counts": dict(Counter(str(r["candidate_action"]) for r in predictions)),
                        "qa_sha256": _file_sha256(staging / qa_name)})
    report: dict[str, object] = {
        "schema_version": (
            "joystick-extraction-v3-geometry" if geometric_base else
            "joystick-extraction-v2-scale" if normalize_scale else "joystick-extraction-v1"
        ),
        "contract_sha256": frozen["contract_sha256"], "scale_regression": scale_check,
        "windows": results, "status": "QA_REQUIRED", "training_allowed": False,
        "automatic_accuracy_verified": False, "candidate_labels_only": True,
        "test_frames_opened": 0, "gpu_seconds": 0, "input_commands_sent": 0,
        "timing": "visible UI state at frame PTS; not inferred preceding action or intention",
    }
    report["report_sha256"] = _object_sha256(report)
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return {k: v for k, v in report.items() if k != "windows"}


def run_native_player_pilot(
    source_root: Path,
    cohort_dir: Path,
    pre_ingest_path: Path,
    output_dir: Path,
    *,
    train_visibility_scan: bool = False,
) -> dict[str, object]:
    """Materialize only the two identity-bound landscape development windows."""
    from PIL import Image, ImageDraw

    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    if output_dir.exists():
        raise ValueError("native pilot output already exists")
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    selected_sources = {
        identity: split
        for identity, split in NATIVE_PLAYER_SOURCES.items()
        if not train_visibility_scan or split == "train"
    }
    if any(cohort.session_splits.get(key) != split for key, split in selected_sources.items()):
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
        if identity in selected_sources:
            sources[identity] = path
    if set(sources) != set(selected_sources):
        raise ValueError("native player source identities unavailable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    rows: list[dict[str, object]] = []
    fractions = (0.05, 0.10, 0.15) if train_visibility_scan else (0.2,)
    jobs = [
        (identity, path, fraction)
        for identity, path in sorted(sources.items())
        for fraction in fractions
    ]
    for identity, path, fraction in jobs:
        if pre_ingest._candidate(path, source_root).candidate_id != identity:
            raise ValueError("native source identity changed")
        arrays = _native_landscape_window(path, start_fraction=fraction)
        positions = green_ring_track(arrays["minimap_rgb"])
        prefix = identity[:8] + (f"-f{round(fraction * 100):02d}" if train_visibility_scan else "")
        basename = prefix + "-native-window.npz"
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
        qa_name = prefix + "-minimap-qa.png"
        sheet.save(staging / qa_name)
        main_name = prefix + "-main-qa.png"
        for index in range(16):
            x, y = index % 4 * 256, index // 4 * 276
            sheet.paste(Image.fromarray(arrays["main_rgb"][index]), (x, y + 20))
        sheet.save(staging / main_name)
        rows.append(
            {
                "session_hash": identity,
                "start_fraction": fraction,
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
        "status": "NATIVE_TRAIN_VISIBILITY_SCAN_QA_ONLY"
        if train_visibility_scan
        else "NATIVE_LANDSCAPE_WINDOWS_MATERIALIZED_QA_ONLY",
        "cohort_sha256": cohort.cohort_sha256,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "sessions": rows,
        "start_fractions": list(fractions),
        "train_visibility_scan": train_visibility_scan,
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


def run_native_anchor_cohort_audit(
    source_root: Path,
    cohort_dir: Path,
    pre_ingest_path: Path,
    output_dir: Path,
    *,
    prior_report_path: Path | None = None,
) -> dict[str, object]:
    """Audit fixed weak-anchor coverage across 8 train and 4 dev source sessions."""
    from PIL import Image, ImageDraw

    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    if output_dir.exists():
        raise ValueError("native anchor audit output already exists")
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    source_bindings = (
        NATIVE_ANCHOR_REPAIR_SOURCES
        if prior_report_path is not None
        else NATIVE_ANCHOR_AUDIT_SOURCES
    )
    prior: dict[str, object] | None = None
    if prior_report_path is not None:
        prior = _load_bound_json(prior_report_path, "report_sha256")
        prior_sessions = cast(list[dict[str, object]], prior.get("sessions"))
        if (
            prior.get("schema_version") != "native-weak-visual-anchor-cohort-audit-v1"
            or prior.get("status") != "WEAK_VISUAL_ANCHOR_COHORT_INSUFFICIENT"
            or prior.get("checks")
            != {
                "train_supported_sessions": True,
                "dev_supported_sessions": False,
                "train_confirmed_frames": True,
                "dev_confirmed_frames": True,
                "session_split_isolation": True,
            }
            or {str(row["session_hash"]): row["split"] for row in prior_sessions}
            != NATIVE_ANCHOR_AUDIT_SOURCES
        ):
            raise ValueError("native anchor repair prior report differs")
    if any(
        cohort.session_splits.get(identity) != split for identity, split in source_bindings.items()
    ):
        raise ValueError("native anchor audit split binding differs")
    sources: dict[str, Path] = {}
    for path in pre_ingest._scan(source_root):
        identity = pre_ingest._candidate(path, source_root).candidate_id
        if identity in source_bindings:
            sources[identity] = path
    if set(sources) != set(source_bindings):
        raise ValueError("native anchor audit source identities unavailable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    sessions: list[dict[str, object]] = []
    for identity, path in sorted(sources.items()):
        maps: list[np.ndarray] = []
        timestamps: list[np.ndarray] = []
        window_ids: list[np.ndarray] = []
        native_hashes: list[np.ndarray] = []
        window_rows: list[dict[str, object]] = []
        map_sheet = Image.new("RGB", (8 * 128, 6 * 144))
        main_sheet = Image.new("RGB", (3 * 256, 3 * 276))
        map_draw = ImageDraw.Draw(map_sheet)
        main_draw = ImageDraw.Draw(main_sheet)
        for window_id, fraction in enumerate(NATIVE_ANCHOR_AUDIT_FRACTIONS):
            arrays = _native_landscape_window(path, start_fraction=fraction)
            candidates = [green_ring_candidates(frame) for frame in arrays["minimap_rgb"]]
            positions, reasons = _confirm_ring_candidates(candidates)
            maps.append(arrays["minimap_rgb"])
            timestamps.append(arrays["timestamp_us"])
            window_ids.append(np.full(16, window_id, dtype=np.int8))
            native_hashes.append(arrays["native_crop_sha256"])
            for frame_index, frame in enumerate(arrays["minimap_rgb"]):
                image = Image.fromarray(frame).resize((128, 128))
                x = frame_index % 8 * 128
                y = (window_id * 2 + frame_index // 8) * 144 + 16
                map_sheet.paste(image, (x, y))
                map_draw.text((x + 2, y - 14), f"w{window_id}:{frame_index}")
                if positions[frame_index] is not None:
                    py, px = cast(tuple[int, int], positions[frame_index])
                    map_draw.ellipse(
                        (x + px // 2 - 8, y + py // 2 - 8, x + px // 2 + 8, y + py // 2 + 8),
                        outline="magenta",
                        width=2,
                    )
            for column, frame_index in enumerate((0, 8, 15)):
                x, y = column * 256, window_id * 276 + 20
                main_sheet.paste(Image.fromarray(arrays["main_rgb"][frame_index]), (x, y))
                main_draw.text((x + 2, y - 18), f"w{window_id}:{frame_index}")
            window_rows.append(
                {
                    "window_id": window_id,
                    "start_fraction": fraction,
                    "start_us": int(arrays["timestamp_us"][0]),
                    "end_us": int(arrays["timestamp_us"][-1]),
                    "confirmed_frames": sum(position is not None for position in positions),
                    "candidate_frames": sum(bool(row) for row in candidates),
                    "candidate_total": sum(map(len, candidates)),
                    "reason_counts": dict(Counter(reasons)),
                    "confirmed_positions_yx": positions,
                }
            )
        data_name = identity[:8] + "-anchor-windows.npz"
        np.savez_compressed(
            staging / data_name,
            minimap_rgb=np.concatenate(maps),
            timestamp_us=np.concatenate(timestamps),
            window_id=np.concatenate(window_ids),
            native_crop_sha256=np.concatenate(native_hashes),
        )
        map_name = identity[:8] + "-anchor-map-qa.png"
        main_name = identity[:8] + "-anchor-main-qa.png"
        map_sheet.save(staging / map_name)
        main_sheet.save(staging / main_name)
        supported = any(cast(int, row["confirmed_frames"]) >= 8 for row in window_rows)
        sessions.append(
            {
                "session_hash": identity,
                "split": source_bindings[identity],
                "windows": window_rows,
                "supported_session": supported,
                "confirmed_frames": sum(cast(int, row["confirmed_frames"]) for row in window_rows),
                "artifacts": [
                    {"basename": name, "sha256": _file_sha256(staging / name)}
                    for name in (data_name, map_name, main_name)
                ],
            }
        )
    combined_sessions = (
        cast(list[dict[str, object]], prior["sessions"]) + sessions
        if prior is not None
        else sessions
    )
    support = {
        split: {
            "sessions": sum(row["split"] == split for row in combined_sessions),
            "supported_sessions": sum(
                row["split"] == split and bool(row["supported_session"])
                for row in combined_sessions
            ),
            "confirmed_frames": sum(
                cast(int, row["confirmed_frames"])
                for row in combined_sessions
                if row["split"] == split
            ),
        }
        for split in ("train", "dev")
    }
    checks = {
        "train_supported_sessions": support["train"]["supported_sessions"] >= 6,
        "dev_supported_sessions": support["dev"]["supported_sessions"] >= 3,
        "train_confirmed_frames": support["train"]["confirmed_frames"] >= 128,
        "dev_confirmed_frames": support["dev"]["confirmed_frames"] >= 48,
        "session_split_isolation": not (
            {row["session_hash"] for row in combined_sessions if row["split"] == "train"}
            & {row["session_hash"] for row in combined_sessions if row["split"] == "dev"}
        ),
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": (
            "native-weak-visual-anchor-cohort-audit-v2"
            if prior is not None
            else "native-weak-visual-anchor-cohort-audit-v1"
        ),
        "status": "WEAK_VISUAL_ANCHOR_COHORT_SUPPORTED_QA_ONLY"
        if passed
        else "WEAK_VISUAL_ANCHOR_COHORT_INSUFFICIENT",
        "cohort_sha256": cohort.cohort_sha256,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "source_sessions": len(combined_sessions),
        "new_source_sessions": len(sessions),
        "windows": len(combined_sessions) * len(NATIVE_ANCHOR_AUDIT_FRACTIONS),
        "new_windows": len(sessions) * len(NATIVE_ANCHOR_AUDIT_FRACTIONS),
        "frames": len(combined_sessions) * len(NATIVE_ANCHOR_AUDIT_FRACTIONS) * 16,
        "new_frames": len(sessions) * len(NATIVE_ANCHOR_AUDIT_FRACTIONS) * 16,
        "fractions": list(NATIVE_ANCHOR_AUDIT_FRACTIONS),
        "selection_note": (
            "four new dev sources selected in anonymous hash order after v1 failure; detector, "
            "fractions, support definition and v1 evidence unchanged"
            if prior is not None
            else "fixed after a separate 15-percent exploratory preflight; formal windows "
            "exclude that fraction and are not a random benchmark"
        ),
        "prior_report_file_sha256": (
            _file_sha256(prior_report_path) if prior_report_path is not None else None
        ),
        "prior_report_sha256": prior["report_sha256"] if prior is not None else None,
        "support": support,
        "checks": checks,
        "sessions": sessions,
        "weak_target": "green-ring visual anchor or unknown",
        "controlled_player_identity_verified": False,
        "hero_identity_verified": False,
        "policy_action_labels_created": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "test_frames_decoded": 0,
        "raw_source_locators_persisted": False,
        "model_runs": 0,
        "gpu_seconds": 0,
        "input_commands_sent": 0,
    }
    report["report_sha256"] = _object_sha256(report)
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return report


def audit_native_anchor_counterfactual(
    source_run: Path,
    repair_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Define balanced visual-relation targets without treating them as executed actions."""
    if output_dir.exists():
        raise ValueError("native anchor counterfactual output already exists")
    source = _load_bound_json(source_run / "report.json", "report_sha256")
    repair = _load_bound_json(repair_report_path, "report_sha256")
    if (
        source.get("schema_version") != "native-weak-visual-anchor-cohort-audit-v1"
        or source.get("status") != "WEAK_VISUAL_ANCHOR_COHORT_INSUFFICIENT"
        or repair.get("schema_version") != "native-weak-visual-anchor-cohort-audit-v2"
        or repair.get("status") != "WEAK_VISUAL_ANCHOR_COHORT_SUPPORTED_QA_ONLY"
        or repair.get("prior_report_file_sha256") != _file_sha256(source_run / "report.json")
        or repair.get("prior_report_sha256") != source.get("report_sha256")
    ):
        raise ValueError("native anchor counterfactual source reports differ")
    repair_run = repair_report_path.parent
    actions = ("STOP", "N", "S", "W", "E", "NW", "NE", "SW", "SE")
    groups: list[dict[str, object]] = []
    opened: list[dict[str, object]] = []
    for audit, directory in ((source, source_run), (repair, repair_run)):
        for session in cast(list[dict[str, object]], audit["sessions"]):
            identity = str(session["session_hash"])
            data_name = identity[:8] + "-anchor-windows.npz"
            artifact = next(
                item
                for item in cast(list[dict[str, object]], session["artifacts"])
                if item["basename"] == data_name
            )
            path = directory / data_name
            if path.is_symlink() or _file_sha256(path) != artifact["sha256"]:
                raise ValueError("native anchor counterfactual cached data differs")
            with np.load(path, allow_pickle=False) as arrays:
                frames = arrays["minimap_rgb"]
                window_ids = arrays["window_id"]
                timestamps = arrays["timestamp_us"]
                if (
                    frames.shape != (48, 256, 256, 3)
                    or frames.dtype != np.uint8
                    or window_ids.shape != (48,)
                    or timestamps.shape != (48,)
                ):
                    raise ValueError("native anchor counterfactual cached arrays differ")
                for window in cast(list[dict[str, object]], session["windows"]):
                    window_id = cast(int, window["window_id"])
                    indices = np.flatnonzero(window_ids == window_id)
                    if len(indices) != 16 or not np.all(np.diff(timestamps[indices]) > 0):
                        raise ValueError("native anchor counterfactual window differs")
                    positions = green_ring_track(frames[indices])
                    expected = [
                        tuple(item) if item is not None else None
                        for item in cast(list[list[int] | None], window["confirmed_positions_yx"])
                    ]
                    if positions != expected:
                        raise ValueError(
                            "native anchor counterfactual tracker reproduction differs"
                        )
                    anchor = positions[-1]
                    goals = (
                        {
                            action: _counterfactual_goal(
                                anchor,
                                action,
                                24,
                                7,
                                canvas_size=256,
                            )
                            for action in actions
                        }
                        if anchor is not None
                        else {}
                    )
                    eligible = (
                        cast(int, window["confirmed_frames"]) >= 8
                        and anchor is not None
                        and all(goal is not None for goal in goals.values())
                    )
                    if eligible:
                        groups.append(
                            {
                                "group_id": f"{identity}:{window_id}",
                                "session_hash": identity,
                                "split": session["split"],
                                "window_id": window_id,
                                "source_npz_sha256": artifact["sha256"],
                                "source_indices": indices.tolist(),
                                "anchor_yx": anchor,
                                "targets_yx": goals,
                                "labels": list(actions),
                            }
                        )
            opened.append(
                {
                    "session_hash": identity,
                    "split": session["split"],
                    "basename": data_name,
                    "sha256": artifact["sha256"],
                }
            )
    support = {
        split: {
            "sessions": len({row["session_hash"] for row in groups if row["split"] == split}),
            "groups": sum(row["split"] == split for row in groups),
            "samples": sum(row["split"] == split for row in groups) * len(actions),
            "per_action": {
                action: sum(row["split"] == split for row in groups) for action in actions
            },
        }
        for split in ("train", "dev")
    }
    train_sessions = {row["session_hash"] for row in groups if row["split"] == "train"}
    dev_sessions = {row["session_hash"] for row in groups if row["split"] == "dev"}
    checks = {
        "train_sessions": cast(int, support["train"]["sessions"]) >= 6,
        "train_groups": cast(int, support["train"]["groups"]) >= 8,
        "dev_sessions": cast(int, support["dev"]["sessions"]) >= 4,
        "dev_groups": cast(int, support["dev"]["groups"]) >= 6,
        "balanced_nine_actions": all(
            len(set(cast(dict[str, int], support[split]["per_action"]).values())) == 1
            for split in ("train", "dev")
        ),
        "session_split_isolation": not (train_sessions & dev_sessions),
        "group_ids_unique": len({row["group_id"] for row in groups}) == len(groups),
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": "native-weak-anchor-counterfactual-data-audit-v1",
        "status": "WEAK_ANCHOR_COUNTERFACTUAL_DATA_SUPPORTED"
        if passed
        else "WEAK_ANCHOR_COUNTERFACTUAL_DATA_INSUFFICIENT",
        "source_report_sha256": source["report_sha256"],
        "source_report_file_sha256": _file_sha256(source_run / "report.json"),
        "repair_report_sha256": repair["report_sha256"],
        "repair_report_file_sha256": _file_sha256(repair_report_path),
        "implementation_sha256": _file_sha256(Path(__file__)),
        "sequence_frames": 16,
        "anchor_frame": 15,
        "goal_distance_pixels": 24,
        "goal_ring_radius_pixels": 7,
        "canvas_size": 256,
        "action_order": list(actions),
        "support": support,
        "checks": checks,
        "groups": groups,
        "opened_cached_files": opened,
        "synthetic_relation_labels_defined": passed,
        "relation_diagnostic_training_allowed": passed,
        "movement_policy_training_allowed": False,
        "executed_action_labels_created": False,
        "controlled_player_identity_verified": False,
        "hero_identity_verified": False,
        "test_frames_read": 0,
        "video_frames_decoded": 0,
        "input_commands_sent": 0,
        "model_runs": 0,
        "gpu_seconds": 0,
        "claim_scope": (
            "counterfactual direction relative to a weak green-ring anchor; not human action, "
            "game-world position, controlled-player identity, or navigation performance"
        ),
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_bytes(_canonical(report) + b"\n")
    return report


def materialize_native_anchor_counterfactual(
    audit_report_path: Path,
    source_run: Path,
    repair_run: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Store source windows once and a reproducible 153-row relation index."""
    if output_dir.exists():
        raise ValueError("native anchor dataset output already exists")
    audit = _load_bound_json(audit_report_path, "report_sha256")
    if (
        audit.get("schema_version") != "native-weak-anchor-counterfactual-data-audit-v1"
        or audit.get("status") != "WEAK_ANCHOR_COUNTERFACTUAL_DATA_SUPPORTED"
        or audit.get("relation_diagnostic_training_allowed") is not True
        or audit.get("movement_policy_training_allowed") is not False
    ):
        raise ValueError("native anchor dataset audit differs")
    directories = (source_run, repair_run)
    source_clips: list[np.ndarray] = []
    group_ids: list[str] = []
    session_hashes: list[str] = []
    splits: list[str] = []
    anchors: list[tuple[int, int]] = []
    logical_group: list[int] = []
    labels: list[int] = []
    targets: list[tuple[int, int]] = []
    source_hashes: list[str] = []
    actions = cast(list[str], audit["action_order"])
    for group_index, group in enumerate(cast(list[dict[str, object]], audit["groups"])):
        session_hash = str(group["session_hash"])
        basename = session_hash[:8] + "-anchor-windows.npz"
        expected_hash = str(group["source_npz_sha256"])
        matches = [path / basename for path in directories if (path / basename).is_file()]
        if (
            len(matches) != 1
            or matches[0].is_symlink()
            or _file_sha256(matches[0]) != expected_hash
        ):
            raise ValueError("native anchor dataset source cache differs")
        indices = np.asarray(cast(list[int], group["source_indices"]), dtype=np.int64)
        with np.load(matches[0], allow_pickle=False) as arrays:
            clip = arrays["minimap_rgb"][indices]
        if clip.shape != (16, 256, 256, 3) or clip.dtype != np.uint8:
            raise ValueError("native anchor dataset source clip differs")
        source_clips.append(clip.copy())
        group_ids.append(str(group["group_id"]))
        session_hashes.append(session_hash)
        splits.append(str(group["split"]))
        anchor_values = cast(list[int], group["anchor_yx"])
        if len(anchor_values) != 2:
            raise ValueError("native anchor dataset anchor differs")
        anchors.append((anchor_values[0], anchor_values[1]))
        source_hashes.append(hashlib.sha256(clip.tobytes()).hexdigest())
        group_targets = cast(dict[str, list[int]], group["targets_yx"])
        if cast(list[str], group["labels"]) != actions:
            raise ValueError("native anchor dataset action order differs")
        for label, action in enumerate(actions):
            logical_group.append(group_index)
            labels.append(label)
            target_values = group_targets[action]
            if len(target_values) != 2:
                raise ValueError("native anchor dataset target differs")
            targets.append((target_values[0], target_values[1]))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    dataset_path = staging / "weak-anchor-counterfactual.npz"
    np.savez_compressed(
        dataset_path,
        source_clips=np.stack(source_clips),
        group_id=np.asarray(group_ids, dtype="U66"),
        session_hash=np.asarray(session_hashes, dtype="U64"),
        split=np.asarray(splits, dtype="U5"),
        anchor_yx=np.asarray(anchors, dtype=np.int16),
        source_clip_sha256=np.asarray(source_hashes, dtype="U64"),
        sample_group_index=np.asarray(logical_group, dtype=np.int16),
        sample_label=np.asarray(labels, dtype=np.int8),
        sample_target_yx=np.asarray(targets, dtype=np.int16),
    )
    split_counts = {
        split: {
            "groups": splits.count(split),
            "samples": sum(splits[index] == split for index in logical_group),
        }
        for split in ("train", "dev")
    }
    report: dict[str, object] = {
        "schema_version": "native-weak-anchor-counterfactual-dataset-v1",
        "status": "WEAK_ANCHOR_COUNTERFACTUAL_DATASET_READY",
        "audit_report_sha256": audit["report_sha256"],
        "audit_report_file_sha256": _file_sha256(audit_report_path),
        "implementation_sha256": _file_sha256(Path(__file__)),
        "dataset_basename": dataset_path.name,
        "dataset_sha256": _file_sha256(dataset_path),
        "source_groups": len(source_clips),
        "logical_samples": len(labels),
        "split_counts": split_counts,
        "action_order": actions,
        "source_storage_deduplicated": True,
        "target_render": {
            "source_canvas": 256,
            "radius": 7,
            "thickness": 2,
            "rgb": [245, 225, 45],
            "render_after_source_copy": True,
            "model_resize": "nearest_even_index_to_128",
        },
        "relation_diagnostic_training_allowed": True,
        "movement_policy_training_allowed": False,
        "executed_action_labels_created": False,
        "controlled_player_identity_verified": False,
        "test_frames_read": 0,
        "video_frames_decoded": 0,
        "model_runs": 0,
        "gpu_seconds": 0,
        "input_commands_sent": 0,
    }
    report["report_sha256"] = _object_sha256(report)
    (staging / "report.json").write_bytes(_canonical(report) + b"\n")
    staging.rename(output_dir)
    return report


def native_death_banner_evidence(frame: np.ndarray) -> tuple[bool, float, float]:
    """Scale the frozen 1600x720 death-banner pixel counts into fractions."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("native death banner requires uint8 RGB")
    height, width = frame.shape[:2]
    y1 = max(1, round(height * 22 / 720))
    x0, x1 = round(width * 0.45), round(width * 0.55)
    roi = frame[:y1, x0:x1].astype(np.int16)
    red = (roi[..., 0] > 80) & (roi[..., 0] - roi[..., 1] > 20) & (roi[..., 0] - roi[..., 2] > 10)
    white = (roi.min(axis=2) > 170) & (roi.max(axis=2) - roi.min(axis=2) < 45)
    red_fraction = float(red.mean())
    white_fraction = float(white.mean())
    return (
        red_fraction >= 1000 / (160 * 22) and white_fraction >= 40 / (160 * 22),
        red_fraction,
        white_fraction,
    )


def native_center_health_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("native health crop requires uint8 RGB")
    height, width = frame.shape[:2]
    crop = frame[
        round(height * 0.15) : round(height * 0.85),
        round(width * 0.30) : round(width * 0.70),
    ]
    rows = np.linspace(0, crop.shape[0] - 1, 128).astype(np.int64)
    columns = np.linspace(0, crop.shape[1] - 1, 128).astype(np.int64)
    return np.ascontiguousarray(crop[rows[:, None], columns[None, :]])


def _scan_native_death_source(
    path: Path,
    session_hash: str,
    split: str,
    health_contract_path: Path,
    qa_dir: Path,
) -> dict[str, object]:
    import av
    from PIL import Image, ImageDraw

    from hok_agent import pre_ingest
    from hok_agent.hierarchical_e1 import (
        HealthBannerConsensusEngine,
        detect_center_health_bar,
        load_health_contract,
    )
    from hok_agent.visual_events import EventEngineIdentity, VisualEvent, VisualEventType

    config, _contract, contract_sha256 = load_health_contract(health_contract_path)
    identity = EventEngineIdentity("native-death-banner-preflight-v1", contract_sha256)
    engine = HealthBannerConsensusEngine(session_hash, identity, config)
    descriptor, opened = pre_ingest._open_regular(path)
    decoded = 0
    sampled = 0
    health_visible = 0
    banner_frames = 0
    banner_without_health = 0
    events: list[VisualEvent] = []
    candidate_images: list[tuple[int, Image.Image]] = []
    context_images: list[tuple[int, Image.Image]] = []
    with os.fdopen(descriptor, "rb") as handle, av.open(handle, mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError("native death preflight requires one video stream")
        stream = streams[0]
        if (
            stream.duration is None
            or stream.time_base is None
            or stream.width <= stream.height
            or pre_ingest._rotation(stream) != 0
        ):
            raise ValueError("native death preflight requires landscape duration geometry")
        duration_us = round(stream.duration * stream.time_base * 1_000_000)
        context_targets = [round(duration_us * fraction) for fraction in (0.1, 0.5, 0.9)]
        context_index = 0
        first_us: int | None = None
        next_sample_us: int | None = None
        for video_frame in container.decode(stream):
            decoded += 1
            if video_frame.pts is None:
                raise ValueError("native death preflight frame has no timestamp")
            timestamp_us = round(video_frame.pts * stream.time_base * 1_000_000)
            first_us = timestamp_us if first_us is None else first_us
            next_sample_us = timestamp_us if next_sample_us is None else next_sample_us
            if timestamp_us < next_sample_us:
                continue
            while next_sample_us <= timestamp_us:
                next_sample_us += 200_000
            rgb = video_frame.to_ndarray(format="rgb24")
            health = detect_center_health_bar(native_center_health_frame(rgb), config)
            banner, red_fraction, white_fraction = native_death_banner_evidence(rgb)
            update = engine.update(
                f"{session_hash}:{sampled:07d}",
                timestamp_us * 1_000,
                health,
                banner,
            )
            events.extend(update.events)
            sampled += 1
            health_visible += int(health.visible)
            banner_frames += int(banner)
            banner_without_health += int(banner and not health.visible)
            keep_candidate = banner and not health.visible and len(candidate_images) < 12
            keep_context = (
                context_index < len(context_targets)
                and timestamp_us - first_us >= context_targets[context_index]
            )
            if keep_candidate or keep_context:
                image = Image.fromarray(rgb)
                image.thumbnail((320, 180))
                draw = ImageDraw.Draw(image)
                x0, x1 = round(image.width * 0.45), round(image.width * 0.55)
                y1 = max(1, round(image.height * 22 / 720))
                draw.rectangle((x0, 0, x1, y1), outline="yellow", width=2)
                draw.text(
                    (2, 2),
                    f"t={timestamp_us / 1e6:.1f} b={int(banner)} h={int(health.visible)} "
                    f"r={red_fraction:.3f} w={white_fraction:.3f}",
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
                if keep_candidate:
                    candidate_images.append((timestamp_us, image.copy()))
                if keep_context:
                    context_images.append((timestamp_us, image.copy()))
                    context_index += 1
        pre_ingest._assert_unchanged(handle.fileno(), opened)
    images = [("candidate", *item) for item in candidate_images] + [
        ("context", *item) for item in context_images
    ]
    qa_name = session_hash[:8] + "-death-cue-qa.png"
    if images:
        columns = 3
        cell_width, cell_height = 320, 200
        sheet = Image.new(
            "RGB", (columns * cell_width, math.ceil(len(images) / columns) * cell_height)
        )
        for index, (kind, timestamp_us, image) in enumerate(images):
            x, y = index % columns * cell_width, index // columns * cell_height
            sheet.paste(image, (x, y + 20))
            ImageDraw.Draw(sheet).text((x + 2, y + 2), f"{kind} {timestamp_us / 1e6:.1f}s")
        sheet.save(qa_dir / qa_name)
    counts = {
        event_type.value: sum(event.event_type == event_type for event in events)
        for event_type in (VisualEventType.DEATH, VisualEventType.RESPAWN)
    }
    return {
        "session_hash": session_hash,
        "split": split,
        "decoded_frames": decoded,
        "sampled_frames": sampled,
        "health_visible_fraction": health_visible / sampled,
        "banner_frames": banner_frames,
        "banner_without_health_frames": banner_without_health,
        "event_counts": counts,
        "event_timestamps_ms": [
            {"type": event.event_type.value, "end_ms": event.end_ns // 1_000_000}
            for event in events
            if event.event_type in {VisualEventType.DEATH, VisualEventType.RESPAWN}
        ],
        "qa_basename": qa_name if images else None,
        "qa_sha256": _file_sha256(qa_dir / qa_name) if images else None,
        "candidate_qa_frames": len(candidate_images),
        "context_qa_frames": len(context_images),
        "source_locator_persisted": False,
    }


def run_native_death_cue_preflight(
    source_root: Path,
    cohort_dir: Path,
    pre_ingest_path: Path,
    health_contract_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    from hok_agent import pre_ingest
    from hok_agent.hierarchical_e1 import load_health_contract
    from hok_agent.v5_data import load_automatic_cohort

    if output_dir.exists():
        raise ValueError("native death preflight output already exists")
    _config, _contract, health_contract_sha256 = load_health_contract(health_contract_path)
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    if any(
        cohort.session_splits.get(identity) != split
        for identity, split in NATIVE_ANCHOR_AUDIT_SOURCES.items()
    ):
        raise ValueError("native death preflight split binding differs")
    sources: dict[str, Path] = {}
    for path in pre_ingest._scan(source_root):
        identity = pre_ingest._candidate(path, source_root).candidate_id
        if identity in NATIVE_ANCHOR_AUDIT_SOURCES:
            sources[identity] = path
    if set(sources) != set(NATIVE_ANCHOR_AUDIT_SOURCES):
        raise ValueError("native death preflight source identities unavailable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    sessions = [
        _scan_native_death_source(
            path,
            identity,
            NATIVE_ANCHOR_AUDIT_SOURCES[identity],
            health_contract_path,
            staging,
        )
        for identity, path in sorted(sources.items())
    ]
    positive = [
        row
        for row in sessions
        if cast(dict[str, int], row["event_counts"])["DEATH"] >= 1
        and cast(dict[str, int], row["event_counts"])["RESPAWN"] >= 1
    ]
    unpaired = [
        row
        for row in sessions
        if (cast(dict[str, int], row["event_counts"])["DEATH"] > 0)
        != (cast(dict[str, int], row["event_counts"])["RESPAWN"] > 0)
    ]
    checks = {
        "minimum_positive_sessions": len(positive) >= 3,
        "no_unpaired_sessions": not unpaired,
        "train_dev_present": {row["split"] for row in sessions} == {"train", "dev"},
        "qa_context_complete": all(cast(int, row["context_qa_frames"]) == 3 for row in sessions),
    }
    payload: dict[str, object] = {
        "schema_version": "native-death-banner-health-preflight-v1",
        "status": "NATIVE_DEATH_CUE_PREFLIGHT_SUPPORTED"
        if all(checks.values())
        else "NATIVE_DEATH_CUE_PREFLIGHT_INSUFFICIENT",
        "cohort_sha256": cohort.cohort_sha256,
        "health_contract_sha256": health_contract_sha256,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "sampling_period_ms": 200,
        "source_sessions": len(sessions),
        "positive_sessions": len(positive),
        "unpaired_sessions": len(unpaired),
        "decoded_frames": sum(cast(int, row["decoded_frames"]) for row in sessions),
        "sampled_frames": sum(cast(int, row["sampled_frames"]) for row in sessions),
        "sessions": sessions,
        "checks": checks,
        "banner_roi_fraction": [0.45, 0.0, 0.55, 22 / 720],
        "banner_red_fraction_minimum": 1000 / (160 * 22),
        "banner_white_fraction_minimum": 40 / (160 * 22),
        "threshold_origin": "frozen mobile death-banner pixel counts normalized by ROI area",
        "candidate_definition": "HealthBannerConsensusEngine over 200ms raw-video samples",
        "raw_full_resolution_frames_persisted": False,
        "derived_full_view_qa_only": True,
        "raw_source_locators_persisted": False,
        "semantic_accuracy_verified": False,
        "reward_allowed": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "video_test_opened": False,
        "input_commands_sent": 0,
        "model_runs": 0,
        "gpu_seconds": 0,
    }
    payload["report_sha256"] = _object_sha256(payload)
    (staging / "report.json").write_bytes(_canonical(payload) + b"\n")
    staging.rename(output_dir)
    return payload


def run_houyi_data_binding_audit(
    source_root: Path,
    cohort_dir: Path,
    pre_ingest_path: Path,
    hero_profile_path: Path,
    summary_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Audit existing identity metadata without decoding a frame or guessing from RGB."""
    import av

    from hok_agent import pre_ingest
    from hok_agent.v5_data import load_automatic_cohort

    if output_dir.exists():
        raise ValueError("Houyi data audit output already exists")
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    evidence = pre_ingest.load_pre_ingest(pre_ingest_path)
    profile = _read_json_no_binding(hero_profile_path)
    if (
        profile.get("schema_version") != "hok-agent-hero-ability-profile-v1"
        or not summary_root.is_dir()
        or summary_root.is_symlink()
    ):
        raise ValueError("Houyi data audit profile or summary root differs")
    metadata_keys: Counter[str] = Counter()
    opened_by_split: Counter[str] = Counter()
    keyword_hits: list[dict[str, str]] = []
    failures = 0
    keywords = ("后羿", "houyi", "hou yi", "射手", "发育路")
    paths = pre_ingest._scan(source_root)
    candidates = pre_ingest._candidate_list(paths, source_root)
    if len(candidates) != len(paths) or len(candidates) != len(cohort.session_splits):
        raise ValueError("Houyi data audit candidate set differs")
    for candidate, path in zip(candidates, paths, strict=True):
        split = cohort.session_splits.get(candidate.candidate_id)
        if split not in {"train", "dev"}:
            continue
        descriptor, opened = pre_ingest._open_regular(path)
        try:
            with os.fdopen(descriptor, "rb") as handle, av.open(handle, mode="r") as container:
                metadata = dict(container.metadata)
                for stream in container.streams:
                    metadata.update(
                        {f"stream.{key}": value for key, value in stream.metadata.items()}
                    )
                pre_ingest._assert_unchanged(handle.fileno(), opened)
        except (OSError, ValueError):
            failures += 1
            continue
        opened_by_split[split] += 1
        metadata_keys.update(metadata.keys())
        text = " ".join(str(value) for value in metadata.values()).lower()
        matched = [keyword for keyword in keywords if keyword.lower() in text]
        if matched:
            keyword_hits.append(
                {
                    "session_hash": candidate.candidate_id,
                    "split": split,
                    "keyword": matched[0],
                }
            )
    summary_count = 0
    hero_field_count = 0
    complete_real_bindings = 0
    declared_heroes: Counter[str] = Counter()
    for summary_path in summary_root.rglob("summary.json"):
        if summary_path.is_symlink() or not summary_path.is_file():
            continue
        try:
            summary = _read_json_no_binding(summary_path)
        except ValueError:
            continue
        summary_count += 1
        hero_fields = {key: value for key, value in summary.items() if "hero" in key.lower()}
        hero_field_count += len(hero_fields)
        for value in hero_fields.values():
            declared_heroes[str(value).lower()] += 1
        if (
            str(summary.get("hero", "")).lower() == "houyi"
            and isinstance(summary.get("hero_identity_sha256"), str)
            and isinstance(summary.get("hero_binding_source"), str)
            and summary.get("derived_roi_rgb_persisted") is True
        ):
            complete_real_bindings += 1
    expected_by_split = Counter(cohort.session_splits.values())
    checks = {
        "all_train_metadata_opened": opened_by_split["train"] == expected_by_split["train"],
        "all_dev_metadata_opened": opened_by_split["dev"] == expected_by_split["dev"],
        "test_container_metadata_opened_zero": True,
        "metadata_failures_zero": failures == 0,
        "configured_houyi_profile": profile.get("profile_status") == "CONFIGURED"
        and str(profile.get("hero_id", "")).upper() == "HOU_YI",
        "complete_real_session_bindings": complete_real_bindings >= 1,
    }
    ready = all(checks.values())
    payload: dict[str, object] = {
        "schema_version": "houyi-existing-data-binding-audit-v1",
        "status": "HOUYI_BOUND_REAL_DATA_AVAILABLE"
        if ready
        else "HOUYI_BOUND_REAL_DATA_NOT_AVAILABLE",
        "cohort_sha256": cohort.cohort_sha256,
        "pre_ingest_sha256": evidence.pre_ingest_sha256,
        "hero_profile_file_sha256": _file_sha256(hero_profile_path),
        "hero_profile_status": profile.get("profile_status"),
        "hero_profile_id": profile.get("hero_id"),
        "opened_container_metadata": dict(opened_by_split),
        "cohort_split_counts": dict(expected_by_split),
        "metadata_keys": dict(metadata_keys),
        "metadata_failures": failures,
        "metadata_keyword_hits": keyword_hits,
        "summaries_scanned": summary_count,
        "summary_hero_fields": hero_field_count,
        "declared_hero_values": dict(declared_heroes),
        "complete_real_houyi_bindings": complete_real_bindings,
        "checks": checks,
        "prior_exploratory_boundary_deviation": {
            "containers_opened": 149,
            "train": 103,
            "dev": 23,
            "test": 23,
            "frames_decoded": 0,
            "hero_keyword_hits": 0,
            "used_for_selection_or_tuning": False,
            "note": "container and stream metadata only; formal audit filters split before open",
        },
        "source_locators_persisted": False,
        "video_frames_decoded": 0,
        "test_frames_decoded": 0,
        "test_container_metadata_opened": 0,
        "human_visual_identity_labels_created": False,
        "hero_data_contract_allowed": ready,
        "training_allowed": False,
        "policy_promotion_allowed": False,
        "input_commands_sent": 0,
        "model_runs": 0,
        "next_step": (
            "create a hero-bound dataset contract"
            if ready
            else "requires episode-scoped hero declaration plus immutable identity evidence"
        ),
    }
    payload["report_sha256"] = _object_sha256(payload)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    (staging / "report.json").write_bytes(_canonical(payload) + b"\n")
    staging.rename(output_dir)
    return payload


def _read_json_no_binding(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Houyi data audit JSON differs") from exc
    if not isinstance(value, dict):
        raise ValueError("Houyi data audit JSON root differs")
    return cast(dict[str, object], value)
