from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.movement_real_rgb import (
    _canonical_content,
    _object_sha256,
    materialize_real_counterfactual_overfit32,
    run_real_player_cue_preflight,
    run_real_player_goal_continuity,
    run_real_player_localization_audit_v2,
    run_real_rgb_goal_canvas,
    run_real_rgb_preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def test_joystick_match_tracks_translation_and_rejects_blank() -> None:
    from hok_agent.movement_real_rgb import _joystick_match

    template = np.random.default_rng(7).normal(size=(25, 25)).astype(np.float32)
    mask = np.ones(template.shape, dtype=np.uint8)
    for x, y in ((20, 30), (70, 55)):
        signal = np.zeros((110, 120), np.float32)
        signal[y:y+25, x:x+25] = template
        center, score, margin, contrast = _joystick_match(signal, template, mask)
        assert center == (x + 12, y + 12)
        assert score > 0.99 and margin > 0.5 and contrast > 0.99
    _, score, _, contrast = _joystick_match(np.zeros((110, 120), np.float32), template, mask)
    assert score < 0.35 and contrast == 0


def test_joystick_stop_unknown_and_dynamic_base(monkeypatch: pytest.MonkeyPatch) -> None:
    from hok_agent import movement_real_rgb as m

    # Base is moving; relative displacement, not screen position, determines direction.
    pairs = [(10, 10, 30, 10, 1), (20, 20, 20, 20, 1), (30, 30, 30, 30, 1),
             (40, 40, 40, 40, 0.1), (50, 50, 50, 50, 1)]
    matches = iter(value for bx, by, kx, ky, contrast in pairs for value in (
        ((bx, by), 0.9, 0.5, contrast), ((kx, ky), 0.9, 0.5, contrast)))
    monkeypatch.setattr(m, "_joystick_match", lambda *_args: next(matches))
    t = dict.fromkeys(("base", "knob", "base_mask", "knob_mask"), np.ones((3, 3)))
    rows = m.extract_joystick_sequence(np.zeros((5, 128, 128, 3), np.uint8), t)
    assert [r["candidate_action"] for r in rows] == ["E", "unknown", "STOP", "unknown", "unknown"]
    assert rows[3]["reason"] == "low_contrast_or_occluded"
    assert rows[4]["reason"] == "center_confirming"


def test_joystick_ambiguous_peak_cannot_be_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    from hok_agent import movement_real_rgb as m

    monkeypatch.setattr(m, "_joystick_match", lambda *_args: ((50, 50), 0.99, 0.01, 1))
    t = dict.fromkeys(("base", "knob", "base_mask", "knob_mask"), np.ones((3, 3)))
    rows = m.extract_joystick_sequence(np.zeros((3, 128, 128, 3), np.uint8), t)
    assert all(r["candidate_action"] == "unknown" for r in rows)


def test_flow_gap_rejoin_and_session_reset() -> None:
    from hok_agent.movement_real_rgb import bridge_player_gaps

    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    frames = np.stack([np.roll(base, i, axis=1) for i in range(8)])
    positions = [(64.0, 64.0), None, None, (64.0, 67.0)]
    result, gaps = bridge_player_gaps(frames[:4], positions, 200)
    assert gaps[0]["accepted"]
    assert np.allclose(result[1:3], [(64, 65), (64, 66)], atol=0.3)
    assert positions[1] is None  # no mutation of original detector evidence
    missing, _ = bridge_player_gaps(frames[:4], [None] * 4, 200)
    assert missing == [None] * 4
    long = [(64.0, 64.0), *([None] * 6), (64.0, 71.0)]
    assert bridge_player_gaps(frames, long, 200)[0] == long
    mismatch = [(64.0, 64.0), None, None, (90.0, 90.0)]
    assert bridge_player_gaps(frames[:4], mismatch, 200)[0] == mismatch
    assert bridge_player_gaps(frames[:4], positions[:-1] + [None], 200)[0][-3:] == [None]*3


def test_flow_rejects_backward_inconsistency(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    from hok_agent.movement_real_rgb import _flow_step

    points = np.full((5, 1, 2), 50, dtype=np.float32)
    calls = 0

    def fake_lk(*args, **kwargs):
        nonlocal calls
        calls += 1
        return points + calls * 4, np.ones((5, 1), np.uint8), None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    gray = np.zeros((128, 128), np.uint8)
    kept, delta, reason = _flow_step(gray, gray, points)
    assert kept is None and delta is None
    assert reason == "insufficient_consistent_points"


def test_flow_actions_only_enter_evaluation() -> None:
    import inspect

    from hok_agent.movement_real_rgb import _flow_response, bridge_player_gaps

    assert list(inspect.signature(bridge_player_gaps).parameters) == [
        "frames", "positions", "period_ms"
    ]
    points = [(64.0, float(32 + i)) for i in range(20)]
    sent = np.ones(20, dtype=bool)
    east = _flow_response(points, np.full(20, 3), sent, 5)
    west = _flow_response(points, np.full(20, 7), sent, 5)
    assert east["passed"] and not west["passed"]
    assert east["valid_fraction_of_all_sent"] == 0.75


def test_native_window_crops_before_resize_and_rejects_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from fractions import Fraction
    from types import SimpleNamespace

    from hok_agent.movement_real_rgb import _joystick_window, _native_landscape_window

    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic descriptor")
    yy, xx = np.indices((180, 320))
    rgb = np.stack([xx % 256, yy, np.zeros_like(xx)], axis=-1).astype(np.uint8)
    stream = SimpleNamespace(
        duration=5000, time_base=Fraction(1, 1000), width=320, height=180, metadata={}
    )
    frames = [SimpleNamespace(pts=1000 + i * 200, to_ndarray=lambda **_kw: rgb) for i in range(16)]

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, *args, **kwargs):
            pass

        def decode(self, requested):
            assert requested is stream
            return iter(frames)

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda *a, **kw: Container()))
    result = _native_landscape_window(source)
    assert result["minimap_rgb"].shape == (16, 256, 256, 3)
    assert result["minimap_rgb"][0, 0, 0].tolist() == [8, 0, 0]
    assert result["minimap_rgb"][0, -1, -1].tolist() == [68, 71, 0]
    assert result["main_rgb"][0, 0, 0].tolist() == [96, 27, 0]
    assert np.diff(result["timestamp_us"]).tolist() == [200000] * 15
    frames[:] = [SimpleNamespace(pts=1000 + i * 100, to_ndarray=lambda **_kw: rgb)
                 for i in range(40)]
    joystick = _joystick_window(source, 0.2)
    assert joystick["rgb"].shape == (40, 99, 112, 3)
    assert np.array_equal(joystick["rgb"][0], rgb[81:, :112])
    assert np.diff(joystick["timestamp_us"]).tolist() == [100000] * 39
    frames[:] = frames[:3]
    with pytest.raises(ValueError, match="incomplete"):
        _joystick_window(source, 0.2)
    with pytest.raises(ValueError, match="incomplete"):
        _native_landscape_window(source)
    stream.width = 100
    with pytest.raises(ValueError, match="landscape"):
        _native_landscape_window(source)


def test_joystick_split_rejected_before_source_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from hok_agent import pre_ingest, v5_data
    from hok_agent.movement_real_rgb import NATIVE_PLAYER_SOURCES, run_joystick_visibility

    monkeypatch.setattr(
        v5_data, "load_automatic_cohort",
        lambda *_args: SimpleNamespace(
            session_splits=dict.fromkeys(NATIVE_PLAYER_SOURCES, "test")
        ),
    )

    def forbidden(*args):
        pytest.fail("source scan/open must not happen after split mismatch")

    monkeypatch.setattr(pre_ingest, "_scan", forbidden)
    with pytest.raises(ValueError, match="split"):
        run_joystick_visibility(tmp_path, tmp_path, tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_ring_cue_requires_hollow_shape_and_rgb() -> None:
    from hok_agent.movement_real_rgb import green_ring_candidates

    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    yy, xx = np.indices((256, 256))
    d = np.hypot(yy - 100, xx - 100)
    rgb[(d >= 9) & (d <= 11)] = (20, 220, 30)
    peaks = green_ring_candidates(rgb)
    assert len(peaks) == 1 and np.linalg.norm(np.asarray(peaks[0]) - [100, 100]) < 3
    assert green_ring_candidates(np.zeros_like(rgb)) == []
    assert green_ring_candidates(np.full_like(rgb, (20, 220, 30))) == []
    with pytest.raises(ValueError, match="256x256"):
        green_ring_candidates(rgb[:128])


def test_native_coordinate_axes_match_sampling_grid() -> None:
    from hok_agent.movement_real_rgb import native_map_point_to_source_xy

    rows = np.linspace(0, round(1080 * 0.4) - 1, 256).astype(int)
    columns = np.linspace(round(2400 * 0.025), round(2400 * 0.215) - 1, 256).astype(int)
    for i in range(256):
        x, y = native_map_point_to_source_xy((i, 255 - i), (2400, 1080))
        assert (x, y) == (columns[255 - i], rows[i])
        assert int(np.argmin(abs(columns - x))) == 255 - i
        assert int(np.argmin(abs(rows - y))) == i
    with pytest.raises(ValueError, match="outside"):
        native_map_point_to_source_xy((256, 0), (2400, 1080))


def test_native_translation_reports_support_and_rejects_vacuous_success() -> None:
    from hok_agent.movement_real_rgb import native_coordinate_diagnostic

    frames = np.zeros((4, 256, 256, 3), dtype=np.uint8)
    yy, xx = np.indices((256, 256))
    for i, frame in enumerate(frames):
        distance = np.hypot(yy - 100 - i, xx - 110)
        frame[(distance >= 9) & (distance <= 11)] = (20, 220, 30)
    original = frames.copy()
    result = native_coordinate_diagnostic(frames)
    assert np.array_equal(frames, original)
    assert result["confirmed_frames"] == 3
    for row in result["translations"]:
        assert row["compared_frames"] == 3
        assert row["maximum_equivariance_error_pixels"] == 0
        assert row["lost_confirmations"] == row["new_confirmations"] == 0
    empty = native_coordinate_diagnostic(np.zeros_like(frames))
    assert empty["confirmed_frames"] == 0
    assert all(row["maximum_equivariance_error_pixels"] is None for row in empty["translations"])


def test_ring_track_nearby_confirmation_missing_and_jump(monkeypatch: pytest.MonkeyPatch) -> None:
    from hok_agent import movement_real_rgb as module

    candidates = iter(
        [
            [(100, 100)],
            [(101, 101), (220, 220)],
            [(102, 102), (221, 221)],
            [],
            [(103, 103)],
            [(104, 104)],
            [(200, 200)],
            [(201, 201)],
        ]
    )
    monkeypatch.setattr(module, "green_ring_candidates", lambda _frame: next(candidates))
    p = module.green_ring_track(np.zeros((8, 256, 256, 3), dtype=np.uint8))
    assert p == [None, (101, 101), (102, 102), None, None, (104, 104), None, (201, 201)]


def test_ring_regions_preserve_margin_and_explain_unknown() -> None:
    from hok_agent.movement_real_rgb import _confirm_ring_candidates, _ring_diagnostic_region

    assert _ring_diagnostic_region((100, 100)) == "interior"
    assert _ring_diagnostic_region((215, 100)) == "edge_margin"
    assert _ring_diagnostic_region((240, 100)) == "context"
    positions, reasons = _confirm_ring_candidates(
        [
            [],
            [(100, 100)],
            [(101, 101)],
            [(150, 150)],
            [(151, 151), (152, 152)],
        ]
    )
    assert positions == [None, None, (101, 101), None, None]
    assert reasons == [
        "no_ring_evidence",
        "awaiting_confirmation",
        "confirmed_visual_cue",
        "discontinuous_candidate",
        "ambiguous_candidates",
    ]


def test_cached_background_audit_never_decodes_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hok_agent import movement_real_rgb as module

    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for identity, split in module.NATIVE_PLAYER_SOURCES.items():
        name = identity[:8] + "-native-window.npz"
        path = source / name
        np.savez_compressed(path, minimap_rgb=np.zeros((16, 256, 256, 3), dtype=np.uint8))
        rows.append(
            {
                "session_hash": identity,
                "split": split,
                "artifacts": [
                    {"basename": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                ],
                "confirmed_green_ring_yx": [None] * 16,
            }
        )
    report = {"status": "NATIVE_LANDSCAPE_WINDOWS_MATERIALIZED_QA_ONLY", "sessions": rows}
    report["report_sha256"] = _object_sha256(report)
    (source / "report.json").write_text(json.dumps(report))

    def forbidden(*_args):
        raise AssertionError("must not decode")

    monkeypatch.setattr(module, "_native_landscape_window", forbidden)
    result = module.audit_native_player_background(source, tmp_path / "audit")
    assert result["video_frames_decoded"] == result["model_runs"] == 0
    assert result["rgb_modified"] is result["runtime_filter_promoted"] is False
    for session in result["sessions"]:
        for variant in session["variants"].values():
            assert variant["reason_counts"] == {"no_ring_evidence": 16}
    with pytest.raises(ValueError, match="already exists"):
        module.audit_native_player_background(source, tmp_path / "audit")
    monkeypatch.setattr(module, "green_ring_candidates", lambda _frame: [(100, 100), (240, 100)])
    filtered = module.audit_native_player_background(source, tmp_path / "filtered")
    for session in filtered["sessions"]:
        assert session["variants"]["unfiltered"]["confirmed_frames"] == 0
        assert session["variants"]["interior_plus_edge"]["confirmed_frames"] == 15
        assert session["candidate_regions"] == {"interior": 16, "context": 16}
    (source / rows[0]["artifacts"][0]["basename"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash differs"):
        module.audit_native_player_background(source, tmp_path / "bad")


def test_native_pilot_opens_selected_sources_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from hok_agent import movement_real_rgb as module
    from hok_agent import pre_ingest, v5_data

    raw = tmp_path / "raw"
    raw.mkdir()
    paths = [raw / f"{i}.mp4" for i in range(3)]
    for path in paths:
        path.write_bytes(b"dummy")
    ids = [pre_ingest._candidate(path, raw).candidate_id for path in paths]
    monkeypatch.setattr(module, "NATIVE_PLAYER_SOURCES", {ids[0]: "train", ids[1]: "dev"})
    monkeypatch.setattr(
        v5_data,
        "load_automatic_cohort",
        lambda *_a: SimpleNamespace(
            session_splits=dict(zip(ids, ["train", "dev", "test"], strict=True)),
            cohort_sha256="0" * 64,
        ),
    )
    decoded = []
    fractions_seen = []

    def decode(path, *, start_fraction=0.2):
        assert path != paths[2]
        decoded.append(path)
        fractions_seen.append(start_fraction)
        return {
            "minimap_rgb": np.zeros((16, 256, 256, 3), dtype=np.uint8),
            "main_rgb": np.zeros((16, 256, 256, 3), dtype=np.uint8),
            "timestamp_us": np.arange(16) * 200000,
            "native_crop_sha256": np.asarray(["a" * 64] * 16, dtype="U64"),
        }

    monkeypatch.setattr(module, "_native_landscape_window", decode)
    output = tmp_path / "out"
    report = module.run_native_player_pilot(raw, tmp_path, tmp_path, output)
    assert set(decoded) == set(paths[:2])
    assert report["test_frames_decoded"] == report["model_runs"] == 0
    assert report["action_labels_created"] is report["training_allowed"] is False
    assert len(list(output.glob("*.npz"))) == 2
    scan = module.run_native_player_pilot(
        raw,
        tmp_path,
        tmp_path,
        tmp_path / "scan",
        train_visibility_scan=True,
    )
    assert decoded[2:] == [paths[0]] * 3
    assert fractions_seen[2:] == [0.05, 0.10, 0.15]
    assert len(scan["sessions"]) == 3 and {row["split"] for row in scan["sessions"]} == {"train"}
    assert {row["start_fraction"] for row in scan["sessions"]} == {0.05, 0.10, 0.15}
    assert len(list((tmp_path / "scan").glob("*.npz"))) == 3
    with pytest.raises(ValueError, match="already exists"):
        module.run_native_player_pilot(raw, tmp_path, tmp_path, output)
    assert set(decoded) == set(paths[:2])


def test_native_anchor_cohort_audit_keeps_groups_and_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from hok_agent import movement_real_rgb as module
    from hok_agent import pre_ingest, v5_data

    raw = tmp_path / "raw"
    raw.mkdir()
    paths = [raw / f"{index:02d}.mp4" for index in range(13)]
    for path in paths:
        path.write_bytes(b"dummy")
    identities = [pre_ingest._candidate(path, raw).candidate_id for path in paths]
    selected = {
        identity: "train" if index < 8 else "dev" for index, identity in enumerate(identities[:12])
    }
    monkeypatch.setattr(module, "NATIVE_ANCHOR_AUDIT_SOURCES", selected)
    monkeypatch.setattr(
        v5_data,
        "load_automatic_cohort",
        lambda *_args: SimpleNamespace(
            session_splits=selected | {identities[12]: "test"}, cohort_sha256="1" * 64
        ),
    )
    decoded: list[tuple[Path, float]] = []
    frames = np.zeros((16, 256, 256, 3), dtype=np.uint8)
    yy, xx = np.indices((256, 256))
    for index, frame in enumerate(frames):
        distance = np.hypot(yy - (100 + index // 4), xx - 110)
        frame[(distance >= 9) & (distance <= 11)] = (20, 220, 30)

    def decode(path: Path, *, start_fraction: float = 0.2) -> dict[str, np.ndarray]:
        assert path != paths[12]
        decoded.append((path, start_fraction))
        return {
            "minimap_rgb": frames.copy(),
            "main_rgb": np.zeros_like(frames),
            "timestamp_us": np.arange(16) * 200_000,
            "native_crop_sha256": np.asarray(["a" * 64] * 16, dtype="U64"),
        }

    monkeypatch.setattr(module, "_native_landscape_window", decode)
    output = tmp_path / "audit"
    report = module.run_native_anchor_cohort_audit(raw, tmp_path, tmp_path, output)
    assert report["status"] == "WEAK_VISUAL_ANCHOR_COHORT_SUPPORTED_QA_ONLY"
    assert len(decoded) == 36
    assert {fraction for _path, fraction in decoded} == {0.1, 0.3, 0.6}
    assert {path for path, _fraction in decoded} == set(paths[:12])
    assert report["support"]["train"] == {
        "sessions": 8,
        "supported_sessions": 8,
        "confirmed_frames": 360,
    }
    assert report["support"]["dev"] == {
        "sessions": 4,
        "supported_sessions": 4,
        "confirmed_frames": 180,
    }
    assert report["training_allowed"] is report["promotion_allowed"] is False
    assert report["test_frames_decoded"] == report["model_runs"] == 0
    assert len(list(output.glob("*.npz"))) == 12
    assert len({row["session_hash"] for row in report["sessions"]}) == 12
    with pytest.raises(ValueError, match="already exists"):
        module.run_native_anchor_cohort_audit(raw, tmp_path, tmp_path, output)
    assert report["policy_action_labels_created"] is False


def test_native_anchor_repair_adds_only_four_new_dev_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from hok_agent import movement_real_rgb as module
    from hok_agent import pre_ingest, v5_data

    raw = tmp_path / "raw"
    raw.mkdir()
    paths = [raw / f"{index:02d}.mp4" for index in range(17)]
    for path in paths:
        path.write_bytes(b"dummy")
    identities = [pre_ingest._candidate(path, raw).candidate_id for path in paths]
    original = {
        identity: "train" if index < 8 else "dev" for index, identity in enumerate(identities[:12])
    }
    repair = {identity: "dev" for identity in identities[12:16]}
    monkeypatch.setattr(module, "NATIVE_ANCHOR_AUDIT_SOURCES", original)
    monkeypatch.setattr(module, "NATIVE_ANCHOR_REPAIR_SOURCES", repair)
    monkeypatch.setattr(
        v5_data,
        "load_automatic_cohort",
        lambda *_args: SimpleNamespace(
            session_splits=original | repair | {identities[16]: "test"},
            cohort_sha256="2" * 64,
        ),
    )
    old_sessions = [
        {
            "session_hash": identity,
            "split": split,
            "supported_session": (index < 7 if split == "train" else index < 10),
            "confirmed_frames": (
                19
                if split == "train" and index < 6
                else 16
                if split == "train" and index == 6
                else 32
                if split == "dev" and index < 10
                else 0
            ),
        }
        for index, (identity, split) in enumerate(original.items())
    ]
    prior = {
        "schema_version": "native-weak-visual-anchor-cohort-audit-v1",
        "status": "WEAK_VISUAL_ANCHOR_COHORT_INSUFFICIENT",
        "checks": {
            "train_supported_sessions": True,
            "dev_supported_sessions": False,
            "train_confirmed_frames": True,
            "dev_confirmed_frames": True,
            "session_split_isolation": True,
        },
        "sessions": old_sessions,
    }
    prior["report_sha256"] = _object_sha256(prior)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior))
    frames = np.zeros((16, 256, 256, 3), dtype=np.uint8)
    yy, xx = np.indices((256, 256))
    distance = np.hypot(yy - 100, xx - 110)
    frames[:, (distance >= 9) & (distance <= 11)] = (20, 220, 30)
    decoded: list[Path] = []

    def decode(path: Path, *, start_fraction: float = 0.2) -> dict[str, np.ndarray]:
        assert path not in paths[:12] and path != paths[16]
        decoded.append(path)
        return {
            "minimap_rgb": frames.copy(),
            "main_rgb": np.zeros_like(frames),
            "timestamp_us": np.arange(16) * 200_000,
            "native_crop_sha256": np.asarray(["b" * 64] * 16, dtype="U64"),
        }

    monkeypatch.setattr(module, "_native_landscape_window", decode)
    output = tmp_path / "repair"
    report = module.run_native_anchor_cohort_audit(
        raw, tmp_path, tmp_path, output, prior_report_path=prior_path
    )
    assert report["schema_version"] == "native-weak-visual-anchor-cohort-audit-v2"
    assert report["status"] == "WEAK_VISUAL_ANCHOR_COHORT_SUPPORTED_QA_ONLY"
    assert len(decoded) == 12 and set(decoded) == set(paths[12:16])
    assert report["source_sessions"] == 16 and report["new_source_sessions"] == 4
    assert report["windows"] == 48 and report["new_windows"] == 12
    assert report["support"]["train"]["sessions"] == 8
    assert report["support"]["dev"]["supported_sessions"] == 6
    assert report["training_allowed"] is report["promotion_allowed"] is False


def test_native_counterfactual_audit_is_balanced_and_non_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hok_agent import movement_real_rgb as module

    source_run = tmp_path / "source"
    repair_run = tmp_path / "repair"
    source_run.mkdir()
    repair_run.mkdir()
    identities = [hashlib.sha256(str(index).encode()).hexdigest() for index in range(10)]
    source_bindings = {
        identity: "train" if index < 6 else "dev" for index, identity in enumerate(identities[:8])
    }
    repair_bindings = {identity: "dev" for identity in identities[8:]}
    monkeypatch.setattr(module, "NATIVE_ANCHOR_AUDIT_SOURCES", source_bindings)
    frames = np.zeros((48, 256, 256, 3), dtype=np.uint8)
    yy, xx = np.indices((256, 256))
    distance = np.hypot(yy - 100, xx - 110)
    frames[:, (distance >= 9) & (distance <= 11)] = (20, 220, 30)
    positions = module.green_ring_track(frames[:16])

    def sessions(bindings: dict[str, str], directory: Path) -> list[dict[str, object]]:
        rows = []
        for identity, split in bindings.items():
            name = identity[:8] + "-anchor-windows.npz"
            path = directory / name
            np.savez_compressed(
                path,
                minimap_rgb=frames,
                timestamp_us=np.arange(48) * 200_000,
                window_id=np.repeat(np.arange(3, dtype=np.int8), 16),
            )
            rows.append(
                {
                    "session_hash": identity,
                    "split": split,
                    "artifacts": [
                        {"basename": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    ],
                    "windows": [
                        {
                            "window_id": window_id,
                            "confirmed_frames": 15,
                            "confirmed_positions_yx": positions,
                        }
                        for window_id in range(3)
                    ],
                }
            )
        return rows

    source = {
        "schema_version": "native-weak-visual-anchor-cohort-audit-v1",
        "status": "WEAK_VISUAL_ANCHOR_COHORT_INSUFFICIENT",
        "sessions": sessions(source_bindings, source_run),
    }
    source["report_sha256"] = _object_sha256(source)
    (source_run / "report.json").write_text(json.dumps(source))
    repair = {
        "schema_version": "native-weak-visual-anchor-cohort-audit-v2",
        "status": "WEAK_VISUAL_ANCHOR_COHORT_SUPPORTED_QA_ONLY",
        "prior_report_file_sha256": hashlib.sha256(
            (source_run / "report.json").read_bytes()
        ).hexdigest(),
        "prior_report_sha256": source["report_sha256"],
        "sessions": sessions(repair_bindings, repair_run),
    }
    repair["report_sha256"] = _object_sha256(repair)
    repair_path = repair_run / "report.json"
    repair_path.write_text(json.dumps(repair))
    output = tmp_path / "audit"
    report = module.audit_native_anchor_counterfactual(source_run, repair_path, output)
    assert report["status"] == "WEAK_ANCHOR_COUNTERFACTUAL_DATA_SUPPORTED"
    assert report["support"]["train"] == {
        "sessions": 6,
        "groups": 18,
        "samples": 162,
        "per_action": {action: 18 for action in report["action_order"]},
    }
    assert report["support"]["dev"]["sessions"] == 4
    assert report["support"]["dev"]["groups"] == 12
    assert report["relation_diagnostic_training_allowed"] is True
    assert report["movement_policy_training_allowed"] is False
    assert report["executed_action_labels_created"] is False
    assert report["video_frames_decoded"] == report["test_frames_read"] == 0
    assert len({row["group_id"] for row in report["groups"]}) == 30
    dataset_dir = tmp_path / "dataset"
    dataset_report = module.materialize_native_anchor_counterfactual(
        output / "report.json", source_run, repair_run, dataset_dir
    )
    assert dataset_report["source_groups"] == 30
    assert dataset_report["logical_samples"] == 270
    assert dataset_report["split_counts"] == {
        "train": {"groups": 18, "samples": 162},
        "dev": {"groups": 12, "samples": 108},
    }
    assert dataset_report["movement_policy_training_allowed"] is False
    with np.load(dataset_dir / "weak-anchor-counterfactual.npz", allow_pickle=False) as arrays:
        assert arrays["source_clips"].shape == (30, 16, 256, 256, 3)
        assert arrays["sample_group_index"].shape == (270,)
        assert arrays["sample_label"].tolist() == list(range(9)) * 30
        assert len(set(arrays["group_id"].tolist())) == 30
        assert not np.shares_memory(arrays["source_clips"][0], frames)
    with pytest.raises(ValueError, match="already exists"):
        module.audit_native_anchor_counterfactual(source_run, repair_path, output)


def test_counterfactual_goal_canvas_size_is_explicit() -> None:
    from hok_agent.movement_real_rgb import _counterfactual_goal, mark_pixel_goal

    assert _counterfactual_goal((120, 120), "SE", 24, 7) is None
    assert _counterfactual_goal((120, 120), "SE", 24, 7, canvas_size=256) == (144, 144)
    source = np.zeros((256, 256, 3), dtype=np.uint8)
    marked = mark_pixel_goal(source, (144, 144))
    assert not source.any()
    assert marked[144, 151].tolist() == [245, 225, 45]
    assert marked[144, 144].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="outside"):
        mark_pixel_goal(source, (2, 2))


def test_native_death_banner_and_health_crops_use_frozen_geometry() -> None:
    from hok_agent.movement_real_rgb import (
        native_center_health_frame,
        native_death_banner_evidence,
    )

    frame = np.zeros((720, 1600, 3), dtype=np.uint8)
    frame[:22, 720:880] = (130, 30, 20)
    frame[:2, 720:800] = (220, 220, 220)
    visible, red_fraction, white_fraction = native_death_banner_evidence(frame)
    assert visible is True
    assert red_fraction >= 1000 / (160 * 22)
    assert white_fraction >= 40 / (160 * 22)
    frame[:22, 720:880] = 0
    assert native_death_banner_evidence(frame)[0] is False
    yy, xx = np.indices((720, 1600))
    geometry = np.stack([xx % 256, yy % 256, np.zeros_like(xx)], axis=-1).astype(np.uint8)
    health = native_center_health_frame(geometry)
    assert health.shape == (128, 128, 3)
    assert health[0, 0].tolist() == [224, 108, 0]
    assert health[-1, -1].tolist() == [95, 99, 0]


def test_native_death_preflight_opens_fixed_train_dev_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from hok_agent import movement_real_rgb as module
    from hok_agent import pre_ingest, v5_data

    raw = tmp_path / "raw"
    raw.mkdir()
    paths = [raw / f"{index}.mp4" for index in range(4)]
    for path in paths:
        path.write_bytes(b"dummy")
    identities = [pre_ingest._candidate(path, raw).candidate_id for path in paths]
    selected = {identities[0]: "train", identities[1]: "train", identities[2]: "dev"}
    monkeypatch.setattr(module, "NATIVE_ANCHOR_AUDIT_SOURCES", selected)
    monkeypatch.setattr(
        v5_data,
        "load_automatic_cohort",
        lambda *_args: SimpleNamespace(
            session_splits=selected | {identities[3]: "test"}, cohort_sha256="3" * 64
        ),
    )
    opened: list[Path] = []

    def scan(
        path: Path,
        session_hash: str,
        split: str,
        _contract: Path,
        _qa_dir: Path,
    ) -> dict[str, object]:
        assert path != paths[3]
        opened.append(path)
        return {
            "session_hash": session_hash,
            "split": split,
            "decoded_frames": 100,
            "sampled_frames": 20,
            "health_visible_fraction": 0.5,
            "banner_frames": 5,
            "banner_without_health_frames": 5,
            "event_counts": {"DEATH": 1, "RESPAWN": 1},
            "event_timestamps_ms": [],
            "qa_basename": None,
            "qa_sha256": None,
            "candidate_qa_frames": 0,
            "context_qa_frames": 3,
            "source_locator_persisted": False,
        }

    monkeypatch.setattr(module, "_scan_native_death_source", scan)
    output = tmp_path / "output"
    report = module.run_native_death_cue_preflight(
        raw, tmp_path, tmp_path, ROOT / "configs/hierarchical_event_e1_health.json", output
    )
    assert set(opened) == set(paths[:3])
    assert report["status"] == "NATIVE_DEATH_CUE_PREFLIGHT_SUPPORTED"
    assert report["positive_sessions"] == 3
    assert report["video_test_opened"] is report["reward_allowed"] is False
    assert report["input_commands_sent"] == report["model_runs"] == 0
    with pytest.raises(ValueError, match="already exists"):
        module.run_native_death_cue_preflight(
            raw, tmp_path, tmp_path, ROOT / "configs/hierarchical_event_e1_health.json", output
        )


def test_houyi_data_audit_filters_test_before_container_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    from hok_agent import movement_real_rgb as module
    from hok_agent import pre_ingest, v5_data

    raw = tmp_path / "raw"
    raw.mkdir()
    paths = [raw / f"{index}.mp4" for index in range(3)]
    for path in paths:
        path.write_bytes(b"dummy")
    candidates = pre_ingest._candidate_list(paths, raw)
    splits = {
        candidates[0].candidate_id: "train",
        candidates[1].candidate_id: "dev",
        candidates[2].candidate_id: "test",
    }
    monkeypatch.setattr(
        v5_data,
        "load_automatic_cohort",
        lambda *_args: SimpleNamespace(session_splits=splits, cohort_sha256="4" * 64),
    )
    monkeypatch.setattr(
        pre_ingest,
        "load_pre_ingest",
        lambda _path: SimpleNamespace(pre_ingest_sha256="5" * 64),
    )
    opened = []

    class Container:
        metadata = {"creation_time": "2026-01-01", "encoder": "generic"}
        streams = [SimpleNamespace(metadata={"handler_name": "VideoHandler"})]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def open_container(handle, *, mode):
        assert mode == "r"
        opened.append(handle)
        return Container()

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_container))
    profile = tmp_path / "hero-profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "hok-agent-hero-ability-profile-v1",
                "profile_status": "TEMPLATE_NOT_CONFIGURED",
                "hero_id": "",
            }
        )
    )
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "summary.json").write_text(json.dumps({"hero": "houyi"}))
    output = tmp_path / "audit"
    report = module.run_houyi_data_binding_audit(
        raw, tmp_path, tmp_path, profile, summaries, output
    )
    assert len(opened) == 2
    assert report["opened_container_metadata"] == {"train": 1, "dev": 1}
    assert report["cohort_split_counts"] == {"train": 1, "dev": 1, "test": 1}
    assert report["test_container_metadata_opened"] == 0
    assert report["test_frames_decoded"] == report["video_frames_decoded"] == 0
    assert report["metadata_keyword_hits"] == []
    assert report["metadata_keys"] == {
        "creation_time": 2,
        "encoder": 2,
        "stream.handler_name": 2,
    }
    assert all(isinstance(value, int) for value in report["metadata_keys"].values())
    assert report["summary_hero_fields"] == 1
    assert report["complete_real_houyi_bindings"] == 0
    assert report["status"] == "HOUYI_BOUND_REAL_DATA_NOT_AVAILABLE"
    assert report["training_allowed"] is report["hero_data_contract_allowed"] is False
    assert str(tmp_path) not in (output / "report.json").read_text()
    with pytest.raises(ValueError, match="already exists"):
        module.run_houyi_data_binding_audit(raw, tmp_path, tmp_path, profile, summaries, output)


CONTRACT = ROOT / "configs" / "movement_real_rgb_preflight_v1.json"
GOAL_CONTRACT = ROOT / "configs" / "movement_real_rgb_goal_canvas_v2.json"
PLAYER_CONTRACT = ROOT / "configs" / "movement_real_player_cue_v1.json"
CONTINUITY_CONTRACT = ROOT / "configs" / "movement_real_player_goal_continuity_v1.json"
COUNTERFACTUAL_DATA_CONTRACT = (
    ROOT / "configs" / "movement_real_counterfactual_overfit32_data_v1.json"
)
LOCALIZATION_AUDIT_CONTRACT = ROOT / "configs" / "movement_real_player_localization_audit_v2.json"


def test_appearance_tracks_without_red_and_abstains_then_reacquires() -> None:
    from hok_agent.movement_real_rgb import _track_appearance

    template = np.full((15, 15, 3), 30, dtype=np.uint8)
    yy, xx = np.indices((15, 15))
    distance = np.hypot(yy - 7, xx - 7)
    template[(distance >= 5) & (distance <= 7)] = (30, 200, 50)
    template[4:7, 5:10] = (80, 80, 130)
    frames = np.full((8, 128, 128, 3), 30, dtype=np.uint8)
    for i, frame in enumerate(frames):
        frame[108:116, 18:26] = (30, 200, 50)  # green non-portrait distractor
        frame[1:16, 111:126] = template  # identical but excluded fixed UI
        if i != 3:
            y, x = (50 + i, 60) if i < 5 else (85, 90)
            frame[y - 7 : y + 8, x - 7 : x + 8] = template
    positions, reasons, _ = _track_appearance(frames, template)
    assert positions[0] is None and positions[1] == (51, 60)
    assert positions[2] == (52, 60)
    assert positions[3:6] == [None, None, None]
    assert positions[6:] == [(85, 90), (85, 90)]
    assert reasons[4] == reasons[5] == "confirming"


def test_appearance_rejects_two_equal_portraits() -> None:
    from hok_agent.movement_real_rgb import _track_appearance

    template = np.zeros((15, 15, 3), dtype=np.uint8)
    template[3:11, 3:11] = (20, 180, 40)
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[30:45, 30:45] = template
    frame[70:85, 70:85] = template
    positions, reasons, _ = _track_appearance(np.stack([frame] * 3), template)
    assert positions == [None] * 3
    assert reasons == ["ambiguous"] * 3


def test_tracking_cli_dispatch_is_lazy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    from hok_agent import cli, movement_real_rgb

    before = set(sys.modules)
    calls = []
    monkeypatch.setattr(
        movement_real_rgb,
        "run_real_player_tracking_audit",
        lambda *args: calls.append(args) or {"status": "DIAGNOSTIC_ONLY"},
    )
    assert (
        cli.main(
            [
                "movement-mvp",
                "--mode",
                "real-player-tracking-audit",
                "--config",
                "source.json",
                "--prior-report",
                "prior.json",
                "--session-root",
                "sessions",
                "--output-dir",
                "out",
            ]
        )
        == 0
    )
    assert calls == [(Path("source.json"), Path("prior.json"), Path("sessions"), Path("out"))]
    assert json.loads(capsys.readouterr().out)["status"] == "DIAGNOSTIC_ONLY"
    assert not any(
        name.startswith(("torch", "av", "hok_agent.mobile_testbed", "hok_agent.movement_mvp_train"))
        for name in set(sys.modules) - before
    )


def test_tracking_audit_bound_offline_source_no_checkpoint(tmp_path: Path) -> None:
    from hok_agent.movement_real_rgb import run_real_player_tracking_audit

    root, prior, contract = _localization_audit_inputs(tmp_path, response_events=True)
    audit = tmp_path / "v2"
    run_real_player_localization_audit_v2(contract, prior, root, audit)
    output = tmp_path / "tracking"
    report = run_real_player_tracking_audit(contract, audit / "report.json", root, output)
    assert len(report["sessions"]) == 3
    assert report["template_session"] == "teacher-session-002"
    assert len(report["template_frame_indices"]) == 16
    assert report["model_runs"] == report["input_commands_sent"] == 0
    assert report["training_allowed"] is report["r2_allowed"] is False
    assert report["false_lock_rate"] is None
    assert {p.suffix for p in output.iterdir()} == {".json", ".png"}
    with pytest.raises(ValueError, match="already exists"):
        run_real_player_tracking_audit(contract, audit / "report.json", root, output)
    shard = root / "teacher-session-002/shards/observations-0000.npz"
    shard.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="shard binding"):
        run_real_player_tracking_audit(contract, audit / "report.json", root, tmp_path / "bad")


def _dataset(tmp_path: Path, *, visible: bool) -> tuple[Path, Path]:
    root = tmp_path / "target"
    shards = root / "shards"
    shards.mkdir(parents=True)
    base_contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    sessions = base_contract["sessions"]
    session_rows = []
    shard_rows = []
    for ordinal, session in enumerate(sessions):
        identity, split = session["session_hash"], session["split"]
        frames = np.zeros((128, 128, 128, 3), dtype=np.uint8)
        frames[:, 24:104] = (30, 40, 30)
        if visible:
            for index, frame in enumerate(frames):
                frame[27:30, 10:13] = (20, 180, 40)
                target_x = 19 + index % 2
                frame[30:33, target_x : target_x + 2] = (200, 40, 30)
        frame_hash = np.asarray(
            [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames], dtype="U64"
        )
        basename = f"{ordinal:06d}-alignment-000000.npz"
        path = shards / basename
        np.savez_compressed(
            path,
            frames=frames,
            session_hash=np.asarray([identity] * len(frames), dtype="U64"),
            timestamp_ms=np.arange(len(frames), dtype=np.int64) * 100,
            rotation_degrees=np.zeros(len(frames), dtype=np.int16),
            frame_hash=frame_hash,
            split=np.asarray([split] * len(frames), dtype="U5"),
        )
        session_rows.append({"session_hash": identity, "split": split})
        shard_rows.append(
            {
                "path": basename,
                "row_count": len(frames),
                "session_hashes": [identity],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source": "target",
                "split": split,
            }
        )
    session_rows.append({"session_hash": "f" * 64, "split": "test"})
    shard_rows.append(
        {
            "path": "must-not-open.npz",
            "row_count": 1,
            "session_hashes": ["f" * 64],
            "sha256": "0" * 64,
            "source": "target",
            "split": "test",
        }
    )
    manifest = {
        "schema_version": "hok-agent-v5-manifest-v2",
        "sessions": session_rows,
        "shards": shard_rows,
    }
    manifest["manifest_sha256"] = _object_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    base_contract["target_manifest_sha256"] = manifest["manifest_sha256"]
    base_contract.pop("contract_sha256")
    base_contract["contract_sha256"] = _object_sha256(base_contract)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(base_contract), encoding="utf-8")
    return root, contract_path


def _goal_contract(tmp_path: Path, target: Path) -> tuple[Path, Path]:
    prior = {
        "schema_version": "movement-real-rgb-observability-report-v1",
        "status": "TARGET_CONDITION_NOT_OBSERVABLE",
        "contract_sha256": "d3755cb682c425dff4e55fc6f7c571b13a7f1ffc63e834899ed8630d14c946ce",
    }
    prior["report_sha256"] = _object_sha256(prior)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    contract = json.loads(GOAL_CONTRACT.read_text(encoding="utf-8"))
    contract["prior_report_sha256"] = prior["report_sha256"]
    contract["target_manifest_sha256"] = manifest["manifest_sha256"]
    contract.pop("contract_sha256")
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "goal-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path, prior_path


def _localization_audit_inputs(tmp_path: Path, *, response_events: bool) -> tuple[Path, Path, Path]:
    session_root = tmp_path / "localization-sessions"
    declarations = []
    for ordinal in range(3):
        basename = f"teacher-session-00{2 if ordinal == 0 else 3 if ordinal == 1 else 5}"
        directory = session_root / basename
        (directory / "shards").mkdir(parents=True)
        frames = np.zeros((192, 128, 128, 3), dtype=np.uint8)
        for index, frame in enumerate(frames):
            frame[6:14, 116:124] = (20, 180, 40)
            frame[7:13, 111:117] = (200, 40, 30)
            if ordinal == 0:
                y = 40 + index // 5
                frame[y : y + 8, 52:60] = (20, 180, 40)
                frame[y + 1 : y + 7, 60:66] = (200, 40, 30)
        sent = np.zeros(len(frames), dtype=np.uint8)
        if ordinal == 0 and response_events:
            sent[np.arange(0, 110, 10)] = 1
        shard = directory / "shards" / "observations-0000.npz"
        np.savez_compressed(
            shard,
            minimap_rgb=frames,
            main_rgb=frames,
            scheduled_elapsed_ms=np.arange(len(frames), dtype=np.int64) * 200,
            movement_id=np.full(len(frames), 5, dtype=np.int8),
            movement_input_sent=sent,
        )
        summary = {
            "status": "PASSED",
            "derived_roi_rgb_persisted": True,
            "raw_frames_persisted": False,
            "observation_shards": [
                {
                    "path": shard.name,
                    "rows": len(frames),
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                }
            ],
        }
        summary["summary_sha256"] = _object_sha256(summary)
        summary_path = directory / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        declarations.append(
            {
                "basename": basename,
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            }
        )
    prior = {"status": "REAL_PLAYER_CUE_PASSED"}
    prior["report_sha256"] = _object_sha256(prior)
    prior_path = tmp_path / "old-player-report.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    contract = json.loads(LOCALIZATION_AUDIT_CONTRACT.read_text(encoding="utf-8"))
    contract["sessions"] = declarations
    contract["prior_report_file_sha256"] = hashlib.sha256(prior_path.read_bytes()).hexdigest()
    contract["prior_report_sha256"] = prior["report_sha256"]
    contract.pop("contract_sha256")
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "localization-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return session_root, prior_path, contract_path


def test_real_rgb_preflight_reads_only_selected_train_dev(tmp_path: Path) -> None:
    target, contract = _dataset(tmp_path, visible=True)
    output = tmp_path / "report"
    report = run_real_rgb_preflight(contract, target, output)
    assert report["status"] == "TARGET_CONDITION_CANDIDATE_SUPPORTED"
    assert report["selected_sessions"] == 3
    assert report["selected_segments"] == 9
    assert report["sampled_frames"] == 288
    assert report["opened_splits"] == ["dev", "train"]
    assert report["test_frames_read"] == 0
    assert report["overall_pair_coverage"] == 1.0
    assert report["marker_jump_fraction"] == 0.0
    assert report["semantic_accuracy_verified"] is False
    assert report["promotion_allowed"] is False
    assert report["r2_allowed"] is False
    assert report["raw_rgb_persisted"] is False
    assert {path.name for path in output.iterdir()} == {"report.json"}
    assert not (target / "shards" / "must-not-open.npz").exists()


def test_localization_v2_excludes_fixed_ui_and_requires_action_response(
    tmp_path: Path,
) -> None:
    session_root, prior, contract = _localization_audit_inputs(tmp_path, response_events=True)
    output = tmp_path / "localization-audit"
    report = run_real_player_localization_audit_v2(contract, prior, session_root, output)
    assert report["status"] == "PLAYER_CUE_PARTIAL_SESSION002_ONLY"
    assert all(report["checks"].values())
    sessions = {row["session"]: row for row in report["sessions"]}
    assert sessions["teacher-session-002"]["filtered_candidate_coverage"] == 1.0
    assert sessions["teacher-session-002"]["response_events"] == 11
    assert sessions["teacher-session-002"]["positive_projection_fraction"] == 1.0
    assert sessions["teacher-session-003"]["raw_candidate_coverage"] == 1.0
    assert sessions["teacher-session-003"]["filtered_candidate_coverage"] == 0.0
    assert sessions["teacher-session-003"]["rejected_ui_candidate_coverage"] == 1.0
    assert sessions["teacher-session-005"]["filtered_candidate_coverage"] == 0.0
    assert len(report["contact_sheets"]) == 3
    assert {path.name for path in output.iterdir()} == {
        "report.json",
        "teacher-session-002-contact.png",
        "teacher-session-003-contact.png",
        "teacher-session-005-contact.png",
    }

    no_response_root, no_response_prior, no_response_contract = _localization_audit_inputs(
        tmp_path / "no-response", response_events=False
    )
    failed = run_real_player_localization_audit_v2(
        no_response_contract,
        no_response_prior,
        no_response_root,
        tmp_path / "no-response-output",
    )
    assert failed["status"] == "PLAYER_CUE_V2_FAILED"
    assert failed["checks"]["session002_response_events"] is False


def test_real_rgb_preflight_reports_non_observable_without_training(tmp_path: Path) -> None:
    target, contract = _dataset(tmp_path, visible=False)
    report = run_real_rgb_preflight(contract, target, tmp_path / "failed")
    assert report["status"] == "TARGET_CONDITION_NOT_OBSERVABLE"
    assert report["overall_pair_coverage"] == 0.0
    assert report["training_called"] is False
    assert report["device_input_commands_sent"] == 0
    assert report["next_action"] == "repair_content_geometry_or_minimap_detector_in_new_contract"


def test_real_rgb_preflight_detects_portrait_content_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    contract_values = json.loads(CONTRACT.read_text(encoding="utf-8"))
    frames = np.zeros((4, 128, 128, 3), dtype=np.uint8)
    frames[:, :, 35:93] = (30, 40, 30)
    canonical, orientation, bounds = _canonical_content(frames, contract_values["content_box"])
    assert canonical.shape == frames.shape
    assert orientation == "counter_clockwise_90"
    assert bounds == (0, 35, 128, 93)

    target, contract = _dataset(tmp_path, visible=True)
    selected = target / "shards" / "000000-alignment-000000.npz"
    selected.write_bytes(selected.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="shard binding differs"):
        run_real_rgb_preflight(contract, target, tmp_path / "tampered-output")


def test_real_rgb_goal_canvas_is_deterministic_and_counterfactual(tmp_path: Path) -> None:
    target, _preflight_contract = _dataset(tmp_path, visible=False)
    contract, prior = _goal_contract(tmp_path, target)
    output = tmp_path / "goal-output"
    report = run_real_rgb_goal_canvas(contract, prior, target, output)
    assert report["status"] == "GOAL_CANVAS_GENERATION_PASSED_SELF_LOCALIZATION_UNRESOLVED"
    assert report["sampled_frames"] == 288
    assert all(report["checks"].values())
    assert report["target_detection_required"] is False
    assert report["player_localization_verified"] is False
    assert report["semantic_lane_coordinate_verified"] is False
    assert report["training_called"] is False
    assert report["r2_allowed"] is False
    assert report["raw_rgb_persisted"] is False
    assert all(row["counterfactual_changed"] for row in report["frame_results"])
    assert all(row["deterministic_repeat"] for row in report["frame_results"])
    assert {path.name for path in output.iterdir()} == {"report.json"}


def test_real_player_cue_uses_existing_minimap_shards_without_labels(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    contract = json.loads(PLAYER_CONTRACT.read_text(encoding="utf-8"))
    declarations = []
    for ordinal in range(3):
        basename = f"teacher-session-{ordinal:03d}"
        directory = session_root / basename
        (directory / "shards").mkdir(parents=True)
        frames = np.zeros((192, 128, 128, 3), dtype=np.uint8)
        for index, frame in enumerate(frames):
            offset = index % 5
            frame[60:68, 60 + offset : 68 + offset] = (20, 180, 40)
            frame[61:67, 68 + offset : 74 + offset] = (200, 40, 30)
        shard = directory / "shards" / "observations-0000.npz"
        np.savez_compressed(
            shard,
            minimap_rgb=frames,
            scheduled_elapsed_ms=np.arange(len(frames), dtype=np.int64) * 200,
        )
        summary = {
            "status": "PASSED",
            "derived_roi_rgb_persisted": True,
            "raw_frames_persisted": False,
            "observation_shards": [
                {
                    "path": shard.name,
                    "rows": len(frames),
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                }
            ],
        }
        summary["summary_sha256"] = _object_sha256(summary)
        summary_path = directory / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        declarations.append(
            {
                "basename": basename,
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            }
        )
    contract["sessions"] = declarations
    contract.pop("contract_sha256")
    contract["contract_sha256"] = _object_sha256(contract)
    contract_path = tmp_path / "player-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    report = run_real_player_cue_preflight(contract_path, session_root, tmp_path / "player-output")
    assert report["status"] == "REAL_PLAYER_CUE_PASSED"
    assert all(row["coverage"] == 1.0 for row in report["sessions"])
    assert all(row["single_candidate_fraction"] == 1.0 for row in report["sessions"])
    assert report["semantic_identity_verified"] is False
    assert report["human_labels_consumed"] is False
    assert report["training_called"] is False
    assert report["device_input_commands_sent"] == 0

    player_report_path = tmp_path / "player-output" / "report.json"
    goal_report = {
        "schema_version": "movement-real-rgb-goal-canvas-report-v2",
        "status": "GOAL_CANVAS_GENERATION_PASSED_SELF_LOCALIZATION_UNRESOLVED",
    }
    goal_report["report_sha256"] = _object_sha256(goal_report)
    goal_report_path = tmp_path / "goal-report.json"
    goal_report_path.write_text(json.dumps(goal_report), encoding="utf-8")
    continuity = json.loads(CONTINUITY_CONTRACT.read_text(encoding="utf-8"))
    continuity["sessions"] = declarations
    continuity["lineage"] = {
        "player_report_file_sha256": hashlib.sha256(player_report_path.read_bytes()).hexdigest(),
        "player_report_sha256": report["report_sha256"],
        "goal_report_file_sha256": hashlib.sha256(goal_report_path.read_bytes()).hexdigest(),
        "goal_report_sha256": goal_report["report_sha256"],
    }
    continuity.pop("contract_sha256")
    continuity["contract_sha256"] = _object_sha256(continuity)
    continuity_path = tmp_path / "continuity-contract.json"
    continuity_path.write_text(json.dumps(continuity), encoding="utf-8")
    combined = run_real_player_goal_continuity(
        continuity_path,
        player_report_path,
        goal_report_path,
        session_root,
        tmp_path / "continuity-output",
    )
    assert combined["status"] == "REAL_PLAYER_GOAL_CONTINUITY_PASSED"
    assert all(combined["checks"].values())
    assert all(row["raw_direction_coverage"] == 1.0 for row in combined["sessions"])
    assert all(row["stable_direction_coverage"] >= 0.9 for row in combined["sessions"])
    assert combined["direction_accuracy_verified"] is False
    assert combined["continuity_only"] is True
    assert combined["policy_training_allowed"] is False
    assert combined["training_called"] is False
    assert combined["test_frames_read"] == 0
    assert combined["device_input_commands_sent"] == 0

    continuity_report_path = tmp_path / "continuity-output" / "report.json"
    data_contract = json.loads(COUNTERFACTUAL_DATA_CONTRACT.read_text(encoding="utf-8"))
    data_contract["sessions"] = declarations
    data_contract["continuity_report_file_sha256"] = hashlib.sha256(
        continuity_report_path.read_bytes()
    ).hexdigest()
    data_contract["continuity_report_sha256"] = combined["report_sha256"]
    data_contract.pop("contract_sha256")
    data_contract["contract_sha256"] = _object_sha256(data_contract)
    data_contract_path = tmp_path / "counterfactual-data-contract.json"
    data_contract_path.write_text(json.dumps(data_contract), encoding="utf-8")
    data_report = materialize_real_counterfactual_overfit32(
        data_contract_path,
        continuity_report_path,
        session_root,
        tmp_path / "counterfactual-data",
    )
    assert data_report["status"] == "PASSED"
    assert data_report["samples"] == 32
    assert data_report["derived_rgb_frames"] == 512
    assert data_report["counterfactual_classes_verified"] == 9
    assert data_report["unique_source_windows"] == 5
    assert data_report["source_windows_nonoverlapping"] is True
    assert data_report["samples_reuse_source_windows_with_different_goals"] is True
    assert data_report["cross_session_windows"] == 0
    assert data_report["labels_are_executed_actions"] is False
    assert data_report["labels_are_geometric_counterfactuals"] is True
    assert data_report["training_called"] is False
    with np.load(tmp_path / "counterfactual-data" / "overfit32.npz") as dataset:
        assert dataset["rgb_sequence"].shape == (32, 16, 128, 128, 3)
        assert sorted(dataset["label"].tolist()) == [0] * 8 + [
            label for label in range(1, 9) for _ in range(3)
        ]

    goal_report["status"] = "tampered"
    goal_report_path.write_text(json.dumps(goal_report), encoding="utf-8")
    with pytest.raises(ValueError, match="self hash differs"):
        run_real_player_goal_continuity(
            continuity_path,
            player_report_path,
            goal_report_path,
            session_root,
            tmp_path / "tampered-continuity-output",
        )
