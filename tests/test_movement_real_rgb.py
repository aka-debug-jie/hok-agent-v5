from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.movement_real_rgb import (
    _canonical_content,
    _object_sha256,
    run_real_rgb_goal_canvas,
    run_real_rgb_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "movement_real_rgb_preflight_v1.json"
GOAL_CONTRACT = ROOT / "configs" / "movement_real_rgb_goal_canvas_v2.json"


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
    canonical, orientation, bounds = _canonical_content(
        frames, contract_values["content_box"]
    )
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
