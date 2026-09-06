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
CONTRACT = ROOT / "configs" / "movement_real_rgb_preflight_v1.json"
GOAL_CONTRACT = ROOT / "configs" / "movement_real_rgb_goal_canvas_v2.json"
PLAYER_CONTRACT = ROOT / "configs" / "movement_real_player_cue_v1.json"
CONTINUITY_CONTRACT = ROOT / "configs" / "movement_real_player_goal_continuity_v1.json"
COUNTERFACTUAL_DATA_CONTRACT = (
    ROOT / "configs" / "movement_real_counterfactual_overfit32_data_v1.json"
)
LOCALIZATION_AUDIT_CONTRACT = (
    ROOT / "configs" / "movement_real_player_localization_audit_v2.json"
)


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


def _localization_audit_inputs(
    tmp_path: Path, *, response_events: bool
) -> tuple[Path, Path, Path]:
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
    session_root, prior, contract = _localization_audit_inputs(
        tmp_path, response_events=True
    )
    output = tmp_path / "localization-audit"
    report = run_real_player_localization_audit_v2(
        contract, prior, session_root, output
    )
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
    report = run_real_player_cue_preflight(
        contract_path, session_root, tmp_path / "player-output"
    )
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
