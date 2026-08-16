from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from hok_agent import t8 as t8_module
from hok_agent.mobile_testbed import (
    ABILITIES,
    AIMS,
    DEMONSTRATOR_DATA_SCHEMA,
    DEMONSTRATOR_SCHEMA,
    DEMONSTRATOR_SESSION_SCHEMA,
    LAYOUT_SCHEMA,
    MOVEMENTS,
    TARGETS,
)
from hok_agent.t8 import (
    CONFIG,
    V2_HEAD_SIZES,
    FactorizedTemporalActor,
    T8Error,
    T8V2FactorizedActor,
    _offline_metrics,
    _v2_class_weights,
    _v2_image_tensor,
    _v2_legal_prediction,
    _v2_metrics,
    _v2_target_index,
    _V2VideoAdapter,
    freeze_t8_split,
    load_t8_data,
)
from hok_agent.t8_v3 import (
    V3_STATE_NAMES,
    V27_FREEZE_SCHEMA,
    V3HybridExecutor,
    V3StateTemporal,
    _v3_decode,
    _v3_labels,
    freeze_t8_v27_failures,
)

LAYOUT_HASH = "a" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layout_file(tmp_path: Path) -> Path:
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": LAYOUT_SCHEMA,
                "screen": {"width": 1600, "height": 720},
                "joystick": {
                    "center": [0.2, 0.8],
                    "radius": 0.12,
                    "forward_vector": [0.0, -1.0],
                    "move_hold_ms": 150,
                    "skill_hold_ms": 250,
                    "aim_radius": 0.18,
                },
                "buttons": {
                    "basic_attack": [0.8, 0.8],
                    "skill1": [0.7, 0.8],
                    "skill2": [0.75, 0.75],
                    "skill3": [0.8, 0.7],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _session(root: Path, name: str, count: int = 90) -> None:
    path = root / name
    path.mkdir()
    actions = [(0, 0, 0, 0, 0)]
    actions.extend((direction, 0, 0, 0, 150) for direction in range(1, len(MOVEMENTS)))
    actions.append((0, 1, 0, 0, 250))
    actions.extend(
        (0, ability, aim, target, 250)
        for ability in range(2, len(ABILITIES))
        for aim in range(1, len(AIMS))
        for target in (0,)
    )
    repeated = (actions * ((count + len(actions) - 1) // len(actions)))[:count]
    movement = np.asarray([action[0] for action in repeated], dtype=np.int8)
    ability = np.asarray([action[1] for action in repeated], dtype=np.int8)
    aim = np.asarray([action[2] for action in repeated], dtype=np.int8)
    target = np.asarray([action[3] for action in repeated], dtype=np.int8)
    hold = np.asarray([action[4] for action in repeated], dtype=np.uint16)
    sent = np.asarray(
        [
            int(not (move == 0 and skill == 0))
            for move, skill in zip(movement, ability, strict=True)
        ],
        dtype=np.uint8,
    )
    timestamps = np.arange(count, dtype=np.int64)
    np.savez_compressed(
        path / "samples-00000.npz",
        frames=np.zeros((count, 8, 128, 128, 3), dtype=np.uint8),
        movement=movement,
        ability=ability,
        aim=aim,
        target=target,
        hold_ms=hold,
        timestamp_ns=timestamps,
        input_sent=sent,
    )
    events = []
    for index in range(count):
        events.append(
            {
                "schema_version": DEMONSTRATOR_SCHEMA,
                "sequence": index,
                "timestamp_ns": int(timestamps[index]),
                "action": {
                    "movement": MOVEMENTS[int(movement[index])],
                    "ability": ABILITIES[int(ability[index])],
                    "aim": AIMS[int(aim[index])],
                    "target": TARGETS[int(target[index])],
                    "hold_ms": int(hold[index]),
                },
                "input_sent": bool(sent[index]),
            }
        )
    (path / "events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events), encoding="utf-8"
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": DEMONSTRATOR_SCHEMA,
                "dataset_schema_version": DEMONSTRATOR_DATA_SCHEMA,
                "status": "COMPLETED",
                "capture_mode": "scrcpy-v4l2",
                "duration_seconds": 300.0,
                "window_frames": 8,
                "samples": count,
                "layout_sha256": LAYOUT_HASH,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": DEMONSTRATOR_SESSION_SCHEMA,
        "summary_sha256": _sha(path / "summary.json"),
        "events_sha256": _sha(path / "events.jsonl"),
        "shards": [{"name": "samples-00000.npz", "sha256": _sha(path / "samples-00000.npz")}],
    }
    manifest["session_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (path / "session-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    dataset = root / "datasets" / "t8-demonstrations-v1"
    dataset.mkdir(parents=True)
    for index in range(1, 9):
        _session(dataset, f"session-{index:03d}")
    freeze_t8_split(dataset_root=dataset, output_path=dataset / "t8-split-v2.json")
    return dataset


def test_t8_loader_uses_frozen_4_2_2_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = load_t8_data(_dataset(tmp_path, monkeypatch))
    assert {name: len(indices) for name, indices in data.splits.items()} == {
        "train": 4,
        "dev": 2,
        "test": 2,
    }


def test_t8_loader_rejects_event_or_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, monkeypatch)
    event = dataset / "session-001" / "events.jsonl"
    event.write_text(
        event.read_text(encoding="utf-8").replace('"sequence": 0', '"sequence": 9', 1),
        encoding="utf-8",
    )
    with pytest.raises(T8Error, match="execution events"):
        load_t8_data(dataset)


def test_t8_actor_is_rgb_only_and_has_five_factor_heads() -> None:
    actor = FactorizedTemporalActor().eval()
    with pytest.raises(T8Error, match="RGB only"):
        actor(np.zeros((1, 3, 128, 128), dtype=np.float32))  # type: ignore[arg-type]
    import torch

    with torch.no_grad():
        outputs = actor(torch.zeros((1, int(CONFIG["frames"]), 3, 128, 128)))
    assert [output.shape[1] for output in outputs] == CONFIG["heads"]


def test_t8_offline_metrics_keep_session_switches_and_illegal_pairs_visible() -> None:
    target = np.asarray(
        [[0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 2], [0, 0, 0, 0, 0]],
        dtype=np.int64,
    )
    predicted = target.copy()
    predicted[1, 1] = 1
    metrics = _offline_metrics(predicted, target, (2, 2))
    assert metrics["true_switch_rate"] == 1.0
    assert metrics["predicted_switch_rate"] == 1.0
    assert metrics["illegal_joint_predictions"] == 1
    assert metrics["heads"]["movement"]["confusion"][1][1] == 1


def test_t8_v2_actor_is_16_frame_rgb_and_legal_decoder_masks_invalid_joint_actions() -> None:
    import torch

    actor = T8V2FactorizedActor().eval()
    with pytest.raises(T8Error, match="Bx16"):
        actor(torch.zeros((1, 8, 3, 128, 128)))
    with torch.no_grad():
        outputs = actor(torch.zeros((1, 16, 3, 128, 128)))
    assert tuple(output.shape[1] for output in outputs) == V2_HEAD_SIZES
    logits = tuple(torch.full((1, size), -20.0) for size in V2_HEAD_SIZES)
    logits[0][0, 0] = 20.0
    logits[1][0, 0] = 80.0
    logits[2][0, 4] = 20.0
    logits[3][0, 3] = 20.0
    assert _v2_legal_prediction(logits).tolist() == [[0, 0, 0, 0]]


def test_t8_v2_conditional_heads_require_and_score_only_active_classes() -> None:
    import torch
    labels = np.asarray(
        [[m, c, a, h, 200] for m in range(len(MOVEMENTS)) for c in range(len(ABILITIES))
         for a in range(1, len(AIMS)) for h in range(1, 4)], dtype=np.int64
    )
    weights = _v2_class_weights(labels, torch.device("cpu"))
    assert all(bool(torch.isfinite(value).all()) for value in weights)
    heads = cast(dict[str, dict[str, object]], _v2_metrics(labels[:, :4], labels)["heads"])
    assert (heads["aim"]["macro_f1"], heads["hold"]["macro_f1"]) == (1.0, 1.0)

def test_t8_v2_adapter_accepts_4d_rgb_and_encodes_once() -> None:
    import torch
    from torch import nn

    actor = T8V2FactorizedActor()
    adapter = _V2VideoAdapter(actor.encoder.state_dict()).eval()

    class CountEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, frames: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.zeros((len(frames), 512))

    counter = CountEncoder()
    adapter.encoder = counter
    tensor = _v2_image_tensor(np.zeros((2, 128, 128, 3), dtype=np.uint8), torch.device("cpu"))
    with torch.no_grad():
        projection, prediction = adapter(tensor)
    assert counter.calls == 1 and projection.shape == prediction.shape == (2, 128)
    with pytest.raises(T8Error, match="Nx128"):
        _v2_image_tensor(np.zeros((2, 16, 128, 128, 3), dtype=np.uint8), torch.device("cpu"))


def test_t8_v2_video_index_is_per_session_and_never_opens_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    target = root / "target"
    shards = target / "shards"
    shards.mkdir(parents=True)
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    rows = []
    for number in range(2):
        path = shards / f"train-{number}.npz"
        np.savez_compressed(
            path,
            frames=np.zeros((4, 128, 128, 3), dtype=np.uint8),
            session_hash=np.asarray(["a" * 64] * 4),
            split=np.asarray(["train"] * 4),
        )
        rows.append({"path": path.name, "split": "train", "sha256": _sha(path)})
    rows.append({"path": "missing-test.npz", "split": "test", "sha256": "x" * 64})
    unsigned = {"shards": rows}
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {**unsigned, "manifest_sha256": digest}
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    indexed, _, count = _v2_target_index(target, "train")
    assert count == 2 and [indices.tolist() for _, indices in indexed] == [[0], [1]]


def test_v21_freeze_has_no_session_upper_bound_and_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    dataset = root / "datasets" / "t8-demonstrations-v2.1"
    dataset.mkdir(parents=True)
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    labels = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 1, 200],
            [0, 1, 0, 1, 200],
            [1, 2, 1, 2, 500],
        ],
        dtype=np.int64,
    )
    sessions = tuple(
        t8_module.V2Session(
            f"{index:064x}",
            dataset / f"session-{index + 1:03d}",
            "a" * 64,
            "b" * 64,
            t8_module.SCRCPY_EXECUTED_ACTION_SOURCE,
            np.zeros((len(labels), 16, 128, 128, 3), dtype=np.uint8),
            labels,
        )
        for index in range(13)
    )
    monkeypatch.setattr(t8_module, "_v21_sessions", lambda _root: sessions)
    first = t8_module.freeze_t8_v21_split(
        dataset_root=dataset, output_path=dataset / "split-a.json"
    )
    second = t8_module.freeze_t8_v21_split(
        dataset_root=dataset, output_path=dataset / "split-b.json"
    )
    assert first == second
    assert first["session_count"] == 13
    splits = cast(dict[str, list[dict[str, str]]], first["splits"])
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 8,
        "dev": 2,
        "test": 3,
    }
    assert len({row["name"] for rows in splits.values() for row in rows}) == 13


@pytest.mark.parametrize("count", [12, 13, 14, 17, 101])
def test_v21_split_allocates_every_session_once_without_an_upper_bound(count: int) -> None:
    splits = t8_module._v21_splits(count)
    allocated = [index for values in splits.values() for index in values]
    assert len(allocated) == count
    assert set(allocated) == set(range(count))
    assert all(splits[name] for name in ("train", "dev", "test"))


def test_inverse_probe_button_score_uses_local_before_after_change() -> None:
    before = np.zeros((128, 128, 3), dtype=np.uint8)
    after = np.zeros((3, 128, 128, 3), dtype=np.uint8)
    after[1, 90:103, 90:103] = 255
    scores = t8_module._button_change_scores(
        before,
        after,
        ((0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)),
    )
    assert scores.argmax() == 1
    assert scores[1] > 200


def test_video_three_class_materializer_abstains_skill3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    source = root / "source"
    shards = source / "shards"
    shards.mkdir(parents=True)
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    shard_path = shards / "train-0000.npz"
    rows = 4
    np.savez_compressed(
        shard_path,
        frames=np.zeros((rows, 16, 128, 128, 3), dtype=np.uint8),
        combat_id=np.arange(rows, dtype=np.int8),
        action_timestamp_ms=np.arange(rows, dtype=np.int64) + 100,
        observation_end_timestamp_ms=np.arange(rows, dtype=np.int64),
        session_hash=np.asarray(["a" * 64] * rows),
        causal_window_sha256=np.asarray(["b" * 64] * rows),
    )
    source_manifest = {
        "schema_version": "hok-agent-t8-video-combat-pseudolabel-candidates-v2",
        "combat_vocabulary": list(ABILITIES[1:]),
        "future_frames_included": False,
        "event_frame_included": False,
        "shards": [
            {
                "path": shard_path.name,
                "split": "train",
                "sha256": _sha(shard_path),
            }
        ],
    }
    (source / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    inverse = root / "inverse.json"
    inverse.write_text(
        json.dumps(
            {
                "schema_version": t8_module.INVERSE_PROBE_SCHEMA,
                "three_class_gate_passed": True,
                "three_class_scope": list(ABILITIES[1:4]),
                "three_class_abstained": ["skill3"],
            }
        ),
        encoding="utf-8",
    )
    result = t8_module.materialize_t8_video_three_class(
        source_dir=source,
        inverse_report_path=inverse,
        output_dir=root / "filtered",
    )
    assert result["rows"] == 3
    assert result["candidate_counts_by_split"]["train"] == {
        "basic_attack": 1,
        "skill1": 1,
        "skill2": 1,
    }
    with np.load(root / "filtered" / "shards" / "train-0000.npz") as filtered:
        assert filtered["combat_id"].tolist() == [0, 1, 2]
    frames, labels, digest = t8_module._load_video_three_class_split(
        root / "filtered", "train"
    )
    assert frames.shape == (3, 16, 128, 128, 3)
    assert labels.tolist() == [0, 1, 2]
    assert digest == result["manifest_sha256"]


def test_video_three_class_temporal_accepts_frozen_features() -> None:
    model = t8_module._VideoCombatTemporal()
    output = model(torch.zeros(2, 16, 512))
    assert output.shape == (2, 3)


def test_video_retrospective_materializer_keeps_event_frame_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    source = root / "source"
    shards = source / "shards"
    shards.mkdir(parents=True)
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    shard_path = shards / "dev-0000.npz"
    np.savez_compressed(
        shard_path,
        frames=np.zeros((4, 16, 128, 128, 3), dtype=np.uint8),
        combat_id=np.arange(4, dtype=np.int8),
        timestamp_ms=np.arange(4, dtype=np.int64),
        session_hash=np.asarray(["a" * 64] * 4),
        causal_window_sha256=np.asarray(["b" * 64] * 4),
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "hok-agent-t8-video-combat-pseudolabel-candidates-v1",
                "combat_vocabulary": list(ABILITIES[1:]),
                "future_frames_included": False,
                "causal_window_includes_action_frame": True,
                "shards": [
                    {
                        "path": shard_path.name,
                        "split": "dev",
                        "sha256": _sha(shard_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    inverse = root / "inverse.json"
    inverse.write_text(
        json.dumps(
            {
                "schema_version": t8_module.INVERSE_PROBE_SCHEMA,
                "three_class_gate_passed": True,
                "three_class_scope": list(ABILITIES[1:4]),
                "three_class_abstained": ["skill3"],
            }
        ),
        encoding="utf-8",
    )
    result = t8_module.materialize_t8_video_three_class(
        source_dir=source,
        inverse_report_path=inverse,
        output_dir=root / "retrospective",
        retrospective=True,
    )
    assert result["task"] == "retrospective_action_recognition"
    assert result["event_frame_included"] is True
    frames, labels, _digest = t8_module._load_video_three_class_split(
        root / "retrospective", "dev", retrospective=True
    )
    assert len(frames) == 3 and labels.tolist() == [0, 1, 2]


def test_retrospective_roi_prediction_uses_frozen_button_centers() -> None:
    frames = np.zeros((16, 128, 128, 3), dtype=np.uint8)
    frames[-1, 46:55, 74:87] = 255
    predicted, ratio = t8_module._retrospective_roi_prediction(
        frames, ((20, 20), (80, 50), (110, 90))
    )
    assert predicted == 1
    assert ratio > 100


def _retrospective_baseline(root: Path, layout_sha256: str = LAYOUT_HASH) -> Path:
    baseline = root / "baseline"
    baseline.mkdir(parents=True)
    payload = {
        "schema_version": t8_module.RETROSPECTIVE_BASELINE_SCHEMA,
        "status": "FROZEN",
        "recognizer": {
            "classes": list(ABILITIES[1:4]),
            "abstained_classes": ["skill3"],
            "window_frames": 16,
            "roi_radius_xy": [6, 4],
            "score": "mean_abs_delta_plus_mean_positive_delta",
            "production_min_top_to_second_ratio": 1.25,
        },
        "layout_sha256": layout_sha256,
        "test_accessed": False,
        "device_input_allowed": False,
        "raw_video_or_source_paths_persisted": False,
    }
    (baseline / "baseline.json").write_text(json.dumps(payload), encoding="utf-8")
    for index in range(10):
        (baseline / f"evidence-{index}.json").write_text("{}\n", encoding="utf-8")
    files = sorted(path for path in baseline.iterdir() if path.name != "SHA256SUMS")
    (baseline / "SHA256SUMS").write_text(
        "".join(f"{_sha(path)}  ./{path.name}\n" for path in files), encoding="utf-8"
    )
    return baseline


def test_retrospective_baseline_verifier_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    baseline = _retrospective_baseline(root)
    assert t8_module.verify_t8_retrospective_baseline(
        baseline_dir=baseline
    )["verified_files"] == 11
    (baseline / "evidence-0.json").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(T8Error, match="hash differs"):
        t8_module.verify_t8_retrospective_baseline(baseline_dir=baseline)


def test_retrospective_batch_emits_event_qc_without_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    target = root / "target"
    shards = target / "shards"
    shards.mkdir(parents=True)
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    layout_path = _layout_file(tmp_path)
    baseline = _retrospective_baseline(root, _sha(layout_path))
    rng = np.random.default_rng(9)
    frames = np.zeros((40, 128, 128, 3), dtype=np.uint8)
    content = rng.integers(40, 220, size=(58, 128, 3), dtype=np.uint8)
    frames[:, 35:93] = content
    frames[20, 79:88, 99:112] = 255
    identity = "a" * 64
    shard_path = shards / "train-0000.npz"
    np.savez_compressed(
        shard_path,
        frames=frames,
        timestamp_ms=np.arange(40, dtype=np.int64) * 100,
        frame_hash=np.asarray([f"{index:064x}" for index in range(40)]),
        session_hash=np.asarray([identity] * 40),
        split=np.asarray(["train"] * 40),
    )
    manifest: dict[str, object] = {
        "schema_version": t8_module.V5_TARGET_MANIFEST_SCHEMA,
        "sessions": [{"session_hash": identity, "split": "train"}],
        "shards": [
            {
                "path": shard_path.name,
                "sha256": _sha(shard_path),
                "split": "train",
                "session_hashes": [identity],
            }
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        t8_module._canonical(manifest)
    ).hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = t8_module.run_t8_retrospective_batch(
        target_dir=target,
        baseline_dir=baseline,
        layout_path=layout_path,
        split="train",
        output_dir=root / "output",
    )
    assert result["session_count"] == 1
    assert result["test_accessed"] is False
    event_text = (root / "output" / "events" / f"{identity}.jsonl").read_text()
    assert "basic_attack" in event_text
    assert "frames" not in event_text and "source_path" not in event_text
    qc = json.loads((root / "output" / "qc" / f"{identity}.json").read_text())
    assert qc["orientation"] == "stored"
    assert qc["raw_rgb_persisted"] is False
    verified = t8_module.verify_t8_retrospective_batch(batch_dir=root / "output")
    assert verified["sessions_verified"] == 1
    assert verified["events_verified"] == 1


def test_retrospective_batch_rejects_test_before_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    with pytest.raises(T8Error, match="train or dev only"):
        t8_module._retrospective_target_index(root / "missing", "test")


def test_retrospective_selective_metrics_counts_abstention_as_missed_recall() -> None:
    scores = np.asarray(
        [[9.0, 1.0, 1.0], [1.1, 1.0, 0.9], [1.0, 9.0, 1.0], [1.0, 1.0, 9.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 2], dtype=np.int64)
    metrics = t8_module._retrospective_selective_metrics(scores, labels, 3, 1.25)
    assert metrics["coverage"] == 0.75
    assert metrics["per_class_recall"] == [0.5, 1.0, 1.0]
    assert metrics["per_class_precision"] == [1.0, 1.0, 1.0]


def test_retrospective_calibration_requires_two_calibration_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(T8Error, match="exactly two"):
        t8_module.run_t8_retrospective_calibration_v2(
            dataset_root=tmp_path / "dataset",
            probe_report_path=tmp_path / "probe.json",
            layout_path=tmp_path / "layout.json",
            baseline_dir=tmp_path / "baseline",
            inverse_calibration_paths=(),
            inverse_holdout_path=tmp_path / "holdout.npz",
            output_dir=tmp_path / "output",
        )


def test_causal_session_rows_end_before_event_and_add_safe_wait() -> None:
    timestamps = np.arange(50, dtype=np.int64) * 100
    frame_hashes = np.asarray([f"{index:064x}" for index in range(50)])
    event = {
        "frame_index": 25,
        "timestamp_ms": 2500,
        "frame_sha256": str(frame_hashes[25]),
        "combat": "basic_attack",
        "confidence": 0.95,
    }
    rows = t8_module._causal_session_rows(timestamps, frame_hashes, [event], 100)
    action = next(row for row in rows if row["label_kind"] == 1)
    wait = next(row for row in rows if row["label_kind"] == 0)
    assert action["combat_id"] == 1
    assert action["observation_end_timestamp_ms"] == 2400
    assert action["end_index"] == 24
    assert action["start_index"] == 9
    assert action["causal_window_sha256"] != ""
    assert wait["combat_id"] == 0
    assert abs(cast(int, wait["label_timestamp_ms"]) - 2500) >= 1000


def test_causal_session_rows_reject_unbound_event() -> None:
    timestamps = np.arange(40, dtype=np.int64) * 100
    frame_hashes = np.asarray([f"{index:064x}" for index in range(40)])
    with pytest.raises(T8Error, match="does not bind"):
        t8_module._causal_session_rows(
            timestamps,
            frame_hashes,
            [
                {
                    "frame_index": 25,
                    "timestamp_ms": 2500,
                    "frame_sha256": "f" * 64,
                    "combat": "skill1",
                    "confidence": 0.9,
                }
            ],
            200,
        )


def test_causal_video_loader_rejects_test_before_dataset_access(tmp_path: Path) -> None:
    with pytest.raises(T8Error, match="train or dev only"):
        t8_module._load_causal_video_split(tmp_path / "missing", "test", 100)


def test_causal_video_temporal_head_has_four_classes() -> None:
    model = t8_module._VideoCausalTemporal().eval()
    with torch.no_grad():
        output = model(torch.zeros((2, 16, 512)))
    assert output.shape == (2, 4)


def test_causal_probe_head_and_stratified_holdout() -> None:
    model = t8_module._VideoCausalTemporal(2).eval()
    with torch.no_grad():
        assert model(torch.zeros((3, 16, 512))).shape == (3, 2)
    labels = np.repeat(np.arange(4, dtype=np.int64), 10)
    train_rows, dev_rows = t8_module._stratified_row_holdout(labels)
    assert not set(train_rows).intersection(map(int, dev_rows))
    assert sorted(np.concatenate((train_rows, dev_rows)).tolist()) == list(range(40))
    assert np.bincount(labels[train_rows], minlength=4).tolist() == [8, 8, 8, 8]
    assert np.bincount(labels[dev_rows], minlength=4).tolist() == [2, 2, 2, 2]


def test_causal_pixel_views_and_matched_wait_are_strictly_pre_action() -> None:
    frame = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3)
    views = t8_module._causal_pixel_views(frame, (0, 35, 128, 93))
    assert views.shape == (3, 128, 128, 3)
    assert np.array_equal(views[0], frame)
    assert not np.array_equal(views[0], views[1])
    timestamps = np.arange(100, dtype=np.int64) * 100
    frame_hashes = np.asarray([f"{index:064x}" for index in range(100)])
    events = [
        {
            "frame_index": 50,
            "timestamp_ms": 5000,
            "frame_sha256": str(frame_hashes[50]),
            "combat": "skill1",
            "confidence": 0.9,
        }
    ]
    rows = t8_module._matched_causal_pixel_rows(timestamps, frame_hashes, events)
    assert len(rows) == 2
    action = next(row for row in rows if row["label_kind"] == 1)
    wait = next(row for row in rows if row["label_kind"] == 0)
    assert action["observation_end_timestamp_ms"] == 4900
    assert action["combat_id"] == 2
    assert abs(cast(int, wait["label_timestamp_ms"]) - 5000) >= 1000
    assert cast(int, action["shift_index"]) < cast(int, action["end_index"])


def test_causal_pixel_loader_rejects_test_before_dataset_access(tmp_path: Path) -> None:
    with pytest.raises(T8Error, match="train or dev only"):
        t8_module._load_causal_pixel_split(tmp_path / "missing", "test")


def test_causal_pixel_probe_accepts_three_views() -> None:
    encoder = T8V2FactorizedActor().encoder.state_dict()
    model = t8_module._CausalPixelProbe(encoder, 2).eval()
    with torch.no_grad():
        output = model(torch.zeros((2, 3, 3, 128, 128)))
    assert output.shape == (2, 2)


def test_visual_teacher_is_deterministic_and_abstains_on_low_activity() -> None:
    views = np.zeros((4, 3, 128, 128, 3), dtype=np.uint8)
    history = views.copy()
    views[1:, 1] = 64
    points = ((0.70, 0.70), (0.80, 0.75), (0.90, 0.80))
    activity, scores = t8_module._visual_teacher_features(views, history, points)
    assert activity.shape == (4,)
    assert scores.shape == (4, 3)
    medians = np.zeros(3, dtype=np.float32)
    scales = np.ones(3, dtype=np.float32)
    first = t8_module._visual_teacher_predict(activity, scores, 0.01, medians, scales)
    second = t8_module._visual_teacher_predict(activity, scores, 0.01, medians, scales)
    assert first[0] == 0
    assert np.array_equal(first, second)


def test_visible_onset_selects_the_first_quiet_button_rise() -> None:
    timestamps = np.arange(12, dtype=np.int64) * 100
    hashes = np.asarray([f"{index:064x}" for index in range(12)])
    content = np.ones(12, dtype=np.float32) * 10.0
    absolute = np.zeros((12, 4), dtype=np.float32)
    positive = np.zeros_like(absolute)
    absolute[6, 0], absolute[7, 0] = 24.0, 50.0
    positive[6, 0], positive[7, 0] = 12.0, 25.0
    event = {
        "frame_index": 7,
        "timestamp_ms": 700,
        "frame_sha256": str(hashes[7]),
        "combat": "basic_attack",
        "confidence": 0.9,
        "session_hash": "a" * 64,
        "split": "train",
        "baseline_sha256": "b" * 64,
        "target_manifest_sha256": "c" * 64,
    }
    onset = t8_module._visible_onset_event(
        event, timestamps, hashes, content, absolute, positive
    )
    assert onset is not None
    assert onset["visible_onset_frame_index"] == 6
    assert onset["onset_offset_frames"] == -1


def test_visible_onset_abstains_on_cross_button_conflict() -> None:
    timestamps = np.arange(12, dtype=np.int64) * 100
    hashes = np.asarray([f"{index:064x}" for index in range(12)])
    content = np.ones(12, dtype=np.float32) * 10.0
    absolute = np.zeros((12, 4), dtype=np.float32)
    positive = np.zeros_like(absolute)
    absolute[7, :2] = (50.0, 45.0)
    positive[7, 0] = 25.0
    event = {
        "frame_index": 7,
        "timestamp_ms": 700,
        "frame_sha256": str(hashes[7]),
        "combat": "basic_attack",
        "confidence": 0.9,
        "session_hash": "a" * 64,
        "split": "dev",
        "baseline_sha256": "b" * 64,
        "target_manifest_sha256": "c" * 64,
    }
    assert (
        t8_module._visible_onset_event(
            event, timestamps, hashes, content, absolute, positive
        )
        is None
    )


def test_combat_causal_rows_are_32_frames_and_strictly_pre_onset() -> None:
    timestamps = np.arange(100, dtype=np.int64) * 100
    hashes = np.asarray([f"{index:064x}" for index in range(100)])
    events = (
        {
            "visible_onset_frame_index": 60,
            "visible_onset_timestamp_ms": 6000,
            "visible_onset_frame_sha256": str(hashes[60]),
            "combat": "skill1",
        },
    )
    rows = t8_module._combat_causal_rows(timestamps, hashes, events)
    assert len(rows) == 2
    action = next(row for row in rows if row["label_kind"] == 1)
    wait = next(row for row in rows if row["label_kind"] == 0)
    assert action["end_index"] - action["start_index"] + 1 == 32
    assert action["observation_end_timestamp_ms"] == 5900
    assert action["visible_onset_timestamp_ms"] == 6000
    assert abs(cast(int, wait["label_timestamp_ms"]) - 6000) >= 1000


def test_combat_causal_loader_rejects_test_before_dataset_access(tmp_path: Path) -> None:
    with pytest.raises(T8Error, match="train or dev only"):
        t8_module._combat_causal_shards(tmp_path / "missing", "test")


def test_combat_causal_rgb_model_accepts_32_frame_two_view_input() -> None:
    encoder = T8V2FactorizedActor().encoder.state_dict()
    model = t8_module._CombatCausalRGB(encoder).eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 32, 2, 3, 128, 128)))
    assert output.shape == (1, 4)


def test_v26_conditional_model_has_gate_and_action_heads() -> None:
    encoder = T8V2FactorizedActor().encoder.state_dict()
    model = t8_module._V26ConditionalCombatRGB(encoder).eval()
    with torch.no_grad():
        gate, action = model(torch.zeros((1, 32, 2, 3, 128, 128)))
    assert gate.shape == (1, 2)
    assert action.shape == (1, 3)


def test_v26_gate_threshold_is_frozen_at_point_65() -> None:
    probabilities = torch.tensor([[0.36, 0.64], [0.35, 0.65]])
    logits = probabilities.log()
    assert t8_module.V26_GATE_DECISION_THRESHOLD == 0.65
    assert t8_module._v26_gate_prediction(logits).tolist() == [0, 1]


def test_v26_live_predictor_emits_combat_confidence_and_entropy(tmp_path: Path) -> None:
    encoder = T8V2FactorizedActor().encoder.state_dict()
    model = t8_module._V26ConditionalCombatRGB(encoder).eval()
    path = tmp_path / "model.safetensors"
    t8_module.save_file(
        model.state_dict(),
        path,
        metadata={
            "schema": t8_module.V26_CONDITIONAL_MODEL_SCHEMA,
            "gate_decision_threshold": str(t8_module.V26_GATE_DECISION_THRESHOLD),
        },
    )
    predictor = t8_module.open_t8_v26_predictor(path, "cpu")
    labels, confidence, entropy = predictor(
        np.zeros((1, 32, 2, 128, 128, 3), dtype=np.uint8)
    )
    assert labels.shape == confidence.shape == entropy.shape == (1,)
    assert 0 <= labels[0] <= 3
    assert 0 <= confidence[0] <= 1
    assert 0 <= entropy[0] <= 1
    stream = t8_module.open_t8_v26_stream_predictor(path, "cpu")
    for _ in range(32):
        stream_labels, stream_confidence, stream_entropy = stream(
            np.zeros((2, 128, 128, 3), dtype=np.uint8)
        )
    np.testing.assert_array_equal(stream_labels, labels)
    np.testing.assert_allclose(stream_confidence, confidence, atol=1e-6)
    np.testing.assert_allclose(stream_entropy, entropy, atol=1e-6)


def test_v26_three_seed_selection_uses_dev_macro_f1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    large = tmp_path / "large"
    run_root = large / "runs" / "formal"
    split_sha256 = "a" * 64
    for seed, macro_f1 in enumerate((0.55, 0.70, 0.64)):
        seed_dir = run_root / f"seed-{seed}"
        seed_dir.mkdir(parents=True)
        model = seed_dir / f"model-seed-{seed}.safetensors"
        model.write_bytes(f"seed-{seed}".encode())
        report = {
            "schema_version": t8_module.V26_CONDITIONAL_PILOT_SCHEMA,
            "seed": seed,
            "pilot_split": False,
            "learnability_gate_passed": True,
            "three_seed_training_allowed": True,
            "test_accessed": False,
            "gate_decision_threshold": t8_module.V26_GATE_DECISION_THRESHOLD,
            "split_sha256": split_sha256,
            "model_sha256": t8_module._sha(model),
            "normal": {
                "metrics": {
                    "four_class": {"macro_f1": macro_f1, "accuracy": 0.8 + seed / 100}
                }
            },
        }
        (seed_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setenv("HOK_LARGE_ROOT", str(large))
    output = run_root / "selection.json"
    selected = t8_module.select_t8_v26_model(run_root=run_root, output_path=output)
    assert selected["selected_seed"] == 1
    assert selected["selected_model"] == "seed-1/model-seed-1.safetensors"
    assert selected["test_accessed"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == selected


def test_v26_sealed_evaluation_rejects_contract_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    large = tmp_path / "large"
    dataset_root = large / "dataset"
    run_root = large / "runs"
    dataset_root.mkdir(parents=True)
    run_root.mkdir(parents=True)
    split_path = dataset_root / "split.json"
    selection_path = run_root / "selection.json"
    split_path.write_text("{}\n", encoding="utf-8")
    selection_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOK_LARGE_ROOT", str(large))

    def fail_if_accessed(_: Path) -> dict[str, object]:
        pytest.fail("test session was accessed before the sealed contract passed")

    monkeypatch.setattr(t8_module, "_v25_session_metadata", fail_if_accessed)
    with pytest.raises(T8Error, match="sealed evaluation contract is invalid"):
        t8_module._v26_selected_test_rows(
            dataset_root, split_path, run_root, selection_path
        )


def test_v26_sealed_evaluation_resolves_only_frozen_test_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    large = tmp_path / "large"
    dataset_root = large / "dataset"
    run_root = large / "runs"
    model_path = run_root / "seed-1" / "model-seed-1.safetensors"
    model_path.parent.mkdir(parents=True)
    dataset_root.mkdir(parents=True)
    model_path.write_bytes(b"selected model")
    sessions = [
        {"name": "session-011", "session_sha256": "b" * 64},
        {"name": "session-012", "session_sha256": "c" * 64},
    ]
    split: dict[str, object] = {
        "schema_version": t8_module.V25_SPLIT_SCHEMA,
        "pilot": False,
        "test_accessed": False,
        "gate_decision_threshold": t8_module.V26_GATE_DECISION_THRESHOLD,
        "splits": {"train": [], "dev": [], "test": sessions},
    }
    split["split_sha256"] = hashlib.sha256(t8_module._canonical(split)).hexdigest()
    split_path = dataset_root / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    selection: dict[str, object] = {
        "schema_version": t8_module.V26_SELECTION_SCHEMA,
        "status": "THREE_SEED_MODEL_SELECTED",
        "split_sha256": split["split_sha256"],
        "gate_decision_threshold": t8_module.V26_GATE_DECISION_THRESHOLD,
        "selected_seed": 1,
        "selected_model": "seed-1/model-seed-1.safetensors",
        "selected_model_sha256": t8_module._sha(model_path),
        "test_accessed": False,
        "shadow_allowed": False,
    }
    selection["selection_sha256"] = hashlib.sha256(
        t8_module._canonical(selection)
    ).hexdigest()
    selection_path = run_root / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setenv("HOK_LARGE_ROOT", str(large))

    def metadata(path: Path) -> dict[str, object]:
        frozen_session = sessions[int(path.name[-3:]) - 11]
        return {
            "session_sha256": frozen_session["session_sha256"],
            "frame_shards": 1,
            "shards": [{"name": "shard-000.npz", "sha256": "d" * 64}],
        }

    monkeypatch.setattr(t8_module, "_v25_session_metadata", metadata)
    root, rows, resolved_selection, resolved_model = t8_module._v26_selected_test_rows(
        dataset_root, split_path, run_root, selection_path
    )
    assert root == dataset_root.resolve()
    assert [row["session"] for row in rows] == ["session-011", "session-012"]
    assert resolved_selection["selected_seed"] == 1
    assert resolved_model == model_path.resolve()


def test_failed_visible_onset_is_readable_only_with_explicit_diagnostic_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    audit = root / "audit" / "onset"
    identity = "a" * 64
    for split in ("train", "dev"):
        event_dir = audit / split / "events"
        event_dir.mkdir(parents=True)
        path = event_dir / f"{identity}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "schema_version": t8_module.VISIBLE_ONSET_EVENT_SCHEMA,
                    "sequence": 0,
                    "session_hash": identity,
                    "split": split,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    report = {
        "schema_version": t8_module.VISIBLE_ONSET_AUDIT_SCHEMA,
        "status": "VISIBLE_ONSET_AUDIT_FAILED",
        "training_eligible": False,
        "video_test_accessed": False,
        "splits": {
            split: {
                "sessions": [
                    {
                        "session_hash": identity,
                        "events_sha256": t8_module._sha(
                            audit / split / "events" / f"{identity}.jsonl"
                        ),
                    }
                ]
            }
            for split in ("train", "dev")
        },
    }
    (audit / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    _root, events, _digest = t8_module._visible_onset_events(
        audit, "train", diagnostic_only=True
    )
    assert len(events[identity]) == 1
    with pytest.raises(T8Error, match="did not admit"):
        t8_module._visible_onset_events(audit, "train")


def test_v25_pilot_loader_rejects_test_before_dataset_access(tmp_path: Path) -> None:
    with pytest.raises(T8Error, match="train or dev only"):
        t8_module._v25_rows(tmp_path / "missing", tmp_path / "split.json", "test")


def test_balanced_combat_order_is_deterministic_and_downsamples_wait() -> None:
    labels = np.asarray([0] * 20 + [1] * 6 + [2] * 2 + [3], dtype=np.int64)
    first = t8_module._balanced_combat_order(labels, 17)
    second = t8_module._balanced_combat_order(labels, 17)
    assert np.array_equal(first, second)
    assert np.bincount(labels[first], minlength=4).tolist() == [6, 6, 6, 6]


def test_v25_loader_reconstructs_32_frame_current_and_shifted_views(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    session = root / "session-001"
    session.mkdir(parents=True)
    frames = np.stack(
        [np.full((128, 128, 3), index, dtype=np.uint8) for index in range(52)]
    )
    frame_path = session / "frames-00000.npz"
    np.savez_compressed(
        frame_path,
        frames=frames,
        timestamp_ns=np.arange(52, dtype=np.int64) * 100_000_000,
    )
    decision_path = session / "samples-00000.npz"
    np.savez_compressed(
        decision_path,
        observation_index=np.asarray([51], dtype=np.int32),
        shifted_observation_index=np.asarray([31], dtype=np.int32),
        combat_id=np.asarray([2], dtype=np.int8),
        observation_end_timestamp_ns=np.asarray([5_100_000_000], dtype=np.int64),
        decision_timestamp_ns=np.asarray([5_200_000_000], dtype=np.int64),
        execution_timestamp_ns=np.asarray([5_300_000_000], dtype=np.int64),
        confidence=np.asarray([0.8], dtype=np.float32),
        input_sent=np.asarray([1], dtype=np.uint8),
    )
    row: dict[str, object] = {
        "session": "session-001",
        "path": decision_path.name,
        "sha256": t8_module._sha(decision_path),
        "frame_shards": [
            {"name": frame_path.name, "sha256": t8_module._sha(frame_path)}
        ],
    }
    loader = t8_module._v25_shard_loader(t8_module._V25FrameCache())
    current, shifted, labels = loader(root, row)
    assert current.shape == (1, 32, 2, 128, 128, 3)
    assert shifted.shape == current.shape
    assert labels.tolist() == [2]
    assert int(current[0, -1, 0, 0, 0, 0]) == 51
    assert int(shifted[0, -1, 0, 0, 0, 0]) == 31


def test_v21_pilot_loader_never_opens_sealed_test_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    dataset = root / "datasets" / "t8-demonstrations-v2.1"
    dataset.mkdir(parents=True)
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    names = tuple(f"session-{index:03d}" for index in range(1, 13))
    for name in names:
        (dataset / name).mkdir()
    splits = t8_module._v21_splits(len(names))
    identities = {name: f"{index:064x}" for index, name in enumerate(names)}
    payload: dict[str, object] = {
        "schema_version": t8_module.V21_SPLIT_SCHEMA,
        "split_seed": t8_module.V21_SPLIT_SEED,
        "config_sha256": t8_module.V2_CONFIG_HASH,
        "layout_sha256": "a" * 64,
        "action_contract_sha256": "b" * 64,
        "session_count": len(names),
        "splits": {
            split: [
                {"name": names[index], "session_sha256": identities[names[index]]}
                for index in indices
            ]
            for split, indices in splits.items()
        },
    }
    payload["split_sha256"] = hashlib.sha256(t8_module._canonical(payload)).hexdigest()
    split_path = dataset / "t8-v2.1-split.json"
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    opened: list[str] = []

    def load(path: Path) -> t8_module.V2Session:
        opened.append(path.name)
        return t8_module.V2Session(
            identities[path.name],
            path,
            "a" * 64,
            "b" * 64,
            t8_module.SCRCPY_EXECUTED_ACTION_SOURCE,
            np.zeros((1, 16, 128, 128, 3), dtype=np.uint8),
            np.zeros((1, 5), dtype=np.int64),
        )

    monkeypatch.setattr(t8_module, "_load_v21_session", load)
    data = t8_module.load_t8_v21_data(dataset, split_path)
    test_names = {names[index] for index in splits["test"]}
    assert not test_names.intersection(opened)
    assert len(data.sessions) == len(splits["train"]) + len(splits["dev"])


def test_v27_is_permanently_frozen() -> None:
    with pytest.raises(T8Error, match="frozen failed"):
        t8_module.run_t8_v27_calibration_pilot(
            dataset_root=Path("missing"),
            train_session=Path("missing-train"),
            dev_session=Path("missing-dev"),
            source_model=Path("missing-model"),
            output_dir=Path("missing-output"),
            device="cpu",
        )


def test_v27_freeze_binds_three_failed_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "large"
    reports = root / "runs"
    reports.mkdir(parents=True)
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    paths: list[Path] = []
    for index in range(3):
        path = reports / f"report-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": t8_module.V27_CALIBRATION_SCHEMA,
                    "status": "CALIBRATION_DIAGNOSIS_REQUIRED",
                    "strict_passed": False,
                    "diagnostic_only": True,
                    "formal_training_allowed": False,
                    "test_accessed": False,
                    "shadow_allowed": False,
                    "device_input_allowed": False,
                    "model_sha256": f"{index:064x}",
                    "train_session_sha256": "a" * 64,
                    "dev_session_sha256": "b" * 64,
                    "thresholds": {"macro_f1": 0.5},
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = freeze_t8_v27_failures(report_paths=paths, output_dir=root / "audit" / "freeze")
    assert result["schema_version"] == V27_FREEZE_SCHEMA
    assert result["rerun_allowed"] is False
    assert len(cast(list[object], result["reports"])) == 3


def test_v3_state_labels_are_observable_and_logically_masked() -> None:
    current = np.zeros((2, 3, 128, 128, 3), dtype=np.uint8)
    history = current.copy()
    current[0, 0, 20:40, 20:50, 0] = 255
    current[:, 2] = 255
    labels, confidence = _v3_labels(
        current,
        history,
        ((0.83, 0.84), (0.76, 0.76), (0.83, 0.64)),
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
    )
    assert labels.shape == (2, len(V3_STATE_NAMES))
    assert confidence.shape == (2,)
    assert labels[0, 0] == 1
    assert np.all(labels[:, 1] <= labels[:, 0])
    assert np.all(labels[:, 1] <= labels[:, 2:].any(axis=1))


def test_v3_temporal_decode_and_hybrid_executor_contract() -> None:
    model = V3StateTemporal()
    state_logits, id_logits = model(torch.zeros(2, 16, 512))
    predicted, probabilities, confidence, abstain = _v3_decode(state_logits, id_logits)
    assert predicted.shape == (2, len(V3_STATE_NAMES))
    assert probabilities.shape == predicted.shape
    assert confidence.shape == abstain.shape == (2,)
    executor = V3HybridExecutor()
    state = {
        **{name: True for name in V3_STATE_NAMES},
        "confidence": 1.0,
        "abstain": False,
    }
    assert executor.decide(0, state).action == "skill2"
    assert executor.decide(1_000, state).action == "skill1"
    assert executor.decide(2_000, state).action == "basic_attack"
    assert executor.decide(3_000, state).action == "basic_attack"
    assert executor.decide(4_000, state).action == "basic_attack"
    assert executor.decide(5_000, state).action == "none"
    assert executor.decide(8_000, state).action == "skill2"
