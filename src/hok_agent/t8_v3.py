"""T8-v3 RGB state perception and deterministic hybrid execution boundary."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.mobile_testbed import ABILITIES, load_layout, load_rgb_teacher_calibration
from hok_agent.t8 import (
    CAUSAL_VIDEO_DATASET_SCHEMA,
    V27_CALIBRATION_SCHEMA,
    _canonical,
    _causal_pixel_views,
    _head_metrics,
    _large_existing,
    _large_new,
    _load_v2_adapter,
    _read_object,
    _retrospective_content_box,
    _retrospective_load_session,
    _retrospective_target_index,
    _sha,
    _V2ResidualBlock,
    _visual_teacher_features,
)

V27_FREEZE_SCHEMA = "hok-agent-t8-v2.7-failure-freeze-v1"
V3_DATASET_SCHEMA = "hok-agent-t8-v3-video-state-dataset-v1"
V3_TRAINING_SCHEMA = "hok-agent-t8-v3-video-state-single-seed-v1"
V3_MODEL_SCHEMA = "hok-agent-t8-v3-video-state-model-v1"
V3_REPLAY_SCHEMA = "hok-agent-t8-v3-hybrid-offline-replay-v1"
V3_STATE_NAMES = (
    "enemy_visible",
    "attack_opportunity",
    "basic_ready",
    "skill1_ready",
    "skill2_ready",
)
V3_WINDOW_FRAMES = 16
V3_FEATURE_SIZE = 512
V3_CONFIDENCE_THRESHOLD = 0.65
V3_STATE_THRESHOLD = 0.50
V3_PRIORITY = ("skill2", "skill1", "basic_attack")
V3_COOLDOWN_MS = {"basic_attack": 1_000, "skill1": 8_000, "skill2": 8_000}
V3_GLOBAL_INTERVAL_MS = 1_000
V3_MAX_IDENTICAL = 3


class T8V3Error(ValueError):
    pass


def freeze_t8_v27_failures(
    *, report_paths: Sequence[Path], output_dir: Path
) -> dict[str, object]:
    if len(report_paths) != 3 or len({path.resolve() for path in report_paths}) != 3:
        raise T8V3Error("T8-v2.7 freeze requires the three distinct calibration reports")
    reports: list[dict[str, object]] = []
    for path in report_paths:
        source = _large_existing(path)
        report = _read_object(source, "T8-v2.7 report is unreadable")
        if (
            report.get("schema_version") != V27_CALIBRATION_SCHEMA
            or report.get("status") != "CALIBRATION_DIAGNOSIS_REQUIRED"
            or report.get("strict_passed") is not False
            or report.get("diagnostic_only") is not True
            or report.get("formal_training_allowed") is not False
            or report.get("test_accessed") is not False
            or report.get("shadow_allowed") is not False
            or report.get("device_input_allowed") is not False
        ):
            raise T8V3Error("T8-v2.7 freeze accepts failed diagnostic reports only")
        reports.append(
            {
                "report_sha256": _sha(source),
                "model_sha256": report.get("model_sha256"),
                "train_session_sha256": report.get("train_session_sha256"),
                "dev_session_sha256": report.get("dev_session_sha256"),
                "thresholds": report.get("thresholds"),
            }
        )
    payload: dict[str, object] = {
        "schema_version": V27_FREEZE_SCHEMA,
        "status": "FROZEN_FAILED",
        "reports": reports,
        "report_count": len(reports),
        "rerun_allowed": False,
        "threshold_changes_allowed": False,
        "four_class_head_training_allowed": False,
        "test_accessed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
        "superseded_by": "t8-v3-video-state-hybrid-v1",
    }
    payload["freeze_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        (staging / "freeze.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return payload


def _v3_feature_manifest(root: Path) -> dict[str, object]:
    manifest = _read_object(root / "manifest.json", "T8-v3 source feature manifest is unreadable")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema_version") != CAUSAL_VIDEO_DATASET_SCHEMA
        or manifest.get("status") != "COMPLETED"
        or manifest.get("feature_shape") != [V3_WINDOW_FRAMES, V3_FEATURE_SIZE]
        or manifest.get("video_test_accessed") is not False
        or manifest.get("event_frame_included") is not False
        or manifest.get("future_frames_included") is not False
        or manifest.get("manifest_sha256")
        != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise T8V3Error("T8-v3 source feature contract is invalid")
    return manifest


def _v3_labels(
    current: np.ndarray,
    history: np.ndarray,
    points: Sequence[tuple[float, float]],
    medians: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _activity, scores = _visual_teacher_features(current, history, points)
    normalized = (scores - medians) / scales
    ready = normalized >= 0.0
    scene = current[:, 0, 8:108, 15:108].astype(np.int16)
    red, green, blue = (scene[..., index] for index in range(3))
    red_mask = (red > 140) & (red - green > 45) & (red - blue > 25)
    red_pixels = red_mask.sum(axis=(1, 2))
    red_row_max = red_mask.sum(axis=2).max(axis=1)
    enemy = (red_pixels >= 400) | (red_row_max >= 11)
    opportunity = enemy & ready.any(axis=1)
    labels = np.column_stack((enemy, opportunity, ready)).astype(np.uint8)
    red_strength = np.maximum(red_pixels / 400.0, red_row_max / 11.0)
    enemy_certainty = np.clip(np.abs(red_strength - 1.0), 0.0, 1.0)
    readiness_certainty = np.mean(np.clip(np.abs(normalized), 0.0, 1.0), axis=1)
    confidence = ((enemy_certainty + readiness_certainty) / 2.0).astype(np.float32)
    return labels, confidence


def materialize_t8_v3_state_dataset(
    *,
    feature_root: Path,
    target_root: Path,
    teacher_report: Path,
    layout_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    features_root = _large_existing(feature_root)
    feature_manifest = _v3_feature_manifest(features_root)
    layout, layout_sha = load_layout(layout_path)
    calibration = load_rgb_teacher_calibration(teacher_report, layout_sha)
    points = [layout.buttons[name] for name in ABILITIES[1:4]]
    if any(point is None for point in points):
        raise T8V3Error("T8-v3 requires the three calibrated combat buttons")
    typed_points = cast(list[tuple[float, float]], points)
    medians = np.asarray(calibration.medians, dtype=np.float32)
    scales = np.asarray(calibration.scales, dtype=np.float32)
    output = _large_new(output_dir)
    source_rows = feature_manifest.get("shards")
    if not isinstance(source_rows, list):
        raise T8V3Error("T8-v3 source feature shard index is invalid")
    counts = {split: {name: 0 for name in V3_STATE_NAMES} for split in ("train", "dev")}
    totals = {"train": 0, "dev": 0}
    session_counts: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    target_sha: str | None = None
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        shard_dir = staging / "shards"
        shard_dir.mkdir()
        for split in ("train", "dev"):
            current_root, current_sha, sessions = _retrospective_target_index(target_root, split)
            if target_sha is None:
                target_sha = current_sha
            elif current_sha != target_sha:
                raise T8V3Error("T8-v3 target manifest changed between train and dev")
            if current_sha != feature_manifest.get("target_manifest_sha256"):
                raise T8V3Error("T8-v3 RGB and V5 feature sources differ")
            expected_count = 103 if split == "train" else 23
            if len(sessions) != expected_count:
                raise T8V3Error("T8-v3 requires the frozen 103/23 video split")
            session_counts[split] = len(sessions)
            by_identity = dict(sessions)
            selected = [
                cast(dict[str, object], row)
                for row in cast(list[object], source_rows)
                if isinstance(row, dict)
                and row.get("split") == split
                and row.get("lag_ms") == 100
            ]
            if {str(row.get("session_hash")) for row in selected} != set(by_identity):
                raise T8V3Error("T8-v3 feature sessions differ from the frozen video split")
            for ordinal, row in enumerate(selected):
                identity = str(row["session_hash"])
                source_name = row.get("path")
                if not isinstance(source_name, str) or Path(source_name).name != source_name:
                    raise T8V3Error("T8-v3 source feature shard name is invalid")
                source_path = features_root / "shards" / source_name
                if _sha(source_path) != row.get("sha256"):
                    raise T8V3Error("T8-v3 source feature shard hash differs")
                with np.load(source_path, allow_pickle=False) as source:
                    feature_values = source["features"]
                    observation_end = source["observation_end_timestamp_ms"]
                    if (
                        feature_values.dtype != np.float16
                        or feature_values.shape[1:] != (V3_WINDOW_FRAMES, V3_FEATURE_SIZE)
                        or observation_end.dtype != np.int64
                        or observation_end.shape != (len(feature_values),)
                    ):
                        raise T8V3Error("T8-v3 source feature tensors are invalid")
                    cached_features = feature_values.copy()
                    cached_timestamps = observation_end.copy()
                frames, timestamps, _hashes = _retrospective_load_session(
                    current_root, split, identity, by_identity[identity]
                )
                canonical, _orientation, content_box = _retrospective_content_box(frames)
                indices = np.searchsorted(timestamps, cached_timestamps)
                if (
                    np.any(indices >= len(timestamps))
                    or not np.array_equal(timestamps[indices], cached_timestamps)
                ):
                    raise T8V3Error("T8-v3 feature timestamps do not bind to source RGB")
                history_indices = np.searchsorted(
                    timestamps, cached_timestamps - 2_000, side="right"
                ) - 1
                history_indices = np.maximum(history_indices, 0)
                current_views = np.stack(
                    [_causal_pixel_views(canonical[int(index)], content_box) for index in indices]
                )
                history_views = np.stack(
                    [
                        _causal_pixel_views(canonical[int(index)], content_box)
                        for index in history_indices
                    ]
                )
                labels, confidence = _v3_labels(
                    current_views, history_views, typed_points, medians, scales
                )
                name = f"{split}-{ordinal:04d}.npz"
                path = shard_dir / name
                np.savez_compressed(
                    path,
                    features=cached_features,
                    state_labels=labels,
                    teacher_confidence=confidence,
                    observation_end_timestamp_ms=cached_timestamps,
                )
                for index, state_name in enumerate(V3_STATE_NAMES):
                    counts[split][state_name] += int(labels[:, index].sum())
                totals[split] += len(labels)
                rows.append(
                    {
                        "path": name,
                        "sha256": _sha(path),
                        "split": split,
                        "session_hash": identity,
                        "rows": len(labels),
                        "positive_counts": {
                            name: int(labels[:, index].sum())
                            for index, name in enumerate(V3_STATE_NAMES)
                        },
                    }
                )
        manifest: dict[str, object] = {
            "schema_version": V3_DATASET_SCHEMA,
            "status": "COMPLETED",
            "task": "rgb_observable_combat_state_distillation",
            "state_names": list(V3_STATE_NAMES),
            "state_definitions": {
                "enemy_visible": "frozen_rgb_red_cue_v1",
                "attack_opportunity": "enemy_visible_and_any_ready",
                "basic_ready": "normalized_basic_button_score_ge_0",
                "skill1_ready": "normalized_skill1_button_score_ge_0",
                "skill2_ready": "normalized_skill2_button_score_ge_0",
                "confidence": "model_id_probability_times_mean_binary_certainty",
                "abstain": f"confidence_lt_{V3_CONFIDENCE_THRESHOLD}",
            },
            "window_frames": V3_WINDOW_FRAMES,
            "state_teacher_history_ms": 2_000,
            "state_teacher_history_padding": "repeat_first_available_frame",
            "feature_shape": [V3_WINDOW_FRAMES, V3_FEATURE_SIZE],
            "feature_source": "frozen_v5_initialized_resnet18",
            "source_feature_manifest_sha256": feature_manifest["manifest_sha256"],
            "adapter_sha256": feature_manifest.get("adapter_sha256"),
            "target_manifest_sha256": target_sha,
            "teacher_report_sha256": calibration.report_sha256,
            "layout_sha256": layout_sha,
            "session_counts": session_counts,
            "sample_counts": totals,
            "positive_counts": counts,
            "shards": rows,
            "video_train_used": True,
            "video_dev_used_for_selection": True,
            "video_test_accessed": False,
            "raw_rgb_persisted": False,
            "raw_video_or_source_paths_persisted": False,
            "structured_state_actor_input": False,
            "pilot_training_allowed": True,
            "device_input_allowed": False,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return manifest


def _v3_dataset_rows(root: Path, split: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    if split not in {"train", "dev"}:
        raise T8V3Error("T8-v3 may open video-train or video-dev only")
    manifest = _read_object(root / "manifest.json", "T8-v3 manifest is unreadable")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema_version") != V3_DATASET_SCHEMA
        or manifest.get("status") != "COMPLETED"
        or manifest.get("state_names") != list(V3_STATE_NAMES)
        or manifest.get("window_frames") != V3_WINDOW_FRAMES
        or manifest.get("feature_shape") != [V3_WINDOW_FRAMES, V3_FEATURE_SIZE]
        or manifest.get("session_counts") != {"train": 103, "dev": 23}
        or manifest.get("video_test_accessed") is not False
        or manifest.get("structured_state_actor_input") is not False
        or manifest.get("manifest_sha256")
        != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise T8V3Error("T8-v3 dataset contract is invalid")
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise T8V3Error("T8-v3 shard index is invalid")
    selected = [
        cast(dict[str, object], row)
        for row in cast(list[object], rows)
        if isinstance(row, dict) and row.get("split") == split
    ]
    if not selected:
        raise T8V3Error("T8-v3 split is empty")
    return manifest, selected


def _load_v3_shard(
    root: Path, row: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    name = row.get("path")
    if not isinstance(name, str) or Path(name).name != name:
        raise T8V3Error("T8-v3 shard name is invalid")
    path = root / "shards" / name
    if _sha(path) != row.get("sha256"):
        raise T8V3Error("T8-v3 shard hash differs")
    with np.load(path, allow_pickle=False) as shard:
        expected = {
            "features",
            "state_labels",
            "teacher_confidence",
            "observation_end_timestamp_ms",
        }
        features = shard["features"]
        labels = shard["state_labels"]
        timestamps = shard["observation_end_timestamp_ms"]
        if (
            set(shard.files) != expected
            or features.dtype != np.float16
            or features.shape[1:] != (V3_WINDOW_FRAMES, V3_FEATURE_SIZE)
            or labels.dtype != np.uint8
            or labels.shape != (len(features), len(V3_STATE_NAMES))
            or np.any(labels > 1)
            or timestamps.dtype != np.int64
            or timestamps.shape != (len(features),)
            or np.any(labels[:, 1] > labels[:, 0])
            or np.any(labels[:, 1] > labels[:, 2:].any(axis=1))
        ):
            raise T8V3Error("T8-v3 shard tensor contract is invalid")
        return features.copy(), labels.astype(np.float32), timestamps.copy()


class V3StateTemporal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Conv1d(V3_FEATURE_SIZE, 256, 1)
        self.temporal = nn.Sequential(*(_V2ResidualBlock(value) for value in (1, 2, 4, 8)))
        self.state_head = nn.Linear(256, len(V3_STATE_NAMES))
        self.id_head = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or tuple(features.shape[1:]) != (
            V3_WINDOW_FRAMES,
            V3_FEATURE_SIZE,
        ):
            raise T8V3Error("T8-v3 temporal input is invalid")
        mixed = self.mix(features.transpose(1, 2))
        current = cast(torch.Tensor, self.temporal(mixed))[..., -1]
        return self.state_head(current), self.id_head(current).squeeze(1)


def _v3_decode(
    state_logits: torch.Tensor, id_logits: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = state_logits.sigmoid().float()
    id_probability = id_logits.sigmoid().float()
    raw = probabilities >= V3_STATE_THRESHOLD
    masked = raw.clone()
    masked[:, 1] &= masked[:, 0] & masked[:, 2:].any(dim=1)
    certainty = torch.maximum(probabilities, 1.0 - probabilities).mean(dim=1)
    confidence = id_probability * certainty
    abstain = confidence < V3_CONFIDENCE_THRESHOLD
    return (
        masked.detach().cpu().numpy().astype(np.uint8),
        probabilities.detach().cpu().numpy(),
        confidence.detach().cpu().numpy(),
        abstain.detach().cpu().numpy(),
    )


def _v3_metrics(
    state_logits: torch.Tensor, id_logits: torch.Tensor, labels: np.ndarray
) -> dict[str, object]:
    predicted, _probabilities, confidence, abstain = _v3_decode(state_logits, id_logits)
    integer_labels = labels.astype(np.int64)
    heads = {
        name: _head_metrics(predicted[:, index].astype(np.int64), integer_labels[:, index], 2)
        for index, name in enumerate(V3_STATE_NAMES)
    }
    macro_f1 = [cast(float, value["macro_f1"]) for value in heads.values()]
    positive_recall = [
        cast(list[float], value["per_class_recall"])[1]
        for value in heads.values()
    ]
    return {
        "heads": heads,
        "mean_head_macro_f1": float(np.mean(macro_f1)),
        "minimum_positive_recall": min(positive_recall),
        "joint_exact_accuracy": float(np.mean(np.all(predicted == integer_labels, axis=1))),
        "confidence_coverage": float(np.mean(~abstain)),
        "mean_confidence": float(np.mean(confidence)),
        "logical_violation_count_after_mask": int(
            np.sum(predicted[:, 1] > (predicted[:, 0] & predicted[:, 2:].any(axis=1)))
        ),
    }


def _predict_v3_features(
    model: V3StateTemporal,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            state, current_id = model(
                torch.from_numpy(features[start : start + batch_size])
                .to(device)
                .float()
            )
            states.append(state.cpu())
            ids.append(current_id.cpu())
    return torch.cat(states), torch.cat(ids)


def _fit_v3_state(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    ood_features: np.ndarray,
    device: torch.device,
    batch_size: int,
    shuffled: bool,
    seed: int,
    epochs: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    labels = train_y.copy()
    if shuffled:
        labels = labels[np.random.default_rng(seed).permutation(len(labels))]
    model = V3StateTemporal().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    positive = labels.sum(axis=0)
    pos_weight = torch.from_numpy((len(labels) - positive) / np.maximum(positive, 1)).to(
        device
    )
    best_f1, best_epoch, best_state = -1.0, 0, {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed * 10_000 + epoch).permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            current_x = torch.from_numpy(train_x[selected]).to(device).float()
            current_y = torch.from_numpy(labels[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            state_logits, id_logits = model(current_x)
            state_loss = nn.functional.binary_cross_entropy_with_logits(
                state_logits, current_y, pos_weight=pos_weight
            )
            real_id_loss = nn.functional.binary_cross_entropy_with_logits(
                id_logits, torch.ones_like(id_logits)
            )
            _ood_states, ood_id = model(
                torch.from_numpy(ood_features).to(device).float()
            )
            ood_loss = nn.functional.binary_cross_entropy_with_logits(
                ood_id, torch.zeros_like(ood_id)
            )
            (state_loss + 0.25 * (real_id_loss + ood_loss)).backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        dev_states, dev_ids = _predict_v3_features(model, dev_x, device, batch_size)
        metrics = _v3_metrics(dev_states, dev_ids, dev_y)
        score = cast(float, metrics["mean_head_macro_f1"])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if not best_state:
        raise T8V3Error("T8-v3 training produced no selected state")
    model.load_state_dict(best_state, strict=True)
    states, ids = _predict_v3_features(model, dev_x, device, batch_size)
    return (
        {
            "best_epoch": best_epoch,
            "shuffled_labels": shuffled,
            "metrics": _v3_metrics(states, ids, dev_y),
        },
        best_state,
    )


def _encode_constant_features(
    encoder_state: Mapping[str, torch.Tensor], device: torch.device
) -> np.ndarray:
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(device).eval()
    images = torch.stack(
        (torch.zeros(3, 128, 128), torch.full((3, 128, 128), 0.5))
    ).to(device)
    with torch.no_grad():
        encoded = cast(torch.Tensor, encoder(images)).cpu().numpy().astype(np.float32)
    return np.repeat(encoded[:, None, :], V3_WINDOW_FRAMES, axis=1)


def train_t8_v3_state_single_seed(
    *,
    dataset_root: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 256,
    seed: int = 0,
    epochs: int = 8,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1 or seed != 0 or epochs != 8:
        raise T8V3Error("T8-v3 requires the frozen single-seed pilot settings")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8V3Error("CUDA is unavailable")
    root = _large_existing(dataset_root)
    manifest, train_rows = _v3_dataset_rows(root, "train")
    dev_manifest, dev_rows = _v3_dataset_rows(root, "dev")
    if dev_manifest["manifest_sha256"] != manifest["manifest_sha256"]:
        raise T8V3Error("T8-v3 dataset changed between train and dev")
    adapter = _large_existing(adapter_checkpoint)
    target = torch.device(device)
    encoder_state, adapter_meta = _load_v2_adapter(adapter, target)
    if (
        _sha(adapter) != manifest.get("adapter_sha256")
        or adapter_meta.get("target_manifest_sha256") != manifest.get("target_manifest_sha256")
    ):
        raise T8V3Error("T8-v3 adapter differs from the frozen feature dataset")

    def load(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
        values = [_load_v3_shard(root, row) for row in rows]
        return np.concatenate([value[0] for value in values]), np.concatenate(
            [value[1] for value in values]
        )

    train_x, train_y = load(train_rows)
    dev_x, dev_y = load(dev_rows)
    ood = _encode_constant_features(encoder_state, target)
    normal, state = _fit_v3_state(
        train_x=train_x,
        train_y=train_y,
        dev_x=dev_x,
        dev_y=dev_y,
        ood_features=ood,
        device=target,
        batch_size=batch_size,
        shuffled=False,
        seed=seed,
        epochs=epochs,
    )
    shuffled, _ = _fit_v3_state(
        train_x=train_x,
        train_y=train_y,
        dev_x=dev_x,
        dev_y=dev_y,
        ood_features=ood,
        device=target,
        batch_size=batch_size,
        shuffled=True,
        seed=seed,
        epochs=epochs,
    )
    model = V3StateTemporal().to(target)
    model.load_state_dict(state, strict=True)
    static_states, static_ids = _predict_v3_features(
        model, np.repeat(dev_x[:, -1:], V3_WINDOW_FRAMES, axis=1), target, batch_size
    )
    reverse_states, reverse_ids = _predict_v3_features(
        model, dev_x[:, ::-1].copy(), target, batch_size
    )
    ood_states, ood_ids = _predict_v3_features(model, ood, target, batch_size)
    _ood_predicted, _ood_probabilities, _ood_confidence, ood_abstain = _v3_decode(
        ood_states, ood_ids
    )
    normal_metrics = cast(dict[str, object], normal["metrics"])
    shuffled_metrics = cast(dict[str, object], shuffled["metrics"])
    normal_f1 = cast(float, normal_metrics["mean_head_macro_f1"])
    shuffled_f1 = cast(float, shuffled_metrics["mean_head_macro_f1"])
    black_gray_abstain = float(np.mean(ood_abstain))
    passed = bool(
        normal_f1 >= 0.70
        and cast(float, normal_metrics["minimum_positive_recall"]) >= 0.55
        and normal_f1 - shuffled_f1 >= 0.15
        and cast(float, normal_metrics["confidence_coverage"]) >= 0.50
        and black_gray_abstain >= 0.95
        and normal_metrics["logical_violation_count_after_mask"] == 0
    )
    output = _large_new(output_dir)
    report: dict[str, object] = {
        "schema_version": V3_TRAINING_SCHEMA,
        "status": "V3_STATE_PILOT_PASSED" if passed else "V3_STATE_PILOT_FAILED",
        "strict_passed": passed,
        "seed": seed,
        "epochs": epochs,
        "model_input": "rgb_only_v5_resnet18_16_frame_causal_features",
        "state_names": list(V3_STATE_NAMES),
        "confidence_threshold": V3_CONFIDENCE_THRESHOLD,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "adapter_sha256": _sha(adapter),
        "v5_source_model_sha256": adapter_meta.get("v5_source_model_sha256"),
        "normal": normal,
        "shuffled": shuffled,
        "normal_minus_shuffled_mean_macro_f1": normal_f1 - shuffled_f1,
        "static_frame": _v3_metrics(static_states, static_ids, dev_y),
        "reversed_time": _v3_metrics(reverse_states, reverse_ids, dev_y),
        "black_gray_abstain_ratio": black_gray_abstain,
        "thresholds": {
            "mean_head_macro_f1": 0.70,
            "minimum_positive_recall": 0.55,
            "shuffled_margin": 0.15,
            "confidence_coverage": 0.50,
            "black_gray_abstain_ratio": 0.95,
            "logical_violations": 0,
        },
        "video_test_accessed": False,
        "offline_replay_allowed": passed,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    combined = {
        **{f"encoder.{key}": value.detach().cpu() for key, value in encoder_state.items()},
        **{f"state.{key}": value.detach().cpu() for key, value in state.items()},
    }
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        model_path = staging / "model-seed-0.safetensors"
        save_file(
            combined,
            model_path,
            metadata={
                "schema": V3_MODEL_SCHEMA,
                "strict_passed": str(passed).lower(),
                "dataset_manifest_sha256": str(manifest["manifest_sha256"]),
                "confidence_threshold": str(V3_CONFIDENCE_THRESHOLD),
            },
        )
        report["model_sha256"] = _sha(model_path)
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _load_v3_model(
    model_path: Path, device: str
) -> tuple[nn.Module, V3StateTemporal, torch.device, dict[str, str]]:
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise T8V3Error("T8-v3 predictor device is unavailable")
    target = torch.device(device)
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = cast(dict[str, str], handle.metadata() or {})
        state = load_file(model_path, device=str(target))
    except (OSError, SafetensorError) as exc:
        raise T8V3Error("T8-v3 model is unreadable") from exc
    if (
        metadata.get("schema") != V3_MODEL_SCHEMA
        or metadata.get("confidence_threshold") != str(V3_CONFIDENCE_THRESHOLD)
    ):
        raise T8V3Error("T8-v3 model metadata is invalid")
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    temporal_state = {
        key.removeprefix("state."): value
        for key, value in state.items()
        if key.startswith("state.")
    }
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(target).eval()
    temporal = V3StateTemporal().to(target)
    temporal.load_state_dict(temporal_state, strict=True)
    temporal.eval()
    return encoder, temporal, target, metadata


def open_t8_v3_stream_predictor(
    model_path: Path, device: str
) -> Callable[[np.ndarray], dict[str, object]]:
    encoder, temporal, target, _metadata = _load_v3_model(model_path, device)
    history: deque[torch.Tensor] = deque(maxlen=V3_WINDOW_FRAMES)

    def predict(frame: np.ndarray) -> dict[str, object]:
        if frame.shape != (128, 128, 3) or frame.dtype != np.uint8:
            raise T8V3Error("T8-v3 stream predictor requires 128x128x3 uint8 RGB")
        tensor = (
            torch.from_numpy(frame).to(target).permute(2, 0, 1).float().div(255.0)[None]
        )
        with torch.no_grad():
            encoded = cast(torch.Tensor, encoder(tensor))[0]
            history.append(encoded)
            values = list(history)
            if len(values) < V3_WINDOW_FRAMES:
                values = [values[0]] * (V3_WINDOW_FRAMES - len(values)) + values
            state_logits, id_logits = temporal(torch.stack(values)[None])
        predicted, probabilities, confidence, abstain = _v3_decode(state_logits, id_logits)
        return {
            **{
                name: bool(predicted[0, index])
                for index, name in enumerate(V3_STATE_NAMES)
            },
            "probabilities": {
                name: float(probabilities[0, index])
                for index, name in enumerate(V3_STATE_NAMES)
            },
            "confidence": float(confidence[0]),
            "abstain": bool(abstain[0]),
        }

    return predict


@dataclass(frozen=True)
class V3HybridDecision:
    action: str
    reason: str


class V3HybridExecutor:
    def __init__(self) -> None:
        self._last_any_ms = -10**12
        self._last_by_action = {name: -10**12 for name in V3_PRIORITY}
        self._last_action = ""
        self._identical = 0

    def decide(self, timestamp_ms: int, state: Mapping[str, object]) -> V3HybridDecision:
        confidence = state.get("confidence")
        if (
            state.get("abstain") is True
            or not isinstance(confidence, (int, float))
            or float(confidence) < V3_CONFIDENCE_THRESHOLD
        ):
            return V3HybridDecision("none", "ABSTAIN")
        if state.get("enemy_visible") is not True or state.get("attack_opportunity") is not True:
            return V3HybridDecision("none", "NO_OPPORTUNITY")
        if timestamp_ms - self._last_any_ms < V3_GLOBAL_INTERVAL_MS:
            return V3HybridDecision("none", "GLOBAL_INTERVAL")
        for action in V3_PRIORITY:
            ready_name = "basic_ready" if action == "basic_attack" else f"{action}_ready"
            if state.get(ready_name) is not True:
                continue
            if timestamp_ms - self._last_by_action[action] < V3_COOLDOWN_MS[action]:
                continue
            if action == self._last_action and self._identical >= V3_MAX_IDENTICAL:
                continue
            self._last_any_ms = timestamp_ms
            self._last_by_action[action] = timestamp_ms
            if action == self._last_action:
                self._identical += 1
            else:
                self._last_action = action
                self._identical = 1
            return V3HybridDecision(action, "PRIORITY_READY")
        return V3HybridDecision("none", "NO_LEGAL_READY_ACTION")


def _accepted_v3_training(report_path: Path, model_path: Path) -> dict[str, object]:
    report = _read_object(report_path, "T8-v3 training report is unreadable")
    if (
        report.get("schema_version") != V3_TRAINING_SCHEMA
        or report.get("status") != "V3_STATE_PILOT_PASSED"
        or report.get("strict_passed") is not True
        or report.get("offline_replay_allowed") is not True
        or report.get("video_test_accessed") is not False
        or report.get("model_sha256") != _sha(model_path)
    ):
        raise T8V3Error("T8-v3 training did not admit offline replay")
    return report


def run_t8_v3_hybrid_replay(
    *,
    dataset_root: Path,
    model_path: Path,
    training_report: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 256,
) -> dict[str, object]:
    root = _large_existing(dataset_root)
    model = _large_existing(model_path)
    training = _accepted_v3_training(training_report, model)
    manifest, rows = _v3_dataset_rows(root, "dev")
    if training.get("dataset_manifest_sha256") != manifest.get("manifest_sha256"):
        raise T8V3Error("T8-v3 replay dataset differs from training")
    _encoder, temporal, target, _metadata = _load_v3_model(model, device)
    events: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()
    accepted_confidence = 0
    samples = 0
    for row in rows:
        features, _labels, timestamps = _load_v3_shard(root, row)
        state_logits, id_logits = _predict_v3_features(temporal, features, target, batch_size)
        predicted, probabilities, confidence, abstain = _v3_decode(state_logits, id_logits)
        executor = V3HybridExecutor()
        for index in np.argsort(timestamps):
            state: dict[str, object] = {
                **{
                    name: bool(predicted[index, ordinal])
                    for ordinal, name in enumerate(V3_STATE_NAMES)
                },
                "confidence": float(confidence[index]),
                "abstain": bool(abstain[index]),
            }
            decision = executor.decide(int(timestamps[index]), state)
            samples += 1
            accepted_confidence += int(not bool(abstain[index]))
            if decision.action != "none":
                action_counts[decision.action] += 1
            events.append(
                {
                    "schema_version": V3_REPLAY_SCHEMA,
                    "sequence": len(events),
                    "session_hash": row["session_hash"],
                    "timestamp_ms": int(timestamps[index]),
                    "states": state,
                    "probabilities": {
                        name: round(float(probabilities[index, ordinal]), 8)
                        for ordinal, name in enumerate(V3_STATE_NAMES)
                    },
                    "hybrid_action": decision.action,
                    "reason": decision.reason,
                    "control_output": False,
                }
            )
    passed = bool(
        samples > 0
        and all(action_counts[name] > 0 for name in V3_PRIORITY)
        and accepted_confidence / samples >= 0.50
    )
    summary: dict[str, object] = {
        "schema_version": V3_REPLAY_SCHEMA,
        "status": "V3_HYBRID_REPLAY_PASSED" if passed else "V3_HYBRID_REPLAY_FAILED",
        "strict_passed": passed,
        "model_sha256": _sha(model),
        "training_report_sha256": _sha(_large_existing(training_report)),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "samples": samples,
        "action_counts": dict(action_counts),
        "confidence_coverage": accepted_confidence / samples,
        "priority": list(V3_PRIORITY),
        "cooldown_ms": V3_COOLDOWN_MS,
        "global_interval_ms": V3_GLOBAL_INTERVAL_MS,
        "maximum_identical_actions": V3_MAX_IDENTICAL,
        "cooldown_violations": 0,
        "priority_violations": 0,
        "identical_action_violations": 0,
        "input_commands_sent": 0,
        "control_output": False,
        "video_test_accessed": False,
        "shadow_allowed": passed,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        event_path = staging / "events.jsonl"
        event_path.write_text(
            "".join(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                for event in events
            ),
            encoding="utf-8",
        )
        summary["events_sha256"] = _sha(event_path)
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return summary
