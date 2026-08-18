"""Offline Operation Policy v1: inverse dynamics, video labels, and causal policy."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.mobile_testbed import ABILITIES, MOVEMENTS, ObservationROIs, load_observation_rois
from hok_agent.t8 import (
    _canonical,
    _head_metrics,
    _large_existing,
    _large_new,
    _load_v2_adapter,
    _read_object,
    _sha,
    _V2ResidualBlock,
)

CONTRACT_SCHEMA: Final = "hok-agent-operation-policy-contract-v1"
CONTRACT_CHECK_SCHEMA: Final = "hok-agent-operation-policy-contract-check-v1"
IDM_SCHEMA: Final = "hok-agent-operation-idm-pilot-v1"
PSEUDOLABEL_SCHEMA: Final = "hok-agent-operation-video-pseudolabel-v1"
POLICY_SCHEMA: Final = "hok-agent-operation-policy-pilot-v1"
DECISION_SCHEMA: Final = "hok-agent-operation-policy-decision-v1"
DIRECT_CONTRACT_SCHEMA: Final = "hok-agent-operation-direct-policy-contract-v1"
DIRECT_POLICY_SCHEMA: Final = "hok-agent-operation-direct-policy-pilot-v1"
MOVEMENT_POLICY_CONTRACT_SCHEMA: Final = "hok-agent-operation-movement-policy-contract-v1"
MOVEMENT_SPLIT_SCHEMA: Final = "hok-agent-operation-movement-split-v1"
MOVEMENT_POLICY_SCHEMA: Final = "hok-agent-operation-movement-policy-pilot-v1"
FEATURE_SIZE: Final = 512
WINDOW_FRAMES: Final = 16
MOVEMENT_SIZE: Final = 9
COMBAT_SIZE: Final = 5


class OperationPolicyError(ValueError):
    pass


EXPECTED_CONTRACT: Final[dict[str, object]] = {
    "schema_version": CONTRACT_SCHEMA,
    "seed": 0,
    "sample_hz": 10,
    "source_sample_hz": 5,
    "window_frames": WINDOW_FRAMES,
    "feature_size": FEATURE_SIZE,
    "idm_feature_shape": [FEATURE_SIZE, 4, 4],
    "pair_lags_ms": [200, 500],
    "movement_vocabulary": list(MOVEMENTS),
    "combat_vocabulary": list(ABILITIES),
    "learned_heads": ["movement", "combat"],
    "deterministic_heads": ["purchase", "hard_stop"],
    "idm_epochs": 12,
    "policy_epochs": 8,
    "teacher_confidence_threshold": 0.8,
    "maximum_negative_to_positive_ratio": 3,
    "minimum_movement_idm_macro_f1": 0.7,
    "minimum_direction_recall": 0.5,
    "minimum_wait_recall": 0.7,
    "minimum_combat_idm_macro_f1": 0.55,
    "minimum_non_none_combat_recall": 0.5,
    "minimum_shuffle_gain": 0.15,
    "minimum_movement_coverage": 0.2,
    "minimum_train_samples_per_class": 100,
    "minimum_dev_samples_per_class": 20,
    "minimum_policy_movement_macro_f1": 0.55,
    "minimum_policy_combat_macro_f1": 0.45,
    "minimum_rgb_gain_over_time_only": 0.08,
    "minimum_tcn_gain": 0.03,
    "minimum_minimap_gain": 0.02,
    "video_train_sessions": 103,
    "video_dev_sessions": 23,
    "video_test_access_allowed": False,
    "human_labels_used": False,
    "semantic_accuracy_verified": False,
    "promotion_allowed": False,
    "control_output": False,
    "device_input_allowed": False,
}

EXPECTED_DIRECT_CONTRACT: Final[dict[str, object]] = {
    "schema_version": DIRECT_CONTRACT_SCHEMA,
    "seed": 0,
    "sample_hz": 5,
    "window_frames": WINDOW_FRAMES,
    "feature_size": FEATURE_SIZE,
    "views": ["main_view", "hud", "minimap"],
    "movement_vocabulary": list(MOVEMENTS),
    "combat_vocabulary": list(ABILITIES),
    "train_combat_sessions": 6,
    "dev_combat_sessions": 2,
    "epochs": 8,
    "maximum_negative_to_positive_ratio": 3,
    "minimum_movement_macro_f1": 0.55,
    "minimum_transition_accuracy": 0.4,
    "minimum_combat_macro_f1": 0.45,
    "minimum_rgb_gain_over_time_only": 0.08,
    "minimum_normal_gain_over_shuffle": 0.15,
    "minimum_tcn_gain": 0.03,
    "human_labels_used": False,
    "semantic_accuracy_verified": False,
    "promotion_allowed": False,
    "control_output": False,
    "device_input_allowed": False,
}

EXPECTED_MOVEMENT_CONTRACT: Final[dict[str, object]] = {
    "schema_version": MOVEMENT_POLICY_CONTRACT_SCHEMA,
    "seed": 0,
    "sample_hz": 5,
    "window_frames": WINDOW_FRAMES,
    "feature_size": FEATURE_SIZE,
    "views": ["main_view", "minimap"],
    "movement_vocabulary": list(MOVEMENTS),
    "pilot_sessions": 4,
    "pilot_split": [3, 1],
    "formal_sessions": 12,
    "formal_split": [8, 2, 2],
    "epochs": 8,
    "maximum_wait_to_movement_ratio": 3,
    "minimum_movement_macro_f1": 0.6,
    "minimum_per_class_recall": 0.4,
    "minimum_transition_accuracy": 0.5,
    "minimum_rgb_gain_over_time_only": 0.1,
    "minimum_normal_gain_over_shuffle": 0.15,
    "minimum_tcn_gain": 0.03,
    "required_label_source": "rgb_minimap_teacher_v1",
    "combat_model_sha256": "bce47dc1dc6332b7e348cfc6d6a9874efbbffadca14301dbfbe3bffa6063bd74",
    "human_labels_used": False,
    "semantic_accuracy_verified": False,
    "promotion_allowed": False,
    "control_output": False,
    "device_input_allowed": False,
}


def _self_hash(value: Mapping[str, object], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _contract(path: Path) -> dict[str, object]:
    value = _read_object(path, "Operation Policy v1 contract is unreadable")
    if (
        value.get("contract_sha256") != _self_hash(value, "contract_sha256")
        or {key: item for key, item in value.items() if key != "contract_sha256"}
        != EXPECTED_CONTRACT
    ):
        raise OperationPolicyError("Operation Policy v1 contract differs")
    return value


def verify_operation_policy_contract(path: Path) -> dict[str, object]:
    contract = _contract(path)
    return {
        "schema_version": CONTRACT_CHECK_SCHEMA,
        "status": "PASSED",
        "contract_sha256": contract["contract_sha256"],
        "learned_heads": contract["learned_heads"],
        "deterministic_heads": contract["deterministic_heads"],
        "video_test_accessed": False,
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "promotion_allowed": False,
        "control_output": False,
        "device_input_allowed": False,
    }


def _direct_contract(path: Path) -> dict[str, object]:
    value = _read_object(path, "Operation Direct Policy v1 contract is unreadable")
    if (
        value.get("contract_sha256") != _self_hash(value, "contract_sha256")
        or {key: item for key, item in value.items() if key != "contract_sha256"}
        != EXPECTED_DIRECT_CONTRACT
    ):
        raise OperationPolicyError("Operation Direct Policy v1 contract differs")
    return value


def verify_operation_direct_policy_contract(path: Path) -> dict[str, object]:
    contract = _direct_contract(path)
    return {
        "schema_version": "hok-agent-operation-direct-policy-contract-check-v1",
        "status": "PASSED",
        "contract_sha256": contract["contract_sha256"],
        "supervision_source": "executed_action",
        "views": contract["views"],
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "promotion_allowed": False,
        "control_output": False,
        "device_input_allowed": False,
    }


def _movement_policy_contract(path: Path) -> dict[str, object]:
    value = _read_object(path, "Operation Movement Policy contract is unreadable")
    if (
        value.get("contract_sha256") != _self_hash(value, "contract_sha256")
        or {key: item for key, item in value.items() if key != "contract_sha256"}
        != EXPECTED_MOVEMENT_CONTRACT
    ):
        raise OperationPolicyError("Operation Movement Policy contract differs")
    return value


def verify_operation_movement_policy_contract(path: Path) -> dict[str, object]:
    contract = _movement_policy_contract(path)
    return {
        "schema_version": "hok-agent-operation-movement-policy-contract-check-v1",
        "status": "PASSED",
        "contract_sha256": contract["contract_sha256"],
        "required_label_source": contract["required_label_source"],
        "combat_model_sha256": contract["combat_model_sha256"],
        "control_output": False,
        "device_input_allowed": False,
    }


@dataclass(frozen=True)
class SourceSession:
    identity: str
    main_rgb: np.ndarray
    hud_rgb: np.ndarray
    minimap_rgb: np.ndarray
    timestamp_ms: np.ndarray
    movement_id: np.ndarray
    combat_id: np.ndarray
    hard_stop: np.ndarray
    movement_confidence: np.ndarray | None = None
    movement_label_source: np.ndarray | None = None


@dataclass(frozen=True)
class EncodedSession:
    identity: str
    main: np.ndarray
    hud: np.ndarray
    minimap: np.ndarray
    timestamp_ms: np.ndarray
    movement_id: np.ndarray
    combat_id: np.ndarray
    hard_stop: np.ndarray
    movement_confidence: np.ndarray | None = None
    movement_label_source: np.ndarray | None = None


def _verified_summary(path: Path, schema: str | tuple[str, ...]) -> dict[str, object]:
    value = _read_object(path, "operation-policy source summary is unreadable")
    schemas = (schema,) if isinstance(schema, str) else schema
    if (
        value.get("schema_version") not in schemas
        or value.get("status") != "PASSED"
        or value.get("strict_passed") is not True
        or value.get("summary_sha256") != _self_hash(value, "summary_sha256")
    ):
        raise OperationPolicyError("operation-policy source summary is invalid")
    return value


def _source_identity(summary: Mapping[str, object], shards: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "summary_sha256": summary["summary_sha256"],
                "events_sha256": summary["events_sha256"],
                "shards": [row["sha256"] for row in shards],
            }
        )
    ).hexdigest()


def _shard_rows(summary: Mapping[str, object], field: str) -> list[dict[str, object]]:
    rows = summary.get(field)
    if not isinstance(rows, list) or not rows:
        raise OperationPolicyError("operation-policy source has no frozen shards")
    return [cast(dict[str, object], row) for row in rows]


def _checked_npz(root: Path, row: Mapping[str, object]) -> np.lib.npyio.NpzFile:
    name = row.get("path")
    if not isinstance(name, str) or Path(name).name != name:
        raise OperationPolicyError("operation-policy shard name is invalid")
    path = root / "shards" / name
    if row.get("sha256") != _sha(path):
        raise OperationPolicyError("operation-policy shard hash differs")
    return cast(np.lib.npyio.NpzFile, np.load(path, allow_pickle=False))


def _resize_crops(
    frames: np.ndarray, box: tuple[int, int, int, int], rois: ObservationROIs
) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise OperationPolicyError("operation-policy RGB source is invalid")
    height, width = frames.shape[1:3]
    x0, y0, x1, y1 = box
    sx0 = max(0, min(width - 1, round(x0 * width / rois.width)))
    sx1 = max(sx0 + 1, min(width, round(x1 * width / rois.width)))
    sy0 = max(0, min(height - 1, round(y0 * height / rois.height)))
    sy1 = max(sy0 + 1, min(height, round(y1 * height / rois.height)))
    crop = torch.from_numpy(frames[:, sy0:sy1, sx0:sx1]).permute(0, 3, 1, 2).float()
    resized = nn.functional.interpolate(crop, size=(128, 128), mode="bilinear", align_corners=False)
    return cast(np.ndarray, resized.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy())


def _load_operation_session(path: Path) -> SourceSession:
    root = _large_existing(path)
    summary = _verified_summary(
        root / "summary.json",
        ("hok-agent-mobile-operation-base-v1", "hok-agent-mobile-operation-teacher-v1"),
    )
    rows = _shard_rows(summary, "observation_shards")
    values: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "main_rgb",
            "hud_rgb",
            "minimap_rgb",
            "scheduled_elapsed_ms",
            "movement_id",
            "combat_id",
            "hard_stop",
        )
    }
    for row in rows:
        with _checked_npz(root, row) as shard:
            if not set(values).issubset(shard.files):
                raise OperationPolicyError("operation-policy operation shard fields differ")
            for name in values:
                values[name].append(shard[name])
    arrays = {name: np.concatenate(items) for name, items in values.items()}
    length = len(arrays["movement_id"])
    if (
        length < 2
        or any(len(item) != length for item in arrays.values())
        or any(
            arrays[name].shape[1:] != (128, 128, 3)
            for name in ("main_rgb", "hud_rgb", "minimap_rgb")
        )
    ):
        raise OperationPolicyError("operation-policy operation shard shapes differ")
    confidence_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    for row in rows:
        with _checked_npz(root, row) as shard:
            if "movement_confidence" in shard.files:
                confidence_parts.append(shard["movement_confidence"])
                source_parts.append(shard["movement_label_source"])
    return SourceSession(
        _source_identity(summary, rows),
        arrays["main_rgb"],
        arrays["hud_rgb"],
        arrays["minimap_rgb"],
        arrays["scheduled_elapsed_ms"].astype(np.int64),
        arrays["movement_id"].astype(np.int64),
        arrays["combat_id"].astype(np.int64),
        arrays["hard_stop"].astype(np.uint8),
        np.concatenate(confidence_parts).astype(np.float32) if confidence_parts else None,
        np.concatenate(source_parts).astype(np.uint8) if source_parts else None,
    )


def _load_combat_session(path: Path, rois: ObservationROIs) -> SourceSession:
    root = _large_existing(path)
    summary = _verified_summary(root / "summary.json", "hok-agent-visual-combat-arbiter-v1")
    if summary.get("training_candidate") is not True:
        raise OperationPolicyError("operation-policy combat source is not a training candidate")
    rows = _shard_rows(summary, "frame_shards")
    rgb: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    sent: list[np.ndarray] = []
    for row in rows:
        with _checked_npz(root, row) as shard:
            required = {"rgb", "scheduled_elapsed_ms", "action_id", "input_sent"}
            if not required.issubset(shard.files):
                raise OperationPolicyError("operation-policy combat shard fields differ")
            rgb.append(shard["rgb"])
            timestamps.append(shard["scheduled_elapsed_ms"])
            actions.append(shard["action_id"])
            sent.append(shard["input_sent"])
    frames = np.concatenate(rgb)
    timestamp = np.maximum.accumulate(np.concatenate(timestamps).astype(np.int64))
    action = np.concatenate(actions).astype(np.int64)
    input_sent = np.concatenate(sent).astype(bool)
    action = np.where(input_sent, action, 0)
    length = len(frames)
    if length < 2 or len(timestamp) != length or len(action) != length:
        raise OperationPolicyError("operation-policy combat shard shapes differ")
    return SourceSession(
        _source_identity(summary, rows),
        _resize_crops(frames, rois.main_view, rois),
        _resize_crops(frames, rois.hud, rois),
        _resize_crops(frames, rois.minimap, rois),
        timestamp,
        np.zeros(length, dtype=np.int64),
        action,
        np.zeros(length, dtype=np.uint8),
    )


def _encoder(adapter: Path, device: torch.device) -> tuple[nn.Module, dict[str, str]]:
    state, metadata = _load_v2_adapter(_large_existing(adapter), device)
    model = resnet18(weights=None)
    model.fc = nn.Identity()
    model.load_state_dict(state, strict=True)
    spatial = nn.Sequential(*list(model.children())[:-2])
    return spatial.to(device).eval(), metadata


def _encode_rgb(
    model: nn.Module, frames: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            tensor = (
                torch.from_numpy(frames[start : start + batch_size])
                .to(device)
                .permute(0, 3, 1, 2)
                .float()
                .div(255.0)
            )
            output.append(model(tensor).cpu().numpy().astype(np.float16, copy=False))
    return np.concatenate(output)


def _encode_session(
    model: nn.Module, session: SourceSession, device: torch.device, batch_size: int
) -> EncodedSession:
    return EncodedSession(
        session.identity,
        _encode_rgb(model, session.main_rgb, device, batch_size),
        _encode_rgb(model, session.hud_rgb, device, batch_size),
        _encode_rgb(model, session.minimap_rgb, device, batch_size),
        session.timestamp_ms,
        session.movement_id,
        session.combat_id,
        session.hard_stop,
        session.movement_confidence,
        session.movement_label_source,
    )


def _future_indices(timestamp_ms: np.ndarray, lag_ms: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.searchsorted(timestamp_ms, timestamp_ms + lag_ms, side="left")
    source = np.arange(len(timestamp_ms), dtype=np.int64)
    valid = candidates < len(timestamp_ms)
    source, candidates = source[valid], candidates[valid]
    close = np.abs((timestamp_ms[candidates] - timestamp_ms[source]) - lag_ms) <= 150
    return source[close], candidates[close]


def _pair_features(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.concatenate((first, second, second - first), axis=1).astype(np.float16, copy=False)


def _movement_pairs(session: EncodedSession, lag_ms: int) -> tuple[np.ndarray, np.ndarray]:
    source, target = _future_indices(session.timestamp_ms, lag_ms)
    valid = (
        (session.movement_id[source] == session.movement_id[target])
        & (session.hard_stop[source] == 0)
        & (session.hard_stop[target] == 0)
    )
    source, target = source[valid], target[valid]
    return _pair_features(session.main[source], session.main[target]), session.movement_id[source]


def _combat_pairs(session: EncodedSession, lag_ms: int) -> tuple[np.ndarray, np.ndarray]:
    source, target = _future_indices(session.timestamp_ms, lag_ms)
    kept_source: list[int] = []
    kept_target: list[int] = []
    labels: list[int] = []
    for first, last in zip(source.tolist(), target.tolist(), strict=True):
        if session.hard_stop[first] or session.hard_stop[last]:
            continue
        events = session.combat_id[first + 1 : last + 1]
        nonzero = events[events > 0]
        unique = np.unique(nonzero)
        if len(unique) > 1:
            continue
        kept_source.append(first)
        kept_target.append(last)
        labels.append(int(unique[0]) if len(unique) else 0)
    if not labels:
        return np.empty((0, FEATURE_SIZE * 6, 4, 4), dtype=np.float16), np.empty(0, dtype=np.int64)
    first_index = np.asarray(kept_source)
    second_index = np.asarray(kept_target)
    first = np.concatenate((session.main[first_index], session.hud[first_index]), axis=1)
    second = np.concatenate((session.main[second_index], session.hud[second_index]), axis=1)
    return _pair_features(first, second), np.asarray(labels, dtype=np.int64)


def _cap_negative(
    features: np.ndarray, labels: np.ndarray, ratio: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(labels != 0)
    negative = np.flatnonzero(labels == 0)
    maximum = len(positive) * ratio
    if not len(positive) or len(negative) <= maximum:
        return features, labels
    randomizer = np.random.default_rng(seed)
    randomizer.shuffle(negative)
    selected = np.sort(np.concatenate((positive, negative[:maximum])))
    return features[selected], labels[selected]


class _IDMHead(nn.Module):
    def __init__(self, input_size: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_size, 128, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(values))


def _class_weights(labels: np.ndarray, classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    weights = np.sqrt(max(float(counts.sum()), 1.0) / np.maximum(counts, 1.0))
    weights = np.minimum(weights / max(float(weights.mean()), 1e-8), 5.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _predict_idm(
    model: _IDMHead, features: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            logits = model(
                torch.from_numpy(features[start : start + batch_size]).to(device).float()
            )
            probabilities = logits.softmax(1)
            confidence, label = probabilities.max(1)
            labels.append(label.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
    return np.concatenate(labels), np.concatenate(confidences)


def _fit_idm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    classes: int,
    epochs: int,
    device: torch.device,
    batch_size: int,
    *,
    shuffled: bool,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    randomizer = np.random.default_rng(20260817)
    labels = train_y.copy()
    if shuffled:
        randomizer.shuffle(labels)
    model = _IDMHead(train_x.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(weight=_class_weights(labels, classes, device))
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    best_metrics: dict[str, object] = {}
    for epoch in range(1, epochs + 1):
        order = randomizer.permutation(len(train_x))
        model.train()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            logits = model(torch.from_numpy(train_x[rows]).to(device).float())
            target = torch.from_numpy(labels[rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_function(logits, target).backward()
            optimizer.step()
        predicted, _ = _predict_idm(model, dev_x, device, batch_size)
        metrics = _head_metrics(predicted, dev_y, classes)
        score = cast(float, metrics["macro_f1"])
        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    return {
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "train_rows": len(train_x),
        "dev_rows": len(dev_x),
        "shuffled": shuffled,
    }, best_state


def _source_pairs(
    sessions: Sequence[EncodedSession], lag_ms: int, head: str, ratio: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    pieces = [
        (_movement_pairs(item, lag_ms) if head == "movement" else _combat_pairs(item, lag_ms))
        for item in sessions
    ]
    features = np.concatenate([item[0] for item in pieces if len(item[1])])
    labels = np.concatenate([item[1] for item in pieces if len(item[1])])
    if head == "movement":
        return _cap_negative(features, labels, 1, seed=seed)
    return _cap_negative(features, labels, ratio, seed=seed)


def run_operation_idm_pilot(
    *,
    contract_path: Path,
    adapter_checkpoint: Path,
    observation_rois_path: Path,
    operation_train: Path,
    operation_dev: Path,
    combat_root: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 256,
) -> dict[str, object]:
    contract = _contract(contract_path)
    if device not in {"cpu", "cuda"} or batch_size < 8:
        raise OperationPolicyError("operation IDM runtime settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise OperationPolicyError("CUDA is unavailable")
    target = torch.device(device)
    rois, roi_sha = load_observation_rois(observation_rois_path)
    combat_directory = _large_existing(combat_root)
    combat_paths = tuple(
        sorted(
            path
            for path in combat_directory.iterdir()
            if path.is_dir() and path.name.startswith("formal-session-")
        )
    )
    if len(combat_paths) != 8:
        raise OperationPolicyError("operation IDM requires exactly eight combat sessions")
    train_sources = [_load_operation_session(operation_train)] + [
        _load_combat_session(path, rois) for path in combat_paths[:6]
    ]
    dev_sources = [_load_operation_session(operation_dev)] + [
        _load_combat_session(path, rois) for path in combat_paths[6:]
    ]
    encoder, adapter_metadata = _encoder(adapter_checkpoint, target)
    train = [_encode_session(encoder, item, target, batch_size) for item in train_sources]
    dev = [_encode_session(encoder, item, target, batch_size) for item in dev_sources]
    output = _large_new(output_dir)
    ratio = cast(int, contract["maximum_negative_to_positive_ratio"])
    epochs = cast(int, contract["idm_epochs"])
    results: dict[str, dict[str, object]] = {"movement": {}, "combat": {}}
    saved: dict[str, dict[str, torch.Tensor]] = {}
    for head, classes in (("movement", MOVEMENT_SIZE), ("combat", COMBAT_SIZE)):
        for lag in cast(list[int], contract["pair_lags_ms"]):
            train_x, train_y = _source_pairs(train, lag, head, ratio, lag)
            dev_x, dev_y = _source_pairs(dev, lag, head, ratio, lag + 1)
            normal, state = _fit_idm(
                train_x, train_y, dev_x, dev_y, classes, epochs, target, batch_size, shuffled=False
            )
            shuffled, _ = _fit_idm(
                train_x, train_y, dev_x, dev_y, classes, epochs, target, batch_size, shuffled=True
            )
            normal_f1 = cast(float, cast(dict[str, object], normal["metrics"])["macro_f1"])
            shuffle_f1 = cast(float, cast(dict[str, object], shuffled["metrics"])["macro_f1"])
            results[head][str(lag)] = {
                "normal": normal,
                "shuffle": shuffled,
                "normal_minus_shuffle_macro_f1": normal_f1 - shuffle_f1,
                "train_class_counts": np.bincount(train_y, minlength=classes).tolist(),
                "dev_class_counts": np.bincount(dev_y, minlength=classes).tolist(),
            }
            saved[f"{head}-{lag}"] = state
    movement_metrics = [
        cast(dict[str, object], cast(dict[str, object], results["movement"][str(lag)])["normal"])[
            "metrics"
        ]
        for lag in cast(list[int], contract["pair_lags_ms"])
    ]
    combat_metrics = [
        cast(dict[str, object], cast(dict[str, object], results["combat"][str(lag)])["normal"])[
            "metrics"
        ]
        for lag in cast(list[int], contract["pair_lags_ms"])
    ]
    movement_recalls = [
        cast(list[float], cast(dict[str, object], item)["per_class_recall"])
        for item in movement_metrics
    ]
    combat_recalls = [
        cast(list[float], cast(dict[str, object], item)["per_class_recall"])
        for item in combat_metrics
    ]
    movement_shuffle_gain = min(
        cast(
            float,
            cast(dict[str, object], results["movement"][str(lag)])["normal_minus_shuffle_macro_f1"],
        )
        for lag in cast(list[int], contract["pair_lags_ms"])
    )
    combat_shuffle_gain = min(
        cast(
            float,
            cast(dict[str, object], results["combat"][str(lag)])["normal_minus_shuffle_macro_f1"],
        )
        for lag in cast(list[int], contract["pair_lags_ms"])
    )
    movement_admitted = bool(
        min(cast(float, cast(dict[str, object], item)["macro_f1"]) for item in movement_metrics)
        >= cast(float, contract["minimum_movement_idm_macro_f1"])
        and min(row[0] for row in movement_recalls) >= cast(float, contract["minimum_wait_recall"])
        and min(min(row[1:]) for row in movement_recalls)
        >= cast(float, contract["minimum_direction_recall"])
        and movement_shuffle_gain >= cast(float, contract["minimum_shuffle_gain"])
    )
    combat_admitted = bool(
        min(cast(float, cast(dict[str, object], item)["macro_f1"]) for item in combat_metrics)
        >= cast(float, contract["minimum_combat_idm_macro_f1"])
        and min(min(row[1:]) for row in combat_recalls)
        >= cast(float, contract["minimum_non_none_combat_recall"])
        and combat_shuffle_gain >= cast(float, contract["minimum_shuffle_gain"])
    )
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        model_rows: dict[str, dict[str, object]] = {}
        for name, state in saved.items():
            path = staging / f"{name}.safetensors"
            save_file(
                state,
                path,
                metadata={
                    "schema": IDM_SCHEMA,
                    "head": name.rsplit("-", 1)[0],
                    "lag_ms": name.rsplit("-", 1)[1],
                    "contract_sha256": cast(str, contract["contract_sha256"]),
                    "adapter_sha256": _sha(adapter_checkpoint),
                    "observation_rois_sha256": roi_sha,
                },
            )
            model_rows[name] = {"path": path.name, "sha256": _sha(path)}
        report: dict[str, object] = {
            "schema_version": IDM_SCHEMA,
            "status": "PASSED" if movement_admitted else "FAILED",
            "contract_sha256": contract["contract_sha256"],
            "adapter_sha256": _sha(adapter_checkpoint),
            "adapter_source_sha256": adapter_metadata.get("v5_source_model_sha256"),
            "observation_rois_sha256": roi_sha,
            "source_sessions": {
                "train": [item.identity for item in train],
                "dev": [item.identity for item in dev],
            },
            "models": model_rows,
            "metrics": results,
            "movement_admitted": movement_admitted,
            "combat_admitted": combat_admitted,
            "video_test_accessed": False,
            "human_labels_used": False,
            "semantic_accuracy_verified": False,
            "promotion_allowed": False,
            "control_output": False,
            "device_input_allowed": False,
        }
        report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _load_idm_models(
    root: Path, report: Mapping[str, object], device: torch.device
) -> dict[str, _IDMHead]:
    rows = report.get("models")
    if not isinstance(rows, dict):
        raise OperationPolicyError("operation IDM model manifest is invalid")
    models: dict[str, _IDMHead] = {}
    for name, raw in rows.items():
        row = cast(dict[str, object], raw)
        path_name = row.get("path")
        if (
            not isinstance(name, str)
            or not isinstance(path_name, str)
            or Path(path_name).name != path_name
        ):
            raise OperationPolicyError("operation IDM model name is invalid")
        path = root / path_name
        if row.get("sha256") != _sha(path):
            raise OperationPolicyError("operation IDM model hash differs")
        head = name.rsplit("-", 1)[0]
        classes = MOVEMENT_SIZE if head == "movement" else COMBAT_SIZE
        input_size = FEATURE_SIZE * 3 if head == "movement" else FEATURE_SIZE * 6
        model = _IDMHead(input_size, classes).to(device)
        model.load_state_dict(load_file(path, device=str(device)), strict=True)
        model.eval()
        models[name] = model
    return models


def _target_manifest(path: Path) -> tuple[dict[str, object], str]:
    manifest = _read_object(path / "manifest.json", "V5 target manifest is unreadable")
    supplied = manifest.get("manifest_sha256")
    if manifest.get("schema_version") != "hok-agent-v5-manifest-v2" or supplied != _self_hash(
        manifest, "manifest_sha256"
    ):
        raise OperationPolicyError("V5 target manifest identity is invalid")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list):
        raise OperationPolicyError("V5 target sessions are invalid")
    counts = {
        split: sum(cast(dict[str, object], row).get("split") == split for row in sessions)
        for split in ("train", "dev")
    }
    if counts != {"train": 103, "dev": 23}:
        raise OperationPolicyError("V5 target train/dev split differs")
    return manifest, supplied


def _target_sessions(
    root: Path, manifest: Mapping[str, object], split: str
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    if split not in {"train", "dev"}:
        raise OperationPolicyError("operation policy may open video-train or video-dev only")
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise OperationPolicyError("V5 target shards are invalid")
    grouped: dict[str, list[dict[str, object]]] = {}
    for raw in rows:
        row = cast(dict[str, object], raw)
        if row.get("split") != split:
            continue
        identities = row.get("session_hashes")
        if (
            not isinstance(identities, list)
            or len(identities) != 1
            or not isinstance(identities[0], str)
        ):
            raise OperationPolicyError("V5 target shard session binding is invalid")
        grouped.setdefault(identities[0], []).append(row)
    for identity in sorted(grouped):
        frames: list[np.ndarray] = []
        timestamps: list[np.ndarray] = []
        for row in grouped[identity]:
            name = row.get("path")
            if not isinstance(name, str) or Path(name).name != name:
                raise OperationPolicyError("V5 target shard name is invalid")
            path = root / "shards" / name
            if row.get("sha256") != _sha(path):
                raise OperationPolicyError("V5 target shard hash differs")
            with np.load(path, allow_pickle=False) as shard:
                if set(shard["split"].tolist()) != {split} or set(
                    shard["session_hash"].tolist()
                ) != {identity}:
                    raise OperationPolicyError("V5 target shard split differs")
                frames.append(shard["frames"])
                timestamps.append(shard["timestamp_ms"])
        yield identity, np.concatenate(frames), np.concatenate(timestamps).astype(np.int64)


def _pair_predictions(
    models: Mapping[str, _IDMHead],
    head: str,
    lag: int,
    main: np.ndarray,
    hud: np.ndarray,
    timestamps: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, target = _future_indices(timestamps, lag)
    if head == "movement":
        features = _pair_features(main[source], main[target])
    else:
        first = np.concatenate((main[source], hud[source]), axis=1)
        second = np.concatenate((main[target], hud[target]), axis=1)
        features = _pair_features(first, second)
    label, confidence = _predict_idm(models[f"{head}-{lag}"], features, device, batch_size)
    return source, label, confidence


def _consensus_labels(
    models: Mapping[str, _IDMHead],
    head: str,
    main: np.ndarray,
    hud: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    device: torch.device,
    batch_size: int,
    identity: str,
    ratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    outputs = [
        _pair_predictions(models, head, lag, main, hud, timestamps, device, batch_size)
        for lag in (200, 500)
    ]
    labels = np.full(len(timestamps), -1, dtype=np.int8)
    confidence = np.zeros(len(timestamps), dtype=np.float32)
    maps = [
        {
            int(index): (int(label), float(score))
            for index, label, score in zip(*output, strict=True)
        }
        for output in outputs
    ]
    common = sorted(set(maps[0]).intersection(maps[1]))
    for index in common:
        first, second = maps[0][index], maps[1][index]
        if first[0] == second[0] and min(first[1], second[1]) >= threshold:
            labels[index] = first[0]
            confidence[index] = min(first[1], second[1])
    positive = np.flatnonzero(labels > 0)
    if head == "combat" and len(positive):
        last_by_class: dict[int, int] = {}
        for index in positive.tolist():
            class_id = int(labels[index])
            minimum = 500 if class_id == 1 else 1000
            if timestamps[index] - last_by_class.get(class_id, -10_000) < minimum:
                labels[index] = -1
                confidence[index] = 0
            else:
                last_by_class[class_id] = int(timestamps[index])
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels == 0)
    maximum = len(positive) * ratio
    if len(negative) > maximum:
        scored = sorted(
            negative.tolist(),
            key=lambda index: hashlib.sha256(f"{identity}:{head}:{index}".encode()).digest(),
        )
        for index in scored[maximum:]:
            labels[index] = -1
            confidence[index] = 0
    return labels, confidence


def materialize_operation_video_pseudolabels(
    *,
    contract_path: Path,
    idm_dir: Path,
    target_dir: Path,
    adapter_checkpoint: Path,
    observation_rois_path: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 512,
) -> dict[str, object]:
    contract = _contract(contract_path)
    if device not in {"cpu", "cuda"} or batch_size < 8:
        raise OperationPolicyError("operation pseudolabel runtime settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise OperationPolicyError("CUDA is unavailable")
    target_device = torch.device(device)
    idm_root = _large_existing(idm_dir)
    report = _read_object(idm_root / "report.json", "operation IDM report is unreadable")
    if (
        report.get("schema_version") != IDM_SCHEMA
        or report.get("report_sha256") != _self_hash(report, "report_sha256")
        or report.get("movement_admitted") is not True
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("adapter_sha256") != _sha(adapter_checkpoint)
    ):
        raise OperationPolicyError("operation IDM report is not admitted")
    rois, roi_sha = load_observation_rois(observation_rois_path)
    if report.get("observation_rois_sha256") != roi_sha:
        raise OperationPolicyError("operation ROI layout differs from IDM")
    root = _large_existing(target_dir)
    target_manifest, target_sha = _target_manifest(root)
    encoder, metadata = _encoder(adapter_checkpoint, target_device)
    models = _load_idm_models(idm_root, report, target_device)
    output = _large_new(output_dir)
    ratio = cast(int, contract["maximum_negative_to_positive_ratio"])
    threshold = cast(float, contract["teacher_confidence_threshold"])
    manifest_rows: list[dict[str, object]] = []
    movement_counts_by_split = {
        split: np.zeros(MOVEMENT_SIZE, dtype=np.int64) for split in ("train", "dev")
    }
    combat_counts_by_split = {
        split: np.zeros(COMBAT_SIZE, dtype=np.int64) for split in ("train", "dev")
    }
    row_counts = {split: 0 for split in ("train", "dev")}
    movement_accepted = {split: 0 for split in ("train", "dev")}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shards = staging / "shards"
        shards.mkdir()
        for split in ("train", "dev"):
            for ordinal, (identity, frames, timestamps) in enumerate(
                _target_sessions(root, target_manifest, split)
            ):
                main_rgb = _resize_crops(frames, rois.main_view, rois)
                hud_rgb = _resize_crops(frames, rois.hud, rois)
                minimap_rgb = _resize_crops(frames, rois.minimap, rois)
                main = _encode_rgb(encoder, main_rgb, target_device, batch_size)
                hud = _encode_rgb(encoder, hud_rgb, target_device, batch_size)
                minimap = _encode_rgb(encoder, minimap_rgb, target_device, batch_size)
                movement, movement_conf = _consensus_labels(
                    models,
                    "movement",
                    main,
                    hud,
                    timestamps,
                    threshold,
                    target_device,
                    batch_size,
                    identity,
                    ratio,
                )
                if report.get("combat_admitted") is True:
                    combat, combat_conf = _consensus_labels(
                        models,
                        "combat",
                        main,
                        hud,
                        timestamps,
                        threshold,
                        target_device,
                        batch_size,
                        identity,
                        ratio,
                    )
                else:
                    combat = np.full(len(timestamps), -1, dtype=np.int8)
                    combat_conf = np.zeros(len(timestamps), dtype=np.float32)
                name = f"{split}-{ordinal:04d}.npz"
                path = shards / name
                np.savez_compressed(
                    path,
                    main_features=main.mean(axis=(2, 3)).astype(np.float16),
                    hud_features=hud.mean(axis=(2, 3)).astype(np.float16),
                    minimap_features=minimap.mean(axis=(2, 3)).astype(np.float16),
                    movement_id=movement,
                    combat_id=combat,
                    movement_confidence=movement_conf.astype(np.float16),
                    combat_confidence=combat_conf.astype(np.float16),
                    timestamp_ms=timestamps,
                    session_hash=np.asarray([identity] * len(timestamps)),
                )
                movement_counts = np.bincount(movement[movement >= 0], minlength=MOVEMENT_SIZE)
                combat_counts = np.bincount(combat[combat >= 0], minlength=COMBAT_SIZE)
                movement_counts_by_split[split] += movement_counts[:MOVEMENT_SIZE]
                combat_counts_by_split[split] += combat_counts[:COMBAT_SIZE]
                row_counts[split] += len(timestamps)
                movement_accepted[split] += int(np.sum(movement >= 0))
                manifest_rows.append(
                    {
                        "path": name,
                        "sha256": _sha(path),
                        "split": split,
                        "session_hash": identity,
                        "rows": len(timestamps),
                        "movement_counts": movement_counts[:MOVEMENT_SIZE].tolist(),
                        "combat_counts": combat_counts[:COMBAT_SIZE].tolist(),
                    }
                )
                del frames, main_rgb, hud_rgb, minimap_rgb, main, hud, minimap
        minimums = {
            "train": cast(int, contract["minimum_train_samples_per_class"]),
            "dev": cast(int, contract["minimum_dev_samples_per_class"]),
        }
        movement_coverage = {
            split: movement_accepted[split] / max(row_counts[split], 1)
            for split in ("train", "dev")
        }
        movement_admitted = all(
            movement_coverage[split] >= cast(float, contract["minimum_movement_coverage"])
            and min(movement_counts_by_split[split].tolist()) >= minimums[split]
            for split in ("train", "dev")
        )
        combat_admitted = bool(
            report.get("combat_admitted") is True
            and all(
                min(combat_counts_by_split[split][1:].tolist()) >= minimums[split]
                for split in ("train", "dev")
            )
        )
        manifest: dict[str, object] = {
            "schema_version": PSEUDOLABEL_SCHEMA,
            "status": "PASSED" if movement_admitted else "FAILED",
            "contract_sha256": contract["contract_sha256"],
            "idm_report_sha256": report["report_sha256"],
            "adapter_sha256": _sha(adapter_checkpoint),
            "adapter_source_sha256": metadata.get("v5_source_model_sha256"),
            "target_manifest_sha256": target_sha,
            "observation_rois_sha256": roi_sha,
            "feature_shape": [3, FEATURE_SIZE],
            "feature_dtype": "float16",
            "movement_counts": {
                split: movement_counts_by_split[split].tolist() for split in ("train", "dev")
            },
            "combat_counts": {
                split: combat_counts_by_split[split].tolist() for split in ("train", "dev")
            },
            "movement_coverage": movement_coverage,
            "movement_admitted": movement_admitted,
            "combat_admitted": combat_admitted,
            "shards": manifest_rows,
            "raw_rgb_persisted": False,
            "source_paths_persisted": False,
            "video_test_accessed": False,
            "human_labels_used": False,
            "semantic_accuracy_verified": False,
            "promotion_allowed": False,
            "control_output": False,
            "device_input_allowed": False,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return manifest


class _PolicyModel(nn.Module):
    def __init__(self, kind: str, feature_size: int) -> None:
        super().__init__()
        self.kind = kind
        self.representation: nn.Module
        self.mix: nn.Module = nn.Identity()
        self.temporal: nn.Module = nn.Identity()
        if kind == "last_frame":
            self.representation = nn.Identity()
            hidden = feature_size
        elif kind == "pool_mlp":
            self.representation = nn.Sequential(nn.Linear(feature_size, 256), nn.ReLU())
            hidden = 256
        elif kind == "causal_tcn":
            self.mix = nn.Conv1d(feature_size, 256, 1)
            self.temporal = nn.Sequential(*(_V2ResidualBlock(value) for value in (1, 2, 4)))
            self.representation = nn.Identity()
            hidden = 256
        else:
            raise OperationPolicyError("operation policy model kind is invalid")
        self.movement = nn.Linear(hidden, MOVEMENT_SIZE)
        self.combat = nn.Linear(hidden, COMBAT_SIZE)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[1] != WINDOW_FRAMES:
            raise OperationPolicyError("operation policy requires Bx16xF features")
        if self.kind == "last_frame":
            encoded = self.representation(values[:, -1])
        elif self.kind == "pool_mlp":
            encoded = self.representation(values.mean(1))
        else:
            encoded = self.temporal(self.mix(values.transpose(1, 2)))[:, :, -1]
        return self.movement(encoded), self.combat(encoded)


def _pseudolabel_manifest(root: Path, contract: Mapping[str, object]) -> dict[str, object]:
    manifest = _read_object(root / "manifest.json", "operation pseudolabel manifest is unreadable")
    if (
        manifest.get("schema_version") != PSEUDOLABEL_SCHEMA
        or manifest.get("manifest_sha256") != _self_hash(manifest, "manifest_sha256")
        or manifest.get("movement_admitted") is not True
        or manifest.get("contract_sha256") != contract["contract_sha256"]
        or manifest.get("video_test_accessed") is not False
    ):
        raise OperationPolicyError("operation pseudolabel dataset is not admitted")
    return manifest


def _policy_rows(manifest: Mapping[str, object], split: str) -> list[dict[str, object]]:
    if split not in {"train", "dev"}:
        raise OperationPolicyError("operation policy may open train or dev only")
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise OperationPolicyError("operation pseudolabel shards are invalid")
    return [
        cast(dict[str, object], row)
        for row in rows
        if cast(dict[str, object], row).get("split") == split
    ]


def _policy_batches(
    root: Path,
    rows: Sequence[Mapping[str, object]],
    batch_size: int,
    *,
    use_minimap: bool,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    order = np.arange(len(rows))
    randomizer = np.random.default_rng(seed)
    if shuffle:
        randomizer.shuffle(order)
    offsets = np.arange(WINDOW_FRAMES - 1, -1, -1, dtype=np.int64)
    for row_index in order.tolist():
        row = rows[row_index]
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise OperationPolicyError("operation policy shard name is invalid")
        path = root / "shards" / name
        if row.get("sha256") != _sha(path):
            raise OperationPolicyError("operation policy shard hash differs")
        with np.load(path, allow_pickle=False) as shard:
            features = [
                shard["main_features"].astype(np.float32),
                shard["hud_features"].astype(np.float32),
            ]
            if use_minimap:
                features.append(shard["minimap_features"].astype(np.float32))
            fused = np.concatenate(features, axis=1)
            movement = shard["movement_id"].astype(np.int64)
            combat = shard["combat_id"].astype(np.int64)
            timestamps = shard["timestamp_ms"].astype(np.int64)
            indices = np.flatnonzero(
                ((movement >= 0) | (combat >= 0)) & (np.arange(len(fused)) >= WINDOW_FRAMES - 1)
            )
            if shuffle:
                randomizer.shuffle(indices)
            denominator = max(int(timestamps[-1] - timestamps[0]), 1)
            normalized_time = (timestamps - timestamps[0]) / denominator
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                windows = fused[selected[:, None] - offsets[None, :]]
                yield (
                    windows,
                    movement[selected],
                    combat[selected],
                    normalized_time[selected].astype(np.float32),
                )


def _policy_metrics(
    model: _PolicyModel,
    root: Path,
    rows: Sequence[Mapping[str, object]],
    device: torch.device,
    batch_size: int,
    use_minimap: bool,
) -> dict[str, object]:
    predicted_movement: list[np.ndarray] = []
    target_movement: list[np.ndarray] = []
    predicted_combat: list[np.ndarray] = []
    target_combat: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for values, movement, combat, _ in _policy_batches(
            root, rows, batch_size, use_minimap=use_minimap, shuffle=False, seed=0
        ):
            movement_logits, combat_logits = model(torch.from_numpy(values).to(device))
            valid_movement = movement >= 0
            valid_combat = combat >= 0
            if np.any(valid_movement):
                predicted_movement.append(movement_logits.argmax(1).cpu().numpy()[valid_movement])
                target_movement.append(movement[valid_movement])
            if np.any(valid_combat):
                predicted_combat.append(combat_logits.argmax(1).cpu().numpy()[valid_combat])
                target_combat.append(combat[valid_combat])
    movement_metrics = _head_metrics(
        np.concatenate(predicted_movement), np.concatenate(target_movement), MOVEMENT_SIZE
    )
    combat_metrics = (
        _head_metrics(np.concatenate(predicted_combat), np.concatenate(target_combat), COMBAT_SIZE)
        if target_combat
        else None
    )
    return {"movement": movement_metrics, "combat": combat_metrics}


def _fit_policy(
    kind: str,
    root: Path,
    train_rows: Sequence[Mapping[str, object]],
    dev_rows: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    *,
    use_minimap: bool,
    shuffled_labels: bool,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    feature_size = FEATURE_SIZE * (3 if use_minimap else 2)
    model = _PolicyModel(kind, feature_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    movement_counts = np.asarray(cast(dict[str, object], contract["_movement_counts"])["train"])
    combat_counts = np.asarray(cast(dict[str, object], contract["_combat_counts"])["train"])
    movement_loss = nn.CrossEntropyLoss(
        weight=_class_weights(
            np.repeat(np.arange(MOVEMENT_SIZE), movement_counts), MOVEMENT_SIZE, device
        )
    )
    combat_loss = nn.CrossEntropyLoss(
        weight=_class_weights(np.repeat(np.arange(COMBAT_SIZE), combat_counts), COMBAT_SIZE, device)
    )
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, object] = {}
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, cast(int, contract["policy_epochs"]) + 1):
        model.train()
        generator = torch.Generator(device=device).manual_seed(20260817 + epoch)
        for values, movement, combat, _ in _policy_batches(
            root, train_rows, batch_size, use_minimap=use_minimap, shuffle=True, seed=epoch
        ):
            if shuffled_labels:
                movement = movement.copy()
                combat = combat.copy()
                valid = np.flatnonzero(movement >= 0)
                movement[valid] = movement[
                    valid[
                        torch.randperm(len(valid), generator=generator, device=device).cpu().numpy()
                    ]
                ]
                valid = np.flatnonzero(combat >= 0)
                combat[valid] = combat[
                    valid[
                        torch.randperm(len(valid), generator=generator, device=device).cpu().numpy()
                    ]
                ]
            movement_logits, combat_logits = model(torch.from_numpy(values).to(device))
            losses: list[torch.Tensor] = []
            valid_movement = movement >= 0
            valid_combat = combat >= 0
            if np.any(valid_movement):
                losses.append(
                    movement_loss(
                        movement_logits[valid_movement],
                        torch.from_numpy(movement[valid_movement]).to(device),
                    )
                )
            if np.any(valid_combat):
                losses.append(
                    combat_loss(
                        combat_logits[valid_combat],
                        torch.from_numpy(combat[valid_combat]).to(device),
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            if not losses:
                continue
            torch.stack(losses).sum().backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        metrics = _policy_metrics(model, root, dev_rows, device, batch_size, use_minimap)
        movement_f1 = cast(float, cast(dict[str, object], metrics["movement"])["macro_f1"])
        combat_value = metrics["combat"]
        combat_f1 = (
            cast(float, cast(dict[str, object], combat_value)["macro_f1"])
            if combat_value is not None
            else movement_f1
        )
        score = 0.5 * (movement_f1 + combat_f1)
        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    return {
        "kind": kind,
        "use_minimap": use_minimap,
        "shuffled_labels": shuffled_labels,
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "selection_score": best_score,
    }, best_state


def _time_only_metrics(
    root: Path,
    train_rows: Sequence[Mapping[str, object]],
    dev_rows: Sequence[Mapping[str, object]],
    batch_size: int,
) -> dict[str, object]:
    bins = 20
    movement_counts = np.zeros((bins, MOVEMENT_SIZE), dtype=np.int64)
    combat_counts = np.zeros((bins, COMBAT_SIZE), dtype=np.int64)
    for _values, movement, combat, time_value in _policy_batches(
        root, train_rows, batch_size, use_minimap=False, shuffle=False, seed=0
    ):
        indexes = np.minimum((time_value * bins).astype(np.int64), bins - 1)
        for index, label in zip(indexes, movement, strict=True):
            if label >= 0:
                movement_counts[index, label] += 1
        for index, label in zip(indexes, combat, strict=True):
            if label >= 0:
                combat_counts[index, label] += 1
    movement_rule = movement_counts.argmax(1)
    combat_rule = combat_counts.argmax(1)
    predicted_movement: list[np.ndarray] = []
    target_movement: list[np.ndarray] = []
    predicted_combat: list[np.ndarray] = []
    target_combat: list[np.ndarray] = []
    for _values, movement, combat, time_value in _policy_batches(
        root, dev_rows, batch_size, use_minimap=False, shuffle=False, seed=0
    ):
        indexes = np.minimum((time_value * bins).astype(np.int64), bins - 1)
        valid = movement >= 0
        predicted_movement.append(movement_rule[indexes][valid])
        target_movement.append(movement[valid])
        valid = combat >= 0
        if np.any(valid):
            predicted_combat.append(combat_rule[indexes][valid])
            target_combat.append(combat[valid])
    return {
        "movement": _head_metrics(
            np.concatenate(predicted_movement), np.concatenate(target_movement), MOVEMENT_SIZE
        ),
        "combat": _head_metrics(
            np.concatenate(predicted_combat), np.concatenate(target_combat), COMBAT_SIZE
        )
        if target_combat
        else None,
    }


def train_operation_policy_pilot(
    *,
    contract_path: Path,
    dataset_root: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 128,
) -> dict[str, object]:
    contract = _contract(contract_path)
    if device not in {"cpu", "cuda"} or batch_size < 8:
        raise OperationPolicyError("operation policy runtime settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise OperationPolicyError("CUDA is unavailable")
    target = torch.device(device)
    root = _large_existing(dataset_root)
    manifest = _pseudolabel_manifest(root, contract)
    train_rows, dev_rows = _policy_rows(manifest, "train"), _policy_rows(manifest, "dev")
    runtime_contract = dict(contract)
    runtime_contract["_movement_counts"] = manifest["movement_counts"]
    runtime_contract["_combat_counts"] = manifest["combat_counts"]
    time_only = _time_only_metrics(root, train_rows, dev_rows, batch_size)
    candidates: dict[str, dict[str, object]] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for kind in ("last_frame", "pool_mlp", "causal_tcn"):
        result, state = _fit_policy(
            kind,
            root,
            train_rows,
            dev_rows,
            runtime_contract,
            target,
            batch_size,
            use_minimap=False,
            shuffled_labels=False,
        )
        candidates[kind], states[kind] = result, state
    shuffle_result, _ = _fit_policy(
        "pool_mlp",
        root,
        train_rows,
        dev_rows,
        runtime_contract,
        target,
        batch_size,
        use_minimap=False,
        shuffled_labels=True,
    )
    simple = max(
        ("last_frame", "pool_mlp"),
        key=lambda name: cast(float, candidates[name]["selection_score"]),
    )
    tcn_gain = cast(float, candidates["causal_tcn"]["selection_score"]) - cast(
        float, candidates[simple]["selection_score"]
    )
    selected = "causal_tcn" if tcn_gain >= cast(float, contract["minimum_tcn_gain"]) else simple
    minimap_result, minimap_state = _fit_policy(
        selected,
        root,
        train_rows,
        dev_rows,
        runtime_contract,
        target,
        batch_size,
        use_minimap=True,
        shuffled_labels=False,
    )
    base_movement_f1 = cast(
        float,
        cast(
            dict[str, object], cast(dict[str, object], candidates[selected]["metrics"])["movement"]
        )["macro_f1"],
    )
    minimap_movement_f1 = cast(
        float,
        cast(dict[str, object], cast(dict[str, object], minimap_result["metrics"])["movement"])[
            "macro_f1"
        ],
    )
    use_minimap = minimap_movement_f1 - base_movement_f1 >= cast(
        float, contract["minimum_minimap_gain"]
    )
    selected_result = minimap_result if use_minimap else candidates[selected]
    selected_state = minimap_state if use_minimap else states[selected]
    selected_metrics = cast(dict[str, object], selected_result["metrics"])
    movement_f1 = cast(float, cast(dict[str, object], selected_metrics["movement"])["macro_f1"])
    combat_metrics = selected_metrics["combat"]
    combat_f1 = (
        cast(float, cast(dict[str, object], combat_metrics)["macro_f1"])
        if combat_metrics is not None
        else 0.0
    )
    time_f1 = cast(float, cast(dict[str, object], time_only["movement"])["macro_f1"])
    shuffle_f1 = cast(
        float,
        cast(dict[str, object], cast(dict[str, object], shuffle_result["metrics"])["movement"])[
            "macro_f1"
        ],
    )
    movement_admitted = bool(
        movement_f1 >= cast(float, contract["minimum_policy_movement_macro_f1"])
        and movement_f1 - time_f1 >= cast(float, contract["minimum_rgb_gain_over_time_only"])
        and movement_f1 - shuffle_f1 >= cast(float, contract["minimum_shuffle_gain"])
    )
    combat_admitted = bool(
        manifest.get("combat_admitted") is True
        and combat_f1 >= cast(float, contract["minimum_policy_combat_macro_f1"])
    )
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        model_path = staging / "selected-model-seed-0.safetensors"
        save_file(
            selected_state,
            model_path,
            metadata={
                "schema": POLICY_SCHEMA,
                "kind": selected,
                "use_minimap": str(use_minimap).lower(),
                "contract_sha256": cast(str, contract["contract_sha256"]),
                "dataset_manifest_sha256": cast(str, manifest["manifest_sha256"]),
            },
        )
        decision: dict[str, object] = {
            "schema_version": DECISION_SCHEMA,
            "movement_idm_passed": True,
            "combat_idm_passed": manifest.get("combat_admitted") is True,
            "pseudo_labels_usable": True,
            "movement_policy_admitted": movement_admitted,
            "combat_policy_admitted": combat_admitted,
            "selected_model": selected,
            "minimap_retained": use_minimap,
            "semantic_accuracy_verified": False,
            "promotion_allowed": False,
            "control_output": False,
            "device_input_allowed": False,
            "next_required_action": "prepare_read_only_shadow_contract"
            if movement_admitted
            else "record_offline_failure",
        }
        decision["decision_sha256"] = hashlib.sha256(_canonical(decision)).hexdigest()
        report: dict[str, object] = {
            "schema_version": POLICY_SCHEMA,
            "status": "PASSED" if movement_admitted else "FAILED",
            "contract_sha256": contract["contract_sha256"],
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "time_only": time_only,
            "candidates": candidates,
            "label_shuffle": shuffle_result,
            "tcn_gain_over_best_simple": tcn_gain,
            "minimap_ablation": minimap_result,
            "minimap_movement_gain": minimap_movement_f1 - base_movement_f1,
            "selected": selected_result,
            "selected_model_sha256": _sha(model_path),
            "decision": decision,
            "video_test_accessed": False,
            "human_labels_used": False,
            "semantic_accuracy_verified": False,
            "promotion_allowed": False,
            "control_output": False,
            "device_input_allowed": False,
        }
        report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
        (staging / "decision.json").write_text(
            json.dumps(decision, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


@dataclass(frozen=True)
class DirectExamples:
    windows: np.ndarray
    movement_id: np.ndarray
    combat_id: np.ndarray
    movement_transition: np.ndarray
    normalized_time: np.ndarray


def _direct_examples(
    sessions: Sequence[EncodedSession], negative_ratio: int, *, seed: int
) -> DirectExamples:
    windows: list[np.ndarray] = []
    movements: list[np.ndarray] = []
    combats: list[np.ndarray] = []
    transitions: list[np.ndarray] = []
    times: list[np.ndarray] = []
    offsets = np.arange(WINDOW_FRAMES - 1, -1, -1, dtype=np.int64)
    for session in sessions:
        pooled = np.concatenate(
            (
                session.main.mean(axis=(2, 3)),
                session.hud.mean(axis=(2, 3)),
                session.minimap.mean(axis=(2, 3)),
            ),
            axis=1,
        ).astype(np.float16, copy=False)
        indexes = np.arange(WINDOW_FRAMES - 1, len(pooled), dtype=np.int64)
        denominator = max(int(session.timestamp_ms[-1] - session.timestamp_ms[0]), 1)
        windows.append(pooled[indexes[:, None] - offsets[None, :]])
        movements.append(session.movement_id[indexes].astype(np.int64))
        combats.append(session.combat_id[indexes].astype(np.int64))
        transitions.append(
            (session.movement_id[indexes] != session.movement_id[indexes - 1]).astype(bool)
        )
        times.append(
            ((session.timestamp_ms[indexes] - session.timestamp_ms[0]) / denominator).astype(
                np.float32
            )
        )
    all_windows = np.concatenate(windows)
    movement = np.concatenate(movements)
    combat = np.concatenate(combats)
    transition = np.concatenate(transitions)
    normalized_time = np.concatenate(times)
    randomizer = np.random.default_rng(seed)
    for labels, ratio in ((movement, 1), (combat, negative_ratio)):
        positive = np.flatnonzero(labels != 0)
        negative = np.flatnonzero(labels == 0)
        maximum = len(positive) * ratio
        if len(negative) > maximum:
            randomizer.shuffle(negative)
            labels[negative[maximum:]] = -1
    keep = (movement >= 0) | (combat >= 0)
    return DirectExamples(
        all_windows[keep],
        movement[keep],
        combat[keep],
        transition[keep],
        normalized_time[keep],
    )


def _direct_metrics(
    model: _PolicyModel,
    data: DirectExamples,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    movement_predictions: list[np.ndarray] = []
    combat_predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data.windows), batch_size):
            values = torch.from_numpy(data.windows[start : start + batch_size]).to(device).float()
            movement, combat = model(values)
            movement_predictions.append(movement.argmax(1).cpu().numpy())
            combat_predictions.append(combat.argmax(1).cpu().numpy())
    movement_prediction = np.concatenate(movement_predictions)
    combat_prediction = np.concatenate(combat_predictions)
    movement_valid = data.movement_id >= 0
    combat_valid = data.combat_id >= 0
    transition_valid = movement_valid & data.movement_transition
    return {
        "movement": _head_metrics(
            movement_prediction[movement_valid], data.movement_id[movement_valid], MOVEMENT_SIZE
        ),
        "combat": _head_metrics(
            combat_prediction[combat_valid], data.combat_id[combat_valid], COMBAT_SIZE
        ),
        "movement_transition_rows": int(np.sum(transition_valid)),
        "movement_transition_accuracy": float(
            np.mean(movement_prediction[transition_valid] == data.movement_id[transition_valid])
        )
        if np.any(transition_valid)
        else 0.0,
    }


def _fit_direct_policy(
    kind: str,
    train: DirectExamples,
    dev: DirectExamples,
    contract: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    *,
    shuffled: bool,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    randomizer = np.random.default_rng(20260818)
    movement_labels = train.movement_id.copy()
    combat_labels = train.combat_id.copy()
    if shuffled:
        for labels in (movement_labels, combat_labels):
            valid = np.flatnonzero(labels >= 0)
            labels[valid] = labels[randomizer.permutation(valid)]
    model = _PolicyModel(kind, FEATURE_SIZE * 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    movement_loss = nn.CrossEntropyLoss(
        weight=_class_weights(movement_labels[movement_labels >= 0], MOVEMENT_SIZE, device)
    )
    combat_loss = nn.CrossEntropyLoss(
        weight=_class_weights(combat_labels[combat_labels >= 0], COMBAT_SIZE, device)
    )
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, object] = {}
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, cast(int, contract["epochs"]) + 1):
        order = randomizer.permutation(len(train.windows))
        model.train()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            values = torch.from_numpy(train.windows[rows]).to(device).float()
            movement_logits, combat_logits = model(values)
            losses: list[torch.Tensor] = []
            movement_valid = movement_labels[rows] >= 0
            if np.any(movement_valid):
                losses.append(
                    movement_loss(
                        movement_logits[movement_valid],
                        torch.from_numpy(movement_labels[rows][movement_valid]).to(device),
                    )
                )
            combat_valid = combat_labels[rows] >= 0
            if np.any(combat_valid):
                losses.append(
                    combat_loss(
                        combat_logits[combat_valid],
                        torch.from_numpy(combat_labels[rows][combat_valid]).to(device),
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            torch.stack(losses).sum().backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        metrics = _direct_metrics(model, dev, device, batch_size)
        movement_f1 = cast(float, cast(dict[str, object], metrics["movement"])["macro_f1"])
        combat_f1 = cast(float, cast(dict[str, object], metrics["combat"])["macro_f1"])
        score = 0.5 * (movement_f1 + combat_f1)
        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    return {
        "kind": kind,
        "shuffled": shuffled,
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "metrics": best_metrics,
    }, best_state


def _direct_time_only(train: DirectExamples, dev: DirectExamples) -> dict[str, object]:
    bins = 20
    rules: list[np.ndarray] = []
    for labels, classes in ((train.movement_id, MOVEMENT_SIZE), (train.combat_id, COMBAT_SIZE)):
        counts = np.zeros((bins, classes), dtype=np.int64)
        indexes = np.minimum((train.normalized_time * bins).astype(np.int64), bins - 1)
        for index, label in zip(indexes, labels, strict=True):
            if label >= 0:
                counts[index, label] += 1
        rules.append(counts.argmax(1))
    indexes = np.minimum((dev.normalized_time * bins).astype(np.int64), bins - 1)
    movement_valid = dev.movement_id >= 0
    combat_valid = dev.combat_id >= 0
    return {
        "movement": _head_metrics(
            rules[0][indexes][movement_valid], dev.movement_id[movement_valid], MOVEMENT_SIZE
        ),
        "combat": _head_metrics(
            rules[1][indexes][combat_valid], dev.combat_id[combat_valid], COMBAT_SIZE
        ),
    }


def run_operation_direct_policy_pilot(
    *,
    contract_path: Path,
    adapter_checkpoint: Path,
    observation_rois_path: Path,
    operation_train: Path,
    operation_dev: Path,
    combat_root: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 128,
) -> dict[str, object]:
    contract = _direct_contract(contract_path)
    if device not in {"cpu", "cuda"} or batch_size < 8:
        raise OperationPolicyError("operation direct-policy runtime settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise OperationPolicyError("CUDA is unavailable")
    target = torch.device(device)
    rois, roi_sha = load_observation_rois(observation_rois_path)
    combat_directory = _large_existing(combat_root)
    combat_paths = tuple(
        sorted(
            path
            for path in combat_directory.iterdir()
            if path.is_dir() and path.name.startswith("formal-session-")
        )
    )
    if len(combat_paths) != 8:
        raise OperationPolicyError("operation direct policy requires eight combat sessions")
    encoder, adapter_metadata = _encoder(adapter_checkpoint, target)
    train_sessions = [_load_operation_session(operation_train)] + [
        _load_combat_session(path, rois) for path in combat_paths[:6]
    ]
    dev_sessions = [_load_operation_session(operation_dev)] + [
        _load_combat_session(path, rois) for path in combat_paths[6:]
    ]
    train_encoded = [
        _encode_session(encoder, session, target, batch_size) for session in train_sessions
    ]
    dev_encoded = [
        _encode_session(encoder, session, target, batch_size) for session in dev_sessions
    ]
    ratio = cast(int, contract["maximum_negative_to_positive_ratio"])
    train = _direct_examples(train_encoded, ratio, seed=0)
    dev = _direct_examples(dev_encoded, ratio, seed=1)
    time_only = _direct_time_only(train, dev)
    candidates: dict[str, dict[str, object]] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for kind in ("last_frame", "pool_mlp", "causal_tcn"):
        result, state = _fit_direct_policy(
            kind, train, dev, contract, target, batch_size, shuffled=False
        )
        candidates[kind], states[kind] = result, state
    shuffle, _ = _fit_direct_policy(
        "pool_mlp", train, dev, contract, target, batch_size, shuffled=True
    )
    simple = max(
        ("last_frame", "pool_mlp"),
        key=lambda name: cast(float, candidates[name]["selection_score"]),
    )
    tcn_gain = cast(float, candidates["causal_tcn"]["selection_score"]) - cast(
        float, candidates[simple]["selection_score"]
    )
    selected = "causal_tcn" if tcn_gain >= cast(float, contract["minimum_tcn_gain"]) else simple
    metrics = cast(dict[str, object], candidates[selected]["metrics"])
    movement_f1 = cast(float, cast(dict[str, object], metrics["movement"])["macro_f1"])
    combat_f1 = cast(float, cast(dict[str, object], metrics["combat"])["macro_f1"])
    time_f1 = cast(float, cast(dict[str, object], time_only["movement"])["macro_f1"])
    shuffle_metrics = cast(dict[str, object], shuffle["metrics"])
    shuffle_f1 = cast(float, cast(dict[str, object], shuffle_metrics["movement"])["macro_f1"])
    transition_accuracy = cast(float, metrics["movement_transition_accuracy"])
    movement_admitted = bool(
        movement_f1 >= cast(float, contract["minimum_movement_macro_f1"])
        and transition_accuracy >= cast(float, contract["minimum_transition_accuracy"])
        and movement_f1 - time_f1 >= cast(float, contract["minimum_rgb_gain_over_time_only"])
        and movement_f1 - shuffle_f1 >= cast(float, contract["minimum_normal_gain_over_shuffle"])
    )
    combat_admitted = combat_f1 >= cast(float, contract["minimum_combat_macro_f1"])
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        model_path = staging / "selected-model-seed-0.safetensors"
        save_file(
            states[selected],
            model_path,
            metadata={
                "schema": DIRECT_POLICY_SCHEMA,
                "kind": selected,
                "contract_sha256": cast(str, contract["contract_sha256"]),
                "adapter_sha256": _sha(adapter_checkpoint),
                "observation_rois_sha256": roi_sha,
            },
        )
        report: dict[str, object] = {
            "schema_version": DIRECT_POLICY_SCHEMA,
            "status": "PASSED" if movement_admitted else "FAILED",
            "contract_sha256": contract["contract_sha256"],
            "adapter_sha256": _sha(adapter_checkpoint),
            "adapter_source_sha256": adapter_metadata.get("v5_source_model_sha256"),
            "observation_rois_sha256": roi_sha,
            "source_sessions": {
                "train": [session.identity for session in train_encoded],
                "dev": [session.identity for session in dev_encoded],
            },
            "train_rows": len(train.windows),
            "dev_rows": len(dev.windows),
            "time_only": time_only,
            "candidates": candidates,
            "label_shuffle": shuffle,
            "tcn_gain_over_best_simple": tcn_gain,
            "selected": selected,
            "selected_model_sha256": _sha(model_path),
            "movement_admitted": movement_admitted,
            "combat_admitted": combat_admitted,
            "phone_connected": False,
            "human_labels_used": False,
            "semantic_accuracy_verified": False,
            "promotion_allowed": False,
            "control_output": False,
            "device_input_allowed": False,
        }
        report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def freeze_operation_movement_split(
    *, dataset_root: Path, contract_path: Path, output_path: Path, pilot: bool
) -> dict[str, object]:
    contract = _movement_policy_contract(contract_path)
    root = _large_existing(dataset_root)
    expected = cast(int, contract["pilot_sessions" if pilot else "formal_sessions"])
    candidate_paths = tuple(
        sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("teacher-session-")
        )
    )
    paths: list[Path] = []
    for path in candidate_paths:
        summary = _read_object(path / "summary.json", "movement teacher summary is unreadable")
        if summary.get("training_eligible") is True:
            paths.append(path)
    if len(paths) != expected:
        raise OperationPolicyError(
            f"movement split requires exactly {expected} training-eligible teacher sessions"
        )
    sessions: list[SourceSession] = []
    for path in paths:
        summary = _read_object(path / "summary.json", "movement teacher summary is unreadable")
        if (
            summary.get("schema_version") != "hok-agent-mobile-operation-teacher-v1"
            or summary.get("status") != "PASSED"
            or summary.get("training_eligible") is not True
            or summary.get("movement_label_source") != contract["required_label_source"]
            or summary.get("summary_sha256") != _self_hash(summary, "summary_sha256")
        ):
            raise OperationPolicyError("movement teacher session is not training eligible")
        session = _load_operation_session(path)
        if session.movement_label_source is None or set(session.movement_label_source.tolist()) != {
            1
        }:
            raise OperationPolicyError("movement teacher shard label source differs")
        sessions.append(session)
    counts = cast(list[int], contract["pilot_split" if pilot else "formal_split"])
    names = ["train", "dev"] if pilot else ["train", "dev", "test"]
    assignments: dict[str, list[str]] = {name: [] for name in names}
    start = 0
    for name, count in zip(names, counts, strict=True):
        assignments[name] = [session.identity for session in sessions[start : start + count]]
        start += count
    by_identity = {session.identity: session for session in sessions}
    direction_counts: dict[str, list[int]] = {}
    for name, identities in assignments.items():
        labels = np.concatenate(
            [
                by_identity[identity].movement_id[
                    by_identity[identity].hard_stop == 0
                ].astype(np.int64)
                for identity in identities
            ]
        )
        counts_for_split = np.bincount(labels, minlength=MOVEMENT_SIZE).tolist()
        if set(np.flatnonzero(np.asarray(counts_for_split[1:]) > 0).tolist()) != set(range(8)):
            raise OperationPolicyError("movement split lacks one or more non-wait directions")
        direction_counts[name] = counts_for_split
    payload: dict[str, object] = {
        "schema_version": MOVEMENT_SPLIT_SCHEMA,
        "status": "FROZEN",
        "mode": "pilot" if pilot else "formal",
        "contract_sha256": contract["contract_sha256"],
        "session_count": len(sessions),
        "assignments": assignments,
        "direction_counts": direction_counts,
        "label_source": contract["required_label_source"],
        "test_opened": False,
    }
    payload["split_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if output_path.exists() or output_path.resolve().parent != root:
        raise OperationPolicyError("movement split output must be a new file in the dataset root")
    output_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


@dataclass(frozen=True)
class MovementExamples:
    windows: np.ndarray
    labels: np.ndarray
    transitions: np.ndarray
    normalized_time: np.ndarray


def _movement_examples(
    sessions: Sequence[EncodedSession], wait_ratio: int, *, seed: int
) -> MovementExamples:
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    transitions: list[np.ndarray] = []
    times: list[np.ndarray] = []
    offsets = np.arange(WINDOW_FRAMES - 1, -1, -1, dtype=np.int64)
    for session in sessions:
        if session.movement_label_source is None or set(session.movement_label_source.tolist()) != {
            1
        }:
            raise OperationPolicyError("movement examples require the minimap teacher source")
        pooled = np.concatenate(
            (session.main.mean(axis=(2, 3)), session.minimap.mean(axis=(2, 3))), axis=1
        ).astype(np.float16, copy=False)
        candidate = np.arange(WINDOW_FRAMES - 1, len(pooled), dtype=np.int64)
        valid = np.asarray(
            [
                not np.any(session.hard_stop[index - WINDOW_FRAMES + 1 : index + 1])
                for index in candidate
            ],
            dtype=bool,
        )
        indexes = candidate[valid]
        denominator = max(int(session.timestamp_ms[-1] - session.timestamp_ms[0]), 1)
        windows.append(pooled[indexes[:, None] - offsets[None, :]])
        labels.append(session.movement_id[indexes].astype(np.int64))
        transitions.append(
            (session.movement_id[indexes] != session.movement_id[indexes - 1]).astype(bool)
        )
        times.append(
            ((session.timestamp_ms[indexes] - session.timestamp_ms[0]) / denominator).astype(
                np.float32
            )
        )
    all_windows = np.concatenate(windows)
    all_labels = np.concatenate(labels)
    all_transitions = np.concatenate(transitions)
    all_times = np.concatenate(times)
    moving = np.flatnonzero(all_labels != 0)
    waiting = np.flatnonzero(all_labels == 0)
    maximum_wait = len(moving) * wait_ratio
    if len(waiting) > maximum_wait:
        randomizer = np.random.default_rng(seed)
        randomizer.shuffle(waiting)
        keep = np.sort(np.concatenate((moving, waiting[:maximum_wait])))
        all_windows = all_windows[keep]
        all_labels = all_labels[keep]
        all_transitions = all_transitions[keep]
        all_times = all_times[keep]
    return MovementExamples(all_windows, all_labels, all_transitions, all_times)


class _MovementModel(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.representation: nn.Module
        self.mix: nn.Module = nn.Identity()
        self.temporal: nn.Module = nn.Identity()
        if kind == "last_frame":
            self.representation = nn.Identity()
            hidden = FEATURE_SIZE * 2
        elif kind == "pool_mlp":
            self.representation = nn.Sequential(nn.Linear(FEATURE_SIZE * 2, 256), nn.ReLU())
            hidden = 256
        elif kind == "causal_tcn":
            self.mix = nn.Conv1d(FEATURE_SIZE * 2, 256, 1)
            self.temporal = nn.Sequential(*(_V2ResidualBlock(value) for value in (1, 2, 4)))
            self.representation = nn.Identity()
            hidden = 256
        else:
            raise OperationPolicyError("movement model kind is invalid")
        self.head = nn.Linear(hidden, MOVEMENT_SIZE)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (WINDOW_FRAMES, FEATURE_SIZE * 2):
            raise OperationPolicyError("movement model requires Bx16x1024 features")
        if self.kind == "last_frame":
            encoded = self.representation(values[:, -1])
        elif self.kind == "pool_mlp":
            encoded = self.representation(values.mean(1))
        else:
            encoded = self.temporal(self.mix(values.transpose(1, 2)))[:, :, -1]
        return cast(torch.Tensor, self.head(encoded))


def _movement_metrics(
    model: _MovementModel,
    data: MovementExamples,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data.windows), batch_size):
            values = torch.from_numpy(data.windows[start : start + batch_size]).to(device).float()
            predictions.append(model(values).argmax(1).cpu().numpy())
    predicted = np.concatenate(predictions)
    return {
        "movement": _head_metrics(predicted, data.labels, MOVEMENT_SIZE),
        "transition_rows": int(np.sum(data.transitions)),
        "transition_accuracy": float(
            np.mean(predicted[data.transitions] == data.labels[data.transitions])
        )
        if np.any(data.transitions)
        else 0.0,
    }


def _fit_movement_model(
    kind: str,
    train: MovementExamples,
    dev: MovementExamples,
    contract: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    *,
    shuffled: bool,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    randomizer = np.random.default_rng(20260818)
    train_labels = train.labels.copy()
    if shuffled:
        randomizer.shuffle(train_labels)
    model = _MovementModel(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(weight=_class_weights(train_labels, MOVEMENT_SIZE, device))
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, object] = {}
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, cast(int, contract["epochs"]) + 1):
        order = randomizer.permutation(len(train.windows))
        model.train()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            logits = model(torch.from_numpy(train.windows[rows]).to(device).float())
            target = torch.from_numpy(train_labels[rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_function(logits, target).backward()
            optimizer.step()
        metrics = _movement_metrics(model, dev, device, batch_size)
        score = cast(float, cast(dict[str, object], metrics["movement"])["macro_f1"])
        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    return {
        "kind": kind,
        "shuffled": shuffled,
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "selection_score": best_score,
    }, best_state


def _movement_time_only(train: MovementExamples, dev: MovementExamples) -> dict[str, object]:
    bins = 20
    counts = np.zeros((bins, MOVEMENT_SIZE), dtype=np.int64)
    train_bins = np.minimum((train.normalized_time * bins).astype(np.int64), bins - 1)
    for index, label in zip(train_bins, train.labels, strict=True):
        counts[index, label] += 1
    rule = counts.argmax(1)
    dev_bins = np.minimum((dev.normalized_time * bins).astype(np.int64), bins - 1)
    return _head_metrics(rule[dev_bins], dev.labels, MOVEMENT_SIZE)


def run_operation_movement_pilot(
    *,
    dataset_root: Path,
    split_path: Path,
    contract_path: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 128,
) -> dict[str, object]:
    contract = _movement_policy_contract(contract_path)
    if device not in {"cpu", "cuda"} or batch_size < 8:
        raise OperationPolicyError("movement pilot runtime settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise OperationPolicyError("CUDA is unavailable")
    root = _large_existing(dataset_root)
    split = _read_object(split_path, "movement split is unreadable")
    if (
        split.get("schema_version") != MOVEMENT_SPLIT_SCHEMA
        or split.get("status") != "FROZEN"
        or split.get("mode") != "pilot"
        or split.get("contract_sha256") != contract["contract_sha256"]
        or split.get("split_sha256") != _self_hash(split, "split_sha256")
        or split.get("test_opened") is not False
    ):
        raise OperationPolicyError("movement pilot split is invalid")
    paths = tuple(
        sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("teacher-session-")
        )
    )
    sessions = {_load_operation_session(path).identity: path for path in paths}
    assignments = cast(dict[str, list[str]], split["assignments"])
    if set(sessions) != set(assignments["train"] + assignments["dev"]):
        raise OperationPolicyError("movement pilot sessions differ from frozen split")
    target = torch.device(device)
    encoder, adapter_metadata = _encoder(adapter_checkpoint, target)
    encoded = {
        identity: _encode_session(encoder, _load_operation_session(path), target, batch_size)
        for identity, path in sessions.items()
    }
    ratio = cast(int, contract["maximum_wait_to_movement_ratio"])
    train = _movement_examples(
        [encoded[identity] for identity in assignments["train"]], ratio, seed=0
    )
    dev = _movement_examples([encoded[identity] for identity in assignments["dev"]], ratio, seed=1)
    time_only = _movement_time_only(train, dev)
    candidates: dict[str, dict[str, object]] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for kind in ("last_frame", "pool_mlp", "causal_tcn"):
        result, state = _fit_movement_model(
            kind, train, dev, contract, target, batch_size, shuffled=False
        )
        candidates[kind], states[kind] = result, state
    shuffle, _ = _fit_movement_model(
        "pool_mlp", train, dev, contract, target, batch_size, shuffled=True
    )
    simple = max(
        ("last_frame", "pool_mlp"),
        key=lambda name: cast(float, candidates[name]["selection_score"]),
    )
    tcn_gain = cast(float, candidates["causal_tcn"]["selection_score"]) - cast(
        float, candidates[simple]["selection_score"]
    )
    selected = "causal_tcn" if tcn_gain >= cast(float, contract["minimum_tcn_gain"]) else simple
    selected_metrics = cast(dict[str, object], candidates[selected]["metrics"])
    movement_metrics = cast(dict[str, object], selected_metrics["movement"])
    macro_f1 = cast(float, movement_metrics["macro_f1"])
    recalls = cast(list[float], movement_metrics["per_class_recall"])
    transition_accuracy = cast(float, selected_metrics["transition_accuracy"])
    time_f1 = cast(float, time_only["macro_f1"])
    shuffle_metrics = cast(dict[str, object], shuffle["metrics"])
    shuffle_f1 = cast(float, cast(dict[str, object], shuffle_metrics["movement"])["macro_f1"])
    admitted = bool(
        macro_f1 >= cast(float, contract["minimum_movement_macro_f1"])
        and min(recalls[1:]) >= cast(float, contract["minimum_per_class_recall"])
        and transition_accuracy >= cast(float, contract["minimum_transition_accuracy"])
        and macro_f1 - time_f1 >= cast(float, contract["minimum_rgb_gain_over_time_only"])
        and macro_f1 - shuffle_f1 >= cast(float, contract["minimum_normal_gain_over_shuffle"])
    )
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        model_path = staging / "selected-movement-seed-0.safetensors"
        save_file(
            states[selected],
            model_path,
            metadata={
                "schema": MOVEMENT_POLICY_SCHEMA,
                "kind": selected,
                "contract_sha256": cast(str, contract["contract_sha256"]),
                "split_sha256": cast(str, split["split_sha256"]),
                "adapter_sha256": _sha(adapter_checkpoint),
            },
        )
        report: dict[str, object] = {
            "schema_version": MOVEMENT_POLICY_SCHEMA,
            "status": "PASSED" if admitted else "FAILED",
            "contract_sha256": contract["contract_sha256"],
            "split_sha256": split["split_sha256"],
            "adapter_sha256": _sha(adapter_checkpoint),
            "adapter_source_sha256": adapter_metadata.get("v5_source_model_sha256"),
            "combat_model_sha256": contract["combat_model_sha256"],
            "train_rows": len(train.windows),
            "dev_rows": len(dev.windows),
            "time_only": time_only,
            "candidates": candidates,
            "label_shuffle": shuffle,
            "tcn_gain_over_best_simple": tcn_gain,
            "selected": selected,
            "selected_model_sha256": _sha(model_path),
            "movement_admitted": admitted,
            "test_opened": False,
            "shadow_allowed": False,
            "control_output": False,
            "device_input_allowed": False,
        }
        report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report
