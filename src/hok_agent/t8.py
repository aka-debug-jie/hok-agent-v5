# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.mobile_testbed import (
    ABILITIES,
    AIMS,
    DEMONSTRATION_SOURCES,
    DEMONSTRATOR_DATA_SCHEMA,
    DEMONSTRATOR_SCHEMA,
    DEMONSTRATOR_SESSION_SCHEMA,
    DEMONSTRATOR_WINDOW_FRAMES,
    DIAGNOSTIC_INVERSE_SOURCE,
    EXECUTED_ACTION_SOURCE,
    KEYBOARD_V2_DATA_SCHEMA,
    KEYBOARD_V2_MIN_FORMAL_SAMPLES,
    KEYBOARD_V2_SCHEMA,
    KEYBOARD_V2_SESSION_SCHEMA,
    KEYBOARD_V21_DATA_SCHEMA,
    KEYBOARD_V21_SCHEMA,
    KEYBOARD_V21_SESSION_SCHEMA,
    MOVEMENTS,
    RGB_TEACHER_DATA_SCHEMA,
    RGB_TEACHER_EXECUTION_LAG_MS,
    RGB_TEACHER_HISTORY_FRAMES,
    RGB_TEACHER_MIN_FORMAL_SAMPLES,
    RGB_TEACHER_SCHEMA,
    RGB_TEACHER_SESSION_SCHEMA,
    RGB_TEACHER_SOURCE,
    RGB_TEACHER_WINDOW_FRAMES,
    SCRCPY_EXECUTED_ACTION_SOURCE,
    SCRCPY_SERVER_SHA256,
    SCRCPY_SERVER_VERSION,
    TARGETS,
    TERMINAL_DEMONSTRATION_SOURCE,
    TOUCH_WINDOW_FRAMES,
    load_layout,
)

MODEL_SCHEMA = "hok-agent-t8-factorized-bc-tcn-v3"
SPLIT_SCHEMA = "hok-agent-t8-split-v2"
TRAINING_SCHEMA = "hok-agent-t8-training-contract-v3"
EVALUATION_SCHEMA = "hok-agent-t8-offline-evaluation-v2"
V5_MODEL_SCHEMA = "hok-agent-v5-model-v1"
V5_SOURCE_ROLE = "v5_causal_source_teacher_source_v1"
V5_CURRENT_SOURCE_MODEL_SHA256 = "9e09650445ef3a4c7fdcf6f39e812375e7060c07f1998bb1e8730aec67e4fae4"
INVERSE_PROBE_SCHEMA = "hok-agent-t8-v2.1-inverse-probe-v1"
INVERSE_PROBE_ARTIFACT_SCHEMA = "hok-agent-t8-v2.1-inverse-probe-artifact-v1"
INVERSE_PROBE_MIN_SCORE_RATIO = 1.7
INVERSE_PROBE_THREE_CLASS_MIN_SCORE_RATIO = 1.25
VIDEO_THREE_CLASS_SCHEMA = "hok-agent-t8-video-combat-three-class-v1"
VIDEO_THREE_CLASS_TRAINING_SCHEMA = "hok-agent-t8-video-combat-three-class-training-v1"
VIDEO_RETROSPECTIVE_THREE_CLASS_SCHEMA = (
    "hok-agent-t8-video-combat-retrospective-three-class-v1"
)
VIDEO_RETROSPECTIVE_TRAINING_SCHEMA = (
    "hok-agent-t8-video-combat-retrospective-training-v1"
)
VIDEO_RETROSPECTIVE_ROI_SCHEMA = "hok-agent-t8-video-retrospective-roi-evaluation-v1"
RETROSPECTIVE_BASELINE_SCHEMA = "hok-agent-t8-retrospective-baseline-v1"
RETROSPECTIVE_EVENT_SCHEMA = "hok-agent-t8-retrospective-event-v1"
RETROSPECTIVE_SESSION_QC_SCHEMA = "hok-agent-t8-retrospective-session-qc-v1"
RETROSPECTIVE_BATCH_SCHEMA = "hok-agent-t8-retrospective-batch-v1"
RETROSPECTIVE_CALIBRATION_SCHEMA = "hok-agent-t8-retrospective-calibration-v2"
CAUSAL_VIDEO_DATASET_SCHEMA = "hok-agent-t8-video-causal-four-class-v1"
CAUSAL_VIDEO_TRAINING_SCHEMA = "hok-agent-t8-video-causal-four-class-pilot-v1"
CAUSAL_VIDEO_DIAGNOSTIC_SCHEMA = "hok-agent-t8-video-causal-learnability-diagnostic-v1"
CAUSAL_PIXEL_DATASET_SCHEMA = "hok-agent-t8-v2.2-causal-pixel-dataset-v1"
CAUSAL_PIXEL_PROBE_SCHEMA = "hok-agent-t8-v2.2-causal-pixel-probe-v1"
VISUAL_TEACHER_REPLAY_SCHEMA = "hok-agent-t8-v2.3-visual-teacher-replay-v1"
VISIBLE_ONSET_EVENT_SCHEMA = "hok-agent-t8-v2.4-visible-onset-event-v1"
VISIBLE_ONSET_AUDIT_SCHEMA = "hok-agent-t8-v2.4-visible-onset-audit-v1"
COMBAT_CAUSAL_DATASET_SCHEMA = "hok-agent-t8-v2.4-combat-causal-dataset-v1"
COMBAT_CAUSAL_PILOT_SCHEMA = "hok-agent-t8-v2.4-combat-causal-pilot-v1"
V25_SPLIT_SCHEMA = "hok-agent-t8-v2.5-rgb-conditioned-split-v1"
V25_PILOT_SCHEMA = "hok-agent-t8-v2.5-rgb-conditioned-pilot-v1"
V25_MODEL_SCHEMA = "hok-agent-t8-v2.5-rgb-conditioned-combat-model-v1"
V26_CONDITIONAL_PILOT_SCHEMA = "hok-agent-t8-v2.6-conditional-combat-pilot-v1"
V26_CONDITIONAL_MODEL_SCHEMA = "hok-agent-t8-v2.6-conditional-combat-model-v1"
V26_SELECTION_SCHEMA = "hok-agent-t8-v2.6-three-seed-selection-v1"
V26_EVALUATION_SCHEMA = "hok-agent-t8-v2.6-sealed-offline-evaluation-v1"
V27_CALIBRATION_SCHEMA = "hok-agent-t8-v2.7-current-scene-head-calibration-v1"
V27_MODEL_SCHEMA = "hok-agent-t8-v2.7-current-scene-head-model-v1"
V27_FROZEN = True
V26_GATE_DECISION_THRESHOLD = 0.65
V5_TARGET_MANIFEST_SCHEMA = "hok-agent-v5-manifest-v2"
SEEDS = (0, 1, 2)
HOLD_VALUES = (0, 150, 250)
MIN_SESSIONS = 6
STRICT_THRESHOLDS = {
    "joint_exact": 0.70,
    "movement": 0.85,
    "ability": 0.85,
    "aim": 0.80,
    "hold_ms": 0.80,
}
MACRO_RECALL_THRESHOLD = 0.75
SWITCH_RATE_ERROR_THRESHOLD = 0.10
CONFIDENCE_THRESHOLD = 0.80
OOD_QUANTILE = 0.995
ARRAY_KEYS = {
    "frames",
    "movement",
    "ability",
    "aim",
    "target",
    "hold_ms",
    "timestamp_ns",
    "input_sent",
}
CONFIG = {
    "frames": 8,
    "backbone": "resnet18-v5-source-initialized",
    "temporal": "causal-depthwise-tcn-2x3",
    "heads": [len(MOVEMENTS), len(ABILITIES), len(AIMS), len(TARGETS), len(HOLD_VALUES)],
    "target_intent": "not_implemented_dynamic_rgb_localization_required",
    "normalization": "uint8_to_float32_div255",
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
}
FRAME_COUNT = 8
HEAD_SIZES = (len(MOVEMENTS), len(ABILITIES), len(AIMS), len(TARGETS), len(HOLD_VALUES))
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class T8Error(ValueError):
    pass


@dataclass(frozen=True)
class Session:
    identity: str
    path: Path
    layout_sha256: str
    event_source: str
    frames: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class T8Data:
    sessions: tuple[Session, ...]
    splits: dict[str, tuple[int, ...]]
    split_sha256: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _large_root() -> Path:
    root_text = os.environ.get("HOK_LARGE_ROOT")
    if not root_text:
        raise T8Error("HOK_LARGE_ROOT is required for T8 artifacts")
    try:
        root = Path(root_text).resolve(strict=True)
    except OSError as exc:
        raise T8Error("HOK_LARGE_ROOT is unavailable") from exc
    if root.is_symlink() or not root.is_dir():
        raise T8Error("HOK_LARGE_ROOT must be a real directory")
    return root


def _large_existing(path: Path) -> Path:
    root = _large_root()
    target = path.resolve(strict=True)
    if target == root or root not in target.parents:
        raise T8Error("T8 input must be below HOK_LARGE_ROOT")
    return target


def _large_new(path: Path) -> Path:
    root = _large_root()
    target = path.resolve()
    if target == root or root not in target.parents or os.path.lexists(path):
        raise T8Error("T8 output must be a new path below HOK_LARGE_ROOT")
    path.parent.mkdir(parents=True, exist_ok=True)
    return target


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_frozen_json(path: Path, payload: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _factor_record(labels: np.ndarray) -> dict[str, object]:
    return {
        "movement": MOVEMENTS[int(labels[0])],
        "ability": ABILITIES[int(labels[1])],
        "aim": AIMS[int(labels[2])],
        "target": TARGETS[int(labels[3])],
        "hold_ms": HOLD_VALUES[int(labels[4])],
    }


def _load_session(path: Path) -> Session:
    summary_path, events_path, manifest_path = (
        path / "summary.json",
        path / "events.jsonl",
        path / "session-manifest.json",
    )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T8Error("demonstration session metadata is unreadable") from exc
    if (
        not isinstance(summary, dict)
        or summary.get("dataset_schema_version") != DEMONSTRATOR_DATA_SCHEMA
        or summary.get("schema_version") != DEMONSTRATOR_SCHEMA
        or summary.get("status") != "COMPLETED"
        or summary.get("capture_mode") != "scrcpy-v4l2"
        or summary.get("window_frames") != DEMONSTRATOR_WINDOW_FRAMES
        or not isinstance(summary.get("duration_seconds"), (int, float))
        or float(cast(float, summary["duration_seconds"])) < 300.0
    ):
        raise T8Error("session has the wrong demonstration schema")
    layout_sha256 = summary.get("layout_sha256")
    if not isinstance(layout_sha256, str) or len(layout_sha256) != 64:
        raise T8Error("session layout identity is invalid")
    event_source = summary.get("event_source", TERMINAL_DEMONSTRATION_SOURCE)
    if event_source not in DEMONSTRATION_SOURCES:
        raise T8Error("session demonstration source is invalid")
    if (
        not isinstance(events, list)
        or not events
        or any(not isinstance(row, dict) for row in events)
    ):
        raise T8Error("session has no valid execution events")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != DEMONSTRATOR_SESSION_SCHEMA
    ):
        raise T8Error("session manifest has the wrong schema")
    pieces: list[dict[str, np.ndarray]] = []
    for shard_path in sorted(path.glob("samples-*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            if set(shard.files) != ARRAY_KEYS:
                raise T8Error("demonstration shard fields are not exact")
            piece = {name: shard[name] for name in ARRAY_KEYS}
        count = len(piece["frames"])
        if piece["frames"].dtype != np.uint8 or piece["frames"].shape != (
            count,
            DEMONSTRATOR_WINDOW_FRAMES,
            128,
            128,
            3,
        ):
            raise T8Error("demonstration frame tensor is invalid")
        if any(piece[name].shape != (count,) for name in ARRAY_KEYS - {"frames"}):
            raise T8Error("demonstration label tensor is invalid")
        pieces.append(piece)
    if not pieces:
        raise T8Error("session has no derived RGB shards")
    frames = np.concatenate([piece["frames"] for piece in pieces])
    labels = np.stack(
        [
            np.concatenate([piece[name] for piece in pieces])
            for name in ("movement", "ability", "aim", "target", "hold_ms")
        ],
        axis=1,
    ).astype(np.int64)
    if len(frames) != len(events) or int(summary.get("samples", -1)) != len(frames):
        raise T8Error("events, summary, and shards disagree")
    limits = (len(MOVEMENTS), len(ABILITIES), len(AIMS), len(TARGETS))
    if any(
        np.any(labels[:, index] < 0) or np.any(labels[:, index] >= bound)
        for index, bound in enumerate(limits)
    ):
        raise T8Error("factorized action index is out of range")
    if np.any((labels[:, 0] != 0) & (labels[:, 1] != 0)):
        raise T8Error("demonstration contains an illegal movement and ability collision")
    try:
        labels[:, 4] = np.array(
            [HOLD_VALUES.index(int(value)) for value in labels[:, 4]], dtype=np.int64
        )
    except ValueError as exc:
        raise T8Error("hold duration is outside the frozen vocabulary") from exc
    sent = np.concatenate([piece["input_sent"] for piece in pieces])
    if sent.dtype != np.uint8 or not np.all(np.isin(sent, (0, 1))):
        raise T8Error("demonstration input result tensor is invalid")
    timestamps = np.concatenate([piece["timestamp_ns"] for piece in pieces])
    for index, event in enumerate(events):
        if (
            event.get("schema_version") != DEMONSTRATOR_SCHEMA
            or event.get("sequence") != index
            or event.get("timestamp_ns") != int(timestamps[index])
            or event.get("action") != _factor_record(labels[index])
            or event.get("input_sent") is not bool(sent[index])
            or event.get("source", TERMINAL_DEMONSTRATION_SOURCE) != event_source
        ):
            raise T8Error("execution events do not bind the derived samples")
    shard_entries = [
        {"name": item.name, "sha256": _sha(item)} for item in sorted(path.glob("samples-*.npz"))
    ]
    expected_manifest = {
        "schema_version": DEMONSTRATOR_SESSION_SCHEMA,
        "summary_sha256": _sha(summary_path),
        "events_sha256": _sha(events_path),
        "shards": shard_entries,
    }
    identity = hashlib.sha256(_canonical(expected_manifest)).hexdigest()
    if manifest != {**expected_manifest, "session_sha256": identity}:
        raise T8Error("session manifest does not bind the session files")
    return Session(identity, path, layout_sha256, cast(str, event_source), frames, labels)


def _sessions(root: Path) -> tuple[Session, ...]:
    expected_names = tuple(f"session-{index:03d}" for index in range(1, 9))
    paths = tuple(root / name for name in expected_names)
    if not root.is_dir() or any(not path.is_dir() for path in paths):
        raise T8Error("T8 v1 requires exactly eight named complete demonstration sessions")
    if any(
        path.is_dir()
        for path in root.iterdir()
        if path.name.startswith("session-") and path.name not in expected_names
    ):
        raise T8Error("T8 v1 demonstration session names are invalid")
    sessions = tuple(_load_session(path) for path in paths)
    if len({session.layout_sha256 for session in sessions}) != 1:
        raise T8Error("demonstration sessions must use one calibrated layout")
    return sessions


def _split_indices() -> dict[str, tuple[int, ...]]:
    return {"train": (0, 1, 2, 3), "dev": (4, 5), "test": (6, 7)}


def _validate_split_coverage(
    sessions: tuple[Session, ...], splits: dict[str, tuple[int, ...]]
) -> None:
    required = (len(MOVEMENTS), len(ABILITIES), len(AIMS), len(TARGETS), len(HOLD_VALUES))
    for indices in splits.values():
        labels = np.concatenate([sessions[index].labels for index in indices])
        if any(
            set(labels[:, index].tolist()) != set(range(size))
            for index, size in enumerate(required)
        ):
            raise T8Error("each frozen split must cover the factor vocabulary")


def freeze_t8_split(*, dataset_root: Path, output_path: Path) -> dict[str, object]:
    root = _large_existing(dataset_root)
    output_path = _large_new(output_path)
    sessions, splits = _sessions(root), _split_indices()
    _validate_split_coverage(sessions, splits)
    payload: dict[str, object] = {
        "schema_version": SPLIT_SCHEMA,
        "layout_sha256": sessions[0].layout_sha256,
        "splits": {
            name: [
                {"name": sessions[index].path.name, "session_sha256": sessions[index].identity}
                for index in indices
            ]
            for name, indices in splits.items()
        },
    }
    payload["split_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    output_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def load_t8_data(root: Path, split_path: Path | None = None) -> T8Data:
    root = _large_existing(root)
    sessions, splits = _sessions(root), _split_indices()
    manifest_path = _large_existing(split_path or root / "t8-split-v2.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T8Error("T8 frozen split manifest is unreadable") from exc
    expected_payload: dict[str, object] = {
        "schema_version": SPLIT_SCHEMA,
        "layout_sha256": sessions[0].layout_sha256,
        "splits": {
            name: [
                {"name": sessions[index].path.name, "session_sha256": sessions[index].identity}
                for index in indices
            ]
            for name, indices in splits.items()
        },
    }
    expected_payload["split_sha256"] = hashlib.sha256(_canonical(expected_payload)).hexdigest()
    if manifest != expected_payload:
        raise T8Error("T8 frozen split manifest does not bind the sessions")
    return T8Data(sessions, splits, cast(str, expected_payload["split_sha256"]))


@dataclass(frozen=True)
class V5Initialization:
    encoder_state: dict[str, torch.Tensor]
    model_sha256: str
    source_manifest_sha256: str
    source_baseline_sha256: str


class FactorizedTemporalActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = resnet18(weights=None)
        base.fc = nn.Identity()
        self.encoder = base
        self.temporal = nn.Sequential(
            nn.Conv1d(512, 512, 3, padding=2, groups=512),
            nn.Tanh(),
            nn.Conv1d(512, 512, 3, padding=4, dilation=2, groups=512),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList(nn.Linear(512, size) for size in HEAD_SIZES)
        self.register_buffer("feature_center", torch.zeros(512))
        self.register_buffer("feature_radius", torch.ones(()))
        self.register_buffer("feature_ood_threshold", torch.ones(()))

    def features(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or tuple(frames.shape[2:]) != (3, 128, 128):
            raise T8Error("actor requires BxTx3x128x128 RGB only")
        batch, steps = frames.shape[:2]
        encoded = cast(torch.Tensor, self.encoder(frames.reshape(batch * steps, 3, 128, 128)))
        temporal = cast(
            torch.Tensor, self.temporal(encoded.reshape(batch, steps, 512).transpose(1, 2))
        )
        return temporal[..., :steps][..., -1]

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.features(frames)
        return tuple(head(features) for head in self.heads)


def _examples(
    data: T8Data, indices: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    frames, labels, lengths = [], [], []
    for index in indices:
        session = data.sessions[index]
        frames.append(session.frames)
        labels.append(session.labels)
        lengths.append(len(session.frames))
    return np.concatenate(frames), np.concatenate(labels), tuple(lengths)


def _normal_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frames).to(device).permute(0, 1, 4, 2, 3).float().div(255.0)


def _loss(outputs: tuple[torch.Tensor, ...], labels: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            nn.functional.cross_entropy(output, labels[:, index])
            for index, output in enumerate(outputs)
        ]
    ).sum()


def _predict_arrays(
    model: FactorizedTemporalActor, frames: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, confidences, distances = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            features = model.features(_normal_tensor(frames[start : start + batch_size], device))
            probabilities = [head(features).softmax(1) for head in model.heads]
            labels.append(torch.stack([item.argmax(1) for item in probabilities], 1).cpu().numpy())
            confidences.append(
                torch.stack([item.max(1).values for item in probabilities], 1)
                .min(1)
                .values.cpu()
                .numpy()
            )
            distances.append(
                (
                    (features - model.feature_center).norm(dim=1)
                    / model.feature_radius.clamp_min(1e-6)
                )
                .cpu()
                .numpy()
            )
    return (
        np.concatenate(labels).astype(np.int64),
        np.concatenate(confidences).astype(np.float32),
        np.concatenate(distances).astype(np.float32),
    )


def _score(
    model: FactorizedTemporalActor,
    frames: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    predicted, confidence, ood = _predict_arrays(model, frames, device, batch_size)
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            stop = min(start + batch_size, len(frames))
            output = model(_normal_tensor(frames[start:stop], device))
            loss_sum += float(
                _loss(output, torch.from_numpy(labels[start:stop]).to(device)).item()
            ) * (stop - start)
    heads = ("movement", "ability", "aim", "target", "hold_ms")
    scores = {
        name: float((predicted[:, index] == labels[:, index]).mean())
        for index, name in enumerate(heads)
    }
    return (
        {
            "loss": loss_sum / len(frames),
            "joint_exact": float((predicted == labels).all(1).mean()),
            **scores,
        },
        predicted,
        confidence,
        ood,
    )


def _freeze_feature_distribution(
    model: FactorizedTemporalActor, frames: np.ndarray, device: torch.device, batch_size: int
) -> None:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            chunks.append(
                model.features(_normal_tensor(frames[start : start + batch_size], device))
                .cpu()
                .numpy()
            )
    values = np.concatenate(chunks)
    center = values.mean(0)
    distances = np.linalg.norm(values - center, axis=1)
    radius = max(float(distances.mean()), 1e-6)
    model.feature_center.copy_(torch.from_numpy(center).to(device))
    model.feature_radius.fill_(radius)
    model.feature_ood_threshold.fill_(float(np.quantile(distances / radius, OOD_QUANTILE)))


def _read_object(path: Path, message: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T8Error(message) from exc
    if not isinstance(value, dict):
        raise T8Error(message)
    return cast(dict[str, object], value)


def _load_v5_initialization(source_dir: Path) -> V5Initialization:
    source_dir = _large_existing(source_dir)
    source, baseline = (
        _read_object(source_dir / "source.json", "V5 source manifest is unreadable"),
        _read_object(source_dir / "source-baseline.json", "V5 source baseline is unreadable"),
    )
    model_name, expected_sha = (
        source.get("selected_model_path"),
        source.get("selected_model_sha256"),
    )
    if (
        source.get("schema_version") != "hok-agent-v5-source-producer-v1"
        or not isinstance(model_name, str)
        or Path(model_name).name != model_name
        or not isinstance(expected_sha, str)
        or expected_sha != V5_CURRENT_SOURCE_MODEL_SHA256
        or baseline.get("selected_model_sha256") != expected_sha
    ):
        raise T8Error("V5 source selection is not a current-contract selected model")
    model_path = source_dir / model_name
    if not model_path.is_file() or _sha(model_path) != expected_sha:
        raise T8Error("V5 source model hash is invalid")
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        state = load_file(model_path, device="cpu")
    except (OSError, SafetensorError) as exc:
        raise T8Error("V5 source model is unreadable") from exc
    if (
        metadata is None
        or metadata.get("schema_version") != V5_MODEL_SCHEMA
        or metadata.get("role") != V5_SOURCE_ROLE
        or metadata.get("source_split_sha256") != source.get("source_split_sha256")
        or metadata.get("source_dataset_sha256") != source.get("dataset_sha256")
        or metadata.get("teacher_sha256") != source.get("teacher_sha256")
    ):
        raise T8Error("V5 source model metadata does not bind its source manifest")
    encoder = {key: value for key, value in state.items() if not key.startswith("fc.")}
    expected = FactorizedTemporalActor().encoder.state_dict()
    if set(encoder) != set(expected) or any(
        encoder[key].shape != expected[key].shape for key in encoder
    ):
        raise T8Error("V5 source encoder is not an exact T8 ResNet18 initialization")
    return V5Initialization(
        encoder,
        expected_sha,
        _sha(source_dir / "source.json"),
        _sha(source_dir / "source-baseline.json"),
    )


def _training_contract(
    data: T8Data, source: V5Initialization, epochs: int, batch_size: int
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": TRAINING_SCHEMA,
        "config_sha256": CONFIG_HASH,
        "layout_sha256": data.sessions[0].layout_sha256,
        "split_sha256": data.split_sha256,
        "v5_source_model_sha256": source.model_sha256,
        "v5_source_manifest_sha256": source.source_manifest_sha256,
        "v5_source_baseline_sha256": source.source_baseline_sha256,
        "seeds": list(SEEDS),
        "selection_metric": "dev_total_cross_entropy",
        "epochs": epochs,
        "batch_size": batch_size,
        "demonstration_sources": sorted({session.event_source for session in data.sessions}),
    }
    payload["training_contract_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _model_metadata(path: Path) -> dict[str, str]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except (OSError, SafetensorError) as exc:
        raise T8Error("T8 model is unreadable") from exc
    if metadata is None:
        raise T8Error("T8 model metadata is missing")
    return dict(metadata)


def _open_t8_model(
    path: Path, layout_sha256: str, device_name: str
) -> tuple[FactorizedTemporalActor, dict[str, str], torch.device]:
    if device_name not in {"cpu", "cuda"} or (
        device_name == "cuda" and not torch.cuda.is_available()
    ):
        raise T8Error("T8 predictor device is unavailable")
    metadata = _model_metadata(path)
    if (
        metadata.get("schema") != MODEL_SCHEMA
        or metadata.get("config_sha256") != CONFIG_HASH
        or metadata.get("layout_sha256") != layout_sha256
    ):
        raise T8Error("T8 model metadata does not bind this layout")
    device = torch.device(device_name)
    model = FactorizedTemporalActor().to(device)
    try:
        model.load_state_dict(load_file(path, device=device_name), strict=True)
    except (RuntimeError, SafetensorError) as exc:
        raise T8Error("T8 model tensors are invalid") from exc
    return model.eval(), metadata, device


def open_t8_predictor(
    path: Path, layout_sha256: str, device_name: str
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    model, _metadata, device = _open_t8_model(path, layout_sha256, device_name)

    def predict(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _predict_arrays(model, frames, device, max(1, len(frames)))

    return predict


def train_t8_bc(
    *,
    dataset_root: Path,
    output_dir: Path,
    v5_source_dir: Path,
    device: str,
    epochs: int = 20,
    batch_size: int = 32,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or epochs < 1 or batch_size < 1:
        raise T8Error("invalid training output, device, or hyperparameters")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    dataset_root, output_dir = _large_existing(dataset_root), _large_new(output_dir)
    data, source, target = (
        load_t8_data(dataset_root),
        _load_v5_initialization(v5_source_dir),
        torch.device(device),
    )
    train_x, train_y, _ = _examples(data, data.splits["train"])
    dev_x, dev_y, _ = _examples(data, data.splits["dev"])
    contract = _training_contract(data, source, epochs, batch_size)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent)))
    results: list[dict[str, object]] = []
    try:
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = FactorizedTemporalActor().to(target)
            model.encoder.load_state_dict(source.encoder_state, strict=True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            best_loss, best_epoch, best_state = math.inf, 0, {}
            for epoch in range(1, epochs + 1):
                model.train()
                order = np.random.permutation(len(train_x))
                for start in range(0, len(order), batch_size):
                    select = order[start : start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = _loss(
                        model(_normal_tensor(train_x[select], target)),
                        torch.from_numpy(train_y[select]).to(target),
                    )
                    loss.backward()  # type: ignore[no-untyped-call]
                    optimizer.step()
                validation = _score(model, dev_x, dev_y, target, batch_size)[0]
                if float(validation["loss"]) < best_loss - 1e-12:
                    best_loss, best_epoch = float(validation["loss"]), epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            if not best_state:
                raise T8Error("T8 validation was not computed")
            model.load_state_dict(best_state, strict=True)
            _freeze_feature_distribution(model, train_x, target, batch_size)
            validation = _score(model, dev_x, dev_y, target, batch_size)[0]
            name = f"model-seed-{seed}.safetensors"
            save_file(
                model.cpu().state_dict(),
                staging / name,
                metadata={
                    "schema": MODEL_SCHEMA,
                    "config_sha256": CONFIG_HASH,
                    "layout_sha256": data.sessions[0].layout_sha256,
                    "seed": str(seed),
                    "training_contract_sha256": cast(str, contract["training_contract_sha256"]),
                    "v5_source_model_sha256": source.model_sha256,
                    "ood_threshold": str(float(model.feature_ood_threshold.item())),
                },
            )
            results.append(
                {
                    "seed": seed,
                    "model": name,
                    "sha256": _sha(staging / name),
                    "best_epoch": best_epoch,
                    "validation": validation,
                }
            )
        selected = min(
            results,
            key=lambda item: (
                float(cast(dict[str, float], item["validation"])["loss"]),
                cast(int, item["seed"]),
            ),
        )
        report: dict[str, object] = {
            "schema_version": TRAINING_SCHEMA,
            "status": "TRAINED_PENDING_OFFLINE_EVALUATION",
            "claim_scope": "authorized_test_app_factorized_behavior_cloning",
            "config_sha256": CONFIG_HASH,
            "layout_sha256": data.sessions[0].layout_sha256,
            "split_sha256": data.split_sha256,
            "training_contract_sha256": contract["training_contract_sha256"],
            "v5_initialization_scope": "source_only_non_promoting",
            "v5_source_model_sha256": source.model_sha256,
            "seeds": results,
            "selected_seed": selected["seed"],
            "selected_model": selected["model"],
            "selected_model_sha256": selected["sha256"],
            "selection_metric": "dev_total_cross_entropy",
            "test_accessed_for_selection": False,
            "raw_video_or_paths_persisted": False,
            "mobile_connected": False,
            "autonomous_execution_claim": False,
            "target_intent_status": "NOT_IMPLEMENTED",
            "demonstration_sources": contract["demonstration_sources"],
        }
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
        return report
    except Exception:
        for item in staging.iterdir():
            item.unlink()
        staging.rmdir()
        raise


def _head_metrics(predicted: np.ndarray, target: np.ndarray, size: int) -> dict[str, object]:
    confusion = np.zeros((size, size), dtype=np.int64)
    for actual, candidate in zip(target, predicted, strict=True):
        confusion[int(actual), int(candidate)] += 1
    diagonal = np.diag(confusion).astype(np.float64)
    support = confusion.sum(1).astype(np.float64)
    predicted_count = confusion.sum(0).astype(np.float64)
    recall = np.divide(diagonal, support, out=np.zeros(size), where=support > 0)
    precision = np.divide(diagonal, predicted_count, out=np.zeros(size), where=predicted_count > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(size),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(diagonal.sum() / support.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_recall": recall.tolist(),
        "confusion": confusion.tolist(),
    }


def _switch_rate(values: np.ndarray, lengths: tuple[int, ...]) -> float:
    changes, opportunities, start = 0, 0, 0
    for length in lengths:
        stop = start + length
        if length > 1:
            changes += int(
                np.any(values[start + 1 : stop] != values[start : stop - 1], axis=1).sum()
            )
            opportunities += length - 1
        start = stop
    return 0.0 if opportunities == 0 else changes / opportunities


def _offline_metrics(
    predicted: np.ndarray, target: np.ndarray, lengths: tuple[int, ...]
) -> dict[str, object]:
    heads = ("movement", "ability", "aim", "target", "hold_ms")
    details = {
        name: _head_metrics(predicted[:, index], target[:, index], HEAD_SIZES[index])
        for index, name in enumerate(heads)
    }
    true_switch, predicted_switch = _switch_rate(target, lengths), _switch_rate(predicted, lengths)
    return {
        "joint_exact": float((predicted == target).all(1).mean()),
        "heads": details,
        "true_switch_rate": true_switch,
        "predicted_switch_rate": predicted_switch,
        "switch_rate_error": abs(predicted_switch - true_switch),
        "illegal_joint_predictions": int(((predicted[:, 0] != 0) & (predicted[:, 1] != 0)).sum()),
    }


def _performance_pass(metrics: Mapping[str, object]) -> bool:
    heads = cast(dict[str, object], metrics["heads"])
    return (
        float(cast(float, metrics["joint_exact"])) >= STRICT_THRESHOLDS["joint_exact"]
        and all(
            float(cast(dict[str, float], heads[name])["accuracy"]) >= STRICT_THRESHOLDS[name]
            and float(cast(dict[str, float], heads[name])["macro_recall"]) >= MACRO_RECALL_THRESHOLD
            for name in STRICT_THRESHOLDS
            if name != "joint_exact"
        )
        and cast(int, metrics["illegal_joint_predictions"]) == 0
    )


def evaluate_t8_offline(
    *,
    dataset_root: Path,
    model_path: Path,
    training_report: Path,
    output_path: Path,
    device: str,
    batch_size: int = 32,
) -> dict[str, object]:
    if batch_size < 1:
        raise T8Error("invalid evaluation batch size")
    dataset_root, model_path, training_report, output_path = (
        _large_existing(dataset_root),
        _large_existing(model_path),
        _large_existing(training_report),
        _large_new(output_path),
    )
    data, training = (
        load_t8_data(dataset_root),
        _read_object(training_report, "T8 training report is unreadable"),
    )
    layout = data.sessions[0].layout_sha256
    if (
        training.get("schema_version") != TRAINING_SCHEMA
        or training.get("status") != "TRAINED_PENDING_OFFLINE_EVALUATION"
        or training.get("layout_sha256") != layout
        or training.get("split_sha256") != data.split_sha256
        or training.get("selected_model_sha256") != _sha(model_path)
    ):
        raise T8Error("T8 training report does not admit this sealed evaluation")
    model, metadata, target = _open_t8_model(model_path, layout, device)
    if metadata.get("training_contract_sha256") != training.get(
        "training_contract_sha256"
    ) or metadata.get("v5_source_model_sha256") != training.get("v5_source_model_sha256"):
        raise T8Error("T8 model does not bind the selected training contract")
    test_x, test_y, lengths = _examples(data, data.splits["test"])
    _scores, predicted, _confidence, ood = _score(model, test_x, test_y, target, batch_size)
    metrics = _offline_metrics(predicted, test_y, lengths)
    shifted = np.concatenate(
        [
            np.roll(test_y[sum(lengths[:index]) : sum(lengths[: index + 1])], 1, axis=0)
            for index in range(len(lengths))
        ]
    )
    shifted_metrics = _offline_metrics(predicted, shifted, lengths)
    controls = np.stack(
        (
            np.zeros((FRAME_COUNT, 128, 128, 3), dtype=np.uint8),
            np.full((FRAME_COUNT, 128, 128, 3), 128, dtype=np.uint8),
        )
    )
    _labels, _confidence, control_ood = _predict_arrays(model, controls, target, 2)
    threshold = float(model.feature_ood_threshold.item())
    negative = {
        "shifted_labels_passed": _performance_pass(shifted_metrics),
        "black_and_gray_ood_rejected": bool(np.all(control_ood > threshold)),
        "test_ood_inlier_rate": float((ood <= threshold).mean()),
    }
    strict = (
        _performance_pass(metrics)
        and float(cast(float, metrics["switch_rate_error"])) <= SWITCH_RATE_ERROR_THRESHOLD
        and not cast(bool, negative["shifted_labels_passed"])
        and cast(bool, negative["black_and_gray_ood_rejected"])
    )
    report: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "PASSED" if strict else "FAILED",
        "strict_passed": strict,
        "model_sha256": _sha(model_path),
        "layout_sha256": layout,
        "split_sha256": data.split_sha256,
        "training_contract_sha256": training["training_contract_sha256"],
        "v5_source_model_sha256": training["v5_source_model_sha256"],
        "thresholds": {
            "accuracy": STRICT_THRESHOLDS,
            "macro_recall": MACRO_RECALL_THRESHOLD,
            "switch_rate_error": SWITCH_RATE_ERROR_THRESHOLD,
            "ood_quantile": OOD_QUANTILE,
        },
        "metrics": metrics,
        "shifted_label_control": shifted_metrics,
        "negative_controls": negative,
        "raw_frames_or_paths_persisted": False,
        "mobile_connected": False,
        "target_intent_status": "NOT_IMPLEMENTED",
    }
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def _inverse_probe_session(
    session_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    session = _large_existing(session_dir)
    summary_path = session / "summary.json"
    events_path = session / "events.jsonl"
    contract_path = session / "action-contract.json"
    manifest_path = session / "session-manifest.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
        ]
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T8Error("T8-v2.1 inverse probe metadata is unreadable") from exc
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != KEYBOARD_V21_SCHEMA
        or summary.get("dataset_schema_version") != KEYBOARD_V21_DATA_SCHEMA
        or summary.get("status") != "COMPLETED"
        or summary.get("event_source") != DIAGNOSTIC_INVERSE_SOURCE
        or summary.get("diagnostic_inverse_probe") is not True
        or summary.get("training_eligible") is not False
        or not isinstance(contract, dict)
        or contract.get("source") != DIAGNOSTIC_INVERSE_SOURCE
    ):
        raise T8Error("T8-v2.1 inverse probe session identity is invalid")
    shards = sorted(session.glob("samples-*.npz"))
    expected = {
        "schema_version": KEYBOARD_V21_SESSION_SCHEMA,
        "summary_sha256": _sha(summary_path),
        "events_sha256": _sha(events_path),
        "action_contract_file_sha256": _sha(contract_path),
        "shards": [{"name": path.name, "sha256": _sha(path)} for path in shards],
    }
    identity = hashlib.sha256(_canonical(expected)).hexdigest()
    if manifest != {**expected, "session_sha256": identity} or not shards:
        raise T8Error("T8-v2.1 inverse probe manifest is invalid")
    timestamps: list[np.ndarray] = []
    last_frames: list[np.ndarray] = []
    for path in shards:
        with np.load(path, allow_pickle=False) as shard:
            if "frames" not in shard.files or "timestamp_ns" not in shard.files:
                raise T8Error("T8-v2.1 inverse probe shard fields are invalid")
            frames = shard["frames"]
            times = shard["timestamp_ns"]
            if (
                frames.dtype != np.uint8
                or frames.ndim != 5
                or frames.shape[1:] != (TOUCH_WINDOW_FRAMES, 128, 128, 3)
                or times.dtype != np.int64
                or times.shape != (len(frames),)
            ):
                raise T8Error("T8-v2.1 inverse probe tensors are invalid")
            last_frames.append(frames[:, -1].copy())
            timestamps.append(times.copy())
    all_times = np.concatenate(timestamps)
    all_frames = np.concatenate(last_frames)
    if len(all_times) < 20 or np.any(np.diff(all_times) <= 0):
        raise T8Error("T8-v2.1 inverse probe timestamps are invalid")
    return summary, cast(list[dict[str, object]], events), all_times, all_frames


def _button_change_scores(
    before: np.ndarray,
    after: np.ndarray,
    buttons: tuple[tuple[float, float], ...],
) -> np.ndarray:
    scores: list[float] = []
    radius = 6
    for x_fraction, y_fraction in buttons:
        x, y = round(127 * x_fraction), round(127 * y_fraction)
        x0, x1 = max(0, x - radius), min(128, x + radius + 1)
        y0, y1 = max(0, y - radius), min(128, y + radius + 1)
        difference = np.abs(
            after[:, y0:y1, x0:x1].astype(np.int16)
            - before[y0:y1, x0:x1].astype(np.int16)
        )
        scores.append(float(difference.mean(axis=(1, 2, 3)).max()))
    return np.asarray(scores, dtype=np.float32)


def materialize_t8_v21_inverse_probe(
    *, session_dir: Path, layout_path: Path, output_dir: Path
) -> dict[str, object]:
    output = _large_new(output_dir)
    summary, events, timestamps, frames = _inverse_probe_session(session_dir)
    layout, layout_sha256 = load_layout(layout_path)
    if summary.get("layout_sha256") != layout_sha256:
        raise T8Error("inverse probe layout does not match the captured session")
    button_names = tuple(ABILITIES[1:])
    button_points = tuple(cast(tuple[float, float], layout.buttons[name]) for name in button_names)
    before_rows: list[np.ndarray] = []
    after_rows: list[np.ndarray] = []
    labels: list[int] = []
    dispatches: list[int] = []
    score_rows: list[np.ndarray] = []
    for event in events:
        action = event.get("action")
        if (
            event.get("event_type") != "semantic_transition"
            or event.get("source") != DIAGNOSTIC_INVERSE_SOURCE
            or not isinstance(action, dict)
            or action.get("ability") not in button_names
            or not isinstance(event.get("dispatch_completed_ns"), int)
        ):
            continue
        dispatch = cast(int, event["dispatch_completed_ns"])
        before_index = int(np.searchsorted(timestamps, dispatch, side="left"))
        if before_index >= len(timestamps) or timestamps[before_index] - dispatch >= 100_000_000:
            continue
        after_indices = [
            int(np.searchsorted(timestamps, dispatch + lag_ms * 1_000_000, side="left"))
            for lag_ms in (200, 300, 500)
        ]
        if any(index >= len(timestamps) for index in after_indices):
            continue
        before = frames[before_index]
        after = frames[np.asarray(after_indices)]
        before_rows.append(before)
        after_rows.append(after)
        label = button_names.index(cast(str, action["ability"]))
        labels.append(label)
        dispatches.append(dispatch)
        score_rows.append(_button_change_scores(before, after, button_points))
    if not labels:
        raise T8Error("inverse probe contains no complete before/action/after rows")
    label_array = np.asarray(labels, dtype=np.int8)
    score_array = np.stack(score_rows)
    predictions = score_array.argmax(axis=1).astype(np.int8)
    ordered_scores = np.sort(score_array, axis=1)
    score_ratios = ordered_scores[:, -1] / np.maximum(ordered_scores[:, -2], 1e-6)
    accepted = score_ratios >= INVERSE_PROBE_MIN_SCORE_RATIO
    recalls = {
        name: (
            float(np.mean(predictions[label_array == index] == index))
            if np.any(label_array == index)
            else 0.0
        )
        for index, name in enumerate(button_names)
    }
    counts = {name: int(np.sum(label_array == index)) for index, name in enumerate(button_names)}
    accuracy = float(np.mean(predictions == label_array))
    shuffled_accuracy = float(np.mean(predictions == np.roll(label_array, 1)))
    accepted_counts = {
        name: int(np.sum(accepted & (label_array == index)))
        for index, name in enumerate(button_names)
    }
    accepted_precision = (
        float(np.mean(predictions[accepted] == label_array[accepted]))
        if np.any(accepted)
        else 0.0
    )
    accepted_coverage = float(np.mean(accepted))
    three_class_accepted = (predictions != button_names.index("skill3")) & (
        score_ratios >= INVERSE_PROBE_THREE_CLASS_MIN_SCORE_RATIO
    )
    three_class_precision = (
        float(
            np.mean(predictions[three_class_accepted] == label_array[three_class_accepted])
        )
        if np.any(three_class_accepted)
        else 0.0
    )
    three_class_coverage = float(np.mean(three_class_accepted))
    three_class_counts = {
        name: int(np.sum(three_class_accepted & (predictions == index)))
        for index, name in enumerate(button_names[:3])
    }
    three_class_shuffled_precision = (
        float(
            np.mean(
                predictions[three_class_accepted]
                == np.roll(label_array, 1)[three_class_accepted]
            )
        )
        if np.any(three_class_accepted)
        else 0.0
    )
    three_class_gate = bool(
        min(three_class_counts.values()) >= 5
        and three_class_precision >= 0.95
        and three_class_coverage >= 0.50
        and three_class_precision - three_class_shuffled_precision >= 0.50
    )
    gate = bool(
        min(counts.values()) >= 5
        and accuracy >= 0.70
        and min(recalls.values()) >= 0.50
        and accuracy - shuffled_accuracy >= 0.25
        and min(accepted_counts.values()) >= 3
        and accepted_precision >= 0.95
        and accepted_coverage >= 0.50
    )
    report: dict[str, object] = {
        "schema_version": INVERSE_PROBE_SCHEMA,
        "status": (
            "PROBE_PASSED"
            if gate
            else "THREE_CLASS_PROBE_PASSED"
            if three_class_gate
            else "PROBE_DIAGNOSIS_REQUIRED"
        ),
        "source_session_sha256": json.loads(
            (session_dir / "session-manifest.json").read_text(encoding="utf-8")
        )["session_sha256"],
        "layout_sha256": layout_sha256,
        "rows": len(labels),
        "class_counts": counts,
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(tuple(recalls.values()))),
        "per_class_recall": recalls,
        "shuffled_accuracy": shuffled_accuracy,
        "normal_minus_shuffled_accuracy": accuracy - shuffled_accuracy,
        "abstention_score_ratio": INVERSE_PROBE_MIN_SCORE_RATIO,
        "accepted_rows": int(np.sum(accepted)),
        "accepted_class_counts": accepted_counts,
        "accepted_precision": accepted_precision,
        "accepted_coverage": accepted_coverage,
        "three_class_scope": list(button_names[:3]),
        "three_class_abstained": ["skill3"],
        "three_class_score_ratio": INVERSE_PROBE_THREE_CLASS_MIN_SCORE_RATIO,
        "three_class_accepted_rows": int(np.sum(three_class_accepted)),
        "three_class_accepted_counts": three_class_counts,
        "three_class_precision": three_class_precision,
        "three_class_coverage": three_class_coverage,
        "three_class_shuffled_precision": three_class_shuffled_precision,
        "three_class_gate_passed": three_class_gate,
        "gate_passed": gate,
        "video_pseudo_label_probe_allowed": gate or three_class_gate,
        "policy_training_allowed": False,
        "formal_demonstration": False,
        "raw_video_persisted": False,
    }
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        artifact = staging / "inverse-probe.npz"
        np.savez_compressed(
            artifact,
            before_rgb=np.stack(before_rows).astype(np.uint8, copy=False),
            after_rgb=np.stack(after_rows).astype(np.uint8, copy=False),
            combat=label_array,
            dispatch_completed_ns=np.asarray(dispatches, dtype=np.int64),
            roi_change_scores=score_array,
            predicted_combat=predictions,
            score_ratio=score_ratios.astype(np.float32),
            accepted=accepted.astype(np.uint8),
            three_class_accepted=three_class_accepted.astype(np.uint8),
        )
        report["artifact_schema_version"] = INVERSE_PROBE_ARTIFACT_SCHEMA
        report["artifact_sha256"] = _sha(artifact)
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": INVERSE_PROBE_ARTIFACT_SCHEMA,
            "artifact_sha256": _sha(artifact),
            "report_sha256": _sha(staging / "report.json"),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def materialize_t8_video_three_class(
    *,
    source_dir: Path,
    inverse_report_path: Path,
    output_dir: Path,
    retrospective: bool = False,
) -> dict[str, object]:
    source = _large_existing(source_dir)
    inverse_path = _large_existing(inverse_report_path)
    output = _large_new(output_dir)
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        inverse = json.loads(inverse_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T8Error("three-class video source metadata is unreadable") from exc
    source_schema = (
        "hok-agent-t8-video-combat-pseudolabel-candidates-v1"
        if retrospective
        else "hok-agent-t8-video-combat-pseudolabel-candidates-v2"
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != source_schema
        or manifest.get("combat_vocabulary") != list(ABILITIES[1:])
        or manifest.get("future_frames_included") is not False
        or (
            manifest.get("causal_window_includes_action_frame") is not True
            if retrospective
            else manifest.get("event_frame_included") is not False
        )
        or not isinstance(manifest.get("shards"), list)
        or not isinstance(inverse, dict)
        or inverse.get("schema_version") != INVERSE_PROBE_SCHEMA
        or inverse.get("three_class_gate_passed") is not True
        or inverse.get("three_class_scope") != list(ABILITIES[1:4])
        or inverse.get("three_class_abstained") != ["skill3"]
    ):
        raise T8Error("three-class video source contract is invalid")
    rows = cast(list[dict[str, object]], manifest["shards"])
    counts = {split: {name: 0 for name in ABILITIES[1:4]} for split in ("train", "dev")}
    output_rows: list[dict[str, object]] = []
    total_rows = 0
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shards_dir = staging / "shards"
        shards_dir.mkdir()
        for row in rows:
            name, split, digest = row.get("path"), row.get("split"), row.get("sha256")
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"(?:train|dev)-[0-9]{4}\.npz", name)
                or split not in {"train", "dev"}
                or not isinstance(digest, str)
            ):
                raise T8Error("three-class video shard manifest is invalid")
            source_shard = source / "shards" / name
            if _sha(source_shard) != digest:
                raise T8Error("three-class video source shard hash differs")
            with np.load(source_shard, allow_pickle=False) as shard:
                expected_fields = (
                    {
                        "frames",
                        "combat_id",
                        "timestamp_ms",
                        "session_hash",
                        "causal_window_sha256",
                    }
                    if retrospective
                    else {
                        "frames",
                        "combat_id",
                        "action_timestamp_ms",
                        "observation_end_timestamp_ms",
                        "session_hash",
                        "causal_window_sha256",
                    }
                )
                if set(shard.files) != expected_fields:
                    raise T8Error("three-class video source shard fields are invalid")
                arrays = {key: shard[key] for key in shard.files}
            combat = arrays.get("combat_id")
            if not isinstance(combat, np.ndarray) or combat.ndim != 1:
                raise T8Error("three-class video source labels are invalid")
            keep = combat < 3
            if not np.any(keep):
                continue
            filtered = {
                key: value[keep]
                for key, value in arrays.items()
                if isinstance(value, np.ndarray) and len(value) == len(combat)
            }
            target = shards_dir / name
            if retrospective:
                np.savez_compressed(
                    target,
                    frames=filtered["frames"],
                    combat_id=filtered["combat_id"],
                    timestamp_ms=filtered["timestamp_ms"],
                    session_hash=filtered["session_hash"],
                    causal_window_sha256=filtered["causal_window_sha256"],
                )
            else:
                np.savez_compressed(
                    target,
                    frames=filtered["frames"],
                    combat_id=filtered["combat_id"],
                    action_timestamp_ms=filtered["action_timestamp_ms"],
                    observation_end_timestamp_ms=filtered["observation_end_timestamp_ms"],
                    session_hash=filtered["session_hash"],
                    causal_window_sha256=filtered["causal_window_sha256"],
                )
            class_counts = {
                class_name: int(np.sum(filtered["combat_id"] == index))
                for index, class_name in enumerate(ABILITIES[1:4])
            }
            for class_name, count in class_counts.items():
                counts[split][class_name] += count
            total_rows += int(len(filtered["combat_id"]))
            output_rows.append(
                {
                    "path": name,
                    "split": split,
                    "row_count": int(len(filtered["combat_id"])),
                    "class_counts": class_counts,
                    "sha256": _sha(target),
                }
            )
        output_manifest: dict[str, object] = {
            "schema_version": (
                VIDEO_RETROSPECTIVE_THREE_CLASS_SCHEMA
                if retrospective
                else VIDEO_THREE_CLASS_SCHEMA
            ),
            "task": (
                "retrospective_action_recognition"
                if retrospective
                else "strict_causal_next_action_diagnostic"
            ),
            "source_manifest_sha256": _sha(source / "manifest.json"),
            "inverse_probe_report_sha256": _sha(inverse_path),
            "combat_vocabulary": list(ABILITIES[1:4]),
            "abstained_classes": ["skill3"],
            "minimum_source_score_ratio": INVERSE_PROBE_THREE_CLASS_MIN_SCORE_RATIO,
            "candidate_counts_by_split": counts,
            "rows": total_rows,
            "shards": output_rows,
            "causal_window_frames": TOUCH_WINDOW_FRAMES,
            "observation_end_lag_ms": None if retrospective else 100,
            "event_frame_included": retrospective,
            "future_frames_included": False,
            "diagnostic_training_allowed": True,
            "formal_policy_training_allowed": False,
            "test_accessed": False,
            "raw_video_or_source_paths_persisted": False,
        }
        output_manifest["manifest_sha256"] = hashlib.sha256(
            _canonical(output_manifest)
        ).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(output_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    return output_manifest


def t8_smoke() -> dict[str, object]:
    actor = FactorizedTemporalActor().eval()
    with torch.no_grad():
        outputs = actor(torch.zeros((1, FRAME_COUNT, 3, 128, 128)))
    return {
        "status": "PASSED",
        "disposition": "NON_PROMOTING_T8_MODEL_SMOKE",
        "head_sizes": [int(item.shape[1]) for item in outputs],
        "config_sha256": CONFIG_HASH,
        "mobile_connected": False,
    }


V2_MODEL_SCHEMA = "hok-agent-t8-v2-auto-factorized-bc-tcn-v1"
V2_SPLIT_SCHEMA = "hok-agent-t8-v2-auto-split-v1"
V2_TRAINING_SCHEMA = "hok-agent-t8-v2-auto-pilot-v1"
V21_MODEL_SCHEMA = "hok-agent-t8-v2.1-live-factorized-bc-tcn-v1"
V21_TRAINING_SCHEMA = "hok-agent-t8-v2.1-live-pilot-v1"
V2_ADAPTER_SCHEMA = "hok-agent-t8-v2-video-adapter-v1"
V2_ARRAY_KEYS = {"frames", "movement", "combat", "aim", "target", "hold_bucket", "hold_ms", "timestamp_ns", "label_source", "input_sent"}
V21_SPLIT_SCHEMA = "hok-agent-t8-v2.1-live-split-v1"
V21_PILOT_SPLIT_SCHEMA = "hok-agent-t8-v2.1-live-pilot-split-v1"
V21_ARRAY_KEYS = V2_ARRAY_KEYS | {"transition_sequence", "last_dispatch_ns"}
V21_SPLIT_SEED = 20260815
V2_HOLD_BUCKETS = ("none", "short", "medium", "long")
V2_HEAD_SIZES = (len(MOVEMENTS), len(ABILITIES), len(AIMS), len(V2_HOLD_BUCKETS))
V2_CONFIG = {"frames": TOUCH_WINDOW_FRAMES, "backbone": "resnet18-v5-source-video-adapted", "temporal": "causal-residual-tcn-1x1-256-dilations-1-2-4", "heads": list(V2_HEAD_SIZES), "target_intent": "none_not_learned", "observation_end_lag_ms": 100, "class_balance": "sqrt_inverse_frequency_cap_5"}
V2_CONFIG_HASH = hashlib.sha256(_canonical(V2_CONFIG)).hexdigest()


@dataclass(frozen=True)
class V2Session:
    identity: str
    path: Path
    layout_sha256: str
    calibration_sha256: str
    source: str
    frames: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class T8V2Data:
    sessions: tuple[V2Session, ...]
    splits: dict[str, tuple[int, ...]]
    split_sha256: str


def _v2_factor_record(labels: np.ndarray) -> dict[str, object]:
    return {
        "movement": MOVEMENTS[int(labels[0])],
        "ability": ABILITIES[int(labels[1])],
        "aim": AIMS[int(labels[2])],
        "target": "none",
        "hold_ms": int(labels[4]),
    }


def _load_v2_session(path: Path) -> V2Session:
    summary_path, events_path, manifest_path, contract_path = path / "summary.json", path / "events.jsonl", path / "session-manifest.json", path / "action-contract.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T8Error("T8-v2 session metadata is unreadable") from exc
    if (not isinstance(summary, dict) or summary.get("schema_version") != KEYBOARD_V2_SCHEMA or summary.get("dataset_schema_version") != KEYBOARD_V2_DATA_SCHEMA or summary.get("status") != "COMPLETED" or summary.get("window_frames") != TOUCH_WINDOW_FRAMES or summary.get("event_source") != EXECUTED_ACTION_SOURCE or summary.get("formal_session") is not True or summary.get("published_as_formal") is not True or not isinstance(summary.get("duration_seconds"), (int, float)) or float(summary["duration_seconds"]) < 300.0 or not isinstance(summary.get("samples"), int) or int(summary["samples"]) < KEYBOARD_V2_MIN_FORMAL_SAMPLES or not isinstance(summary.get("factor_coverage"), dict) or summary["factor_coverage"].get("complete") is not True):
        raise T8Error("T8-v2 session summary is invalid")
    layout, calibration = summary.get("layout_sha256"), summary.get("action_contract_sha256")
    if not isinstance(layout, str) or not isinstance(calibration, str) or len(layout) != 64 or len(calibration) != 64 or not isinstance(contract, dict) or contract.get("source") != EXECUTED_ACTION_SOURCE or contract.get("layout_sha256") != layout or hashlib.sha256(_canonical(contract)).hexdigest() != calibration:
        raise T8Error("T8-v2 session identity is invalid")
    pieces: list[dict[str, np.ndarray]] = []
    for shard_path in sorted(path.glob("samples-*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            if set(shard.files) != V2_ARRAY_KEYS:
                raise T8Error("T8-v2 shard fields are invalid")
            pieces.append({key: shard[key] for key in V2_ARRAY_KEYS})
    if not pieces:
        raise T8Error("T8-v2 session has no shards")
    frames = np.concatenate([item["frames"] for item in pieces])
    columns = ("movement", "combat", "aim", "target", "hold_ms")
    labels = np.stack([np.concatenate([item[key] for item in pieces]) for key in columns], axis=1).astype(np.int64)
    buckets = np.concatenate([item["hold_bucket"] for item in pieces]).astype(np.int64)
    timestamps = np.concatenate([item["timestamp_ns"] for item in pieces]).astype(np.int64)
    sources = np.concatenate([item["label_source"] for item in pieces])
    sent = np.concatenate([item["input_sent"] for item in pieces]).astype(np.uint8)
    if (frames.dtype != np.uint8 or frames.shape != (len(labels), TOUCH_WINDOW_FRAMES, 128, 128, 3) or np.any(labels[:, 0] < 0) or np.any(labels[:, 0] >= len(MOVEMENTS)) or np.any(labels[:, 1] < 0) or np.any(labels[:, 1] >= len(ABILITIES)) or np.any(labels[:, 2] < 0) or np.any(labels[:, 2] >= len(AIMS)) or np.any(labels[:, 3] != 0) or np.any(buckets < 0) or np.any(buckets >= len(V2_HOLD_BUCKETS)) or set(sources.tolist()) != {1} or set(sent.tolist()) - {0, 1} or np.any((sent == 0) & np.any(labels[:, :2] != 0, axis=1))):
        raise T8Error("T8-v2 factor tensors are invalid")
    if (len(events) != len(labels) or len(timestamps) != len(labels) or np.any(np.diff(timestamps) < 0)):
        raise T8Error("T8-v2 timestamps/events are invalid")
    for index, event in enumerate(events):
        if (not isinstance(event, dict) or event.get("schema_version") != KEYBOARD_V2_SCHEMA or event.get("sequence") != index or event.get("timestamp_ns") != int(timestamps[index]) or event.get("frame_sha256") != hashlib.sha256(frames[index, -1].tobytes()).hexdigest() or event.get("action") != _v2_factor_record(labels[index]) or event.get("hold_bucket") != int(buckets[index]) or event.get("source") != EXECUTED_ACTION_SOURCE or event.get("input_sent") is not bool(sent[index])):
            raise T8Error("T8-v2 events do not bind samples")
    shards = [{"name": item.name, "sha256": _sha(item)} for item in sorted(path.glob("samples-*.npz"))]
    expected = {"schema_version": KEYBOARD_V2_SESSION_SCHEMA, "summary_sha256": _sha(summary_path), "events_sha256": _sha(events_path), "action_contract_file_sha256": _sha(contract_path), "shards": shards}
    identity = hashlib.sha256(_canonical(expected)).hexdigest()
    if manifest != {**expected, "session_sha256": identity}:
        raise T8Error("T8-v2 manifest does not bind session")
    return V2Session(
        identity,
        path,
        layout,
        calibration,
        EXECUTED_ACTION_SOURCE,
        frames,
        np.column_stack((labels[:, :3], buckets, labels[:, 4])),
    )


def _v2_session_names() -> tuple[str, ...]:
    return tuple(f"session-{index:03d}" for index in range(1, 13))


def _v2_sessions(root: Path, names: tuple[str, ...] | None = None) -> tuple[V2Session, ...]:
    all_names = _v2_session_names()
    selected_names = all_names if names is None else names
    if not selected_names or any(name not in all_names for name in selected_names):
        raise T8Error("T8-v2 session selection is invalid")
    paths = tuple(root / name for name in all_names)
    if (
        not root.is_dir()
        or any(not path.is_dir() for path in paths)
        or any(
            path.is_dir()
            for path in root.iterdir()
            if path.name.startswith("session-") and path.name not in all_names
        )
    ):
        raise T8Error("T8-v2 requires exactly twelve named sessions")
    sessions = tuple(_load_v2_session(root / name) for name in selected_names)
    if (
        len({item.layout_sha256 for item in sessions}) != 1
        or len({item.calibration_sha256 for item in sessions}) != 1
    ):
        raise T8Error("T8-v2 sessions must use one layout and touch calibration")
    return sessions


def _v2_splits() -> dict[str, tuple[int, ...]]:
    return {"train": tuple(range(8)), "dev": (8, 9), "test": (10, 11)}


def _v2_validate_coverage(
    sessions: tuple[V2Session, ...], splits: Mapping[str, tuple[int, ...]]
) -> None:
    for indices in splits.values():
        labels = np.concatenate([sessions[index].labels for index in indices])
        if (
            set(labels[:, 0]) != set(range(len(MOVEMENTS)))
            or set(labels[:, 1]) != set(range(len(ABILITIES)))
            or set(labels[labels[:, 1] >= 2, 2]) != set(range(1, len(AIMS)))
            or set(labels[np.any(labels[:, :2] != 0, axis=1), 3]) != {1, 2, 3}
        ):
            raise T8Error("T8-v2 split factor coverage is incomplete")


def freeze_t8_v2_split(*, dataset_root: Path, output_path: Path) -> dict[str, object]:
    root, output = _large_existing(dataset_root), _large_new(output_path)
    sessions, splits = _v2_sessions(root), _v2_splits()
    _v2_validate_coverage(sessions, splits)
    payload: dict[str, object] = {
        "schema_version": V2_SPLIT_SCHEMA,
        "config_sha256": V2_CONFIG_HASH,
        "layout_sha256": sessions[0].layout_sha256,
        "action_contract_sha256": sessions[0].calibration_sha256,
        "splits": {
            name: [
                {"name": sessions[index].path.name, "session_sha256": sessions[index].identity}
                for index in indices
            ]
            for name, indices in splits.items()
        },
    }
    payload["split_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _write_frozen_json(output, payload)
    return payload


def load_t8_v2_data(
    root: Path, split_path: Path | None = None, *, include_test: bool = True
) -> T8V2Data:
    directory = _large_existing(root)
    names = _v2_session_names()
    sessions = _v2_sessions(directory, names if include_test else names[:10])
    splits = _v2_splits() if include_test else {"train": tuple(range(8)), "dev": (8, 9)}
    identities = {session.path.name: session.identity for session in sessions}
    if not include_test:
        for name in names[10:]:
            manifest = _read_object(
                directory / name / "session-manifest.json", "T8-v2 test manifest is unreadable"
            )
            identity = manifest.get("session_sha256")
            if not isinstance(identity, str) or len(identity) != 64:
                raise T8Error("T8-v2 test manifest identity is invalid")
            identities[name] = identity
    full_splits = _v2_splits()
    expected: dict[str, object] = {
        "schema_version": V2_SPLIT_SCHEMA,
        "config_sha256": V2_CONFIG_HASH,
        "layout_sha256": sessions[0].layout_sha256,
        "action_contract_sha256": sessions[0].calibration_sha256,
        "splits": {
            name: [
                {"name": sessions[index].path.name, "session_sha256": sessions[index].identity}
                if include_test or index < len(sessions)
                else {"name": names[index], "session_sha256": identities[names[index]]}
                for index in indices
            ]
            for name, indices in full_splits.items()
        },
    }
    expected["split_sha256"] = hashlib.sha256(_canonical(expected)).hexdigest()
    path = _large_existing(split_path or directory / "t8-v2-split.json")
    if _read_object(path, "T8-v2 split is unreadable") != expected:
        raise T8Error("T8-v2 split manifest does not bind sessions")
    return T8V2Data(sessions, splits, cast(str, expected["split_sha256"]))


def _load_v21_session(path: Path) -> V2Session:
    summary_path = path / "summary.json"
    events_path = path / "events.jsonl"
    manifest_path = path / "session-manifest.json"
    contract_path = path / "action-contract.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T8Error("T8-v2.1 session metadata is unreadable") from exc
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != KEYBOARD_V21_SCHEMA
        or summary.get("dataset_schema_version") != KEYBOARD_V21_DATA_SCHEMA
        or summary.get("status") != "COMPLETED"
        or summary.get("window_frames") != TOUCH_WINDOW_FRAMES
        or summary.get("event_source") != SCRCPY_EXECUTED_ACTION_SOURCE
        or summary.get("formal_session") is not True
        or summary.get("published_as_formal") is not True
        or not isinstance(summary.get("duration_seconds"), (int, float))
        or float(summary["duration_seconds"]) < 300
        or not isinstance(summary.get("samples"), int)
        or int(summary["samples"]) < 2850
    ):
        raise T8Error("T8-v2.1 session summary is invalid")
    layout = summary.get("layout_sha256")
    action_contract = summary.get("action_contract_sha256")
    if (
        not isinstance(layout, str)
        or len(layout) != 64
        or not isinstance(action_contract, str)
        or len(action_contract) != 64
        or not isinstance(contract, dict)
        or contract.get("source") != SCRCPY_EXECUTED_ACTION_SOURCE
        or contract.get("layout_sha256") != layout
        or contract.get("executor") != "pinned_scrcpy_1.25_multitouch_v1"
        or contract.get("scrcpy_server_version") != SCRCPY_SERVER_VERSION
        or contract.get("scrcpy_server_sha256") != SCRCPY_SERVER_SHA256
        or contract.get("pointer_roles") != {"joystick": 0, "combat": 1}
        or summary.get("scrcpy_server_version") != SCRCPY_SERVER_VERSION
        or summary.get("scrcpy_server_sha256") != SCRCPY_SERVER_SHA256
        or hashlib.sha256(_canonical(contract)).hexdigest() != action_contract
    ):
        raise T8Error("T8-v2.1 session identity is invalid")
    pieces: list[dict[str, np.ndarray]] = []
    for shard_path in sorted(path.glob("samples-*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            if set(shard.files) != V21_ARRAY_KEYS:
                raise T8Error("T8-v2.1 shard fields are invalid")
            pieces.append({key: shard[key] for key in V21_ARRAY_KEYS})
    if not pieces:
        raise T8Error("T8-v2.1 session has no shards")
    frames = np.concatenate([piece["frames"] for piece in pieces])
    columns = ("movement", "combat", "aim", "target", "hold_ms")
    labels = np.stack(
        [np.concatenate([piece[key] for piece in pieces]) for key in columns], axis=1
    ).astype(np.int64)
    buckets = np.concatenate([piece["hold_bucket"] for piece in pieces]).astype(np.int64)
    timestamps = np.concatenate([piece["timestamp_ns"] for piece in pieces]).astype(np.int64)
    transitions = np.concatenate(
        [piece["transition_sequence"] for piece in pieces]
    ).astype(np.int64)
    dispatches = np.concatenate([piece["last_dispatch_ns"] for piece in pieces]).astype(np.int64)
    sources = np.concatenate([piece["label_source"] for piece in pieces])
    sent = np.concatenate([piece["input_sent"] for piece in pieces]).astype(np.uint8)
    if (
        frames.dtype != np.uint8
        or frames.shape != (len(labels), TOUCH_WINDOW_FRAMES, 128, 128, 3)
        or len(labels) != int(summary["samples"])
        or np.any(labels[:, 0] < 0)
        or np.any(labels[:, 0] >= len(MOVEMENTS))
        or np.any(labels[:, 1] < 0)
        or np.any(labels[:, 1] >= len(ABILITIES))
        or np.any(labels[:, 2] < 0)
        or np.any(labels[:, 2] >= len(AIMS))
        or np.any(labels[:, 3] != 0)
        or np.any(buckets < 0)
        or np.any(buckets >= len(V2_HOLD_BUCKETS))
        or set(sources.tolist()) != {1}
        or set(sent.tolist()) - {0, 1}
        or len(timestamps) != len(labels)
        or np.any(np.diff(timestamps) < 0)
        or np.any(np.diff(transitions) < 0)
        or np.any(dispatches > timestamps)
    ):
        raise T8Error("T8-v2.1 factor tensors are invalid")
    semantic_sequences: list[int] = []
    for index, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or event.get("schema_version") != KEYBOARD_V21_SCHEMA
            or event.get("sequence") != index
            or event.get("source") != SCRCPY_EXECUTED_ACTION_SOURCE
            or event.get("event_type") not in {"semantic_transition", "combat_conflict"}
        ):
            raise T8Error("T8-v2.1 semantic events are invalid")
        if event["event_type"] == "semantic_transition":
            sequence = event.get("transition_sequence")
            if not isinstance(sequence, int):
                raise T8Error("T8-v2.1 transition sequence is invalid")
            semantic_sequences.append(sequence)
    if semantic_sequences != list(range(1, len(semantic_sequences) + 1)):
        raise T8Error("T8-v2.1 transitions are not contiguous")
    shards = [
        {"name": item.name, "sha256": _sha(item)}
        for item in sorted(path.glob("samples-*.npz"))
    ]
    expected = {
        "schema_version": KEYBOARD_V21_SESSION_SCHEMA,
        "summary_sha256": _sha(summary_path),
        "events_sha256": _sha(events_path),
        "action_contract_file_sha256": _sha(contract_path),
        "shards": shards,
    }
    identity = hashlib.sha256(_canonical(expected)).hexdigest()
    if manifest != {**expected, "session_sha256": identity}:
        raise T8Error("T8-v2.1 manifest does not bind session")
    return V2Session(
        identity,
        path,
        layout,
        action_contract,
        SCRCPY_EXECUTED_ACTION_SOURCE,
        frames,
        np.column_stack((labels[:, :3], buckets, labels[:, 4])),
    )


def _v21_session_names(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        raise T8Error("T8-v2.1 dataset root is unavailable")
    invalid = [
        item.name
        for item in root.iterdir()
        if item.is_dir()
        and item.name.startswith("session-")
        and re.fullmatch(r"session-[0-9]+", item.name) is None
    ]
    if invalid:
        raise T8Error("T8-v2.1 dataset has invalid session directory names")
    names = tuple(
        sorted(
            item.name
            for item in root.iterdir()
            if item.is_dir() and re.fullmatch(r"session-[0-9]+", item.name)
        )
    )
    if not names:
        raise T8Error("T8-v2.1 dataset has no named sessions")
    return names


def _v21_splits(count: int) -> dict[str, tuple[int, ...]]:
    if count < 12:
        raise T8Error("T8-v2.1 freeze requires at least twelve sessions")
    indices = list(range(count))
    random.Random(V21_SPLIT_SEED).shuffle(indices)
    train_count, dev_count = (2 * count) // 3, count // 6
    return {
        "train": tuple(indices[:train_count]),
        "dev": tuple(indices[train_count : train_count + dev_count]),
        "test": tuple(indices[train_count + dev_count :]),
    }


def _v21_sessions(root: Path) -> tuple[V2Session, ...]:
    sessions = tuple(_load_v21_session(root / name) for name in _v21_session_names(root))
    if (
        len({session.layout_sha256 for session in sessions}) != 1
        or len({session.calibration_sha256 for session in sessions}) != 1
    ):
        raise T8Error("T8-v2.1 sessions must use one layout and action contract")
    return sessions


def _v21_validate_coverage(
    sessions: tuple[V2Session, ...], splits: Mapping[str, tuple[int, ...]]
) -> None:
    for indices in splits.values():
        labels = np.concatenate([sessions[index].labels for index in indices])
        if (
            not np.any(labels[:, 0] == 0)
            or not np.any(labels[:, 0] != 0)
            or not np.any(labels[:, 1] != 0)
            or not np.any(labels[:, 2] != 0)
            or not np.any(labels[:, 3] != 0)
        ):
            raise T8Error("T8-v2.1 split lacks movement, combat, aim, or hold coverage")


def freeze_t8_v21_split(*, dataset_root: Path, output_path: Path) -> dict[str, object]:
    root, output = _large_existing(dataset_root), _large_new(output_path)
    sessions = _v21_sessions(root)
    splits = _v21_splits(len(sessions))
    _v21_validate_coverage(sessions, splits)
    payload: dict[str, object] = {
        "schema_version": V21_SPLIT_SCHEMA,
        "split_seed": V21_SPLIT_SEED,
        "config_sha256": V2_CONFIG_HASH,
        "layout_sha256": sessions[0].layout_sha256,
        "action_contract_sha256": sessions[0].calibration_sha256,
        "session_count": len(sessions),
        "splits": {
            name: [
                {"name": sessions[index].path.name, "session_sha256": sessions[index].identity}
                for index in indices
            ]
            for name, indices in splits.items()
        },
    }
    payload["split_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _write_frozen_json(output, payload)
    return payload


def freeze_t8_v21_pilot_split(*, dataset_root: Path, output_path: Path) -> dict[str, object]:
    root, output = _large_existing(dataset_root), _large_new(output_path)
    sessions = _v21_sessions(root)
    if len(sessions) != 3:
        raise T8Error("T8-v2.1 pilot split requires exactly three sessions")
    splits = {"train": (0, 1), "dev": (2,), "test": ()}
    for indices in (splits["train"], splits["dev"]):
        labels = np.concatenate([sessions[index].labels for index in indices])
        if (
            not np.any(labels[:, 0] != 0)
            or not np.any(labels[:, 1] != 0)
            or not np.any(labels[:, 2] != 0)
            or not np.any(labels[:, 3] != 0)
        ):
            raise T8Error("T8-v2.1 pilot split lacks movement, combat, aim, or hold coverage")
    payload: dict[str, object] = {
        "schema_version": V21_PILOT_SPLIT_SCHEMA,
        "split_seed": None,
        "config_sha256": V2_CONFIG_HASH,
        "layout_sha256": sessions[0].layout_sha256,
        "action_contract_sha256": sessions[0].calibration_sha256,
        "session_count": len(sessions),
        "splits": {
            name: [
                {"name": sessions[index].path.name, "session_sha256": sessions[index].identity}
                for index in indices
            ]
            for name, indices in splits.items()
        },
    }
    payload["split_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _write_frozen_json(output, payload)
    return payload


def load_t8_v21_data(root: Path, split_path: Path | None = None) -> T8V2Data:
    directory = _large_existing(root)
    manifest_path = _large_existing(split_path or directory / "t8-v2.1-split.json")
    manifest = _read_object(manifest_path, "T8-v2.1 split is unreadable")
    splits_value = manifest.get("splits")
    if not isinstance(splits_value, dict):
        raise T8Error("T8-v2.1 split fields are invalid")
    split_rows = cast(dict[str, object], splits_value)
    names: list[str] = []
    split_names: dict[str, list[str]] = {}
    schema = manifest.get("schema_version")
    pilot = schema == V21_PILOT_SPLIT_SCHEMA
    if schema not in {V21_SPLIT_SCHEMA, V21_PILOT_SPLIT_SCHEMA}:
        raise T8Error("T8-v2.1 split schema is invalid")
    for split in ("train", "dev", "test"):
        rows = split_rows.get(split)
        if not isinstance(rows, list) or (not rows and not (pilot and split == "test")):
            raise T8Error("T8-v2.1 split fields are invalid")
        current: list[str] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                raise T8Error("T8-v2.1 split session is invalid")
            current.append(cast(str, row["name"]))
        split_names[split] = current
        names.extend(current)
    canonical_names = _v21_session_names(directory)
    if len(names) != len(set(names)) or set(names) != set(canonical_names):
        raise T8Error("T8-v2.1 split does not allocate every session once")
    deterministic = (
        {"train": (0, 1), "dev": (2,), "test": ()}
        if pilot and len(canonical_names) == 3
        else _v21_splits(len(canonical_names))
    )
    if any(split_names[split] != [canonical_names[index] for index in indices] for split, indices in deterministic.items()):
        raise T8Error("T8-v2.1 split allocation is not deterministic")
    loaded_names = split_names["train"] + split_names["dev"]
    sessions = tuple(_load_v21_session(directory / name) for name in loaded_names)
    identities = {session.path.name: session.identity for session in sessions}
    for row in cast(list[dict[str, object]], split_rows["test"]):
        name, identity = row.get("name"), row.get("session_sha256")
        if not isinstance(name, str) or not isinstance(identity, str) or len(identity) != 64:
            raise T8Error("T8-v2.1 sealed test identity is invalid")
        identities[name] = identity
    expected = {key: value for key, value in manifest.items() if key != "split_sha256"}
    if (
        expected.get("schema_version") != schema
        or expected.get("split_seed") != (None if pilot else V21_SPLIT_SEED)
        or expected.get("config_sha256") != V2_CONFIG_HASH
        or expected.get("session_count") != len(names)
        or expected.get("layout_sha256") != sessions[0].layout_sha256
        or expected.get("action_contract_sha256") != sessions[0].calibration_sha256
        or hashlib.sha256(_canonical(expected)).hexdigest() != manifest.get("split_sha256")
    ):
        raise T8Error("T8-v2.1 split identity is invalid")
    for rows in cast(dict[str, list[dict[str, object]]], splits_value).values():
        for row in rows:
            if row.get("session_sha256") != identities.get(cast(str, row.get("name"))):
                raise T8Error("T8-v2.1 split does not bind session identities")
    train_count = len(split_names["train"])
    splits = {
        "train": tuple(range(train_count)),
        "dev": tuple(range(train_count, len(sessions))),
    }
    return T8V2Data(sessions, splits, cast(str, manifest["split_sha256"]))


class _V2ResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(256, 256, 3, dilation=dilation, groups=256)
        self.pointwise = nn.Conv1d(256, 256, 1)
        self.activation = nn.GELU()
        self.dilation = dilation

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padded = nn.functional.pad(values, (2 * self.dilation, 0))
        output = self.pointwise(self.activation(self.depthwise(padded)))
        return cast(torch.Tensor, values + output)


class T8V2FactorizedActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.mix = nn.Conv1d(512, 256, 1)
        self.temporal = nn.Sequential(*(_V2ResidualBlock(value) for value in (1, 2, 4)))
        self.heads = nn.ModuleList(nn.Linear(256, size) for size in V2_HEAD_SIZES)

    def features(self, frames: torch.Tensor) -> torch.Tensor:
        if (
            frames.ndim != 5
            or tuple(frames.shape[1:])[-3:] != (3, 128, 128)
            or frames.shape[1] != TOUCH_WINDOW_FRAMES
        ):
            raise T8Error("T8-v2 actor requires Bx16x3x128x128 RGB only")
        batch = frames.shape[0]
        encoded = cast(
            torch.Tensor, self.encoder(frames.reshape(batch * TOUCH_WINDOW_FRAMES, 3, 128, 128))
        )
        mixed = self.mix(encoded.reshape(batch, TOUCH_WINDOW_FRAMES, 512).transpose(1, 2))
        return cast(torch.Tensor, self.temporal(mixed))[..., -1]

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(head(self.features(frames)) for head in self.heads)


def _v2_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frames).to(device).permute(0, 1, 4, 2, 3).float().div(255.0)


def _v2_examples(data: T8V2Data, split: str) -> tuple[np.ndarray, np.ndarray]:
    frames, labels = [], []
    for index in data.splits[split]:
        session = data.sessions[index]
        frames.append(session.frames)
        labels.append(session.labels)
    return (np.concatenate(frames), np.concatenate(labels))


def _v2_class_weights(labels: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    masks = (
        np.ones(len(labels), dtype=bool),
        np.ones(len(labels), dtype=bool),
        labels[:, 1] >= 2,
        np.any(labels[:, :2] != 0, axis=1),
    )
    result = []
    for index, (size, mask) in enumerate(zip(V2_HEAD_SIZES, masks, strict=True)):
        counts = np.bincount(labels[mask, index], minlength=size).astype(np.float32)
        if np.any((counts if index < 2 else counts[1:]) == 0):
            raise T8Error("T8-v2 train split has an unrepresented active class")
        if index >= 2:
            counts[0] = float(counts[1:].mean())
        weights = np.minimum(np.sqrt(counts.sum() / counts), 5.0)
        result.append(torch.from_numpy(weights / weights.mean()).to(device))
    return tuple(result)


def _v2_loss(
    outputs: tuple[torch.Tensor, ...], labels: torch.Tensor, weights: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    masks = (
        torch.ones(len(labels), dtype=torch.bool, device=labels.device),
        torch.ones(len(labels), dtype=torch.bool, device=labels.device),
        labels[:, 1] >= 2,
        torch.any(labels[:, :2] != 0, dim=1),
    )
    losses = []
    for index, mask in enumerate(masks):
        if not bool(mask.any()):
            raise T8Error("T8-v2 conditional loss has no active rows")
        losses.append(
            nn.functional.cross_entropy(
                outputs[index][mask], labels[mask, index], weight=weights[index]
            )
        )
    return torch.stack(losses).sum()


def _v2_legal_prediction(outputs: tuple[torch.Tensor, ...]) -> np.ndarray:
    probabilities = [value.softmax(1).detach().cpu().numpy() for value in outputs]
    result: list[list[int]] = []
    for row in range(len(probabilities[0])):
        best, best_score = (0, 0, 0, 0), -math.inf
        for movement in range(len(MOVEMENTS)):
            for combat in range(len(ABILITIES)):
                aims = (0,) if combat < 2 else tuple(range(len(AIMS)))
                holds = (0,) if movement == 0 and combat == 0 else (1, 2, 3)
                for aim in aims:
                    for hold in holds:
                        score = sum(
                            math.log(max(float(probabilities[index][row, value]), 1e-300))
                            for index, value in enumerate((movement, combat, aim, hold))
                        )
                        if score > best_score:
                            best, best_score = (movement, combat, aim, hold), score
        result.append(list(best))
    return np.asarray(result, dtype=np.int64)


def _v2_metrics(predicted: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    def head(index: int, size: int, mask: np.ndarray, offset: int = 0) -> dict[str, object]:
        if not mask.any():
            raise T8Error("T8-v2 metric has no active rows")
        return _head_metrics(
            predicted[mask, index] - offset, labels[mask, index] - offset, size - offset
        )

    masks = (
        np.ones(len(labels), dtype=bool),
        np.ones(len(labels), dtype=bool),
        labels[:, 1] >= 2,
        np.any(labels[:, :2] != 0, axis=1),
    )
    scores = {
        "movement": head(0, len(MOVEMENTS), masks[0]),
        "combat": head(1, len(ABILITIES), masks[1]),
        "aim": head(2, len(AIMS), masks[2], 1),
        "hold": head(3, len(V2_HOLD_BUCKETS), masks[3], 1),
    }
    return {
        "joint_exact": float((predicted == labels[:, :4]).all(1).mean()),
        "heads": scores,
        "legal_predictions": bool(
            all(
                (row[2] == 0 if row[1] < 2 else True)
                and (row[3] == 0 if row[0] == 0 and row[1] == 0 else row[3] > 0)
                for row in predicted
            )
        ),
    }


def _v2_primary_gate(metrics: Mapping[str, object], plurality_joint: float) -> bool:
    heads = cast(dict[str, dict[str, object]], metrics["heads"])
    recalls = [
        float(value)
        for head in heads.values()
        for value in cast(list[float], head["per_class_recall"])
    ]
    return bool(
        float(cast(float, metrics["joint_exact"])) >= 0.30
        and float(cast(float, heads["movement"]["macro_f1"])) >= 0.55
        and float(cast(float, heads["combat"]["macro_f1"])) >= 0.55
        and float(cast(float, heads["aim"]["macro_f1"])) >= 0.45
        and float(cast(float, heads["hold"]["macro_f1"])) >= 0.45
        and min(recalls) >= 0.30
        and float(cast(float, metrics["joint_exact"])) >= plurality_joint + 0.10
        and bool(metrics["legal_predictions"])
    )
class _V2VideoAdapter(nn.Module):
    def __init__(self, encoder_state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        encoder.load_state_dict(encoder_state, strict=True)
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128)
        )
        self.predictor = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projector(self.encoder(frames))
        return projected, self.predictor(projected)


def _v2_target_index(
    target_dir: Path, split: str
) -> tuple[tuple[tuple[Path, np.ndarray], ...], str, int]:
    if split not in {"train", "dev"}:
        raise T8Error("T8-v2 video adapter may not open video-test")
    manifest_path = target_dir / "manifest.json"
    manifest = _read_object(manifest_path, "V5 target manifest is unreadable")
    supplied = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        not isinstance(supplied, str)
        or supplied != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise T8Error("V5 target manifest hash is invalid")
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise T8Error("V5 target manifest shards are invalid")
    indexed: list[tuple[Path, np.ndarray]] = []
    session_ordinals: dict[str, int] = {}
    selected_count = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != split:
            continue
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise T8Error("V5 target shard name is invalid")
        path = target_dir / "shards" / name
        if row.get("sha256") != _sha(path):
            raise T8Error("V5 target shard hash is invalid")
        with np.load(path, allow_pickle=False) as shard:
            split_values = shard["split"]
            sessions = shard["session_hash"]
            if (
                set(split_values.tolist()) != {split}
                or shard["frames"].dtype != np.uint8
                or shard["frames"].shape[1:] != (128, 128, 3)
                or len(sessions) != len(shard["frames"])
            ):
                raise T8Error("V5 target shard split is invalid")
            selected: list[int] = []
            for index, session_value in enumerate(sessions.tolist()):
                session = str(session_value)
                ordinal = session_ordinals.get(session, 0)
                if ordinal % 5 == 0:
                    selected.append(index)
                session_ordinals[session] = ordinal + 1
            if selected:
                indices = np.asarray(selected, dtype=np.int64)
                indexed.append((path, indices))
                selected_count += len(indices)
    if not indexed:
        raise T8Error("T8-v2 video subset is empty")
    return tuple(indexed), supplied, selected_count


def _v2_image_batches(
    indexed: tuple[tuple[Path, np.ndarray], ...], batch_size: int, *, seed: int | None
) -> Iterator[np.ndarray]:
    order = np.arange(len(indexed))
    randomizer = np.random.default_rng(seed)
    if seed is not None:
        randomizer.shuffle(order)
    pending: list[np.ndarray] = []
    pending_count = 0
    for item_index in order:
        path, selected = indexed[int(item_index)]
        selected = selected.copy()
        if seed is not None:
            randomizer.shuffle(selected)
        with np.load(path, allow_pickle=False) as shard:
            pending.append(shard["frames"][selected])
        pending_count += len(selected)
        while pending_count >= batch_size:
            joined = np.concatenate(pending)
            yield joined[:batch_size]
            remainder = joined[batch_size:]
            pending = [remainder] if len(remainder) else []
            pending_count = len(remainder)
    if pending_count >= 2:
        yield np.concatenate(pending)


def _v2_image_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[1:] != (128, 128, 3) or frames.dtype != np.uint8:
        raise T8Error("T8-v2 adapter requires Nx128x128x3 uint8 RGB")
    return torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div(255.0)


def _v2_simsiam_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return -nn.functional.cosine_similarity(first, second.detach(), dim=1).mean()


def train_t8_v2_video_adapter(
    *,
    v5_source_dir: Path,
    target_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 256,
    epochs: int = 5,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 2 or epochs != 5:
        raise T8Error("T8-v2 adapter settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    output = _large_new(output_dir)
    source, target = _load_v5_initialization(v5_source_dir), torch.device(device)
    directory = _large_existing(target_dir)
    train_index, manifest_sha, train_count = _v2_target_index(directory, "train")
    dev_index, dev_manifest_sha, dev_count = _v2_target_index(directory, "dev")
    if manifest_sha != dev_manifest_sha:
        raise T8Error("V5 target manifest changed during adapter setup")
    torch.manual_seed(0)
    model = _V2VideoAdapter(source.encoder_state).to(target)
    for name, parameter in model.encoder.named_parameters():
        parameter.requires_grad = name.startswith(("conv1", "layer1", "layer2"))
    for module in model.encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (item for item in model.parameters() if item.requires_grad), lr=3e-4, weight_decay=1e-4
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    candidates: list[dict[str, int | str | float]] = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            for module in model.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            for frames in _v2_image_batches(train_index, batch_size, seed=epoch):
                batch = _v2_image_tensor(frames, target)
                first, second = batch, torch.flip(batch, dims=(3,))
                first_projection, first_prediction = model(first)
                second_projection, second_prediction = model(second)
                optimizer.zero_grad(set_to_none=True)
                loss = 0.5 * (
                    _v2_simsiam_loss(first_prediction, second_projection)
                    + _v2_simsiam_loss(second_prediction, first_projection)
                )
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            model.eval()
            losses = []
            with torch.no_grad():
                for frames in _v2_image_batches(dev_index, batch_size, seed=None):
                    batch = _v2_image_tensor(frames, target)
                    first_projection, first_prediction = model(batch)
                    second_projection, _ = model(torch.flip(batch, dims=(3,)))
                    losses.append(
                        float(_v2_simsiam_loss(first_prediction, second_projection).item())
                    )
            name = f"adapter-epoch-{epoch}.safetensors"
            save_file(
                model.cpu().state_dict(),
                staging / name,
                metadata={
                    "schema": V2_ADAPTER_SCHEMA,
                    "v5_source_model_sha256": source.model_sha256,
                    "target_manifest_sha256": manifest_sha,
                    "epoch": str(epoch),
                    "sample_rule": "per_session_ordinal_mod_5_eq_0",
                },
            )
            model.to(target)
            candidates.append(
                {
                    "epoch": epoch,
                    "model": name,
                    "sha256": _sha(staging / name),
                    "video_dev_loss": float(np.mean(losses)),
                }
            )
        selected = min(
            candidates,
            key=lambda item: (
                float(cast(float, item["video_dev_loss"])),
                int(cast(int, item["epoch"])),
            ),
        )
        report = {
            "schema_version": V2_ADAPTER_SCHEMA,
            "status": "VIDEO_ADAPTER_SELECTED",
            "v5_source_model_sha256": source.model_sha256,
            "target_manifest_sha256": manifest_sha,
            "sampling_rule": "per_session_ordinal_mod_5_eq_0",
            "train_samples": train_count,
            "dev_samples": dev_count,
            "seed": 0,
            "candidates": candidates,
            "selected": selected,
            "video_test_accessed": False,
        }
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return report
    except Exception:
        for item in staging.iterdir():
            item.unlink()
        staging.rmdir()
        raise


def _load_v2_adapter(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    metadata = _model_metadata(path)
    if metadata.get("schema") != V2_ADAPTER_SCHEMA:
        raise T8Error("T8-v2 adapter metadata is invalid")
    state = load_file(path, device=str(device))
    encoder = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    expected = T8V2FactorizedActor().encoder.state_dict()
    if set(encoder) != set(expected) or any(
        encoder[key].shape != expected[key].shape for key in encoder
    ):
        raise T8Error("T8-v2 adapter encoder is invalid")
    return encoder, metadata


def _v2_predict(
    model: T8V2FactorizedActor, frames: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            values.append(
                _v2_legal_prediction(model(_v2_tensor(frames[start : start + batch_size], device)))
            )
    return np.concatenate(values)


def _v2_plurality(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(
        [np.bincount(labels[:, index]).argmax() for index in range(4)], dtype=np.int64
    )
    candidate = _v2_legal_prediction(
        tuple(
            torch.nn.functional.one_hot(
                torch.tensor([values[index]]),
                num_classes=V2_HEAD_SIZES[index],
            ).float()
            * 30
            for index in range(4)
        )
    )
    return cast(np.ndarray, candidate[0])


class _VideoCombatTemporal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Conv1d(512, 256, 1)
        self.temporal = nn.Sequential(*(_V2ResidualBlock(value) for value in (1, 2, 4)))
        self.head = nn.Linear(256, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (TOUCH_WINDOW_FRAMES, 512):
            raise T8Error("three-class combat model requires Bx16x512 frozen features")
        mixed = self.mix(features.transpose(1, 2))
        return cast(torch.Tensor, self.head(cast(torch.Tensor, self.temporal(mixed))[..., -1]))


def _load_video_three_class_split(
    root: Path, split: str, *, retrospective: bool = False
) -> tuple[np.ndarray, np.ndarray, str]:
    if split not in {"train", "dev"}:
        raise T8Error("three-class video training may not access test")
    manifest_path = root / "manifest.json"
    manifest = _read_object(manifest_path, "three-class video manifest is unreadable")
    supplied = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    expected_schema = (
        VIDEO_RETROSPECTIVE_THREE_CLASS_SCHEMA
        if retrospective
        else VIDEO_THREE_CLASS_SCHEMA
    )
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("task")
        not in (
            {"retrospective_action_recognition"}
            if retrospective
            else {None, "strict_causal_next_action_diagnostic"}
        )
        or manifest.get("combat_vocabulary") != list(ABILITIES[1:4])
        or manifest.get("abstained_classes") != ["skill3"]
        or manifest.get("diagnostic_training_allowed") is not True
        or manifest.get("formal_policy_training_allowed") is not False
        or manifest.get("test_accessed") is not False
        or manifest.get("event_frame_included") is not retrospective
        or supplied != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise T8Error("three-class video manifest contract is invalid")
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise T8Error("three-class video manifest shards are invalid")
    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != split:
            continue
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise T8Error("three-class video shard name is invalid")
        path = root / "shards" / name
        if row.get("sha256") != _sha(path):
            raise T8Error("three-class video shard hash differs")
        with np.load(path, allow_pickle=False) as shard:
            current_frames = shard["frames"]
            current_labels = shard["combat_id"]
            if (
                current_frames.dtype != np.uint8
                or current_frames.shape[1:] != (TOUCH_WINDOW_FRAMES, 128, 128, 3)
                or current_labels.dtype != np.int8
                or current_labels.shape != (len(current_frames),)
                or np.any(current_labels < 0)
                or np.any(current_labels >= 3)
            ):
                raise T8Error("three-class video tensors are invalid")
            frames.append(current_frames)
            labels.append(current_labels.astype(np.int64))
    if not frames:
        raise T8Error("three-class video split is empty")
    return np.concatenate(frames), np.concatenate(labels), supplied


def _retrospective_roi_prediction(
    frames: np.ndarray, centers: tuple[tuple[int, int], ...]
) -> tuple[int, float]:
    if frames.shape != (TOUCH_WINDOW_FRAMES, 128, 128, 3) or len(centers) != 3:
        raise T8Error("retrospective ROI input is invalid")
    scores = _retrospective_feature_scores(
        frames,
        centers,
        radius_xy=(6, 4),
        temporal="previous",
        score_mode="abs_plus_positive",
    )
    ordered = sorted(scores.tolist())
    ratio = ordered[-1] / max(ordered[-2], 1e-6)
    return int(np.argmax(scores)), ratio


def _retrospective_feature_scores(
    frames: np.ndarray,
    centers: tuple[tuple[int, int], ...],
    *,
    radius_xy: tuple[int, int],
    temporal: str,
    score_mode: str,
) -> np.ndarray:
    if frames.shape != (TOUCH_WINDOW_FRAMES, 128, 128, 3) or len(centers) not in {3, 4}:
        raise T8Error("retrospective feature input is invalid")
    if temporal not in {"previous", "median_previous_three"}:
        raise T8Error("retrospective temporal feature is invalid")
    if score_mode not in {"absolute", "positive", "abs_plus_positive"}:
        raise T8Error("retrospective score feature is invalid")
    radius_x, radius_y = radius_xy
    scores: list[float] = []
    for x, y in centers:
        x0, x1 = max(0, x - radius_x), min(128, x + radius_x + 1)
        y0, y1 = max(0, y - radius_y), min(128, y + radius_y + 1)
        current = frames[-1, y0:y1, x0:x1].astype(np.float32)
        previous = (
            frames[-2, y0:y1, x0:x1].astype(np.float32)
            if temporal == "previous"
            else np.median(frames[-4:-1, y0:y1, x0:x1].astype(np.float32), axis=0)
        )
        delta = current - previous
        absolute = float(np.abs(delta).mean())
        positive = float(np.maximum(delta, 0).mean())
        scores.append(
            absolute
            if score_mode == "absolute"
            else positive
            if score_mode == "positive"
            else absolute + positive
        )
    return np.asarray(scores, dtype=np.float32)


def evaluate_t8_video_retrospective_roi(
    *,
    dataset_root: Path,
    probe_report_path: Path,
    inverse_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    root = _large_existing(dataset_root)
    probe_path = _large_existing(probe_report_path)
    inverse_path = _large_existing(inverse_report_path)
    output = _large_new(output_dir)
    manifest = _read_object(root / "manifest.json", "retrospective manifest is unreadable")
    probe = _read_object(probe_path, "video action probe report is unreadable")
    inverse = _read_object(inverse_path, "inverse probe report is unreadable")
    if (
        manifest.get("schema_version") != VIDEO_RETROSPECTIVE_THREE_CLASS_SCHEMA
        or manifest.get("task") != "retrospective_action_recognition"
        or manifest.get("event_frame_included") is not True
        or manifest.get("test_accessed") is not False
        or probe.get("schema_version") != "hok-agent-t8-video-action-probe-v7"
        or probe.get("promotion_allowed") is not False
        or not isinstance(probe.get("sessions"), list)
        or inverse.get("schema_version") != INVERSE_PROBE_SCHEMA
        or inverse.get("three_class_gate_passed") is not True
        or inverse.get("three_class_precision") != 1.0
    ):
        raise T8Error("retrospective ROI evidence contract is invalid")
    centers: dict[str, tuple[tuple[int, int], ...]] = {}
    for session in cast(list[dict[str, object]], probe["sessions"]):
        session_hash, points = session.get("session_hash"), session.get("centers_xy")
        if not isinstance(session_hash, str) or not isinstance(points, dict):
            raise T8Error("retrospective ROI session centers are invalid")
        selected: list[tuple[int, int]] = []
        for button_name in ABILITIES[1:4]:
            point = points.get(button_name)
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(isinstance(value, int) for value in point)
            ):
                raise T8Error("retrospective ROI button center is invalid")
            selected.append((int(point[0]), int(point[1])))
        centers[session_hash] = tuple(selected)
    shard_rows = manifest.get("shards")
    if not isinstance(shard_rows, list):
        raise T8Error("retrospective ROI shard manifest is invalid")
    predictions: dict[str, list[int]] = {"train": [], "dev": []}
    labels: dict[str, list[int]] = {"train": [], "dev": []}
    ratios: dict[str, list[float]] = {"train": [], "dev": []}
    for row in shard_rows:
        if not isinstance(row, dict) or row.get("split") not in {"train", "dev"}:
            raise T8Error("retrospective ROI split is invalid")
        split = cast(str, row["split"])
        shard_name = row.get("path")
        if not isinstance(shard_name, str):
            raise T8Error("retrospective ROI shard name is invalid")
        path = root / "shards" / shard_name
        if row.get("sha256") != _sha(path):
            raise T8Error("retrospective ROI shard hash differs")
        with np.load(path, allow_pickle=False) as shard:
            for frames, label, session_hash in zip(
                shard["frames"], shard["combat_id"], shard["session_hash"], strict=True
            ):
                session_centers = centers.get(str(session_hash))
                if session_centers is None:
                    raise T8Error("retrospective ROI session has no frozen centers")
                predicted, ratio = _retrospective_roi_prediction(frames, session_centers)
                predictions[split].append(predicted)
                labels[split].append(int(label))
                ratios[split].append(ratio)
    metrics: dict[str, dict[str, object]] = {}
    for split in ("train", "dev"):
        predicted_array = np.asarray(predictions[split], dtype=np.int64)
        target = np.asarray(labels[split], dtype=np.int64)
        metrics[split] = _head_metrics(predicted_array, target, 3)
        metrics[split]["median_top_to_second_ratio"] = float(np.median(ratios[split]))
    dev_metrics = metrics["dev"]
    dev_predictions = np.asarray(predictions["dev"], dtype=np.int64)
    dev_labels = np.asarray(labels["dev"], dtype=np.int64)
    shifted = _head_metrics(dev_predictions, np.roll(dev_labels, 1), 3)
    recalls = cast(list[float], dev_metrics["per_class_recall"])
    gate = bool(
        float(cast(float, dev_metrics["accuracy"])) >= 0.95
        and float(cast(float, dev_metrics["macro_f1"])) >= 0.85
        and min(recalls) >= 0.60
        and float(cast(float, dev_metrics["macro_f1"]))
        - float(cast(float, shifted["macro_f1"]))
        >= 0.50
    )
    report: dict[str, object] = {
        "schema_version": VIDEO_RETROSPECTIVE_ROI_SCHEMA,
        "status": "PASSED" if gate else "FAILED",
        "task": "retrospective_action_recognition",
        "method": "frozen_session_centers_last_frame_local_onset_v1",
        "roi_radius_xy": [6, 4],
        "dataset_manifest_sha256": manifest.get("manifest_sha256"),
        "probe_report_sha256": _sha(probe_path),
        "inverse_probe_report_sha256": _sha(inverse_path),
        "metrics": metrics,
        "shifted_dev_metrics": shifted,
        "gate_passed": gate,
        "automated_annotation_allowed": gate,
        "independent_precision_source": "controlled_inverse_probe_three_class",
        "self_consistency_only": True,
        "test_accessed": False,
        "formal_policy_training_allowed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def verify_t8_retrospective_baseline(*, baseline_dir: Path) -> dict[str, object]:
    baseline = _large_existing(baseline_dir)
    payload = _read_object(baseline / "baseline.json", "retrospective baseline is unreadable")
    recognizer = payload.get("recognizer")
    if (
        payload.get("schema_version") != RETROSPECTIVE_BASELINE_SCHEMA
        or payload.get("status") != "FROZEN"
        or not isinstance(recognizer, dict)
        or recognizer.get("classes") != list(ABILITIES[1:4])
        or recognizer.get("abstained_classes") != ["skill3"]
        or recognizer.get("window_frames") != TOUCH_WINDOW_FRAMES
        or recognizer.get("roi_radius_xy") != [6, 4]
        or recognizer.get("score") != "mean_abs_delta_plus_mean_positive_delta"
        or recognizer.get("production_min_top_to_second_ratio") != 1.25
        or payload.get("test_accessed") is not False
        or payload.get("device_input_allowed") is not False
        or payload.get("raw_video_or_source_paths_persisted") is not False
    ):
        raise T8Error("retrospective baseline contract is invalid")
    sums_path = baseline / "SHA256SUMS"
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise T8Error("retrospective baseline checksums are unavailable") from exc
    checked = 0
    for line in lines:
        digest, separator, relative_text = line.partition("  ")
        relative_text = relative_text.removeprefix("./")
        relative = Path(relative_text)
        if (
            separator != "  "
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.name == "SHA256SUMS"
        ):
            raise T8Error("retrospective baseline checksum row is invalid")
        path = baseline / relative
        if not path.is_file() or path.is_symlink() or _sha(path) != digest:
            raise T8Error("retrospective baseline file hash differs")
        checked += 1
    if checked < 10:
        raise T8Error("retrospective baseline is incomplete")
    return {
        "schema_version": RETROSPECTIVE_BASELINE_SCHEMA,
        "status": "VERIFIED",
        "baseline_sha256": _sha(baseline / "baseline.json"),
        "sha256sums_sha256": _sha(sums_path),
        "verified_files": checked,
        "test_accessed": False,
        "device_input_sent": False,
    }


def _retrospective_target_index(
    target_dir: Path, split: str, session_hashes: Sequence[str] = ()
) -> tuple[Path, str, tuple[tuple[str, tuple[dict[str, object], ...]], ...]]:
    if split not in {"train", "dev"}:
        raise T8Error("retrospective batch may open train or dev only")
    target = _large_existing(target_dir)
    manifest = _read_object(target / "manifest.json", "V5 target manifest is unreadable")
    supplied = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema_version") != V5_TARGET_MANIFEST_SCHEMA
        or not isinstance(supplied, str)
        or supplied != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or not isinstance(manifest.get("sessions"), list)
        or not isinstance(manifest.get("shards"), list)
    ):
        raise T8Error("V5 target manifest contract is invalid")
    selected = set(session_hashes)
    sessions: list[str] = []
    for row in cast(list[dict[str, object]], manifest["sessions"]):
        identity = row.get("session_hash")
        if (
            row.get("split") == split
            and isinstance(identity, str)
            and (not selected or identity in selected)
        ):
            sessions.append(identity)
    if selected and selected != set(sessions):
        raise T8Error("requested retrospective session is absent from split")
    indexed: dict[str, list[dict[str, object]]] = {identity: [] for identity in sessions}
    for row in cast(list[dict[str, object]], manifest["shards"]):
        if row.get("split") != split:
            continue
        identities = row.get("session_hashes")
        name = row.get("path")
        if (
            not isinstance(identities, list)
            or len(identities) != 1
            or not isinstance(identities[0], str)
            or not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(row.get("sha256"), str)
        ):
            raise T8Error("V5 target shard index is invalid")
        identity = identities[0]
        if identity in indexed:
            indexed[identity].append(row)
    result: list[tuple[str, tuple[dict[str, object], ...]]] = []
    for identity in sessions:
        rows = tuple(indexed[identity])
        if not rows:
            raise T8Error("V5 target session has no shards")
        result.append((identity, rows))
    if not result:
        raise T8Error("retrospective batch split is empty")
    return target, supplied, tuple(result)


def _retrospective_load_session(
    target: Path, split: str, identity: str, rows: Sequence[Mapping[str, object]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    hashes: list[np.ndarray] = []
    for row in rows:
        name = cast(str, row["path"])
        path = target / "shards" / name
        if _sha(path) != row.get("sha256"):
            raise T8Error("V5 target shard hash differs")
        with np.load(path, allow_pickle=False) as shard:
            required = {"frames", "timestamp_ms", "frame_hash", "session_hash", "split"}
            if not required.issubset(shard.files):
                raise T8Error("V5 target shard fields are incomplete")
            current_frames = shard["frames"]
            current_times = shard["timestamp_ms"]
            current_hashes = shard["frame_hash"]
            if (
                current_frames.dtype != np.uint8
                or current_frames.shape[1:] != (128, 128, 3)
                or current_times.dtype != np.int64
                or current_times.shape != (len(current_frames),)
                or current_hashes.shape != (len(current_frames),)
                or set(map(str, shard["session_hash"].tolist())) != {identity}
                or set(map(str, shard["split"].tolist())) != {split}
            ):
                raise T8Error("V5 target shard tensor contract is invalid")
            frames.append(current_frames.copy())
            timestamps.append(current_times.copy())
            hashes.append(current_hashes.astype("U64", copy=True))
    all_frames = np.concatenate(frames)
    all_times = np.concatenate(timestamps)
    all_hashes = np.concatenate(hashes)
    if len(all_frames) < TOUCH_WINDOW_FRAMES + 6 or np.any(np.diff(all_times) <= 0):
        raise T8Error("retrospective session timestamps are invalid")
    return all_frames, all_times, all_hashes


def _retrospective_content_box(
    frames: np.ndarray,
) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
    sample_indices = np.linspace(0, len(frames) - 1, min(64, len(frames)), dtype=np.int64)

    def bounds(values: np.ndarray) -> tuple[int, int, int, int] | None:
        mask = values.astype(np.float32).mean(axis=2) > 5.0
        rows = np.flatnonzero(mask.mean(axis=1) >= 0.05)
        columns = np.flatnonzero(mask.mean(axis=0) >= 0.05)
        if not len(rows) or not len(columns):
            return None
        return int(columns[0]), int(rows[0]), int(columns[-1] + 1), int(rows[-1] + 1)

    sampled = [bounds(frames[index]) for index in sample_indices]
    valid = [value for value in sampled if value is not None]
    if not valid:
        raise T8Error("retrospective session has no visible content")
    median = tuple(int(round(float(np.median([row[index] for row in valid])))) for index in range(4))
    width, height = median[2] - median[0], median[3] - median[1]
    orientation = "stored"
    canonical = frames
    if height > width * 1.5:
        canonical = np.rot90(frames, 1, axes=(1, 2))
        orientation = "counter_clockwise_90"
        sampled = [bounds(canonical[index]) for index in sample_indices]
        valid = [value for value in sampled if value is not None]
        median = tuple(
            int(round(float(np.median([row[index] for row in valid])))) for index in range(4)
        )
        width, height = median[2] - median[0], median[3] - median[1]
    if width < height * 1.5 or width < 32 or height < 20:
        raise T8Error("retrospective content orientation is ambiguous")
    return canonical, orientation, cast(tuple[int, int, int, int], median)


def _retrospective_centers(
    content_box: tuple[int, int, int, int], layout: Mapping[str, tuple[float, float]]
) -> tuple[tuple[int, int], ...]:
    x0, y0, x1, y1 = content_box
    return tuple(
        (
            min(x1 - 1, round(x0 + (x1 - x0) * layout[name][0])),
            min(y1 - 1, round(y0 + (y1 - y0) * layout[name][1])),
        )
        for name in ABILITIES[1:]
    )


def _retrospective_session_candidates(
    frames: np.ndarray,
    content_box: tuple[int, int, int, int],
    centers: tuple[tuple[int, int], ...],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    x0, y0, x1, y1 = content_box
    content = frames[:, y0:y1, x0:x1]
    content_change = np.zeros(len(frames), dtype=np.float32)
    for start in range(1, len(frames), 256):
        stop = min(len(frames), start + 256)
        difference = np.abs(
            content[start:stop].astype(np.int16) - content[start - 1 : stop - 1].astype(np.int16)
        )
        content_change[start:stop] = difference.mean(axis=(1, 2, 3))
    raw: list[dict[str, object]] = []
    diagnostics = {
        "raw_candidates": 0,
        "same_button_deduplicated": 0,
        "cross_button_conflicts": 0,
        "overlay_rejected": 0,
        "minimap_rejected": 0,
        "skill3_persistence_rejected": 0,
    }
    for combat_id, (center_x, center_y) in enumerate(centers):
        crop = frames[
            :,
            max(0, center_y - 4) : min(128, center_y + 5),
            max(0, center_x - 6) : min(128, center_x + 7),
        ].astype(np.int16)
        delta = crop[1:] - crop[:-1]
        roi_difference = np.zeros(len(frames), dtype=np.float32)
        onset = np.zeros(len(frames), dtype=np.float32)
        roi_difference[1:] = np.abs(delta).mean(axis=(1, 2, 3))
        onset[1:] = np.maximum(delta, 0).mean(axis=(1, 2, 3))
        decay = np.zeros(len(frames), dtype=np.float32)
        decay[:-5] = np.abs(crop[:-5] - crop[5:]).mean(axis=(1, 2, 3))
        threshold = max(18.0, float(np.quantile(roi_difference[1:], 0.995)))
        indices = np.flatnonzero(
            (roi_difference >= threshold)
            & (onset >= 20.0)
            & (decay >= 15.0)
            & (np.arange(len(frames)) >= TOUCH_WINDOW_FRAMES - 1)
            & (np.arange(len(frames)) < len(frames) - 5)
        )
        button_rows: list[dict[str, object]] = []
        for index in indices.tolist():
            if combat_id == 3 and float(
                np.max(roi_difference[max(1, index - 3) : index], initial=0.0)
            ) >= threshold * 0.5:
                diagnostics["skill3_persistence_rejected"] += 1
                continue
            ratio = float(roi_difference[index] / max(content_change[index], 1e-6))
            if ratio < 1.5:
                continue
            current = frames[index]
            center = current[
                y0 + round((y1 - y0) * 0.15) : y0 + round((y1 - y0) * 0.85),
                x0 + round((x1 - x0) * 0.2) : x0 + round((x1 - x0) * 0.8),
            ].astype(np.int16)
            spread = center.max(axis=2) - center.min(axis=2)
            dark_gray = float(np.mean((spread <= 25) & (center.mean(axis=2) <= 100)))
            if dark_gray > 0.25:
                diagnostics["overlay_rejected"] += 1
                continue
            minimap = current[y0 : y0 + round((y1 - y0) * 0.38), x0 : x0 + round((x1 - x0) * 0.23)].astype(np.float32)
            horizontal = np.abs(np.diff(minimap, axis=1)).mean() if minimap.shape[1] > 1 else 0.0
            vertical = np.abs(np.diff(minimap, axis=0)).mean() if minimap.shape[0] > 1 else 0.0
            minimap_edge = float((horizontal + vertical) / 2.0)
            if minimap_edge < 7.0:
                diagnostics["minimap_rejected"] += 1
                continue
            score = float(roi_difference[index] * decay[index] * ratio)
            button_rows.append(
                {
                    "frame_index": index,
                    "combat_id": combat_id,
                    "score": score,
                    "roi_difference": float(roi_difference[index]),
                    "onset": float(onset[index]),
                    "decay": float(decay[index]),
                    "roi_to_content_ratio": ratio,
                    "center_dark_gray_fraction": dark_gray,
                    "minimap_mean_edge_strength": minimap_edge,
                }
            )
        diagnostics["raw_candidates"] += len(button_rows)
        cursor = 0
        while cursor < len(button_rows):
            group = [button_rows[cursor]]
            cursor += 1
            while (
                cursor < len(button_rows)
                and cast(int, button_rows[cursor]["frame_index"])
                - cast(int, group[-1]["frame_index"])
                <= 8
            ):
                group.append(button_rows[cursor])
                cursor += 1
            raw.append(max(group, key=lambda row: cast(float, row["score"])))
            diagnostics["same_button_deduplicated"] += len(group) - 1
    raw.sort(key=lambda row: cast(int, row["frame_index"]))
    accepted: list[dict[str, object]] = []
    cursor = 0
    while cursor < len(raw):
        group = [raw[cursor]]
        cursor += 1
        while (
            cursor < len(raw)
            and cast(int, raw[cursor]["frame_index"])
            - cast(int, group[-1]["frame_index"])
            <= 2
        ):
            group.append(raw[cursor])
            cursor += 1
        ordered = sorted(group, key=lambda row: cast(float, row["score"]), reverse=True)
        if len(ordered) > 1:
            ratio = cast(float, ordered[0]["score"]) / max(cast(float, ordered[1]["score"]), 1e-6)
            if ratio < 1.25:
                diagnostics["cross_button_conflicts"] += 1
                continue
            ordered[0]["best_to_second_ratio"] = ratio
        else:
            ordered[0]["best_to_second_ratio"] = None
        accepted.append(ordered[0])
    return accepted, diagnostics


def run_t8_retrospective_batch(
    *,
    target_dir: Path,
    baseline_dir: Path,
    layout_path: Path,
    split: str,
    output_dir: Path,
    session_hashes: Sequence[str] = (),
) -> dict[str, object]:
    baseline = verify_t8_retrospective_baseline(baseline_dir=baseline_dir)
    target, target_sha256, sessions = _retrospective_target_index(
        target_dir, split, session_hashes
    )
    output = _large_new(output_dir)
    layout, layout_sha256 = load_layout(layout_path)
    baseline_payload = _read_object(
        _large_existing(baseline_dir) / "baseline.json", "retrospective baseline is unreadable"
    )
    if baseline_payload.get("layout_sha256") != layout_sha256:
        raise T8Error("retrospective batch layout differs from baseline")
    production_ratio = cast(
        float,
        cast(dict[str, object], baseline_payload["recognizer"])[
            "production_min_top_to_second_ratio"
        ],
    )
    button_layout: dict[str, tuple[float, float]] = {}
    for name in ABILITIES[1:]:
        point = layout.buttons.get(name)
        if point is None:
            raise T8Error("retrospective layout has no combat button")
        button_layout[name] = point
    session_reports: list[dict[str, object]] = []
    total_events = 0
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "events").mkdir()
        (staging / "qc").mkdir()
        for identity, rows in sessions:
            frames, timestamps, frame_hashes = _retrospective_load_session(
                target, split, identity, rows
            )
            canonical, orientation, content_box = _retrospective_content_box(frames)
            centers = _retrospective_centers(content_box, button_layout)
            candidates, diagnostics = _retrospective_session_candidates(
                canonical, content_box, centers
            )
            events: list[dict[str, object]] = []
            abstentions = {"skill3": 0, "classifier_disagreement": 0, "low_confidence": 0}
            for candidate in candidates:
                index = cast(int, candidate["frame_index"])
                combat_id = cast(int, candidate["combat_id"])
                if combat_id == 3:
                    abstentions["skill3"] += 1
                    continue
                predicted, ratio = _retrospective_roi_prediction(
                    canonical[index - TOUCH_WINDOW_FRAMES + 1 : index + 1], centers[:3]
                )
                if predicted != combat_id:
                    abstentions["classifier_disagreement"] += 1
                    continue
                if ratio < production_ratio:
                    abstentions["low_confidence"] += 1
                    continue
                event = {
                    "schema_version": RETROSPECTIVE_EVENT_SCHEMA,
                    "sequence": len(events),
                    "session_hash": identity,
                    "split": split,
                    "timestamp_ms": int(timestamps[index]),
                    "frame_index": index,
                    "combat": ABILITIES[combat_id + 1],
                    "roi_score": candidate["score"],
                    "second_score_ratio": ratio,
                    "confidence": float(1.0 - 1.0 / ratio),
                    "confidence_kind": "one_minus_second_over_top",
                    "frame_sha256": str(frame_hashes[index]),
                    "baseline_sha256": baseline["baseline_sha256"],
                    "target_manifest_sha256": target_sha256,
                }
                events.append(event)
            event_path = staging / "events" / f"{identity}.jsonl"
            event_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
                encoding="utf-8",
            )
            counts = {
                name: sum(row["combat"] == name for row in events) for name in ABILITIES[1:4]
            }
            qc: dict[str, object] = {
                "schema_version": RETROSPECTIVE_SESSION_QC_SCHEMA,
                "status": "PASSED" if events else "PASSED_NO_ACCEPTED_EVENTS",
                "session_hash": identity,
                "split": split,
                "evaluation_only": split == "dev",
                "frames_scanned": len(frames),
                "duration_ms": int(timestamps[-1] - timestamps[0]),
                "timestamps_monotonic": True,
                "orientation": orientation,
                "content_box_xyxy": list(content_box),
                "centers_xy": {
                    name: list(centers[index]) for index, name in enumerate(ABILITIES[1:])
                },
                "detector_candidates": len(candidates),
                "accepted_events": len(events),
                "accepted_counts": counts,
                "abstentions": abstentions,
                "diagnostics": diagnostics,
                "events_per_minute": (
                    len(events) * 60_000.0 / max(int(timestamps[-1] - timestamps[0]), 1)
                ),
                "events_sha256": _sha(event_path),
                "raw_rgb_persisted": False,
                "raw_video_or_source_paths_persisted": False,
                "test_accessed": False,
                "device_input_sent": False,
            }
            qc_path = staging / "qc" / f"{identity}.json"
            qc_path.write_text(json.dumps(qc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            session_reports.append(
                {
                    "session_hash": identity,
                    "status": qc["status"],
                    "events": len(events),
                    "events_sha256": qc["events_sha256"],
                    "qc_sha256": _sha(qc_path),
                }
            )
            total_events += len(events)
        manifest: dict[str, object] = {
            "schema_version": RETROSPECTIVE_BATCH_SCHEMA,
            "status": "COMPLETED",
            "split": split,
            "evaluation_only": split == "dev",
            "session_count": len(session_reports),
            "event_count": total_events,
            "baseline_sha256": baseline["baseline_sha256"],
            "target_manifest_sha256": target_sha256,
            "layout_sha256": layout_sha256,
            "sessions": session_reports,
            "test_accessed": False,
            "raw_rgb_persisted": False,
            "raw_video_or_source_paths_persisted": False,
            "device_input_sent": False,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return manifest


def verify_t8_retrospective_batch(*, batch_dir: Path) -> dict[str, object]:
    root = _large_existing(batch_dir)
    manifest = _read_object(root / "manifest.json", "retrospective batch manifest is unreadable")
    supplied = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    sessions = manifest.get("sessions")
    split = manifest.get("split")
    if (
        manifest.get("schema_version") != RETROSPECTIVE_BATCH_SCHEMA
        or manifest.get("status") != "COMPLETED"
        or split not in {"train", "dev"}
        or not isinstance(sessions, list)
        or supplied != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or manifest.get("test_accessed") is not False
        or manifest.get("raw_rgb_persisted") is not False
        or manifest.get("raw_video_or_source_paths_persisted") is not False
        or manifest.get("device_input_sent") is not False
    ):
        raise T8Error("retrospective batch manifest contract is invalid")
    total_events = 0
    forbidden = {"frames", "raw_rgb", "raw_video", "source_path", "device_path"}
    for session in cast(list[dict[str, object]], sessions):
        identity = session.get("session_hash")
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise T8Error("retrospective batch session identity is invalid")
        event_path = root / "events" / f"{identity}.jsonl"
        qc_path = root / "qc" / f"{identity}.json"
        if (
            _sha(event_path) != session.get("events_sha256")
            or _sha(qc_path) != session.get("qc_sha256")
        ):
            raise T8Error("retrospective batch output hash differs")
        qc = _read_object(qc_path, "retrospective session QC is unreadable")
        if (
            qc.get("schema_version") != RETROSPECTIVE_SESSION_QC_SCHEMA
            or qc.get("session_hash") != identity
            or qc.get("split") != split
            or qc.get("evaluation_only") is not (split == "dev")
            or qc.get("events_sha256") != session.get("events_sha256")
            or qc.get("test_accessed") is not False
            or qc.get("raw_rgb_persisted") is not False
            or qc.get("raw_video_or_source_paths_persisted") is not False
            or qc.get("device_input_sent") is not False
        ):
            raise T8Error("retrospective session QC contract is invalid")
        try:
            rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
        except (OSError, json.JSONDecodeError) as exc:
            raise T8Error("retrospective event stream is unreadable") from exc
        for sequence, event in enumerate(rows):
            if (
                not isinstance(event, dict)
                or event.get("schema_version") != RETROSPECTIVE_EVENT_SCHEMA
                or event.get("sequence") != sequence
                or event.get("session_hash") != identity
                or event.get("split") != split
                or event.get("combat") not in ABILITIES[1:4]
                or forbidden.intersection(event)
            ):
                raise T8Error("retrospective event contract is invalid")
        if len(rows) != session.get("events") or len(rows) != qc.get("accepted_events"):
            raise T8Error("retrospective event count differs")
        total_events += len(rows)
    if len(sessions) != manifest.get("session_count") or total_events != manifest.get("event_count"):
        raise T8Error("retrospective batch aggregate count differs")
    return {
        "schema_version": RETROSPECTIVE_BATCH_SCHEMA,
        "status": "VERIFIED",
        "manifest_sha256": supplied,
        "split": split,
        "sessions_verified": len(sessions),
        "events_verified": total_events,
        "test_accessed": False,
        "raw_rgb_persisted": False,
        "device_input_sent": False,
    }


def _retrospective_candidate_data(
    dataset_root: Path, probe_report_path: Path
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, tuple[tuple[int, int], ...]],
    str,
    str,
]:
    root = _large_existing(dataset_root)
    probe_path = _large_existing(probe_report_path)
    manifest = _read_object(root / "manifest.json", "retrospective candidate manifest is unreadable")
    probe = _read_object(probe_path, "retrospective probe report is unreadable")
    if (
        manifest.get("schema_version")
        != "hok-agent-t8-video-combat-pseudolabel-candidates-v1"
        or manifest.get("combat_vocabulary") != list(ABILITIES[1:])
        or manifest.get("causal_window_includes_action_frame") is not True
        or manifest.get("test_tensors_accessed") is not False
        or not isinstance(manifest.get("shards"), list)
        or probe.get("schema_version") != "hok-agent-t8-video-action-probe-v7"
        or probe.get("test_accessed") is not False
        or not isinstance(probe.get("sessions"), list)
    ):
        raise T8Error("retrospective candidate evidence contract is invalid")
    centers: dict[str, tuple[tuple[int, int], ...]] = {}
    for session in cast(list[dict[str, object]], probe["sessions"]):
        identity, values = session.get("session_hash"), session.get("centers_xy")
        if not isinstance(identity, str) or not isinstance(values, dict):
            raise T8Error("retrospective candidate centers are invalid")
        points: list[tuple[int, int]] = []
        for name in ABILITIES[1:]:
            point = values.get(name)
            if not isinstance(point, list) or len(point) != 2 or not all(
                isinstance(value, int) for value in point
            ):
                raise T8Error("retrospective candidate button center is invalid")
            points.append((cast(int, point[0]), cast(int, point[1])))
        centers[identity] = tuple(points)
    pieces: dict[str, dict[str, list[np.ndarray]]] = {
        "train": {"frames": [], "labels": [], "sessions": []},
        "dev": {"frames": [], "labels": [], "sessions": []},
    }
    for row in cast(list[dict[str, object]], manifest["shards"]):
        shard_split, shard_name = row.get("split"), row.get("path")
        if (
            shard_split not in {"train", "dev"}
            or not isinstance(shard_name, str)
            or Path(shard_name).name != shard_name
        ):
            raise T8Error("retrospective candidate shard index is invalid")
        path = root / "shards" / shard_name
        if row.get("sha256") != _sha(path):
            raise T8Error("retrospective candidate shard hash differs")
        with np.load(path, allow_pickle=False) as shard:
            frames, labels, sessions = (
                shard["frames"],
                shard["combat_id"],
                shard["session_hash"],
            )
            if (
                frames.dtype != np.uint8
                or frames.shape[1:] != (TOUCH_WINDOW_FRAMES, 128, 128, 3)
                or labels.dtype != np.int8
                or labels.shape != (len(frames),)
                or np.any(labels < 0)
                or np.any(labels >= 4)
                or sessions.shape != (len(frames),)
                or any(str(value) not in centers for value in sessions.tolist())
            ):
                raise T8Error("retrospective candidate shard tensor is invalid")
            pieces[shard_split]["frames"].append(frames.copy())
            pieces[shard_split]["labels"].append(labels.astype(np.int64))
            pieces[shard_split]["sessions"].append(sessions.astype("U64"))
    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in ("train", "dev"):
        if not pieces[split]["frames"]:
            raise T8Error("retrospective candidate split is empty")
        data[split] = (
            np.concatenate(pieces[split]["frames"]),
            np.concatenate(pieces[split]["labels"]),
            np.concatenate(pieces[split]["sessions"]),
        )
    return data, centers, _sha(root / "manifest.json"), _sha(probe_path)


def _retrospective_inverse_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    artifact = _large_existing(path)
    with np.load(artifact, allow_pickle=False) as shard:
        required = {"before_rgb", "after_rgb", "combat"}
        if not required.issubset(shard.files):
            raise T8Error("retrospective inverse artifact is incomplete")
        before, after, labels = shard["before_rgb"], shard["after_rgb"], shard["combat"]
        if (
            before.dtype != np.uint8
            or before.shape[1:] != (128, 128, 3)
            or after.dtype != np.uint8
            or after.shape != (len(before), 3, 128, 128, 3)
            or labels.dtype != np.int8
            or labels.shape != (len(before),)
            or np.any(labels < 0)
            or np.any(labels >= 4)
        ):
            raise T8Error("retrospective inverse tensor is invalid")
        return before.copy(), after.copy(), labels.astype(np.int64)


def _retrospective_video_scores(
    frames: np.ndarray,
    sessions: np.ndarray,
    centers: Mapping[str, tuple[tuple[int, int], ...]],
    config: Mapping[str, object],
) -> np.ndarray:
    scores = np.empty((len(frames), 4), dtype=np.float32)
    radius = cast(list[int], config["roi_radius_xy"])
    for index, (window, identity) in enumerate(zip(frames, sessions, strict=True)):
        scores[index] = _retrospective_feature_scores(
            window,
            centers[str(identity)],
            radius_xy=(radius[0], radius[1]),
            temporal=cast(str, config["temporal"]),
            score_mode=cast(str, config["score_mode"]),
        )
    return scores


def _retrospective_inverse_scores(
    before: np.ndarray,
    after: np.ndarray,
    centers: tuple[tuple[int, int], ...],
    config: Mapping[str, object],
) -> np.ndarray:
    radius = cast(list[int], config["roi_radius_xy"])
    radius_x, radius_y = radius
    result = np.empty((len(before), 4), dtype=np.float32)
    for combat_id, (center_x, center_y) in enumerate(centers):
        current = after[
            :,
            :,
            max(0, center_y - radius_y) : min(128, center_y + radius_y + 1),
            max(0, center_x - radius_x) : min(128, center_x + radius_x + 1),
        ].astype(np.float32)
        reference = before[
            :,
            max(0, center_y - radius_y) : min(128, center_y + radius_y + 1),
            max(0, center_x - radius_x) : min(128, center_x + radius_x + 1),
        ].astype(np.float32)
        delta = current - reference[:, None]
        absolute = np.abs(delta).mean(axis=(2, 3, 4))
        positive = np.maximum(delta, 0).mean(axis=(2, 3, 4))
        mode = config["score_mode"]
        values = absolute if mode == "absolute" else positive if mode == "positive" else absolute + positive
        result[:, combat_id] = values.max(axis=1)
    return result


def _retrospective_selective_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    threshold: float | Sequence[float],
) -> dict[str, object]:
    mask = labels < class_count
    current_scores = scores[mask, :class_count]
    current_labels = labels[mask]
    predicted = current_scores.argmax(axis=1)
    ordered = np.sort(current_scores, axis=1)
    ratios = ordered[:, -1] / np.maximum(ordered[:, -2], 1e-6)
    if isinstance(threshold, Sequence):
        thresholds = np.asarray(threshold, dtype=np.float32)
        if thresholds.shape != (class_count,):
            raise T8Error("retrospective class thresholds are invalid")
        accepted = ratios >= thresholds[predicted]
    else:
        accepted = ratios >= threshold
    recalls = [
        float(np.mean(accepted[current_labels == index] & (predicted[current_labels == index] == index)))
        for index in range(class_count)
    ]
    precisions = [
        (
            float(np.mean(current_labels[accepted & (predicted == index)] == index))
            if np.any(accepted & (predicted == index))
            else 0.0
        )
        for index in range(class_count)
    ]
    return {
        "rows": len(current_labels),
        "coverage": float(np.mean(accepted)),
        "accepted_accuracy": (
            float(np.mean(predicted[accepted] == current_labels[accepted])) if np.any(accepted) else 0.0
        ),
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "accepted_counts": [int(np.sum(accepted & (predicted == index))) for index in range(class_count)],
    }


def run_t8_retrospective_calibration_v2(
    *,
    dataset_root: Path,
    probe_report_path: Path,
    layout_path: Path,
    baseline_dir: Path,
    inverse_calibration_paths: Sequence[Path],
    inverse_holdout_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if len(inverse_calibration_paths) != 2:
        raise T8Error("retrospective v2 requires exactly two calibration inverse artifacts")
    baseline = verify_t8_retrospective_baseline(baseline_dir=baseline_dir)
    data, centers, dataset_sha256, probe_sha256 = _retrospective_candidate_data(
        dataset_root, probe_report_path
    )
    layout, layout_sha256 = load_layout(layout_path)
    points: list[tuple[int, int]] = []
    for name in ABILITIES[1:]:
        point = layout.buttons.get(name)
        if point is None:
            raise T8Error("retrospective calibration layout has no combat button")
        points.append((round(127 * point[0]), round(127 * point[1])))
    inverse_calibration = [_retrospective_inverse_data(path) for path in inverse_calibration_paths]
    inverse_holdout = _retrospective_inverse_data(inverse_holdout_path)
    configs: list[dict[str, object]] = [
        {
            "roi_radius_xy": [radius_x, radius_y],
            "temporal": temporal,
            "score_mode": score_mode,
        }
        for radius_x, radius_y in ((3, 2), (4, 3), (6, 4))
        for temporal in ("previous", "median_previous_three")
        for score_mode in ("absolute", "positive", "abs_plus_positive")
    ]
    thresholds = (1.05, 1.10, 1.25, 1.50, 1.70)
    train_frames, train_labels, train_sessions = data["train"]
    dev_frames, dev_labels, dev_sessions = data["dev"]
    ranked_three: list[tuple[tuple[float, ...], dict[str, object], dict[str, object], dict[str, object]]] = []
    ranked_four: list[tuple[tuple[float, ...], dict[str, object], dict[str, object], dict[str, object]]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for config in configs:
        key = hashlib.sha256(_canonical(config)).hexdigest()
        train_scores = _retrospective_video_scores(train_frames, train_sessions, centers, config)
        dev_scores = _retrospective_video_scores(dev_frames, dev_sessions, centers, config)
        calibration_scores = np.concatenate(
            [_retrospective_inverse_scores(before, after, tuple(points), config) for before, after, _ in inverse_calibration]
        )
        calibration_labels = np.concatenate([labels for _, _, labels in inverse_calibration])
        holdout_scores = _retrospective_inverse_scores(
            inverse_holdout[0], inverse_holdout[1], tuple(points), config
        )
        cache[key] = (train_scores, dev_scores, calibration_scores, holdout_scores)
        for threshold in thresholds:
            train_three = _retrospective_selective_metrics(train_scores, train_labels, 3, threshold)
            inverse_three = _retrospective_selective_metrics(
                calibration_scores, calibration_labels, 3, threshold
            )
            train_recall = cast(list[float], train_three["per_class_recall"])
            train_precision = cast(list[float], train_three["per_class_precision"])
            inverse_recall = cast(list[float], inverse_three["per_class_recall"])
            inverse_precision = cast(list[float], inverse_three["per_class_precision"])
            candidate: dict[str, object] = {
                **config,
                "min_top_to_second_ratio": threshold,
            }
            if (
                min(train_precision) >= 0.95
                and train_recall[0] >= 0.75
                and min(train_recall[1:]) >= 0.90
                and min(inverse_precision) >= 0.95
                and inverse_recall[0] >= 0.80
                and min(inverse_recall[1:]) >= 0.70
            ):
                rank = (
                    min(train_recall + inverse_recall),
                    train_recall[0],
                    cast(float, train_three["coverage"]),
                    -float(
                        cast(list[int], config["roi_radius_xy"])[0]
                        * cast(list[int], config["roi_radius_xy"])[1]
                    ),
                    -threshold,
                )
                ranked_three.append((rank, candidate, train_three, inverse_three))
            train_four = _retrospective_selective_metrics(train_scores, train_labels, 4, threshold)
            inverse_four = _retrospective_selective_metrics(
                calibration_scores, calibration_labels, 4, threshold
            )
            four_recall = cast(list[float], train_four["per_class_recall"])
            four_precision = cast(list[float], train_four["per_class_precision"])
            inverse_four_recall = cast(list[float], inverse_four["per_class_recall"])
            inverse_four_precision = cast(list[float], inverse_four["per_class_precision"])
            if (
                min(four_precision) >= 0.90
                and four_recall[3] >= 0.75
                and min(inverse_four_precision) >= 0.95
                and inverse_four_recall[3] >= 0.60
            ):
                rank = (
                    min(four_recall + inverse_four_recall),
                    four_recall[3],
                    cast(float, train_four["coverage"]),
                    -float(
                        cast(list[int], config["roi_radius_xy"])[0]
                        * cast(list[int], config["roi_radius_xy"])[1]
                    ),
                    -threshold,
                )
                ranked_four.append((rank, candidate, train_four, inverse_four))

        def choose_class_thresholds(
            class_count: int,
            current_train_scores: np.ndarray,
            current_calibration_scores: np.ndarray,
            current_calibration_labels: np.ndarray,
        ) -> list[float] | None:
            chosen: list[float] = []
            for class_id in range(class_count):
                options: list[tuple[tuple[float, ...], float]] = []
                for threshold in thresholds:
                    train_metrics = _retrospective_selective_metrics(
                        current_train_scores, train_labels, class_count, threshold
                    )
                    inverse_metrics = _retrospective_selective_metrics(
                        current_calibration_scores,
                        current_calibration_labels,
                        class_count,
                        threshold,
                    )
                    train_recall = cast(list[float], train_metrics["per_class_recall"])[
                        class_id
                    ]
                    train_precision = cast(
                        list[float], train_metrics["per_class_precision"]
                    )[class_id]
                    inverse_recall = cast(
                        list[float], inverse_metrics["per_class_recall"]
                    )[class_id]
                    inverse_precision = cast(
                        list[float], inverse_metrics["per_class_precision"]
                    )[class_id]
                    required_train_precision = (
                        0.90 if class_count == 4 and class_id == 3 else 0.95 if class_id == 0 else 0.98
                    )
                    required_train_recall = (
                        0.75 if class_id in {0, 3} else 0.90
                    )
                    required_inverse_recall = (
                        0.60 if class_count == 4 and class_id == 3 else 0.80 if class_id == 0 else 0.70
                    )
                    if (
                        train_precision >= required_train_precision
                        and train_recall >= required_train_recall
                        and inverse_precision >= 0.95
                        and inverse_recall >= required_inverse_recall
                    ):
                        options.append(
                            (
                                (
                                    min(train_recall, inverse_recall),
                                    train_recall,
                                    inverse_recall,
                                    -threshold,
                                ),
                                threshold,
                            )
                        )
                if not options:
                    return None
                chosen.append(max(options, key=lambda row: row[0])[1])
            return chosen

        for class_count, ranked in ((3, ranked_three), (4, ranked_four)):
            class_thresholds = choose_class_thresholds(
                class_count, train_scores, calibration_scores, calibration_labels
            )
            if class_thresholds is None:
                continue
            train_metrics = _retrospective_selective_metrics(
                train_scores, train_labels, class_count, class_thresholds
            )
            inverse_metrics = _retrospective_selective_metrics(
                calibration_scores, calibration_labels, class_count, class_thresholds
            )
            recalls = cast(list[float], train_metrics["per_class_recall"]) + cast(
                list[float], inverse_metrics["per_class_recall"]
            )
            radius = cast(list[int], config["roi_radius_xy"])
            rank = (
                min(recalls),
                recalls[0] if class_count == 3 else recalls[3],
                cast(float, train_metrics["coverage"]),
                -float(radius[0] * radius[1]),
                -sum(class_thresholds),
            )
            class_config: dict[str, object] = {
                **config,
                "min_top_to_second_ratio_by_class": class_thresholds,
            }
            ranked.append((rank, class_config, train_metrics, inverse_metrics))

    def validate(
        ranked: list[tuple[tuple[float, ...], dict[str, object], dict[str, object], dict[str, object]]],
        class_count: int,
    ) -> dict[str, object] | None:
        if not ranked:
            return None
        _rank, config, train_metrics, calibration_metrics = max(ranked, key=lambda row: row[0])
        key = hashlib.sha256(
            _canonical({name: config[name] for name in ("roi_radius_xy", "temporal", "score_mode")})
        ).hexdigest()
        _train_scores, dev_scores, _calibration_scores, holdout_scores = cache[key]
        threshold: float | Sequence[float] = (
            cast(list[float], config["min_top_to_second_ratio_by_class"])
            if "min_top_to_second_ratio_by_class" in config
            else cast(float, config["min_top_to_second_ratio"])
        )
        return {
            "config": config,
            "train": train_metrics,
            "inverse_calibration": calibration_metrics,
            "dev": _retrospective_selective_metrics(dev_scores, dev_labels, class_count, threshold),
            "inverse_holdout": _retrospective_selective_metrics(
                holdout_scores, inverse_holdout[2], class_count, threshold
            ),
        }

    three = validate(ranked_three, 3)
    four = validate(ranked_four, 4)
    basic_promotion = False
    if three is not None:
        dev = cast(dict[str, object], three["dev"])
        holdout = cast(dict[str, object], three["inverse_holdout"])
        dev_recall = cast(list[float], dev["per_class_recall"])
        dev_precision = cast(list[float], dev["per_class_precision"])
        holdout_recall = cast(list[float], holdout["per_class_recall"])
        holdout_precision = cast(list[float], holdout["per_class_precision"])
        basic_promotion = bool(
            dev_recall[0] >= 0.75
            and dev_precision[0] >= 0.95
            and min(dev_recall[1:]) >= 0.90
            and min(dev_precision[1:]) >= 0.98
            and holdout_recall[0] >= 0.80
            and holdout_precision[0] >= 0.95
        )
    skill3_enabled = False
    if four is not None:
        dev = cast(dict[str, object], four["dev"])
        holdout = cast(dict[str, object], four["inverse_holdout"])
        skill3_enabled = bool(
            cast(list[float], dev["per_class_recall"])[3] >= 0.75
            and cast(list[float], dev["per_class_precision"])[3] >= 0.90
            and cast(list[float], holdout["per_class_recall"])[3] >= 0.60
            and cast(list[float], holdout["per_class_precision"])[3] >= 0.95
        )
    report: dict[str, object] = {
        "schema_version": RETROSPECTIVE_CALIBRATION_SCHEMA,
        "status": "UPGRADE_PASSED" if basic_promotion or skill3_enabled else "BASELINE_RETAINED",
        "baseline_sha256": baseline["baseline_sha256"],
        "dataset_manifest_file_sha256": dataset_sha256,
        "probe_report_sha256": probe_sha256,
        "layout_sha256": layout_sha256,
        "inverse_calibration_sha256": [_sha(_large_existing(path)) for path in inverse_calibration_paths],
        "inverse_holdout_sha256": _sha(_large_existing(inverse_holdout_path)),
        "search_space_size": len(configs) * len(thresholds),
        "selection_uses": ["video_train", "inverse_calibration_1", "inverse_calibration_2"],
        "validation_uses": ["video_dev", "inverse_holdout"],
        "three_class": three,
        "four_class": four,
        "basic_promotion_allowed": basic_promotion,
        "skill3_enabled": skill3_enabled,
        "skill3_temporal_gate": "quiet_onset_decay_refractory_cross_button_conflict",
        "test_accessed": False,
        "formal_policy_training_allowed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _causal_batch_events(
    batch_dir: Path, split: str, target_manifest_sha256: str
) -> tuple[Path, dict[str, tuple[dict[str, object], ...]], str]:
    if split not in {"train", "dev"}:
        raise T8Error("causal video data may open train or dev only")
    root = _large_existing(batch_dir)
    verified = verify_t8_retrospective_batch(batch_dir=root)
    manifest = _read_object(root / "manifest.json", "retrospective batch manifest is unreadable")
    if (
        verified.get("split") != split
        or manifest.get("target_manifest_sha256") != target_manifest_sha256
        or manifest.get("test_accessed") is not False
    ):
        raise T8Error("causal video event batch differs from target or split")
    indexed: dict[str, tuple[dict[str, object], ...]] = {}
    for session in cast(list[dict[str, object]], manifest["sessions"]):
        identity = cast(str, session["session_hash"])
        try:
            events = tuple(
                cast(dict[str, object], json.loads(line))
                for line in (root / "events" / f"{identity}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise T8Error("causal video event stream is unreadable") from exc
        indexed[identity] = events
    return root, indexed, cast(str, verified["manifest_sha256"])


def _causal_session_rows(
    timestamps: np.ndarray,
    frame_hashes: np.ndarray,
    events: Sequence[Mapping[str, object]],
    lag_ms: int,
) -> tuple[dict[str, object], ...]:
    if lag_ms not in {100, 200, 300}:
        raise T8Error("causal video lag must be 100, 200, or 300 ms")
    event_times: list[int] = []
    action_rows: list[dict[str, object]] = []
    for event in events:
        frame_index = event.get("frame_index")
        timestamp = event.get("timestamp_ms")
        confidence = event.get("confidence")
        combat = event.get("combat")
        if (
            not isinstance(frame_index, int)
            or not isinstance(timestamp, int)
            or not isinstance(confidence, (int, float))
            or combat not in ABILITIES[1:4]
            or frame_index < 0
            or frame_index >= len(timestamps)
            or int(timestamps[frame_index]) != timestamp
            or str(frame_hashes[frame_index]) != event.get("frame_sha256")
        ):
            raise T8Error("causal video event does not bind to the target frame")
        event_times.append(timestamp)
        end_index = int(np.searchsorted(timestamps, timestamp - lag_ms, side="right") - 1)
        start_index = end_index - TOUCH_WINDOW_FRAMES + 1
        if start_index < 0:
            continue
        observation_end = int(timestamps[end_index])
        if observation_end > timestamp - lag_ms or observation_end >= timestamp:
            raise T8Error("causal video window leaks the action frame")
        action_rows.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "combat_id": ABILITIES.index(combat),
                "label_timestamp_ms": timestamp,
                "event_timestamp_ms": timestamp,
                "observation_end_timestamp_ms": observation_end,
                "event_confidence": float(confidence),
                "label_kind": 1,
            }
        )
    event_time_array = np.asarray(event_times, dtype=np.int64)
    wait_candidates: list[int] = []
    last_end = int(np.searchsorted(timestamps, int(timestamps[-1]) - lag_ms, side="right") - 1)
    for end_index in range(TOUCH_WINDOW_FRAMES - 1, last_end + 1, 10):
        label_timestamp = int(timestamps[end_index]) + lag_ms
        if len(event_time_array) and int(np.min(np.abs(event_time_array - label_timestamp))) < 1000:
            continue
        wait_candidates.append(end_index)
    wait_count = min(len(action_rows), len(wait_candidates))
    selected_waits: list[int] = []
    if wait_count:
        selected_waits = [
            wait_candidates[index]
            for index in np.linspace(0, len(wait_candidates) - 1, wait_count, dtype=np.int64)
        ]
    rows = list(action_rows)
    for end_index in selected_waits:
        rows.append(
            {
                "start_index": end_index - TOUCH_WINDOW_FRAMES + 1,
                "end_index": end_index,
                "combat_id": 0,
                "label_timestamp_ms": int(timestamps[end_index]) + lag_ms,
                "event_timestamp_ms": -1,
                "observation_end_timestamp_ms": int(timestamps[end_index]),
                "event_confidence": 1.0,
                "label_kind": 0,
            }
        )
    rows.sort(
        key=lambda row: (
            cast(int, row["label_timestamp_ms"]),
            cast(int, row["label_kind"]),
            cast(int, row["combat_id"]),
        )
    )
    for row in rows:
        start, end = cast(int, row["start_index"]), cast(int, row["end_index"])
        row["causal_window_sha256"] = hashlib.sha256(
            _canonical([str(value) for value in frame_hashes[start : end + 1]])
        ).hexdigest()
    return tuple(rows)


def _causal_encode_frames(
    encoder: nn.Module,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            tensor = _v2_image_tensor(frames[start : start + batch_size], device)
            features.append(cast(torch.Tensor, encoder(tensor)).cpu().numpy())
    return np.concatenate(features).astype(np.float32, copy=False)


def materialize_t8_causal_video_dataset(
    *,
    target_dir: Path,
    train_events_dir: Path,
    dev_events_dir: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 256,
    lags_ms: Sequence[int] = (100, 200, 300),
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1 or tuple(lags_ms) != (100, 200, 300):
        raise T8Error("causal video materialization settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    target = torch.device(device)
    target_root, target_sha256, split_sessions = _retrospective_target_index(target_dir, "train")
    dev_root, dev_target_sha256, dev_sessions = _retrospective_target_index(target_dir, "dev")
    if target_root != dev_root or target_sha256 != dev_target_sha256:
        raise T8Error("causal video target manifest changed during setup")
    train_root, train_events, train_batch_sha = _causal_batch_events(
        train_events_dir, "train", target_sha256
    )
    dev_event_root, dev_events, dev_batch_sha = _causal_batch_events(
        dev_events_dir, "dev", target_sha256
    )
    if set(train_events) != {identity for identity, _ in split_sessions} or set(dev_events) != {
        identity for identity, _ in dev_sessions
    }:
        raise T8Error("causal video event sessions differ from target split")
    adapter = _large_existing(adapter_checkpoint)
    encoder_state, adapter_meta = _load_v2_adapter(adapter, target)
    if adapter_meta.get("target_manifest_sha256") != target_sha256:
        raise T8Error("causal video adapter differs from target manifest")
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(target).eval()
    output = _large_new(output_dir)
    shard_rows: list[dict[str, object]] = []
    counts = {
        str(lag): {
            split: {name: 0 for name in ABILITIES[:4]} for split in ("train", "dev")
        }
        for lag in lags_ms
    }
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shards_dir = staging / "shards"
        shards_dir.mkdir()
        for split, sessions, event_index in (
            ("train", split_sessions, train_events),
            ("dev", dev_sessions, dev_events),
        ):
            for ordinal, (identity, source_rows) in enumerate(sessions):
                frames, timestamps, frame_hashes = _retrospective_load_session(
                    target_root, split, identity, source_rows
                )
                frame_features = _causal_encode_frames(encoder, frames, target, batch_size)
                for lag_ms in lags_ms:
                    examples = _causal_session_rows(
                        timestamps, frame_hashes, event_index[identity], lag_ms
                    )
                    if not examples:
                        continue
                    feature_rows = np.stack(
                        [
                            frame_features[
                                cast(int, row["start_index"]) : cast(int, row["end_index"]) + 1
                            ]
                            for row in examples
                        ]
                    ).astype(np.float16, copy=False)
                    label_ids = np.asarray(
                        [cast(int, row["combat_id"]) for row in examples], dtype=np.int8
                    )
                    name = f"lag-{lag_ms}-{split}-{ordinal:04d}.npz"
                    path = shards_dir / name
                    np.savez_compressed(
                        path,
                        features=feature_rows,
                        combat_id=label_ids,
                        label_timestamp_ms=np.asarray(
                            [cast(int, row["label_timestamp_ms"]) for row in examples],
                            dtype=np.int64,
                        ),
                        event_timestamp_ms=np.asarray(
                            [cast(int, row["event_timestamp_ms"]) for row in examples],
                            dtype=np.int64,
                        ),
                        observation_end_timestamp_ms=np.asarray(
                            [cast(int, row["observation_end_timestamp_ms"]) for row in examples],
                            dtype=np.int64,
                        ),
                        event_confidence=np.asarray(
                            [cast(float, row["event_confidence"]) for row in examples],
                            dtype=np.float32,
                        ),
                        label_kind=np.asarray(
                            [cast(int, row["label_kind"]) for row in examples], dtype=np.int8
                        ),
                        session_hash=np.asarray([identity] * len(examples)),
                        causal_window_sha256=np.asarray(
                            [cast(str, row["causal_window_sha256"]) for row in examples]
                        ),
                    )
                    class_counts = {
                        action: int(np.sum(label_ids == action_id))
                        for action_id, action in enumerate(ABILITIES[:4])
                    }
                    for action, count in class_counts.items():
                        counts[str(lag_ms)][split][action] += count
                    shard_rows.append(
                        {
                            "path": name,
                            "sha256": _sha(path),
                            "split": split,
                            "lag_ms": lag_ms,
                            "session_hash": identity,
                            "rows": len(examples),
                            "class_counts": class_counts,
                        }
                    )
                del frames, frame_features
        manifest: dict[str, object] = {
            "schema_version": CAUSAL_VIDEO_DATASET_SCHEMA,
            "status": "COMPLETED",
            "task": "strict_causal_next_action_four_class_diagnostic",
            "lags_ms": list(lags_ms),
            "window_frames": TOUCH_WINDOW_FRAMES,
            "feature_shape": [TOUCH_WINDOW_FRAMES, 512],
            "feature_dtype": "float16",
            "encoder": "selected_t8_v2_video_adapter_frozen",
            "adapter_sha256": _sha(adapter),
            "adapter_source_sha256": adapter_meta.get("v5_source_model_sha256"),
            "target_manifest_sha256": target_sha256,
            "event_batch_manifest_sha256": {
                "train": train_batch_sha,
                "dev": dev_batch_sha,
            },
            "combat_vocabulary": list(ABILITIES[:4]),
            "wait_rule": "prediction_time_at_least_1000ms_from_any_accepted_event",
            "wait_sampling": "deterministic_up_to_session_action_count",
            "class_balance": "sqrt_inverse_frequency_cap_5",
            "counts": counts,
            "shards": shard_rows,
            "event_frame_included": False,
            "future_frames_included": False,
            "video_test_accessed": False,
            "raw_rgb_persisted": False,
            "raw_video_or_source_paths_persisted": False,
            "formal_policy_training_allowed": False,
            "shadow_allowed": False,
            "device_input_allowed": False,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    del train_root, dev_event_root
    return manifest


def _load_causal_video_split(
    dataset_root: Path, split: str, lag_ms: int
) -> tuple[np.ndarray, np.ndarray, str]:
    if split not in {"train", "dev"}:
        raise T8Error("causal video training may open train or dev only")
    root = _large_existing(dataset_root)
    manifest = _read_object(root / "manifest.json", "causal video manifest is unreadable")
    supplied = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema_version") != CAUSAL_VIDEO_DATASET_SCHEMA
        or manifest.get("status") != "COMPLETED"
        or lag_ms not in cast(list[int], manifest.get("lags_ms"))
        or manifest.get("combat_vocabulary") != list(ABILITIES[:4])
        or manifest.get("event_frame_included") is not False
        or manifest.get("future_frames_included") is not False
        or manifest.get("video_test_accessed") is not False
        or supplied != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise T8Error("causal video dataset contract is invalid")
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise T8Error("causal video shard manifest is invalid")
    for row in cast(list[dict[str, object]], rows):
        if row.get("split") != split or row.get("lag_ms") != lag_ms:
            continue
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise T8Error("causal video shard name is invalid")
        path = root / "shards" / name
        if _sha(path) != row.get("sha256"):
            raise T8Error("causal video shard hash differs")
        with np.load(path, allow_pickle=False) as shard:
            expected = {
                "features",
                "combat_id",
                "label_timestamp_ms",
                "event_timestamp_ms",
                "observation_end_timestamp_ms",
                "event_confidence",
                "label_kind",
                "session_hash",
                "causal_window_sha256",
            }
            current_x, current_y = shard["features"], shard["combat_id"]
            observation_end = shard["observation_end_timestamp_ms"]
            event_timestamp = shard["event_timestamp_ms"]
            label_kind = shard["label_kind"]
            if (
                set(shard.files) != expected
                or current_x.dtype != np.float16
                or current_x.shape[1:] != (TOUCH_WINDOW_FRAMES, 512)
                or current_y.dtype != np.int8
                or current_y.shape != (len(current_x),)
                or np.any(current_y < 0)
                or np.any(current_y >= 4)
                or np.any(
                    (label_kind == 1)
                    & (observation_end > event_timestamp - lag_ms)
                )
            ):
                raise T8Error("causal video shard tensor contract is invalid")
            features.append(current_x.astype(np.float32))
            labels.append(current_y.astype(np.int64))
    if not features:
        raise T8Error("causal video split is empty")
    return np.concatenate(features), np.concatenate(labels), supplied


class _VideoCausalTemporal(nn.Module):
    def __init__(self, class_count: int = 4) -> None:
        super().__init__()
        if class_count < 2:
            raise T8Error("causal video model requires at least two classes")
        self.mix = nn.Conv1d(512, 256, 1)
        self.temporal = nn.Sequential(*(_V2ResidualBlock(value) for value in (1, 2, 4)))
        self.head = nn.Linear(256, class_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (TOUCH_WINDOW_FRAMES, 512):
            raise T8Error("causal video model requires Bx16x512 frozen features")
        mixed = self.mix(features.transpose(1, 2))
        return cast(torch.Tensor, self.head(cast(torch.Tensor, self.temporal(mixed))[..., -1]))


def _predict_causal_temporal(
    model: _VideoCausalTemporal,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predicted: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            predicted.append(
                model(torch.from_numpy(features[start : start + batch_size]).to(device))
                .argmax(1)
                .cpu()
                .numpy()
            )
    return np.concatenate(predicted)


def _fit_causal_four_class(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    shuffled: bool,
    class_count: int = 4,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    labels = train_y.copy()
    if shuffled:
        labels = labels[np.random.default_rng(0).permutation(len(labels))]
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    if np.any(counts == 0):
        raise T8Error("causal video train split lacks a class")
    weight_values = np.minimum(np.sqrt(counts.sum() / counts), 5.0)
    weights = torch.from_numpy(weight_values / weight_values.mean()).to(device)
    torch.manual_seed(0)
    model = _VideoCausalTemporal(class_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state = math.inf, 0, {}
    for epoch in range(1, 13):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(torch.from_numpy(train_x[selected]).to(device)),
                torch.from_numpy(labels[selected]).to(device),
                weight=weights,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        validation_loss, validation_rows = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(dev_x), batch_size):
                batch_y = torch.from_numpy(dev_y[start : start + batch_size]).to(device)
                current = nn.functional.cross_entropy(
                    model(torch.from_numpy(dev_x[start : start + batch_size]).to(device)),
                    batch_y,
                    weight=weights,
                    reduction="sum",
                )
                validation_loss += float(current.item())
                validation_rows += len(batch_y)
        current_loss = validation_loss / validation_rows
        if current_loss < best_loss:
            best_loss, best_epoch = current_loss, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if not best_state:
        raise T8Error("causal video pilot did not validate")
    model.load_state_dict(best_state, strict=True)
    metrics = _head_metrics(
        _predict_causal_temporal(model, dev_x, device, batch_size), dev_y, class_count
    )
    return (
        {
            "best_epoch": best_epoch,
            "best_dev_weighted_cross_entropy": best_loss,
            "metrics": metrics,
            "shuffled_labels": shuffled,
        },
        best_state,
    )


def run_t8_causal_video_pilot(
    *,
    dataset_root: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 256,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise T8Error("causal video pilot settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    root = _large_existing(dataset_root)
    adapter = _large_existing(adapter_checkpoint)
    target = torch.device(device)
    candidates: list[dict[str, object]] = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    manifest_sha: str | None = None
    for lag_ms in (100, 200, 300):
        train_x, train_y, current_sha = _load_causal_video_split(root, "train", lag_ms)
        dev_x, dev_y, dev_sha = _load_causal_video_split(root, "dev", lag_ms)
        if current_sha != dev_sha or (manifest_sha is not None and current_sha != manifest_sha):
            raise T8Error("causal video manifest changed during training")
        manifest_sha = current_sha
        normal, state = _fit_causal_four_class(
            train_x, train_y, dev_x, dev_y, target, batch_size, shuffled=False
        )
        shuffled, _ = _fit_causal_four_class(
            train_x, train_y, dev_x, dev_y, target, batch_size, shuffled=True
        )
        model = _VideoCausalTemporal().to(target)
        model.load_state_dict(state, strict=True)
        static_x = np.repeat(dev_x[:, -1:, :], TOUCH_WINDOW_FRAMES, axis=1)
        static_metrics = _head_metrics(
            _predict_causal_temporal(model, static_x, target, batch_size), dev_y, 4
        )
        normal_metrics = cast(dict[str, object], normal["metrics"])
        shuffled_metrics = cast(dict[str, object], shuffled["metrics"])
        normal_f1 = float(cast(float, normal_metrics["macro_f1"]))
        shuffled_f1 = float(cast(float, shuffled_metrics["macro_f1"]))
        static_f1 = float(cast(float, static_metrics["macro_f1"]))
        plurality = int(np.bincount(train_y, minlength=4).argmax())
        plurality_accuracy = float(np.mean(dev_y == plurality))
        recalls = cast(list[float], normal_metrics["per_class_recall"])
        gate = bool(
            float(cast(float, normal_metrics["accuracy"])) >= plurality_accuracy + 0.05
            and normal_f1 >= 0.45
            and float(cast(float, normal_metrics["macro_recall"])) >= 0.45
            and recalls[0] >= 0.50
            and min(recalls[1:]) >= 0.30
            and normal_f1 - shuffled_f1 >= 0.15
            and normal_f1 - static_f1 >= 0.10
        )
        candidates.append(
            {
                "lag_ms": lag_ms,
                "normal": normal,
                "shuffled": shuffled,
                "static_frame": static_metrics,
                "plurality_class": ABILITIES[plurality],
                "plurality_dev_accuracy": plurality_accuracy,
                "normal_minus_shuffled_macro_f1": normal_f1 - shuffled_f1,
                "normal_minus_static_macro_f1": normal_f1 - static_f1,
                "gate_passed": gate,
                "train_rows": len(train_y),
                "dev_rows": len(dev_y),
            }
        )
        states[lag_ms] = state
    selected = max(
        candidates,
        key=lambda row: (
            cast(
                float,
                cast(dict[str, object], cast(dict[str, object], row["normal"])["metrics"])[
                    "macro_f1"
                ],
            ),
            -cast(float, cast(dict[str, object], row["normal"])["best_dev_weighted_cross_entropy"]),
            -cast(int, row["lag_ms"]),
        ),
    )
    selected_lag = cast(int, selected["lag_ms"])
    gate_passed = bool(selected["gate_passed"])
    report: dict[str, object] = {
        "schema_version": CAUSAL_VIDEO_TRAINING_SCHEMA,
        "status": "PILOT_PASSED" if gate_passed else "PILOT_DIAGNOSIS_REQUIRED",
        "task": "strict_causal_next_action_four_class_diagnostic",
        "seed": 0,
        "epochs": 12,
        "selection_uses": "video_dev_only",
        "selection_metric": "dev_macro_f1_then_weighted_cross_entropy",
        "dataset_manifest_sha256": manifest_sha,
        "adapter_sha256": _sha(adapter),
        "combat_vocabulary": list(ABILITIES[:4]),
        "candidates": candidates,
        "selected_lag_ms": selected_lag,
        "gate_passed": gate_passed,
        "video_test_accessed": False,
        "formal_policy_training_allowed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        model_path = staging / "causal-combat-temporal-seed0.safetensors"
        save_file(
            states[selected_lag],
            model_path,
            metadata={
                "schema": CAUSAL_VIDEO_TRAINING_SCHEMA,
                "dataset_manifest_sha256": cast(str, manifest_sha),
                "adapter_sha256": _sha(adapter),
                "selected_lag_ms": str(selected_lag),
                "seed": "0",
            },
        )
        report["model_sha256"] = _sha(model_path)
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _causal_probe_bundle(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    class_count: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    normal, state = _fit_causal_four_class(
        train_x,
        train_y,
        dev_x,
        dev_y,
        device,
        batch_size,
        shuffled=False,
        class_count=class_count,
    )
    shuffled, _ = _fit_causal_four_class(
        train_x,
        train_y,
        dev_x,
        dev_y,
        device,
        batch_size,
        shuffled=True,
        class_count=class_count,
    )
    model = _VideoCausalTemporal(class_count).to(device)
    model.load_state_dict(state, strict=True)
    train_metrics = _head_metrics(
        _predict_causal_temporal(model, train_x, device, batch_size), train_y, class_count
    )
    static_x = np.repeat(dev_x[:, -1:, :], TOUCH_WINDOW_FRAMES, axis=1)
    static_metrics = _head_metrics(
        _predict_causal_temporal(model, static_x, device, batch_size), dev_y, class_count
    )
    normal_metrics = cast(dict[str, object], normal["metrics"])
    shuffled_metrics = cast(dict[str, object], shuffled["metrics"])
    normal_f1 = cast(float, normal_metrics["macro_f1"])
    return {
        "normal": normal,
        "shuffled": shuffled,
        "train_metrics": train_metrics,
        "static_frame": static_metrics,
        "normal_minus_shuffled_macro_f1": normal_f1
        - cast(float, shuffled_metrics["macro_f1"]),
        "normal_minus_static_macro_f1": normal_f1
        - cast(float, static_metrics["macro_f1"]),
    }


def _stratified_row_holdout(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_rows: list[np.ndarray] = []
    dev_rows: list[np.ndarray] = []
    randomizer = np.random.default_rng(20260815)
    for class_id in range(int(labels.max()) + 1):
        rows = np.flatnonzero(labels == class_id)
        randomizer.shuffle(rows)
        cut = max(1, min(len(rows) - 1, int(len(rows) * 0.8)))
        train_rows.append(rows[:cut])
        dev_rows.append(rows[cut:])
    return np.concatenate(train_rows), np.concatenate(dev_rows)


def run_t8_causal_video_diagnostic(
    *,
    dataset_root: Path,
    pilot_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 256,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise T8Error("causal video diagnostic settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    target = torch.device(device)
    root = _large_existing(dataset_root)
    pilot = _large_existing(pilot_dir)
    pilot_report = _read_object(pilot / "report.json", "causal pilot report is unreadable")
    lag_ms = pilot_report.get("selected_lag_ms")
    if (
        pilot_report.get("schema_version") != CAUSAL_VIDEO_TRAINING_SCHEMA
        or pilot_report.get("status") != "PILOT_DIAGNOSIS_REQUIRED"
        or pilot_report.get("gate_passed") is not False
        or lag_ms not in {100, 200, 300}
        or pilot_report.get("video_test_accessed") is not False
    ):
        raise T8Error("causal pilot diagnosis contract is invalid")
    selected_lag = lag_ms
    train_x, train_y, manifest_sha = _load_causal_video_split(root, "train", selected_lag)
    dev_x, dev_y, dev_sha = _load_causal_video_split(root, "dev", selected_lag)
    if dev_sha != manifest_sha or pilot_report.get("dataset_manifest_sha256") != manifest_sha:
        raise T8Error("causal diagnostic dataset differs from pilot")
    model_path = pilot / "causal-combat-temporal-seed0.safetensors"
    metadata = _model_metadata(model_path)
    if (
        metadata.get("schema") != CAUSAL_VIDEO_TRAINING_SCHEMA
        or metadata.get("dataset_manifest_sha256") != manifest_sha
        or metadata.get("selected_lag_ms") != str(selected_lag)
    ):
        raise T8Error("causal pilot model metadata is invalid")
    cross_model = _VideoCausalTemporal().to(target)
    cross_model.load_state_dict(load_file(model_path, device=str(target)), strict=True)
    cross_session = {
        "train_metrics": _head_metrics(
            _predict_causal_temporal(cross_model, train_x, target, batch_size), train_y, 4
        ),
        "dev_metrics": _head_metrics(
            _predict_causal_temporal(cross_model, dev_x, target, batch_size), dev_y, 4
        ),
    }
    binary = _causal_probe_bundle(
        train_x,
        (train_y > 0).astype(np.int64),
        dev_x,
        (dev_y > 0).astype(np.int64),
        2,
        target,
        batch_size,
    )
    train_action = train_y > 0
    dev_action = dev_y > 0
    action_only = _causal_probe_bundle(
        train_x[train_action],
        train_y[train_action] - 1,
        dev_x[dev_action],
        dev_y[dev_action] - 1,
        3,
        target,
        batch_size,
    )
    mixed_train, mixed_dev = _stratified_row_holdout(train_y)
    mixed_session = _causal_probe_bundle(
        train_x[mixed_train],
        train_y[mixed_train],
        train_x[mixed_dev],
        train_y[mixed_dev],
        4,
        target,
        batch_size,
    )
    binary_normal = cast(
        dict[str, object], cast(dict[str, object], binary["normal"])["metrics"]
    )
    action_normal = cast(
        dict[str, object], cast(dict[str, object], action_only["normal"])["metrics"]
    )
    mixed_normal = cast(
        dict[str, object], cast(dict[str, object], mixed_session["normal"])["metrics"]
    )
    cross_dev = cross_session["dev_metrics"]
    binary_signal = bool(
        cast(float, binary_normal["macro_f1"]) >= 0.55
        and cast(float, binary["normal_minus_shuffled_macro_f1"]) >= 0.15
        and cast(float, binary["normal_minus_static_macro_f1"]) >= 0.10
    )
    action_recalls = cast(list[float], action_normal["per_class_recall"])
    action_signal = bool(
        cast(float, action_normal["macro_f1"]) >= 0.45
        and min(action_recalls) >= 0.25
        and cast(float, action_only["normal_minus_shuffled_macro_f1"]) >= 0.15
        and cast(float, action_only["normal_minus_static_macro_f1"]) >= 0.10
    )
    train_fit = bool(
        cast(
            float,
            cross_session["train_metrics"]["macro_f1"],
        )
        >= 0.60
    )
    domain_shift = bool(
        cast(float, mixed_normal["macro_f1"]) >= 0.45
        and cast(float, mixed_normal["macro_f1"]) - cast(float, cross_dev["macro_f1"])
        >= 0.15
    )
    conditional_32_frame_allowed = binary_signal and action_signal
    if conditional_32_frame_allowed:
        diagnosis = "CONDITIONAL_MODEL_SIGNAL_FOUND"
    elif domain_shift:
        diagnosis = "SESSION_DOMAIN_SHIFT"
    elif train_fit:
        diagnosis = "CROSS_SESSION_GENERALIZATION_FAILURE"
    else:
        diagnosis = "NO_CAUSAL_LABEL_SIGNAL_IN_FROZEN_FEATURES"
    report: dict[str, object] = {
        "schema_version": CAUSAL_VIDEO_DIAGNOSTIC_SCHEMA,
        "status": "DIAGNOSIS_COMPLETED",
        "diagnosis": diagnosis,
        "selected_lag_ms": selected_lag,
        "dataset_manifest_sha256": manifest_sha,
        "pilot_report_sha256": _sha(pilot / "report.json"),
        "pilot_model_sha256": _sha(model_path),
        "cross_session_four_class": cross_session,
        "binary_action_vs_wait": binary,
        "action_only_three_class": action_only,
        "mixed_session_row_holdout_four_class": mixed_session,
        "gates": {
            "train_fit": train_fit,
            "binary_signal": binary_signal,
            "action_signal": action_signal,
            "domain_shift": domain_shift,
            "conditional_32_frame_allowed": conditional_32_frame_allowed,
        },
        "video_test_accessed": False,
        "formal_policy_training_allowed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _resize_pixel_crop(
    frame: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 128):
        raise T8Error("causal pixel crop is invalid")
    xs = np.linspace(x0, x1 - 1, 128, dtype=np.int64)
    ys = np.linspace(y0, y1 - 1, 128, dtype=np.int64)
    return frame[ys[:, None], xs[None, :]].astype(np.uint8, copy=False)


def _causal_pixel_views(
    frame: np.ndarray, content_box: tuple[int, int, int, int]
) -> np.ndarray:
    if frame.shape != (128, 128, 3) or frame.dtype != np.uint8:
        raise T8Error("causal pixel frame is invalid")
    x0, y0, x1, y1 = content_box
    width, height = x1 - x0, y1 - y0
    gameplay = (x0, y0, max(x0 + 1, x0 + round(width * 0.72)), y1)
    hud = (
        min(x1 - 1, x0 + round(width * 0.52)),
        min(y1 - 1, y0 + round(height * 0.30)),
        x1,
        y1,
    )
    return np.stack(
        (frame, _resize_pixel_crop(frame, gameplay), _resize_pixel_crop(frame, hud))
    )


def _matched_causal_pixel_rows(
    timestamps: np.ndarray,
    frame_hashes: np.ndarray,
    events: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    event_times = np.asarray(
        [cast(int, event.get("timestamp_ms")) for event in events], dtype=np.int64
    )
    if np.any(event_times < 0):
        raise T8Error("causal pixel event timestamp is invalid")
    safe_candidates: list[int] = []
    for end_index in range(len(timestamps)):
        observation_end = int(timestamps[end_index])
        if observation_end < 2000 or observation_end + 100 > int(timestamps[-1]):
            continue
        label_time = observation_end + 100
        if len(event_times) and int(np.min(np.abs(event_times - label_time))) < 1000:
            continue
        safe_candidates.append(end_index)
    unused = set(safe_candidates)
    rows: list[dict[str, object]] = []
    for event in events:
        frame_index = event.get("frame_index")
        event_time = event.get("timestamp_ms")
        combat = event.get("combat")
        confidence = event.get("confidence")
        if (
            not isinstance(frame_index, int)
            or not isinstance(event_time, int)
            or combat not in ABILITIES[1:4]
            or not isinstance(confidence, (int, float))
            or frame_index < 0
            or frame_index >= len(timestamps)
            or int(timestamps[frame_index]) != event_time
            or str(frame_hashes[frame_index]) != event.get("frame_sha256")
        ):
            raise T8Error("causal pixel event does not bind to target frame")
        end_index = int(np.searchsorted(timestamps, event_time - 100, side="right") - 1)
        shift_index = int(
            np.searchsorted(timestamps, int(timestamps[end_index]) - 2000, side="right") - 1
        )
        if end_index < 0 or shift_index < 0 or not unused:
            continue
        wait_index = min(
            unused,
            key=lambda index: (
                abs(int(timestamps[index]) + 100 - event_time),
                int(timestamps[index]),
            ),
        )
        wait_shift = int(
            np.searchsorted(timestamps, int(timestamps[wait_index]) - 2000, side="right") - 1
        )
        if wait_shift < 0:
            continue
        unused.remove(wait_index)
        rows.extend(
            (
                {
                    "end_index": end_index,
                    "shift_index": shift_index,
                    "combat_id": ABILITIES.index(combat),
                    "label_timestamp_ms": event_time,
                    "event_timestamp_ms": event_time,
                    "observation_end_timestamp_ms": int(timestamps[end_index]),
                    "frame_sha256": str(frame_hashes[end_index]),
                    "confidence": float(confidence),
                    "label_kind": 1,
                },
                {
                    "end_index": wait_index,
                    "shift_index": wait_shift,
                    "combat_id": 0,
                    "label_timestamp_ms": int(timestamps[wait_index]) + 100,
                    "event_timestamp_ms": -1,
                    "observation_end_timestamp_ms": int(timestamps[wait_index]),
                    "frame_sha256": str(frame_hashes[wait_index]),
                    "confidence": 1.0,
                    "label_kind": 0,
                },
            )
        )
    rows.sort(
        key=lambda row: (
            cast(int, row["label_timestamp_ms"]),
            cast(int, row["label_kind"]),
            cast(int, row["combat_id"]),
        )
    )
    return tuple(rows)


def materialize_t8_causal_pixel_dataset(
    *,
    target_dir: Path,
    train_events_dir: Path,
    dev_events_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    target_root, target_sha, train_sessions = _retrospective_target_index(target_dir, "train")
    dev_root, dev_target_sha, dev_sessions = _retrospective_target_index(target_dir, "dev")
    if target_root != dev_root or target_sha != dev_target_sha:
        raise T8Error("causal pixel target manifest changed during setup")
    _, train_events, train_batch_sha = _causal_batch_events(
        train_events_dir, "train", target_sha
    )
    _, dev_events, dev_batch_sha = _causal_batch_events(dev_events_dir, "dev", target_sha)
    if set(train_events) != {identity for identity, _ in train_sessions} or set(dev_events) != {
        identity for identity, _ in dev_sessions
    }:
        raise T8Error("causal pixel event sessions differ from target split")
    output = _large_new(output_dir)
    shard_rows: list[dict[str, object]] = []
    counts = {
        split: {name: 0 for name in ABILITIES[:4]} for split in ("train", "dev")
    }
    orientation_counts: dict[str, dict[str, int]] = {
        split: {} for split in ("train", "dev")
    }
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shards_dir = staging / "shards"
        shards_dir.mkdir()
        for split, sessions, event_index in (
            ("train", train_sessions, train_events),
            ("dev", dev_sessions, dev_events),
        ):
            for ordinal, (identity, source_rows) in enumerate(sessions):
                frames, timestamps, frame_hashes = _retrospective_load_session(
                    target_root, split, identity, source_rows
                )
                canonical, orientation, content_box = _retrospective_content_box(frames)
                rows = _matched_causal_pixel_rows(
                    timestamps, frame_hashes, event_index[identity]
                )
                if not rows:
                    continue
                views = np.stack(
                    [
                        _causal_pixel_views(canonical[cast(int, row["end_index"])], content_box)
                        for row in rows
                    ]
                )
                shifted_views = np.stack(
                    [
                        _causal_pixel_views(
                            canonical[cast(int, row["shift_index"])], content_box
                        )
                        for row in rows
                    ]
                )
                labels = np.asarray(
                    [cast(int, row["combat_id"]) for row in rows], dtype=np.int8
                )
                name = f"{split}-{ordinal:04d}.npz"
                path = shards_dir / name
                np.savez_compressed(
                    path,
                    views=views,
                    shifted_views=shifted_views,
                    combat_id=labels,
                    label_timestamp_ms=np.asarray(
                        [cast(int, row["label_timestamp_ms"]) for row in rows], dtype=np.int64
                    ),
                    event_timestamp_ms=np.asarray(
                        [cast(int, row["event_timestamp_ms"]) for row in rows], dtype=np.int64
                    ),
                    observation_end_timestamp_ms=np.asarray(
                        [
                            cast(int, row["observation_end_timestamp_ms"])
                            for row in rows
                        ],
                        dtype=np.int64,
                    ),
                    frame_sha256=np.asarray(
                        [cast(str, row["frame_sha256"]) for row in rows]
                    ),
                    confidence=np.asarray(
                        [cast(float, row["confidence"]) for row in rows], dtype=np.float32
                    ),
                    label_kind=np.asarray(
                        [cast(int, row["label_kind"]) for row in rows], dtype=np.int8
                    ),
                    session_hash=np.asarray([identity] * len(rows)),
                )
                class_counts = {
                    name: int(np.sum(labels == class_id))
                    for class_id, name in enumerate(ABILITIES[:4])
                }
                for action, count in class_counts.items():
                    counts[split][action] += count
                orientation_counts[split][orientation] = (
                    orientation_counts[split].get(orientation, 0) + 1
                )
                shard_rows.append(
                    {
                        "path": name,
                        "sha256": _sha(path),
                        "split": split,
                        "session_hash": identity,
                        "rows": len(rows),
                        "class_counts": class_counts,
                        "orientation": orientation,
                    }
                )
        manifest: dict[str, object] = {
            "schema_version": CAUSAL_PIXEL_DATASET_SCHEMA,
            "status": "COMPLETED",
            "task": "strict_causal_raw_pixel_learnability_probe",
            "observation_end_lag_ms": 100,
            "time_shift_control_ms": 2000,
            "views": ["full", "gameplay_left_72_percent", "combat_hud"],
            "view_shape": [3, 128, 128, 3],
            "combat_vocabulary": list(ABILITIES[:4]),
            "wait_rule": "same_session_nearest_unique_at_least_1000ms_from_event",
            "counts": counts,
            "orientation_counts": orientation_counts,
            "target_manifest_sha256": target_sha,
            "event_batch_manifest_sha256": {
                "train": train_batch_sha,
                "dev": dev_batch_sha,
            },
            "shards": shard_rows,
            "event_frame_included": False,
            "future_frames_included": False,
            "video_test_accessed": False,
            "raw_video_or_source_paths_persisted": False,
            "formal_policy_training_allowed": False,
            "shadow_allowed": False,
            "device_input_allowed": False,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return manifest


def _load_causal_pixel_split(
    dataset_root: Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if split not in {"train", "dev"}:
        raise T8Error("causal pixel probe may open train or dev only")
    root = _large_existing(dataset_root)
    manifest = _read_object(root / "manifest.json", "causal pixel manifest is unreadable")
    supplied = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema_version") != CAUSAL_PIXEL_DATASET_SCHEMA
        or manifest.get("status") != "COMPLETED"
        or manifest.get("observation_end_lag_ms") != 100
        or manifest.get("time_shift_control_ms") != 2000
        or manifest.get("combat_vocabulary") != list(ABILITIES[:4])
        or manifest.get("event_frame_included") is not False
        or manifest.get("future_frames_included") is not False
        or manifest.get("video_test_accessed") is not False
        or supplied != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise T8Error("causal pixel dataset contract is invalid")
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise T8Error("causal pixel shard manifest is invalid")
    views: list[np.ndarray] = []
    shifted: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for row in cast(list[dict[str, object]], rows):
        if row.get("split") != split:
            continue
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise T8Error("causal pixel shard name is invalid")
        path = root / "shards" / name
        if _sha(path) != row.get("sha256"):
            raise T8Error("causal pixel shard hash differs")
        with np.load(path, allow_pickle=False) as shard:
            expected = {
                "views",
                "shifted_views",
                "combat_id",
                "label_timestamp_ms",
                "event_timestamp_ms",
                "observation_end_timestamp_ms",
                "frame_sha256",
                "confidence",
                "label_kind",
                "session_hash",
            }
            current, current_shifted = shard["views"], shard["shifted_views"]
            current_labels = shard["combat_id"]
            label_kind = shard["label_kind"]
            observation_end = shard["observation_end_timestamp_ms"]
            event_timestamp = shard["event_timestamp_ms"]
            if (
                set(shard.files) != expected
                or current.dtype != np.uint8
                or current.shape[1:] != (3, 128, 128, 3)
                or current_shifted.dtype != np.uint8
                or current_shifted.shape != current.shape
                or current_labels.dtype != np.int8
                or current_labels.shape != (len(current),)
                or np.any(current_labels < 0)
                or np.any(current_labels >= 4)
                or np.any((label_kind == 1) & (observation_end > event_timestamp - 100))
            ):
                raise T8Error("causal pixel shard tensor contract is invalid")
            views.append(current.copy())
            shifted.append(current_shifted.copy())
            labels.append(current_labels.astype(np.int64))
    if not views:
        raise T8Error("causal pixel split is empty")
    return (
        np.concatenate(views),
        np.concatenate(labels),
        np.concatenate(shifted),
        supplied,
    )


def _pixel_probe_tensor(views: np.ndarray, device: torch.device) -> torch.Tensor:
    if views.ndim != 5 or views.shape[1:] != (3, 128, 128, 3) or views.dtype != np.uint8:
        raise T8Error("causal pixel probe requires Nx3x128x128x3 uint8 views")
    return (
        torch.from_numpy(views)
        .to(device)
        .permute(0, 1, 4, 2, 3)
        .float()
        .div(255.0)
    )


class _CausalPixelProbe(nn.Module):
    def __init__(self, encoder_state: Mapping[str, torch.Tensor], class_count: int) -> None:
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.encoder.load_state_dict(encoder_state, strict=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder.layer4.parameters():
            parameter.requires_grad = True
        self.head = nn.Sequential(nn.Linear(512 * 3, 256), nn.ReLU(), nn.Linear(256, class_count))

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 5 or views.shape[1:] != (3, 3, 128, 128):
            raise T8Error("causal pixel probe tensor is invalid")
        encoded = cast(
            torch.Tensor,
            self.encoder(views.reshape(len(views) * 3, 3, 128, 128)),
        ).reshape(len(views), 3 * 512)
        return cast(torch.Tensor, self.head(encoded))


def _balanced_probe_order(labels: np.ndarray, epoch: int, class_count: int) -> np.ndarray:
    randomizer = np.random.default_rng(epoch)
    class_rows = [np.flatnonzero(labels == class_id) for class_id in range(class_count)]
    if any(not len(rows) for rows in class_rows):
        raise T8Error("causal pixel probe train split lacks a class")
    selected_classes = randomizer.integers(0, class_count, size=len(labels))
    order = np.empty(len(labels), dtype=np.int64)
    for class_id, rows in enumerate(class_rows):
        positions = np.flatnonzero(selected_classes == class_id)
        order[positions] = randomizer.choice(rows, size=len(positions), replace=True)
    return order


def _predict_pixel_probe(
    model: _CausalPixelProbe,
    views: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predicted: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(views), batch_size):
            predicted.append(
                model(_pixel_probe_tensor(views[start : start + batch_size], device))
                .argmax(1)
                .cpu()
                .numpy()
            )
    return np.concatenate(predicted)


def _fit_causal_pixel_probe(
    train_views: np.ndarray,
    train_y: np.ndarray,
    dev_views: np.ndarray,
    dev_y: np.ndarray,
    encoder_state: Mapping[str, torch.Tensor],
    class_count: int,
    device: torch.device,
    batch_size: int,
    *,
    shuffled: bool,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    labels = train_y.copy()
    if shuffled:
        labels = labels[np.random.default_rng(0).permutation(len(labels))]
    torch.manual_seed(0)
    model = _CausalPixelProbe(encoder_state, class_count).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
        weight_decay=1e-4,
    )
    best_loss, best_epoch, best_state = math.inf, 0, {}
    for epoch in range(1, 7):
        model.train()
        for name, module in model.encoder.named_modules():
            if isinstance(module, nn.BatchNorm2d) and not name.startswith("layer4"):
                module.eval()
        order = _balanced_probe_order(labels, epoch, class_count)
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(_pixel_probe_tensor(train_views[selected], device)),
                torch.from_numpy(labels[selected]).to(device),
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        validation_loss, validation_rows = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(dev_views), batch_size):
                current_y = torch.from_numpy(dev_y[start : start + batch_size]).to(device)
                current_loss = nn.functional.cross_entropy(
                    model(
                        _pixel_probe_tensor(dev_views[start : start + batch_size], device)
                    ),
                    current_y,
                    reduction="sum",
                )
                validation_loss += float(current_loss.item())
                validation_rows += len(current_y)
        current = validation_loss / validation_rows
        if current < best_loss:
            best_loss, best_epoch = current, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if not best_state:
        raise T8Error("causal pixel probe did not validate")
    model.load_state_dict(best_state, strict=True)
    return (
        {
            "best_epoch": best_epoch,
            "best_dev_cross_entropy": best_loss,
            "metrics": _head_metrics(
                _predict_pixel_probe(model, dev_views, device, batch_size), dev_y, class_count
            ),
            "train_metrics": _head_metrics(
                _predict_pixel_probe(model, train_views, device, batch_size),
                train_y,
                class_count,
            ),
            "shuffled_labels": shuffled,
        },
        best_state,
    )


def _flat_pixel_views(views: np.ndarray) -> np.ndarray:
    means = np.rint(views.astype(np.float32).mean(axis=(2, 3), keepdims=True)).astype(np.uint8)
    return np.broadcast_to(means, views.shape).copy()


def _causal_pixel_probe_bundle(
    train_views: np.ndarray,
    train_y: np.ndarray,
    dev_views: np.ndarray,
    dev_y: np.ndarray,
    shifted_dev_views: np.ndarray,
    encoder_state: Mapping[str, torch.Tensor],
    class_count: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    normal, state = _fit_causal_pixel_probe(
        train_views,
        train_y,
        dev_views,
        dev_y,
        encoder_state,
        class_count,
        device,
        batch_size,
        shuffled=False,
    )
    shuffled, _ = _fit_causal_pixel_probe(
        train_views,
        train_y,
        dev_views,
        dev_y,
        encoder_state,
        class_count,
        device,
        batch_size,
        shuffled=True,
    )
    model = _CausalPixelProbe(encoder_state, class_count).to(device)
    model.load_state_dict(state, strict=True)
    flat_metrics = _head_metrics(
        _predict_pixel_probe(model, _flat_pixel_views(dev_views), device, batch_size),
        dev_y,
        class_count,
    )
    shifted_metrics = _head_metrics(
        _predict_pixel_probe(model, shifted_dev_views, device, batch_size),
        dev_y,
        class_count,
    )
    normal_metrics = cast(dict[str, object], normal["metrics"])
    normal_f1 = cast(float, normal_metrics["macro_f1"])
    return (
        {
            "normal": normal,
            "shuffled": shuffled,
            "flat_frame": flat_metrics,
            "time_shift_2000ms": shifted_metrics,
            "normal_minus_shuffled_macro_f1": normal_f1
            - cast(float, cast(dict[str, object], shuffled["metrics"])["macro_f1"]),
            "normal_minus_flat_macro_f1": normal_f1 - cast(float, flat_metrics["macro_f1"]),
            "normal_minus_shifted_macro_f1": normal_f1
            - cast(float, shifted_metrics["macro_f1"]),
        },
        state,
    )


def run_t8_causal_pixel_probe(
    *,
    dataset_root: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 64,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise T8Error("causal pixel probe settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    target = torch.device(device)
    root = _large_existing(dataset_root)
    train_views, train_y, _, manifest_sha = _load_causal_pixel_split(root, "train")
    dev_views, dev_y, shifted_dev, dev_sha = _load_causal_pixel_split(root, "dev")
    if dev_sha != manifest_sha:
        raise T8Error("causal pixel manifest changed during training")
    manifest = _read_object(root / "manifest.json", "causal pixel manifest is unreadable")
    adapter = _large_existing(adapter_checkpoint)
    encoder_state, adapter_meta = _load_v2_adapter(adapter, target)
    if adapter_meta.get("target_manifest_sha256") != manifest.get("target_manifest_sha256"):
        raise T8Error("causal pixel adapter differs from dataset target")
    binary, binary_state = _causal_pixel_probe_bundle(
        train_views,
        (train_y > 0).astype(np.int64),
        dev_views,
        (dev_y > 0).astype(np.int64),
        shifted_dev,
        encoder_state,
        2,
        target,
        batch_size,
    )
    train_action, dev_action = train_y > 0, dev_y > 0
    action, action_state = _causal_pixel_probe_bundle(
        train_views[train_action],
        train_y[train_action] - 1,
        dev_views[dev_action],
        dev_y[dev_action] - 1,
        shifted_dev[dev_action],
        encoder_state,
        3,
        target,
        batch_size,
    )
    binary_metrics = cast(
        dict[str, object], cast(dict[str, object], binary["normal"])["metrics"]
    )
    action_metrics = cast(
        dict[str, object], cast(dict[str, object], action["normal"])["metrics"]
    )
    binary_gate = bool(
        cast(float, binary_metrics["macro_f1"]) >= 0.60
        and cast(float, binary["normal_minus_shuffled_macro_f1"]) >= 0.15
        and cast(float, binary["normal_minus_flat_macro_f1"]) >= 0.15
        and cast(float, binary["normal_minus_shifted_macro_f1"]) >= 0.10
    )
    action_gate = bool(
        cast(float, action_metrics["macro_f1"]) >= 0.45
        and min(cast(list[float], action_metrics["per_class_recall"])) >= 0.25
        and cast(float, action["normal_minus_shuffled_macro_f1"]) >= 0.15
        and cast(float, action["normal_minus_flat_macro_f1"]) >= 0.15
        and cast(float, action["normal_minus_shifted_macro_f1"]) >= 0.10
    )
    conditional_allowed = binary_gate and action_gate
    report: dict[str, object] = {
        "schema_version": CAUSAL_PIXEL_PROBE_SCHEMA,
        "status": "PIXEL_SIGNAL_FOUND" if conditional_allowed else "VISUAL_TEACHER_REQUIRED",
        "dataset_manifest_sha256": manifest_sha,
        "adapter_sha256": _sha(adapter),
        "encoder_trainable_scope": "resnet18_layer4_plus_probe_head",
        "seed": 0,
        "epochs": 6,
        "binary_action_vs_wait": binary,
        "action_only_three_class": action,
        "gates": {
            "binary_gate": binary_gate,
            "action_gate": action_gate,
            "conditional_model_allowed": conditional_allowed,
        },
        "video_test_accessed": False,
        "formal_policy_training_allowed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        save_file(binary_state, staging / "binary-pixel-probe.safetensors")
        save_file(action_state, staging / "action-pixel-probe.safetensors")
        report["binary_model_sha256"] = _sha(staging / "binary-pixel-probe.safetensors")
        report["action_model_sha256"] = _sha(staging / "action-pixel-probe.safetensors")
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _visual_teacher_features(
    views: np.ndarray, history_views: np.ndarray, layout_points: Sequence[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    if views.shape != history_views.shape or views.shape[1:] != (3, 128, 128, 3):
        raise T8Error("visual teacher replay views are invalid")
    activity = np.mean(
        np.abs(views[:, 1].astype(np.float32) - history_views[:, 1].astype(np.float32)),
        axis=(1, 2, 3),
    ) / 255.0
    hud = views[:, 2].astype(np.float32) / 255.0
    scores = np.zeros((len(views), len(layout_points)), dtype=np.float32)
    for index, (x, y) in enumerate(layout_points):
        center_x = round((x - 0.52) / 0.48 * 127)
        center_y = round((y - 0.30) / 0.70 * 127)
        x0, x1 = max(0, center_x - 6), min(128, center_x + 7)
        y0, y1 = max(0, center_y - 5), min(128, center_y + 6)
        if x0 >= x1 or y0 >= y1:
            raise T8Error("visual teacher button ROI is invalid")
        patch = hud[:, y0:y1, x0:x1]
        maximum, minimum = patch.max(axis=3), patch.min(axis=3)
        scores[:, index] = 0.55 * maximum.mean(axis=(1, 2)) + 0.45 * (
            maximum - minimum
        ).mean(axis=(1, 2))
    return activity.astype(np.float32), scores


def _visual_teacher_predict(
    activity: np.ndarray,
    scores: np.ndarray,
    activity_threshold: float,
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[1] != 3 or np.any(scales <= 0):
        raise T8Error("visual teacher calibration is invalid")
    normalized = (scores - medians) / scales
    actions = normalized.argmax(axis=1).astype(np.int64) + 1
    actions[activity < activity_threshold] = 0
    return cast(np.ndarray, actions)


def run_t8_visual_teacher_replay(
    *,
    dataset_root: Path,
    pixel_probe_dir: Path,
    layout_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    root = _large_existing(dataset_root)
    probe = _large_existing(pixel_probe_dir)
    probe_report = _read_object(probe / "report.json", "causal pixel report is unreadable")
    if (
        probe_report.get("schema_version") != CAUSAL_PIXEL_PROBE_SCHEMA
        or probe_report.get("status") != "VISUAL_TEACHER_REQUIRED"
        or probe_report.get("video_test_accessed") is not False
        or cast(dict[str, object], probe_report.get("gates")).get(
            "conditional_model_allowed"
        )
        is not False
    ):
        raise T8Error("visual teacher replay requires a failed causal pixel probe")
    layout, layout_sha = load_layout(layout_path)
    points: list[tuple[float, float]] = []
    for name in ABILITIES[1:4]:
        point = layout.buttons.get(name)
        if point is None:
            raise T8Error("visual teacher layout lacks a combat button")
        points.append(point)
    train_views, train_labels, train_history, manifest_sha = _load_causal_pixel_split(
        root, "train"
    )
    dev_views, dev_labels, dev_history, dev_sha = _load_causal_pixel_split(root, "dev")
    if dev_sha != manifest_sha or probe_report.get("dataset_manifest_sha256") != manifest_sha:
        raise T8Error("visual teacher replay dataset differs from pixel probe")
    train_activity, train_scores = _visual_teacher_features(train_views, train_history, points)
    dev_activity, dev_scores = _visual_teacher_features(dev_views, dev_history, points)
    history_activity, history_scores = _visual_teacher_features(
        dev_history, dev_views, points
    )
    activity_threshold = float(np.quantile(train_activity, 0.50))
    medians = np.median(train_scores, axis=0).astype(np.float32)
    lower = np.quantile(train_scores, 0.25, axis=0)
    upper = np.quantile(train_scores, 0.75, axis=0)
    scales = np.maximum((upper - lower).astype(np.float32), 1e-4)
    train_actions = _visual_teacher_predict(
        train_activity, train_scores, activity_threshold, medians, scales
    )
    dev_actions = _visual_teacher_predict(
        dev_activity, dev_scores, activity_threshold, medians, scales
    )
    repeated = _visual_teacher_predict(
        dev_activity, dev_scores, activity_threshold, medians, scales
    )
    history_actions = _visual_teacher_predict(
        history_activity, history_scores, activity_threshold, medians, scales
    )

    def counts(values: np.ndarray) -> dict[str, int]:
        return {
            name: int(np.sum(values == index)) for index, name in enumerate(ABILITIES[:4])
        }

    dev_counts = counts(dev_actions)
    dev_total = max(len(dev_actions), 1)
    fractions = {name: value / dev_total for name, value in dev_counts.items()}
    coverage_gate = bool(
        0.30 <= fractions["none"] <= 0.70
        and min(fractions[name] for name in ABILITIES[1:4]) >= 0.05
    )
    deterministic = bool(np.array_equal(dev_actions, repeated))
    history_agreement = float(np.mean(dev_actions == history_actions))
    temporal_gate = history_agreement <= 0.90
    ready = coverage_gate and deterministic and temporal_gate
    report: dict[str, object] = {
        "schema_version": VISUAL_TEACHER_REPLAY_SCHEMA,
        "status": "OFFLINE_TEACHER_READY" if ready else "TEACHER_RULE_DIAGNOSIS_REQUIRED",
        "decision_source": "rgb_activity_plus_normalized_combat_button_appearance",
        "history_ms": 2000,
        "calibration_uses": "video_train_only",
        "validation_uses": "video_dev_only",
        "dataset_manifest_sha256": manifest_sha,
        "pixel_probe_report_sha256": _sha(probe / "report.json"),
        "layout_sha256": layout_sha,
        "activity_threshold": activity_threshold,
        "button_score_medians": medians.tolist(),
        "button_score_iqr": scales.tolist(),
        "combat_vocabulary": list(ABILITIES[:4]),
        "train_decision_counts": counts(train_actions),
        "dev_decision_counts": dev_counts,
        "dev_decision_fractions": fractions,
        "dev_history_action_agreement": history_agreement,
        "deterministic_repeat": deterministic,
        "coverage_gate": coverage_gate,
        "temporal_gate": temporal_gate,
        "offline_teacher_ready": ready,
        "agreement_with_retrospective_labels": {
            "train": float(np.mean(train_actions == train_labels)),
            "dev": float(np.mean(dev_actions == dev_labels)),
            "not_a_promotion_metric": True,
        },
        "training_eligible": False,
        "live_execution_allowed": False,
        "video_test_accessed": False,
        "formal_policy_training_allowed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _visible_onset_traces(
    frames: np.ndarray,
    content_box: tuple[int, int, int, int],
    centers: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frames.ndim != 4 or frames.shape[1:] != (128, 128, 3) or len(centers) != 4:
        raise T8Error("visible-onset session tensors are invalid")
    x0, y0, x1, y1 = content_box
    content = frames[:, y0:y1, x0:x1].astype(np.int16)
    content_change = np.zeros(len(frames), dtype=np.float32)
    content_change[1:] = np.abs(content[1:] - content[:-1]).mean(axis=(1, 2, 3))
    absolute = np.zeros((len(frames), len(centers)), dtype=np.float32)
    positive = np.zeros_like(absolute)
    for button, (center_x, center_y) in enumerate(centers):
        crop = frames[
            :,
            max(0, center_y - 4) : min(128, center_y + 5),
            max(0, center_x - 6) : min(128, center_x + 7),
        ].astype(np.int16)
        delta = crop[1:] - crop[:-1]
        absolute[1:, button] = np.abs(delta).mean(axis=(1, 2, 3))
        positive[1:, button] = np.maximum(delta, 0).mean(axis=(1, 2, 3))
    return content_change, absolute, positive


def _visible_onset_event(
    event: Mapping[str, object],
    timestamps: np.ndarray,
    frame_hashes: np.ndarray,
    content_change: np.ndarray,
    absolute: np.ndarray,
    positive: np.ndarray,
) -> dict[str, object] | None:
    peak_index = event.get("frame_index")
    combat = event.get("combat")
    if (
        not isinstance(peak_index, int)
        or combat not in ABILITIES[1:4]
        or peak_index < 3
        or peak_index >= len(timestamps)
        or int(timestamps[peak_index]) != event.get("timestamp_ms")
        or str(frame_hashes[peak_index]) != event.get("frame_sha256")
    ):
        raise T8Error("visible-onset source event does not bind to RGB")
    button = ABILITIES.index(combat) - 1
    peak_absolute = float(absolute[peak_index, button])
    peak_positive = float(positive[peak_index, button])
    if peak_absolute < 18.0:
        return None
    onset_index = peak_index
    for candidate in range(peak_index - 3, peak_index + 1):
        ratio = float(absolute[candidate, button] / max(content_change[candidate], 1e-6))
        if (
            float(absolute[candidate, button]) >= max(18.0, 0.35 * peak_absolute)
            and float(positive[candidate, button]) >= max(3.0, 0.20 * peak_positive)
            and ratio >= 1.25
        ):
            onset_index = candidate
            break
    preceding = absolute[max(1, onset_index - 2) : onset_index, button]
    pre_quiet = not len(preceding) or float(np.max(preceding)) < 0.35 * peak_absolute
    correct_ratio = float(
        absolute[onset_index, button] / max(content_change[onset_index], 1e-6)
    )
    other = np.delete(absolute[onset_index], button)
    cross_button_ratio = float(np.max(other, initial=0.0) / max(peak_absolute, 1e-6))
    if not pre_quiet or cross_button_ratio >= 0.80:
        return None
    confidence = min(
        1.0,
        max(
            0.0,
            min(
                float(absolute[onset_index, button]) / max(peak_absolute, 1e-6),
                correct_ratio / 1.25,
                1.0 - cross_button_ratio,
            ),
        ),
    )
    return {
        "schema_version": VISIBLE_ONSET_EVENT_SCHEMA,
        "sequence": 0,
        "session_hash": event["session_hash"],
        "split": event["split"],
        "combat": combat,
        "visible_onset_timestamp_ms": int(timestamps[onset_index]),
        "visible_onset_frame_index": onset_index,
        "visible_onset_frame_sha256": str(frame_hashes[onset_index]),
        "peak_timestamp_ms": int(timestamps[peak_index]),
        "peak_frame_index": peak_index,
        "peak_frame_sha256": str(frame_hashes[peak_index]),
        "onset_offset_frames": onset_index - peak_index,
        "onset_confidence": confidence,
        "class_confidence": float(cast(float, event["confidence"])),
        "peak_roi_absolute": peak_absolute,
        "onset_roi_absolute": float(absolute[onset_index, button]),
        "onset_roi_positive": float(positive[onset_index, button]),
        "onset_to_content_ratio": correct_ratio,
        "cross_button_ratio": cross_button_ratio,
        "source_event_schema": RETROSPECTIVE_EVENT_SCHEMA,
        "source_baseline_sha256": event["baseline_sha256"],
        "target_manifest_sha256": event["target_manifest_sha256"],
    }


def run_t8_visible_onset_audit(
    *,
    target_dir: Path,
    train_events_dir: Path,
    dev_events_dir: Path,
    layout_path: Path,
    calibration_report_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    target, target_sha, train_sessions = _retrospective_target_index(target_dir, "train")
    dev_target, dev_target_sha, dev_sessions = _retrospective_target_index(target_dir, "dev")
    if target != dev_target or target_sha != dev_target_sha:
        raise T8Error("visible-onset target changed between splits")
    _, train_events, train_batch_sha = _causal_batch_events(
        train_events_dir, "train", target_sha
    )
    _, dev_events, dev_batch_sha = _causal_batch_events(dev_events_dir, "dev", target_sha)
    calibration_path = _large_existing(calibration_report_path)
    calibration = _read_object(
        calibration_path, "visible-onset calibration report is unreadable"
    )
    four_class = calibration.get("four_class")
    if (
        calibration.get("schema_version") != RETROSPECTIVE_CALIBRATION_SCHEMA
        or calibration.get("status") != "BASELINE_RETAINED"
        or calibration.get("test_accessed") is not False
        or not isinstance(four_class, dict)
    ):
        raise T8Error("visible-onset calibration contract is invalid")
    inverse = cast(dict[str, object], four_class).get("inverse_holdout")
    if not isinstance(inverse, dict) or not isinstance(inverse.get("per_class_precision"), list):
        raise T8Error("visible-onset inverse holdout is unavailable")
    precision_values = cast(list[object], inverse["per_class_precision"])
    if any(not isinstance(value, (int, float)) for value in precision_values):
        raise T8Error("visible-onset inverse precision is invalid")
    inverse_precision = [float(cast(float, value)) for value in precision_values]
    inverse_gate = len(inverse_precision) >= 3 and min(inverse_precision[:3]) >= 0.95
    layout, layout_sha = load_layout(layout_path)
    button_layout: dict[str, tuple[float, float]] = {}
    for name in ABILITIES[1:]:
        point = layout.buttons.get(name)
        if point is None:
            raise T8Error("visible-onset layout lacks a combat button")
        button_layout[name] = point
    output = _large_new(output_dir)
    aggregate: dict[str, dict[str, object]] = {}
    split_payload = (
        ("train", train_sessions, train_events),
        ("dev", dev_sessions, dev_events),
    )
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for split, sessions, event_index in split_payload:
            event_output = staging / split / "events"
            event_output.mkdir(parents=True)
            source_counts = {name: 0 for name in ABILITIES[1:4]}
            retained_counts = {name: 0 for name in ABILITIES[1:4]}
            offsets = {str(value): 0 for value in range(-3, 1)}
            rejected = 0
            session_rows: list[dict[str, object]] = []
            for identity, rows in sessions:
                frames, timestamps, frame_hashes = _retrospective_load_session(
                    target, split, identity, rows
                )
                canonical, orientation, content_box = _retrospective_content_box(frames)
                centers = _retrospective_centers(content_box, button_layout)
                content_change, absolute, positive = _visible_onset_traces(
                    canonical, content_box, centers
                )
                accepted: list[dict[str, object]] = []
                for event in event_index[identity]:
                    combat = cast(str, event["combat"])
                    source_counts[combat] += 1
                    current = _visible_onset_event(
                        event,
                        timestamps,
                        frame_hashes,
                        content_change,
                        absolute,
                        positive,
                    )
                    if current is None:
                        rejected += 1
                        continue
                    current["sequence"] = len(accepted)
                    accepted.append(current)
                    retained_counts[combat] += 1
                    offsets[str(current["onset_offset_frames"])] += 1
                path = event_output / f"{identity}.jsonl"
                path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted),
                    encoding="utf-8",
                )
                session_rows.append(
                    {
                        "session_hash": identity,
                        "orientation": orientation,
                        "source_events": len(event_index[identity]),
                        "retained_events": len(accepted),
                        "events_sha256": _sha(path),
                    }
                )
            source_total = sum(source_counts.values())
            retained_total = sum(retained_counts.values())
            fractions = {
                name: retained_counts[name] / max(source_counts[name], 1)
                for name in ABILITIES[1:4]
            }
            aggregate[split] = {
                "session_count": len(session_rows),
                "source_event_count": source_total,
                "retained_event_count": retained_total,
                "retained_fraction": retained_total / max(source_total, 1),
                "ambiguity_fraction": rejected / max(source_total, 1),
                "source_counts": source_counts,
                "retained_counts": retained_counts,
                "per_class_retained_fraction": fractions,
                "onset_offset_counts": offsets,
                "sessions": session_rows,
            }
        dev = aggregate["dev"]
        dev_retained = cast(dict[str, float], dev["per_class_retained_fraction"])
        coverage_gate = bool(
            cast(float, dev["retained_fraction"]) >= 0.75
            and min(dev_retained.values()) >= 0.60
            and cast(float, dev["ambiguity_fraction"]) <= 0.10
        )
        passed = coverage_gate and inverse_gate
        report: dict[str, object] = {
            "schema_version": VISIBLE_ONSET_AUDIT_SCHEMA,
            "status": "VISIBLE_ONSET_AUDIT_PASSED" if passed else "VISIBLE_ONSET_AUDIT_FAILED",
            "search_offsets_frames": [-3, -2, -1, 0],
            "frame_interval_ms": 100,
            "thresholds": {
                "minimum_roi_absolute": 18.0,
                "minimum_peak_fraction": 0.35,
                "minimum_positive": 3.0,
                "minimum_positive_peak_fraction": 0.20,
                "minimum_roi_to_content_ratio": 1.25,
                "maximum_cross_button_ratio": 0.80,
                "quiet_peak_fraction": 0.35,
            },
            "target_manifest_sha256": target_sha,
            "train_event_batch_sha256": train_batch_sha,
            "dev_event_batch_sha256": dev_batch_sha,
            "layout_sha256": layout_sha,
            "calibration_report_sha256": _sha(calibration_path),
            "inverse_holdout_per_class_precision": inverse_precision,
            "inverse_precision_gate": inverse_gate,
            "coverage_gate": coverage_gate,
            "splits": aggregate,
            "training_eligible": passed,
            "combat_subpolicy_only": True,
            "video_test_accessed": False,
            "formal_policy_training_allowed": False,
            "shadow_allowed": False,
            "device_input_allowed": False,
        }
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _visible_onset_events(
    audit_dir: Path, split: str, *, diagnostic_only: bool = False
) -> tuple[Path, dict[str, tuple[dict[str, object], ...]], str]:
    if split not in {"train", "dev"}:
        raise T8Error("visible-onset audit may open train or dev only")
    root = _large_existing(audit_dir)
    report = _read_object(root / "report.json", "visible-onset report is unreadable")
    admitted_status = (
        report.get("status") == "VISIBLE_ONSET_AUDIT_FAILED"
        and report.get("training_eligible") is False
        if diagnostic_only
        else report.get("status") == "VISIBLE_ONSET_AUDIT_PASSED"
        and report.get("training_eligible") is True
    )
    if (
        report.get("schema_version") != VISIBLE_ONSET_AUDIT_SCHEMA
        or not admitted_status
        or report.get("video_test_accessed") is not False
    ):
        raise T8Error("visible-onset audit did not admit causal materialization")
    splits = report.get("splits")
    current = splits.get(split) if isinstance(splits, dict) else None
    sessions = current.get("sessions") if isinstance(current, dict) else None
    if not isinstance(sessions, list):
        raise T8Error("visible-onset audit session index is invalid")
    indexed: dict[str, tuple[dict[str, object], ...]] = {}
    for session in cast(list[dict[str, object]], sessions):
        identity = session.get("session_hash")
        if not isinstance(identity, str):
            raise T8Error("visible-onset session identity is invalid")
        path = root / split / "events" / f"{identity}.jsonl"
        if _sha(path) != session.get("events_sha256"):
            raise T8Error("visible-onset event hash differs")
        try:
            events = tuple(
                cast(dict[str, object], json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise T8Error("visible-onset event stream is unreadable") from exc
        if any(
            event.get("schema_version") != VISIBLE_ONSET_EVENT_SCHEMA
            or event.get("sequence") != sequence
            or event.get("session_hash") != identity
            or event.get("split") != split
            for sequence, event in enumerate(events)
        ):
            raise T8Error("visible-onset event contract is invalid")
        indexed[identity] = events
    return root, indexed, _sha(root / "report.json")


def _combat_causal_rows(
    timestamps: np.ndarray,
    frame_hashes: np.ndarray,
    events: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    event_times = np.asarray(
        [cast(int, event["visible_onset_timestamp_ms"]) for event in events], dtype=np.int64
    )
    unused: set[int] = set()
    for end_index in range(31, len(timestamps)):
        label_time = int(timestamps[end_index]) + 100
        shift_end = int(
            np.searchsorted(timestamps, int(timestamps[end_index]) - 2000, side="right") - 1
        )
        if shift_end < 31:
            continue
        if len(event_times) and int(np.min(np.abs(event_times - label_time))) < 1000:
            continue
        unused.add(end_index)
    rows: list[dict[str, object]] = []
    for event in events:
        onset_index = event.get("visible_onset_frame_index")
        onset_time = event.get("visible_onset_timestamp_ms")
        onset_hash = event.get("visible_onset_frame_sha256")
        combat = event.get("combat")
        if (
            not isinstance(onset_index, int)
            or not isinstance(onset_time, int)
            or combat not in ABILITIES[1:4]
            or onset_index < 0
            or onset_index >= len(timestamps)
            or int(timestamps[onset_index]) != onset_time
            or str(frame_hashes[onset_index]) != onset_hash
        ):
            raise T8Error("combat-causal onset does not bind to target RGB")
        end_index = int(np.searchsorted(timestamps, onset_time - 100, side="right") - 1)
        shift_end = int(
            np.searchsorted(timestamps, int(timestamps[end_index]) - 2000, side="right") - 1
        )
        if end_index < 31 or shift_end < 31 or not unused:
            continue
        wait_index = min(
            unused,
            key=lambda index: (
                abs(int(timestamps[index]) + 100 - onset_time),
                int(timestamps[index]),
            ),
        )
        unused.remove(wait_index)
        wait_shift = int(
            np.searchsorted(timestamps, int(timestamps[wait_index]) - 2000, side="right") - 1
        )
        rows.extend(
            (
                {
                    "start_index": end_index - 31,
                    "end_index": end_index,
                    "shift_start_index": shift_end - 31,
                    "shift_end_index": shift_end,
                    "combat_id": ABILITIES.index(combat),
                    "label_timestamp_ms": onset_time,
                    "visible_onset_timestamp_ms": onset_time,
                    "observation_end_timestamp_ms": int(timestamps[end_index]),
                    "window_sha256": hashlib.sha256(
                        "".join(map(str, frame_hashes[end_index - 31 : end_index + 1])).encode()
                    ).hexdigest(),
                    "label_kind": 1,
                },
                {
                    "start_index": wait_index - 31,
                    "end_index": wait_index,
                    "shift_start_index": wait_shift - 31,
                    "shift_end_index": wait_shift,
                    "combat_id": 0,
                    "label_timestamp_ms": int(timestamps[wait_index]) + 100,
                    "visible_onset_timestamp_ms": -1,
                    "observation_end_timestamp_ms": int(timestamps[wait_index]),
                    "window_sha256": hashlib.sha256(
                        "".join(map(str, frame_hashes[wait_index - 31 : wait_index + 1])).encode()
                    ).hexdigest(),
                    "label_kind": 0,
                },
            )
        )
    rows.sort(
        key=lambda row: (
            cast(int, row["label_timestamp_ms"]),
            cast(int, row["label_kind"]),
            cast(int, row["combat_id"]),
        )
    )
    return tuple(rows)


def _combat_session_views(
    frames: np.ndarray, content_box: tuple[int, int, int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = content_box
    width, height = x1 - x0, y1 - y0
    hud = (
        min(x1 - 1, x0 + round(width * 0.52)),
        min(y1 - 1, y0 + round(height * 0.30)),
        x1,
        y1,
    )
    xs = np.linspace(hud[0], hud[2] - 1, 128, dtype=np.int64)
    ys = np.linspace(hud[1], hud[3] - 1, 128, dtype=np.int64)
    hud_frames = frames[:, ys[:, None], xs[None, :]].astype(np.uint8, copy=False)
    return np.stack((frames, hud_frames), axis=1)


def materialize_t8_combat_causal_dataset(
    *,
    target_dir: Path,
    onset_audit_dir: Path,
    output_dir: Path,
    diagnostic_only: bool = False,
) -> dict[str, object]:
    target, target_sha, train_sessions = _retrospective_target_index(target_dir, "train")
    dev_target, dev_sha, dev_sessions = _retrospective_target_index(target_dir, "dev")
    if target != dev_target or target_sha != dev_sha:
        raise T8Error("combat-causal target changed between splits")
    _, train_events, audit_sha = _visible_onset_events(
        onset_audit_dir, "train", diagnostic_only=diagnostic_only
    )
    _, dev_events, dev_audit_sha = _visible_onset_events(
        onset_audit_dir, "dev", diagnostic_only=diagnostic_only
    )
    if audit_sha != dev_audit_sha:
        raise T8Error("combat-causal onset audit changed between splits")
    output = _large_new(output_dir)
    shards: list[dict[str, object]] = []
    counts = {split: {name: 0 for name in ABILITIES[:4]} for split in ("train", "dev")}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shard_dir = staging / "shards"
        shard_dir.mkdir()
        for split, sessions, indexed in (
            ("train", train_sessions, train_events),
            ("dev", dev_sessions, dev_events),
        ):
            for ordinal, (identity, source_rows) in enumerate(sessions):
                frames, timestamps, frame_hashes = _retrospective_load_session(
                    target, split, identity, source_rows
                )
                canonical, orientation, content_box = _retrospective_content_box(frames)
                rows = _combat_causal_rows(timestamps, frame_hashes, indexed[identity])
                if not rows:
                    continue
                session_views = _combat_session_views(canonical, content_box)
                views = np.stack(
                    [
                        session_views[
                            cast(int, row["start_index"]) : cast(int, row["end_index"]) + 1
                        ]
                        for row in rows
                    ]
                )
                shifted = np.stack(
                    [
                        session_views[
                            cast(int, row["shift_start_index"]) : cast(int, row["shift_end_index"])
                            + 1
                        ]
                        for row in rows
                    ]
                )
                labels = np.asarray(
                    [cast(int, row["combat_id"]) for row in rows], dtype=np.int8
                )
                name = f"{split}-{ordinal:04d}.npz"
                path = shard_dir / name
                np.savez_compressed(
                    path,
                    views=views,
                    shifted_views=shifted,
                    combat_id=labels,
                    label_timestamp_ms=np.asarray(
                        [cast(int, row["label_timestamp_ms"]) for row in rows], dtype=np.int64
                    ),
                    visible_onset_timestamp_ms=np.asarray(
                        [cast(int, row["visible_onset_timestamp_ms"]) for row in rows],
                        dtype=np.int64,
                    ),
                    observation_end_timestamp_ms=np.asarray(
                        [cast(int, row["observation_end_timestamp_ms"]) for row in rows],
                        dtype=np.int64,
                    ),
                    window_sha256=np.asarray([cast(str, row["window_sha256"]) for row in rows]),
                    label_kind=np.asarray(
                        [cast(int, row["label_kind"]) for row in rows], dtype=np.int8
                    ),
                    session_hash=np.asarray([identity] * len(rows)),
                )
                class_counts = {
                    action: int(np.sum(labels == class_id))
                    for class_id, action in enumerate(ABILITIES[:4])
                }
                for action, count in class_counts.items():
                    counts[split][action] += count
                shards.append(
                    {
                        "path": name,
                        "sha256": _sha(path),
                        "split": split,
                        "session_hash": identity,
                        "orientation": orientation,
                        "rows": len(rows),
                        "class_counts": class_counts,
                    }
                )
        manifest: dict[str, object] = {
            "schema_version": COMBAT_CAUSAL_DATASET_SCHEMA,
            "status": "DIAGNOSTIC_ONLY" if diagnostic_only else "COMPLETED",
            "diagnostic_only": diagnostic_only,
            "training_eligible": not diagnostic_only,
            "window_frames": 32,
            "sample_hz": 10,
            "observation_end_lag_ms": 100,
            "time_shift_control_ms": 2000,
            "views": ["full", "combat_hud"],
            "combat_vocabulary": list(ABILITIES[:4]),
            "target_manifest_sha256": target_sha,
            "visible_onset_audit_sha256": audit_sha,
            "counts": counts,
            "shards": shards,
            "event_frame_included": False,
            "future_frames_included": False,
            "raw_video_or_source_paths_persisted": False,
            "video_test_accessed": False,
            "formal_policy_training_allowed": False,
            "shadow_allowed": False,
            "device_input_allowed": False,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return manifest


def _combat_causal_shards(
    dataset_root: Path, split: str
) -> tuple[Path, list[dict[str, object]], str]:
    if split not in {"train", "dev"}:
        raise T8Error("combat-causal loader may open train or dev only")
    root = _large_existing(dataset_root)
    manifest = _read_object(root / "manifest.json", "combat-causal manifest is unreadable")
    supplied = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema_version") != COMBAT_CAUSAL_DATASET_SCHEMA
        or manifest.get("status")
        not in {"COMPLETED", "DIAGNOSTIC_ONLY"}
        or manifest.get("diagnostic_only")
        is not (manifest.get("status") == "DIAGNOSTIC_ONLY")
        or manifest.get("training_eligible")
        is not (manifest.get("status") == "COMPLETED")
        or manifest.get("window_frames") != 32
        or manifest.get("sample_hz") != 10
        or manifest.get("observation_end_lag_ms") != 100
        or manifest.get("time_shift_control_ms") != 2000
        or manifest.get("views") != ["full", "combat_hud"]
        or manifest.get("combat_vocabulary") != list(ABILITIES[:4])
        or manifest.get("event_frame_included") is not False
        or manifest.get("future_frames_included") is not False
        or manifest.get("video_test_accessed") is not False
        or supplied != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise T8Error("combat-causal dataset contract is invalid")
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise T8Error("combat-causal shard index is invalid")
    selected = [row for row in cast(list[dict[str, object]], rows) if row.get("split") == split]
    if not selected:
        raise T8Error("combat-causal split is empty")
    return root, selected, supplied


def _load_combat_causal_shard(
    root: Path, row: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    name = row.get("path")
    if not isinstance(name, str) or Path(name).name != name:
        raise T8Error("combat-causal shard name is invalid")
    path = root / "shards" / name
    if _sha(path) != row.get("sha256"):
        raise T8Error("combat-causal shard hash differs")
    with np.load(path, allow_pickle=False) as shard:
        expected = {
            "views",
            "shifted_views",
            "combat_id",
            "label_timestamp_ms",
            "visible_onset_timestamp_ms",
            "observation_end_timestamp_ms",
            "window_sha256",
            "label_kind",
            "session_hash",
        }
        views = shard["views"]
        shifted = shard["shifted_views"]
        labels = shard["combat_id"]
        onset = shard["visible_onset_timestamp_ms"]
        observation_end = shard["observation_end_timestamp_ms"]
        label_kind = shard["label_kind"]
        if (
            set(shard.files) != expected
            or views.dtype != np.uint8
            or views.shape[1:] != (32, 2, 128, 128, 3)
            or shifted.dtype != np.uint8
            or shifted.shape != views.shape
            or labels.dtype != np.int8
            or labels.shape != (len(views),)
            or np.any(labels < 0)
            or np.any(labels >= 4)
            or np.any((label_kind == 1) & (observation_end > onset - 100))
        ):
            raise T8Error("combat-causal shard tensor contract is invalid")
        return views.copy(), shifted.copy(), labels.astype(np.int64)


def _combat_rgb_tensor(views: np.ndarray, device: torch.device) -> torch.Tensor:
    if views.ndim != 6 or views.shape[1:] != (32, 2, 128, 128, 3):
        raise T8Error("combat-causal probe requires Bx32x2x128x128x3 RGB")
    return (
        torch.from_numpy(views)
        .to(device)
        .permute(0, 1, 2, 5, 3, 4)
        .float()
        .div(255.0)
    )


class _CombatCausalRGB(nn.Module):
    def __init__(self, encoder_state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.encoder.load_state_dict(encoder_state, strict=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder.layer4.parameters():
            parameter.requires_grad = True
        self.mix = nn.Conv1d(1024, 256, 1)
        self.temporal = nn.Sequential(*(_V2ResidualBlock(value) for value in (1, 2, 4, 8)))
        self.head = nn.Linear(256, 4)

    def _features(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 6 or tuple(views.shape[1:]) != (32, 2, 3, 128, 128):
            raise T8Error("combat-causal model input is invalid")
        batch = len(views)
        encoded = cast(
            torch.Tensor,
            self.encoder(views.reshape(batch * 64, 3, 128, 128)),
        ).reshape(batch, 32, 1024)
        mixed = self.mix(encoded.transpose(1, 2))
        return cast(torch.Tensor, self.temporal(mixed))[..., -1]

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self._features(views)))


class _V26ConditionalCombatRGB(nn.Module):
    def __init__(self, encoder_state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.features = _CombatCausalRGB(encoder_state)
        del self.features.head
        self.gate_head = nn.Linear(256, 2)
        self.action_head = nn.Linear(256, 3)

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features._features(views)
        return self.gate_head(features), self.action_head(features)


def _load_t8_v26_model(model_path: Path, device: str) -> tuple[_V26ConditionalCombatRGB, torch.device]:
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise T8Error("T8-v2.6 predictor device is unavailable")
    target = torch.device(device)
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        state = load_file(model_path, device=str(target))
    except (OSError, SafetensorError) as exc:
        raise T8Error("T8-v2.6 model is unreadable") from exc
    if (
        metadata.get("schema") != V26_CONDITIONAL_MODEL_SCHEMA
        or metadata.get("gate_decision_threshold") != str(V26_GATE_DECISION_THRESHOLD)
    ):
        raise T8Error("T8-v2.6 model metadata is invalid")
    prefix = "features.encoder."
    encoder_state = {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }
    model = _V26ConditionalCombatRGB(encoder_state).to(target)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, target


def _v26_decode_logits(
    gate_logits: torch.Tensor, action_logits: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate = gate_logits.softmax(1)
    action = action_logits.softmax(1)
    probabilities = torch.cat((gate[:, :1], gate[:, 1:] * action), dim=1).float()
    active = gate[:, 1] >= V26_GATE_DECISION_THRESHOLD
    labels = torch.where(active, action.argmax(1) + 1, torch.zeros_like(active.long()))
    confidence = probabilities.gather(1, labels[:, None]).squeeze(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1) / math.log(4)
    return labels.cpu().numpy(), confidence.cpu().numpy(), entropy.cpu().numpy()


def open_t8_v26_predictor(
    model_path: Path, device: str
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    model, target = _load_t8_v26_model(model_path, device)

    def predict(views: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad(), torch.autocast(
            device_type=target.type, enabled=target.type == "cuda"
        ):
            gate_logits, action_logits = model(_combat_rgb_tensor(views, target))
        return _v26_decode_logits(gate_logits, action_logits)

    return predict


def open_t8_v26_stream_predictor(
    model_path: Path, device: str
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    model, target = _load_t8_v26_model(model_path, device)
    history: deque[torch.Tensor] = deque(maxlen=32)

    def predict(views: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if views.shape != (2, 128, 128, 3) or views.dtype != np.uint8:
            raise T8Error("T8-v2.6 stream predictor requires 2x128x128x3 uint8 RGB")
        tensor = (
            torch.from_numpy(views)
            .to(target)
            .permute(0, 3, 1, 2)
            .float()
            .div(255.0)
        )
        with torch.no_grad(), torch.autocast(
            device_type=target.type, enabled=target.type == "cuda"
        ):
            encoded = cast(torch.Tensor, model.features.encoder(tensor)).reshape(1024)
            history.append(encoded)
            current = list(history)
            if len(current) < 32:
                current = [current[0]] * (32 - len(current)) + current
            mixed = model.features.mix(torch.stack(current)[None].transpose(1, 2))
            features = cast(torch.Tensor, model.features.temporal(mixed))[..., -1]
            gate_logits = model.gate_head(features)
            action_logits = model.action_head(features)
        return _v26_decode_logits(gate_logits, action_logits)

    return predict


def _v26_gate_prediction(gate_logits: torch.Tensor) -> torch.Tensor:
    return (gate_logits.softmax(1)[:, 1] >= V26_GATE_DECISION_THRESHOLD).long()


def _predict_combat_causal(
    model: _CombatCausalRGB,
    root: Path,
    rows: Sequence[Mapping[str, object]],
    device: torch.device,
    batch_size: int,
    control: str = "normal",
    loader: Callable[
        [Path, Mapping[str, object]], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = _load_combat_causal_shard,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted: list[np.ndarray] = []
    action_predicted: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            views, shifted, current_y = loader(root, row)
            if control == "shifted":
                views = shifted
            elif control == "static":
                views = np.repeat(views[:, -1:], 32, axis=1)
            elif control != "normal":
                raise T8Error("combat-causal control is invalid")
            for start in range(0, len(views), batch_size):
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(_combat_rgb_tensor(views[start : start + batch_size], device))
                predicted.append(logits.argmax(1).cpu().numpy())
                action_predicted.append(logits[:, 1:].argmax(1).cpu().numpy())
            labels.append(current_y)
    return np.concatenate(predicted), np.concatenate(action_predicted), np.concatenate(labels)


def _combat_metrics(
    predicted: np.ndarray, action_predicted: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    active = labels > 0
    return {
        "four_class": _head_metrics(predicted, labels, 4),
        "action_vs_wait": _head_metrics((predicted > 0).astype(np.int64), active.astype(np.int64), 2),
        "action_only": _head_metrics(action_predicted[active], labels[active] - 1, 3),
    }


def _predict_v26_conditional(
    model: _V26ConditionalCombatRGB,
    root: Path,
    rows: Sequence[Mapping[str, object]],
    device: torch.device,
    batch_size: int,
    control: str = "normal",
    loader: Callable[
        [Path, Mapping[str, object]], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = _load_combat_causal_shard,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted: list[np.ndarray] = []
    action_predicted: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            views, shifted, current_y = loader(root, row)
            if control == "shifted":
                views = shifted
            elif control == "static":
                views = np.repeat(views[:, -1:], 32, axis=1)
            elif control != "normal":
                raise T8Error("T8-v2.6 conditional control is invalid")
            for start in range(0, len(views), batch_size):
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    gate_logits, action_logits = model(
                        _combat_rgb_tensor(views[start : start + batch_size], device)
                    )
                gate = _v26_gate_prediction(gate_logits).cpu().numpy()
                action = action_logits.argmax(1).cpu().numpy()
                predicted.append(np.where(gate == 0, 0, action + 1))
                action_predicted.append(action)
            labels.append(current_y)
    return np.concatenate(predicted), np.concatenate(action_predicted), np.concatenate(labels)


def _balanced_combat_order(labels: np.ndarray, seed: int) -> np.ndarray:
    if labels.ndim != 1 or len(labels) == 0 or np.any(labels < 0) or np.any(labels >= 4):
        raise T8Error("combat labels cannot be balanced")
    generator = np.random.default_rng(seed)
    indices = [np.flatnonzero(labels == class_id) for class_id in range(4)]
    target = max((len(current) for current in indices[1:]), default=0)
    if target == 0:
        return generator.permutation(len(labels))
    balanced = [
        generator.choice(current, size=target, replace=len(current) < target)
        for current in indices
        if len(current)
    ]
    return generator.permutation(np.concatenate(balanced))


def _fit_combat_causal(
    root: Path,
    train_rows: Sequence[Mapping[str, object]],
    dev_rows: Sequence[Mapping[str, object]],
    encoder_state: Mapping[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    class_weights: torch.Tensor,
    *,
    shuffled: bool,
    seed: int = 0,
    shuffle_rows: bool = True,
    balanced_sampling: bool = False,
    loader: Callable[
        [Path, Mapping[str, object]], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = _load_combat_causal_shard,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    model = _CombatCausalRGB(encoder_state).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_f1, best_epoch, best_state = -1.0, 0, {}
    for epoch in range(1, 7):
        model.train()
        for name, module in model.encoder.named_modules():
            if isinstance(module, nn.BatchNorm2d) and not name.startswith("layer4"):
                module.eval()
        ordered_rows = list(train_rows)
        if shuffle_rows:
            random.Random(seed * 1000 + epoch).shuffle(ordered_rows)
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0
        for ordinal, row in enumerate(ordered_rows):
            views, _shifted, labels = loader(root, row)
            if shuffled:
                labels = labels[
                    np.random.default_rng(seed * 100_000 + ordinal).permutation(len(labels))
                ]
            order_seed = seed * 100_000 + epoch + ordinal * 1000
            order = (
                _balanced_combat_order(labels, order_seed)
                if balanced_sampling
                else np.random.default_rng(order_seed).permutation(len(labels))
            )
            for start in range(0, len(order), batch_size):
                selected = order[start : start + batch_size]
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(_combat_rgb_tensor(views[selected], device))
                    loss = nn.functional.cross_entropy(
                        logits,
                        torch.from_numpy(labels[selected]).to(device),
                        weight=class_weights,
                    ) / 4.0
                scaler.scale(loss).backward()
                accumulation += 1
                if accumulation % 4 == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        if accumulation % 4:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        predicted, action_predicted, dev_y = _predict_combat_causal(
            model, root, dev_rows, device, batch_size, loader=loader
        )
        metrics = _combat_metrics(predicted, action_predicted, dev_y)
        macro_f1 = cast(float, cast(dict[str, object], metrics["four_class"])["macro_f1"])
        if macro_f1 > best_f1:
            best_f1, best_epoch = macro_f1, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if not best_state:
        raise T8Error("combat-causal pilot did not validate")
    model.load_state_dict(best_state, strict=True)
    predicted, action_predicted, dev_y = _predict_combat_causal(
        model, root, dev_rows, device, batch_size, loader=loader
    )
    return (
        {
            "best_epoch": best_epoch,
            "metrics": _combat_metrics(predicted, action_predicted, dev_y),
            "shuffled_labels": shuffled,
            "balanced_sampling": balanced_sampling,
        },
        best_state,
    )


def _fit_v26_conditional(
    root: Path,
    train_rows: Sequence[Mapping[str, object]],
    dev_rows: Sequence[Mapping[str, object]],
    encoder_state: Mapping[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    gate_weights: torch.Tensor,
    action_weights: torch.Tensor,
    *,
    shuffled: bool,
    seed: int = 0,
    initial_state: Mapping[str, torch.Tensor] | None = None,
    heads_only: bool = False,
    learning_rate: float = 1e-4,
    epochs: int = 6,
    balanced_sampling: bool = False,
    loader: Callable[
        [Path, Mapping[str, object]], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = _load_combat_causal_shard,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    model = _V26ConditionalCombatRGB(encoder_state).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    if heads_only:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_f1, best_epoch, best_state = -1.0, 0, {}
    for epoch in range(1, epochs + 1):
        model.train()
        if heads_only:
            model.features.eval()
        else:
            for name, module in model.features.encoder.named_modules():
                if isinstance(module, nn.BatchNorm2d) and not name.startswith("layer4"):
                    module.eval()
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0
        for ordinal, row in enumerate(train_rows):
            views, _shifted, labels = loader(root, row)
            if shuffled:
                labels = labels[
                    np.random.default_rng(seed * 100_000 + ordinal).permutation(len(labels))
                ]
            order_seed = seed * 100_000 + epoch + ordinal * 1000
            order = (
                _balanced_combat_order(labels, order_seed)
                if balanced_sampling
                else np.random.default_rng(order_seed).permutation(len(labels))
            )
            for start in range(0, len(order), batch_size):
                selected = order[start : start + batch_size]
                targets = torch.from_numpy(labels[selected]).to(device)
                active = targets > 0
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    gate_logits, action_logits = model(
                        _combat_rgb_tensor(views[selected], device)
                    )
                    gate_loss = nn.functional.cross_entropy(
                        gate_logits, active.long(), weight=gate_weights
                    )
                    loss = gate_loss
                    if bool(active.any()):
                        loss = loss + nn.functional.cross_entropy(
                            action_logits[active], targets[active] - 1, weight=action_weights
                        )
                    loss = loss / 4.0
                scaler.scale(loss).backward()
                accumulation += 1
                if accumulation % 4 == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        if accumulation % 4:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        predicted, action_predicted, dev_y = _predict_v26_conditional(
            model, root, dev_rows, device, batch_size, loader=loader
        )
        metrics = _combat_metrics(predicted, action_predicted, dev_y)
        macro_f1 = cast(float, cast(dict[str, object], metrics["four_class"])["macro_f1"])
        if macro_f1 > best_f1:
            best_f1, best_epoch = macro_f1, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if not best_state:
        raise T8Error("T8-v2.6 conditional pilot did not validate")
    model.load_state_dict(best_state, strict=True)
    predicted, action_predicted, dev_y = _predict_v26_conditional(
        model, root, dev_rows, device, batch_size, loader=loader
    )
    return (
        {
            "best_epoch": best_epoch,
            "metrics": _combat_metrics(predicted, action_predicted, dev_y),
            "shuffled_labels": shuffled,
            "conditional_heads": True,
            "heads_only": heads_only,
            "balanced_sampling": balanced_sampling,
        },
        best_state,
    )


def run_t8_combat_causal_pilot(
    *,
    dataset_root: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 8,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise T8Error("combat-causal pilot settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    target = torch.device(device)
    root, train_rows, manifest_sha = _combat_causal_shards(dataset_root, "train")
    dev_root, dev_rows, dev_sha = _combat_causal_shards(dataset_root, "dev")
    if root != dev_root or manifest_sha != dev_sha:
        raise T8Error("combat-causal dataset changed between splits")
    manifest = _read_object(root / "manifest.json", "combat-causal manifest is unreadable")
    diagnostic_only = manifest.get("diagnostic_only") is True
    adapter = _large_existing(adapter_checkpoint)
    encoder_state, adapter_meta = _load_v2_adapter(adapter, target)
    if adapter_meta.get("target_manifest_sha256") != manifest.get("target_manifest_sha256"):
        raise T8Error("combat-causal adapter differs from dataset target")
    train_counts = cast(dict[str, int], cast(dict[str, object], manifest["counts"])["train"])
    counts = np.asarray([train_counts[name] for name in ABILITIES[:4]], dtype=np.float32)
    if np.any(counts <= 0):
        raise T8Error("combat-causal train split lacks a class")
    weights = torch.from_numpy(counts.sum() / (4.0 * counts)).to(target)
    normal, state = _fit_combat_causal(
        root,
        train_rows,
        dev_rows,
        encoder_state,
        target,
        batch_size,
        weights,
        shuffled=False,
    )
    shuffled, _ = _fit_combat_causal(
        root,
        train_rows,
        dev_rows,
        encoder_state,
        target,
        batch_size,
        weights,
        shuffled=True,
    )
    model = _CombatCausalRGB(encoder_state).to(target)
    model.load_state_dict(state, strict=True)
    static_pred, static_action, dev_y = _predict_combat_causal(
        model, root, dev_rows, target, batch_size, "static"
    )
    shifted_pred, shifted_action, shifted_y = _predict_combat_causal(
        model, root, dev_rows, target, batch_size, "shifted"
    )
    if not np.array_equal(dev_y, shifted_y):
        raise T8Error("combat-causal controls changed dev labels")
    static_metrics = _combat_metrics(static_pred, static_action, dev_y)
    shifted_metrics = _combat_metrics(shifted_pred, shifted_action, dev_y)
    normal_metrics = cast(dict[str, object], normal["metrics"])
    four = cast(dict[str, object], normal_metrics["four_class"])
    binary = cast(dict[str, object], normal_metrics["action_vs_wait"])
    action = cast(dict[str, object], normal_metrics["action_only"])
    shuffled_four = cast(
        dict[str, object], cast(dict[str, object], shuffled["metrics"])["four_class"]
    )
    static_four = cast(dict[str, object], static_metrics["four_class"])
    shifted_four = cast(dict[str, object], shifted_metrics["four_class"])
    normal_f1 = cast(float, four["macro_f1"])
    plurality = float(np.max(np.bincount(dev_y, minlength=4)) / len(dev_y))
    margins = {
        "normal_minus_shuffled_macro_f1": normal_f1 - cast(float, shuffled_four["macro_f1"]),
        "normal_minus_static_macro_f1": normal_f1 - cast(float, static_four["macro_f1"]),
        "normal_minus_shifted_macro_f1": normal_f1 - cast(float, shifted_four["macro_f1"]),
    }
    passed = bool(
        cast(float, four["accuracy"]) >= plurality + 0.10
        and normal_f1 >= 0.40
        and cast(float, binary["macro_f1"]) >= 0.60
        and cast(float, action["macro_f1"]) >= 0.45
        and min(cast(list[float], action["per_class_recall"])) >= 0.30
        and margins["normal_minus_shuffled_macro_f1"] >= 0.15
        and margins["normal_minus_static_macro_f1"] >= 0.10
        and margins["normal_minus_shifted_macro_f1"] >= 0.10
    )
    report: dict[str, object] = {
        "schema_version": COMBAT_CAUSAL_PILOT_SCHEMA,
        "status": (
            "DIAGNOSTIC_SIGNAL_FOUND"
            if diagnostic_only and passed
            else "DIAGNOSTIC_SIGNAL_NOT_FOUND"
            if diagnostic_only
            else "COMBAT_CAUSAL_PILOT_PASSED"
            if passed
            else "CAUSAL_COMBAT_SIGNAL_NOT_FOUND"
        ),
        "diagnostic_only": diagnostic_only,
        "dataset_manifest_sha256": manifest_sha,
        "adapter_sha256": _sha(adapter),
        "seed": 0,
        "epochs": 6,
        "window_frames": 32,
        "views": ["full", "combat_hud"],
        "encoder_trainable_scope": "resnet18_layer4_plus_causal_tcn_and_head",
        "normal": normal,
        "shuffled": shuffled,
        "static_frame": static_metrics,
        "time_shift_2000ms": shifted_metrics,
        "plurality_accuracy": plurality,
        "margins": margins,
        "diagnostic_signal_found": passed,
        "combat_subpolicy_three_seed_allowed": passed and not diagnostic_only,
        "full_t8_policy_training_allowed": False,
        "video_test_accessed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        save_file(state, staging / "combat-causal-seed0.safetensors")
        report["model_sha256"] = _sha(staging / "combat-causal-seed0.safetensors")
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _v25_session_names(*, pilot: bool) -> tuple[str, ...]:
    return (
        ("session-001", "session-002", "session-003", "session-009")
        if pilot
        else tuple(f"session-{index:03d}" for index in range(1, 13))
    )


def _v25_session_metadata(path: Path, *, diagnostic: bool = False) -> dict[str, object]:
    summary_path = path / "summary.json"
    events_path = path / "events.jsonl"
    contract_path = path / "action-contract.json"
    manifest_path = path / "session-manifest.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
        ]
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T8Error("T8-v2.5 session metadata is unreadable") from exc
    if not isinstance(summary, dict):
        raise T8Error("T8-v2.5 session contract is invalid")
    formal_contract = bool(
        summary.get("status") == "COMPLETED"
        and summary.get("formal_session") is True
        and summary.get("published_as_formal") is True
        and summary.get("training_eligible") is True
        and isinstance(summary.get("duration_seconds"), (int, float))
        and float(summary["duration_seconds"]) >= 300
        and isinstance(summary.get("samples"), int)
        and int(summary["samples"]) >= RGB_TEACHER_MIN_FORMAL_SAMPLES
    )
    diagnostic_contract = bool(
        summary.get("status") in {"COMPLETED", "ACTION_CAP_REACHED"}
        and summary.get("formal_session") is False
        and summary.get("published_as_formal") is False
        and summary.get("training_eligible") is False
        and summary.get("input_enabled") is True
        and isinstance(summary.get("executed_actions"), int)
        and int(summary["executed_actions"]) > 0
        and summary.get("environment_input_commands_sent") == 0
        and isinstance(summary.get("samples"), int)
        and int(summary["samples"]) >= 100
    )
    if (
        summary.get("schema_version") != RGB_TEACHER_SCHEMA
        or summary.get("dataset_schema_version") != RGB_TEACHER_DATA_SCHEMA
        or summary.get("event_source") != RGB_TEACHER_SOURCE
        or summary.get("window_frames") != RGB_TEACHER_WINDOW_FRAMES
        or summary.get("video_test_accessed") is not False
        or (diagnostic and not diagnostic_contract)
        or (not diagnostic and not formal_contract)
        or not isinstance(contract, dict)
        or contract.get("schema_version") != RGB_TEACHER_DATA_SCHEMA
        or contract.get("source") != RGB_TEACHER_SOURCE
        or contract.get("layout_sha256") != summary.get("layout_sha256")
        or contract.get("teacher_report_sha256") != summary.get("teacher_report_sha256")
        or hashlib.sha256(_canonical(contract)).hexdigest()
        != summary.get("action_contract_sha256")
    ):
        raise T8Error("T8-v2.5 session contract is invalid")
    frame_paths = sorted(path.glob("frames-*.npz"))
    decision_paths = sorted(path.glob("samples-*.npz"))
    expected: dict[str, object] = {
        "schema_version": RGB_TEACHER_SESSION_SCHEMA,
        "summary_sha256": _sha(summary_path),
        "events_sha256": _sha(events_path),
        "action_contract_file_sha256": _sha(contract_path),
        "frame_shards": [
            {"name": item.name, "sha256": _sha(item)} for item in frame_paths
        ],
        "shards": [
            {"name": item.name, "sha256": _sha(item)} for item in decision_paths
        ],
    }
    identity = hashlib.sha256(_canonical(expected)).hexdigest()
    if manifest != {**expected, "session_sha256": identity} or not frame_paths or not decision_paths:
        raise T8Error("T8-v2.5 session manifest is invalid")
    frames: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    for item in frame_paths:
        with np.load(item, allow_pickle=False) as shard:
            if set(shard.files) != {"frames", "timestamp_ns"}:
                raise T8Error("T8-v2.5 frame shard fields are invalid")
            current_frames, current_times = shard["frames"], shard["timestamp_ns"]
            if (
                current_frames.dtype != np.uint8
                or current_frames.shape[1:] != (128, 128, 3)
                or current_times.dtype != np.int64
                or current_times.shape != (len(current_frames),)
            ):
                raise T8Error("T8-v2.5 frame tensors are invalid")
            frames.append(current_frames)
            timestamps.append(current_times)
    all_frames, all_times = np.concatenate(frames), np.concatenate(timestamps)
    if np.any(np.diff(all_times) < 0):
        raise T8Error("T8-v2.5 frame timestamps are not monotonic")
    labels: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    shifted: list[np.ndarray] = []
    observation_times: list[np.ndarray] = []
    decisions: list[np.ndarray] = []
    executions: list[np.ndarray] = []
    sent_values: list[np.ndarray] = []
    expected_fields = {
        "observation_index",
        "shifted_observation_index",
        "combat_id",
        "observation_end_timestamp_ns",
        "decision_timestamp_ns",
        "execution_timestamp_ns",
        "confidence",
        "input_sent",
    }
    for item in decision_paths:
        with np.load(item, allow_pickle=False) as shard:
            if set(shard.files) != expected_fields:
                raise T8Error("T8-v2.5 decision shard fields are invalid")
            size = len(shard["combat_id"])
            if any(shard[name].shape != (size,) for name in expected_fields):
                raise T8Error("T8-v2.5 decision tensors are invalid")
            labels.append(shard["combat_id"].astype(np.int64))
            observations.append(shard["observation_index"].astype(np.int64))
            shifted.append(shard["shifted_observation_index"].astype(np.int64))
            observation_times.append(
                shard["observation_end_timestamp_ns"].astype(np.int64)
            )
            decisions.append(shard["decision_timestamp_ns"].astype(np.int64))
            executions.append(shard["execution_timestamp_ns"].astype(np.int64))
            sent_values.append(shard["input_sent"].astype(np.uint8))
    combat = np.concatenate(labels)
    observation = np.concatenate(observations)
    shifted_observation = np.concatenate(shifted)
    bound_observation_times = np.concatenate(observation_times)
    decision_times = np.concatenate(decisions)
    execution_times = np.concatenate(executions)
    sent = np.concatenate(sent_values)
    if (
        len(events) != len(combat)
        or len(combat) != summary.get("samples")
        or np.any(combat < 0)
        or np.any(combat >= 4)
        or np.any(observation < RGB_TEACHER_WINDOW_FRAMES - 1)
        or np.any(observation >= len(all_frames))
        or np.any(shifted_observation < RGB_TEACHER_WINDOW_FRAMES - 1)
        or np.any(shifted_observation != observation - RGB_TEACHER_HISTORY_FRAMES)
        or np.any(bound_observation_times != all_times[observation])
        or np.any(all_times[observation] > decision_times)
        or np.any(np.diff(decision_times) < 0)
        or np.any((combat == 0) & ((sent != 0) | (execution_times != -1)))
        or np.any((combat > 0) & ((sent != 1) | (execution_times < decision_times + RGB_TEACHER_EXECUTION_LAG_MS * 1_000_000)))
        or set(combat.tolist()) != set(range(4))
    ):
        raise T8Error("T8-v2.5 causal decision binding is invalid")
    for index, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or event.get("schema_version") != RGB_TEACHER_SCHEMA
            or event.get("sequence") != index
            or event.get("source") != RGB_TEACHER_SOURCE
            or event.get("combat") != ABILITIES[int(combat[index])]
            or event.get("observation_end_timestamp_ns") != int(all_times[observation[index]])
            or event.get("decision_timestamp_ns") != int(decision_times[index])
            or event.get("execution_timestamp_ns") != int(execution_times[index])
            or event.get("input_sent") is not bool(sent[index])
            or event.get("frame_sha256")
            != hashlib.sha256(all_frames[observation[index]].tobytes()).hexdigest()
        ):
            raise T8Error("T8-v2.5 events do not bind RGB decisions")
    return {
        "name": path.name,
        "session_sha256": identity,
        "layout_sha256": summary["layout_sha256"],
        "teacher_report_sha256": summary["teacher_report_sha256"],
        "action_contract_sha256": summary["action_contract_sha256"],
        "counts": {
            name: int(np.sum(combat == index))
            for index, name in enumerate(ABILITIES[:4])
        },
        "rows": len(combat),
        "frame_shards": expected["frame_shards"],
        "shards": expected["shards"],
    }


def freeze_t8_v25_split(
    *, dataset_root: Path, output_path: Path, pilot: bool = False
) -> dict[str, object]:
    root, output = _large_existing(dataset_root), _large_new(output_path)
    names = _v25_session_names(pilot=pilot)
    sessions = [_v25_session_metadata(root / name) for name in names]
    if (
        len({session["layout_sha256"] for session in sessions}) != 1
        or len({session["teacher_report_sha256"] for session in sessions}) != 1
        or len({session["action_contract_sha256"] for session in sessions}) != 1
    ):
        raise T8Error("T8-v2.5 sessions do not share one frozen contract")
    split_names = (
        {"train": names[:3], "dev": names[3:], "test": ()}
        if pilot
        else {"train": names[:8], "dev": names[8:10], "test": names[10:]}
    )
    indexed = {cast(str, session["name"]): session for session in sessions}
    for split in ("train", "dev"):
        totals = {action: 0 for action in ABILITIES[:4]}
        for name in split_names[split]:
            counts = cast(dict[str, int], indexed[name]["counts"])
            for action in ABILITIES[:4]:
                totals[action] += counts[action]
        if any(value <= 0 for value in totals.values()):
            raise T8Error("T8-v2.5 split lacks a combat class")
    payload: dict[str, object] = {
        "schema_version": V25_SPLIT_SCHEMA,
        "pilot": pilot,
        "layout_sha256": sessions[0]["layout_sha256"],
        "teacher_report_sha256": sessions[0]["teacher_report_sha256"],
        "action_contract_sha256": sessions[0]["action_contract_sha256"],
        "window_frames": RGB_TEACHER_WINDOW_FRAMES,
        "gate_decision_threshold": V26_GATE_DECISION_THRESHOLD,
        "gate_threshold_selection": "lowest_validation_grid_value_passing_macro_f1_and_action_recall",
        "splits": {
            split: [
                {
                    "name": name,
                    "session_sha256": indexed[name]["session_sha256"],
                }
                for name in selected
            ]
            for split, selected in split_names.items()
        },
        "test_accessed": False,
    }
    payload["split_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _write_frozen_json(output, payload)
    return payload


class _V25FrameCache:
    def __init__(self) -> None:
        self._session = ""
        self._views: np.ndarray | None = None

    def views(self, root: Path, row: Mapping[str, object]) -> np.ndarray:
        session = cast(str, row["session"])
        if session != self._session:
            session_path = root / session
            pieces: list[np.ndarray] = []
            for entry in cast(list[dict[str, object]], row["frame_shards"]):
                path = session_path / cast(str, entry["name"])
                if _sha(path) != entry.get("sha256"):
                    raise T8Error("T8-v2.5 frame shard changed after split freeze")
                with np.load(path, allow_pickle=False) as shard:
                    pieces.append(shard["frames"].copy())
            frames = np.concatenate(pieces)
            self._views = _combat_session_views(frames, (0, 0, 128, 128))
            self._session = session
        assert self._views is not None
        return self._views


def _v25_rows(
    dataset_root: Path, split_path: Path, split: str
) -> tuple[Path, list[dict[str, object]], dict[str, object]]:
    if split not in {"train", "dev"}:
        raise T8Error("T8-v2.5 pilot may open train or dev only")
    root = _large_existing(dataset_root)
    frozen = _read_object(_large_existing(split_path), "T8-v2.5 split is unreadable")
    supplied = frozen.get("split_sha256")
    unsigned = {key: value for key, value in frozen.items() if key != "split_sha256"}
    splits = frozen.get("splits")
    selected = splits.get(split) if isinstance(splits, dict) else None
    if (
        frozen.get("schema_version") != V25_SPLIT_SCHEMA
        or frozen.get("window_frames") != RGB_TEACHER_WINDOW_FRAMES
        or frozen.get("test_accessed") is not False
        or supplied != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or not isinstance(selected, list)
        or not selected
    ):
        raise T8Error("T8-v2.5 split contract is invalid")
    rows: list[dict[str, object]] = []
    for frozen_session in cast(list[dict[str, object]], selected):
        name = frozen_session.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"session-[0-9]{3}", name):
            raise T8Error("T8-v2.5 split session name is invalid")
        metadata = _v25_session_metadata(root / name)
        if metadata["session_sha256"] != frozen_session.get("session_sha256"):
            raise T8Error("T8-v2.5 session changed after split freeze")
        for shard in cast(list[dict[str, object]], metadata["shards"]):
            rows.append(
                {
                    "session": name,
                    "path": shard["name"],
                    "sha256": shard["sha256"],
                    "frame_shards": metadata["frame_shards"],
                }
            )
    return root, rows, frozen


def _v25_shard_loader(
    cache: _V25FrameCache,
) -> Callable[[Path, Mapping[str, object]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    def load(
        root: Path, row: Mapping[str, object]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        name, session = row.get("path"), row.get("session")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(session, str)
        ):
            raise T8Error("T8-v2.5 decision shard name is invalid")
        path = root / session / name
        if _sha(path) != row.get("sha256"):
            raise T8Error("T8-v2.5 decision shard changed after split freeze")
        session_views = cache.views(root, row)
        with np.load(path, allow_pickle=False) as shard:
            observation = shard["observation_index"].astype(np.int64)
            shifted_observation = shard["shifted_observation_index"].astype(np.int64)
            labels = shard["combat_id"].astype(np.int64)
        views = np.stack(
            [session_views[index - 31 : index + 1] for index in observation]
        )
        shifted = np.stack(
            [session_views[index - 31 : index + 1] for index in shifted_observation]
        )
        return views, shifted, labels

    return load


def run_t8_v25_pilot(
    *,
    dataset_root: Path,
    split_path: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1 or seed not in SEEDS:
        raise T8Error("T8-v2.5 pilot settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    target = torch.device(device)
    root, train_rows, split = _v25_rows(dataset_root, split_path, "train")
    dev_root, dev_rows, dev_split = _v25_rows(dataset_root, split_path, "dev")
    if root != dev_root or split.get("split_sha256") != dev_split.get("split_sha256"):
        raise T8Error("T8-v2.5 split changed between train and dev")
    counts = np.zeros(4, dtype=np.float32)
    for row in train_rows:
        with np.load(root / cast(str, row["session"]) / cast(str, row["path"]), allow_pickle=False) as shard:
            counts += np.bincount(shard["combat_id"], minlength=4)
    if np.any(counts <= 0):
        raise T8Error("T8-v2.5 train split lacks a class")
    gate_counts = np.asarray([counts[0], counts[1:].sum()], dtype=np.float32)
    action_counts = counts[1:]
    gate_weights = torch.from_numpy(gate_counts.sum() / (2.0 * gate_counts)).to(target)
    action_weights = torch.from_numpy(action_counts.sum() / (3.0 * action_counts)).to(target)
    adapter = _large_existing(adapter_checkpoint)
    encoder_state, adapter_meta = _load_v2_adapter(adapter, target)
    cache = _V25FrameCache()
    loader = _v25_shard_loader(cache)
    normal, state = _fit_v26_conditional(
        root,
        train_rows,
        dev_rows,
        encoder_state,
        target,
        batch_size,
        gate_weights,
        action_weights,
        shuffled=False,
        seed=seed,
        loader=loader,
    )
    shuffled, _ = _fit_v26_conditional(
        root,
        train_rows,
        dev_rows,
        encoder_state,
        target,
        batch_size,
        gate_weights,
        action_weights,
        shuffled=True,
        seed=seed,
        loader=loader,
    )
    model = _V26ConditionalCombatRGB(encoder_state).to(target)
    model.load_state_dict(state, strict=True)
    static_pred, static_action, dev_y = _predict_v26_conditional(
        model, root, dev_rows, target, batch_size, "static", loader=loader
    )
    shifted_pred, shifted_action, shifted_y = _predict_v26_conditional(
        model, root, dev_rows, target, batch_size, "shifted", loader=loader
    )
    if not np.array_equal(dev_y, shifted_y):
        raise T8Error("T8-v2.5 controls changed dev labels")
    static_metrics = _combat_metrics(static_pred, static_action, dev_y)
    shifted_metrics = _combat_metrics(shifted_pred, shifted_action, dev_y)
    metrics = cast(dict[str, object], normal["metrics"])
    four = cast(dict[str, object], metrics["four_class"])
    action = cast(dict[str, object], metrics["action_only"])
    normal_f1 = cast(float, four["macro_f1"])
    shuffled_f1 = cast(
        float,
        cast(
            dict[str, object], cast(dict[str, object], shuffled["metrics"])["four_class"]
        )["macro_f1"],
    )
    static_f1 = cast(float, cast(dict[str, object], static_metrics["four_class"])["macro_f1"])
    shifted_f1 = cast(float, cast(dict[str, object], shifted_metrics["four_class"])["macro_f1"])
    margins = {
        "normal_minus_shuffled_macro_f1": normal_f1 - shuffled_f1,
        "normal_minus_static_macro_f1": normal_f1 - static_f1,
        "normal_minus_shifted_macro_f1": normal_f1 - shifted_f1,
    }
    passed = bool(
        cast(float, four["accuracy"]) >= 0.45
        and normal_f1 >= 0.50
        and min(cast(list[float], action["per_class_recall"])) >= 0.35
        and margins["normal_minus_shuffled_macro_f1"] >= 0.15
        and margins["normal_minus_static_macro_f1"] >= 0.10
        and margins["normal_minus_shifted_macro_f1"] >= 0.10
    )
    report: dict[str, object] = {
        "schema_version": V26_CONDITIONAL_PILOT_SCHEMA,
        "status": "PILOT_REVIEW_READY" if passed else "PILOT_DIAGNOSIS_REQUIRED",
        "architecture": "resnet18_layer4_plus_causal_tcn_conditional_wait_action_and_three_action_heads_v1",
        "seed": seed,
        "split_sha256": split["split_sha256"],
        "pilot_split": split["pilot"],
        "adapter_sha256": _sha(adapter),
        "adapter_source_sha256": adapter_meta.get("v5_source_model_sha256"),
        "window_frames": RGB_TEACHER_WINDOW_FRAMES,
        "gate_decision_threshold": V26_GATE_DECISION_THRESHOLD,
        "gate_threshold_selection": "lowest_validation_grid_value_passing_macro_f1_and_action_recall",
        "normal": normal,
        "shuffled": shuffled,
        "static_frame": static_metrics,
        "time_shift_2000ms": shifted_metrics,
        "margins": margins,
        "learnability_gate_passed": passed,
        "three_seed_training_allowed": passed and split.get("pilot") is False,
        "test_accessed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        save_file(
            state,
            staging / f"model-seed-{seed}.safetensors",
            metadata={
                "schema": V26_CONDITIONAL_MODEL_SCHEMA,
                "seed": str(seed),
                "split_sha256": cast(str, split["split_sha256"]),
                "adapter_sha256": _sha(adapter),
                "gate_decision_threshold": str(V26_GATE_DECISION_THRESHOLD),
            },
        )
        report["model_sha256"] = _sha(staging / f"model-seed-{seed}.safetensors")
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def select_t8_v26_model(*, run_root: Path, output_path: Path) -> dict[str, object]:
    root, output = _large_existing(run_root), _large_new(output_path)
    entries: list[dict[str, object]] = []
    split_sha256 = ""
    for seed in SEEDS:
        report_path = root / f"seed-{seed}" / "report.json"
        model_path = root / f"seed-{seed}" / f"model-seed-{seed}.safetensors"
        report = _read_object(report_path, "T8-v2.6 seed report is unreadable")
        metrics = report.get("normal")
        four = (
            cast(dict[str, object], cast(dict[str, object], metrics).get("metrics", {})).get(
                "four_class"
            )
            if isinstance(metrics, dict)
            else None
        )
        current_split = report.get("split_sha256")
        if (
            report.get("schema_version") != V26_CONDITIONAL_PILOT_SCHEMA
            or report.get("seed") != seed
            or report.get("pilot_split") is not False
            or report.get("learnability_gate_passed") is not True
            or report.get("three_seed_training_allowed") is not True
            or report.get("test_accessed") is not False
            or report.get("gate_decision_threshold") != V26_GATE_DECISION_THRESHOLD
            or not isinstance(current_split, str)
            or (split_sha256 and current_split != split_sha256)
            or not isinstance(four, dict)
            or not isinstance(four.get("macro_f1"), (int, float))
            or report.get("model_sha256") != _sha(_large_existing(model_path))
        ):
            raise T8Error("T8-v2.6 three-seed selection contract is invalid")
        split_sha256 = current_split
        entries.append(
            {
                "seed": seed,
                "dev_four_class_macro_f1": float(four["macro_f1"]),
                "dev_four_class_accuracy": float(four["accuracy"]),
                "model": f"seed-{seed}/model-seed-{seed}.safetensors",
                "model_sha256": report["model_sha256"],
                "report_sha256": _sha(report_path),
            }
        )
    selected = max(
        entries,
        key=lambda item: (cast(float, item["dev_four_class_macro_f1"]), -cast(int, item["seed"])),
    )
    payload: dict[str, object] = {
        "schema_version": V26_SELECTION_SCHEMA,
        "status": "THREE_SEED_MODEL_SELECTED",
        "selection_metric": "dev_four_class_macro_f1",
        "split_sha256": split_sha256,
        "gate_decision_threshold": V26_GATE_DECISION_THRESHOLD,
        "seeds": entries,
        "selected_seed": selected["seed"],
        "selected_model": selected["model"],
        "selected_model_sha256": selected["model_sha256"],
        "test_accessed": False,
        "shadow_allowed": False,
    }
    payload["selection_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _write_frozen_json(output, payload)
    return payload


def _v26_selected_test_rows(
    dataset_root: Path,
    split_path: Path,
    run_root: Path,
    selection_path: Path,
) -> tuple[Path, list[dict[str, object]], dict[str, object], Path]:
    root = _large_existing(dataset_root)
    run = _large_existing(run_root)
    frozen = _read_object(_large_existing(split_path), "T8-v2.6 split is unreadable")
    selection_file = _large_existing(selection_path)
    selection = _read_object(selection_file, "T8-v2.6 selection is unreadable")
    frozen_unsigned = {key: value for key, value in frozen.items() if key != "split_sha256"}
    selection_unsigned = {
        key: value for key, value in selection.items() if key != "selection_sha256"
    }
    selected_seed = selection.get("selected_seed")
    selected_name = selection.get("selected_model")
    splits = frozen.get("splits")
    test_sessions = splits.get("test") if isinstance(splits, dict) else None
    if (
        frozen.get("schema_version") != V25_SPLIT_SCHEMA
        or frozen.get("pilot") is not False
        or frozen.get("test_accessed") is not False
        or frozen.get("gate_decision_threshold") != V26_GATE_DECISION_THRESHOLD
        or frozen.get("split_sha256")
        != hashlib.sha256(_canonical(frozen_unsigned)).hexdigest()
        or selection.get("schema_version") != V26_SELECTION_SCHEMA
        or selection.get("status") != "THREE_SEED_MODEL_SELECTED"
        or selection.get("test_accessed") is not False
        or selection.get("shadow_allowed") is not False
        or selection.get("split_sha256") != frozen.get("split_sha256")
        or selection.get("gate_decision_threshold") != V26_GATE_DECISION_THRESHOLD
        or selection.get("selection_sha256")
        != hashlib.sha256(_canonical(selection_unsigned)).hexdigest()
        or selected_seed not in SEEDS
        or selected_name != f"seed-{selected_seed}/model-seed-{selected_seed}.safetensors"
        or not isinstance(test_sessions, list)
        or len(test_sessions) != 2
        or [session.get("name") for session in test_sessions if isinstance(session, dict)]
        != ["session-011", "session-012"]
    ):
        raise T8Error("T8-v2.6 sealed evaluation contract is invalid")
    model_path = _large_existing(run / selected_name)
    if selection.get("selected_model_sha256") != _sha(model_path):
        raise T8Error("T8-v2.6 selected model changed before sealed evaluation")
    rows: list[dict[str, object]] = []
    for frozen_session in cast(list[dict[str, object]], test_sessions):
        name = frozen_session.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"session-[0-9]{3}", name):
            raise T8Error("T8-v2.6 sealed test session name is invalid")
        metadata = _v25_session_metadata(root / name)
        if metadata["session_sha256"] != frozen_session.get("session_sha256"):
            raise T8Error("T8-v2.6 sealed test session changed after split freeze")
        for shard in cast(list[dict[str, object]], metadata["shards"]):
            rows.append(
                {
                    "session": name,
                    "path": shard["name"],
                    "sha256": shard["sha256"],
                    "frame_shards": metadata["frame_shards"],
                }
            )
    return root, rows, selection, model_path


def _v26_session_lengths(root: Path, rows: Sequence[Mapping[str, object]]) -> tuple[int, ...]:
    lengths: list[int] = []
    current_session = ""
    for row in rows:
        session, name = cast(str, row["session"]), cast(str, row["path"])
        with np.load(root / session / name, allow_pickle=False) as shard:
            length = len(shard["combat_id"])
        if session == current_session:
            lengths[-1] += length
        else:
            current_session = session
            lengths.append(length)
    return tuple(lengths)


def evaluate_t8_v26_offline(
    *,
    dataset_root: Path,
    split_path: Path,
    run_root: Path,
    selection_path: Path,
    output_path: Path,
    device: str,
    batch_size: int = 8,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise T8Error("T8-v2.6 sealed evaluation settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    root, rows, selection, model_path = _v26_selected_test_rows(
        dataset_root, split_path, run_root, selection_path
    )
    output = _large_new(output_path)
    target = torch.device(device)
    state = load_file(model_path, device=str(target))
    prefix = "features.encoder."
    encoder_state = {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }
    model = _V26ConditionalCombatRGB(encoder_state).to(target)
    model.load_state_dict(state, strict=True)
    loader = _v25_shard_loader(_V25FrameCache())
    predicted, action_predicted, labels = _predict_v26_conditional(
        model, root, rows, target, batch_size, loader=loader
    )
    static_predicted, static_action, static_labels = _predict_v26_conditional(
        model, root, rows, target, batch_size, "static", loader=loader
    )
    shifted_predicted, shifted_action, shifted_labels = _predict_v26_conditional(
        model, root, rows, target, batch_size, "shifted", loader=loader
    )
    if not np.array_equal(labels, static_labels) or not np.array_equal(labels, shifted_labels):
        raise T8Error("T8-v2.6 sealed controls changed test labels")
    metrics = _combat_metrics(predicted, action_predicted, labels)
    static_metrics = _combat_metrics(static_predicted, static_action, labels)
    shifted_metrics = _combat_metrics(shifted_predicted, shifted_action, labels)
    four = cast(dict[str, object], metrics["four_class"])
    action = cast(dict[str, object], metrics["action_only"])
    normal_f1 = cast(float, four["macro_f1"])
    static_f1 = cast(float, cast(dict[str, object], static_metrics["four_class"])["macro_f1"])
    shifted_f1 = cast(float, cast(dict[str, object], shifted_metrics["four_class"])["macro_f1"])
    lengths = _v26_session_lengths(root, rows)
    true_switch = _switch_rate(labels[:, None], lengths)
    predicted_switch = _switch_rate(predicted[:, None], lengths)
    switch_error = abs(predicted_switch - true_switch)
    margins = {
        "normal_minus_static_macro_f1": normal_f1 - static_f1,
        "normal_minus_shifted_macro_f1": normal_f1 - shifted_f1,
    }
    passed = bool(
        cast(float, four["accuracy"]) >= 0.45
        and normal_f1 >= 0.50
        and min(cast(list[float], action["per_class_recall"])) >= 0.35
        and margins["normal_minus_static_macro_f1"] >= 0.10
        and margins["normal_minus_shifted_macro_f1"] >= 0.10
        and switch_error <= SWITCH_RATE_ERROR_THRESHOLD
    )
    report: dict[str, object] = {
        "schema_version": V26_EVALUATION_SCHEMA,
        "status": "SEALED_OFFLINE_EVALUATION_PASSED" if passed else "SEALED_OFFLINE_EVALUATION_FAILED",
        "strict_passed": passed,
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": _sha(_large_existing(selection_path)),
        "split_sha256": selection["split_sha256"],
        "selected_seed": selection["selected_seed"],
        "model_sha256": _sha(model_path),
        "gate_decision_threshold": V26_GATE_DECISION_THRESHOLD,
        "metrics": metrics,
        "static_frame": static_metrics,
        "time_shift_2000ms": shifted_metrics,
        "margins": margins,
        "switch_rate": {
            "true": true_switch,
            "predicted": predicted_switch,
            "absolute_error": switch_error,
        },
        "thresholds": {
            "four_class_accuracy": 0.45,
            "four_class_macro_f1": 0.50,
            "minimum_action_recall": 0.35,
            "static_margin": 0.10,
            "shifted_margin": 0.10,
            "switch_rate_error": SWITCH_RATE_ERROR_THRESHOLD,
        },
        "test_sessions": ["session-011", "session-012"],
        "test_accessed": True,
        "raw_frames_video_or_paths_persisted": False,
        "device_input_allowed": False,
        "shadow_allowed": passed,
    }
    _write_frozen_json(output, report)
    return report


def _v27_diagnostic_rows(
    root: Path, session_path: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    session = _large_existing(session_path)
    if session.parent != root / "diagnostics":
        raise T8Error("T8-v2.7 calibration session must be one direct diagnostic session")
    metadata = _v25_session_metadata(session, diagnostic=True)
    relative = session.relative_to(root).as_posix()
    rows = [
        {
            "session": relative,
            "path": shard["name"],
            "sha256": shard["sha256"],
            "frame_shards": metadata["frame_shards"],
        }
        for shard in cast(list[dict[str, object]], metadata["shards"])
    ]
    return rows, metadata


def run_t8_v27_calibration_pilot(
    *,
    dataset_root: Path,
    train_session: Path,
    dev_session: Path,
    source_model: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 8,
) -> dict[str, object]:
    if V27_FROZEN:
        raise T8Error("T8-v2.7 is frozen failed; further calibration is disabled")
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise T8Error("T8-v2.7 calibration settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    root = _large_existing(dataset_root)
    train_rows, train_meta = _v27_diagnostic_rows(root, train_session)
    dev_rows, dev_meta = _v27_diagnostic_rows(root, dev_session)
    if (
        train_meta["session_sha256"] == dev_meta["session_sha256"]
        or train_meta["layout_sha256"] != dev_meta["layout_sha256"]
        or train_meta["action_contract_sha256"] != dev_meta["action_contract_sha256"]
    ):
        raise T8Error("T8-v2.7 calibration train/dev identities are invalid")
    target = torch.device(device)
    model_path = _large_existing(source_model)
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        source_state = load_file(model_path, device=str(target))
    except (OSError, SafetensorError) as exc:
        raise T8Error("T8-v2.7 source model is unreadable") from exc
    if (
        metadata.get("schema") != V26_CONDITIONAL_MODEL_SCHEMA
        or metadata.get("gate_decision_threshold") != str(V26_GATE_DECISION_THRESHOLD)
    ):
        raise T8Error("T8-v2.7 source model metadata is invalid")
    prefix = "features.encoder."
    encoder_state = {
        key.removeprefix(prefix): value
        for key, value in source_state.items()
        if key.startswith(prefix)
    }
    train_counts = cast(dict[str, int], train_meta["counts"])
    counts = np.asarray([train_counts[name] for name in ABILITIES[:4]], dtype=np.float32)
    if np.any(counts <= 0):
        raise T8Error("T8-v2.7 calibration train session lacks a class")
    gate_counts = np.asarray([counts[0], counts[1:].sum()], dtype=np.float32)
    gate_weights = torch.from_numpy(gate_counts.sum() / (2.0 * gate_counts)).to(target)
    action_weights = torch.from_numpy(counts[1:].sum() / (3.0 * counts[1:])).to(target)
    loader = _v25_shard_loader(_V25FrameCache())
    source = _V26ConditionalCombatRGB(encoder_state).to(target)
    source.load_state_dict(source_state, strict=True)
    source_pred, source_action, dev_y = _predict_v26_conditional(
        source, root, dev_rows, target, batch_size, loader=loader
    )
    source_metrics = _combat_metrics(source_pred, source_action, dev_y)
    normal, state = _fit_v26_conditional(
        root,
        train_rows,
        dev_rows,
        encoder_state,
        target,
        batch_size,
        gate_weights,
        action_weights,
        shuffled=False,
        initial_state=source_state,
        heads_only=True,
        learning_rate=1e-3,
        epochs=20,
        balanced_sampling=True,
        loader=loader,
    )
    shuffled, _ = _fit_v26_conditional(
        root,
        train_rows,
        dev_rows,
        encoder_state,
        target,
        batch_size,
        gate_weights,
        action_weights,
        shuffled=True,
        initial_state=source_state,
        heads_only=True,
        learning_rate=1e-3,
        epochs=20,
        balanced_sampling=True,
        loader=loader,
    )
    calibrated = _V26ConditionalCombatRGB(encoder_state).to(target)
    calibrated.load_state_dict(state, strict=True)
    static_pred, static_action, static_y = _predict_v26_conditional(
        calibrated, root, dev_rows, target, batch_size, "static", loader=loader
    )
    shifted_pred, shifted_action, shifted_y = _predict_v26_conditional(
        calibrated, root, dev_rows, target, batch_size, "shifted", loader=loader
    )
    if not np.array_equal(dev_y, static_y) or not np.array_equal(dev_y, shifted_y):
        raise T8Error("T8-v2.7 controls changed dev labels")
    metrics = cast(dict[str, object], normal["metrics"])
    four = cast(dict[str, object], metrics["four_class"])
    action = cast(dict[str, object], metrics["action_only"])
    normal_f1 = cast(float, four["macro_f1"])
    source_f1 = cast(
        float, cast(dict[str, object], source_metrics["four_class"])["macro_f1"]
    )
    shuffled_f1 = cast(
        float,
        cast(
            dict[str, object], cast(dict[str, object], shuffled["metrics"])["four_class"]
        )["macro_f1"],
    )
    static_metrics = _combat_metrics(static_pred, static_action, dev_y)
    shifted_metrics = _combat_metrics(shifted_pred, shifted_action, dev_y)
    static_f1 = cast(
        float, cast(dict[str, object], static_metrics["four_class"])["macro_f1"]
    )
    shifted_f1 = cast(
        float, cast(dict[str, object], shifted_metrics["four_class"])["macro_f1"]
    )
    margins = {
        "normal_minus_source_macro_f1": normal_f1 - source_f1,
        "normal_minus_shuffled_macro_f1": normal_f1 - shuffled_f1,
        "normal_minus_static_macro_f1": normal_f1 - static_f1,
        "normal_minus_shifted_macro_f1": normal_f1 - shifted_f1,
    }
    passed = bool(
        cast(float, four["accuracy"]) >= 0.70
        and normal_f1 >= 0.50
        and min(cast(list[float], action["per_class_recall"])) >= 0.35
        and margins["normal_minus_source_macro_f1"] >= 0.15
        and margins["normal_minus_shuffled_macro_f1"] >= 0.15
        and margins["normal_minus_static_macro_f1"] >= 0.05
        and margins["normal_minus_shifted_macro_f1"] >= 0.05
    )
    report: dict[str, object] = {
        "schema_version": V27_CALIBRATION_SCHEMA,
        "status": "CALIBRATION_SIGNAL_FOUND" if passed else "CALIBRATION_DIAGNOSIS_REQUIRED",
        "strict_passed": passed,
        "source_model_sha256": _sha(model_path),
        "train_session_sha256": train_meta["session_sha256"],
        "dev_session_sha256": dev_meta["session_sha256"],
        "layout_sha256": train_meta["layout_sha256"],
        "gate_decision_threshold": V26_GATE_DECISION_THRESHOLD,
        "trainable_scope": "conditional_gate_and_action_heads_only",
        "epochs": 20,
        "learning_rate": 1e-3,
        "source_dev_metrics": source_metrics,
        "normal": normal,
        "shuffled": shuffled,
        "static_frame": static_metrics,
        "time_shift_2000ms": shifted_metrics,
        "margins": margins,
        "thresholds": {
            "accuracy": 0.70,
            "macro_f1": 0.50,
            "minimum_action_recall": 0.35,
            "source_improvement": 0.15,
            "shuffled_margin": 0.15,
            "static_margin": 0.05,
            "shifted_margin": 0.05,
        },
        "diagnostic_only": True,
        "formal_training_allowed": False,
        "test_accessed": False,
        "shadow_allowed": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        model_output = staging / "model.safetensors"
        save_file(
            state,
            model_output,
            metadata={
                "schema": V27_MODEL_SCHEMA,
                "source_model_sha256": _sha(model_path),
                "gate_decision_threshold": str(V26_GATE_DECISION_THRESHOLD),
            },
        )
        report["model_sha256"] = _sha(model_output)
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _frozen_video_features(
    frames: np.ndarray,
    encoder_state: Mapping[str, torch.Tensor],
    device: torch.device,
    window_batch_size: int,
) -> np.ndarray:
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(device).eval()
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frames), window_batch_size):
            batch = frames[start : start + window_batch_size]
            tensor = _v2_tensor(batch, device)
            encoded = cast(
                torch.Tensor,
                encoder(tensor.reshape(len(batch) * TOUCH_WINDOW_FRAMES, 3, 128, 128)),
            )
            features.append(
                encoded.reshape(len(batch), TOUCH_WINDOW_FRAMES, 512).cpu().numpy()
            )
    return np.concatenate(features).astype(np.float32, copy=False)


def _fit_video_three_class(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    shuffled: bool,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    labels = train_y.copy()
    if shuffled:
        labels = labels[np.random.default_rng(0).permutation(len(labels))]
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    if np.any(counts == 0):
        raise T8Error("three-class video train split lacks a class")
    weights_array = np.minimum(np.sqrt(counts.sum() / counts), 5.0)
    weights = torch.from_numpy(weights_array / weights_array.mean()).to(device)
    torch.manual_seed(0)
    model = _VideoCombatTemporal().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state = math.inf, 0, {}
    for epoch in range(1, 13):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(train_x[rows]).to(device))
            loss = nn.functional.cross_entropy(
                logits, torch.from_numpy(labels[rows]).to(device), weight=weights
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        total_loss, total_rows = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(dev_x), batch_size):
                batch_x = torch.from_numpy(dev_x[start : start + batch_size]).to(device)
                batch_y = torch.from_numpy(dev_y[start : start + batch_size]).to(device)
                current = nn.functional.cross_entropy(
                    model(batch_x), batch_y, weight=weights, reduction="sum"
                )
                total_loss += float(current.item())
                total_rows += len(batch_y)
        validation_loss = total_loss / total_rows
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if not best_state:
        raise T8Error("three-class video diagnostic did not validate")
    model.load_state_dict(best_state, strict=True)
    predicted: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dev_x), batch_size):
            predicted.append(
                model(torch.from_numpy(dev_x[start : start + batch_size]).to(device))
                .argmax(1)
                .cpu()
                .numpy()
            )
    metrics = _head_metrics(np.concatenate(predicted), dev_y, 3)
    return (
        {
            "best_epoch": best_epoch,
            "best_dev_weighted_cross_entropy": best_loss,
            "metrics": metrics,
            "shuffled_labels": shuffled,
        },
        best_state,
    )


def run_t8_video_three_class_pilot(
    *,
    dataset_root: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 64,
    retrospective: bool = False,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise T8Error("three-class video pilot settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    root = _large_existing(dataset_root)
    adapter = _large_existing(adapter_checkpoint)
    output = _large_new(output_dir)
    target = torch.device(device)
    train_frames, train_y, manifest_sha = _load_video_three_class_split(
        root, "train", retrospective=retrospective
    )
    dev_frames, dev_y, dev_manifest_sha = _load_video_three_class_split(
        root, "dev", retrospective=retrospective
    )
    if manifest_sha != dev_manifest_sha:
        raise T8Error("three-class video manifest changed during setup")
    encoder, adapter_meta = _load_v2_adapter(adapter, target)
    feature_batch_size = min(batch_size, 32)
    train_x = _frozen_video_features(train_frames, encoder, target, feature_batch_size)
    dev_x = _frozen_video_features(dev_frames, encoder, target, feature_batch_size)
    del train_frames, dev_frames
    normal, normal_state = _fit_video_three_class(
        train_x, train_y, dev_x, dev_y, target, batch_size, shuffled=False
    )
    shuffled, _ = _fit_video_three_class(
        train_x, train_y, dev_x, dev_y, target, batch_size, shuffled=True
    )
    normal_metrics = cast(dict[str, object], normal["metrics"])
    shuffled_metrics = cast(dict[str, object], shuffled["metrics"])
    normal_macro_f1 = float(cast(float, normal_metrics["macro_f1"]))
    shuffled_macro_f1 = float(cast(float, shuffled_metrics["macro_f1"]))
    plurality = int(np.bincount(train_y, minlength=3).argmax())
    plurality_accuracy = float(np.mean(dev_y == plurality))
    recalls = cast(list[float], normal_metrics["per_class_recall"])
    gate = bool(
        float(cast(float, normal_metrics["accuracy"])) >= plurality_accuracy + 0.10
        and float(cast(float, normal_metrics["macro_recall"])) >= 0.55
        and normal_macro_f1 >= 0.55
        and min(recalls) >= 0.30
        and normal_macro_f1 - shuffled_macro_f1 >= 0.15
    )
    report: dict[str, object] = {
        "schema_version": (
            VIDEO_RETROSPECTIVE_TRAINING_SCHEMA
            if retrospective
            else VIDEO_THREE_CLASS_TRAINING_SCHEMA
        ),
        "task": (
            "retrospective_action_recognition"
            if retrospective
            else "strict_causal_next_action_diagnostic"
        ),
        "status": "PILOT_PASSED" if gate else "PILOT_DIAGNOSIS_REQUIRED",
        "seed": 0,
        "epochs": 12,
        "selection_metric": "dev_weighted_cross_entropy_only",
        "encoder": "selected_t8_v2_video_adapter_frozen",
        "adapter_sha256": _sha(adapter),
        "adapter_source_sha256": adapter_meta.get("v5_source_model_sha256"),
        "dataset_manifest_sha256": manifest_sha,
        "train_rows": len(train_y),
        "dev_rows": len(dev_y),
        "combat_vocabulary": list(ABILITIES[1:4]),
        "abstained_classes": ["skill3"],
        "plurality_dev_accuracy": plurality_accuracy,
        "normal": normal,
        "shuffled": shuffled,
        "normal_minus_shuffled_macro_f1": normal_macro_f1 - shuffled_macro_f1,
        "gate_passed": gate,
        "test_accessed": False,
        "formal_policy_training_allowed": False,
        "shadow_allowed": False,
    }
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        save_file(
            normal_state,
            staging / "combat-temporal-seed0.safetensors",
            metadata={
                "schema": (
                    VIDEO_RETROSPECTIVE_TRAINING_SCHEMA
                    if retrospective
                    else VIDEO_THREE_CLASS_TRAINING_SCHEMA
                ),
                "adapter_sha256": _sha(adapter),
                "dataset_manifest_sha256": manifest_sha,
                "seed": "0",
            },
        )
        report["model_sha256"] = _sha(staging / "combat-temporal-seed0.safetensors")
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def train_t8_v2_pilot(
    *,
    dataset_root: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    epochs: int = 12,
    batch_size: int = 32,
    shuffled_labels: bool = False,
    lineage: str = "v2",
    split_path: Path | None = None,
) -> dict[str, object]:
    if (
        device not in {"cpu", "cuda"}
        or epochs != 12
        or batch_size < 1
        or lineage not in {"v2", "v2.1"}
    ):
        raise T8Error("T8-v2 pilot settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise T8Error("CUDA is unavailable")
    output = _large_new(output_dir)
    data = (
        load_t8_v21_data(dataset_root, split_path)
        if lineage == "v2.1"
        else load_t8_v2_data(dataset_root, include_test=False)
    )
    target = torch.device(device)
    train_x, train_y = _v2_examples(data, "train")
    dev_x, dev_y = _v2_examples(data, "dev")
    if shuffled_labels:
        train_y = train_y.copy()
        train_y[:, :4] = train_y[np.random.default_rng(0).permutation(len(train_y)), :4]
    encoder, adapter_meta = _load_v2_adapter(_large_existing(adapter_checkpoint), target)
    torch.manual_seed(0)
    model = T8V2FactorizedActor().to(target)
    model.encoder.load_state_dict(encoder, strict=True)
    weights = _v2_class_weights(train_y, target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, best_epoch, best_state = math.inf, 0, {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = _v2_loss(
                model(_v2_tensor(train_x[rows], target)),
                torch.from_numpy(train_y[rows, :4]).to(target),
                weights,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        with torch.no_grad():
            dev_outputs: list[list[torch.Tensor]] = [[], [], [], []]
            for start in range(0, len(dev_x), batch_size):
                for index, values in enumerate(
                    model(_v2_tensor(dev_x[start : start + batch_size], target))
                ):
                    dev_outputs[index].append(values)
            validation_loss = float(
                _v2_loss(
                    tuple(torch.cat(values) for values in dev_outputs),
                    torch.from_numpy(dev_y[:, :4]).to(target),
                    weights,
                ).item()
            )
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if not best_state:
        raise T8Error("T8-v2 pilot did not validate")
    model.load_state_dict(best_state, strict=True)
    predicted = _v2_predict(model, dev_x, target, batch_size)
    metrics = _v2_metrics(predicted, dev_y)
    plurality = _v2_plurality(train_y[:, :4])
    plurality_joint = float(
        (np.broadcast_to(plurality, predicted.shape) == dev_y[:, :4]).all(1).mean()
    )
    primary_gate = _v2_primary_gate(metrics, plurality_joint)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    try:
        model_name: str | None = None
        model_sha256: str | None = None
        if not shuffled_labels:
            model_name = "model-seed-0.safetensors"
            save_file(
                model.cpu().state_dict(),
                staging / model_name,
                metadata={
                    "schema": V21_MODEL_SCHEMA if lineage == "v2.1" else V2_MODEL_SCHEMA,
                    "config_sha256": V2_CONFIG_HASH,
                    "layout_sha256": data.sessions[0].layout_sha256,
                    "split_sha256": data.split_sha256,
                    "adapter_sha256": _sha(adapter_checkpoint),
                    "dataset_lineage": lineage,
                    "shuffled_labels": "false",
                },
            )
            model_sha256 = _sha(staging / model_name)
        report: dict[str, object] = {
            "schema_version": (
                V21_TRAINING_SCHEMA if lineage == "v2.1" else V2_TRAINING_SCHEMA
            ),
            "status": "CONTROL_RECORDED" if shuffled_labels else "CANDIDATE_RECORDED",
            "seed": 0,
            "best_epoch": best_epoch,
            "metrics": metrics,
            "plurality_joint": plurality_joint,
            "adapter_sha256": _sha(adapter_checkpoint),
            "adapter_source_sha256": adapter_meta.get("v5_source_model_sha256"),
            "split_sha256": data.split_sha256,
            "layout_sha256": data.sessions[0].layout_sha256,
            "dataset_lineage": lineage,
            "shuffled_labels": shuffled_labels,
            "primary_gate_passed": primary_gate,
            "test_accessed": False,
            "observation_end_lag_ms": 100,
            "model": model_name,
            "model_sha256": model_sha256,
            "raw_video_or_paths_persisted": False,
        }
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        return report
    except Exception:
        for item in staging.iterdir():
            item.unlink()
        staging.rmdir()
        raise


def run_t8_v2_pilot_pair(
    *,
    dataset_root: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    epochs: int = 12,
    batch_size: int = 32,
    lineage: str = "v2",
    split_path: Path | None = None,
) -> dict[str, object]:
    output = _large_new(output_dir)
    normal_dir = output.parent / f"{output.name}-normal"
    shuffled_dir = output.parent / f"{output.name}-shuffled"
    normal = train_t8_v2_pilot(
        dataset_root=dataset_root,
        adapter_checkpoint=adapter_checkpoint,
        output_dir=normal_dir,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lineage=lineage,
        split_path=split_path,
    )
    shuffled = train_t8_v2_pilot(
        dataset_root=dataset_root,
        adapter_checkpoint=adapter_checkpoint,
        output_dir=shuffled_dir,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        shuffled_labels=True,
        lineage=lineage,
        split_path=split_path,
    )
    normal_joint = float(cast(float, cast(dict[str, object], normal["metrics"])["joint_exact"]))
    shuffled_joint = float(
        cast(float, cast(dict[str, object], shuffled["metrics"])["joint_exact"])
    )
    pilot_only = bool(
        lineage == "v2.1"
        and split_path is not None
        and _read_object(
            _large_existing(split_path), "T8-v2.1 pilot split is unreadable"
        ).get("schema_version")
        == V21_PILOT_SPLIT_SCHEMA
    )
    normal_heads = cast(
        dict[str, dict[str, object]], cast(dict[str, object], normal["metrics"])["heads"]
    )
    shuffled_heads = cast(
        dict[str, dict[str, object]], cast(dict[str, object], shuffled["metrics"])["heads"]
    )
    early_learnability = bool(
        normal_joint - shuffled_joint >= 0.05
        and float(cast(float, normal_heads["movement"]["macro_f1"]))
        - float(cast(float, shuffled_heads["movement"]["macro_f1"]))
        >= 0.10
        and float(cast(float, normal_heads["combat"]["macro_f1"]))
        - float(cast(float, shuffled_heads["combat"]["macro_f1"]))
        >= 0.10
    )
    shuffled_materially_failed = bool(
        early_learnability
        if pilot_only
        else shuffled_joint <= normal_joint - 0.10 and not shuffled["primary_gate_passed"]
    )
    admitted = bool(
        early_learnability
        if pilot_only
        else normal["primary_gate_passed"] and shuffled_materially_failed
    )
    report: dict[str, object] = {
        "schema_version": V21_TRAINING_SCHEMA if lineage == "v2.1" else V2_TRAINING_SCHEMA,
        "status": "PILOT_REVIEW_READY" if admitted else "PILOT_DIAGNOSIS_REQUIRED",
        "normal_report_sha256": _sha(normal_dir / "report.json"),
        "normal_model_sha256": normal["model_sha256"],
        "shuffled_report_sha256": _sha(shuffled_dir / "report.json"),
        "normal_joint_exact": normal_joint,
        "shuffled_joint_exact": shuffled_joint,
        "joint_gap": normal_joint - shuffled_joint,
        "pilot_only": pilot_only,
        "early_learnability_passed": early_learnability,
        "normal_primary_gate_passed": normal["primary_gate_passed"],
        "dataset_lineage": lineage,
        "shuffled_materially_failed": shuffled_materially_failed,
        "test_accessed": False,
        "next_stage": "MANUAL_PILOT_REVIEW",
    }
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report
