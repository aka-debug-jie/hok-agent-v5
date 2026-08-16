# ruff: noqa: E501, E302, E305, E701, E702
"""Strict V5 contracts; CPU paths are explicitly non-promoting contract smokes."""

from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import math
import os
import stat
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeGuard, cast

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18  # type: ignore[import-untyped]

from .arena import FactorizedAction, PixelArena, Side
from .policies import RandomPolicy
from .pre_ingest import READY as PRE_INGEST_READY
from .pre_ingest import PreIngestError, PreIngestEvidence, load_pre_ingest

ACTION_TYPES = ("wait", "forward", "backward", "attack_hero", "attack_tower", "attack_crystal")
VIEW_KEYS = ("t-100", "t", "t+100")
AUGMENTATION_KEYS = ("identity", "color-v1", "translate-x2-y-2-v1")
SPLITS = {"train", "dev", "test"}
SOURCES = {"source", "target"}
NPZ_FIELDS = (
    "frames",
    "session_hash",
    "timestamp_ms",
    "pts",
    "time_base",
    "rotation_degrees",
    "frame_hash",
    "privacy_transform_sha256",
    "owner_attestation_sha256",
    "privacy_review_sha256",
    "alignment_hash",
    "split",
    "source",
)
NPZ_DTYPES = {
    "frames": np.dtype("uint8"),
    "session_hash": np.dtype("<U64"),
    "timestamp_ms": np.dtype("int64"),
    "pts": np.dtype("int64"),
    "time_base": np.dtype("int64"),
    "rotation_degrees": np.dtype("int16"),
    "frame_hash": np.dtype("<U64"),
    "privacy_transform_sha256": np.dtype("<U64"),
    "owner_attestation_sha256": np.dtype("<U64"),
    "privacy_review_sha256": np.dtype("<U64"),
    "alignment_hash": np.dtype("<U64"),
    "split": np.dtype("<U5"),
    "source": np.dtype("<U6"),
}
FORBIDDEN_DERIVED_KEYS = {
    "label",
    "labels",
    "legal",
    "legal_mask",
    "legal_actions",
    "reward",
    "state",
    "structured_state",
    "path",
    "raw_path",
    "audio",
    "account",
    "audit",
    "reviewer",
    "privacy_mask",
}
RENDERER_SPEC = {
    "id": "pixelarena-v5-source-rgb128-v1",
    "shape": [128, 128, 3],
    "dtype": "uint8",
    "hud": "six-health-bars-no-tick",
    "distortion": "color-quantize-downsample-blur-shift-le4",
}
PRIVACY_SPEC = {
    "transform_id": "rotation-normalized-zero-redaction-letterbox-rgb128-v1",
    "rotation_degrees": [0, 90, 180, 270],
    "rotation_direction": "clockwise",
    "output_shape": [128, 128, 3],
    "output_dtype": "uint8",
    "zero_redaction": "owner-authorized-no-pixel-redaction",
    "privacy_recipe": {
        "name": "aspect-fit-letterbox-no-crop-no-mask-v1",
        "aspect_fit": True,
        "resize": "bilinear",
        "padding": {"style": "black", "mode": "letterbox"},
        "crop": False,
        "mask": False,
    },
}
MANIFEST_SCHEMA = "hok-agent-v5-manifest-v2"
PRIVACY_CONTEXT_SCHEMA = "hok-agent-v5-privacy-context-v1"
OWNER_ATTESTATION_SCHEMA = "hok-agent-v5-owner-attestation-v1"
COMPONENT_COHORT_SCHEMA = "hok-agent-v5-component-cohort-v1"
SOURCE_SCHEMA = "hok-agent-v5-source-dataset-v1"
SOURCE_PRODUCER_SCHEMA = "hok-agent-v5-source-producer-v1"
SOURCE_BASELINE_SCHEMA = "hok-agent-v5-source-baseline-v1"
MODEL_SCHEMA = "hok-agent-v5-model-v1"
RESUME_SCHEMA = "hok-agent-v5-simsiam-resume-v1"
PREDICTION_SCHEMA = "hok-agent-v5-predictions-v1"
MODEL_PREDICTION_SCHEMA = "hok-agent-v5-model-predictions-v2"
AUDIT_SCHEMA = "hok-agent-v5-audit-v1"
LEDGER_SCHEMA = "hok-agent-v5-mean-teacher-ledger-v1"
RELEASE_SCHEMA = "hok-agent-v5-release-v1"
COLLAPSE_BLOCK = "NO_ACCEPTED_REAL_DOMAIN_COLLAPSE_EVIDENCE"
SOURCE_SEEDS = (0, 1, 2)
SOURCE_EPISODES = 128
SOURCE_TRAINING = {
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "epochs": 50,
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
}
COLLAPSE_THRESHOLDS = {
    "variance_ratio_min": 0.25,
    "effective_rank_ratio_min": 0.50,
    "top_eigen_share_increase_max": 0.10,
    "black_constant_distance_ratio_min": 0.80,
}


class AlignmentError(ValueError):
    """Malformed or incomplete V5 evidence."""


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    component_id: str
    parent_id: str | None = None
    near_duplicates: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidatePrediction:
    session_id: str
    timestamp_ms: int
    model_id: str
    view_id: str
    probs: tuple[float, ...]
    augmentation_id: str = "identity"
    ood_score: float = 0.0
    black_control_ok: bool = True
    constant_control_ok: bool = True
    cut: bool = False


@dataclass(frozen=True)
class PseudoLabel:
    session_id: str
    timestamp_ms: int
    class_id: int
    confidence: float
    margin: float
    evidence_hash: str


@dataclass(frozen=True)
class PseudoFilterReport:
    groups: int
    accepted: int
    rejected_by_reason: dict[str, int]
    filter_floor_met: bool


@dataclass(frozen=True)
class SourceDataset:
    frames: np.ndarray
    labels: np.ndarray
    split: np.ndarray | None = None


@dataclass(frozen=True)
class AcceptedPseudoDataset:
    frames: np.ndarray
    labels: np.ndarray
    artifact_path: Path
    artifact_sha256: str
    manifest_sha256: str
    predictions_sha256: str


@dataclass(frozen=True)
class ModelPredictionEvidence:
    directory: Path
    manifest_sha256: str
    source_model_sha256: str
    adapted_model_sha256: str
    config_sha256: str
    shard_paths: tuple[Path, ...]


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    checkpoint_sha256: str
    config_hash: str
    metrics: dict[str, object]
    promoting: bool


@dataclass(frozen=True)
class AuditClip:
    clip_id: str
    session_hash: str
    timestamp_ms: int
    frame_path: Path


@dataclass(frozen=True)
class AuditLabel:
    clip_id: str
    reviewer: str
    observed_action: str
    validity: bool


@dataclass(frozen=True)
class AuditPrediction:
    clip_id: str
    action: str
    accepted: bool
    confidence: float
    baselines: tuple[str, str]


@dataclass(frozen=True)
class SourceRegression:
    accuracy_before: float
    accuracy_after: float
    recall_before: tuple[float, ...]
    recall_after: tuple[float, ...]


@dataclass(frozen=True)
class ReleaseGate:
    passed: bool
    allowed_classes: tuple[str, ...]
    kappa: float
    overall_precision: float
    class_precision: dict[str, float]
    coverage: float
    ood_false_accept: float
    baseline_deltas: tuple[float, float]
    source_accuracy_drop: float
    max_recall_drop: float
    evidence_hash: str
    collapse_status: str
    release_path: Path | None


@dataclass(frozen=True)
class V5Manifest:
    path: Path
    manifest_sha256: str
    pre_ingest_sha256: str
    component_cohort_sha256: str
    privacy_context_sha256: str
    split_binding_sha256: str
    privacy_transform_sha256: str
    owner_attestation_sha256: str
    session_splits: dict[str, str]
    shard_paths: tuple[Path, ...]
    shard_sha256: dict[Path, str]
    split_row_counts: dict[str, int]
    shard_bindings: dict[Path, ManifestShardBinding] = field(default_factory=dict)
    privacy_reviews: dict[str, str] = field(default_factory=dict)
    snapshot_dir: Path | None = None
    snapshot_paths: dict[Path, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestShardBinding:
    row_count: int
    session_hashes: frozenset[str]
    split: str
    source: str


@dataclass(frozen=True)
class PrivacyBinding:
    privacy_context_sha256: str
    privacy_transform_sha256: str
    owner_attestation_sha256: str
    privacy_reviews: dict[str, str]
    components: dict[str, str]


@dataclass(frozen=True)
class ComponentCohort:
    component_cohort_sha256: str
    pre_ingest_sha256: str
    owner_attestation_sha256: str
    component_hashes: frozenset[str]


OwnerComponentConfirmation = ComponentCohort


@dataclass(frozen=True)
class BoundV5Release:
    release_sha256: str
    model_sha256: str
    allowed_classes: tuple[str, ...]
    class_thresholds: dict[str, float]
    manifest_sha256: str = ""
    split_binding_sha256: str = ""


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value: object) -> int:
    return int(cast(Any, value))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _is_sha(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and _valid_sha(value)


_PUBLIC_KEYS = (
    "side",
    "self_position",
    "opponent_position",
    "self_health",
    "opponent_health",
    "own_tower_health",
    "enemy_tower_health",
    "own_crystal_health",
    "enemy_crystal_health",
)


def strip_to_public_observation(observation: Mapping[str, object]) -> dict[str, object]:
    return {key: observation[key] for key in _PUBLIC_KEYS if key in observation}


def _action_name(action: object) -> str:
    get = (
        action.get
        if isinstance(action, Mapping)
        else lambda key, default="": getattr(action, key, default)
    )
    kind, target, direction = (
        str(get("action_type", "")),
        str(get("target", "")),
        str(get("direction", "")),
    )
    if kind == "wait":
        return "wait"
    if kind == "move" and direction in {"forward", "backward"}:
        return direction
    if kind == "attack" and target == "enemy_hero":
        return "attack_hero"
    if kind == "attack" and target == "enemy_tower":
        return "attack_tower"
    if kind == "attack" and target == "enemy_crystal":
        return "attack_crystal"
    raise AlignmentError("action outside exact V5 six-class vocabulary")


class CausalSourceTeacher:
    def select(self, observation: Mapping[str, object], legal: Sequence[object]) -> object:
        public = strip_to_public_observation(observation)
        indexed: dict[str, object] = {}
        for action in legal:
            indexed.setdefault(_action_name(action), action)
        if "wait" not in indexed:
            raise AlignmentError("legal set must include wait")
        self_hp = _integer(public.get("self_health", 0))
        opponent_hp = _integer(public.get("opponent_health", 0))
        priorities = (
            (True, "attack_crystal"),
            (True, "attack_tower"),
            (self_hp >= opponent_hp, "attack_hero"),
            (self_hp < opponent_hp, "backward"),
            (self_hp < opponent_hp, "wait"),
            (True, "forward"),
            (True, "wait"),
        )
        for condition, name in priorities:
            if condition and name in indexed:
                return indexed[name]
        raise AssertionError("wait rule is unreachable")


def _draw_bar(draw: ImageDraw.ImageDraw, y: int, value: int, maximum: int, color: str) -> None:
    draw.rectangle((8, y, 119, y + 4), fill="#20252d")
    width = round(111 * min(1.0, max(0.0, value / maximum)))
    if width:
        draw.rectangle((8, y, 8 + width, y + 4), fill=color)


def _shift_without_wrap(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(frame)
    src_x = slice(max(0, -dx), min(128, 128 - dx))
    src_y = slice(max(0, -dy), min(128, 128 - dy))
    dst_x = slice(max(0, dx), min(128, 128 + dx))
    dst_y = slice(max(0, dy), min(128, 128 + dy))
    out[dst_y, dst_x] = frame[src_y, src_x]
    return out


def source_render_128_rgb(observation: Mapping[str, object], *, render_seed: int = 0) -> np.ndarray:
    """Render an independent deterministic source view with no tick/progress channel."""
    public = strip_to_public_observation(observation)
    if isinstance(render_seed, bool) or not isinstance(render_seed, int):
        raise AlignmentError("render_seed must be an integer")
    rng = np.random.default_rng(render_seed)
    side = str(public.get("side", "blue"))
    base = (30, 55, 72) if side == "blue" else (65, 43, 52)
    image = Image.new("RGB", (128, 128), base)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 45, 127, 127), fill=(46, 72, 55))
    draw.line((8, 102, 120, 102), fill=(151, 135, 91), width=5)
    own_x = 12 + round(10.4 * min(10, max(0, _integer(public.get("self_position", 0)))))
    opp_x = 12 + round(10.4 * min(10, max(0, _integer(public.get("opponent_position", 0)))))
    draw.ellipse((own_x - 5, 91, own_x + 5, 101), fill="#23bde2")
    draw.ellipse((opp_x - 5, 103, opp_x + 5, 113), fill="#df5148")
    bars = (
        (2, "self_health", 6, "#58cf72"),
        (8, "opponent_health", 6, "#dd5b62"),
        (14, "own_tower_health", 4, "#dbc267"),
        (20, "enemy_tower_health", 4, "#b99b50"),
        (26, "own_crystal_health", 6, "#bd73d2"),
        (32, "enemy_crystal_health", 6, "#9755ac"),
    )
    for y, key, maximum, bar_color in bars:
        _draw_bar(draw, y, _integer(public.get(key, 0)), maximum, bar_color)
    pixels = np.asarray(image, dtype=np.int16)
    pixels = np.clip(pixels + rng.integers(-7, 8, size=(1, 1, 3)), 0, 255).astype(np.uint8)
    pixels = (pixels // 8 * 8).astype(np.uint8)
    compressed = Image.fromarray(pixels).resize((112, 112), Image.Resampling.BILINEAR)
    compressed = compressed.resize((128, 128), Image.Resampling.BILINEAR)
    blurred = np.asarray(compressed.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.uint8)
    return _shift_without_wrap(blurred, int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))


def source_renderer_hash() -> str:
    related = (
        _json,
        _sha,
        _integer,
        strip_to_public_observation,
        _draw_bar,
        _shift_without_wrap,
        source_render_128_rgb,
    )
    implementation = "\n".join(inspect.getsource(function) for function in related)
    return _sha((_json(RENDERER_SPEC) + _json(_PUBLIC_KEYS) + implementation).encode())


def _read_regular(path: Path, suffix: str | None = None) -> bytes:
    if suffix is not None and path.suffix != suffix:
        raise AlignmentError(f"artifact must use {suffix}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise AlignmentError("artifact must be a non-symlink regular file") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AlignmentError("artifact must be a non-symlink regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _file_sha(path: Path) -> str:
    return _sha(_read_regular(path))


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AlignmentError("JSON keys must be unique")
        result[key] = value
    return result


def _strict_json_bytes(data: bytes) -> object:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AlignmentError(f"invalid number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AlignmentError("invalid strict UTF-8 JSON") from exc


def _load_json(path: Path) -> dict[str, object]:
    return _load_json_bytes(_read_regular(path, ".json"))


def _load_json_bytes(data: bytes) -> dict[str, object]:
    value = _strict_json_bytes(data)
    if type(value) is not dict:
        raise AlignmentError("JSON artifact must be an object")
    return cast(dict[str, object], value)


def _object(value: object, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != fields:
        raise AlignmentError(f"{name} fields are not exact")
    return cast(dict[str, object], value)


def _self_hash(payload: Mapping[str, object], field: str) -> str:
    supplied = payload.get(field)
    if not isinstance(supplied, str) or not _valid_sha(supplied):
        raise AlignmentError(f"{field} must be lowercase SHA-256")
    if (
        _sha(_json({key: value for key, value in payload.items() if key != field}).encode())
        != supplied
    ):
        raise AlignmentError(f"{field} mismatch")
    return supplied


def _number(value: object, *, low: float | None = None, high: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise AlignmentError("numeric evidence must be finite and non-boolean")
    result = float(value)
    if (low is not None and result < low) or (high is not None and result > high):
        raise AlignmentError("numeric evidence is outside its domain")
    return result


def _write_exclusive(path: Path, data: bytes, *, sync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            if sync:
                os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_pair_exclusive(
    first_path: Path,
    first_bytes: bytes,
    second_path: Path,
    second_bytes: bytes,
) -> None:
    _write_exclusive(first_path, first_bytes)
    try:
        _write_exclusive(second_path, second_bytes)
    except BaseException:
        first_path.unlink(missing_ok=True)
        raise


class _Groups:
    def __init__(self, ids: Sequence[str]) -> None:
        self.parent = {value: value for value in ids}

    def find(self, value: str) -> str:
        while value != self.parent[value]:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _linked_groups(records: Sequence[SessionRecord]) -> tuple[tuple[str, ...], ...]:
    ids = [row.session_id for row in records]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise AlignmentError("session ids must be unique and non-empty")
    known, groups = set(ids), _Groups(ids)
    components_by_id: dict[str, str] = {}
    for row in records:
        if (
            not _valid_sha(row.component_id)
            or row.parent_id == row.session_id
            or len(set(row.near_duplicates)) != len(row.near_duplicates)
            or row.session_id in row.near_duplicates
        ):
            raise AlignmentError("invalid component/self/duplicate relationship")
        references = ({row.parent_id} if row.parent_id is not None else set()) | set(
            row.near_duplicates
        )
        if not references.issubset(known):
            raise AlignmentError("parent and near-duplicate references must be known")
        if row.component_id in components_by_id:
            groups.union(row.session_id, components_by_id[row.component_id])
        else:
            components_by_id[row.component_id] = row.session_id
        for reference in references:
            groups.union(row.session_id, reference)
    components: dict[str, list[str]] = defaultdict(list)
    for session_id in ids:
        components[groups.find(session_id)].append(session_id)
    return tuple(
        sorted(
            (tuple(sorted(values)) for values in components.values()), key=lambda values: values[0]
        )
    )


def _rgb(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim == 2:
        value = np.repeat(value[:, :, None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] not in {3, 4}:
        raise AlignmentError("frame must be HWC RGB/RGBA")
    value = value[:, :, :3]
    if value.dtype != np.uint8:
        raise AlignmentError("frames must be uint8")
    return value


def zero_redaction_letterbox_rgb(frame: np.ndarray, rotation_degrees: int) -> np.ndarray:
    if isinstance(rotation_degrees, bool) or rotation_degrees not in {0, 90, 180, 270}:
        raise AlignmentError("rotation_degrees must be one of 0/90/180/270")
    value = np.rot90(_rgb(frame), k=-(rotation_degrees // 90))
    scale = min(128 / value.shape[1], 128 / value.shape[0])
    size = (max(1, round(value.shape[1] * scale)), max(1, round(value.shape[0] * scale)))
    resized = np.asarray(Image.fromarray(value).resize(size, Image.Resampling.BILINEAR))
    output = np.zeros((128, 128, 3), dtype=np.uint8)
    x, y = (128 - size[0]) // 2, (128 - size[1]) // 2
    output[y : y + size[1], x : x + size[0]] = resized
    return output


def privacy_transform_hash() -> str:
    return _sha(
        (
            _json(PRIVACY_SPEC)
            + inspect.getsource(_rgb)
            + inspect.getsource(zero_redaction_letterbox_rgb)
        ).encode()
    )


def load_owner_attestation(path: Path) -> str:
    fields = {
        "schema_version",
        "recording_owner",
        "local_research_only",
        "zero_redaction_authorized",
        "redistribution",
        "owner_attestation_sha256",
    }
    payload = _object(_load_json(path), fields, "owner attestation")
    if (
        payload["schema_version"] != OWNER_ATTESTATION_SCHEMA
        or payload["recording_owner"] is not True
        or payload["local_research_only"] is not True
        or payload["zero_redaction_authorized"] is not True
        or payload["redistribution"] is not False
    ):
        raise AlignmentError("owner attestation does not authorize local privacy review")
    return _self_hash(payload, "owner_attestation_sha256")


def load_privacy_context(context_path: Path, owner_attestation_path: Path) -> PrivacyBinding:
    owner_hash = load_owner_attestation(owner_attestation_path)
    fields = {
        "schema_version",
        "transform",
        "privacy_transform_sha256",
        "owner_attestation_sha256",
        "reviews",
        "privacy_context_sha256",
    }
    payload = _object(_load_json(context_path), fields, "privacy context")
    if (
        payload["schema_version"] != PRIVACY_CONTEXT_SCHEMA
        or payload["transform"] != PRIVACY_SPEC
        or payload["privacy_transform_sha256"] != privacy_transform_hash()
        or payload["owner_attestation_sha256"] != owner_hash
        or not isinstance(payload["reviews"], list)
    ):
        raise AlignmentError("privacy context binding is invalid")
    reviews: dict[str, str] = {}
    components: dict[str, str] = {}
    for value in payload["reviews"]:
        row = _object(
            value,
            {
                "session_hash",
                "component_hash",
                "zero_redaction_authorized",
                "privacy_review_sha256",
            },
            "privacy review",
        )
        session, component, authorized = (
            row["session_hash"],
            row["component_hash"],
            row["zero_redaction_authorized"],
        )
        if (
            not _is_sha(session)
            or not _is_sha(component)
            or session in reviews
            or authorized is not True
        ):
            raise AlignmentError("privacy review is invalid")
        reviews[session] = _self_hash(row, "privacy_review_sha256")
        components[session] = component
    if not reviews:
        raise AlignmentError("privacy context requires reviewed sessions")
    context_hash = _self_hash(payload, "privacy_context_sha256")
    return PrivacyBinding(
        context_hash, payload["privacy_transform_sha256"], owner_hash, reviews, components
    )


def load_component_cohort(
    path: Path | None, pre_ingest_sha256: str, owner_attestation_sha256: str
) -> ComponentCohort:
    if path is None:
        raise AlignmentError("component cohort artifact is required")
    fields = {
        "schema_version",
        "pre_ingest_sha256",
        "owner_attestation_sha256",
        "component_hashes",
        "component_cohort_sha256",
    }
    payload = _object(_load_json(path), fields, "component cohort evidence")
    components = payload["component_hashes"]
    if (
        payload["schema_version"] != COMPONENT_COHORT_SCHEMA
        or payload["pre_ingest_sha256"] != pre_ingest_sha256
        or payload["owner_attestation_sha256"] != owner_attestation_sha256
        or not isinstance(components, list)
        or len(components) < 12
        or components != sorted(components)
        or len(set(cast(list[object], components))) != len(components)
        or any(not _is_sha(value) for value in components)
    ):
        raise AlignmentError("component cohort binding is invalid")
    confirmation_hash = _self_hash(payload, "component_cohort_sha256")
    return ComponentCohort(
        confirmation_hash,
        pre_ingest_sha256,
        owner_attestation_sha256,
        frozenset(cast(list[str], components)),
    )


def load_owner_component_confirmation(
    path: Path | None, pre_ingest_sha256: str, owner_attestation_sha256: str
) -> OwnerComponentConfirmation:
    return load_component_cohort(path, pre_ingest_sha256, owner_attestation_sha256)


def _load_target_evidence(
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path | None,
) -> tuple[PreIngestEvidence, PrivacyBinding, OwnerComponentConfirmation]:
    try:
        pre_ingest = load_pre_ingest(pre_ingest_path)
    except PreIngestError as exc:
        raise AlignmentError("pre-ingest evidence is invalid") from exc
    if pre_ingest.disposition != PRE_INGEST_READY:
        raise AlignmentError("pre-ingest evidence is not ready for component split")
    privacy = load_privacy_context(privacy_context_path, owner_attestation_path)
    confirmation = load_component_cohort(
        owner_component_confirmation_path,
        pre_ingest.pre_ingest_sha256,
        privacy.owner_attestation_sha256,
    )
    if confirmation.component_hashes != set(pre_ingest.component_of.values()):
        raise AlignmentError("component cohort member is absent from pre-ingest evidence")
    if any(
        pre_ingest.component_of.get(session) != component
        or component not in confirmation.component_hashes
        for session, component in privacy.components.items()
    ):
        raise AlignmentError("privacy context component differs from pre-ingest component cohort")
    return pre_ingest, privacy, confirmation


def _session_hash(item: Mapping[str, object]) -> str:
    supplied = item.get("session_hash")
    if supplied is not None:
        value = str(supplied)
        if not _valid_sha(value):
            raise AlignmentError("session_hash must be lowercase SHA-256")
        return value
    session_id = str(item.get("session_id", ""))
    if not session_id:
        raise AlignmentError("session identity is required")
    return _sha(session_id.encode())


def write_npz_shards(
    records: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    shard_size: int = 256,
    pre_ingest_path: Path | None = None,
    privacy_context_path: Path | None = None,
    owner_attestation_path: Path | None = None,
    owner_component_confirmation_path: Path | None = None,
    name_prefix: str = "",
    start_index: int = 0,
    frames_already_normalized: bool = False,
) -> tuple[Path, ...]:
    if shard_size < 1 or start_index < 0 or Path(name_prefix).name != name_prefix:
        raise AlignmentError("shard_size must be positive")
    evidence_paths = (
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
    )
    if any(path is not None for path in evidence_paths) and any(
        path is None for path in evidence_paths
    ):
        raise AlignmentError("target evidence paths are an exact set")
    privacy = (
        None
        if privacy_context_path is None
        else _load_target_evidence(
            cast(Path, pre_ingest_path),
            privacy_context_path,
            cast(Path, owner_attestation_path),
            owner_component_confirmation_path,
        )[1]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for shard, offset in enumerate(range(0, len(records), shard_size)):
        rows = records[offset : offset + shard_size]
        columns: dict[str, list[Any]] = {key: [] for key in NPZ_FIELDS}
        for item in rows:
            if FORBIDDEN_DERIVED_KEYS.intersection(item):
                raise AlignmentError("derived shards contain forbidden training/privacy fields")
            split, source = str(item.get("split", "")), str(item.get("source", ""))
            if split not in SPLITS or source not in SOURCES:
                raise AlignmentError("invalid split/source enum")
            if source == "target" and privacy is None:
                raise AlignmentError("target frames require reviewed privacy context")
            session = _session_hash(item)
            if source == "target" and session not in cast(PrivacyBinding, privacy).privacy_reviews:
                raise AlignmentError("target session lacks privacy review")
            rotation = item.get("rotation_degrees")
            if source == "target" and rotation is None:
                raise AlignmentError("target rotation_degrees is required")
            if rotation is not None and (
                type(rotation) is not int or rotation not in {0, 90, 180, 270}
            ):
                raise AlignmentError("rotation_degrees must be one of 0/90/180/270")
            rotation_degrees = 0 if rotation is None else rotation
            source_frame = np.asarray(item["frame"])
            if frames_already_normalized:
                if source_frame.dtype != np.uint8 or source_frame.shape != (128, 128, 3):
                    raise AlignmentError("pre-normalized target frame has invalid dtype/shape")
                frame = source_frame
            else:
                frame = zero_redaction_letterbox_rgb(source_frame, rotation_degrees)
            timestamp, pts = _integer(item["timestamp_ms"]), _integer(item["pts"])
            base = item["time_base"]
            if not isinstance(base, Sequence) or isinstance(base, (str, bytes)) or len(base) != 2:
                raise AlignmentError("time_base must be an integer numerator/denominator pair")
            time_base = (int(base[0]), int(base[1]))
            if time_base[1] <= 0:
                raise AlignmentError("time_base denominator must be positive")
            frame_hash = _sha(frame.tobytes())
            transform_hash = privacy.privacy_transform_sha256 if privacy is not None else "0" * 64
            owner_hash = privacy.owner_attestation_sha256 if privacy is not None else "0" * 64
            review_hash = (
                privacy.privacy_reviews[session]
                if privacy is not None and session in privacy.privacy_reviews
                else "0" * 64
            )
            alignment = _sha(
                _json(
                    [
                        session,
                        timestamp,
                        pts,
                        time_base,
                        rotation_degrees,
                        frame_hash,
                        transform_hash,
                        owner_hash,
                        review_hash,
                        split,
                        source,
                    ]
                ).encode()
            )
            values = (
                frame,
                session,
                timestamp,
                pts,
                time_base,
                rotation_degrees,
                frame_hash,
                transform_hash,
                owner_hash,
                review_hash,
                alignment,
                split,
                source,
            )
            for key, value in zip(NPZ_FIELDS, values, strict=True):
                columns[key].append(value)
        payload = {key: np.asarray(value, dtype=NPZ_DTYPES[key]) for key, value in columns.items()}
        stem = (
            f"alignment-{shard:04d}"
            if not name_prefix and start_index == 0
            else f"{name_prefix}alignment-{start_index + shard:06d}"
        )
        path, buffer = output_dir / f"{stem}.npz", io.BytesIO()
        np.savez(buffer, **payload)  # type: ignore[arg-type]
        _write_exclusive(path, buffer.getvalue())
        paths.append(path)
    return tuple(paths)


def _load_npz_bytes(data: bytes, *, verify_rows: bool = True) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != set(NPZ_FIELDS):
                raise AlignmentError("NPZ fields are not exact")
            payload = {key: archive[key].copy() for key in NPZ_FIELDS}
    except (OSError, ValueError) as exc:
        if isinstance(exc, AlignmentError):
            raise
        raise AlignmentError("invalid NPZ artifact") from exc
    count = len(payload["frames"])
    if payload["frames"].shape != (count, 128, 128, 3) or payload["time_base"].shape != (count, 2):
        raise AlignmentError("invalid NPZ frame/time-base shape")
    for key in NPZ_FIELDS:
        if payload[key].dtype != NPZ_DTYPES[key] or len(payload[key]) != count:
            raise AlignmentError(f"invalid dtype/length for {key}")
    if not verify_rows:
        return payload
    for index in range(count):
        split, source = str(payload["split"][index]), str(payload["source"][index])
        session, frame_hash = str(payload["session_hash"][index]), str(payload["frame_hash"][index])
        bound_hashes = tuple(
            str(payload[key][index])
            for key in (
                "privacy_transform_sha256",
                "owner_attestation_sha256",
                "privacy_review_sha256",
            )
        )
        rotation_degrees = int(payload["rotation_degrees"][index])
        if (
            split not in SPLITS
            or source not in SOURCES
            or rotation_degrees not in {0, 90, 180, 270}
            or not _valid_sha(session)
            or not all(_valid_sha(value) for value in bound_hashes)
            or int(payload["time_base"][index][1]) <= 0
        ):
            raise AlignmentError("invalid NPZ enum/hash")
        if _sha(payload["frames"][index].tobytes()) != frame_hash:
            raise AlignmentError("frame hash mismatch")
        material = [
            session,
            int(payload["timestamp_ms"][index]),
            int(payload["pts"][index]),
            tuple(int(x) for x in payload["time_base"][index]),
            rotation_degrees,
            frame_hash,
            *bound_hashes,
            split,
            source,
        ]
        if _sha(_json(material).encode()) != str(payload["alignment_hash"][index]):
            raise AlignmentError("alignment hash mismatch")
    return payload


def _load_npz(path: Path, *, verify_rows: bool = True) -> dict[str, np.ndarray]:
    return _load_npz_bytes(_read_regular(path, ".npz"), verify_rows=verify_rows)


def load_npz_shards(paths: Sequence[Path]) -> tuple[dict[str, np.ndarray], ...]:
    return tuple(_load_npz(path) for path in paths)


def split_binding_hash(session_splits: Mapping[str, str]) -> str:
    return _sha(_json(sorted(session_splits.items())).encode())


def _verify_manifest_shard(
    shard: Mapping[str, np.ndarray],
    binding: ManifestShardBinding,
    session_splits: Mapping[str, str],
    privacy_transform_sha256: str,
    owner_attestation_sha256: str,
    privacy_reviews: Mapping[str, str],
    active_session: dict[str, str | None],
    closed_sessions: dict[str, set[str]],
    last_timestamp: dict[tuple[str, str], int],
) -> None:
    actual_sessions = {str(item) for item in shard["session_hash"]}
    if (
        len(shard["frames"]) != binding.row_count
        or actual_sessions != binding.session_hashes
        or set(map(str, shard["split"])) != {binding.split}
        or set(map(str, shard["source"])) != {binding.source}
        or any(session_splits[item] != binding.split for item in actual_sessions)
        or set(map(str, shard["privacy_transform_sha256"])) != {privacy_transform_sha256}
        or set(map(str, shard["owner_attestation_sha256"])) != {owner_attestation_sha256}
        or any(
            str(review) != privacy_reviews[str(session)]
            for session, review in zip(
                shard["session_hash"], shard["privacy_review_sha256"], strict=True
            )
        )
    ):
        raise AlignmentError("manifest shard row/session/split mismatch")
    for session, timestamp in zip(shard["session_hash"], shard["timestamp_ms"], strict=True):
        session_text, timestamp_value = str(session), int(timestamp)
        previous, key = active_session[binding.split], (binding.split, session_text)
        if previous != session_text:
            if session_text in closed_sessions[binding.split]:
                raise AlignmentError("manifest session frames must be contiguous")
            if previous is not None:
                closed_sessions[binding.split].add(previous)
            active_session[binding.split] = session_text
        if timestamp_value <= last_timestamp.get(key, -1):
            raise AlignmentError("manifest frames must be session/timestamp ordered and unique")
        last_timestamp[key] = timestamp_value


def _all_clean_components_lexicographic_ceil15pct_v1(components: Sequence[str]) -> dict[str, str]:
    values = tuple(sorted(components))
    if (
        len(values) < 12
        or len(set(values)) != len(values)
        or any(not _is_sha(value) for value in values)
    ):
        raise AlignmentError("automatic cohort requires at least 12 unique components")
    held_out = math.ceil(0.15 * len(values))
    train_count = len(values) - 2 * held_out
    if train_count < 8 or held_out < 2:
        raise AlignmentError("manifest components do not satisfy 8/2/2")
    return {
        component: "train"
        if index < train_count
        else "dev"
        if index < train_count + held_out
        else "test"
        for index, component in enumerate(values)
    }


def load_v5_manifest(
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path | None,
    shard_paths: Sequence[Path],
    *,
    verify_rows: bool = True,
    verify_shards: bool = True,
    snapshot_dir: Path | None = None,
) -> V5Manifest:
    fields = {
        "schema_version",
        "pre_ingest_sha256",
        "component_cohort_sha256",
        "privacy_context_sha256",
        "privacy_transform_sha256",
        "owner_attestation_sha256",
        "split_binding_sha256",
        "sessions",
        "shards",
        "manifest_sha256",
    }
    payload = _object(_load_json(manifest_path), fields, "manifest")
    if payload["schema_version"] != MANIFEST_SCHEMA:
        raise AlignmentError("manifest schema is invalid")
    manifest_hash = _self_hash(payload, "manifest_sha256")
    pre_ingest, privacy, confirmation = _load_target_evidence(
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
    )
    pre_ingest_sha = payload["pre_ingest_sha256"]
    if pre_ingest_sha != pre_ingest.pre_ingest_sha256:
        raise AlignmentError("manifest pre-ingest hash mismatch")
    confirmation_hash, privacy_context = (
        payload["component_cohort_sha256"],
        payload["privacy_context_sha256"],
    )
    privacy_transform, owner_attestation, split_binding = (
        payload["privacy_transform_sha256"],
        payload["owner_attestation_sha256"],
        payload["split_binding_sha256"],
    )
    if not all(
        _is_sha(value)
        for value in (
            pre_ingest_sha,
            confirmation_hash,
            privacy_context,
            privacy_transform,
            owner_attestation,
            split_binding,
        )
    ):
        raise AlignmentError("manifest evidence hashes are invalid")
    if (
        confirmation_hash != confirmation.component_cohort_sha256
        or privacy_context != privacy.privacy_context_sha256
        or privacy_transform != privacy.privacy_transform_sha256
        or owner_attestation != privacy.owner_attestation_sha256
    ):
        raise AlignmentError("manifest owner/privacy evidence binding mismatch")
    if not isinstance(payload["sessions"], list) or not isinstance(payload["shards"], list):
        raise AlignmentError("manifest sessions/shards must be lists")
    records: list[SessionRecord] = []
    declared_splits: dict[str, str] = {}
    for value in payload["sessions"]:
        row = _object(
            value,
            {
                "session_hash",
                "component_hash",
                "parent_hash",
                "near_duplicate_hashes",
                "split",
                "privacy_review_sha256",
            },
            "session",
        )
        session, component, parent, duplicates, split, review = (
            row[key]
            for key in (
                "session_hash",
                "component_hash",
                "parent_hash",
                "near_duplicate_hashes",
                "split",
                "privacy_review_sha256",
            )
        )
        if (
            not _is_sha(session)
            or not _is_sha(component)
            or not _is_sha(review)
            or (parent is not None and not _is_sha(parent))
            or not isinstance(duplicates, list)
            or any(not _is_sha(item) for item in duplicates)
            or not isinstance(split, str)
            or split not in SPLITS
        ):
            raise AlignmentError("invalid manifest session row")
        if pre_ingest.component_of.get(session) != component:
            raise AlignmentError("manifest session/component differs from pre-ingest evidence")
        if (
            privacy.components.get(session) != component
            or privacy.privacy_reviews.get(session) != review
        ):
            raise AlignmentError("manifest session differs from reviewed privacy context")
        records.append(
            SessionRecord(session, component, parent, tuple(cast(list[str], duplicates)))
        )
        declared_splits[session] = split
    if len(declared_splits) != len(records):
        raise AlignmentError("manifest sessions must be unique")
    if set(declared_splits) != set(privacy.privacy_reviews):
        raise AlignmentError("manifest sessions must exactly match reviewed privacy context")
    if {record.component_id for record in records} != confirmation.component_hashes:
        raise AlignmentError("manifest components must exactly match component cohort")
    expected_component_splits = _all_clean_components_lexicographic_ceil15pct_v1(
        tuple(pre_ingest.component_of.values())
    )
    expected_session_splits = {
        session: expected_component_splits[component]
        for session, component in sorted(pre_ingest.component_of.items())
    }
    if declared_splits != expected_session_splits:
        raise AlignmentError("manifest split allocation does not match all-clean lexicographic rule")
    if split_binding_hash(declared_splits) != split_binding:
        raise AlignmentError("split binding mismatch")
    components = _linked_groups(records)
    if len(components) < 12:
        raise AlignmentError("manifest requires at least 12 connected components")
    component_of = {record.session_id: record.component_id for record in records}
    component_splits: list[str] = []
    for component_members in components:
        if len({component_of[item] for item in component_members}) != 1:
            raise AlignmentError("linked sessions must share component hash")
        splits = {declared_splits[item] for item in component_members}
        if len(splits) != 1:
            raise AlignmentError("linked sessions cross manifest splits")
        component_splits.append(next(iter(splits)))
    counts = {name: component_splits.count(name) for name in SPLITS}
    if counts["train"] < 8 or counts["dev"] < 2 or counts["test"] < 2:
        raise AlignmentError("manifest components do not satisfy 8/2/2")
    reviews = {
        record.session_id: cast(
            str, cast(dict[str, object], payload["sessions"][index])["privacy_review_sha256"]
        )
        for index, record in enumerate(records)
    }
    supplied = {path.name: path for path in shard_paths}
    if len(supplied) != len(shard_paths):
        raise AlignmentError("shard paths must be unique")
    if snapshot_dir is not None and (
        snapshot_dir.is_symlink()
        or not snapshot_dir.is_dir()
        or any(snapshot_dir.iterdir())
    ):
        raise AlignmentError("training snapshot directory must be a new real directory")
    seen_sessions: set[str] = set()
    expected_names: set[str] = set()
    ordered_names: list[str] = []
    shard_sha256: dict[Path, str] = {}
    shard_bindings: dict[Path, ManifestShardBinding] = {}
    split_row_counts = {split: 0 for split in SPLITS}
    active_session: dict[str, str | None] = {split: None for split in SPLITS}
    closed_sessions: dict[str, set[str]] = {split: set() for split in SPLITS}
    last_timestamp: dict[tuple[str, str], int] = {}
    for value in payload["shards"]:
        row = _object(
            value, {"path", "sha256", "row_count", "session_hashes", "split", "source"}, "shard"
        )
        name, digest, row_count, sessions, split, source = (
            row["path"],
            row["sha256"],
            row["row_count"],
            row["session_hashes"],
            row["split"],
            row["source"],
        )
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or Path(name).suffix != ".npz"
            or name in expected_names
            or not isinstance(digest, str)
            or not _valid_sha(digest)
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
            or not isinstance(sessions, list)
            or not sessions
            or len(set(cast(list[object], sessions))) != len(sessions)
            or any(not isinstance(item, str) or item not in declared_splits for item in sessions)
            or split not in SPLITS
            or source != "target"
        ):
            raise AlignmentError("invalid manifest shard row")
        expected_names.add(name)
        ordered_names.append(name)
        path = supplied.get(name)
        if path is None:
            raise AlignmentError("manifest shard path/hash mismatch")
        actual_sessions = set(cast(list[str], sessions))
        binding = ManifestShardBinding(
            row_count,
            frozenset(actual_sessions),
            split,
            source,
        )
        if not verify_shards:
            try:
                regular = stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
            except OSError as exc:
                raise AlignmentError("manifest shard path/hash mismatch") from exc
            if not regular or any(declared_splits[item] != split for item in actual_sessions):
                raise AlignmentError("manifest shard row/session/split mismatch")
        else:
            shard_bytes = _read_regular(path, ".npz")
            if _sha(shard_bytes) != digest:
                raise AlignmentError("manifest shard path/hash mismatch")
            shard = _load_npz_bytes(shard_bytes, verify_rows=verify_rows)
            _verify_manifest_shard(
                shard,
                binding,
                declared_splits,
                privacy_transform,
                owner_attestation,
                reviews,
                active_session,
                closed_sessions,
                last_timestamp,
            )
        seen_sessions.update(binding.session_hashes)
        shard_sha256[path] = digest
        shard_bindings[path] = binding
        split_row_counts[split] += row_count
    if set(supplied) != expected_names or seen_sessions != set(declared_splits):
        raise AlignmentError("manifest paths and sessions must exactly cover shards")
    return V5Manifest(
        manifest_path,
        manifest_hash,
        pre_ingest_sha,
        confirmation.component_cohort_sha256,
        privacy.privacy_context_sha256,
        split_binding,
        privacy_transform,
        owner_attestation,
        declared_splits,
        tuple(supplied[name] for name in ordered_names),
        shard_sha256,
        split_row_counts,
        shard_bindings,
        reviews,
        snapshot_dir,
    )


def iter_v5_shards(
    manifest: V5Manifest,
    split: str | None = None,
    *,
    verify_rows: bool = True,
    verify_hash: bool = True,
    prefetch_shards: int = 1,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield one manifest-bound NPZ shard at a time; never retain the dataset."""
    if (split is not None and split not in SPLITS) or not 1 <= prefetch_shards <= 8:
        raise AlignmentError("invalid target split/prefetch setting")
    active_session: dict[str, str | None] = {name: None for name in SPLITS}
    closed_sessions: dict[str, set[str]] = {name: set() for name in SPLITS}
    last_timestamp: dict[tuple[str, str], int] = {}
    paths = []
    for path in manifest.shard_paths:
        binding = manifest.shard_bindings.get(path)
        if split is not None and binding is not None and binding.split != split:
            continue
        paths.append(path)

    def load(path: Path) -> tuple[Path, bytes, dict[str, np.ndarray], bool]:
        bound_path = manifest.snapshot_paths.get(path, path)
        data = _read_regular(bound_path, ".npz")
        must_bind = manifest.snapshot_dir is not None and bound_path == path
        if must_bind and _sha(data) != manifest.shard_sha256[path]:
            raise AlignmentError("target shard changed after manifest verification")
        return path, data, _load_npz_bytes(data, verify_rows=verify_rows or must_bind), must_bind

    def records() -> Iterator[tuple[Path, bytes, dict[str, np.ndarray], bool]]:
        if prefetch_shards == 1:
            yield from map(load, paths)
            return
        with ThreadPoolExecutor(max_workers=prefetch_shards) as executor:
            yield from executor.map(load, paths)

    for path, data, shard, must_bind in records():
        binding = manifest.shard_bindings.get(path)
        if (verify_hash or must_bind) and _sha(data) != manifest.shard_sha256[path]:
            raise AlignmentError("target shard changed after manifest verification")
        if must_bind:
            if binding is None:
                raise AlignmentError("training snapshot requires manifest shard binding")
            _verify_manifest_shard(
                shard,
                binding,
                manifest.session_splits,
                manifest.privacy_transform_sha256,
                manifest.owner_attestation_sha256,
                manifest.privacy_reviews,
                active_session,
                closed_sessions,
                last_timestamp,
            )
            snapshot_path = cast(Path, manifest.snapshot_dir) / path.name
            _write_exclusive(snapshot_path, data, sync=False)
            os.chmod(snapshot_path, stat.S_IRUSR)
            manifest.snapshot_paths[path] = snapshot_path
        if split is None or set(map(str, shard["split"])) == {split}:
            yield shard


def build_training_config(
    *,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    mean_teacher_epochs: int = 20,
) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": "hok-agent-v5-training-config-v1",
        "architecture": "torchvision-resnet18-simsiam-c6",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "mean_teacher_epochs": mean_teacher_epochs,
        "ema_decay": 0.999,
        "source_target_ratio": "50/50",
        "trainable": "stem+layer1+layer2+projector+predictor",
        "frozen": "fc+layer3+layer4+all_batchnorm",
        "audit_input": False,
    }
    config["config_sha256"] = _sha(_json(config).encode())
    return config


def load_v5_training_config(path: Path) -> dict[str, object]:
    fields = set(build_training_config())
    config = _object(_load_json(path), fields, "training config")
    if (
        config["schema_version"] != "hok-agent-v5-training-config-v1"
        or config["architecture"] != "torchvision-resnet18-simsiam-c6"
        or config["optimizer"] != "AdamW"
        or config["ema_decay"] != 0.999
        or config["source_target_ratio"] != "50/50"
        or config["audit_input"] is not False
    ):
        raise AlignmentError("training config identity is invalid")
    for key in ("batch_size", "epochs", "mean_teacher_epochs"):
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], int)
            or cast(int, config[key]) < 1
        ):
            raise AlignmentError("training counts must be positive integers")
    _number(config["learning_rate"], low=0.0)
    _number(config["weight_decay"], low=0.0)
    _self_hash(config, "config_sha256")
    return config


def _probabilities(values: tuple[float, ...]) -> tuple[int, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all() or (array < 0).any():
        raise AlignmentError("pseudo probabilities must be finite non-negative C=6 vectors")
    if not math.isclose(float(array.sum()), 1.0, abs_tol=1e-6):
        raise AlignmentError("pseudo probabilities must already be normalized")
    order = np.argsort(-array)
    return int(order[0]), float(array[order[0]]), float(array[order[0]] - array[order[1]])


def filter_pseudo_labels(
    predictions: Sequence[CandidatePrediction],
    *,
    min_confidence: float = 0.995,
    min_margin: float = 0.5,
    ood_limit: float = 0.05,
    min_interval_ms: int = 500,
    max_per_session_class: int = 500,
    max_global_class: int = 2000,
    require_augmentation_views: bool = False,
) -> tuple[tuple[PseudoLabel, ...], PseudoFilterReport]:
    if min_confidence < 0.995 or min_margin < 0.5 or ood_limit > 0.05:
        raise AlignmentError("frozen pseudo thresholds cannot be relaxed")
    if min_interval_ms < 500 or max_per_session_class > 500 or max_global_class > 2000:
        raise AlignmentError("frozen pseudo spacing/caps cannot be relaxed")
    grouped: dict[tuple[str, int], list[CandidatePrediction]] = defaultdict(list)
    for row in predictions:
        _probabilities(row.probs)
        grouped[(row.session_id, row.timestamp_ms)].append(row)
    rejected: dict[str, int] = defaultdict(int)
    accepted: list[PseudoLabel] = []
    session_counts: dict[tuple[str, int], int] = defaultdict(int)
    global_counts: dict[int, int] = defaultdict(int)
    last: dict[str, int] = defaultdict(lambda: -(10**18))
    for (session, timestamp), rows in sorted(grouped.items()):
        combos = {(row.model_id, row.view_id, row.augmentation_id): row for row in rows}
        augmentations = AUGMENTATION_KEYS if require_augmentation_views else ("identity",)
        required = {
            (model, view, augmentation)
            for model in ("source", "student")
            for view in VIEW_KEYS
            for augmentation in augmentations
        }
        reason = ""
        if set(combos) != required or len(rows) != len(required):
            reason = "exact_views"
        elif any(row.cut for row in rows):
            reason = "cut"
        elif any(
            row.ood_score > ood_limit or not row.black_control_ok or not row.constant_control_ok
            for row in rows
        ):
            reason = "ood_control"
        stats = [_probabilities(row.probs) for row in rows]
        classes = {value[0] for value in stats}
        if not reason and len(classes) != 1:
            reason = "agreement"
        if not reason and any(
            value[1] < min_confidence or value[2] < min_margin for value in stats
        ):
            reason = "confidence"
        class_id = stats[0][0]
        if not reason and timestamp - last[session] < min_interval_ms:
            reason = "interval"
        if not reason and session_counts[(session, class_id)] >= max_per_session_class:
            reason = "session_class_cap"
        if not reason and global_counts[class_id] >= max_global_class:
            reason = "global_class_cap"
        if reason:
            rejected[reason] += 1
            continue
        evidence = _sha(
            _json(
                [asdict(row) for row in sorted(rows, key=lambda x: (x.model_id, x.view_id))]
            ).encode()
        )
        accepted.append(
            PseudoLabel(
                session,
                timestamp,
                class_id,
                min(x[1] for x in stats),
                min(x[2] for x in stats),
                evidence,
            )
        )
        last[session] = timestamp
        session_counts[(session, class_id)] += 1
        global_counts[class_id] += 1
    report = PseudoFilterReport(len(grouped), len(accepted), dict(rejected), len(accepted) >= 200)
    return tuple(accepted), report


class ResNet18SimSiam(nn.Module):
    def __init__(self, source_state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        source = resnet18(weights=None, num_classes=6)
        normalized = {
            (key.removeprefix("network.") if key.startswith("network.") else key): value
            for key, value in source_state.items()
        }
        try:
            source.load_state_dict(normalized, strict=True)
        except RuntimeError as exc:
            raise AlignmentError("source checkpoint is not an exact ResNet18 C=6 state") from exc
        self.encoder = source
        self.classifier = copy.deepcopy(source.fc)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.predictor = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 128)
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        for name, parameter in self.encoder.named_parameters():
            if name.startswith(("conv1", "layer1", "layer2")) and ".bn" not in name:
                parameter.requires_grad = True
        for module in (self.projector, self.predictor):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.freeze_batch_norm()

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = cast(torch.Tensor, self.encoder(frames))
        projection = self.projector(features)
        return self.classifier(features), projection, self.predictor(projection)

    def freeze_batch_norm(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False


def _validate_source(data: SourceDataset) -> None:
    if type(data) is not SourceDataset or data.frames.dtype != np.uint8 or data.frames.ndim != 4:
        raise AlignmentError("source must be a typed uint8 NHWC dataset")
    if data.labels.dtype != np.int64 or data.labels.shape != (len(data.frames),):
        raise AlignmentError("source labels must be int64 and row-aligned")
    if not len(data.frames) or (data.labels < 0).any() or (data.labels >= 6).any():
        raise AlignmentError("source labels must use exact C=6")
    if data.split is not None and (
        data.split.dtype != np.dtype("<U10")
        or data.split.shape != (len(data.frames),)
        or set(map(str, data.split)) != {"train", "validation"}
    ):
        raise AlignmentError("source split must be exact train/validation rows")


def _source_train_validation(data: SourceDataset) -> tuple[np.ndarray, np.ndarray]:
    if data.split is None:
        return data.frames, data.frames
    train = data.frames[data.split == "train"]
    validation = data.frames[data.split == "validation"]
    if not len(train) or not len(validation):
        raise AlignmentError("source train/validation split is empty")
    return train, validation


def _tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(frames).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    )


def _runtime_prefetch_shards() -> int:
    try:
        value = int(os.environ.get("HOK_V5_PREFETCH_SHARDS", "1"))
    except ValueError as exc:
        raise AlignmentError("HOK_V5_PREFETCH_SHARDS must be an integer") from exc
    if not 1 <= value <= 8:
        raise AlignmentError("HOK_V5_PREFETCH_SHARDS must be in 1..8")
    return value


def _target_batches(
    manifest: V5Manifest,
    split: str,
    batch_size: int,
    seed: int,
    *,
    bound_at_start: bool = False,
    prefetch_shards: int = 1,
) -> Iterator[np.ndarray]:
    """Deterministically shuffle one bounded target batch at a time."""
    if batch_size < 1:
        raise AlignmentError("target batch size must be positive")
    random = np.random.default_rng(seed)
    pending: list[np.ndarray] = []
    for shard in iter_v5_shards(
        manifest,
        split,
        verify_rows=not bound_at_start,
        verify_hash=not bound_at_start,
        prefetch_shards=prefetch_shards,
    ):
        indices = random.permutation(len(shard["frames"]))
        for index in indices:
            pending.append(shard["frames"][int(index)])
            if len(pending) == batch_size:
                yield np.stack(pending)
                pending.clear()


def _cycled_target_batches(
    manifest: V5Manifest,
    split: str,
    batch_size: int,
    seed: int,
    *,
    bound_at_start: bool = False,
    prefetch_shards: int = 1,
) -> Iterator[np.ndarray]:
    cycle = 0
    while True:
        yielded = False
        for batch in _target_batches(
            manifest,
            split,
            batch_size,
            seed + cycle,
            bound_at_start=bound_at_start,
            prefetch_shards=prefetch_shards,
        ):
            yielded = True
            yield batch
        if not yielded:
            raise AlignmentError("target split cannot produce a complete batch")
        cycle += 1


def _augment(batch: torch.Tensor) -> torch.Tensor:
    return torch.clamp(batch + torch.randn_like(batch) * 0.02, 0.0, 1.0)


def _siam_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -F.cosine_similarity(prediction, target.detach(), dim=1).mean()


def _collapse_metrics(model: ResNet18SimSiam, frames: torch.Tensor) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        _, embeddings, _ = model(frames)
        values = embeddings.cpu().numpy().astype(np.float64)
        centered = values - values.mean(0, keepdims=True)
        covariance = centered.T @ centered / max(1, len(values) - 1)
        spectrum = np.linalg.eigvalsh(covariance)[::-1].clip(0)
        weights = spectrum / max(float(spectrum.sum()), 1e-12)
        effective_rank = float(np.exp(-(weights * np.log(weights + 1e-12)).sum()))
        controls = torch.stack((torch.zeros_like(frames[0]), torch.full_like(frames[0], 0.5)))
        _, control_embeddings, _ = model(controls)
    return {
        "embedding_variance": float(values.var(axis=0).mean()),
        "effective_rank": effective_rank,
        "covariance_spectrum": spectrum.tolist(),
        "black_constant_distance": float(
            torch.norm(control_embeddings[0] - control_embeddings[1]).item()
        ),
        "collapse_status": COLLAPSE_BLOCK,
    }


def _source_network(state: Mapping[str, torch.Tensor], device: torch.device) -> nn.Module:
    network = resnet18(weights=None, num_classes=6).to(device)
    try:
        network.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise AlignmentError("source checkpoint is not an exact ResNet18 C=6 state") from exc
    return cast(nn.Module, network.eval())


def _penultimate(model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    network: Any = model.encoder if isinstance(model, ResNet18SimSiam) else model
    with torch.no_grad():
        value = network.conv1(frames)
        value = network.bn1(value)
        value = network.relu(value)
        value = network.maxpool(value)
        value = network.layer1(value)
        value = network.layer2(value)
        value = network.layer3(value)
        value = network.layer4(value)
        return torch.flatten(network.avgpool(value), 1)


def _embedding_summary(model: nn.Module, frames: torch.Tensor) -> dict[str, float]:
    values = _penultimate(model.eval(), frames).detach().cpu().numpy().astype(np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise AlignmentError("source validation embeddings are not computable")
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.linalg.eigvalsh(centered.T @ centered / (len(values) - 1)).clip(0)
    total = float(spectrum.sum())
    weights = spectrum / total if total > 0 else spectrum
    rank = float(np.exp(-(weights * np.log(weights + 1e-12)).sum())) if total > 0 else 0.0
    controls = torch.stack((torch.zeros_like(frames[0]), torch.full_like(frames[0], 0.5)))
    control_features = _penultimate(model, controls)
    summary = {
        "embedding_variance": float(values.var(axis=0).mean()),
        "effective_rank": rank,
        "top_eigen_share": float(spectrum[-1] / total) if total > 0 else 0.0,
        "black_constant_distance": float(
            torch.norm(control_features[0] - control_features[1]).item()
        ),
    }
    if (
        not all(
            math.isfinite(summary[key]) and summary[key] > 0
            for key in ("embedding_variance", "effective_rank", "black_constant_distance")
        )
        or not 0 < summary["top_eigen_share"] <= 1
    ):
        raise AlignmentError("source validation baseline is not computable")
    return summary


def _collapse_report(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, object]:
    keys = ("embedding_variance", "effective_rank", "top_eigen_share", "black_constant_distance")
    if (
        set(before) != set(keys)
        or set(after) != set(keys)
        or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in (*before.values(), *after.values())
        )
    ):
        raise AlignmentError("collapse summaries are not computable")
    ratios = {
        "variance_ratio": float(after["embedding_variance"] / before["embedding_variance"]),
        "effective_rank_ratio": float(after["effective_rank"] / before["effective_rank"]),
        "top_eigen_share_increase": float(after["top_eigen_share"] - before["top_eigen_share"]),
        "black_constant_distance_ratio": float(
            after["black_constant_distance"] / before["black_constant_distance"]
        ),
    }
    passed = (
        ratios["variance_ratio"] >= COLLAPSE_THRESHOLDS["variance_ratio_min"]
        and ratios["effective_rank_ratio"] >= COLLAPSE_THRESHOLDS["effective_rank_ratio_min"]
        and ratios["top_eigen_share_increase"]
        <= COLLAPSE_THRESHOLDS["top_eigen_share_increase_max"]
        and ratios["black_constant_distance_ratio"]
        >= COLLAPSE_THRESHOLDS["black_constant_distance_ratio_min"]
    )
    return {
        "schema_version": "hok-agent-v5-collapse-v1",
        "before": dict(before),
        "after": dict(after),
        "ratios": ratios,
        "thresholds": COLLAPSE_THRESHOLDS,
        "collapse_status": "PASS" if passed else "BLOCKED",
    }


def action_schema_hash() -> str:
    return _sha(_json(ACTION_TYPES).encode())


def causal_source_teacher_hash() -> str:
    functions = (strip_to_public_observation, _action_name, CausalSourceTeacher.select)
    return _sha(
        ("\n".join(inspect.getsource(value) for value in functions) + _json(ACTION_TYPES)).encode()
    )


def source_producer_hash() -> str:
    functions = (
        _source_rows,
        _fit_source_seed,
        produce_v5_source,
        _source_replay,
        _source_metadata_fields,
        _source_network,
        _penultimate,
        _embedding_summary,
        _collapse_report,
    )
    return _sha(
        (
            _json([SOURCE_EPISODES, SOURCE_SEEDS, SOURCE_TRAINING, COLLAPSE_THRESHOLDS])
            + "\n".join(inspect.getsource(value) for value in functions)
        ).encode()
    )


def _source_rows(episodes: int) -> list[dict[str, object]]:
    if episodes < 2:
        raise AlignmentError("source producer needs at least two full episodes")
    rows: list[dict[str, object]] = []
    for seed in range(episodes):
        split = "validation" if seed % 5 == 0 else "train"
        for side in ("blue", "red"):
            arena = PixelArena()
            arena.reset(seed)
            other: Side = "red" if side == "blue" else "blue"
            random_policy = RandomPolicy(seed, other)
            teacher = CausalSourceTeacher()
            while not arena.state.terminal:
                legal = arena.legal_actions(side)
                public = strip_to_public_observation(arena.observe(side))
                chosen = cast(FactorizedAction, teacher.select(public, legal))
                names = [_action_name(action) for action in legal]
                class_id = ACTION_TYPES.index(_action_name(chosen))
                frame = source_render_128_rgb(public, render_seed=seed)
                frame_sha256 = _sha(frame.tobytes())
                alignment = _sha(_json([frame_sha256, class_id]).encode())
                rows.append(
                    {
                        "arena_seed": seed,
                        "side": side,
                        "tick": int(arena.state.tick),
                        "split": split,
                        "render_seed": seed,
                        "observation": public,
                        "legal_actions": names,
                        "class_id": class_id,
                        "frame": frame,
                        "frame_sha256": frame_sha256,
                        "alignment_hash": alignment,
                    }
                )
                other_action = random_policy.select(other, arena.legal_actions(other))
                if side == "blue":
                    arena.step(chosen, other_action)
                else:
                    arena.step(other_action, chosen)
    by_split = {
        split: {cast(int, row["class_id"]) for row in rows if row["split"] == split}
        for split in ("train", "validation")
    }
    if any(classes != set(range(6)) for classes in by_split.values()):
        raise AlignmentError("source corpus lacks six-class split coverage")
    unique: dict[str, dict[str, object]] = {}
    frame_labels: dict[str, tuple[object, object]] = {}
    for row in rows:
        frame_sha256 = cast(str, row["frame_sha256"])
        prior_frame = frame_labels.setdefault(frame_sha256, (row["class_id"], row["split"]))
        if prior_frame != (row["class_id"], row["split"]):
            raise AlignmentError("source frame has conflicting label or split")
        prior = unique.get(cast(str, row["alignment_hash"]))
        if prior is not None and (
            prior["class_id"] != row["class_id"] or prior["split"] != row["split"]
        ):
            raise AlignmentError("source frame has conflicting label or split")
        unique.setdefault(cast(str, row["alignment_hash"]), row)
    return [unique[key] for key in sorted(unique)]


def _fit_source_seed(
    data: SourceDataset, seed: int, device: str
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if data.split is None:
        raise AlignmentError("source producer dataset requires train/validation split")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    run_device = torch.device(device)
    model = _source_network(resnet18(weights=None, num_classes=6).state_dict(), run_device)
    model.train()
    train_frames, validation_frames = _source_train_validation(data)
    train_labels = data.labels[data.split == "train"]
    validation_labels = data.labels[data.split == "validation"]
    train_x, validation_x = (
        _tensor(train_frames, run_device),
        _tensor(validation_frames, run_device),
    )
    train_y, validation_y = (
        torch.from_numpy(train_labels).to(run_device),
        torch.from_numpy(validation_labels).to(run_device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cast(float, SOURCE_TRAINING["learning_rate"])),
        weight_decay=float(cast(float, SOURCE_TRAINING["weight_decay"])),
    )
    batch = min(int(cast(int, SOURCE_TRAINING["batch_size"])), len(train_x))
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = 0
    for epoch in range(1, int(cast(int, SOURCE_TRAINING["epochs"])) + 1):
        order = torch.randperm(len(train_x), device=run_device)
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            optimizer.zero_grad()
            train_loss = F.cross_entropy(cast(torch.Tensor, model(train_x[index])), train_y[index])
            torch.autograd.backward(train_loss)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                F.cross_entropy(cast(torch.Tensor, model(validation_x)), validation_y)
            )
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        model.train()
    if not best_state:
        raise AlignmentError("source validation CE was not computed")
    return best_state, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_cross_entropy": best_loss,
    }


def _large_output_dir(output_dir: Path) -> Path:
    root_text = os.environ.get("HOK_LARGE_ROOT")
    if not root_text:
        raise AlignmentError("HOK_LARGE_ROOT is required for source artifacts")
    root = Path(root_text).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise AlignmentError("HOK_LARGE_ROOT must be a real directory")
    resolved = output_dir.resolve()
    if resolved == root or root not in resolved.parents:
        raise AlignmentError("source output must be below HOK_LARGE_ROOT")
    return resolved


def _large_output_path(path: Path) -> Path:
    root_text = os.environ.get("HOK_LARGE_ROOT")
    if not root_text:
        raise AlignmentError("HOK_LARGE_ROOT is required for V5 output artifacts")
    root = Path(root_text).resolve(strict=True)
    target = path.resolve()
    if root.is_symlink() or root not in target.parents:
        raise AlignmentError("V5 output artifact must be below HOK_LARGE_ROOT")
    if os.path.lexists(path):
        raise AlignmentError("V5 output artifact already exists")
    return target


def produce_v5_source(*, output_dir: Path, device: str = "cuda") -> dict[str, object]:
    output_dir = _large_output_dir(output_dir)
    if os.path.lexists(output_dir):
        raise AlignmentError("source output directory already exists")
    rows = _source_rows(SOURCE_EPISODES)
    frames = np.stack([cast(np.ndarray, row["frame"]) for row in rows])
    labels = np.asarray([cast(int, row["class_id"]) for row in rows], dtype=np.int64)
    split = np.asarray([cast(str, row["split"]) for row in rows], dtype="<U10")
    alignment = np.asarray([cast(str, row["alignment_hash"]) for row in rows], dtype="<U64")
    data = SourceDataset(frames, labels, split)
    _validate_source(data)
    producer_sha = source_producer_hash()
    split_sha = _sha(
        _json(
            sorted((str(key), str(value)) for key, value in zip(alignment, split, strict=True))
        ).encode()
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        source_path, metadata_path = staging / "source.npz", staging / "source.json"
        buffer = io.BytesIO()
        np.savez(buffer, frames=frames, class_id=labels, alignment_hash=alignment)
        _write_exclusive(source_path, buffer.getvalue())
        source_sha = _sha(source_path.read_bytes())
        training_sha = _sha(_json(SOURCE_TRAINING).encode())
        seed_reports: list[dict[str, object]] = []
        states: dict[int, dict[str, torch.Tensor]] = {}
        for seed in SOURCE_SEEDS:
            state, report = _fit_source_seed(data, seed, device)
            states[seed] = state
            seed_reports.append(report)
        selected = min(
            seed_reports,
            key=lambda row: (
                float(cast(float, row["validation_cross_entropy"])),
                int(cast(int, row["seed"])),
            ),
        )
        models: list[dict[str, object]] = []
        for seed in SOURCE_SEEDS:
            model_path = staging / f"source-seed-{seed}.safetensors"
            model_metadata = {
                "schema_version": MODEL_SCHEMA,
                "role": "v5_causal_source_teacher_source_v1",
                "renderer_sha256": source_renderer_hash(),
                "teacher_sha256": causal_source_teacher_hash(),
                "action_schema_sha256": action_schema_hash(),
                "source_dataset_sha256": source_sha,
                "producer_sha256": producer_sha,
                "source_split_sha256": split_sha,
                "source_training_sha256": training_sha,
                "training_seed": str(seed),
            }
            model_bytes = _model_bytes(states[seed], model_metadata)
            _write_exclusive(model_path, model_bytes)
            report = next(row for row in seed_reports if row["seed"] == seed)
            models.append({**report, "path": model_path.name, "model_sha256": _sha(model_bytes)})
        selected_model = next(row for row in models if row["seed"] == selected["seed"])
        validation_frames = _tensor(data.frames[data.split == "validation"], torch.device("cpu"))
        baseline_network = _source_network(
            states[int(cast(int, selected["seed"]))], torch.device("cpu")
        )
        before = _embedding_summary(baseline_network, validation_frames)
        collapse = _collapse_report(before, before)
        baseline: dict[str, object] = {
            "schema_version": SOURCE_BASELINE_SCHEMA,
            "source_dataset_sha256": source_sha,
            "producer_sha256": producer_sha,
            "source_split_sha256": split_sha,
            "source_training_sha256": training_sha,
            "selection_metric": "source_validation_cross_entropy",
            "models": models,
            "selected_seed": selected["seed"],
            "selected_model_sha256": selected_model["model_sha256"],
            "source_validation_collapse": collapse,
            "collapse_thresholds": COLLAPSE_THRESHOLDS,
        }
        baseline["baseline_sha256"] = _sha(_json(baseline).encode())
        baseline_path = staging / "source-baseline.json"
        _write_exclusive(baseline_path, (_json(baseline) + "\n").encode())
        metadata: dict[str, object] = {
            "schema_version": SOURCE_PRODUCER_SCHEMA,
            "dataset_path": source_path.name,
            "dataset_sha256": source_sha,
            "renderer_id": RENDERER_SPEC["id"],
            "renderer_sha256": source_renderer_hash(),
            "teacher_id": "causal-source-teacher-v1",
            "teacher_sha256": causal_source_teacher_hash(),
            "action_types": list(ACTION_TYPES),
            "action_schema_sha256": action_schema_hash(),
            "producer_sha256": producer_sha,
            "source_split_sha256": split_sha,
            "selected_seed": selected["seed"],
            "selected_model_path": str(selected_model["path"]),
            "selected_model_sha256": selected_model["model_sha256"],
            "source_baseline_path": baseline_path.name,
            "source_baseline_sha256": baseline["baseline_sha256"],
            "rows": [
                {key: value for key, value in row.items() if key not in {"frame", "legal_actions"}}
                for row in rows
            ],
        }
        metadata["source_metadata_sha256"] = _sha(_json(metadata).encode())
        _write_exclusive(metadata_path, (_json(metadata) + "\n").encode())
        staging.rename(output_dir)
        return {
            "status": "PASSED",
            "disposition": "SOURCE_ONLY_NON_PROMOTING",
            "output_dir": str(output_dir),
            "selected_seed": selected["seed"],
            "source_rows": len(rows),
            "source_baseline_sha256": baseline["baseline_sha256"],
            "formal_release_written": False,
        }
    except BaseException:
        for path in staging.iterdir():
            if path.is_file():
                path.unlink()
        staging.rmdir()
        raise


_MODEL_FIELDS = {
    "v5_causal_source_teacher": {
        "schema_version",
        "role",
        "manifest_sha256",
        "split_binding_sha256",
        "config_sha256",
        "renderer_sha256",
        "teacher_sha256",
        "action_schema_sha256",
        "source_dataset_sha256",
        "source_metadata_sha256",
    },
    "v5_causal_source_teacher_source_v1": {
        "schema_version",
        "role",
        "renderer_sha256",
        "teacher_sha256",
        "action_schema_sha256",
        "source_dataset_sha256",
        "producer_sha256",
        "source_split_sha256",
        "source_training_sha256",
        "training_seed",
    },
    "v5_simsiam_adapted": {
        "schema_version",
        "role",
        "manifest_sha256",
        "split_binding_sha256",
        "config_sha256",
        "renderer_sha256",
        "teacher_sha256",
        "action_schema_sha256",
        "source_model_sha256",
        "source_dataset_sha256",
        "source_metadata_sha256",
        "training_seed",
        "collapse_metrics_sha256",
        "collapse_metrics_json",
    },
    "v5_mean_teacher_ema": {
        "schema_version",
        "role",
        "manifest_sha256",
        "split_binding_sha256",
        "config_sha256",
        "renderer_sha256",
        "teacher_sha256",
        "action_schema_sha256",
        "source_model_sha256",
        "source_dataset_sha256",
        "source_metadata_sha256",
        "adapted_model_sha256",
        "pseudo_sha256",
        "predictions_sha256",
        "collapse_metrics_sha256",
        "round",
    },
}


def _load_model(path: Path, role: str) -> tuple[dict[str, torch.Tensor], dict[str, str], str]:
    data = _read_regular(path, ".safetensors")
    if len(data) < 8:
        raise AlignmentError("truncated safetensors artifact")
    length = int.from_bytes(data[:8], "little")
    header = _strict_json_bytes(data[8 : 8 + length])
    if type(header) is not dict:
        raise AlignmentError("invalid safetensors header")
    metadata = cast(dict[str, object], header).get("__metadata__")
    if type(metadata) is not dict or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in cast(dict[object, object], metadata).items()
    ):
        raise AlignmentError("safetensors metadata must be a string map")
    parsed = cast(dict[str, str], metadata)
    _object(parsed, _MODEL_FIELDS.get(role, set()), "model metadata")
    if parsed.get("schema_version") != MODEL_SCHEMA or parsed.get("role") != role:
        raise AlignmentError("model role/schema mismatch")
    for key, value in parsed.items():
        if key.endswith("sha256") and not _valid_sha(value):
            raise AlignmentError("model metadata hashes are invalid")
    try:
        state = load_safetensors(data)
    except Exception as exc:
        raise AlignmentError("invalid safetensors tensor payload") from exc
    if not state:
        raise AlignmentError("model tensor payload cannot be empty")
    return state, parsed, _sha(data)


def _model_bytes(state: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> bytes:
    return save_safetensors(
        {key: value.detach().cpu().contiguous() for key, value in state.items()},
        metadata=dict(metadata),
    )


def _training_paths(output_checkpoint: Path) -> tuple[Path, Path]:
    stem = output_checkpoint.name.removesuffix(".safetensors")
    return (
        output_checkpoint.with_name(f"{stem}.resume.safetensors"),
        output_checkpoint.with_name(f"{stem}.progress.jsonl"),
    )


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise AlignmentError("checkpoint destination cannot be a symlink")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _append_progress(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        data = (_json(payload) + "\n").encode()
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resume_bytes(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    payload: Mapping[str, object],
    *,
    cuda: bool,
) -> bytes:
    tensors = {f"model/{key}": value for key, value in model.state_dict().items()}
    parameters = dict(model.named_parameters())
    optimizer_scalars: dict[str, dict[str, object]] = {}
    for name, parameter in parameters.items():
        values: dict[str, object] = {}
        for key, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor):
                tensors[f"optimizer/{name}/{key}"] = value
            elif type(value) in {bool, int, float}:
                values[key] = value
            else:
                raise AlignmentError("optimizer state is not resumable")
        if values:
            optimizer_scalars[name] = values
    tensors["rng/cpu"] = torch.get_rng_state()
    if cuda:
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"rng/cuda/{index}"] = state
    full_payload = dict(payload)
    full_payload.update(
        {
            "optimizer_scalars": optimizer_scalars,
            "scheduler_state": scheduler.state_dict(),  # type: ignore[no-untyped-call]
            "cuda_rng_count": len(torch.cuda.get_rng_state_all()) if cuda else 0,
        }
    )
    encoded = _json(full_payload)
    return save_safetensors(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        metadata={
            "schema_version": RESUME_SCHEMA,
            "role": "v5_simsiam_resume",
            "payload_json": encoded,
            "payload_sha256": _sha(encoded.encode()),
        },
    )


def _load_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected: Mapping[str, object],
    *,
    cuda: bool,
) -> tuple[int, float]:
    data = _read_regular(path, ".safetensors")
    if len(data) < 9:
        raise AlignmentError("truncated resume checkpoint")
    length = int.from_bytes(data[:8], "little")
    header = _strict_json_bytes(data[8 : 8 + length])
    metadata = cast(dict[str, object], header).get("__metadata__") if type(header) is dict else None
    if type(metadata) is not dict or set(metadata) != {
        "schema_version",
        "role",
        "payload_json",
        "payload_sha256",
    }:
        raise AlignmentError("resume metadata is not exact")
    meta = cast(dict[str, str], metadata)
    if (
        any(not isinstance(value, str) for value in meta.values())
        or meta["schema_version"] != RESUME_SCHEMA
        or meta["role"] != "v5_simsiam_resume"
        or _sha(meta["payload_json"].encode()) != meta["payload_sha256"]
    ):
        raise AlignmentError("resume metadata binding is invalid")
    payload = _strict_json_bytes(meta["payload_json"].encode())
    if type(payload) is not dict or any(payload.get(key) != value for key, value in expected.items()):
        raise AlignmentError("resume checkpoint does not match this training invocation")
    completed = payload.get("completed_epochs")
    elapsed = payload.get("elapsed_seconds")
    if type(completed) is not int or completed < 1:
        raise AlignmentError("resume progress is invalid")
    completed_epochs = completed
    elapsed_seconds = _number(elapsed, low=0.0)
    try:
        tensors = load_safetensors(data)
    except Exception as exc:
        raise AlignmentError("resume tensor payload is invalid") from exc
    model_state = {key.removeprefix("model/"): value for key, value in tensors.items() if key.startswith("model/")}
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise AlignmentError("resume model state is invalid") from exc
    parameters = dict(model.named_parameters())
    scalars = payload.get("optimizer_scalars")
    scheduler_state = payload.get("scheduler_state")
    if type(scalars) is not dict or type(scheduler_state) is not dict:
        raise AlignmentError("resume optimizer/scheduler state is invalid")
    for name, parameter in parameters.items():
        state = optimizer.state[parameter]
        scalar_state = cast(dict[str, object], scalars).get(name, {})
        if type(scalar_state) is not dict:
            raise AlignmentError("resume optimizer scalar state is invalid")
        for key, value in cast(dict[str, object], scalar_state).items():
            state[key] = value
        prefix = f"optimizer/{name}/"
        for key, value in tensors.items():
            if key.startswith(prefix):
                state[key.removeprefix(prefix)] = value.to(next(model.parameters()).device)
    scheduler.load_state_dict(cast(dict[str, Any], scheduler_state))
    optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
    if "rng/cpu" not in tensors:
        raise AlignmentError("resume CPU RNG state is missing")
    torch.set_rng_state(tensors["rng/cpu"])
    cuda_count = payload.get("cuda_rng_count")
    if type(cuda_count) is not int or cuda_count != (len(torch.cuda.get_rng_state_all()) if cuda else 0):
        raise AlignmentError("resume CUDA topology does not match")
    assert type(cuda_count) is int
    if cuda:
        torch.cuda.set_rng_state_all([tensors[f"rng/cuda/{index}"] for index in range(cuda_count)])
    return completed_epochs, elapsed_seconds


def _source_npz(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != {"frames", "class_id", "alignment_hash"}:
                raise AlignmentError("source NPZ fields are not exact")
            frames, labels, alignment = (
                archive[key].copy() for key in ("frames", "class_id", "alignment_hash")
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, AlignmentError):
            raise
        raise AlignmentError("invalid source NPZ") from exc
    if (
        frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[1:] != (128, 128, 3)
        or labels.dtype != np.int64
        or labels.shape != (len(frames),)
        or alignment.dtype != np.dtype("<U64")
        or alignment.shape != (len(frames),)
        or not len(frames)
    ):
        raise AlignmentError("source NPZ dtype/shape is invalid")
    return frames, labels, alignment


def _action_object(name: str) -> dict[str, str]:
    if name == "wait":
        return {"action_type": "wait", "target": "none", "direction": "none"}
    if name in {"forward", "backward"}:
        return {"action_type": "move", "target": "none", "direction": name}
    return {
        "action_type": "attack",
        "target": {
            "attack_hero": "enemy_hero",
            "attack_tower": "enemy_tower",
            "attack_crystal": "enemy_crystal",
        }[name],
        "direction": "none",
    }


def _source_replay(seed: int, side: Side, tick: int) -> tuple[dict[str, object], list[str], int]:
    if isinstance(seed, bool) or not isinstance(seed, int) or tick < 0:
        raise AlignmentError("source replay identity is invalid")
    arena = PixelArena()
    arena.reset(seed)
    other: Side = "red" if side == "blue" else "blue"
    random_policy = RandomPolicy(seed, other)
    teacher = CausalSourceTeacher()
    for current_tick in range(tick + 1):
        legal = arena.legal_actions(side)
        public = strip_to_public_observation(arena.observe(side))
        chosen = cast(FactorizedAction, teacher.select(public, legal))
        names = [_action_name(action) for action in legal]
        class_id = ACTION_TYPES.index(_action_name(chosen))
        if current_tick == tick:
            return public, names, class_id
        other_legal = arena.legal_actions(other)
        other_action = random_policy.select(other, other_legal)
        if side == "blue":
            arena.step(chosen, other_action)
        else:
            arena.step(other_action, chosen)
        if arena.state.terminal:
            break
    raise AlignmentError("source replay tick is not reachable")


def _source_metadata_fields(schema: object) -> set[str]:
    base = {
        "schema_version",
        "manifest_sha256",
        "config_sha256",
        "dataset_path",
        "dataset_sha256",
        "renderer_id",
        "renderer_sha256",
        "teacher_id",
        "teacher_sha256",
        "action_types",
        "action_schema_sha256",
        "rows",
        "source_metadata_sha256",
    }
    if schema == SOURCE_PRODUCER_SCHEMA:
        return {
            "schema_version",
            "dataset_path",
            "dataset_sha256",
            "renderer_id",
            "renderer_sha256",
            "teacher_id",
            "teacher_sha256",
            "action_types",
            "action_schema_sha256",
            "producer_sha256",
            "source_split_sha256",
            "selected_seed",
            "selected_model_path",
            "selected_model_sha256",
            "source_baseline_path",
            "source_baseline_sha256",
            "rows",
            "source_metadata_sha256",
        }
    if schema == SOURCE_SCHEMA:
        return base
    raise AlignmentError("source metadata schema is invalid")


def _source_baseline(
    path: Path,
    *,
    data: SourceDataset,
    source_bytes_sha256: str,
    source_model_path: Path,
    source_model_sha256: str,
    source_state: Mapping[str, torch.Tensor],
    producer_sha256: str,
    source_split_sha256: str,
) -> str:
    payload = _object(
        _load_json(path),
        {
            "schema_version",
            "source_dataset_sha256",
            "producer_sha256",
            "source_split_sha256",
            "source_training_sha256",
            "selection_metric",
            "models",
            "selected_seed",
            "selected_model_sha256",
            "source_validation_collapse",
            "collapse_thresholds",
            "baseline_sha256",
        },
        "source baseline",
    )
    baseline_sha = _self_hash(payload, "baseline_sha256")
    if (
        payload["schema_version"] != SOURCE_BASELINE_SCHEMA
        or payload["source_dataset_sha256"] != source_bytes_sha256
        or payload["producer_sha256"] != producer_sha256
        or payload["source_split_sha256"] != source_split_sha256
        or payload["source_training_sha256"] != _sha(_json(SOURCE_TRAINING).encode())
        or payload["selection_metric"] != "source_validation_cross_entropy"
        or payload["selected_model_sha256"] != source_model_sha256
        or not isinstance(payload["selected_seed"], int)
        or isinstance(payload["selected_seed"], bool)
        or not isinstance(payload["models"], list)
        or payload["collapse_thresholds"] != COLLAPSE_THRESHOLDS
    ):
        raise AlignmentError("source baseline binding is invalid")
    models = payload["models"]
    if len(models) != len(SOURCE_SEEDS) or {
        row.get("seed") for row in models if isinstance(row, dict)
    } != set(SOURCE_SEEDS):
        raise AlignmentError("source baseline seed reports are invalid")
    selected = next(
        (
            row
            for row in models
            if isinstance(row, dict) and row.get("seed") == payload["selected_seed"]
        ),
        None,
    )
    if (
        selected is None
        or selected.get("model_sha256") != source_model_sha256
        or selected.get("path") != source_model_path.name
    ):
        raise AlignmentError("source baseline selected model mismatch")
    if (
        selected["seed"]
        != min(
            models,
            key=lambda row: (
                float(cast(int | float, cast(dict[str, object], row)["validation_cross_entropy"])),
                int(cast(int, cast(dict[str, object], row)["seed"])),
            ),
        )["seed"]
    ):
        raise AlignmentError("source baseline selected seed is not validation-CE optimal")
    if data.split is None:
        raise AlignmentError("source baseline requires a source validation split")
    validation = _tensor(data.frames[data.split == "validation"], torch.device("cpu"))
    actual = _collapse_report(
        _embedding_summary(_source_network(source_state, torch.device("cpu")), validation),
        _embedding_summary(_source_network(source_state, torch.device("cpu")), validation),
    )
    if payload["source_validation_collapse"] != actual:
        raise AlignmentError("source baseline collapse evidence mismatch")
    return baseline_sha


def _source_bundle(
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    shard_paths: Sequence[Path],
    config_path: Path,
    source_model_path: Path,
) -> tuple[SourceDataset, dict[str, torch.Tensor], dict[str, str], str]:
    raw_payload = _load_json(source_metadata_path)
    payload = _object(
        raw_payload, _source_metadata_fields(raw_payload.get("schema_version")), "source metadata"
    )
    metadata_hash = _self_hash(payload, "source_metadata_sha256")
    source_bytes = _read_regular(source_dataset_path, ".npz")
    producer = payload["schema_version"] == SOURCE_PRODUCER_SCHEMA
    manifest: V5Manifest | None = None
    config: dict[str, object] | None = None
    if not producer:
        manifest = load_v5_manifest(
            manifest_path,
            pre_ingest_path,
            privacy_context_path,
            owner_attestation_path,
            owner_component_confirmation_path,
            shard_paths,
        )
        config = load_v5_training_config(config_path)
    common = (
        payload["dataset_path"] == source_dataset_path.name
        and payload["dataset_sha256"] == _sha(source_bytes)
        and payload["renderer_id"] == RENDERER_SPEC["id"]
        and payload["renderer_sha256"] == source_renderer_hash()
        and payload["teacher_id"] == "causal-source-teacher-v1"
        and payload["teacher_sha256"] == causal_source_teacher_hash()
        and payload["action_types"] == list(ACTION_TYPES)
        and payload["action_schema_sha256"] == action_schema_hash()
    )
    fixed = common and (
        (
            producer
            and payload["producer_sha256"] == source_producer_hash()
            and _is_sha(payload["source_split_sha256"])
            and type(payload["selected_seed"]) is int
            and _is_sha(payload["selected_model_sha256"])
            and _is_sha(payload["source_baseline_sha256"])
            and payload["selected_model_path"] == source_model_path.name
            and payload["source_baseline_path"] == "source-baseline.json"
        )
        or (
            not producer
            and manifest is not None
            and config is not None
            and payload["manifest_sha256"] == manifest.manifest_sha256
            and payload["config_sha256"] == config["config_sha256"]
        )
    )
    if not fixed or not isinstance(payload["rows"], list):
        raise AlignmentError("source provenance binding is invalid")
    frames, labels, alignment = _source_npz(source_bytes)
    if len(payload["rows"]) != len(frames) or len(set(map(str, alignment))) != len(alignment):
        raise AlignmentError("source rows/alignment are not exact and unique")
    actual = {str(value): index for index, value in enumerate(alignment)}
    seen: set[str] = set()
    splits = np.full(len(frames), "train", dtype="<U10")
    for value in payload["rows"]:
        fields = {"alignment_hash", "observation", "class_id"}
        if producer:
            fields |= {"arena_seed", "side", "tick", "split", "render_seed", "frame_sha256"}
        else:
            fields.add("legal_actions")
        row = _object(value, fields, "source row")
        alignment_hash, observation, class_id = (
            row["alignment_hash"],
            row["observation"],
            row["class_id"],
        )
        if (
            not isinstance(alignment_hash, str)
            or alignment_hash not in actual
            or alignment_hash in seen
            or type(observation) is not dict
            or set(cast(dict[object, object], observation)) != set(_PUBLIC_KEYS)
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < 6
        ):
            raise AlignmentError("invalid causal source row")
        public = cast(dict[str, object], observation)
        if public["side"] not in {"blue", "red"} or any(
            isinstance(public[key], bool) or not isinstance(public[key], int)
            for key in _PUBLIC_KEYS
            if key != "side"
        ):
            raise AlignmentError("source observation is not exact public state")
        index = actual[alignment_hash]
        legal = row.get("legal_actions")
        if producer:
            seed, side, tick, split, render_seed, frame_sha256 = (
                row["arena_seed"],
                row["side"],
                row["tick"],
                row["split"],
                row["render_seed"],
                row["frame_sha256"],
            )
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not isinstance(side, str)
                or side not in {"blue", "red"}
                or isinstance(tick, bool)
                or not isinstance(tick, int)
                or not isinstance(split, str)
                or split not in {"train", "validation"}
                or isinstance(render_seed, bool)
                or not isinstance(render_seed, int)
                or render_seed != seed
                or not _is_sha(frame_sha256)
                or frame_sha256 != _sha(frames[index].tobytes())
            ):
                raise AlignmentError("source producer row identity is invalid")
            replay_public, legal, replay_class = _source_replay(seed, cast(Side, side), tick)
        if (
            not isinstance(legal, list)
            or not legal
            or len(set(cast(list[object], legal))) != len(legal)
            or any(name not in ACTION_TYPES for name in legal)
        ):
            raise AlignmentError("source legal replay is invalid")
        selected = CausalSourceTeacher().select(
            public, tuple(_action_object(cast(str, name)) for name in legal)
        )
        if (
            ACTION_TYPES[class_id] != _action_name(selected)
            or labels[index] != class_id
            or (not producer and not np.array_equal(frames[index], source_render_128_rgb(public)))
            or _sha(_json([_sha(frames[index].tobytes()), class_id]).encode()) != alignment_hash
        ):
            raise AlignmentError("source frame/teacher/class alignment mismatch")
        if producer:
            if (
                replay_public != public
                or replay_class != class_id
                or not np.array_equal(
                    frames[index], source_render_128_rgb(public, render_seed=cast(int, render_seed))
                )
            ):
                raise AlignmentError("source producer replay mismatch")
            splits[index] = split
        seen.add(alignment_hash)
    if seen != set(actual):
        raise AlignmentError("source metadata does not cover dataset")
    if producer:
        split_hash = _sha(
            _json(
                sorted((str(key), str(value)) for key, value in zip(alignment, splits, strict=True))
            ).encode()
        )
        if (
            payload["source_split_sha256"] != split_hash
            or payload["producer_sha256"] != source_producer_hash()
        ):
            raise AlignmentError("source producer split/provenance binding is invalid")
    role = "v5_causal_source_teacher_source_v1" if producer else "v5_causal_source_teacher"
    state, model_meta, model_sha = _load_model(source_model_path, role)
    if producer:
        expected = {
            "renderer_sha256": source_renderer_hash(),
            "teacher_sha256": causal_source_teacher_hash(),
            "action_schema_sha256": action_schema_hash(),
            "source_dataset_sha256": cast(str, payload["dataset_sha256"]),
            "producer_sha256": source_producer_hash(),
            "source_split_sha256": cast(str, payload["source_split_sha256"]),
            "source_training_sha256": _sha(_json(SOURCE_TRAINING).encode()),
            "training_seed": str(payload["selected_seed"]),
        }
    else:
        if manifest is None or config is None:
            raise AlignmentError("legacy source requires target evidence")
        expected = {
            "manifest_sha256": manifest.manifest_sha256,
            "split_binding_sha256": manifest.split_binding_sha256,
            "config_sha256": cast(str, config["config_sha256"]),
            "renderer_sha256": source_renderer_hash(),
            "teacher_sha256": causal_source_teacher_hash(),
            "action_schema_sha256": action_schema_hash(),
            "source_dataset_sha256": cast(str, payload["dataset_sha256"]),
            "source_metadata_sha256": metadata_hash,
        }
    if any(model_meta[key] != value for key, value in expected.items()):
        raise AlignmentError("source model metadata binding mismatch")
    data = SourceDataset(frames, labels, splits if producer else None)
    _validate_source(data)
    if (
        producer
        and _source_baseline(
            source_metadata_path.with_name(cast(str, payload["source_baseline_path"])),
            data=data,
            source_bytes_sha256=cast(str, payload["dataset_sha256"]),
            source_model_path=source_model_path,
            source_model_sha256=model_sha,
            source_state=state,
            producer_sha256=cast(str, payload["producer_sha256"]),
            source_split_sha256=cast(str, payload["source_split_sha256"]),
        )
        != payload["source_baseline_sha256"]
    ):
        raise AlignmentError("source baseline hash mismatch")
    return data, state, model_meta, model_sha


def load_v5_source_dataset(
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    shard_paths: Sequence[Path],
    config_path: Path,
    source_model_path: Path,
) -> SourceDataset:
    return _source_bundle(
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        shard_paths,
        config_path,
        source_model_path,
    )[0]


def _adapted_model(state: Mapping[str, torch.Tensor]) -> ResNet18SimSiam:
    source_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.") and key not in {"encoder.fc.weight", "encoder.fc.bias"}
    }
    source_state["fc.weight"], source_state["fc.bias"] = (
        state["classifier.weight"],
        state["classifier.bias"],
    )
    model = ResNet18SimSiam(source_state)
    model.load_state_dict(state, strict=True)
    return model


def train_shallow_simsiam(
    *,
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    target_shards: Sequence[Path],
    config_path: Path,
    source_checkpoint: Path,
    output_checkpoint: Path,
    device: str = "cpu",
    seed: int = 0,
    resume: bool = False,
) -> TrainingResult:
    output_checkpoint = _large_output_path(output_checkpoint)
    resume_checkpoint, progress_log = _training_paths(output_checkpoint)
    if output_checkpoint.exists():
        raise AlignmentError("final adapted checkpoint already exists")
    if resume:
        if not resume_checkpoint.is_file():
            raise AlignmentError("resume requested but no resume checkpoint exists")
    elif resume_checkpoint.exists() or progress_log.exists():
        raise AlignmentError("existing training progress requires explicit --resume")
    source, source_state, source_meta, source_sha = _source_bundle(
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
        config_path,
        source_checkpoint,
    )
    root = Path(os.environ["HOK_LARGE_ROOT"]).resolve(strict=True)
    snapshot_parent = root / "staging"
    snapshot_parent.mkdir(parents=True, exist_ok=True)
    if snapshot_parent.is_symlink() or not snapshot_parent.is_dir():
        raise AlignmentError("V5 training snapshot parent must be a real directory")
    snapshot = tempfile.TemporaryDirectory(prefix=".v5-train-", dir=snapshot_parent)
    manifest, config = (
        load_v5_manifest(
            manifest_path,
            pre_ingest_path,
            privacy_context_path,
            owner_attestation_path,
            owner_component_confirmation_path,
            target_shards,
            verify_rows=False,
            verify_shards=False,
            snapshot_dir=Path(snapshot.name),
        ),
        load_v5_training_config(config_path),
    )
    target_count = manifest.split_row_counts["train"]
    if not target_count:
        raise AlignmentError("target training split is empty")
    batch_size, epochs = cast(int, config["batch_size"]), cast(int, config["epochs"])
    if batch_size < 2 or batch_size % 2:
        raise AlignmentError("50/50 batches require a positive even batch size")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    run_device = torch.device(device)
    model = ResNet18SimSiam(source_state).to(run_device)
    model.train()
    model.freeze_batch_norm()
    source_train, source_validation = _source_train_validation(source)
    source_labels = (
        source.labels if source.split is None else source.labels[source.split == "train"]
    )
    half = batch_size // 2
    if target_count < half:
        raise AlignmentError("target training split is smaller than one half-batch")
    prefetch_shards = _runtime_prefetch_shards()
    steps_per_epoch = max(math.ceil(len(source_train) / half), math.ceil(target_count / half))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cast(float, config["learning_rate"]),
        weight_decay=cast(float, config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, epochs * steps_per_epoch)
    )
    binding = {
        "manifest_sha256": manifest.manifest_sha256,
        "split_binding_sha256": manifest.split_binding_sha256,
        "config_sha256": cast(str, config["config_sha256"]),
        "source_model_sha256": source_sha,
        "training_seed": seed,
        "steps_per_epoch": steps_per_epoch,
        "target_prefetch_shards": prefetch_shards,
    }
    start_epoch, elapsed_prior = (
        _load_resume(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            binding,
            cuda=device == "cuda",
        )
        if resume
        else (0, 0.0)
    )
    if start_epoch >= epochs:
        raise AlignmentError("resume checkpoint has already completed this training config")
    started = time.monotonic() - elapsed_prior
    for epoch in range(start_epoch, epochs):
        source_random = np.random.default_rng(seed + epoch)
        target_batches = _cycled_target_batches(
            manifest,
            "train",
            half,
            seed + epoch,
            bound_at_start=True,
            prefetch_shards=prefetch_shards,
        )
        loss_total = torch.zeros((), device=run_device)
        for _ in range(steps_per_epoch):
            source_index = source_random.integers(len(source_train), size=half)
            source_x = _tensor(source_train[source_index], run_device)
            source_y = torch.from_numpy(source_labels[source_index]).to(run_device)
            target_x = _tensor(next(target_batches), run_device)
            source_logits, _, _ = model(_augment(source_x))
            first, second = _augment(target_x), _augment(target_x)
            _, z1, p1 = model(first)
            _, z2, p2 = model(second)
            loss = F.cross_entropy(source_logits, source_y) + 0.5 * (
                _siam_loss(p1, z2) + _siam_loss(p2, z1)
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            scheduler.step()
            model.freeze_batch_norm()
            loss_total += loss.detach()
        completed_epochs = epoch + 1
        elapsed_seconds = time.monotonic() - started
        _write_atomic(
            resume_checkpoint,
            _resume_bytes(
                model,
                optimizer,
                scheduler,
                {
                    **binding,
                    "completed_epochs": completed_epochs,
                    "elapsed_seconds": elapsed_seconds,
                },
                cuda=device == "cuda",
            ),
        )
        _append_progress(
            progress_log,
            {
                "schema_version": "hok-agent-v5-training-progress-v1",
                "epoch": completed_epochs,
                "epochs": epochs,
                "global_step": completed_epochs * steps_per_epoch,
                "steps_per_epoch": steps_per_epoch,
                "mean_loss": float((loss_total / steps_per_epoch).cpu()),
                "elapsed_seconds": elapsed_seconds,
                **binding,
            },
        )
    validation = _tensor(source_validation, run_device)
    before = _embedding_summary(_source_network(source_state, run_device), validation)
    after = _embedding_summary(model, validation)
    metrics = _collapse_report(before, after)
    collapse_json = _json(metrics)
    collapse_hash = _sha(collapse_json.encode())
    metadata = {
        "schema_version": MODEL_SCHEMA,
        "role": "v5_simsiam_adapted",
        "manifest_sha256": manifest.manifest_sha256,
        "split_binding_sha256": manifest.split_binding_sha256,
        "config_sha256": cast(str, config["config_sha256"]),
        "renderer_sha256": source_renderer_hash(),
        "teacher_sha256": causal_source_teacher_hash(),
        "action_schema_sha256": action_schema_hash(),
        "source_model_sha256": source_sha,
        "source_dataset_sha256": source_meta["source_dataset_sha256"],
        "source_metadata_sha256": source_meta["source_metadata_sha256"],
        "training_seed": str(seed),
        "collapse_metrics_sha256": collapse_hash,
        "collapse_metrics_json": collapse_json,
    }
    output_bytes = _model_bytes(model.state_dict(), metadata)
    _write_exclusive(output_checkpoint, output_bytes)
    metrics.update(
        {
            "cpu_smoke": device == "cpu",
            "training_steps": epochs * steps_per_epoch,
            "progress_log": str(progress_log),
            "resume_checkpoint": str(resume_checkpoint),
            "disposition": "NON_PROMOTING_FRAMEWORK_ONLY",
        }
    )
    result = TrainingResult(
        output_checkpoint,
        _sha(output_bytes),
        cast(str, config["config_sha256"]),
        metrics,
        False,
    )
    snapshot.cleanup()
    return result


_PSEUDO_FIELDS = (
    "frames",
    "class_id",
    "session_hash",
    "timestamp_ms",
    "alignment_hash",
    "evidence_hash",
    "manifest_sha256",
    "predictions_sha256",
    "source_model_sha256",
    "adapted_model_sha256",
    "config_sha256",
)


def _target_index(manifest: V5Manifest) -> dict[tuple[str, int], tuple[np.ndarray, str, str]]:
    result = {}
    for shard in iter_v5_shards(manifest):
        for frame, session, timestamp, alignment, split in zip(
            shard["frames"],
            shard["session_hash"],
            shard["timestamp_ms"],
            shard["alignment_hash"],
            shard["split"],
            strict=True,
        ):
            result[(str(session), int(timestamp))] = (frame, str(alignment), str(split))
    return result


_MODEL_PREDICTION_FIELDS = (
    "session_hash",
    "anchor_timestamp_ms",
    "frame_timestamp_ms",
    "frame_alignment_hash",
    "model_role",
    "view_id",
    "augmentation_id",
    "probs",
    "ood_score",
    "black_control_ok",
    "constant_control_ok",
    "cut",
)
_MODEL_PREDICTION_DTYPES = {
    "session_hash": np.dtype("<U64"),
    "anchor_timestamp_ms": np.dtype("int64"),
    "frame_timestamp_ms": np.dtype("int64"),
    "frame_alignment_hash": np.dtype("<U64"),
    "model_role": np.dtype("<U7"),
    "view_id": np.dtype("<U6"),
    "augmentation_id": np.dtype("<U24"),
    "probs": np.dtype("float32"),
    "ood_score": np.dtype("float32"),
    "black_control_ok": np.dtype("bool"),
    "constant_control_ok": np.dtype("bool"),
    "cut": np.dtype("bool"),
}
_MODEL_PREDICTION_MANIFEST_FIELDS = {
    "schema_version",
    "manifest_sha256",
    "source_model_sha256",
    "adapted_model_sha256",
    "config_sha256",
    "inference_contract_sha256",
    "ood_metric",
    "controls",
    "controls_sha256",
    "shards",
    "predictions_sha256",
}


def prediction_inference_contract() -> dict[str, object]:
    return {
        "schema_version": "hok-agent-v5-inference-contract-v1",
        "model_roles": ["source", "student"],
        "temporal_offsets_ms": [-100, 0, 100],
        "augmentation_views": list(AUGMENTATION_KEYS),
        "normalization": "uint8_to_float32_div255",
        "color_view": "affine_contrast_0.92_channel_offsets_2_-2_1",
        "translation_view": "constant_fill_translate_x2_y-2",
        "scene_cut_mean_absolute_delta_max": 0.35,
        "ood_metric": "source_student_total_variation_v1",
        "control_rule": "black_and_constant_max_softmax_below_0.995",
    }


def prediction_inference_contract_hash() -> str:
    return _sha(_json(prediction_inference_contract()).encode())


def _safe_views(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (128, 128, 3):
        raise AlignmentError("prediction views require uint8 NHWC 128 RGB frames")
    color = np.clip(
        (frames.astype(np.float32) - 127.5) * 0.92 + 127.5 + np.asarray((2, -2, 1)), 0, 255
    ).astype(np.uint8)
    translated = np.zeros_like(frames)
    translated[:, :-2, 2:] = frames[:, 2:, :-2]
    return frames, color, translated


def _prediction_binding(
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    target_shards: Sequence[Path],
    config_path: Path,
    source_model_path: Path,
    adapted_model_path: Path,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, str],
    str,
    V5Manifest,
    dict[str, object],
    dict[str, torch.Tensor],
    str,
]:
    _, source_state, source_meta, source_sha = _source_bundle(
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
        config_path,
        source_model_path,
    )
    manifest = load_v5_manifest(
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
    )
    config = load_v5_training_config(config_path)
    adapted_state, adapted_meta, adapted_sha = _load_model(adapted_model_path, "v5_simsiam_adapted")
    expected = {
        "manifest_sha256": manifest.manifest_sha256,
        "split_binding_sha256": manifest.split_binding_sha256,
        "config_sha256": cast(str, config["config_sha256"]),
        "source_model_sha256": source_sha,
        "source_dataset_sha256": source_meta["source_dataset_sha256"],
        "source_metadata_sha256": source_meta["source_metadata_sha256"],
        "renderer_sha256": source_renderer_hash(),
        "teacher_sha256": causal_source_teacher_hash(),
        "action_schema_sha256": action_schema_hash(),
    }
    if any(adapted_meta[key] != value for key, value in expected.items()):
        raise AlignmentError("adapted model binding mismatch")
    return source_state, source_meta, source_sha, manifest, config, adapted_state, adapted_sha


def _bound_target_shards(manifest: V5Manifest) -> Iterator[tuple[Path, dict[str, np.ndarray]]]:
    for path in manifest.shard_paths:
        data = _read_regular(path, ".npz")
        if _sha(data) != manifest.shard_sha256[path]:
            raise AlignmentError("target shard changed after manifest verification")
        shard = _load_npz_bytes(data)
        if set(map(str, shard["split"])) == {"train"}:
            yield path, shard


def _control_summary(
    source_model: nn.Module, student_model: ResNet18SimSiam, device: torch.device
) -> dict[str, object]:
    black, constant = (
        np.zeros((1, 128, 128, 3), dtype=np.uint8),
        np.full((1, 128, 128, 3), 128, dtype=np.uint8),
    )
    control_views = tuple(
        np.concatenate((_safe_views(black)[index], _safe_views(constant)[index]))
        for index in range(len(AUGMENTATION_KEYS))
    )
    controls = _tensor(np.concatenate(control_views), device)
    with torch.no_grad():
        source = F.softmax(cast(torch.Tensor, source_model(controls)), dim=1)
        student = F.softmax(student_model(controls)[0], dim=1)
    source_values, student_values = source.max(dim=1).values.cpu().numpy(), student.max(dim=1).values.cpu().numpy()
    maxima = {
        "source_black_max_probability": float(source_values[::2].max()),
        "source_constant_max_probability": float(source_values[1::2].max()),
        "student_black_max_probability": float(student_values[::2].max()),
        "student_constant_max_probability": float(student_values[1::2].max()),
    }
    return {
        **maxima,
        "black_control_ok": max(
            maxima["source_black_max_probability"], maxima["student_black_max_probability"]
        )
        < 0.995,
        "constant_control_ok": max(
            maxima["source_constant_max_probability"], maxima["student_constant_max_probability"]
        )
        < 0.995,
    }


def _prediction_windows(shard: Mapping[str, np.ndarray]) -> tuple[tuple[int, int, int], ...]:
    sessions, timestamps = shard["session_hash"], shard["timestamp_ms"]
    return tuple(
        (index - 1, index, index + 1)
        for index in range(1, len(sessions) - 1)
        if (
            str(sessions[index - 1]) == str(sessions[index]) == str(sessions[index + 1])
            and int(timestamps[index]) - int(timestamps[index - 1]) == 100
            and int(timestamps[index + 1]) - int(timestamps[index]) == 100
        )
    )


def _empty_model_prediction_arrays() -> dict[str, np.ndarray]:
    return {
        key: np.empty((0, 6), dtype=dtype)
        if key == "probs"
        else np.empty(0, dtype=dtype)
        for key, dtype in _MODEL_PREDICTION_DTYPES.items()
    }


def _model_prediction_arrays(
    shard: Mapping[str, np.ndarray],
    source_model: nn.Module,
    student_model: ResNet18SimSiam,
    device: torch.device,
    batch_size: int,
    controls: Mapping[str, object],
) -> dict[str, np.ndarray]:
    if batch_size < 1:
        raise AlignmentError("prediction batch size must be positive")
    windows = _prediction_windows(shard)
    if not windows:
        return _empty_model_prediction_arrays()
    columns: dict[str, list[Any]] = {key: [] for key in _MODEL_PREDICTION_FIELDS}
    frames = shard["frames"]
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        flat = np.stack([frames[index] for window in batch for index in window])
        views = _safe_views(flat)
        x = _tensor(np.concatenate(views), device)
        with torch.no_grad():
            source_probs = F.softmax(cast(torch.Tensor, source_model(x)), dim=1).cpu().numpy()
            student_probs = F.softmax(student_model(x)[0], dim=1).cpu().numpy()
        for local_index, window in enumerate(batch):
            anchor_index = window[1]
            cut = any(
                float(
                    np.abs(
                        frames[right].astype(np.int16) - frames[left].astype(np.int16)
                    ).mean()
                    / 255.0
                )
                > 0.35
                for left, right in zip(window, window[1:], strict=False)
            )
            for augmentation_index, augmentation in enumerate(AUGMENTATION_KEYS):
                for offset, view in enumerate(VIEW_KEYS):
                    position = augmentation_index * len(batch) * 3 + local_index * 3 + offset
                    disagreement = float(
                        0.5 * np.abs(source_probs[position] - student_probs[position]).sum()
                    )
                    for role, probs in (
                        ("source", source_probs[position]),
                        ("student", student_probs[position]),
                    ):
                        columns["session_hash"].append(str(shard["session_hash"][anchor_index]))
                        columns["anchor_timestamp_ms"].append(int(shard["timestamp_ms"][anchor_index]))
                        columns["frame_timestamp_ms"].append(
                            int(shard["timestamp_ms"][window[offset]])
                        )
                        columns["frame_alignment_hash"].append(
                            str(shard["alignment_hash"][window[offset]])
                        )
                        columns["model_role"].append(role)
                        columns["view_id"].append(view)
                        columns["augmentation_id"].append(augmentation)
                        columns["probs"].append(probs)
                        columns["ood_score"].append(disagreement)
                        columns["black_control_ok"].append(bool(controls["black_control_ok"]))
                        columns["constant_control_ok"].append(bool(controls["constant_control_ok"]))
                        columns["cut"].append(cut)
    arrays: dict[str, np.ndarray] = {}
    for key, dtype in _MODEL_PREDICTION_DTYPES.items():
        arrays[key] = np.asarray(columns[key], dtype=dtype)
    if arrays["probs"].shape != (len(arrays["session_hash"]), 6):
        raise AlignmentError("model prediction probability rows are malformed")
    return arrays


def _write_model_prediction_shard(path: Path, arrays: Mapping[str, np.ndarray]) -> bytes:
    if set(arrays) != set(_MODEL_PREDICTION_FIELDS):
        raise AlignmentError("model prediction fields are not exact")
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # type: ignore[arg-type]
    data = buffer.getvalue()
    _write_exclusive(path, data)
    return data


def generate_v5_model_predictions(
    *,
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    target_shards: Sequence[Path],
    config_path: Path,
    source_model_path: Path,
    adapted_model_path: Path,
    output_dir: Path,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict[str, object]:
    """Write bounded, model-recomputed V5 train-split prediction evidence."""
    (
        source_state,
        _,
        source_sha,
        manifest,
        config,
        adapted_state,
        adapted_sha,
    ) = _prediction_binding(
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
        config_path,
        source_model_path,
        adapted_model_path,
    )
    target = _large_output_dir(output_dir)
    if os.path.lexists(target):
        raise AlignmentError("model prediction output directory already exists")
    run_device = torch.device(device)
    source_model = _source_network(source_state, run_device).eval()
    student_model = _adapted_model(adapted_state).to(run_device).eval()
    student_model.freeze_batch_norm()
    controls = _control_summary(source_model, student_model, run_device)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        shard_dir = staging / "shards"
        shard_dir.mkdir()
        entries: list[dict[str, object]] = []
        rows = 0
        for index, (source_path, shard) in enumerate(_bound_target_shards(manifest)):
            arrays = _model_prediction_arrays(
                shard, source_model, student_model, run_device, batch_size, controls
            )
            path = shard_dir / f"prediction-{index:05d}.npz"
            prediction_bytes = _write_model_prediction_shard(path, arrays)
            count = len(arrays["session_hash"])
            rows += count
            entries.append(
                {
                    "target_path": source_path.name,
                    "target_sha256": manifest.shard_sha256[source_path],
                    "path": path.name,
                    "sha256": _sha(prediction_bytes),
                    "row_count": count,
                }
            )
        control_hash = _sha(_json(controls).encode())
        payload: dict[str, object] = {
            "schema_version": MODEL_PREDICTION_SCHEMA,
            "manifest_sha256": manifest.manifest_sha256,
            "source_model_sha256": source_sha,
            "adapted_model_sha256": adapted_sha,
            "config_sha256": config["config_sha256"],
            "inference_contract_sha256": prediction_inference_contract_hash(),
            "ood_metric": "source_student_total_variation_v1",
            "controls": controls,
            "controls_sha256": control_hash,
            "shards": entries,
        }
        payload["predictions_sha256"] = _sha(_json(payload).encode())
        _write_exclusive(staging / "manifest.json", (_json(payload) + "\n").encode())
        staging.rename(target)
        return {
            "status": "PASSED",
            "disposition": "MODEL_GENERATED_NON_PROMOTING_PREDICTION_EVIDENCE",
            "output_dir": str(target),
            "prediction_rows": rows,
            "prediction_shards": len(entries),
            "predictions_sha256": payload["predictions_sha256"],
            "controls": controls,
            "human_labels_consumed": False,
            "real_domain_advice_released": False,
        }
    except BaseException:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                path.rmdir()
        staging.rmdir()
        raise


def _load_model_prediction_shard(path: Path) -> tuple[dict[str, np.ndarray], str]:
    data = _read_regular(path, ".npz")
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != set(_MODEL_PREDICTION_FIELDS):
                raise AlignmentError("model prediction NPZ fields are not exact")
            arrays = {key: archive[key].copy() for key in _MODEL_PREDICTION_FIELDS}
    except (OSError, ValueError) as exc:
        if isinstance(exc, AlignmentError):
            raise
        raise AlignmentError("invalid model prediction NPZ") from exc
    count = len(arrays["session_hash"])
    for key, dtype in _MODEL_PREDICTION_DTYPES.items():
        expected_shape = (count, 6) if key == "probs" else (count,)
        if arrays[key].dtype != dtype or arrays[key].shape != expected_shape:
            raise AlignmentError("model prediction NPZ dtype/shape is invalid")
    for index in range(count):
        if (
            not _valid_sha(str(arrays["session_hash"][index]))
            or not _valid_sha(str(arrays["frame_alignment_hash"][index]))
            or str(arrays["model_role"][index]) not in {"source", "student"}
            or str(arrays["view_id"][index]) not in VIEW_KEYS
            or str(arrays["augmentation_id"][index]) not in AUGMENTATION_KEYS
            or not math.isfinite(float(arrays["ood_score"][index]))
            or float(arrays["ood_score"][index]) < 0
        ):
            raise AlignmentError("model prediction row identity is invalid")
        _probabilities(tuple(float(value) for value in arrays["probs"][index]))
    return arrays, _sha(data)


def _model_prediction_manifest(
    directory: Path,
    manifest: V5Manifest,
    source_sha: str,
    adapted_sha: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    payload = _object(
        _load_json(directory / "manifest.json"),
        _MODEL_PREDICTION_MANIFEST_FIELDS,
        "model prediction manifest",
    )
    predictions_sha = _self_hash(payload, "predictions_sha256")
    if (
        payload["schema_version"] != MODEL_PREDICTION_SCHEMA
        or payload["manifest_sha256"] != manifest.manifest_sha256
        or payload["source_model_sha256"] != source_sha
        or payload["adapted_model_sha256"] != adapted_sha
        or payload["config_sha256"] != config["config_sha256"]
        or payload["inference_contract_sha256"] != prediction_inference_contract_hash()
        or payload["ood_metric"] != "source_student_total_variation_v1"
        or not isinstance(payload["controls"], dict)
        or not isinstance(payload["shards"], list)
        or not _is_sha(payload["controls_sha256"])
        or predictions_sha != payload["predictions_sha256"]
    ):
        raise AlignmentError("model prediction manifest binding is invalid")
    controls = cast(dict[str, object], payload["controls"])
    if (
        set(controls)
        != {
            "source_black_max_probability",
            "source_constant_max_probability",
            "student_black_max_probability",
            "student_constant_max_probability",
            "black_control_ok",
            "constant_control_ok",
        }
        or _sha(_json(controls).encode()) != payload["controls_sha256"]
        or any(
            not isinstance(controls[key], bool)
            for key in ("black_control_ok", "constant_control_ok")
        )
        or any(
            not math.isfinite(_number(controls[key], low=0.0, high=1.0))
            for key in (
                "source_black_max_probability",
                "source_constant_max_probability",
                "student_black_max_probability",
                "student_constant_max_probability",
            )
        )
    ):
        raise AlignmentError("model prediction controls are invalid")
    return payload


def _stream_filter_group(
    rows: Sequence[CandidatePrediction],
    last: dict[str, int],
    session_counts: dict[tuple[str, int], int],
    global_counts: dict[int, int],
) -> tuple[PseudoLabel | None, str | None]:
    if len(rows) != 18:
        return None, "exact_views"
    session, timestamp = rows[0].session_id, rows[0].timestamp_ms
    if any(row.session_id != session or row.timestamp_ms != timestamp for row in rows):
        return None, "identity"
    combos = {(row.model_id, row.view_id, row.augmentation_id) for row in rows}
    required = {
        (model, view, augmentation)
        for model in ("source", "student")
        for view in VIEW_KEYS
        for augmentation in AUGMENTATION_KEYS
    }
    if combos != required:
        return None, "exact_views"
    stats = [_probabilities(row.probs) for row in rows]
    if any(row.cut for row in rows):
        return None, "cut"
    if any(
        row.ood_score > 0.05 or not row.black_control_ok or not row.constant_control_ok for row in rows
    ):
        return None, "ood_control"
    classes = {value[0] for value in stats}
    if len(classes) != 1:
        return None, "agreement"
    if any(value[1] < 0.995 or value[2] < 0.5 for value in stats):
        return None, "confidence"
    class_id = stats[0][0]
    if timestamp - last.get(session, -(10**18)) < 500:
        return None, "interval"
    if session_counts.get((session, class_id), 0) >= 500:
        return None, "session_class_cap"
    if global_counts.get(class_id, 0) >= 2000:
        return None, "global_class_cap"
    evidence = _sha(
        _json(
            [
                {
                    "model_id": row.model_id,
                    "view_id": row.view_id,
                    "augmentation_id": row.augmentation_id,
                    "probs": row.probs,
                    "ood_score": row.ood_score,
                    "black_control_ok": row.black_control_ok,
                    "constant_control_ok": row.constant_control_ok,
                    "cut": row.cut,
                }
                for row in sorted(rows, key=lambda row: (row.model_id, row.view_id, row.augmentation_id))
            ]
        ).encode()
    )
    last[session] = timestamp
    session_counts[(session, class_id)] = session_counts.get((session, class_id), 0) + 1
    global_counts[class_id] = global_counts.get(class_id, 0) + 1
    return PseudoLabel(session, timestamp, class_id, min(value[1] for value in stats), min(value[2] for value in stats), evidence), None


def _model_prediction_evidence(
    predictions_path: Path,
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    target_shards: Sequence[Path],
    config_path: Path,
    source_model_path: Path,
    adapted_model_path: Path,
    device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], PseudoFilterReport]:
    (
        source_state,
        _,
        source_sha,
        manifest,
        config,
        adapted_state,
        adapted_sha,
    ) = _prediction_binding(
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
        config_path,
        source_model_path,
        adapted_model_path,
    )
    payload = _model_prediction_manifest(
        predictions_path, manifest, source_sha, adapted_sha, config
    )
    entries = cast(list[object], payload["shards"])
    run_device = torch.device(device)
    source_model = _source_network(source_state, run_device).eval()
    student_model = _adapted_model(adapted_state).to(run_device).eval()
    student_model.freeze_batch_norm()
    controls = _control_summary(source_model, student_model, run_device)
    if _sha(_json(controls).encode()) != payload["controls_sha256"]:
        raise AlignmentError("model prediction controls do not reproduce")
    last: dict[str, int] = {}
    session_counts: dict[tuple[str, int], int] = {}
    global_counts: dict[int, int] = {}
    rejected: dict[str, int] = defaultdict(int)
    selected_frames: list[np.ndarray] = []
    labels: list[int] = []
    sessions: list[str] = []
    timestamps: list[int] = []
    alignments: list[str] = []
    evidence: list[str] = []
    group_count = 0
    prediction_paths: list[Path] = []
    for entry, (target_path, target_shard) in zip(
        entries, _bound_target_shards(manifest), strict=True
    ):
        row = _object(
            entry,
            {"target_path", "target_sha256", "path", "sha256", "row_count"},
            "model prediction shard",
        )
        name, target_sha, prediction_name, prediction_sha, row_count = (
            row["target_path"],
            row["target_sha256"],
            row["path"],
            row["sha256"],
            row["row_count"],
        )
        if (
            name != target_path.name
            or target_sha != manifest.shard_sha256[target_path]
            or not isinstance(prediction_name, str)
            or Path(prediction_name).name != prediction_name
            or not _is_sha(prediction_sha)
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 0
        ):
            raise AlignmentError("model prediction shard binding is invalid")
        prediction_path = predictions_path / "shards" / prediction_name
        arrays, actual_sha = _load_model_prediction_shard(prediction_path)
        expected = _model_prediction_arrays(
            target_shard, source_model, student_model, run_device, batch_size, controls
        )
        if (
            actual_sha != prediction_sha
            or len(arrays["session_hash"]) != row_count
            or any(
                arrays[key].shape != expected[key].shape
                or arrays[key].dtype != expected[key].dtype
                or (
                    not np.allclose(arrays[key], expected[key], atol=1e-6, rtol=0.0)
                    if key in {"probs", "ood_score"}
                    else not np.array_equal(arrays[key], expected[key])
                )
                for key in _MODEL_PREDICTION_FIELDS
            )
        ):
            raise AlignmentError("model prediction rows do not reproduce")
        lookup = {
            (str(session), int(timestamp)): (frame, str(alignment))
            for frame, session, timestamp, alignment in zip(
                target_shard["frames"],
                target_shard["session_hash"],
                target_shard["timestamp_ms"],
                target_shard["alignment_hash"],
                strict=True,
            )
        }
        for start in range(0, len(arrays["session_hash"]), 18):
            group_count += 1
            stop = start + 18
            if stop > len(arrays["session_hash"]):
                raise AlignmentError("model prediction group is incomplete")
            rows = tuple(
                CandidatePrediction(
                    str(arrays["session_hash"][row_index]),
                    int(arrays["anchor_timestamp_ms"][row_index]),
                    str(arrays["model_role"][row_index]),
                    str(arrays["view_id"][row_index]),
                    tuple(float(value) for value in arrays["probs"][row_index]),
                    augmentation_id=str(arrays["augmentation_id"][row_index]),
                    ood_score=float(arrays["ood_score"][row_index]),
                    black_control_ok=bool(arrays["black_control_ok"][row_index]),
                    constant_control_ok=bool(arrays["constant_control_ok"][row_index]),
                    cut=bool(arrays["cut"][row_index]),
                )
                for row_index in range(start, stop)
            )
            label, reason = _stream_filter_group(rows, last, session_counts, global_counts)
            if reason is not None:
                rejected[reason] += 1
                continue
            assert label is not None
            frame, alignment = lookup.get((label.session_id, label.timestamp_ms), (None, ""))
            if frame is None:
                raise AlignmentError("accepted pseudo anchor does not join target shard")
            selected_frames.append(np.array(frame, dtype=np.uint8, copy=True))
            labels.append(label.class_id)
            sessions.append(label.session_id)
            timestamps.append(label.timestamp_ms)
            alignments.append(alignment)
            evidence.append(label.evidence_hash)
        prediction_paths.append(prediction_path)
    if {path.name for path in prediction_paths} != {
        cast(str, cast(dict[str, object], entry)["path"]) for entry in entries
    }:
        raise AlignmentError("model prediction paths are not exact")
    frames = (
        np.stack(selected_frames)
        if selected_frames
        else np.empty((0, 128, 128, 3), dtype=np.uint8)
    )
    arrays = {
        "frames": frames,
        "class_id": np.asarray(labels, dtype=np.int64),
        "session_hash": np.asarray(sessions, dtype="<U64"),
        "timestamp_ms": np.asarray(timestamps, dtype=np.int64),
        "alignment_hash": np.asarray(alignments, dtype="<U64"),
        "evidence_hash": np.asarray(evidence, dtype="<U64"),
        "manifest_sha256": np.asarray(manifest.manifest_sha256, dtype="<U64"),
        "predictions_sha256": np.asarray(payload["predictions_sha256"], dtype="<U64"),
        "source_model_sha256": np.asarray(source_sha, dtype="<U64"),
        "adapted_model_sha256": np.asarray(adapted_sha, dtype="<U64"),
        "config_sha256": np.asarray(config["config_sha256"], dtype="<U64"),
    }
    return arrays, PseudoFilterReport(
        group_count, len(selected_frames), dict(rejected), len(selected_frames) >= 200
    )


def _prediction_evidence(
    predictions_path: Path,
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    target_shards: Sequence[Path],
    config_path: Path,
    source_model_path: Path,
    adapted_model_path: Path,
    prediction_device: str = "cpu",
    prediction_batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], PseudoFilterReport]:
    if not predictions_path.is_dir() or predictions_path.is_symlink():
        raise AlignmentError("V5 pseudo materialization requires model-generated evidence directory")
    return _model_prediction_evidence(
        predictions_path,
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
        config_path,
        source_model_path,
        adapted_model_path,
        prediction_device,
        prediction_batch_size,
    )
    _, _, source_meta, source_sha = _source_bundle(
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
        config_path,
        source_model_path,
    )
    manifest, config = (
        load_v5_manifest(
            manifest_path,
            pre_ingest_path,
            privacy_context_path,
            owner_attestation_path,
            owner_component_confirmation_path,
            target_shards,
        ),
        load_v5_training_config(config_path),
    )
    _, adapted_meta, adapted_sha = _load_model(adapted_model_path, "v5_simsiam_adapted")
    expected_adapted = {
        "manifest_sha256": manifest.manifest_sha256,
        "split_binding_sha256": manifest.split_binding_sha256,
        "config_sha256": cast(str, config["config_sha256"]),
        "source_model_sha256": source_sha,
        "source_dataset_sha256": source_meta["source_dataset_sha256"],
        "source_metadata_sha256": source_meta["source_metadata_sha256"],
        "renderer_sha256": source_renderer_hash(),
        "teacher_sha256": causal_source_teacher_hash(),
        "action_schema_sha256": action_schema_hash(),
    }
    if any(adapted_meta[key] != value for key, value in expected_adapted.items()):
        raise AlignmentError("adapted model binding mismatch")
    payload = _object(
        _load_json(predictions_path),
        {
            "schema_version",
            "manifest_sha256",
            "source_model_sha256",
            "adapted_model_sha256",
            "config_sha256",
            "rows",
            "predictions_sha256",
        },
        "prediction evidence",
    )
    predictions_hash = _self_hash(payload, "predictions_sha256")
    if (
        payload["schema_version"] != PREDICTION_SCHEMA
        or payload["manifest_sha256"] != manifest.manifest_sha256
        or payload["source_model_sha256"] != source_sha
        or payload["adapted_model_sha256"] != adapted_sha
        or payload["config_sha256"] != config["config_sha256"]
        or not isinstance(payload["rows"], list)
    ):
        raise AlignmentError("prediction evidence binding mismatch")
    index, candidates, raw_groups = _target_index(manifest), [], defaultdict(list)
    delta = {"t-100": -100, "t": 0, "t+100": 100}
    fields = {
        "session_hash",
        "anchor_timestamp_ms",
        "frame_timestamp_ms",
        "frame_alignment_hash",
        "model_role",
        "view_id",
        "probs",
        "ood_score",
        "black_control_ok",
        "constant_control_ok",
        "cut",
    }
    for value in payload["rows"]:
        row = _object(value, fields, "prediction row")
        session, anchor, frame_time, alignment, role, view, probs = (
            row["session_hash"],
            row["anchor_timestamp_ms"],
            row["frame_timestamp_ms"],
            row["frame_alignment_hash"],
            row["model_role"],
            row["view_id"],
            row["probs"],
        )
        if (
            not isinstance(session, str)
            or not _valid_sha(session)
            or isinstance(anchor, bool)
            or not isinstance(anchor, int)
            or isinstance(frame_time, bool)
            or not isinstance(frame_time, int)
            or not isinstance(alignment, str)
            or role not in {"source", "student"}
            or view not in VIEW_KEYS
            or not isinstance(probs, list)
            or row["black_control_ok"] not in {True, False}
            or type(row["black_control_ok"]) is not bool
            or type(row["constant_control_ok"]) is not bool
            or type(row["cut"]) is not bool
        ):
            raise AlignmentError("invalid prediction row")
        target = index.get((session, frame_time))
        if (
            frame_time != anchor + delta[view]
            or target is None
            or target[1] != alignment
            or target[2] != "train"
        ):
            raise AlignmentError("prediction row does not join target/train shard")
        probability_tuple = tuple(_number(item, low=0.0, high=1.0) for item in probs)
        candidate = CandidatePrediction(
            session,
            anchor,
            role,
            view,
            probability_tuple,
            ood_score=_number(row["ood_score"], low=0.0),
            black_control_ok=row["black_control_ok"],
            constant_control_ok=row["constant_control_ok"],
            cut=row["cut"],
        )
        candidates.append(candidate)
        raw_groups[(session, anchor)].append(row)
    accepted, report = filter_pseudo_labels(candidates)
    selected_frames, labels, sessions, timestamps, alignments, evidence = [], [], [], [], [], []
    for label in accepted:
        frame, alignment, split = index[(label.session_id, label.timestamp_ms)]
        if split != "train":
            raise AlignmentError("accepted pseudo anchor is not train")
        selected_frames.append(frame)
        labels.append(label.class_id)
        sessions.append(label.session_id)
        timestamps.append(label.timestamp_ms)
        alignments.append(alignment)
        evidence.append(
            _sha(
                _json(
                    sorted(
                        raw_groups[(label.session_id, label.timestamp_ms)],
                        key=lambda row: (cast(str, row["model_role"]), cast(str, row["view_id"])),
                    )
                ).encode()
            )
        )
    frames = (
        np.stack(selected_frames) if selected_frames else np.empty((0, 128, 128, 3), dtype=np.uint8)
    )
    arrays = {
        "frames": frames.astype(np.uint8),
        "class_id": np.asarray(labels, dtype=np.int64),
        "session_hash": np.asarray(sessions, dtype="<U64"),
        "timestamp_ms": np.asarray(timestamps, dtype=np.int64),
        "alignment_hash": np.asarray(alignments, dtype="<U64"),
        "evidence_hash": np.asarray(evidence, dtype="<U64"),
        "manifest_sha256": np.asarray(manifest.manifest_sha256, dtype="<U64"),
        "predictions_sha256": np.asarray(predictions_hash, dtype="<U64"),
        "source_model_sha256": np.asarray(source_sha, dtype="<U64"),
        "adapted_model_sha256": np.asarray(adapted_sha, dtype="<U64"),
        "config_sha256": np.asarray(config["config_sha256"], dtype="<U64"),
    }
    return arrays, report


def _pseudo_artifact(path: Path, expected: Mapping[str, np.ndarray]) -> AcceptedPseudoDataset:
    data = _read_regular(path, ".npz")
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != set(_PSEUDO_FIELDS):
                raise AlignmentError("pseudo fields are not exact")
            actual = {key: archive[key].copy() for key in _PSEUDO_FIELDS}
    except (OSError, ValueError) as exc:
        if isinstance(exc, AlignmentError):
            raise
        raise AlignmentError("invalid pseudo NPZ") from exc
    if any(
        actual[key].dtype != expected[key].dtype
        or actual[key].shape != expected[key].shape
        or not np.array_equal(actual[key], expected[key])
        for key in _PSEUDO_FIELDS
    ):
        raise AlignmentError("pseudo artifact differs from re-filtered path evidence")
    return AcceptedPseudoDataset(
        actual["frames"],
        actual["class_id"],
        path,
        _sha(data),
        str(actual["manifest_sha256"].item()),
        str(actual["predictions_sha256"].item()),
    )


def materialize_v5_pseudo(
    *,
    predictions_path: Path,
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    target_shards: Sequence[Path],
    config_path: Path,
    source_model_path: Path,
    adapted_model_path: Path,
    output_path: Path,
    prediction_device: str = "cpu",
    prediction_batch_size: int = 256,
) -> tuple[AcceptedPseudoDataset, PseudoFilterReport]:
    arrays, report = _prediction_evidence(
        predictions_path,
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
        config_path,
        source_model_path,
        adapted_model_path,
        prediction_device,
        prediction_batch_size,
    )
    output_path = _large_output_path(output_path)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # type: ignore[arg-type]
    _write_exclusive(output_path, buffer.getvalue())
    return _pseudo_artifact(output_path, arrays), report


def load_accepted_pseudo_artifact(
    path: Path,
    *,
    predictions_path: Path,
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    target_shards: Sequence[Path],
    config_path: Path,
    source_model_path: Path,
    adapted_model_path: Path,
    prediction_device: str = "cpu",
    prediction_batch_size: int = 256,
) -> AcceptedPseudoDataset:
    arrays, _ = _prediction_evidence(
        predictions_path,
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
        config_path,
        source_model_path,
        adapted_model_path,
        prediction_device,
        prediction_batch_size,
    )
    return _pseudo_artifact(path, arrays)


def run_mean_teacher_round(
    *,
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    target_shards: Sequence[Path],
    predictions_path: Path,
    pseudo_path: Path,
    source_model_path: Path,
    adapted_checkpoint: Path,
    config_path: Path,
    ema_checkpoint: Path,
    round_ledger: Path,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, object]:
    source, _, source_meta, source_sha = _source_bundle(
        source_metadata_path,
        source_dataset_path,
        manifest_path,
        pre_ingest_path,
        privacy_context_path,
        owner_attestation_path,
        owner_component_confirmation_path,
        target_shards,
        config_path,
        source_model_path,
    )
    manifest, config = (
        load_v5_manifest(
            manifest_path,
            pre_ingest_path,
            privacy_context_path,
            owner_attestation_path,
            owner_component_confirmation_path,
            target_shards,
        ),
        load_v5_training_config(config_path),
    )
    pseudo = load_accepted_pseudo_artifact(
        pseudo_path,
        predictions_path=predictions_path,
        source_metadata_path=source_metadata_path,
        source_dataset_path=source_dataset_path,
        manifest_path=manifest_path,
        pre_ingest_path=pre_ingest_path,
        privacy_context_path=privacy_context_path,
        owner_attestation_path=owner_attestation_path,
        owner_component_confirmation_path=owner_component_confirmation_path,
        target_shards=target_shards,
        config_path=config_path,
        source_model_path=source_model_path,
        adapted_model_path=adapted_checkpoint,
        prediction_device=device,
    )
    if len(pseudo.frames) < 200:
        raise AlignmentError("Mean Teacher requires at least 200 accepted pseudo rows")
    ema_checkpoint = _large_output_path(ema_checkpoint)
    round_ledger = _large_output_path(round_ledger)
    state, adapted_meta, adapted_sha = _load_model(adapted_checkpoint, "v5_simsiam_adapted")
    expected = {
        "manifest_sha256": manifest.manifest_sha256,
        "split_binding_sha256": manifest.split_binding_sha256,
        "config_sha256": cast(str, config["config_sha256"]),
        "source_model_sha256": source_sha,
        "source_dataset_sha256": source_meta["source_dataset_sha256"],
        "source_metadata_sha256": source_meta["source_metadata_sha256"],
    }
    if any(adapted_meta[key] != value for key, value in expected.items()):
        raise AlignmentError("Mean Teacher adapted binding mismatch")
    torch.manual_seed(seed)
    run_device = torch.device(device)
    student = _adapted_model(state).to(run_device)
    teacher = copy.deepcopy(student).to(run_device).eval()
    student.train()
    student.freeze_batch_norm()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    source_train = source.frames if source.split is None else source.frames[source.split == "train"]
    source_labels = source.labels if source.split is None else source.labels[source.split == "train"]
    if len(source_train) != len(source_labels):
        raise AlignmentError("Mean Teacher source train rows are misaligned with labels")
    batch_size, epochs, ema_decay = (
        cast(int, config["batch_size"]),
        cast(int, config["mean_teacher_epochs"]),
        cast(float, config["ema_decay"]),
    )
    if batch_size < 2 or batch_size % 2:
        raise AlignmentError("Mean Teacher requires an even batch size")
    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=float(cast(int | float, config["learning_rate"])),
        weight_decay=float(cast(int | float, config["weight_decay"])),
    )
    half = max(1, batch_size // 2)
    steps = max(math.ceil(len(source_train) / half), math.ceil(len(pseudo.frames) / half))
    random = np.random.default_rng(seed)
    for _ in range(epochs):
        for _ in range(steps):
            source_index = random.integers(len(source_train), size=half)
            pseudo_index = random.integers(len(pseudo.frames), size=half)
            source_x = _tensor(source_train[source_index], run_device)
            source_y = torch.from_numpy(source_labels[source_index]).to(run_device)
            target_view = _augment(_tensor(pseudo.frames[pseudo_index], run_device))
            pseudo_y = torch.from_numpy(pseudo.labels[pseudo_index]).to(run_device)
            source_logits, _, _ = student(_augment(source_x))
            pseudo_logits, _, _ = student(target_view)
            with torch.no_grad():
                teacher_logits, _, _ = teacher(target_view)
            loss = (
                F.cross_entropy(source_logits, source_y)
                + F.cross_entropy(pseudo_logits, pseudo_y)
                + 0.5
                * F.kl_div(
                    F.log_softmax(pseudo_logits, dim=1),
                    F.softmax(teacher_logits, dim=1),
                    reduction="batchmean",
                )
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            student.freeze_batch_norm()
            with torch.no_grad():
                for target_parameter, source_parameter in zip(
                    teacher.parameters(), student.parameters(), strict=True
                ):
                    target_parameter.mul_(ema_decay).add_(source_parameter, alpha=1 - ema_decay)
    ema_meta = {
        "schema_version": MODEL_SCHEMA,
        "role": "v5_mean_teacher_ema",
        "manifest_sha256": manifest.manifest_sha256,
        "split_binding_sha256": manifest.split_binding_sha256,
        "config_sha256": cast(str, config["config_sha256"]),
        "renderer_sha256": source_renderer_hash(),
        "teacher_sha256": causal_source_teacher_hash(),
        "action_schema_sha256": action_schema_hash(),
        "source_model_sha256": source_sha,
        "source_dataset_sha256": source_meta["source_dataset_sha256"],
        "source_metadata_sha256": source_meta["source_metadata_sha256"],
        "adapted_model_sha256": adapted_sha,
        "pseudo_sha256": pseudo.artifact_sha256,
        "predictions_sha256": pseudo.predictions_sha256,
        "collapse_metrics_sha256": adapted_meta["collapse_metrics_sha256"],
        "round": "1",
    }
    ema_bytes = _model_bytes(teacher.state_dict(), ema_meta)
    ema_sha = _sha(ema_bytes)
    payload: dict[str, object] = {
        "schema_version": LEDGER_SCHEMA,
        "round": 1,
        "source_model_sha256": source_sha,
        "source_dataset_sha256": source_meta["source_dataset_sha256"],
        "source_metadata_sha256": source_meta["source_metadata_sha256"],
        "adapted_model_sha256": adapted_sha,
        "pseudo_sha256": pseudo.artifact_sha256,
        "predictions_sha256": pseudo.predictions_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "config_sha256": config["config_sha256"],
        "ema_model_sha256": ema_sha,
        "ema_decay": ema_decay,
        "weights": {"source_ce": 1.0, "pseudo_ce": 1.0, "kl": 0.5},
        "epochs": epochs,
        "batch_size": batch_size,
        "collapse_status": COLLAPSE_BLOCK,
    }
    payload["ledger_sha256"] = _sha(_json(payload).encode())
    _write_pair_exclusive(ema_checkpoint, ema_bytes, round_ledger, (_json(payload) + "\n").encode())
    return payload


def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise AlignmentError("kappa requires paired labels")
    classes = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    expected = sum(left.count(value) * right.count(value) for value in classes) / len(left) ** 2
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def _audit_diagnostics(
    labels: Sequence[AuditLabel],
    predictions: Sequence[AuditPrediction],
    regression: SourceRegression,
    evidence_hash: str,
) -> ReleaseGate:
    by_clip: dict[str, list[AuditLabel]] = defaultdict(list)
    for row in labels:
        if row.observed_action not in ACTION_TYPES:
            raise AlignmentError("audit action outside exact vocabulary")
        by_clip[row.clip_id].append(row)
    if len(by_clip) != 300 or any(
        len(rows) != 2 or len({row.reviewer for row in rows}) != 2 for rows in by_clip.values()
    ):
        raise AlignmentError("V5 audit composition is exactly 300 clips x two reviewers")
    prediction_map = {row.clip_id: row for row in predictions}
    if len(prediction_map) != 300 or set(prediction_map) != set(by_clip):
        raise AlignmentError("audit requires one sealed prediction per clip")
    ordered = sorted(by_clip)
    left = [
        by_clip[key][0].observed_action if by_clip[key][0].validity else "INVALID"
        for key in ordered
    ]
    right = [
        by_clip[key][1].observed_action if by_clip[key][1].validity else "INVALID"
        for key in ordered
    ]
    kappa = _cohen_kappa(left, right)
    valid_consensus: dict[str, str] = {
        key: left[index]
        for index, key in enumerate(ordered)
        if left[index] == right[index] and left[index] != "INVALID"
    }
    invalid_consensus = {
        key for index, key in enumerate(ordered) if left[index] == right[index] == "INVALID"
    }
    accepted = [prediction_map[key] for key in valid_consensus if prediction_map[key].accepted]
    correct = [row for row in accepted if row.action == valid_consensus[row.clip_id]]
    overall = len(correct) / max(1, len(accepted))
    coverage = len(accepted) / 300
    ood_false = sum(prediction_map[key].accepted for key in invalid_consensus) / max(
        1, len(invalid_consensus)
    )
    class_precision = {
        name: sum(
            row.action == valid_consensus[row.clip_id] for row in accepted if row.action == name
        )
        / max(1, sum(row.action == name for row in accepted))
        for name in ACTION_TYPES
    }
    baselines = tuple(
        overall
        - sum(row.baselines[index] == valid_consensus[row.clip_id] for row in accepted)
        / max(1, len(accepted))
        for index in range(2)
    )
    accuracy_drop = regression.accuracy_before - regression.accuracy_after
    if len(regression.recall_before) != 6 or len(regression.recall_after) != 6:
        raise AlignmentError("source regression requires six class recalls")
    recall_drop = max(
        a - b for a, b in zip(regression.recall_before, regression.recall_after, strict=True)
    )
    return ReleaseGate(
        False,
        (),
        kappa,
        overall,
        class_precision,
        coverage,
        ood_false,
        cast(tuple[float, float], baselines),
        accuracy_drop,
        recall_drop,
        evidence_hash,
        COLLAPSE_BLOCK,
        None,
    )


def _load_ledger(path: Path) -> dict[str, object]:
    fields = {
        "schema_version",
        "round",
        "source_model_sha256",
        "source_dataset_sha256",
        "source_metadata_sha256",
        "adapted_model_sha256",
        "pseudo_sha256",
        "predictions_sha256",
        "manifest_sha256",
        "config_sha256",
        "ema_model_sha256",
        "ema_decay",
        "weights",
        "epochs",
        "batch_size",
        "collapse_status",
        "ledger_sha256",
    }
    payload = _object(_load_json(path), fields, "Mean Teacher ledger")
    if (
        payload["schema_version"] != LEDGER_SCHEMA
        or payload["round"] != 1
        or payload["ema_decay"] != 0.999
        or payload["weights"] != {"source_ce": 1.0, "pseudo_ce": 1.0, "kl": 0.5}
        or payload["collapse_status"] != COLLAPSE_BLOCK
    ):
        raise AlignmentError("Mean Teacher ledger contract is invalid")
    if any(
        not isinstance(payload[key], str) or not _valid_sha(cast(str, payload[key]))
        for key in fields
        if key.endswith("sha256")
    ):
        raise AlignmentError("Mean Teacher ledger hashes are invalid")
    _self_hash(payload, "ledger_sha256")
    return payload


def path_only_v5_release_gate(
    *,
    manifest_path: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    target_shards: Sequence[Path],
    sealed_predictions_path: Path,
    sealed_audit_path: Path,
    model_path: Path,
    mean_teacher_ledger_path: Path,
    config_path: Path,
    release_path: Path,
) -> ReleaseGate:
    if os.path.lexists(release_path):
        raise AlignmentError("release path must not already exist")
    manifest, config = (
        load_v5_manifest(
            manifest_path,
            pre_ingest_path,
            privacy_context_path,
            owner_attestation_path,
            owner_component_confirmation_path,
            target_shards,
        ),
        load_v5_training_config(config_path),
    )
    _, model_meta, model_sha = _load_model(model_path, "v5_mean_teacher_ema")
    ledger = _load_ledger(mean_teacher_ledger_path)
    if (
        ledger["ema_model_sha256"] != model_sha
        or ledger["manifest_sha256"] != manifest.manifest_sha256
        or ledger["config_sha256"] != config["config_sha256"]
        or model_meta["split_binding_sha256"] != manifest.split_binding_sha256
        or any(
            model_meta[key] != ledger[key]
            for key in (
                "manifest_sha256",
                "config_sha256",
                "source_model_sha256",
                "source_dataset_sha256",
                "source_metadata_sha256",
                "adapted_model_sha256",
                "pseudo_sha256",
                "predictions_sha256",
            )
        )
    ):
        raise AlignmentError("EMA model/ledger/manifest/config binding mismatch")
    sealed = _object(
        _load_json(sealed_predictions_path),
        {
            "schema_version",
            "manifest_sha256",
            "model_sha256",
            "config_sha256",
            "selection_sha256",
            "source_regression",
            "rows",
            "predictions_sha256",
        },
        "sealed predictions",
    )
    predictions_hash = _self_hash(sealed, "predictions_sha256")
    if (
        sealed["schema_version"] != "hok-agent-v5-sealed-predictions-v1"
        or sealed["manifest_sha256"] != manifest.manifest_sha256
        or sealed["model_sha256"] != model_sha
        or sealed["config_sha256"] != config["config_sha256"]
        or not _is_sha(sealed["selection_sha256"])
        or not isinstance(sealed["rows"], list)
    ):
        raise AlignmentError("sealed prediction binding is invalid")
    regression_row = _object(
        sealed["source_regression"],
        {"accuracy_before", "accuracy_after", "recall_before", "recall_after"},
        "source regression",
    )
    before, after = regression_row["recall_before"], regression_row["recall_after"]
    if (
        not isinstance(before, list)
        or not isinstance(after, list)
        or len(before) != 6
        or len(after) != 6
    ):
        raise AlignmentError("source regression requires six recalls")
    regression = SourceRegression(
        _number(regression_row["accuracy_before"], low=0.0, high=1.0),
        _number(regression_row["accuracy_after"], low=0.0, high=1.0),
        tuple(_number(value, low=0.0, high=1.0) for value in before),
        tuple(_number(value, low=0.0, high=1.0) for value in after),
    )
    index, predictions, identities, clips, sealed_identities = (
        _target_index(manifest),
        [],
        {},
        set(),
        set(),
    )
    prediction_fields = {
        "clip_id",
        "session_hash",
        "timestamp_ms",
        "alignment_hash",
        "action",
        "accepted",
        "confidence",
        "baselines",
    }
    for value in sealed["rows"]:
        row = _object(value, prediction_fields, "sealed prediction row")
        clip, session, timestamp, alignment, action, accepted, baselines = (
            row["clip_id"],
            row["session_hash"],
            row["timestamp_ms"],
            row["alignment_hash"],
            row["action"],
            row["accepted"],
            row["baselines"],
        )
        if (
            not isinstance(clip, str)
            or not clip
            or clip in clips
            or not isinstance(session, str)
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or not isinstance(alignment, str)
            or action not in ACTION_TYPES
            or type(accepted) is not bool
            or not isinstance(baselines, list)
            or len(baselines) != 2
            or any(item not in ACTION_TYPES for item in baselines)
        ):
            raise AlignmentError("invalid sealed prediction row")
        target = index.get((session, timestamp))
        identity = (session, timestamp, alignment)
        if (
            target is None
            or target[1] != alignment
            or target[2] != "test"
            or identity in sealed_identities
        ):
            raise AlignmentError("sealed prediction must join target/test")
        confidence = _number(row["confidence"], low=0.0, high=1.0)
        clips.add(clip)
        sealed_identities.add(identity)
        identities[clip] = identity
        predictions.append(
            AuditPrediction(
                clip,
                action,
                accepted,
                confidence,
                (cast(str, baselines[0]), cast(str, baselines[1])),
            )
        )
    if len(predictions) != 300:
        raise AlignmentError("sealed predictions require exactly 300 unique clips")
    audit = _object(
        _load_json(sealed_audit_path),
        {"schema_version", "manifest_sha256", "selection_sha256", "rows", "audit_sha256"},
        "sealed audit",
    )
    audit_hash = _self_hash(audit, "audit_sha256")
    if (
        audit["schema_version"] != AUDIT_SCHEMA
        or audit["manifest_sha256"] != manifest.manifest_sha256
        or audit["selection_sha256"] != sealed["selection_sha256"]
        or not isinstance(audit["rows"], list)
    ):
        raise AlignmentError("sealed audit binding is invalid")
    labels: list[AuditLabel] = []
    per_clip: dict[str, set[str]] = defaultdict(set)
    reviewers: set[str] = set()
    audit_fields = {
        "clip_id",
        "session_hash",
        "timestamp_ms",
        "alignment_hash",
        "reviewer",
        "observed_action",
        "validity",
    }
    for value in audit["rows"]:
        row = _object(value, audit_fields, "audit row")
        clip, reviewer, action, validity = (
            row["clip_id"],
            row["reviewer"],
            row["observed_action"],
            row["validity"],
        )
        audit_identity = (row["session_hash"], row["timestamp_ms"], row["alignment_hash"])
        if (
            not isinstance(clip, str)
            or clip not in identities
            or audit_identity != identities[clip]
            or not isinstance(reviewer, str)
            or not reviewer
            or reviewer in per_clip[clip]
            or action not in ACTION_TYPES
            or type(validity) is not bool
        ):
            raise AlignmentError("invalid or unbound audit row")
        per_clip[clip].add(reviewer)
        reviewers.add(reviewer)
        labels.append(AuditLabel(clip, reviewer, action, validity))
    if (
        len(labels) != 600
        or set(per_clip) != clips
        or len(reviewers) != 2
        or any(values != reviewers for values in per_clip.values())
    ):
        raise AlignmentError("V5 audit is exactly 300 clips x the same two reviewers")
    evidence_hash = _sha(
        _json(
            [
                manifest.manifest_sha256,
                model_sha,
                predictions_hash,
                audit_hash,
                ledger["ledger_sha256"],
            ]
        ).encode()
    )
    gate = _audit_diagnostics(labels, predictions, regression, evidence_hash)
    # The source thresholds are frozen, but no accepted real-domain collapse evidence exists.
    # Formal promotion and release writing therefore remain fail-closed.
    return gate


def _release_payload(data: bytes) -> dict[str, object]:
    payload = _load_json_bytes(data)
    fields = {
        "schema_version",
        "model_sha256",
        "alignment_sha256",
        "audit_sha256",
        "config_sha256",
        "overall_pass",
        "allowed_classes",
        "class_thresholds",
        "thresholds",
        "thresholds_hash",
        "release_sha256",
    }
    _object(payload, fields, "release")
    if payload["schema_version"] != RELEASE_SCHEMA or payload["overall_pass"] is not True:
        raise AlignmentError("release schema/gate is invalid")
    if any(
        not _is_sha(payload[key])
        for key in (
            "model_sha256",
            "alignment_sha256",
            "audit_sha256",
            "config_sha256",
            "thresholds_hash",
            "release_sha256",
        )
    ):
        raise AlignmentError("release hashes are invalid")
    allowed = payload["allowed_classes"]
    class_thresholds = payload["class_thresholds"]
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(value not in ACTION_TYPES for value in allowed)
        or len(set(allowed)) != len(allowed)
    ):
        raise AlignmentError("release classes are invalid")
    if (
        not isinstance(class_thresholds, dict)
        or set(class_thresholds) != set(allowed)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0.75 <= float(value) <= 1.0
            for value in class_thresholds.values()
        )
    ):
        raise AlignmentError("release class thresholds are invalid")
    frozen = {
        "kappa": 0.70,
        "overall_precision": 0.85,
        "per_class_precision": 0.75,
        "coverage": 0.30,
        "ood_false_accept": 0.05,
        "baseline_delta": 0.05,
        "source_accuracy_drop": 0.02,
        "source_recall_drop": 0.05,
    }
    if payload["thresholds"] != frozen:
        raise AlignmentError("release thresholds differ from frozen V5 contract")
    if _sha(_json(payload["thresholds"]).encode()) != payload["thresholds_hash"]:
        raise AlignmentError("release thresholds hash mismatch")
    _self_hash(payload, "release_sha256")
    return payload


def load_release(path: Path) -> dict[str, object]:
    """Validate a candidate release, then fail closed until collapse is frozen."""
    _release_payload(_read_regular(path, ".json"))
    raise AlignmentError(COLLAPSE_BLOCK)


def load_bound_v5_release(release_path: Path, model_path: Path) -> BoundV5Release:
    release_bytes = _read_regular(release_path, ".json")
    payload = _release_payload(release_bytes)
    _, metadata, model_sha = _load_model(model_path, "v5_mean_teacher_ema")
    if (
        payload["model_sha256"] != model_sha
        or payload["alignment_sha256"] != metadata["manifest_sha256"]
        or payload["config_sha256"] != metadata["config_sha256"]
    ):
        raise AlignmentError("release does not bind actual model bytes/metadata")
    thresholds = cast(dict[str, int | float], payload["class_thresholds"])
    bound = BoundV5Release(
        _sha(release_bytes),
        model_sha,
        tuple(cast(list[str], payload["allowed_classes"])),
        {key: float(value) for key, value in thresholds.items()},
        metadata["manifest_sha256"],
        metadata["split_binding_sha256"],
    )
    del bound
    raise AlignmentError(COLLAPSE_BLOCK)


def launch_offline_audit_ui(
    clips: Sequence[AuditClip], *, reviewer: str, output_path: Path, enable_gui: bool = True
) -> dict[str, object]:
    """Retired: V5/V6 no longer collect human per-frame action labels."""
    del clips, reviewer, output_path, enable_gui
    raise AlignmentError(
        "per-frame action audit is retired; use a future gameplay-quality post-training contract"
    )
