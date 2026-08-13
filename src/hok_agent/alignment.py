# ruff: noqa: E501, E302, E305
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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageTk
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18  # type: ignore[import-untyped]

ACTION_TYPES = ("wait", "forward", "backward", "attack_hero", "attack_tower", "attack_crystal")
VIEW_KEYS = ("t-100", "t", "t+100")
SPLITS = {"train", "dev", "test"}
SOURCES = {"source", "target"}
NPZ_FIELDS = (
    "frames",
    "session_hash",
    "timestamp_ms",
    "pts",
    "time_base",
    "frame_hash",
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
    "frame_hash": np.dtype("<U64"),
    "alignment_hash": np.dtype("<U64"),
    "split": np.dtype("<U5"),
    "source": np.dtype("<U6"),
}
FORBIDDEN_DERIVED_KEYS = {
    "label",
    "labels",
    "legal",
    "reward",
    "state",
    "path",
    "account",
    "audit",
    "reviewer",
}
RENDERER_SPEC = {
    "id": "pixelarena-v5-source-rgb128-v1",
    "shape": [128, 128, 3],
    "dtype": "uint8",
    "hud": "six-health-bars-no-tick",
    "distortion": "color-quantize-downsample-blur-shift-le4",
}
MANIFEST_SCHEMA = "hok-agent-v5-manifest-v1"
SOURCE_SCHEMA = "hok-agent-v5-source-dataset-v1"
MODEL_SCHEMA = "hok-agent-v5-model-v1"
PREDICTION_SCHEMA = "hok-agent-v5-predictions-v1"
AUDIT_SCHEMA = "hok-agent-v5-audit-v1"
LEDGER_SCHEMA = "hok-agent-v5-mean-teacher-ledger-v1"
RELEASE_SCHEMA = "hok-agent-v5-release-v1"
COLLAPSE_BLOCK = "UNSPECIFIED_NUMERIC_THRESHOLD_BLOCKING"
class AlignmentError(ValueError):
    """Malformed or incomplete V5 evidence."""
@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    family_id: str
    parent_id: str | None = None
    near_duplicates: tuple[str, ...] = ()
@dataclass(frozen=True)
class CandidatePrediction:
    session_id: str
    timestamp_ms: int
    model_id: str
    view_id: str
    probs: tuple[float, ...]
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
@dataclass(frozen=True)
class AcceptedPseudoDataset:
    frames: np.ndarray
    labels: np.ndarray
    artifact_path: Path
    artifact_sha256: str
    manifest_sha256: str
    predictions_sha256: str
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
    session_splits: dict[str, str]
    shard_paths: tuple[Path, ...]
    shards: tuple[dict[str, np.ndarray], ...]
@dataclass(frozen=True)
class BoundV5Release:
    release_sha256: str
    model_sha256: str
    allowed_classes: tuple[str, ...]
    class_thresholds: dict[str, float]
def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
def _integer(value: object) -> int:
    return int(cast(Any, value))
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def _valid_sha(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
def _is_sha(value: object) -> bool:
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
    get = action.get if isinstance(action, Mapping) else lambda key, default="": getattr(action, key, default)
    kind, target, direction = str(get("action_type", "")), str(get("target", "")), str(get("direction", ""))
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
def source_render_128_rgb(observation: Mapping[str, object]) -> np.ndarray:
    """Render an independent deterministic source view with no tick/progress channel."""
    public = strip_to_public_observation(observation)
    seed = int(_sha(_json(public).encode())[:16], 16)
    rng = np.random.default_rng(seed)
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
    related = (_json, _sha, _integer, strip_to_public_observation, _draw_bar, _shift_without_wrap, source_render_128_rgb)
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
            parse_constant=lambda value: (_ for _ in ()).throw(AlignmentError(f"invalid number {value}")),
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
    if _sha(_json({key: value for key, value in payload.items() if key != field}).encode()) != supplied:
        raise AlignmentError(f"{field} mismatch")
    return supplied
def _number(value: object, *, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise AlignmentError("numeric evidence must be finite and non-boolean")
    result = float(value)
    if (low is not None and result < low) or (high is not None and result > high):
        raise AlignmentError("numeric evidence is outside its domain")
    return result
def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
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
    families: dict[str, str] = {}
    for row in records:
        if not row.family_id or row.parent_id == row.session_id or len(set(row.near_duplicates)) != len(row.near_duplicates) or row.session_id in row.near_duplicates:
            raise AlignmentError("invalid family/self/duplicate relationship")
        references = ({row.parent_id} if row.parent_id is not None else set()) | set(row.near_duplicates)
        if not references.issubset(known):
            raise AlignmentError("parent and near-duplicate references must be known")
        if row.family_id in families:
            groups.union(row.session_id, families[row.family_id])
        else:
            families[row.family_id] = row.session_id
        for reference in references:
            groups.union(row.session_id, reference)
    components: dict[str, list[str]] = defaultdict(list)
    for session_id in ids:
        components[groups.find(session_id)].append(session_id)
    return tuple(sorted((tuple(sorted(values)) for values in components.values()), key=lambda values: values[0]))
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
def privacy_mask_and_letterbox_rgb(frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    value = _rgb(frame).copy()
    if mask is not None:
        if mask.dtype != np.bool_ or mask.shape != value.shape[:2]:
            raise AlignmentError("privacy mask must be boolean and match frame")
        value[mask] = 0
    scale = min(128 / value.shape[1], 128 / value.shape[0])
    size = (max(1, round(value.shape[1] * scale)), max(1, round(value.shape[0] * scale)))
    resized = np.asarray(Image.fromarray(value).resize(size, Image.Resampling.BILINEAR))
    output = np.zeros((128, 128, 3), dtype=np.uint8)
    x, y = (128 - size[0]) // 2, (128 - size[1]) // 2
    output[y : y + size[1], x : x + size[0]] = resized
    return output
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
    records: Sequence[Mapping[str, object]], output_dir: Path, *, shard_size: int = 256
) -> tuple[Path, ...]:
    if shard_size < 1:
        raise AlignmentError("shard_size must be positive")
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
            mask = item.get("privacy_mask")
            if source == "target" and mask is None:
                raise AlignmentError("target frames require an explicit privacy mask")
            frame = privacy_mask_and_letterbox_rgb(
                np.asarray(item["frame"]), None if mask is None else np.asarray(mask)
            )
            session = _session_hash(item)
            timestamp, pts = _integer(item["timestamp_ms"]), _integer(item["pts"])
            base = item["time_base"]
            if not isinstance(base, Sequence) or isinstance(base, (str, bytes)) or len(base) != 2:
                raise AlignmentError("time_base must be an integer numerator/denominator pair")
            time_base = (int(base[0]), int(base[1]))
            if time_base[1] <= 0:
                raise AlignmentError("time_base denominator must be positive")
            frame_hash = _sha(frame.tobytes())
            alignment = _sha(
                _json([session, timestamp, pts, time_base, frame_hash, split, source]).encode()
            )
            values = (frame, session, timestamp, pts, time_base, frame_hash, alignment, split, source)
            for key, value in zip(NPZ_FIELDS, values, strict=True):
                columns[key].append(value)
        payload = {
            key: np.asarray(value, dtype=NPZ_DTYPES[key]) for key, value in columns.items()
        }
        path, buffer = output_dir / f"alignment-{shard:04d}.npz", io.BytesIO()
        np.savez(buffer, **payload)  # type: ignore[arg-type]
        _write_exclusive(path, buffer.getvalue())
        paths.append(path)
    return tuple(paths)
def _load_npz_bytes(data: bytes) -> dict[str, np.ndarray]:
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
    for index in range(count):
        split, source = str(payload["split"][index]), str(payload["source"][index])
        session, frame_hash = str(payload["session_hash"][index]), str(payload["frame_hash"][index])
        if split not in SPLITS or source not in SOURCES or not _valid_sha(session) or int(payload["time_base"][index][1]) <= 0:
            raise AlignmentError("invalid NPZ enum/hash")
        if _sha(payload["frames"][index].tobytes()) != frame_hash:
            raise AlignmentError("frame hash mismatch")
        material = [session, int(payload["timestamp_ms"][index]), int(payload["pts"][index]), tuple(int(x) for x in payload["time_base"][index]), frame_hash, split, source]
        if _sha(_json(material).encode()) != str(payload["alignment_hash"][index]):
            raise AlignmentError("alignment hash mismatch")
    return payload
def _load_npz(path: Path) -> dict[str, np.ndarray]:
    return _load_npz_bytes(_read_regular(path, ".npz"))
def load_npz_shards(paths: Sequence[Path]) -> tuple[dict[str, np.ndarray], ...]:
    return tuple(_load_npz(path) for path in paths)
def load_v5_manifest(manifest_path: Path, shard_paths: Sequence[Path]) -> V5Manifest:
    payload = _object(_load_json(manifest_path), {"schema_version", "sessions", "shards", "manifest_sha256"}, "manifest")
    if payload["schema_version"] != MANIFEST_SCHEMA:
        raise AlignmentError("manifest schema is invalid")
    manifest_hash = _self_hash(payload, "manifest_sha256")
    if not isinstance(payload["sessions"], list) or not isinstance(payload["shards"], list):
        raise AlignmentError("manifest sessions/shards must be lists")
    records, declared_splits = [], {}
    for value in payload["sessions"]:
        row = _object(value, {"session_hash", "family_id", "parent_hash", "near_duplicate_hashes", "split"}, "session")
        session, family, parent, duplicates, split = row["session_hash"], row["family_id"], row["parent_hash"], row["near_duplicate_hashes"], row["split"]
        if not isinstance(session, str) or not _valid_sha(session) or not isinstance(family, str) or not family or (parent is not None and (not isinstance(parent, str) or not _valid_sha(parent))) or not isinstance(duplicates, list) or any(not isinstance(item, str) or not _valid_sha(item) for item in duplicates) or split not in SPLITS:
            raise AlignmentError("invalid manifest session row")
        records.append(SessionRecord(session, family, parent, tuple(cast(list[str], duplicates))))
        declared_splits[session] = split
    if len(records) < 12 or len(declared_splits) != len(records):
        raise AlignmentError("manifest requires unique sessions and at least 12 rows")
    for component in _linked_groups(records):
        if len({declared_splits[item] for item in component}) != 1:
            raise AlignmentError("linked sessions cross manifest splits")
    counts = {name: sum(value == name for value in declared_splits.values()) for name in SPLITS}
    if counts["train"] < 8 or counts["dev"] < 2 or counts["test"] < 2:
        raise AlignmentError("manifest does not satisfy 8/2/2")
    supplied = {path.name: path for path in shard_paths}
    if len(supplied) != len(shard_paths):
        raise AlignmentError("shard paths must be unique")
    loaded, seen_sessions, seen_alignment, seen_key, expected_names = [], set(), set(), set(), set()
    for value in payload["shards"]:
        row = _object(value, {"path", "sha256", "row_count", "session_hashes", "split", "source"}, "shard")
        name, digest, row_count, sessions, split, source = row["path"], row["sha256"], row["row_count"], row["session_hashes"], row["split"], row["source"]
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix != ".npz" or name in expected_names or not isinstance(digest, str) or not _valid_sha(digest) or isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1 or not isinstance(sessions, list) or not sessions or len(set(cast(list[object], sessions))) != len(sessions) or any(not isinstance(item, str) or item not in declared_splits for item in sessions) or split not in SPLITS or source != "target":
            raise AlignmentError("invalid manifest shard row")
        expected_names.add(name)
        path = supplied.get(name)
        if path is None:
            raise AlignmentError("manifest shard path/hash mismatch")
        shard_bytes = _read_regular(path, ".npz")
        if _sha(shard_bytes) != digest:
            raise AlignmentError("manifest shard path/hash mismatch")
        shard = _load_npz_bytes(shard_bytes)
        actual_sessions = {str(item) for item in shard["session_hash"]}
        if len(shard["frames"]) != row_count or actual_sessions != set(cast(list[str], sessions)) or set(map(str, shard["split"])) != {split} or set(map(str, shard["source"])) != {source} or any(declared_splits[item] != split for item in actual_sessions):
            raise AlignmentError("manifest shard row/session/split mismatch")
        for session, timestamp, alignment in zip(shard["session_hash"], shard["timestamp_ms"], shard["alignment_hash"], strict=True):
            key = (str(session), int(timestamp))
            if str(alignment) in seen_alignment or key in seen_key:
                raise AlignmentError("manifest alignment/session timestamp must be globally unique")
            seen_alignment.add(str(alignment))
            seen_key.add(key)
        seen_sessions.update(actual_sessions)
        loaded.append(shard)
    if set(supplied) != expected_names or seen_sessions != set(declared_splits):
        raise AlignmentError("manifest paths and sessions must exactly cover shards")
    return V5Manifest(manifest_path, manifest_hash, declared_splits, tuple(supplied[name] for name in sorted(supplied)), tuple(loaded))
def build_training_config(*, batch_size: int = 256, learning_rate: float = 3e-4, weight_decay: float = 1e-4, epochs: int = 50, mean_teacher_epochs: int = 20) -> dict[str, object]:
    config: dict[str, object] = {"schema_version": "hok-agent-v5-training-config-v1", "architecture": "torchvision-resnet18-simsiam-c6", "optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": weight_decay, "batch_size": batch_size, "epochs": epochs, "mean_teacher_epochs": mean_teacher_epochs, "ema_decay": 0.999, "source_target_ratio": "50/50", "trainable": "stem+layer1+layer2+projector+predictor", "frozen": "fc+layer3+layer4+all_batchnorm", "audit_input": False}
    config["config_sha256"] = _sha(_json(config).encode())
    return config
def load_v5_training_config(path: Path) -> dict[str, object]:
    fields = set(build_training_config())
    config = _object(_load_json(path), fields, "training config")
    if config["schema_version"] != "hok-agent-v5-training-config-v1" or config["architecture"] != "torchvision-resnet18-simsiam-c6" or config["optimizer"] != "AdamW" or config["ema_decay"] != 0.999 or config["source_target_ratio"] != "50/50" or config["audit_input"] is not False:
        raise AlignmentError("training config identity is invalid")
    for key in ("batch_size", "epochs", "mean_teacher_epochs"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or cast(int, config[key]) < 1:
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
    last: dict[str, int] = defaultdict(lambda: -10**18)
    for (session, timestamp), rows in sorted(grouped.items()):
        combos = {(row.model_id, row.view_id): row for row in rows}
        required = {(model, view) for model in ("source", "student") for view in VIEW_KEYS}
        reason = ""
        if set(combos) != required or len(rows) != 6:
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
        if not reason and any(value[1] < min_confidence or value[2] < min_margin for value in stats):
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
        evidence = _sha(_json([asdict(row) for row in sorted(rows, key=lambda x: (x.model_id, x.view_id))]).encode())
        accepted.append(
            PseudoLabel(session, timestamp, class_id, min(x[1] for x in stats), min(x[2] for x in stats), evidence)
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
        self.projector = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128))
        self.predictor = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 128))
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
def _tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frames).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
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
        "black_constant_distance": float(torch.norm(control_embeddings[0] - control_embeddings[1]).item()),
        "collapse_status": COLLAPSE_BLOCK,
    }
def action_schema_hash() -> str:
    return _sha(_json(ACTION_TYPES).encode())
def causal_source_teacher_hash() -> str:
    functions = (strip_to_public_observation, _action_name, CausalSourceTeacher.select)
    return _sha(("\n".join(inspect.getsource(value) for value in functions) + _json(ACTION_TYPES)).encode())
_MODEL_FIELDS = {
    "v5_causal_source_teacher": {"schema_version", "role", "manifest_sha256", "config_sha256", "renderer_sha256", "teacher_sha256", "action_schema_sha256", "source_dataset_sha256", "source_metadata_sha256"},
    "v5_simsiam_adapted": {"schema_version", "role", "manifest_sha256", "config_sha256", "renderer_sha256", "teacher_sha256", "action_schema_sha256", "source_model_sha256", "source_dataset_sha256", "source_metadata_sha256", "collapse_metrics_sha256"},
    "v5_mean_teacher_ema": {"schema_version", "role", "manifest_sha256", "config_sha256", "renderer_sha256", "teacher_sha256", "action_schema_sha256", "source_model_sha256", "source_dataset_sha256", "source_metadata_sha256", "adapted_model_sha256", "pseudo_sha256", "predictions_sha256", "collapse_metrics_sha256", "round"},
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
    if type(metadata) is not dict or any(not isinstance(key, str) or not isinstance(value, str) for key, value in cast(dict[object, object], metadata).items()):
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
    return save_safetensors({key: value.detach().cpu().contiguous() for key, value in state.items()}, metadata=dict(metadata))
def _source_npz(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != {"frames", "class_id", "alignment_hash"}:
                raise AlignmentError("source NPZ fields are not exact")
            frames, labels, alignment = (archive[key].copy() for key in ("frames", "class_id", "alignment_hash"))
    except (OSError, ValueError) as exc:
        if isinstance(exc, AlignmentError):
            raise
        raise AlignmentError("invalid source NPZ") from exc
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (128, 128, 3) or labels.dtype != np.int64 or labels.shape != (len(frames),) or alignment.dtype != np.dtype("<U64") or alignment.shape != (len(frames),) or not len(frames):
        raise AlignmentError("source NPZ dtype/shape is invalid")
    return frames, labels, alignment
def _action_object(name: str) -> dict[str, str]:
    if name == "wait":
        return {"action_type": "wait", "target": "none", "direction": "none"}
    if name in {"forward", "backward"}:
        return {"action_type": "move", "target": "none", "direction": name}
    return {"action_type": "attack", "target": {"attack_hero": "enemy_hero", "attack_tower": "enemy_tower", "attack_crystal": "enemy_crystal"}[name], "direction": "none"}
def _source_bundle(source_metadata_path: Path, source_dataset_path: Path, manifest_path: Path, shard_paths: Sequence[Path], config_path: Path, source_model_path: Path) -> tuple[SourceDataset, dict[str, torch.Tensor], dict[str, str], str]:
    manifest, config = load_v5_manifest(manifest_path, shard_paths), load_v5_training_config(config_path)
    payload = _object(_load_json(source_metadata_path), {"schema_version", "manifest_sha256", "config_sha256", "dataset_path", "dataset_sha256", "renderer_id", "renderer_sha256", "teacher_id", "teacher_sha256", "action_types", "action_schema_sha256", "rows", "source_metadata_sha256"}, "source metadata")
    metadata_hash = _self_hash(payload, "source_metadata_sha256")
    source_bytes = _read_regular(source_dataset_path, ".npz")
    fixed = (payload["schema_version"] == SOURCE_SCHEMA and payload["manifest_sha256"] == manifest.manifest_sha256 and payload["config_sha256"] == config["config_sha256"] and payload["dataset_path"] == source_dataset_path.name and payload["dataset_sha256"] == _sha(source_bytes) and payload["renderer_id"] == RENDERER_SPEC["id"] and payload["renderer_sha256"] == source_renderer_hash() and payload["teacher_id"] == "causal-source-teacher-v1" and payload["teacher_sha256"] == causal_source_teacher_hash() and payload["action_types"] == list(ACTION_TYPES) and payload["action_schema_sha256"] == action_schema_hash())
    if not fixed or not isinstance(payload["rows"], list):
        raise AlignmentError("source provenance binding is invalid")
    frames, labels, alignment = _source_npz(source_bytes)
    if len(payload["rows"]) != len(frames) or len(set(map(str, alignment))) != len(alignment):
        raise AlignmentError("source rows/alignment are not exact and unique")
    actual = {str(value): index for index, value in enumerate(alignment)}
    seen: set[str] = set()
    for value in payload["rows"]:
        row = _object(value, {"alignment_hash", "observation", "legal_actions", "class_id"}, "source row")
        alignment_hash, observation, legal, class_id = row["alignment_hash"], row["observation"], row["legal_actions"], row["class_id"]
        if not isinstance(alignment_hash, str) or alignment_hash not in actual or alignment_hash in seen or type(observation) is not dict or set(cast(dict[object, object], observation)) != set(_PUBLIC_KEYS) or not isinstance(legal, list) or not legal or len(set(cast(list[object], legal))) != len(legal) or any(name not in ACTION_TYPES for name in legal) or isinstance(class_id, bool) or not isinstance(class_id, int) or not 0 <= class_id < 6:
            raise AlignmentError("invalid causal source row")
        public = cast(dict[str, object], observation)
        if public["side"] not in {"blue", "red"} or any(isinstance(public[key], bool) or not isinstance(public[key], int) for key in _PUBLIC_KEYS if key != "side"):
            raise AlignmentError("source observation is not exact public state")
        index = actual[alignment_hash]
        selected = CausalSourceTeacher().select(public, tuple(_action_object(cast(str, name)) for name in legal))
        if ACTION_TYPES[class_id] != _action_name(selected) or labels[index] != class_id or not np.array_equal(frames[index], source_render_128_rgb(public)) or _sha(_json([_sha(frames[index].tobytes()), class_id]).encode()) != alignment_hash:
            raise AlignmentError("source frame/teacher/class alignment mismatch")
        seen.add(alignment_hash)
    if seen != set(actual):
        raise AlignmentError("source metadata does not cover dataset")
    state, model_meta, model_sha = _load_model(source_model_path, "v5_causal_source_teacher")
    expected = {"manifest_sha256": manifest.manifest_sha256, "config_sha256": cast(str, config["config_sha256"]), "renderer_sha256": source_renderer_hash(), "teacher_sha256": causal_source_teacher_hash(), "action_schema_sha256": action_schema_hash(), "source_dataset_sha256": cast(str, payload["dataset_sha256"]), "source_metadata_sha256": metadata_hash}
    if any(model_meta[key] != value for key, value in expected.items()):
        raise AlignmentError("source model metadata binding mismatch")
    data = SourceDataset(frames, labels)
    _validate_source(data)
    return data, state, model_meta, model_sha
def load_v5_source_dataset(source_metadata_path: Path, source_dataset_path: Path, manifest_path: Path, shard_paths: Sequence[Path], config_path: Path, source_model_path: Path) -> SourceDataset:
    return _source_bundle(source_metadata_path, source_dataset_path, manifest_path, shard_paths, config_path, source_model_path)[0]
def _adapted_model(state: Mapping[str, torch.Tensor]) -> ResNet18SimSiam:
    source_state = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.") and key not in {"encoder.fc.weight", "encoder.fc.bias"}}
    source_state["fc.weight"], source_state["fc.bias"] = state["classifier.weight"], state["classifier.bias"]
    model = ResNet18SimSiam(source_state)
    model.load_state_dict(state, strict=True)
    return model
def train_shallow_simsiam(
    *,
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
    target_shards: Sequence[Path],
    config_path: Path,
    source_checkpoint: Path,
    output_checkpoint: Path,
    device: str = "cpu",
    seed: int = 0,
) -> TrainingResult:
    source, source_state, source_meta, source_sha = _source_bundle(source_metadata_path, source_dataset_path, manifest_path, target_shards, config_path, source_checkpoint)
    manifest, config = load_v5_manifest(manifest_path, target_shards), load_v5_training_config(config_path)
    target_frames = [shard["frames"][shard["split"] == "train"] for shard in manifest.shards]
    if not target_frames or not sum(len(value) for value in target_frames):
        raise AlignmentError("target training split is empty")
    target = np.concatenate(target_frames)
    batch_size, epochs = cast(int, config["batch_size"]), cast(int, config["epochs"])
    if batch_size < 2 or batch_size % 2:
        raise AlignmentError("50/50 batches require a positive even batch size")
    torch.manual_seed(seed)
    run_device = torch.device(device)
    model = ResNet18SimSiam(source_state).to(run_device)
    model.train()
    model.freeze_batch_norm()
    source_x, target_x = _tensor(source.frames, run_device), _tensor(target, run_device)
    source_y = torch.from_numpy(source.labels).to(run_device)
    half = batch_size // 2
    steps_per_epoch = max(math.ceil(len(source_x) / half), math.ceil(len(target_x) / half))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cast(float, config["learning_rate"]), weight_decay=cast(float, config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs * steps_per_epoch))
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            source_index = torch.randint(len(source_x), (half,), device=run_device)
            target_index = torch.randint(len(target_x), (half,), device=run_device)
            source_logits, _, _ = model(_augment(source_x[source_index]))
            first, second = _augment(target_x[target_index]), _augment(target_x[target_index])
            _, z1, p1 = model(first)
            _, z2, p2 = model(second)
            loss = F.cross_entropy(source_logits, source_y[source_index]) + 0.5 * (_siam_loss(p1, z2) + _siam_loss(p2, z1))
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            scheduler.step()
            model.freeze_batch_norm()
    metrics = _collapse_metrics(model, target_x[: min(len(target_x), 64)])
    collapse_hash = _sha(_json(metrics).encode())
    metadata = {"schema_version": MODEL_SCHEMA, "role": "v5_simsiam_adapted", "manifest_sha256": manifest.manifest_sha256, "config_sha256": cast(str, config["config_sha256"]), "renderer_sha256": source_renderer_hash(), "teacher_sha256": causal_source_teacher_hash(), "action_schema_sha256": action_schema_hash(), "source_model_sha256": source_sha, "source_dataset_sha256": source_meta["source_dataset_sha256"], "source_metadata_sha256": source_meta["source_metadata_sha256"], "collapse_metrics_sha256": collapse_hash}
    output_bytes = _model_bytes(model.state_dict(), metadata)
    _write_exclusive(output_checkpoint, output_bytes)
    metrics.update(
        {
            "cpu_smoke": device == "cpu",
            "training_steps": epochs * steps_per_epoch,
            "disposition": "NON_PROMOTING_FRAMEWORK_ONLY",
        }
    )
    return TrainingResult(
        output_checkpoint,
        _sha(output_bytes),
        cast(str, config["config_sha256"]),
        metrics,
        False,
    )
_PSEUDO_FIELDS = ("frames", "class_id", "session_hash", "timestamp_ms", "alignment_hash", "evidence_hash", "manifest_sha256", "predictions_sha256", "source_model_sha256", "adapted_model_sha256", "config_sha256")
def _target_index(manifest: V5Manifest) -> dict[tuple[str, int], tuple[np.ndarray, str, str]]:
    result = {}
    for shard in manifest.shards:
        for frame, session, timestamp, alignment, split in zip(shard["frames"], shard["session_hash"], shard["timestamp_ms"], shard["alignment_hash"], shard["split"], strict=True):
            result[(str(session), int(timestamp))] = (frame, str(alignment), str(split))
    return result
def _prediction_evidence(predictions_path: Path, source_metadata_path: Path, source_dataset_path: Path, manifest_path: Path, target_shards: Sequence[Path], config_path: Path, source_model_path: Path, adapted_model_path: Path) -> tuple[dict[str, np.ndarray], PseudoFilterReport]:
    _, _, source_meta, source_sha = _source_bundle(source_metadata_path, source_dataset_path, manifest_path, target_shards, config_path, source_model_path)
    manifest, config = load_v5_manifest(manifest_path, target_shards), load_v5_training_config(config_path)
    _, adapted_meta, adapted_sha = _load_model(adapted_model_path, "v5_simsiam_adapted")
    expected_adapted = {"manifest_sha256": manifest.manifest_sha256, "config_sha256": cast(str, config["config_sha256"]), "source_model_sha256": source_sha, "source_dataset_sha256": source_meta["source_dataset_sha256"], "source_metadata_sha256": source_meta["source_metadata_sha256"], "renderer_sha256": source_renderer_hash(), "teacher_sha256": causal_source_teacher_hash(), "action_schema_sha256": action_schema_hash()}
    if any(adapted_meta[key] != value for key, value in expected_adapted.items()):
        raise AlignmentError("adapted model binding mismatch")
    payload = _object(_load_json(predictions_path), {"schema_version", "manifest_sha256", "source_model_sha256", "adapted_model_sha256", "config_sha256", "rows", "predictions_sha256"}, "prediction evidence")
    predictions_hash = _self_hash(payload, "predictions_sha256")
    if payload["schema_version"] != PREDICTION_SCHEMA or payload["manifest_sha256"] != manifest.manifest_sha256 or payload["source_model_sha256"] != source_sha or payload["adapted_model_sha256"] != adapted_sha or payload["config_sha256"] != config["config_sha256"] or not isinstance(payload["rows"], list):
        raise AlignmentError("prediction evidence binding mismatch")
    index, candidates, raw_groups = _target_index(manifest), [], defaultdict(list)
    delta = {"t-100": -100, "t": 0, "t+100": 100}
    fields = {"session_hash", "anchor_timestamp_ms", "frame_timestamp_ms", "frame_alignment_hash", "model_role", "view_id", "probs", "ood_score", "black_control_ok", "constant_control_ok", "cut"}
    for value in payload["rows"]:
        row = _object(value, fields, "prediction row")
        session, anchor, frame_time, alignment, role, view, probs = row["session_hash"], row["anchor_timestamp_ms"], row["frame_timestamp_ms"], row["frame_alignment_hash"], row["model_role"], row["view_id"], row["probs"]
        if not isinstance(session, str) or not _valid_sha(session) or isinstance(anchor, bool) or not isinstance(anchor, int) or isinstance(frame_time, bool) or not isinstance(frame_time, int) or not isinstance(alignment, str) or role not in {"source", "student"} or view not in VIEW_KEYS or not isinstance(probs, list) or row["black_control_ok"] not in {True, False} or type(row["black_control_ok"]) is not bool or type(row["constant_control_ok"]) is not bool or type(row["cut"]) is not bool:
            raise AlignmentError("invalid prediction row")
        target = index.get((session, frame_time))
        if frame_time != anchor + delta[view] or target is None or target[1] != alignment or target[2] != "train":
            raise AlignmentError("prediction row does not join target/train shard")
        probability_tuple = tuple(_number(item, low=0.0, high=1.0) for item in probs)
        candidate = CandidatePrediction(session, anchor, role, view, probability_tuple, _number(row["ood_score"], low=0.0), row["black_control_ok"], row["constant_control_ok"], row["cut"])
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
        evidence.append(_sha(_json(sorted(raw_groups[(label.session_id, label.timestamp_ms)], key=lambda row: (cast(str, row["model_role"]), cast(str, row["view_id"])))).encode()))
    frames = np.stack(selected_frames) if selected_frames else np.empty((0, 128, 128, 3), dtype=np.uint8)
    arrays = {"frames": frames.astype(np.uint8), "class_id": np.asarray(labels, dtype=np.int64), "session_hash": np.asarray(sessions, dtype="<U64"), "timestamp_ms": np.asarray(timestamps, dtype=np.int64), "alignment_hash": np.asarray(alignments, dtype="<U64"), "evidence_hash": np.asarray(evidence, dtype="<U64"), "manifest_sha256": np.asarray(manifest.manifest_sha256, dtype="<U64"), "predictions_sha256": np.asarray(predictions_hash, dtype="<U64"), "source_model_sha256": np.asarray(source_sha, dtype="<U64"), "adapted_model_sha256": np.asarray(adapted_sha, dtype="<U64"), "config_sha256": np.asarray(config["config_sha256"], dtype="<U64")}
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
    if any(actual[key].dtype != expected[key].dtype or actual[key].shape != expected[key].shape or not np.array_equal(actual[key], expected[key]) for key in _PSEUDO_FIELDS):
        raise AlignmentError("pseudo artifact differs from re-filtered path evidence")
    return AcceptedPseudoDataset(actual["frames"], actual["class_id"], path, _sha(data), str(actual["manifest_sha256"].item()), str(actual["predictions_sha256"].item()))
def materialize_v5_pseudo(*, predictions_path: Path, source_metadata_path: Path, source_dataset_path: Path, manifest_path: Path, target_shards: Sequence[Path], config_path: Path, source_model_path: Path, adapted_model_path: Path, output_path: Path) -> tuple[AcceptedPseudoDataset, PseudoFilterReport]:
    arrays, report = _prediction_evidence(predictions_path, source_metadata_path, source_dataset_path, manifest_path, target_shards, config_path, source_model_path, adapted_model_path)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # type: ignore[arg-type]
    _write_exclusive(output_path, buffer.getvalue())
    return _pseudo_artifact(output_path, arrays), report
def load_accepted_pseudo_artifact(path: Path, *, predictions_path: Path, source_metadata_path: Path, source_dataset_path: Path, manifest_path: Path, target_shards: Sequence[Path], config_path: Path, source_model_path: Path, adapted_model_path: Path) -> AcceptedPseudoDataset:
    arrays, _ = _prediction_evidence(predictions_path, source_metadata_path, source_dataset_path, manifest_path, target_shards, config_path, source_model_path, adapted_model_path)
    return _pseudo_artifact(path, arrays)
def run_mean_teacher_round(
    *,
    source_metadata_path: Path,
    source_dataset_path: Path,
    manifest_path: Path,
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
    if os.path.lexists(ema_checkpoint) or os.path.lexists(round_ledger):
        raise AlignmentError("Mean Teacher EMA/round ledger already exists")
    source, _, source_meta, source_sha = _source_bundle(source_metadata_path, source_dataset_path, manifest_path, target_shards, config_path, source_model_path)
    manifest, config = load_v5_manifest(manifest_path, target_shards), load_v5_training_config(config_path)
    pseudo = load_accepted_pseudo_artifact(pseudo_path, predictions_path=predictions_path, source_metadata_path=source_metadata_path, source_dataset_path=source_dataset_path, manifest_path=manifest_path, target_shards=target_shards, config_path=config_path, source_model_path=source_model_path, adapted_model_path=adapted_checkpoint)
    if not len(pseudo.frames):
        raise AlignmentError("Mean Teacher requires accepted pseudo rows")
    state, adapted_meta, adapted_sha = _load_model(adapted_checkpoint, "v5_simsiam_adapted")
    expected = {"manifest_sha256": manifest.manifest_sha256, "config_sha256": cast(str, config["config_sha256"]), "source_model_sha256": source_sha, "source_dataset_sha256": source_meta["source_dataset_sha256"], "source_metadata_sha256": source_meta["source_metadata_sha256"]}
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
    source_x, source_y = _tensor(source.frames, run_device), torch.from_numpy(source.labels).to(run_device)
    pseudo_x = _tensor(pseudo.frames, run_device)
    pseudo_y = torch.from_numpy(pseudo.labels).to(run_device)
    batch_size, epochs, ema_decay = cast(int, config["batch_size"]), cast(int, config["mean_teacher_epochs"]), cast(float, config["ema_decay"])
    if batch_size < 2 or batch_size % 2:
        raise AlignmentError("Mean Teacher requires an even batch size")
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=float(cast(int | float, config["learning_rate"])), weight_decay=float(cast(int | float, config["weight_decay"])))
    half = max(1, batch_size // 2)
    steps = max(math.ceil(len(source_x) / half), math.ceil(len(pseudo_x) / half))
    for _ in range(epochs):
        for _ in range(steps):
            si = torch.randint(len(source_x), (half,), device=run_device)
            pi = torch.randint(len(pseudo_x), (half,), device=run_device)
            source_logits, _, _ = student(_augment(source_x[si]))
            target_view = _augment(pseudo_x[pi])
            pseudo_logits, _, _ = student(target_view)
            with torch.no_grad():
                teacher_logits, _, _ = teacher(target_view)
            loss = F.cross_entropy(source_logits, source_y[si]) + F.cross_entropy(pseudo_logits, pseudo_y[pi]) + 0.5 * F.kl_div(F.log_softmax(pseudo_logits, dim=1), F.softmax(teacher_logits, dim=1), reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            student.freeze_batch_norm()
            with torch.no_grad():
                for target_parameter, source_parameter in zip(teacher.parameters(), student.parameters(), strict=True):
                    target_parameter.mul_(ema_decay).add_(source_parameter, alpha=1 - ema_decay)
    ema_meta = {"schema_version": MODEL_SCHEMA, "role": "v5_mean_teacher_ema", "manifest_sha256": manifest.manifest_sha256, "config_sha256": cast(str, config["config_sha256"]), "renderer_sha256": source_renderer_hash(), "teacher_sha256": causal_source_teacher_hash(), "action_schema_sha256": action_schema_hash(), "source_model_sha256": source_sha, "source_dataset_sha256": source_meta["source_dataset_sha256"], "source_metadata_sha256": source_meta["source_metadata_sha256"], "adapted_model_sha256": adapted_sha, "pseudo_sha256": pseudo.artifact_sha256, "predictions_sha256": pseudo.predictions_sha256, "collapse_metrics_sha256": adapted_meta["collapse_metrics_sha256"], "round": "1"}
    ema_bytes = _model_bytes(teacher.state_dict(), ema_meta)
    ema_sha = _sha(ema_bytes)
    payload: dict[str, object] = {"schema_version": LEDGER_SCHEMA, "round": 1, "source_model_sha256": source_sha, "source_dataset_sha256": source_meta["source_dataset_sha256"], "source_metadata_sha256": source_meta["source_metadata_sha256"], "adapted_model_sha256": adapted_sha, "pseudo_sha256": pseudo.artifact_sha256, "predictions_sha256": pseudo.predictions_sha256, "manifest_sha256": manifest.manifest_sha256, "config_sha256": config["config_sha256"], "ema_model_sha256": ema_sha, "ema_decay": ema_decay, "weights": {"source_ce": 1.0, "pseudo_ce": 1.0, "kl": 0.5}, "epochs": epochs, "batch_size": batch_size, "collapse_status": COLLAPSE_BLOCK}
    payload["ledger_sha256"] = _sha(_json(payload).encode())
    _write_exclusive(ema_checkpoint, ema_bytes)
    _write_exclusive(round_ledger, (_json(payload) + "\n").encode())
    return payload
def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise AlignmentError("kappa requires paired labels")
    classes = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    expected = sum(left.count(value) * right.count(value) for value in classes) / len(left) ** 2
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)
def _audit_diagnostics(labels: Sequence[AuditLabel], predictions: Sequence[AuditPrediction], regression: SourceRegression, evidence_hash: str) -> ReleaseGate:
    by_clip: dict[str, list[AuditLabel]] = defaultdict(list)
    for row in labels:
        if row.observed_action not in ACTION_TYPES:
            raise AlignmentError("audit action outside exact vocabulary")
        by_clip[row.clip_id].append(row)
    if len(by_clip) != 300 or any(len(rows) != 2 or len({row.reviewer for row in rows}) != 2 for rows in by_clip.values()):
        raise AlignmentError("V5 audit composition is exactly 300 clips x two reviewers")
    prediction_map = {row.clip_id: row for row in predictions}
    if len(prediction_map) != 300 or set(prediction_map) != set(by_clip):
        raise AlignmentError("audit requires one sealed prediction per clip")
    ordered = sorted(by_clip)
    left = [by_clip[key][0].observed_action if by_clip[key][0].validity else "INVALID" for key in ordered]
    right = [by_clip[key][1].observed_action if by_clip[key][1].validity else "INVALID" for key in ordered]
    kappa = _cohen_kappa(left, right)
    valid_consensus: dict[str, str] = {
        key: left[index]
        for index, key in enumerate(ordered)
        if left[index] == right[index] and left[index] != "INVALID"
    }
    invalid_consensus = {key for index, key in enumerate(ordered) if left[index] == right[index] == "INVALID"}
    accepted = [prediction_map[key] for key in valid_consensus if prediction_map[key].accepted]
    correct = [row for row in accepted if row.action == valid_consensus[row.clip_id]]
    overall = len(correct) / max(1, len(accepted))
    coverage = len(accepted) / 300
    ood_false = sum(prediction_map[key].accepted for key in invalid_consensus) / max(1, len(invalid_consensus))
    class_precision = {
        name: sum(row.action == valid_consensus[row.clip_id] for row in accepted if row.action == name)
        / max(1, sum(row.action == name for row in accepted))
        for name in ACTION_TYPES
    }
    baselines = tuple(
        overall
        - sum(row.baselines[index] == valid_consensus[row.clip_id] for row in accepted) / max(1, len(accepted))
        for index in range(2)
    )
    accuracy_drop = regression.accuracy_before - regression.accuracy_after
    if len(regression.recall_before) != 6 or len(regression.recall_after) != 6:
        raise AlignmentError("source regression requires six class recalls")
    recall_drop = max(a - b for a, b in zip(regression.recall_before, regression.recall_after, strict=True))
    return ReleaseGate(False, (), kappa, overall, class_precision, coverage, ood_false, cast(tuple[float, float], baselines), accuracy_drop, recall_drop, evidence_hash, COLLAPSE_BLOCK, None)
def _load_ledger(path: Path) -> dict[str, object]:
    fields = {"schema_version", "round", "source_model_sha256", "source_dataset_sha256", "source_metadata_sha256", "adapted_model_sha256", "pseudo_sha256", "predictions_sha256", "manifest_sha256", "config_sha256", "ema_model_sha256", "ema_decay", "weights", "epochs", "batch_size", "collapse_status", "ledger_sha256"}
    payload = _object(_load_json(path), fields, "Mean Teacher ledger")
    if payload["schema_version"] != LEDGER_SCHEMA or payload["round"] != 1 or payload["ema_decay"] != 0.999 or payload["weights"] != {"source_ce": 1.0, "pseudo_ce": 1.0, "kl": 0.5} or payload["collapse_status"] != COLLAPSE_BLOCK:
        raise AlignmentError("Mean Teacher ledger contract is invalid")
    if any(not isinstance(payload[key], str) or not _valid_sha(cast(str, payload[key])) for key in fields if key.endswith("sha256")):
        raise AlignmentError("Mean Teacher ledger hashes are invalid")
    _self_hash(payload, "ledger_sha256")
    return payload
def path_only_v5_release_gate(*, manifest_path: Path, target_shards: Sequence[Path], sealed_predictions_path: Path, sealed_audit_path: Path, model_path: Path, mean_teacher_ledger_path: Path, config_path: Path, release_path: Path) -> ReleaseGate:
    if os.path.lexists(release_path):
        raise AlignmentError("release path must not already exist")
    manifest, config = load_v5_manifest(manifest_path, target_shards), load_v5_training_config(config_path)
    _, model_meta, model_sha = _load_model(model_path, "v5_mean_teacher_ema")
    ledger = _load_ledger(mean_teacher_ledger_path)
    if ledger["ema_model_sha256"] != model_sha or ledger["manifest_sha256"] != manifest.manifest_sha256 or ledger["config_sha256"] != config["config_sha256"] or any(model_meta[key] != ledger[key] for key in ("manifest_sha256", "config_sha256", "source_model_sha256", "source_dataset_sha256", "source_metadata_sha256", "adapted_model_sha256", "pseudo_sha256", "predictions_sha256")):
        raise AlignmentError("EMA model/ledger/manifest/config binding mismatch")
    sealed = _object(_load_json(sealed_predictions_path), {"schema_version", "manifest_sha256", "model_sha256", "config_sha256", "selection_sha256", "source_regression", "rows", "predictions_sha256"}, "sealed predictions")
    predictions_hash = _self_hash(sealed, "predictions_sha256")
    if sealed["schema_version"] != "hok-agent-v5-sealed-predictions-v1" or sealed["manifest_sha256"] != manifest.manifest_sha256 or sealed["model_sha256"] != model_sha or sealed["config_sha256"] != config["config_sha256"] or not _is_sha(sealed["selection_sha256"]) or not isinstance(sealed["rows"], list):
        raise AlignmentError("sealed prediction binding is invalid")
    regression_row = _object(sealed["source_regression"], {"accuracy_before", "accuracy_after", "recall_before", "recall_after"}, "source regression")
    before, after = regression_row["recall_before"], regression_row["recall_after"]
    if not isinstance(before, list) or not isinstance(after, list) or len(before) != 6 or len(after) != 6:
        raise AlignmentError("source regression requires six recalls")
    regression = SourceRegression(_number(regression_row["accuracy_before"], low=0.0, high=1.0), _number(regression_row["accuracy_after"], low=0.0, high=1.0), tuple(_number(value, low=0.0, high=1.0) for value in before), tuple(_number(value, low=0.0, high=1.0) for value in after))
    index, predictions, identities, clips, sealed_identities = _target_index(manifest), [], {}, set(), set()
    prediction_fields = {"clip_id", "session_hash", "timestamp_ms", "alignment_hash", "action", "accepted", "confidence", "baselines"}
    for value in sealed["rows"]:
        row = _object(value, prediction_fields, "sealed prediction row")
        clip, session, timestamp, alignment, action, accepted, baselines = row["clip_id"], row["session_hash"], row["timestamp_ms"], row["alignment_hash"], row["action"], row["accepted"], row["baselines"]
        if not isinstance(clip, str) or not clip or clip in clips or not isinstance(session, str) or isinstance(timestamp, bool) or not isinstance(timestamp, int) or not isinstance(alignment, str) or action not in ACTION_TYPES or type(accepted) is not bool or not isinstance(baselines, list) or len(baselines) != 2 or any(item not in ACTION_TYPES for item in baselines):
            raise AlignmentError("invalid sealed prediction row")
        target = index.get((session, timestamp))
        identity = (session, timestamp, alignment)
        if target is None or target[1] != alignment or target[2] != "test" or identity in sealed_identities:
            raise AlignmentError("sealed prediction must join target/test")
        confidence = _number(row["confidence"], low=0.0, high=1.0)
        clips.add(clip)
        sealed_identities.add(identity)
        identities[clip] = identity
        predictions.append(AuditPrediction(clip, action, accepted, confidence, (cast(str, baselines[0]), cast(str, baselines[1]))))
    if len(predictions) != 300:
        raise AlignmentError("sealed predictions require exactly 300 unique clips")
    audit = _object(_load_json(sealed_audit_path), {"schema_version", "manifest_sha256", "selection_sha256", "rows", "audit_sha256"}, "sealed audit")
    audit_hash = _self_hash(audit, "audit_sha256")
    if audit["schema_version"] != AUDIT_SCHEMA or audit["manifest_sha256"] != manifest.manifest_sha256 or audit["selection_sha256"] != sealed["selection_sha256"] or not isinstance(audit["rows"], list):
        raise AlignmentError("sealed audit binding is invalid")
    labels: list[AuditLabel] = []
    per_clip: dict[str, set[str]] = defaultdict(set)
    reviewers: set[str] = set()
    audit_fields = {"clip_id", "session_hash", "timestamp_ms", "alignment_hash", "reviewer", "observed_action", "validity"}
    for value in audit["rows"]:
        row = _object(value, audit_fields, "audit row")
        clip, reviewer, action, validity = row["clip_id"], row["reviewer"], row["observed_action"], row["validity"]
        audit_identity = (row["session_hash"], row["timestamp_ms"], row["alignment_hash"])
        if not isinstance(clip, str) or clip not in identities or audit_identity != identities[clip] or not isinstance(reviewer, str) or not reviewer or reviewer in per_clip[clip] or action not in ACTION_TYPES or type(validity) is not bool:
            raise AlignmentError("invalid or unbound audit row")
        per_clip[clip].add(reviewer)
        reviewers.add(reviewer)
        labels.append(AuditLabel(clip, reviewer, action, validity))
    if len(labels) != 600 or set(per_clip) != clips or len(reviewers) != 2 or any(values != reviewers for values in per_clip.values()):
        raise AlignmentError("V5 audit is exactly 300 clips x the same two reviewers")
    evidence_hash = _sha(_json([manifest.manifest_sha256, model_sha, predictions_hash, audit_hash, ledger["ledger_sha256"]]).encode())
    gate = _audit_diagnostics(labels, predictions, regression, evidence_hash)
    # No authority-defined numeric collapse threshold exists. Diagnostics are retained,
    # but formal promotion and release writing remain fail-closed.
    return gate
def _release_payload(data: bytes) -> dict[str, object]:
    payload = _load_json_bytes(data)
    fields = {"schema_version", "model_sha256", "alignment_sha256", "audit_sha256", "config_sha256", "overall_pass", "allowed_classes", "class_thresholds", "thresholds", "thresholds_hash", "release_sha256"}
    _object(payload, fields, "release")
    if payload["schema_version"] != RELEASE_SCHEMA or payload["overall_pass"] is not True:
        raise AlignmentError("release schema/gate is invalid")
    if any(not _is_sha(payload[key]) for key in ("model_sha256", "alignment_sha256", "audit_sha256", "config_sha256", "thresholds_hash", "release_sha256")):
        raise AlignmentError("release hashes are invalid")
    allowed = payload["allowed_classes"]
    class_thresholds = payload["class_thresholds"]
    if not isinstance(allowed, list) or not allowed or any(value not in ACTION_TYPES for value in allowed) or len(set(allowed)) != len(allowed):
        raise AlignmentError("release classes are invalid")
    if not isinstance(class_thresholds, dict) or set(class_thresholds) != set(allowed) or any(isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)) or not 0.75 <= float(value) <= 1.0 for value in class_thresholds.values()):
        raise AlignmentError("release class thresholds are invalid")
    frozen = {"kappa": 0.70, "overall_precision": 0.85, "per_class_precision": 0.75, "coverage": 0.30, "ood_false_accept": 0.05, "baseline_delta": 0.05, "source_accuracy_drop": 0.02, "source_recall_drop": 0.05}
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
    if payload["model_sha256"] != model_sha or payload["alignment_sha256"] != metadata["manifest_sha256"] or payload["config_sha256"] != metadata["config_sha256"]:
        raise AlignmentError("release does not bind actual model bytes/metadata")
    thresholds = cast(dict[str, int | float], payload["class_thresholds"])
    bound = BoundV5Release(_sha(release_bytes), model_sha, tuple(cast(list[str], payload["allowed_classes"])), {key: float(value) for key, value in thresholds.items()})
    del bound
    raise AlignmentError(COLLAPSE_BLOCK)
def launch_offline_audit_ui(
    clips: Sequence[AuditClip], *, reviewer: str, output_path: Path, enable_gui: bool = True
) -> dict[str, object]:
    """Blind offline Pillow/Tk entry point.  No prediction is accepted or displayed."""
    if not reviewer or len({clip.clip_id for clip in clips}) != len(clips):
        raise AlignmentError("reviewer and unique clips are required")
    summary: dict[str, object] = {"clips": len(clips), "reviewer": reviewer, "blind": True}
    if not enable_gui:
        return summary
    import tkinter as tk
    root = tk.Tk()
    root.title("V5 blind offline audit")
    label = tk.Label(root)
    label.pack()
    index = 0
    photo: ImageTk.PhotoImage | None = None
    def record(action: str, validity: bool) -> None:
        nonlocal index, photo
        clip = clips[index]
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(_json(asdict(AuditLabel(clip.clip_id, reviewer, action, validity))) + "\n")
        index += 1
        if index == len(clips):
            root.destroy()
            return
        photo = ImageTk.PhotoImage(Image.open(clips[index].frame_path).convert("RGB"))
        label.configure(image=photo)
    def valid_callback(action: str) -> Any:
        def callback() -> None:
            record(action, True)
        return callback
    def invalid_callback() -> None:
        record("wait", False)
    if clips:
        photo = ImageTk.PhotoImage(Image.open(clips[0].frame_path).convert("RGB"))
        label.configure(image=photo)
    for action in ACTION_TYPES:
        tk.Button(root, text=action, command=valid_callback(action)).pack(side="left")
    tk.Button(root, text="invalid", command=invalid_callback).pack(side="left")
    root.mainloop()
    return summary
