# ruff: noqa: E501, E302, E305
"""Strict V5 contracts; CPU paths are explicitly non-promoting contract smokes."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageTk
from safetensors.torch import load_file, save_file
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
class AlignmentError(ValueError):
    """Malformed or incomplete V5 evidence."""
@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    family_id: str
    parent_id: str | None = None
    near_duplicates: tuple[str, ...] = ()
@dataclass(frozen=True)
class ManifestSplit:
    session_id: str
    split: str
    family_id: str
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
    release_eligible: bool
@dataclass(frozen=True)
class SourceDataset:
    frames: np.ndarray
    labels: np.ndarray
@dataclass(frozen=True)
class UnlabeledTargetDataset:
    frames: np.ndarray
@dataclass(frozen=True)
class AcceptedPseudoDataset:
    frames: np.ndarray
    labels: np.ndarray
    artifact_path: Path
    artifact_sha256: str
    alignment_hash: str
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
    release_path: Path | None
def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
def _integer(value: object) -> int:
    return int(cast(Any, value))
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())
def _valid_sha(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
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
def split_session_manifest(records: Sequence[SessionRecord]) -> tuple[ManifestSplit, ...]:
    if len(records) < 12:
        raise AlignmentError("V5 requires at least 12 sessions")
    ids = [row.session_id for row in records]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise AlignmentError("session ids must be unique and non-empty")
    groups = _Groups(ids)
    by_family: dict[str, str] = {}
    known = set(ids)
    for row in records:
        if not row.family_id:
            raise AlignmentError("family id is required")
        if row.family_id in by_family:
            groups.union(row.session_id, by_family[row.family_id])
        else:
            by_family[row.family_id] = row.session_id
        if row.parent_id in known:
            groups.union(row.session_id, row.parent_id)
        for duplicate in row.near_duplicates:
            if duplicate in known:
                groups.union(row.session_id, duplicate)
    components: dict[str, list[str]] = defaultdict(list)
    for session_id in ids:
        components[groups.find(session_id)].append(session_id)
    chunks = sorted((tuple(sorted(values)) for values in components.values()), key=lambda x: x[0])
    minima = (8, 2, 2)
    states: dict[tuple[int, int, int], tuple[str, ...]] = {(0, 0, 0): ()}
    for chunk in chunks:
        updated: dict[tuple[int, int, int], tuple[str, ...]] = {}
        for counts, choices in states.items():
            for index, split in enumerate(("train", "dev", "test")):
                next_counts = list(counts)
                next_counts[index] = min(minima[index], next_counts[index] + len(chunk))
                state = (next_counts[0], next_counts[1], next_counts[2])
                updated.setdefault(state, choices + (split,))
        states = updated
    solution = states.get(minima)
    if solution is None:
        raise AlignmentError("linked components cannot satisfy 8/2/2 split")
    assignment = {sid: split for chunk, split in zip(chunks, solution, strict=True) for sid in chunk}
    return tuple(
        ManifestSplit(row.session_id, assignment[row.session_id], row.family_id)
        for row in sorted(records, key=lambda value: value.session_id)
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
        path = output_dir / f"alignment-{shard:04d}.npz"
        np.savez(path, **payload)  # type: ignore[arg-type]
        paths.append(path)
    return tuple(paths)
def load_npz_shards(paths: Sequence[Path]) -> tuple[dict[str, np.ndarray], ...]:
    result: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(sorted(archive.files)) != tuple(sorted(NPZ_FIELDS)):
                raise AlignmentError("NPZ fields are not exact")
            payload = {key: archive[key] for key in NPZ_FIELDS}
        count = len(payload["frames"])
        for key in NPZ_FIELDS:
            if payload[key].dtype != NPZ_DTYPES[key] or len(payload[key]) != count:
                raise AlignmentError(f"invalid dtype/length for {key}")
        if payload["frames"].shape != (count, 128, 128, 3):
            raise AlignmentError("invalid frames shape")
        if payload["time_base"].shape != (count, 2):
            raise AlignmentError("invalid time_base shape")
        for index in range(count):
            split, source = str(payload["split"][index]), str(payload["source"][index])
            session, frame_hash = str(payload["session_hash"][index]), str(payload["frame_hash"][index])
            if split not in SPLITS or source not in SOURCES or not _valid_sha(session) or int(payload["time_base"][index][1]) <= 0:
                raise AlignmentError("invalid NPZ enum/hash")
            if _sha(payload["frames"][index].tobytes()) != frame_hash:
                raise AlignmentError("frame hash mismatch")
            material = [
                session,
                int(payload["timestamp_ms"][index]),
                int(payload["pts"][index]),
                tuple(int(x) for x in payload["time_base"][index]),
                frame_hash,
                split,
                source,
            ]
            if _sha(_json(material).encode()) != str(payload["alignment_hash"][index]):
                raise AlignmentError("alignment hash mismatch")
        result.append(payload)
    return tuple(result)
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
def collapse_pseudo_metrics(labels: Sequence[PseudoLabel]) -> dict[str, object]:
    counts = {name: 0 for name in ACTION_TYPES}
    for row in labels:
        counts[ACTION_TYPES[row.class_id]] += 1
    return {"count": len(labels), "by_class": counts, "release_eligible": len(labels) >= 200}
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
def build_training_config(
    *, batch_size: int = 256, learning_rate: float = 3e-4, weight_decay: float = 1e-4, epochs: int = 50
) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": "torchvision-resnet18-simsiam-c6",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "schedule": "cosine",
        "source_target_ratio": "50/50",
        "trainable": "stem+layer1+layer2+projector+predictor",
        "frozen": "fc+layer3+layer4+all_batchnorm",
        "audit_input": False,
    }
    config["hash"] = _sha(_json(config).encode())
    return config
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
    }
def _verified_state(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    if not _valid_sha(expected_sha256) or _file_sha(path) != expected_sha256:
        raise AlignmentError("checkpoint SHA-256 mismatch")
    return load_file(path, device="cpu")
def train_shallow_simsiam(
    source: SourceDataset,
    target: UnlabeledTargetDataset,
    *,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
    output_checkpoint: Path,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
) -> TrainingResult:
    _validate_source(source)
    if type(target) is not UnlabeledTargetDataset or target.frames.dtype != np.uint8 or target.frames.ndim != 4 or not len(target.frames):
        raise AlignmentError("target must be a typed non-empty unlabeled uint8 NHWC dataset")
    if batch_size < 2 or batch_size % 2:
        raise AlignmentError("50/50 batches require a positive even batch size")
    torch.manual_seed(seed)
    run_device = torch.device(device)
    model = ResNet18SimSiam(_verified_state(source_checkpoint, source_checkpoint_sha256)).to(run_device)
    model.train()
    model.freeze_batch_norm()
    source_x, target_x = _tensor(source.frames, run_device), _tensor(target.frames, run_device)
    source_y = torch.from_numpy(source.labels).to(run_device)
    half = batch_size // 2
    steps_per_epoch = max(math.ceil(len(source_x) / half), math.ceil(len(target_x) / half))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=weight_decay)
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
    config = build_training_config(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, output_checkpoint, metadata={"source_sha256": source_checkpoint_sha256, "config_hash": str(config["hash"])})
    metrics.update(
        {
            "cpu_smoke": device == "cpu",
            "training_steps": epochs * steps_per_epoch,
            "disposition": "NON_PROMOTING_FRAMEWORK_ONLY",
        }
    )
    return TrainingResult(
        output_checkpoint,
        _file_sha(output_checkpoint),
        str(config["hash"]),
        metrics,
        False,
    )
def write_accepted_pseudo_artifact(
    frames: np.ndarray, labels: np.ndarray, alignment_hash: str, path: Path
) -> AcceptedPseudoDataset:
    if frames.dtype != np.uint8 or frames.ndim != 4 or labels.dtype != np.int64 or labels.shape != (len(frames),):
        raise AlignmentError("invalid accepted pseudo dataset")
    if not _valid_sha(alignment_hash) or (labels < 0).any() or (labels >= 6).any():
        raise AlignmentError("invalid pseudo class/alignment hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, frames=frames, class_id=labels, alignment_hash=np.asarray(alignment_hash, dtype="<U64"))
    return AcceptedPseudoDataset(frames, labels, path, _file_sha(path), alignment_hash)
def load_accepted_pseudo_artifact(path: Path, expected_sha256: str) -> AcceptedPseudoDataset:
    if not _valid_sha(expected_sha256) or _file_sha(path) != expected_sha256:
        raise AlignmentError("pseudo artifact SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"frames", "class_id", "alignment_hash"}:
            raise AlignmentError("pseudo artifact fields are not exact")
        frames, labels, alignment = archive["frames"], archive["class_id"], archive["alignment_hash"]
    value = str(alignment.item())
    if frames.dtype != np.uint8 or labels.dtype != np.int64 or alignment.dtype != np.dtype("<U64"):
        raise AlignmentError("pseudo artifact dtypes are not exact")
    return AcceptedPseudoDataset(frames, labels, path, expected_sha256, value)
def _adapted_model(state: Mapping[str, torch.Tensor]) -> ResNet18SimSiam:
    source_state = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.") and key != "encoder.fc.weight" and key != "encoder.fc.bias"}
    source_state["fc.weight"] = state["classifier.weight"]
    source_state["fc.bias"] = state["classifier.bias"]
    model = ResNet18SimSiam(source_state)
    model.load_state_dict(state, strict=True)
    return model
def run_mean_teacher_round(
    source: SourceDataset,
    pseudo: AcceptedPseudoDataset,
    *,
    adapted_checkpoint: Path,
    adapted_checkpoint_sha256: str,
    round_ledger: Path,
    epochs: int = 20,
    batch_size: int = 128,
    ema_decay: float = 0.999,
    device: str = "cpu",
) -> dict[str, object]:
    _validate_source(source)
    if type(pseudo) is not AcceptedPseudoDataset:
        raise AlignmentError("Mean Teacher requires a hash-bound accepted pseudo artifact")
    verified_pseudo = load_accepted_pseudo_artifact(pseudo.artifact_path, pseudo.artifact_sha256)
    if verified_pseudo.alignment_hash != pseudo.alignment_hash:
        raise AlignmentError("pseudo artifact alignment hash mismatch")
    if round_ledger.exists():
        raise AlignmentError("Mean Teacher round ledger already exists")
    if ema_decay != 0.999:
        raise AlignmentError("EMA is frozen at 0.999")
    state = _verified_state(adapted_checkpoint, adapted_checkpoint_sha256)
    run_device = torch.device(device)
    student = _adapted_model(state).to(run_device)
    teacher = copy.deepcopy(student).to(run_device).eval()
    student.train()
    student.freeze_batch_norm()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    source_x, source_y = _tensor(source.frames, run_device), torch.from_numpy(source.labels).to(run_device)
    pseudo_x = _tensor(verified_pseudo.frames, run_device)
    pseudo_y = torch.from_numpy(verified_pseudo.labels).to(run_device)
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-4)
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
    payload = {
        "round": 1,
        "adapted_checkpoint_sha256": adapted_checkpoint_sha256,
        "pseudo_artifact_sha256": verified_pseudo.artifact_sha256,
        "alignment_hash": verified_pseudo.alignment_hash,
        "weights": {"source_ce": 1.0, "pseudo_ce": 1.0, "kl": 0.5},
        "ema": ema_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "promoting": False,
        "disposition": "NON_PROMOTING_FRAMEWORK_ONLY",
    }
    round_ledger.parent.mkdir(parents=True, exist_ok=True)
    round_ledger.write_text(_json(payload) + "\n", encoding="utf-8")
    return payload
def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise AlignmentError("kappa requires paired labels")
    classes = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    expected = sum(left.count(value) * right.count(value) for value in classes) / len(left) ** 2
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)
def audit_evidence_hash(
    labels: Sequence[AuditLabel], predictions: Sequence[AuditPrediction], regression: SourceRegression
) -> str:
    payload = {
        "labels": [asdict(row) for row in sorted(labels, key=lambda x: (x.clip_id, x.reviewer))],
        "predictions": [asdict(row) for row in sorted(predictions, key=lambda x: x.clip_id)],
        "source_regression": asdict(regression),
    }
    return _sha(_json(payload).encode())
def offline_audit_release_gate(
    labels: Sequence[AuditLabel],
    predictions: Sequence[AuditPrediction],
    regression: SourceRegression,
    *,
    evidence_hash: str,
    model_hash: str,
    alignment_hash: str,
    config_hash: str,
    release_path: Path,
) -> ReleaseGate:
    if any(not _valid_sha(value) for value in (evidence_hash, model_hash, alignment_hash, config_hash)):
        raise AlignmentError("release inputs must be SHA-256 bound")
    if audit_evidence_hash(labels, predictions, regression) != evidence_hash:
        raise AlignmentError("audit evidence hash mismatch")
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
    del release_path
    # This function intentionally computes audit diagnostics only.  Formal release
    # remains disabled until model/config/session/audit artifacts are loaded from
    # paths and independently re-hashed; caller-supplied in-memory rows are not a
    # promotion boundary.
    return ReleaseGate(False, (), kappa, overall, class_precision, coverage, ood_false, cast(tuple[float, float], baselines), accuracy_drop, recall_drop, evidence_hash, None)
def load_release(path: Path) -> dict[str, object]:
    """Load the hash-bound release only; V6 never receives audit labels."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = {"schema_version", "model_sha256", "alignment_sha256", "audit_sha256", "config_sha256", "overall_pass", "allowed_classes", "class_thresholds", "thresholds", "thresholds_hash", "release_sha256"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise AlignmentError("release fields are not exact")
    if payload["schema_version"] != "hok-agent-v5-release-v1" or payload["overall_pass"] is not True:
        raise AlignmentError("release schema/gate is invalid")
    if any(not isinstance(payload[key], str) or not _valid_sha(payload[key]) for key in ("model_sha256", "alignment_sha256", "audit_sha256", "config_sha256", "thresholds_hash", "release_sha256")):
        raise AlignmentError("release hashes are invalid")
    allowed = payload["allowed_classes"]
    class_thresholds = payload["class_thresholds"]
    if not isinstance(allowed, list) or not allowed or any(value not in ACTION_TYPES for value in allowed) or len(set(allowed)) != len(allowed):
        raise AlignmentError("release classes are invalid")
    if not isinstance(class_thresholds, dict) or set(class_thresholds) != set(allowed) or any(not isinstance(value, (int, float)) or value < 0.75 for value in class_thresholds.values()):
        raise AlignmentError("release class thresholds are invalid")
    frozen = {"kappa": 0.70, "overall_precision": 0.85, "per_class_precision": 0.75, "coverage": 0.30, "ood_false_accept": 0.05, "baseline_delta": 0.05, "source_accuracy_drop": 0.02, "source_recall_drop": 0.05}
    if payload["thresholds"] != frozen:
        raise AlignmentError("release thresholds differ from frozen V5 contract")
    if _sha(_json(payload["thresholds"]).encode()) != payload["thresholds_hash"]:
        raise AlignmentError("release thresholds hash mismatch")
    signature = payload.pop("release_sha256")
    if _sha(_json(payload).encode()) != signature:
        raise AlignmentError("release payload hash mismatch")
    payload["release_sha256"] = signature
    return cast(dict[str, object], payload)
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
