from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.rich_arena import (
    ACTION_TYPES,
    CANONICAL_ACTION_TEMPLATES,
    DIRECTIONS,
    MACROS,
    SKILLS,
    TARGETS,
    ArenaConfig,
    FactorizedAction,
    ReplayError,
    RichNullPolicy,
    RichPixelArena,
    RichRandomPolicy,
    RichTeacherPolicy,
    Side,
    canonical_actions,
    ego_action,
    record_rich_trace,
    verify_rich_trace,
)
from hok_agent.rich_renderer import RENDERER_HASH, RENDERER_ID, render

TYPES = ACTION_TYPES
ACTIONS: Final = canonical_actions()
TEMPLATE_COUNT = len(CANONICAL_ACTION_TEMPLATES)
SPLITS: Final = {"fit": 0, "acquisition": 1, "validation": 2, "test": 3}
SPLIT_NAMES = {value: key for key, value in SPLITS.items()}
TRAINING_SEEDS: Final = (0, 1, 2)
TRAINING_CONFIG: Final = {
    "architecture": "resnet18-weights-none-rgb-factorized-v2",
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 50,
    "validation_patience": 8,
    "class_weight": "inverse_sqrt_frequency",
    "augmentation": "deterministic_shift_color",
    "precision": "float32",
}
MODEL_SCHEMA = "rich-pixel-rgb-resnet18-v2"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


DIRECTION_FRAME = "ego-view-red-rotated-180"
ACTION_HASH = _sha(
    _canonical(
        {"actions": [action.to_dict() for action in ACTIONS], "direction_frame": DIRECTION_FRAME}
    ).encode()
)
TRAINING_HASH = _sha(_canonical(TRAINING_CONFIG).encode())
DATA_KEYS: Final = {
    "frames",
    "macro",
    "type",
    "direction",
    "target",
    "skill",
    "template",
    "group_ids",
    "ticks",
    "render_seeds",
    "variants",
    "splits",
    "frame_hashes",
}


class RichPixelError(ValueError):
    pass


@dataclass(frozen=True)
class TraceStep:
    observation: dict[str, object]
    action: FactorizedAction
    template: int
    tick: int


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    side: Side
    group_id: str
    split: int
    render_seeds: tuple[int, ...]
    outcome: str
    completed: bool


@dataclass
class RichPixelData:
    frames: np.ndarray
    macro: np.ndarray
    type: np.ndarray
    direction: np.ndarray
    target: np.ndarray
    skill: np.ndarray
    template: np.ndarray
    group_ids: np.ndarray
    ticks: np.ndarray
    render_seeds: np.ndarray
    variants: np.ndarray
    splits: np.ndarray
    frame_hashes: np.ndarray
    episodes: list[EpisodeSpec]
    config: ArenaConfig


def _template(action: FactorizedAction) -> int:
    for index, (macro, kind, target, direction, skill) in enumerate(CANONICAL_ACTION_TEMPLATES):
        if (
            action.macro == macro
            and action.action_type == kind
            and action.target == target
            and action.skill == skill
            and (direction == "{direction}" or action.direction == direction)
        ):
            return index
    raise RichPixelError(f"action is not canonical: {action}")


def _indices(action: FactorizedAction) -> tuple[int, int, int, int, int]:
    return (
        MACROS.index(action.macro),
        TYPES.index(action.action_type),
        DIRECTIONS.index(action.direction),
        TARGETS.index(action.target),
        SKILLS.index(action.skill),
    )


def _ego_action(action: FactorizedAction, side: Side) -> FactorizedAction:
    return ego_action(action, side)


def _group_splits(groups: Sequence[str], formal: bool) -> dict[str, int]:
    ordered = sorted(set(groups))
    if formal and len(ordered) != 512:
        raise RichPixelError(f"formal collection requires 512 groups; got {len(ordered)}")
    if len(ordered) <= 4:
        return {group: index % 4 for index, group in enumerate(ordered)}
    counts = [round(len(ordered) * 0.56), round(len(ordered) * 0.14), round(len(ordered) * 0.15)]
    counts.append(len(ordered) - sum(counts))
    if min(counts) < 1:
        raise RichPixelError("all four grouped splits must be non-empty")
    result: dict[str, int] = {}
    start = 0
    for split, count in enumerate(counts):
        for group in ordered[start : start + count]:
            result[group] = split
        start += count
    return result


def _render_seeds(group: str, variants: int) -> tuple[int, ...]:
    base = int(group[:15], 16)
    return tuple((base + 1_000_003 * index) % (2**31 - 1) for index in range(variants))


def _teacher_episode(
    seed: int, side: Side
) -> tuple[str, tuple[TraceStep, ...], str, bool]:
    arena = RichPixelArena()
    arena.reset(seed)
    teacher = RichTeacherPolicy()
    opponent = RichRandomPolicy(seed, "red" if side == "blue" else "blue")
    steps: list[TraceStep] = []
    while not arena.state.terminal:
        observation = arena.observe(side)
        blue = (
            teacher.select(
                "blue", arena.legal_actions("blue"), arena.state.tick, arena.observe("blue")
            )
            if side == "blue"
            else opponent.select("blue", arena.legal_actions("blue"))
        )
        red = (
            teacher.select(
                "red", arena.legal_actions("red"), arena.state.tick, arena.observe("red")
            )
            if side == "red"
            else opponent.select("red", arena.legal_actions("red"))
        )
        selected = blue if side == "blue" else red
        steps.append(TraceStep(observation, selected, _template(selected), arena.state.tick))
        arena.step(blue, red)
    expected = f"{side}_win_crystal_destroyed"
    trajectory = [
        {"tick": step.tick, "observation": step.observation, "action": step.action.to_dict()}
        for step in steps
    ]
    return (
        _sha(_canonical(trajectory).encode()),
        tuple(steps),
        arena.state.outcome,
        arena.state.outcome == expected,
    )


def collect_rich_data(
    episode_seeds: range = range(256), variants: int = 2, enforce: bool = True
) -> RichPixelData:
    if variants != 2 and enforce:
        raise RichPixelError("formal collection requires exactly two render variants")
    if variants < 1:
        raise RichPixelError("at least one render variant is required")
    records: list[tuple[int, Side, str, tuple[TraceStep, ...], str, bool]] = []
    for seed in episode_seeds:
        for side in ("blue", "red"):
            typed_side: Side = "blue" if side == "blue" else "red"
            group, steps, outcome, completed = _teacher_episode(seed, typed_side)
            records.append((seed, typed_side, group, steps, outcome, completed))
    assignments = _group_splits([row[2] for row in records], enforce)
    frames: list[np.ndarray] = []
    labels: list[tuple[int, int, int, int, int]] = []
    templates: list[int] = []
    groups: list[bytes] = []
    ticks: list[int] = []
    render_seeds: list[int] = []
    variant_rows: list[int] = []
    split_rows: list[int] = []
    hashes: list[bytes] = []
    semantic_labels: dict[str, tuple[int, int, int, int, int]] = {}
    episodes: list[EpisodeSpec] = []
    for seed, side, group, steps, outcome, completed in records:
        seeds = _render_seeds(group, variants)
        episodes.append(
            EpisodeSpec(seed, side, group, assignments[group], seeds, outcome, completed)
        )
        for step in steps:
            label = _indices(_ego_action(step.action, cast(Side, step.observation["side"])))
            semantic_hash = _sha(render(step.observation, 0).tobytes())
            previous = semantic_labels.setdefault(semantic_hash, label)
            if previous != label:
                raise RichPixelError("same semantic RGB frame has conflicting action labels")
            for variant, render_seed in enumerate(seeds):
                frame = render(step.observation, render_seed)
                frames.append(frame)
                labels.append(label)
                templates.append(step.template)
                groups.append(group.encode())
                ticks.append(step.tick)
                render_seeds.append(render_seed)
                variant_rows.append(variant)
                split_rows.append(assignments[group])
                hashes.append(_sha(frame.tobytes()).encode())
    matrix = np.asarray(labels, dtype=np.uint8)
    data = RichPixelData(
        np.stack(frames),
        matrix[:, 0],
        matrix[:, 1],
        matrix[:, 2],
        matrix[:, 3],
        matrix[:, 4],
        np.asarray(templates, dtype=np.uint8),
        np.asarray(groups, dtype="S64"),
        np.asarray(ticks, dtype=np.uint16),
        np.asarray(render_seeds, dtype=np.int64),
        np.asarray(variant_rows, dtype=np.uint8),
        np.asarray(split_rows, dtype=np.uint8),
        np.asarray(hashes, dtype="S64"),
        episodes,
        ArenaConfig(),
    )
    _validate_collection(data, formal=enforce)
    return data


def _validate_collection(data: RichPixelData, formal: bool) -> None:
    count = len(data.frames)
    arrays = (
        data.macro,
        data.type,
        data.direction,
        data.target,
        data.skill,
        data.template,
        data.group_ids,
        data.ticks,
        data.render_seeds,
        data.variants,
        data.splits,
        data.frame_hashes,
    )
    if (
        data.frames.dtype != np.uint8
        or data.frames.shape != (count, 128, 128, 3)
        or any(len(value) != count for value in arrays)
    ):
        raise RichPixelError("dataset shape or length mismatch")
    if any(
        value.dtype != np.uint8
        for value in (
            data.macro,
            data.type,
            data.direction,
            data.target,
            data.skill,
            data.template,
            data.variants,
            data.splits,
        )
    ):
        raise RichPixelError("labels, variants and splits must be uint8")
    group_split: dict[bytes, int] = {}
    group_variants: dict[bytes, set[int]] = {}
    for group, split, variant in zip(
        data.group_ids.tolist(), data.splits.tolist(), data.variants.tolist(), strict=True
    ):
        if group in group_split and group_split[group] != int(split):
            raise RichPixelError("trajectory group crosses splits")
        group_split[group] = int(split)
        group_variants.setdefault(group, set()).add(int(variant))
    if formal:
        if len(data.episodes) != 512 or len(group_split) != 512:
            raise RichPixelError("formal dataset requires 512 grouped episodes")
        if any(values != {0, 1} for values in group_variants.values()):
            raise RichPixelError("every formal group requires exactly two render variants")
        completion = sum(int(episode.completed) for episode in data.episodes) / 512
        if completion < 0.90:
            raise RichPixelError(f"teacher completion below 0.90: {completion:.4f}")
        for split_name, split in SPLITS.items():
            for template in range(TEMPLATE_COUNT):
                count_template = int(np.sum((data.splits == split) & (data.template == template)))
                if count_template < 20:
                    raise RichPixelError(
                        f"template {template} in {split_name} below 20: {count_template}"
                    )


def dataset_summary(data: RichPixelData) -> dict[str, object]:
    completion = sum(int(item.completed) for item in data.episodes) / max(1, len(data.episodes))
    return {
        "episodes": len(data.episodes),
        "samples": len(data.frames),
        "trajectory_groups": len(set(data.group_ids.tolist())),
        "render_variants": sorted(set(data.variants.tolist())),
        "teacher_completion": completion,
        "split_groups": {
            name: sum(item.split == split for item in data.episodes)
            for name, split in SPLITS.items()
        },
        "split_samples": {
            name: int(np.sum(data.splits == split)) for name, split in SPLITS.items()
        },
        "template_counts": {
            str(index): int(np.sum(data.template == index)) for index in range(TEMPLATE_COUNT)
        },
        "frame_shape": [128, 128, 3],
        "frame_dtype": str(data.frames.dtype),
    }


def write_dataset(path: Path, data: RichPixelData) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            frames=data.frames,
            macro=data.macro,
            type=data.type,
            direction=data.direction,
            target=data.target,
            skill=data.skill,
            template=data.template,
            group_ids=data.group_ids,
            ticks=data.ticks,
            render_seeds=data.render_seeds,
            variants=data.variants,
            splits=data.splits,
            frame_hashes=data.frame_hashes,
        )
        handle.flush()
        os.fsync(handle.fileno())


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != DATA_KEYS:
                raise RichPixelError("invalid rich dataset field set")
            data = {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError) as exc:
        raise RichPixelError("invalid fixed-dtype NPZ") from exc
    count = len(data["frames"])
    expected = {
        "frames": np.dtype(np.uint8),
        "macro": np.dtype(np.uint8),
        "type": np.dtype(np.uint8),
        "direction": np.dtype(np.uint8),
        "target": np.dtype(np.uint8),
        "skill": np.dtype(np.uint8),
        "template": np.dtype(np.uint8),
        "group_ids": np.dtype("S64"),
        "ticks": np.dtype(np.uint16),
        "render_seeds": np.dtype(np.int64),
        "variants": np.dtype(np.uint8),
        "splits": np.dtype(np.uint8),
        "frame_hashes": np.dtype("S64"),
    }
    if data["frames"].shape != (count, 128, 128, 3) or count == 0:
        raise RichPixelError("invalid dataset frame shape")
    for key, dtype in expected.items():
        if data[key].dtype != dtype or len(data[key]) != count:
            raise RichPixelError(f"invalid dataset array: {key}")
    limits = {
        "macro": len(MACROS),
        "type": len(TYPES),
        "direction": len(DIRECTIONS),
        "target": len(TARGETS),
        "skill": len(SKILLS),
        "template": TEMPLATE_COUNT,
        "variants": 2,
        "splits": 4,
    }
    if any(bool(np.any(data[key] >= limit)) for key, limit in limits.items()):
        raise RichPixelError("dataset label out of range")
    if not np.array_equal(
        data["frame_hashes"],
        np.asarray([_sha(frame.tobytes()).encode() for frame in data["frames"]], dtype="S64"),
    ):
        raise RichPixelError("dataset frame hash mismatch")
    mapping: dict[bytes, int] = {}
    for group, split in zip(data["group_ids"].tolist(), data["splits"].tolist(), strict=True):
        if group in mapping and mapping[group] != int(split):
            raise RichPixelError("dataset group leakage")
        mapping[group] = int(split)
    return data


class RichPixelActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(1))
        self.heads = nn.ModuleList(
            nn.Linear(512, size)
            for size in (len(MACROS), len(TYPES), len(DIRECTIONS), len(TARGETS), len(SKILLS))
        )

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(frames)
        values = [head(features) for head in self.heads]
        return values[0], values[1], values[2], values[3], values[4]


Logits = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _normalize(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(frames)
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(127.5)
        .sub_(1)
    )


def _scores(logits: Logits, actions: Sequence[FactorizedAction]) -> torch.Tensor:
    if not actions:
        raise RichPixelError("decoder requires at least one canonical action")
    result = torch.empty(
        (logits[0].shape[0], len(actions)), device=logits[0].device, dtype=logits[0].dtype
    )
    for column, action in enumerate(actions):
        indices = _indices(action)
        result[:, column] = sum(head[:, index] for head, index in zip(logits, indices, strict=True))
    return result


def _decode_raw_template(logits: Logits) -> torch.Tensor:
    """Return indices into the complete fixed canonical action list."""
    return _scores(logits, ACTIONS).argmax(1)


def decode_raw(logits: Logits) -> tuple[FactorizedAction, ...]:
    return tuple(ACTIONS[int(index)] for index in _decode_raw_template(logits).tolist())


def _decode_legal_template(
    logits: Logits, legal: tuple[FactorizedAction, ...], side: Side = "blue"
) -> tuple[FactorizedAction, FactorizedAction]:
    if logits[0].shape[0] != 1:
        raise RichPixelError("execution decoder requires batch size one")
    if not legal or any(action not in ACTIONS for action in legal):
        raise RichPixelError("execution legal domain must be non-empty and canonical")
    raw = _ego_action(decode_raw(logits)[0], side)
    ego_legal = tuple(_ego_action(action, side) for action in legal)
    executed = legal[int(_scores(logits, ego_legal).argmax(1).item())]
    return executed, raw


def _weights(labels: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=classes).astype(np.float64)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def _deterministic(seed: int, device: torch.device) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _augment(batch: torch.Tensor, seed: int, epoch: int, index: int) -> torch.Tensor:
    rng = np.random.default_rng(seed * 1_000_003 + epoch * 1009 + index)
    dy, dx = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
    shifted = torch.roll(batch, shifts=(dy, dx), dims=(2, 3))
    return shifted.mul(float(rng.uniform(0.96, 1.04))).clamp(-1, 1)


def _metrics(
    actor: RichPixelActor, data: RichPixelData, indices: np.ndarray, device: torch.device
) -> dict[str, object]:
    if len(indices) == 0:
        raise RichPixelError("metrics split is empty")
    predictions: list[np.ndarray] = []
    raw_heads: list[np.ndarray] = []
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            logits = actor(_normalize(data.frames[indices[start : start + 256]], device))
            raw_heads.append(
                np.column_stack([head.argmax(1).cpu().numpy() for head in logits])
            )
            predicted_actions = decode_raw(logits)
            predictions.append(
                np.asarray([_indices(action) for action in predicted_actions], dtype=np.uint8)
            )
    pred = np.concatenate(predictions)
    raw = np.concatenate(raw_heads)
    truth = np.column_stack(
        tuple(
            getattr(data, name)[indices]
            for name in ("macro", "type", "direction", "target", "skill")
        )
    )
    factor_accuracy = {
        name: float(np.mean(raw[:, column] == truth[:, column]))
        for column, name in enumerate(("macro", "type", "direction", "target", "skill"))
    }
    pred_template = np.asarray(
        [
            _template(ACTIONS[int(index)])
            for index in [
                ACTIONS.index(
                    FactorizedAction(
                        MACROS[int(row[0])],
                        TYPES[int(row[1])],
                        TARGETS[int(row[3])],
                        DIRECTIONS[int(row[2])],
                        SKILLS[int(row[4])],
                    )
                )
                for row in pred
            ]
        ],
        dtype=np.uint8,
    )
    template_recall: dict[str, float] = {}
    for template in range(TEMPLATE_COUNT):
        mask = data.template[indices] == template
        template_recall[str(template)] = (
            float(np.mean(pred_template[mask] == template)) if np.any(mask) else 0.0
        )
    return {
        "joint_accuracy": float(np.mean(np.all(pred == truth, axis=1))),
        "balanced_accuracy": float(np.mean(tuple(template_recall.values()))),
        "factor_accuracy": factor_accuracy,
        "template_recall": template_recall,
    }


def _classification_pass(metrics: dict[str, object]) -> bool:
    factors = cast(dict[str, float], metrics["factor_accuracy"])
    recalls = cast(dict[str, float], metrics["template_recall"])
    return (
        cast(float, metrics["joint_accuracy"]) >= 0.90
        and cast(float, metrics["balanced_accuracy"]) >= 0.85
        and all(value >= 0.95 for value in factors.values())
        and all(value >= 0.70 for value in recalls.values())
    )


def _validation_loss(
    actor: RichPixelActor,
    data: RichPixelData,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    total = 0.0
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            selected = indices[start : start + 256]
            logits = actor(_normalize(data.frames[selected], device))
            targets = [
                torch.from_numpy(getattr(data, name)[selected].astype(np.int64)).to(device)
                for name in ("macro", "type", "direction", "target", "skill")
            ]
            total += sum(
                float(nn.functional.cross_entropy(head, target, reduction="sum").item())
                for head, target in zip(logits, targets, strict=True)
            )
    return total / len(indices)


def train_actor(
    data: RichPixelData,
    seed: int,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 128,
    patience: int = 8,
) -> tuple[RichPixelActor, dict[str, object]]:
    _deterministic(seed, device)
    actor = RichPixelActor().to(device)
    fit, validation = (
        np.flatnonzero(data.splits == SPLITS["fit"]),
        np.flatnonzero(data.splits == SPLITS["validation"]),
    )
    if not len(fit) or not len(validation):
        raise RichPixelError("fit and validation splits are required")
    weights = [
        _weights(getattr(data, name)[fit], size).to(device)
        for name, size in zip(
            ("macro", "type", "direction", "target", "skill"),
            (len(MACROS), len(TYPES), len(DIRECTIONS), len(TARGETS), len(SKILLS)),
            strict=True,
        )
    ]
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, {}
    for epoch in range(1, epochs + 1):
        order = fit.copy()
        np.random.default_rng(seed * 1009 + epoch).shuffle(order)
        actor.train()
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            selected = order[start : start + batch_size]
            logits = actor(
                _augment(_normalize(data.frames[selected], device), seed, epoch, batch_index)
            )
            targets = [
                torch.from_numpy(getattr(data, name)[selected].astype(np.int64)).to(device)
                for name in ("macro", "type", "direction", "target", "skill")
            ]
            losses = [
                nn.functional.cross_entropy(head, target, weight=weight)
                for head, target, weight in zip(logits, targets, weights, strict=True)
            ]
            loss = torch.stack(losses).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        value = _validation_loss(actor, data, validation, device)
        if value < best_loss - 1e-7:
            best_loss, best_epoch, stale = value, epoch, 0
            best_state = {
                key: tensor.detach().cpu().clone() for key, tensor in actor.state_dict().items()
            }
        else:
            stale += 1
        if stale >= patience:
            break
    if not best_state:
        raise RichPixelError("training produced no checkpoint")
    actor.load_state_dict(best_state)
    test = _metrics(actor, data, np.flatnonzero(data.splits == SPLITS["test"]), device)
    return actor, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test": test,
        "test_passed": _classification_pass(test),
    }


def _metadata(config: ArenaConfig, seed: int) -> dict[str, str]:
    return {
        "schema": MODEL_SCHEMA,
        "config_hash": config.digest,
        "renderer_id": RENDERER_ID,
        "renderer_hash": RENDERER_HASH,
        "action_hash": ACTION_HASH,
        "training_hash": TRAINING_HASH,
        "training_seed": str(seed),
        "head_sizes": "6,4,9,5,5",
        "rgb_only": "true",
        "weights": "none",
        "claim_scope": "pixelarena_engineering",
    }


def save_model(path: Path, actor: RichPixelActor, config: ArenaConfig, seed: int) -> None:
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in actor.state_dict().items()},
        path,
        metadata=_metadata(config, seed),
    )


def load_model(path: Path, config: ArenaConfig | None = None) -> tuple[RichPixelActor, int]:
    expected_config = config or ArenaConfig()
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except (OSError, SafetensorError) as exc:
        raise RichPixelError("invalid safetensors file") from exc
    if metadata is None or set(metadata) != set(_metadata(expected_config, 0)):
        raise RichPixelError("model metadata field set mismatch")
    seed_text = metadata.get("training_seed", "")
    if not seed_text.isdigit() or metadata != _metadata(expected_config, int(seed_text)):
        raise RichPixelError("model config/renderer/action/training metadata mismatch")
    actor = RichPixelActor()
    try:
        state = load_file(path, device="cpu")
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise RichPixelError("model contains non-finite tensors")
        actor.load_state_dict(state, strict=True)
    except (RuntimeError, SafetensorError) as exc:
        raise RichPixelError("model tensor contract mismatch") from exc
    actor.eval()
    return actor, int(seed_text)


def _rollout(
    actor: RichPixelActor, device: torch.device, side: Side, opponent_name: str, seed: int
) -> dict[str, object]:
    arena = RichPixelArena()
    arena.reset(seed)
    opponent = (
        RichRandomPolicy(seed, "red" if side == "blue" else "blue")
        if opponent_name == "random"
        else RichNullPolicy()
    )
    raw_illegal = corrections = executed_illegal = steps = 0
    actor.eval()
    while not arena.state.terminal:
        legal = arena.legal_actions(side)
        observation = arena.observe(side)
        with torch.no_grad():
            logits = actor(_normalize(render(observation, seed)[None], device))
        executed, raw = _decode_legal_template(logits, legal, side)
        raw_illegal += int(raw not in legal)
        corrections += int(raw != executed)
        executed_illegal += int(executed not in legal)
        other_side: Side = "red" if side == "blue" else "blue"
        other_legal = arena.legal_actions(other_side)
        other = opponent.select(other_side, other_legal, arena.state.tick)
        arena.step(executed if side == "blue" else other, other if side == "blue" else executed)
        steps += 1
    return {
        "side": side,
        "opponent": opponent_name,
        "seed": seed,
        "steps": steps,
        "completed": arena.state.outcome == f"{side}_win_crystal_destroyed",
        "outcome": arena.state.outcome,
        "raw_illegal": raw_illegal,
        "corrections": corrections,
        "executed_illegal": executed_illegal,
    }


def _closed_loop_gate(actor: RichPixelActor, device: torch.device) -> dict[str, object]:
    rows = [
        _rollout(actor, device, side, opponent, seed)
        for seed in range(20)
        for side in ("blue", "red")
        for opponent in ("null", "random")
    ]
    total_steps = sum(cast(int, row["steps"]) for row in rows)
    rate = {
        opponent: sum(bool(row["completed"]) for row in rows if row["opponent"] == opponent)
        / sum(row["opponent"] == opponent for row in rows)
        for opponent in ("null", "random")
    }
    random_rows = [row for row in rows if row["opponent"] == "random"]
    side_rates = {
        side: sum(bool(row["completed"]) for row in random_rows if row["side"] == side)
        / sum(row["side"] == side for row in random_rows)
        for side in ("blue", "red")
    }
    raw = sum(cast(int, row["raw_illegal"]) for row in rows) / total_steps
    corrections = sum(cast(int, row["corrections"]) for row in rows) / total_steps
    executed = sum(cast(int, row["executed_illegal"]) for row in rows)
    teacher_rows = [
        _teacher_episode(seed, side)[3]
        for seed in range(20)
        for side in ("blue", "red")
    ]
    teacher_rate = sum(teacher_rows) / len(teacher_rows)
    side_gap = abs(side_rates["blue"] - side_rates["red"])
    checks = {
        "null_completion_at_least_0_95": rate["null"] >= 0.95,
        "random_completion_at_least_0_90": rate["random"] >= 0.90,
        "random_at_least_0_95_teacher": rate["random"] >= 0.95 * teacher_rate,
        "blue_red_gap_at_most_0_05": side_gap <= 0.05,
        "raw_illegal_at_most_0_02": raw <= 0.02,
        "mask_correction_at_most_0_02": corrections <= 0.02,
        "executed_illegal_zero": executed == 0,
    }
    passed = (
        rate["null"] >= 0.95
        and rate["random"] >= 0.90
        and rate["random"] >= 0.95 * teacher_rate
        and side_gap <= 0.05
        and raw <= 0.02
        and corrections <= 0.02
        and executed == 0
    )
    return {
        "passed": passed,
        "null_completion": rate["null"],
        "random_completion": rate["random"],
        "side_completion": side_rates,
        "blue_red_completion_gap": side_gap,
        "matched_teacher_completion": teacher_rate,
        "raw_illegal_rate": raw,
        "correction_rate": corrections,
        "executed_illegal": executed,
        "checks": checks,
        "failed_checks": [name for name, accepted in checks.items() if not accepted],
        "rows": rows,
    }


def _controls(
    actor: RichPixelActor, data: RichPixelData, device: torch.device, evaluated_seed: int
) -> dict[str, object]:
    indices = np.flatnonzero(data.splits == SPLITS["test"])
    actual = _metrics(actor, data, indices, device)
    original = data.frames
    permutation = np.random.default_rng(71).permutation(indices)
    try:
        data.frames = original.copy()
        data.frames[indices] = 0
        black_metrics = _metrics(actor, data, indices, device)
        data.frames = original.copy()
        data.frames[indices] = original[permutation]
        mismatch_metrics = _metrics(actor, data, indices, device)
    finally:
        data.frames = original
    actual_joint = cast(float, actual["joint_accuracy"])
    black_drop = actual_joint - cast(float, black_metrics["joint_accuracy"])
    mismatch_drop = actual_joint - cast(float, mismatch_metrics["joint_accuracy"])
    checks = {
        "black_joint_accuracy_drop_over_0_20": black_drop > 0.20,
        "mismatch_joint_accuracy_drop_over_0_20": mismatch_drop > 0.20,
    }
    passed = cast(float, black_metrics["joint_accuracy"]) + 0.20 < actual_joint and cast(
        float, mismatch_metrics["joint_accuracy"]
    ) + 0.20 < actual_joint
    return {
        "passed": passed,
        "evaluated_seed": evaluated_seed,
        "domain": "synthetic_pixelarena",
        "actual": actual,
        "black": black_metrics,
        "mismatched_frames": mismatch_metrics,
        "joint_accuracy_drop": {"black": black_drop, "mismatched": mismatch_drop},
        "checks": checks,
        "failed_checks": [name for name, accepted in checks.items() if not accepted],
    }


def _formal_failure_report(
    data: RichPixelData,
    runs: list[dict[str, object]],
    controls: dict[str, object],
    closed: list[dict[str, object]],
    replay: dict[str, object],
    best: int,
    runtime: dict[str, str],
) -> dict[str, object]:
    failed_checks = [
        f"controls.{name}" for name in cast(list[str], controls["failed_checks"])
    ]
    for item in closed:
        seed = cast(int, item["training_seed"])
        failed_checks.extend(
            f"closed_loop.seed_{seed}.{name}"
            for name in cast(list[str], item["failed_checks"])
        )
    return {
        "kind": "rich_pixel_v7_formal_report_v2",
        "status": "FAILED",
        "disposition": "NON_PROMOTING_DIAGNOSTIC",
        "failed_stage": "controls_or_closed_loop",
        "failed_checks": failed_checks,
        "error": "control or closed-loop formal threshold failed",
        "promotion_eligible": False,
        "claim_scope": "pixelarena_engineering",
        "hok_capability_claim": False,
        "config_hash": data.config.digest,
        "renderer_hash": RENDERER_HASH,
        "action_hash": ACTION_HASH,
        "training_hash": TRAINING_HASH,
        "dataset": dataset_summary(data),
        "training_runs": runs,
        "selected_evaluation_seed": TRAINING_SEEDS[best],
        "selection_rule": "minimum_best_validation_loss",
        "controls": controls,
        "closed_loop": closed,
        "fresh_process_replay": replay,
        "latency": {"status": "NOT_RUN", "reason": "prior_formal_gate_failed"},
        "files": {},
        "retained_artifacts": ["report.json"],
        "models_retained": False,
        "runtime": runtime,
    }


def _publish_failed_report(output: Path, report: dict[str, object]) -> Path:
    failed = output.with_name(f"{output.name}.failed-{time.time_ns()}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{failed.name}.tmp-", dir=output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if {path.name for path in temporary.iterdir()} != {"report.json"}:
            raise RichPixelError("failed-report atomic directory field set mismatch")
        temporary.rename(failed)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return failed


def _latency(actor: RichPixelActor, device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        raise RichPixelError("formal p95 protocol requires CUDA")
    sample = torch.zeros((1, 3, 128, 128), device=device)
    actor.eval()
    with torch.no_grad():
        for _ in range(100):
            actor(sample)
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(500):
            start = time.perf_counter()
            actor(sample)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return {
        "device": "cuda",
        "batch": 1,
        "dtype": "float32",
        "warmup": 100,
        "samples": 500,
        "scope": "forward_only_synchronized",
        "p95_ms": float(np.percentile(times, 95)),
    }


def _smoke() -> dict[str, object]:
    data = collect_rich_data(range(1), variants=2, enforce=False)
    actor = RichPixelActor().eval()
    with tempfile.TemporaryDirectory(prefix="rich-pixel-smoke-") as directory:
        root = Path(directory)
        dataset, model = root / "dataset.npz", root / "model.safetensors"
        write_dataset(dataset, data)
        arrays = load_dataset(dataset)
        save_model(model, actor, data.config, 0)
        loaded, seed = load_model(model, data.config)
        shapes = [
            list(value.shape) for value in loaded(_normalize(data.frames[:1], torch.device("cpu")))
        ]
    return {
        "status": "PASSED",
        "disposition": "NON_PROMOTING_CPU_SMOKE",
        "samples": len(data.frames),
        "arrays": sorted(arrays),
        "seed": seed,
        "head_shapes": shapes,
        "training": "not_run_non_promoting_smoke",
    }


def accept_rich_pixel(
    output: Path | None, device_name: str, smoke: bool = False
) -> dict[str, object]:
    if smoke:
        if device_name != "cpu" or output is not None:
            raise RichPixelError("CPU smoke is non-promoting and accepts no output directory")
        return _smoke()
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RichPixelError(
            "formal Rich Pixel acceptance requires available CUDA; no evidence was written"
        )
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RichPixelError(
            "formal Rich Pixel acceptance requires NVIDIA GeForce RTX 4090"
        )
    if output is None or output.exists():
        raise RichPixelError("formal acceptance requires a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        data = collect_rich_data()
        replay_path = temporary / "replay.jsonl"
        record_rich_trace(replay_path, "teacher", "null", 0)
        replay = verify_rich_trace(replay_path)
        if replay.get("verified") is not True:
            raise RichPixelError("fresh-process replay verification failed")
        original = [
            json.loads(line)
            for line in replay_path.read_text(encoding="utf-8").splitlines()
        ]
        tamper_rejected: dict[str, bool] = {}
        for kind in ("config", "action", "event"):
            changed = json.loads(_canonical(original))
            if kind == "config":
                changed[0]["config_hash"] = "0" * 64
            elif kind == "action":
                changed[1]["blue_action"]["auxiliary"] = 1
            else:
                changed[1]["events"] = [{"tampered": True}]
            tampered_path = temporary / f"replay-tampered-{kind}.jsonl"
            tampered_path.write_text(
                "".join(_canonical(row) + "\n" for row in changed), encoding="utf-8"
            )
            try:
                verify_rich_trace(tampered_path)
            except ReplayError:
                tamper_rejected[kind] = True
            else:
                tamper_rejected[kind] = False
            tampered_path.unlink()
        if not all(tamper_rejected.values()):
            raise RichPixelError("fresh-process replay tamper rejection failed")
        replay["tamper_rejected"] = tamper_rejected
        replay_path.unlink()
        dataset_path = temporary / "dataset.npz"
        write_dataset(dataset_path, data)
        load_dataset(dataset_path)
        actors: list[RichPixelActor] = []
        runs: list[dict[str, object]] = []
        for seed in TRAINING_SEEDS:
            actor, metrics = train_actor(data, seed, torch.device("cuda:0"))
            if not bool(metrics["test_passed"]):
                raise RichPixelError(f"classification thresholds failed for seed {seed}")
            actors.append(actor)
            runs.append(metrics)
        best = min(
            range(3),
            key=lambda index: cast(float, runs[index]["best_validation_loss"]),
        )
        controls = _controls(
            actors[best], data, torch.device("cuda:0"), TRAINING_SEEDS[best]
        )
        closed = []
        for seed, actor in zip(TRAINING_SEEDS, actors, strict=True):
            result = _closed_loop_gate(actor, torch.device("cuda:0"))
            result["training_seed"] = seed
            closed.append(result)
        if not bool(controls["passed"]) or not all(bool(item["passed"]) for item in closed):
            failed_report = _formal_failure_report(
                data,
                runs,
                controls,
                closed,
                replay,
                best,
                {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0),
                },
            )
            failed_output = _publish_failed_report(output, failed_report)
            raise RichPixelError(
                "control or closed-loop formal threshold failed; "
                f"status=FAILED report retained at {failed_output / 'report.json'}"
            )
        latency = _latency(actors[best], torch.device("cuda:0"))
        if cast(float, latency["p95_ms"]) > 10.0:
            raise RichPixelError("formal CUDA forward p95 exceeds 10 ms")
        files: dict[str, str] = {}
        for actor, seed in zip(actors, TRAINING_SEEDS, strict=True):
            path = temporary / f"model-seed-{seed}.safetensors"
            save_model(path, actor, data.config, seed)
            load_model(path, data.config)
            files[path.name] = _sha(path.read_bytes())
        files[dataset_path.name] = _sha(dataset_path.read_bytes())
        report = {
            "kind": "rich_pixel_v7_formal_report_v2",
            "status": "PASSED",
            "claim_scope": "pixelarena_engineering",
            "hok_capability_claim": False,
            "config_hash": data.config.digest,
            "renderer_hash": RENDERER_HASH,
            "action_hash": ACTION_HASH,
            "training_hash": TRAINING_HASH,
            "dataset": dataset_summary(data),
            "training_runs": runs,
            "controls": controls,
            "closed_loop": closed,
            "fresh_process_replay": replay,
            "latency": latency,
            "files": files,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
            },
        }
        report_path = temporary / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        if set(path.name for path in temporary.iterdir()) != {
            "dataset.npz",
            "model-seed-0.safetensors",
            "model-seed-1.safetensors",
            "model-seed-2.safetensors",
            "report.json",
        }:
            raise RichPixelError("formal atomic directory field set mismatch")
        temporary.rename(output)
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
