# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import cast

import numpy as np
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torchvision import __version__ as torchvision_version  # type: ignore[import-untyped]
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.arena import ArenaConfig, FactorizedAction, PixelArena, Side
from hok_agent.policies import NullPolicy, RandomPolicy, TacticalTeacher
from hok_agent.renderer import RENDERER_HASH, RENDERER_ID, render
from hok_agent.safety import check_project

ACTIONS = (("wait", "none", "none"), ("move", "none", "forward"), ("move", "none", "backward"), ("attack", "enemy_hero", "none"), ("attack", "enemy_tower", "none"), ("attack", "enemy_crystal", "none"))
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}
ACTION_HASH = hashlib.sha256(json.dumps(ACTIONS, separators=(",", ":")).encode("utf-8")).hexdigest()
SPLITS = {"fit": 0, "acquisition": 1, "validation": 2, "test": 3}
SPLIT_NAMES = {value: key for key, value in SPLITS.items()}
DATA_KEYS = {"frames", "actions", "group_ids", "ticks", "render_seeds", "splits", "frame_hashes", "sources"}
MODEL_SCHEMA = "pixelarena-rgb-resnet18-v1"
TRAINING_SEEDS = (0, 1, 2)
TRAINING_CONFIG = {"architecture": "torchvision-resnet18-weights-none-6", "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 128, "max_epochs": 50, "validation_patience": 8, "schedule": "cosine", "class_weight": "inverse_sqrt_frequency", "augmentation": "deterministic_color_scale_and_translation_le2px", "precision": "float32"}
TRAINING_HASH = hashlib.sha256(json.dumps(TRAINING_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class PixelError(ValueError):
    pass


@dataclass(frozen=True)
class TraceStep:
    observation: dict[str, object]
    action: int
    legal: tuple[bool, ...]
    tick: int


@dataclass(frozen=True)
class EpisodeTrace:
    seed: int
    side: Side
    group_id: str
    steps: tuple[TraceStep, ...]
    outcome: str
    completed: bool


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
class PixelData:
    frames: np.ndarray
    actions: np.ndarray
    group_ids: np.ndarray
    ticks: np.ndarray
    render_seeds: np.ndarray
    splits: np.ndarray
    frame_hashes: np.ndarray
    sources: np.ndarray
    legal: np.ndarray
    episodes: list[EpisodeSpec]
    config: ArenaConfig


@dataclass(frozen=True)
class Rollout:
    seed: int
    side: Side
    render_seed: int
    opponent: str
    outcome: str
    terminal: bool
    completed: bool
    steps: int
    raw_illegal: int
    corrections: int


class PixelActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = resnet18(weights=None, num_classes=len(ACTIONS))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(frames))


def _action_key(action: FactorizedAction) -> tuple[str, str, str]:
    return action.action_type, action.target, action.direction


def _action_index(action: FactorizedAction) -> int:
    try:
        return ACTION_INDEX[_action_key(action)]
    except KeyError as exc:
        raise PixelError("action outside fixed six-class vocabulary") from exc


def _legal_mask(legal: tuple[FactorizedAction, ...]) -> tuple[bool, ...]:
    indices = {_action_index(action) for action in legal}
    return tuple(index in indices for index in range(len(ACTIONS)))


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _teacher_episode(seed: int, side: Side) -> EpisodeTrace:
    arena = PixelArena()
    arena.reset(seed)
    teacher = TacticalTeacher()
    other: Side = "red" if side == "blue" else "blue"
    opponent = RandomPolicy(seed, other)
    steps: list[TraceStep] = []
    identity: list[dict[str, object]] = []
    while not arena.state.terminal:
        blue_legal = arena.legal_actions("blue")
        red_legal = arena.legal_actions("red")
        observation = arena.observe(side)
        if side == "blue":
            selected = teacher.select("blue", blue_legal, arena.state.tick)
            blue = selected
            red = opponent.select("red", red_legal)
            selected_legal = blue_legal
        else:
            blue = opponent.select("blue", blue_legal)
            selected = teacher.select("red", red_legal, arena.state.tick)
            red = selected
            selected_legal = red_legal
        action = _action_index(selected)
        steps.append(TraceStep(observation, action, _legal_mask(selected_legal), arena.state.tick))
        response = arena.step(blue, red)
        identity.append({"public_state": response["observation"], "blue_action": blue.to_dict(), "red_action": red.to_dict(), "events": response["events"]})
    group = _sha_bytes(_canonical({"side": side, "trajectory": identity}).encode("utf-8"))
    expected = f"{side}_win_crystal_destroyed"
    return EpisodeTrace(seed, side, group, tuple(steps), arena.state.outcome, arena.state.outcome == expected)


def _group_splits(groups: set[str]) -> dict[str, int]:
    ordered = sorted(groups)
    count = len(ordered)
    fit_end = int(count * 0.56)
    acquisition_end = int(count * 0.70)
    validation_end = int(count * 0.85)
    if min(fit_end, acquisition_end - fit_end, validation_end - acquisition_end) < 1:
        raise PixelError("not enough trajectory groups for four splits")
    result: dict[str, int] = {}
    for index, group in enumerate(ordered):
        if index < fit_end:
            split = SPLITS["fit"]
        elif index < acquisition_end:
            split = SPLITS["acquisition"]
        elif index < validation_end:
            split = SPLITS["validation"]
        else:
            split = SPLITS["test"]
        result[group] = split
    return result


def _render_seeds(group_id: str, variants: int) -> tuple[int, ...]:
    base = int(group_id[:16], 16)
    return tuple((base + variant * 1_000_003) % (2**31 - 1) for variant in range(variants))


def collect_pixel_data(episode_seeds: range = range(128), variants: int = 2, enforce: bool = True) -> PixelData:
    traces = [_teacher_episode(seed, side) for seed in episode_seeds for side in ("blue", "red")]
    assignments = _group_splits({trace.group_id for trace in traces})
    frames: list[np.ndarray] = []
    actions: list[int] = []
    groups: list[bytes] = []
    ticks: list[int] = []
    seeds: list[int] = []
    splits: list[int] = []
    hashes: list[bytes] = []
    legal: list[tuple[bool, ...]] = []
    episodes: list[EpisodeSpec] = []
    for trace in traces:
        split = assignments[trace.group_id]
        render_seeds = _render_seeds(trace.group_id, variants)
        episodes.append(EpisodeSpec(trace.seed, trace.side, trace.group_id, split, render_seeds, trace.outcome, trace.completed))
        for step in trace.steps:
            for render_seed in render_seeds:
                frame = render(step.observation, render_seed)
                frames.append(frame)
                actions.append(step.action)
                groups.append(trace.group_id.encode("ascii"))
                ticks.append(step.tick)
                seeds.append(render_seed)
                splits.append(split)
                hashes.append(_sha_bytes(frame.tobytes()).encode("ascii"))
                legal.append(step.legal)
    data = PixelData(np.stack(frames), np.asarray(actions, dtype=np.uint8), np.asarray(groups, dtype="S64"), np.asarray(ticks, dtype=np.uint16), np.asarray(seeds, dtype=np.int64), np.asarray(splits, dtype=np.uint8), np.asarray(hashes, dtype="S64"), np.zeros(len(frames), dtype=np.uint8), np.asarray(legal, dtype=np.bool_), episodes, PixelArena().config)
    if enforce:
        _validate_collection(data)
    return data


def _validate_collection(data: PixelData) -> None:
    if len(data.episodes) != 256:
        raise PixelError("formal collection must contain 256 episodes")
    if not all(episode.outcome != "ongoing" for episode in data.episodes):
        raise PixelError("episode did not reach terminal")
    completion = sum(episode.completed for episode in data.episodes) / len(data.episodes)
    if completion < 0.95:
        raise PixelError(f"teacher crystal completion below 95%: {completion:.4f}")
    for split_name, split in SPLITS.items():
        counts = Counter(data.actions[data.splits == split].tolist())
        if any(counts[action] < 20 for action in range(len(ACTIONS))):
            raise PixelError(f"split {split_name} lacks 20 samples for every action: {counts}")
    group_to_split: dict[bytes, int] = {}
    for group, split in zip(data.group_ids.tolist(), data.splits.tolist(), strict=True):
        previous = group_to_split.setdefault(group, int(split))
        if previous != split:
            raise PixelError("trajectory group crosses dataset splits")


def dataset_summary(data: PixelData) -> dict[str, object]:
    split_counts = Counter(data.splits.tolist())
    class_by_split = {name: {str(action): int(np.sum(data.actions[data.splits == code] == action)) for action in range(len(ACTIONS))} for name, code in SPLITS.items()}
    return {"episodes": len(data.episodes), "teacher_crystal_completion": sum(episode.completed for episode in data.episodes) / len(data.episodes), "samples": len(data.frames), "trajectory_groups": len(set(data.group_ids.tolist())), "split_counts": {SPLIT_NAMES[key]: value for key, value in sorted(split_counts.items())}, "class_counts_by_split": class_by_split, "frame_shape": list(data.frames.shape[1:]), "frame_dtype": str(data.frames.dtype)}


def write_dataset(path: Path, data: PixelData) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, frames=data.frames, actions=data.actions, group_ids=data.group_ids, ticks=data.ticks, render_seeds=data.render_seeds, splits=data.splits, frame_hashes=data.frame_hashes, sources=data.sources)
        handle.flush()
        os.fsync(handle.fileno())


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != DATA_KEYS:
            raise PixelError("invalid pixel dataset fields")
        values = {name: archive[name].copy() for name in archive.files}
    expected = {"frames": np.dtype(np.uint8), "actions": np.dtype(np.uint8), "group_ids": np.dtype("S64"), "ticks": np.dtype(np.uint16), "render_seeds": np.dtype(np.int64), "splits": np.dtype(np.uint8), "frame_hashes": np.dtype("S64"), "sources": np.dtype(np.uint8)}
    count = len(values["actions"])
    if values["frames"].shape != (count, 128, 128, 3):
        raise PixelError("invalid pixel frame shape")
    if any(values[name].dtype != dtype for name, dtype in expected.items()):
        raise PixelError("invalid pixel dataset dtype")
    if any(len(value) != count for value in values.values()):
        raise PixelError("pixel dataset arrays differ in length")
    if count == 0:
        raise PixelError("pixel dataset is empty")
    if bool(np.any(values["actions"] >= len(ACTIONS))):
        raise PixelError("invalid pixel action index")
    if bool(np.any(values["splits"] >= len(SPLITS))):
        raise PixelError("invalid pixel split index")
    if bool(np.any(values["sources"] > 1)):
        raise PixelError("invalid pixel source index")
    group_to_split: dict[bytes, int] = {}
    for group, split in zip(values["group_ids"].tolist(), values["splits"].tolist(), strict=True):
        previous = group_to_split.setdefault(group, int(split))
        if previous != split:
            raise PixelError("trajectory group crosses dataset splits")
    expected_hashes = np.asarray([_sha_bytes(frame.tobytes()).encode("ascii") for frame in values["frames"]], dtype="S64")
    if not bool(np.array_equal(expected_hashes, values["frame_hashes"])):
        raise PixelError("pixel frame hash mismatch")
    return values


def _normalize(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    return tensor.div_(127.5).sub_(1.0)


def _augment(batch: torch.Tensor, seed: int, epoch: int, batch_index: int) -> torch.Tensor:
    rng = Random(seed * 1_000_003 + epoch * 10_007 + batch_index)
    dy, dx = rng.randint(-2, 2), rng.randint(-2, 2)
    shifted = torch.zeros_like(batch)
    source_y = slice(max(0, -dy), min(128, 128 - dy))
    target_y = slice(max(0, dy), min(128, 128 + dy))
    source_x = slice(max(0, -dx), min(128, 128 - dx))
    target_x = slice(max(0, dx), min(128, 128 + dx))
    shifted[:, :, target_y, target_x] = batch[:, :, source_y, source_x]
    return shifted.mul(rng.uniform(0.9, 1.1)).clamp_(-1.0, 1.0)


def _metric_arrays(actor: PixelActor, frames: np.ndarray, labels: np.ndarray, legal: np.ndarray, device: torch.device, batch_size: int = 256) -> dict[str, object]:
    predictions: list[int] = []
    loss_sum = 0.0
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            end = min(len(frames), start + batch_size)
            x = _normalize(frames[start:end], device)
            y = torch.from_numpy(labels[start:end].astype(np.int64)).to(device)
            logits = actor(x)
            loss_sum += float(nn.functional.cross_entropy(logits, y, reduction="sum").item())
            predictions.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())
    predicted = np.asarray(predictions, dtype=np.int64)
    recalls = {str(action): float(np.mean(predicted[labels == action] == action)) if bool(np.any(labels == action)) else 0.0 for action in range(len(ACTIONS))}
    illegal = int(np.sum(~legal[np.arange(len(predicted)), predicted]))
    return {"cross_entropy": loss_sum / len(labels), "exact_accuracy": float(np.mean(predicted == labels)), "balanced_accuracy": sum(recalls.values()) / len(recalls), "per_class_recall": recalls, "raw_illegal_top1_rate": illegal / len(labels), "mask_correction_rate": illegal / len(labels), "executed_illegal_actions": 0}


def _split_metrics(actor: PixelActor, data: PixelData, split: int, device: torch.device) -> dict[str, object]:
    selected = np.flatnonzero(data.splits == split)
    return _metric_arrays(actor, data.frames[selected], data.actions[selected], data.legal[selected], device)


def _classification_pass(metrics: dict[str, object]) -> bool:
    recalls = cast(dict[str, float], metrics["per_class_recall"])
    return cast(float, metrics["exact_accuracy"]) >= 0.95 and cast(float, metrics["balanced_accuracy"]) >= 0.90 and min(recalls.values()) >= 0.80 and cast(float, metrics["raw_illegal_top1_rate"]) <= 0.01 and cast(float, metrics["mask_correction_rate"]) <= 0.01 and cast(int, metrics["executed_illegal_actions"]) == 0


def _configure_determinism(seed: int, device: torch.device) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def train_actor(data: PixelData, seed: int, device: torch.device, epochs: int = 50, batch_size: int = 128, patience: int = 8) -> tuple[PixelActor, dict[str, object]]:
    _configure_determinism(seed, device)
    actor = PixelActor().to(device)
    if sum(parameter.numel() for parameter in actor.parameters()) > 12_000_000:
        raise PixelError("pixel actor exceeds 12M parameters")
    fit = np.flatnonzero(data.splits == SPLITS["fit"])
    validation = np.flatnonzero(data.splits == SPLITS["validation"])
    counts = np.bincount(data.actions[fit], minlength=len(ACTIONS)).astype(np.float64)
    class_weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss = math.inf
    best_epoch = stale = 0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, epochs + 1):
        order = fit.copy()
        np.random.default_rng(seed * 10_000 + epoch).shuffle(order)
        actor.train()
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            indices = order[start : start + batch_size]
            x = _augment(_normalize(data.frames[indices], device), seed, epoch, batch_index)
            y = torch.from_numpy(data.actions[indices].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(actor(x), y, weight=weight)
            torch.autograd.backward(loss)
            optimizer.step()
        scheduler.step()
        metrics = _metric_arrays(actor, data.frames[validation], data.actions[validation], data.legal[validation], device)
        loss_value = cast(float, metrics["cross_entropy"])
        if loss_value < best_loss - 1e-7:
            best_loss = loss_value
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    actor.load_state_dict(best_state)
    actor.to(device).eval()
    validation_metrics = _split_metrics(actor, data, SPLITS["validation"], device)
    return actor, {"seed": seed, "epochs": best_epoch, "validation": validation_metrics, "validation_passed": _classification_pass(validation_metrics)}


def _model_metadata(config: ArenaConfig, seed: int) -> dict[str, str]:
    return {"schema_version": MODEL_SCHEMA, "arena_config_hash": config.digest, "renderer_contract_hash": RENDERER_HASH, "action_vocabulary_hash": ACTION_HASH, "architecture": "torchvision-resnet18-weights-none-6", "training_contract_hash": TRAINING_HASH, "training_seed": str(seed), "claim_scope": "pixelarena_engineering", "hok_capability_claim": "false", "gamecore_equivalence_claim": "false", "normalization": "x/127.5-1"}


def save_model(path: Path, actor: PixelActor, config: ArenaConfig, seed: int) -> None:
    state = {key: value.detach().cpu().contiguous() for key, value in actor.state_dict().items()}
    save_file(state, path, metadata=_model_metadata(config, seed))


def load_model(path: Path, config: ArenaConfig | None = None) -> tuple[PixelActor, int]:
    expected_config = config or PixelArena().config
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except SafetensorError as exc:
        raise PixelError("invalid safetensors model") from exc
    if metadata is None or set(metadata) != set(_model_metadata(expected_config, 0)):
        raise PixelError("invalid pixel model metadata fields")
    seed_text = metadata.get("training_seed", "")
    if not seed_text.isdigit():
        raise PixelError("invalid pixel model training seed")
    seed = int(seed_text)
    if metadata != _model_metadata(expected_config, seed):
        raise PixelError("pixel model contract mismatch")
    actor = PixelActor()
    try:
        state = load_file(path, device="cpu")
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise PixelError("pixel model tensors must be finite")
        actor.load_state_dict(state, strict=True)
    except (RuntimeError, SafetensorError) as exc:
        raise PixelError("invalid pixel model tensors") from exc
    actor.eval()
    return actor, seed


def infer_rgb_frames(model_path: Path, frames: np.ndarray, device_name: str) -> tuple[list[int], list[float], int]:
    """Run the frozen RGB Actor without exposing Torch at the Shadow boundary."""
    predict, seed = open_rgb_predictor(model_path, device_name)
    predictions, confidences = predict(frames)
    return predictions, confidences, seed


def open_rgb_predictor(model_path: Path, device_name: str) -> tuple[Callable[[np.ndarray], tuple[list[int], list[float]]], int]:
    """Load one RGB Actor and return a reusable RGB-only batch predictor."""
    if device_name not in {"cpu", "cuda"}:
        raise PixelError("device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise PixelError("CUDA requested but unavailable")
    device = torch.device(device_name)
    actor, seed = load_model(model_path)
    actor.to(device).eval()

    def predict(frames: np.ndarray) -> tuple[list[int], list[float]]:
        if frames.ndim != 4 or frames.shape[1:] != (128, 128, 3) or frames.dtype != np.uint8:
            raise PixelError("RGB inference expects uint8 NHWC 128x128 frames")
        with torch.no_grad():
            probabilities = torch.softmax(actor(_normalize(frames, device)), dim=1)
        if not bool(torch.isfinite(probabilities).all()):
            raise PixelError("pixel model produced non-finite probabilities")
        confidence, prediction = probabilities.max(dim=1)
        return prediction.cpu().tolist(), confidence.cpu().tolist()

    return predict, seed


def _choose_action(actor: PixelActor, observation: dict[str, object], legal: tuple[FactorizedAction, ...], render_seed: int, device: torch.device) -> tuple[FactorizedAction, int, int]:
    frame = render(observation, render_seed)
    with torch.no_grad():
        logits = actor(_normalize(frame[None, ...], device))[0]
    raw = int(logits.argmax().item())
    legal_by_index = {_action_index(action): action for action in legal}
    selected_index = max(legal_by_index, key=lambda index: float(logits[index]))
    return (legal_by_index[selected_index], int(raw not in legal_by_index), int(raw != selected_index))


def _actor_episode(actor: PixelActor, seed: int, side: Side, opponent_name: str, render_seed: int, device: torch.device, collect_teacher_labels: bool = False) -> tuple[Rollout, list[tuple[np.ndarray, int, tuple[bool, ...], int]], str]:
    arena = PixelArena()
    arena.reset(seed)
    other: Side = "red" if side == "blue" else "blue"
    opponent = NullPolicy() if opponent_name == "null" else RandomPolicy(seed, other)
    teacher = TacticalTeacher()
    collected: list[tuple[np.ndarray, int, tuple[bool, ...], int]] = []
    identity: list[dict[str, object]] = []
    raw_illegal = corrections = steps = 0
    actor.eval()
    while not arena.state.terminal:
        blue_legal = arena.legal_actions("blue")
        red_legal = arena.legal_actions("red")
        observation = arena.observe(side)
        actor_legal = blue_legal if side == "blue" else red_legal
        if collect_teacher_labels:
            label = teacher.select(side, actor_legal, arena.state.tick)
            collected.append((render(observation, render_seed), _action_index(label), _legal_mask(actor_legal), arena.state.tick))
        selected, invalid, corrected = _choose_action(actor, observation, actor_legal, render_seed, device)
        raw_illegal += invalid
        corrections += corrected
        if side == "blue":
            blue = selected
            red = opponent.select("red", red_legal)
        else:
            blue = opponent.select("blue", blue_legal)
            red = selected
        response = arena.step(blue, red)
        identity.append({"public_state": response["observation"], "blue_action": blue.to_dict(), "red_action": red.to_dict(), "events": response["events"]})
        steps += 1
    group = _sha_bytes(_canonical({"side": side, "trajectory": identity}).encode("utf-8"))
    expected = f"{side}_win_crystal_destroyed"
    return (Rollout(seed, side, render_seed, opponent_name, arena.state.outcome, arena.state.terminal, arena.state.outcome == expected, steps, raw_illegal, corrections), collected, group)


def _rollout_document(rollout: Rollout) -> dict[str, object]:
    return {"seed": rollout.seed, "side": rollout.side, "render_seed": rollout.render_seed, "opponent": rollout.opponent, "outcome": rollout.outcome, "terminal": rollout.terminal, "completed": rollout.completed, "steps": rollout.steps, "raw_illegal": rollout.raw_illegal, "mask_corrections": rollout.corrections, "executed_illegal": 0}


def _acquisition(actor: PixelActor, data: PixelData, device: torch.device) -> tuple[dict[str, object], list[tuple[np.ndarray, int, tuple[bool, ...], int, str, int]]]:
    specs = [episode for episode in data.episodes if episode.split == SPLITS["acquisition"]]
    rollouts: list[Rollout] = []
    dagger: list[tuple[np.ndarray, int, tuple[bool, ...], int, str, int]] = []
    for spec in specs:
        render_seed = spec.render_seeds[0]
        rollout, collected, group = _actor_episode(actor, spec.seed, spec.side, "random", render_seed, device, True)
        rollouts.append(rollout)
        fit_group = _sha_bytes(f"dagger:{group}".encode("ascii"))
        dagger.extend((frame, action, legal, tick, fit_group, render_seed) for frame, action, legal, tick in collected)
    actor_rate = sum(row.completed for row in rollouts) / len(rollouts)
    teacher_rate = sum(spec.completed for spec in specs) / len(specs)
    total_steps = sum(row.steps for row in rollouts)
    correction_rate = sum(row.corrections for row in rollouts) / total_steps
    return {"episodes": len(rollouts), "actor_completion": actor_rate, "teacher_completion": teacher_rate, "relative_completion": actor_rate / teacher_rate if teacher_rate else 0.0, "mask_correction_rate": correction_rate, "triggered": actor_rate < 0.95 * teacher_rate or correction_rate > 0.01}, dagger


def _append_dagger(data: PixelData, samples: list[tuple[np.ndarray, int, tuple[bool, ...], int, str, int]]) -> PixelData:
    frames = np.stack([sample[0] for sample in samples])
    actions = np.asarray([sample[1] for sample in samples], dtype=np.uint8)
    groups = np.asarray([sample[4].encode("ascii") for sample in samples], dtype="S64")
    ticks = np.asarray([sample[3] for sample in samples], dtype=np.uint16)
    render_seeds = np.asarray([sample[5] for sample in samples], dtype=np.int64)
    hashes = np.asarray([_sha_bytes(frame.tobytes()).encode("ascii") for frame in frames], dtype="S64")
    legal = np.asarray([sample[2] for sample in samples], dtype=np.bool_)
    return PixelData(np.concatenate((data.frames, frames)), np.concatenate((data.actions, actions)), np.concatenate((data.group_ids, groups)), np.concatenate((data.ticks, ticks)), np.concatenate((data.render_seeds, render_seeds)), np.concatenate((data.splits, np.full(len(samples), SPLITS["fit"], dtype=np.uint8))), np.concatenate((data.frame_hashes, hashes)), np.concatenate((data.sources, np.ones(len(samples), dtype=np.uint8))), np.concatenate((data.legal, legal)), data.episodes, data.config)


def _train_all(data: PixelData, device: torch.device, epochs: int = 50) -> tuple[list[PixelActor], list[dict[str, object]]]:
    actors: list[PixelActor] = []
    runs: list[dict[str, object]] = []
    for seed in TRAINING_SEEDS:
        actor, metrics = train_actor(data, seed, device, epochs=epochs)
        actors.append(actor)
        runs.append(metrics)
    return actors, runs


def _closed_loop_gate(actor: PixelActor, device: torch.device) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    teacher_completed: list[bool] = []
    for seed in range(1000, 1032):
        for side_name in ("blue", "red"):
            side: Side = side_name
            render_seed = 2_000_000 + seed * 2 + (0 if side == "blue" else 1)
            for opponent in ("null", "random"):
                rollout, _, _ = _actor_episode(actor, seed, side, opponent, render_seed, device, False)
                document = _rollout_document(rollout)
                if opponent == "random":
                    teacher = _teacher_episode(seed, side)
                    document["matched_teacher_completed"] = teacher.completed
                    teacher_completed.append(teacher.completed)
                rows.append(document)
    null_rows = [row for row in rows if row["opponent"] == "null"]
    random_rows = [row for row in rows if row["opponent"] == "random"]
    random_rate = sum(bool(row["completed"]) for row in random_rows) / len(random_rows)
    teacher_rate = sum(teacher_completed) / len(teacher_completed)
    side_rates = {side: sum(bool(row["completed"]) for row in random_rows if row["side"] == side) / sum(row["side"] == side for row in random_rows) for side in ("blue", "red")}
    total_steps = sum(cast(int, row["steps"]) for row in rows)
    raw_rate = sum(cast(int, row["raw_illegal"]) for row in rows) / total_steps
    correction_rate = sum(cast(int, row["mask_corrections"]) for row in rows) / total_steps
    passed = all(bool(row["terminal"]) for row in rows) and all(bool(row["completed"]) for row in null_rows) and random_rate >= 0.90 and random_rate >= 0.95 * teacher_rate and abs(side_rates["blue"] - side_rates["red"]) <= 0.05 and raw_rate <= 0.01 and correction_rate <= 0.01
    return {"passed": passed, "null_completion": sum(bool(row["completed"]) for row in null_rows) / len(null_rows), "random_completion": random_rate, "matched_teacher_completion": teacher_rate, "random_completion_by_side": side_rates, "raw_illegal_rate": raw_rate, "mask_correction_rate": correction_rate, "executed_illegal_actions": 0, "rows": rows}


def _mismatched_frames(frames: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mismatch = np.empty_like(frames)
    for label in range(len(ACTIONS)):
        targets = np.flatnonzero(labels == label)
        sources = np.flatnonzero(labels == (label + 1) % len(ACTIONS))
        mismatch[targets] = frames[np.resize(sources, len(targets))]
    return mismatch


def _controls(actor: PixelActor, data: PixelData, device: torch.device) -> dict[str, object]:
    selected = np.flatnonzero(data.splits == SPLITS["test"])
    frames = data.frames[selected]
    labels = data.actions[selected]
    legal = data.legal[selected]
    black = _metric_arrays(actor, np.zeros_like(frames), labels, legal, device)
    mismatch = _metric_arrays(actor, _mismatched_frames(frames, labels), labels, legal, device)
    return {"black_frames": black, "mismatched_frames": mismatch, "passed": not _classification_pass(black) and not _classification_pass(mismatch)}


def _latency(actor: PixelActor, device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        raise PixelError("formal latency gate requires CUDA")
    actor.eval()
    sample = torch.zeros((1, 3, 128, 128), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(100):
            actor(sample)
        torch.cuda.synchronize()
        elapsed: list[float] = []
        for _ in range(1000):
            start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()  # type: ignore[no-untyped-call]
            actor(sample)
            end.record()  # type: ignore[no-untyped-call]
            end.synchronize()
            elapsed.append(float(start.elapsed_time(end)))  # type: ignore[no-untyped-call]
    p95 = float(np.percentile(np.asarray(elapsed), 95))
    return {"batch": 1, "dtype": "float32", "warmup": 100, "measurements": 1000, "scope": "normalized_cuda_tensor_forward_only", "p95_ms": p95, "passed": p95 <= 10.0}


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_json(path: Path, document: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish(temporary: Path, output: Path) -> None:
    if output.exists():
        raise PixelError(f"output already exists: {output}")
    temporary.rename(output)


def _smoke() -> dict[str, object]:
    device = torch.device("cpu")
    data = collect_pixel_data(range(4), variants=1, enforce=False)
    actor, training = train_actor(data, 0, device, epochs=1, batch_size=32, patience=1)
    with tempfile.TemporaryDirectory(prefix="hok-agent-pixel-smoke-") as directory:
        root = Path(directory)
        dataset_path = root / "dataset.npz"
        model_path = root / "model.safetensors"
        write_dataset(dataset_path, data)
        loaded_data = load_dataset(dataset_path)
        save_model(model_path, actor, data.config, 0)
        loaded_actor, loaded_seed = load_model(model_path, data.config)
        logits = loaded_actor(_normalize(data.frames[:1], device))
    return {"status": "PASSED", "disposition": "NON_PROMOTING_CPU_SMOKE", "samples": len(data.frames), "dataset_arrays": sorted(loaded_data), "model_seed": loaded_seed, "logits_shape": list(logits.shape), "parameters": sum(parameter.numel() for parameter in actor.parameters()), "training": training}


def accept_pixel_v3(output: Path | None, device_name: str, smoke: bool) -> dict[str, object]:
    if smoke:
        if device_name != "cpu" or output is not None:
            raise PixelError("pixel smoke requires --device cpu and no output directory")
        return _smoke()
    if device_name != "cuda":
        raise PixelError("formal Pixel V3 acceptance requires --device cuda")
    if not torch.cuda.is_available():
        raise PixelError("CUDA is unavailable for formal Pixel V3 acceptance")
    if output is None:
        raise PixelError("formal Pixel V3 acceptance requires --output-dir")
    if output.exists():
        raise PixelError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    device = torch.device("cuda:0")
    stage = "collect"
    evidence: dict[str, object] = {}
    base = {"kind": "minimal_v3_pixel_bc_report_v1", "claim_scope": "pixelarena_engineering", "hok_capability_claim": False, "gamecore_equivalence_claim": False, "commercial_client_action_output": False, "actor_input": "rgb_only"}
    try:
        if not torch.__version__.startswith("2.5.1") or not torchvision_version.startswith("0.20.1") or torch.version.cuda != "12.1":
            raise PixelError("formal runtime requires Torch 2.5.1, torchvision 0.20.1, CUDA 12.1")
        device_identity = torch.cuda.get_device_name(0)
        if "RTX 4090" not in device_identity:
            raise PixelError(f"formal runtime requires RTX 4090, found: {device_identity}")
        static_checks = check_project()
        if not bool(static_checks["passed"]):
            raise PixelError(f"static checks failed: {static_checks['findings']}")
        evidence["static_checks"] = static_checks
        data = collect_pixel_data()
        evidence["dataset"] = dataset_summary(data)
        dataset_path = temporary / "dataset.npz"
        write_dataset(dataset_path, data)
        load_dataset(dataset_path)
        stage = "initial_train"
        actors, training = _train_all(data, device)
        if not all(bool(run["validation_passed"]) for run in training):
            raise PixelError("one or more initial models failed the validation gate")
        initial_training = [dict(run) for run in training]
        evidence["initial_training_runs"] = initial_training
        best_index = min(range(len(training)), key=lambda index: float(cast(float, cast(dict[str, object], training[index]["validation"])["cross_entropy"])))
        stage = "acquisition"
        acquisition_before, dagger_samples = _acquisition(actors[best_index], data, device)
        evidence["acquisition_before"] = acquisition_before
        dagger_rounds = 0
        if bool(acquisition_before["triggered"]):
            dagger_rounds = 1
            data = _append_dagger(data, dagger_samples)
            stage = "dagger_train"
            actors, training = _train_all(data, device)
            if not all(bool(run["validation_passed"]) for run in training):
                raise PixelError("one or more DAgger models failed the validation gate")
            best_index = min(range(len(training)), key=lambda index: float(cast(float, cast(dict[str, object], training[index]["validation"])["cross_entropy"])))
            write_dataset(dataset_path, data)
            load_dataset(dataset_path)
        acquisition_after, _ = _acquisition(actors[best_index], data, device)
        evidence["dagger_rounds"] = dagger_rounds
        evidence["acquisition_after"] = acquisition_after
        if bool(acquisition_after["triggered"]):
            raise PixelError("acquisition gate failed after the allowed DAgger disposition")
        stage = "sealed_test"
        for actor, run in zip(actors, training, strict=True):
            test_metrics = _split_metrics(actor, data, SPLITS["test"], device)
            run["test"] = test_metrics
            run["test_passed"] = _classification_pass(test_metrics)
        evidence["training_runs"] = training
        if not all(bool(run["test_passed"]) for run in training):
            raise PixelError("one or more models failed the sealed test gate")
        controls = _controls(actors[best_index], data, device)
        evidence["controls"] = controls
        if not bool(controls["passed"]):
            raise PixelError("black or mismatched frame control did not fail the main gate")
        stage = "closed_loop"
        closed_loop = [_closed_loop_gate(actor, device) for actor in actors]
        evidence["closed_loop"] = closed_loop
        if not all(bool(result["passed"]) for result in closed_loop):
            raise PixelError("one or more models failed formal closed-loop evaluation")
        stage = "latency"
        latency = _latency(actors[best_index], device)
        evidence["latency"] = latency
        if not bool(latency["passed"]):
            raise PixelError("promoted model failed the CUDA latency gate")
        stage = "models"
        model_hashes: dict[str, str] = {}
        for actor, seed in zip(actors, TRAINING_SEEDS, strict=True):
            path = temporary / f"model-seed-{seed}.safetensors"
            save_model(path, actor, data.config, seed)
            load_model(path, data.config)
            model_hashes[path.name] = _sha(path)
        best_model = f"model-seed-{TRAINING_SEEDS[best_index]}.safetensors"
        stage = "report"
        report = {**base, "status": "PASSED", "environment_identity": data.config.identity, "arena_config_hash": data.config.digest, "renderer": {"identity": RENDERER_ID, "hash": RENDERER_HASH}, "action_vocabulary": [list(action) for action in ACTIONS], "action_vocabulary_hash": ACTION_HASH, "dataset": dataset_summary(data), "training_contract": {"hash": TRAINING_HASH, **TRAINING_CONFIG}, "initial_training_runs": initial_training, "training_runs": training, "best_model": best_model, "dagger": {"rounds": dagger_rounds, "acquisition_before": acquisition_before, "acquisition_after": acquisition_after}, "controls": controls, "closed_loop": closed_loop, "latency": latency, "runtime": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision_version, "cuda_runtime": torch.version.cuda, "device": device_identity, "platform": platform.platform()}, "files": {"dataset.npz": _sha(dataset_path), **model_hashes}, "static_checks": static_checks}
        _write_json(temporary / "report.json", report)
        _publish(temporary, output)
        return report
    except Exception as exc:
        for child in temporary.iterdir():
            if child.is_file():
                child.unlink()
        failure = {**base, "status": "FAILED", "stage": stage, "error_type": type(exc).__name__, "error": str(exc), "python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision_version, "evidence": evidence}
        _write_json(temporary / "report.json", failure)
        _publish(temporary, output)
        raise PixelError(f"Pixel V3 failed at {stage}; report: {output / 'report.json'}") from exc
