from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from hok_agent.hierarchical_p1v2_movement_branch import MovementBranch
from hok_agent.movement_mvp import (
    MOVEMENT_ACTIONS,
    StageAMovement,
    mark_visible_target,
    rgb_geometry_movement,
    rule_movement_in_range,
    stage_c_arena,
    to_arena_action,
)
from hok_agent.rich_arena import wait_action
from hok_agent.rich_renderer import render


class TaskSpecificMovement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.project = nn.Linear(64 * 8 * 8, 128)
        self.temporal = nn.GRU(128, 128, batch_first=True)
        self.head = nn.Linear(128, len(MOVEMENT_ACTIONS))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels, height, width = clips.shape
        features = self.spatial(clips.reshape(batch * sequence, channels, height, width))
        features = self.project(features.flatten(1)).reshape(batch, sequence, 128)
        _output, hidden = self.temporal(features)
        return cast(torch.Tensor, self.head(hidden[-1]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(clips: torch.Tensor, device: torch.device) -> torch.Tensor:
    return clips.to(device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 127.5 - 1.0


def train_step(
    model: nn.Module,
    clips: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    freeze_batch_norm: bool = False,
) -> tuple[float, float]:
    model.train()
    if isinstance(model, MovementBranch):
        model.freeze_shared_trunk()
    if freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    logits = model(clips)
    loss = nn.functional.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    gradient_squares = [
        torch.sum(parameter.grad.detach() ** 2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = float(torch.sqrt(torch.stack(gradient_squares).sum()))
    optimizer.step()
    return float(loss.detach()), gradient_norm


def _evaluate(
    model: MovementBranch | TaskSpecificMovement,
    clips: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
    *,
    training_mode: bool,
    freeze_batch_norm: bool,
) -> tuple[float, float, list[float]]:
    saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.train(training_mode)
    if training_mode and isinstance(model, MovementBranch):
        model.freeze_shared_trunk()
    if training_mode and freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(clips), batch_size):
            logits_parts.append(model(_batch(clips[start : start + batch_size], device)).cpu())
    logits = torch.cat(logits_parts)
    loss = float(nn.functional.cross_entropy(logits, labels))
    predicted = logits.argmax(1)
    accuracy = float((predicted == labels).float().mean())
    recalls = []
    for label in range(len(MOVEMENT_ACTIONS)):
        selected = labels == label
        recalls.append(float((predicted[selected] == label).float().mean()))
    model.load_state_dict(saved)
    model.eval()
    return accuracy, loss, recalls


def run_overfit32(
    config_path: Path,
    dataset_path: Path,
    representation_path: Path | None,
    output_dir: Path,
    *,
    device_name: str,
    freeze_batch_norm: bool = False,
    architecture: str = "task-specific",
) -> dict[str, object]:
    raw = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
    stage = cast(dict[str, object], raw["stage_b"])
    batch_size = int(cast(int, stage["batch_size"]))
    maximum_updates = int(cast(int, stage["maximum_updates"]))
    learning_rate = float(cast(float, stage["learning_rate"]))
    weight_decay = float(cast(float, stage["weight_decay"]))
    minimum_accuracy = float(cast(float, stage["minimum_eval_accuracy"]))
    maximum_loss = float(cast(float, stage["maximum_eval_loss"]))
    attempt_limit = int(cast(int, stage["diagnostic_attempt_limit"]))
    attempts_used = int(cast(int, stage["diagnostic_attempts_used"]))
    if attempts_used >= attempt_limit:
        raise ValueError("stage B diagnostic attempt limit is exhausted")
    seed = int(cast(int, raw["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")

    with np.load(dataset_path, allow_pickle=False) as data:
        clips = torch.from_numpy(data["rgb_sequence"].copy())
        labels = torch.from_numpy(data["label"].astype(np.int64))
    if clips.shape != (32, 16, 128, 128, 3) or sorted(labels.tolist()) != [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        *[label for label in range(1, 9) for _ in range(3)],
    ]:
        raise ValueError("stage B requires the frozen 32-sample nine-action dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "diagnostic-last.safetensors"
    report_path = output_dir / "report.json"
    if checkpoint_path.exists() or report_path.exists():
        raise ValueError("stage B diagnostic output already exists")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if architecture == "p0-branch":
        if representation_path is None:
            raise ValueError("p0-branch explicitly requires a representation checkpoint")
        representation_sha256 = _sha256(representation_path)
        model: MovementBranch | TaskSpecificMovement = MovementBranch(
            load_file(representation_path, device="cpu"), output_actions=len(MOVEMENT_ACTIONS)
        ).to(device)
    elif architecture == "task-specific":
        representation_sha256 = "not_used"
        model = TaskSpecificMovement().to(device)
    else:
        raise ValueError("unknown Movement MVP diagnostic architecture")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    order_generator = torch.Generator().manual_seed(seed)
    initial = parameters[-1].detach().clone()
    first_update_before = [parameter.detach().clone() for parameter in parameters]
    first_update_loss = math.nan
    first_update_gradient_norm = math.nan
    first_update_parameter_changed = False
    first_update_finite = False
    losses, gradient_norms = [], []
    started = time.monotonic()
    for update in range(maximum_updates):
        if update % (len(clips) // batch_size) == 0:
            order = torch.randperm(len(clips), generator=order_generator)
        start = update % (len(clips) // batch_size) * batch_size
        selected = order[start : start + batch_size]
        loss, gradient_norm = train_step(
            model,
            _batch(clips[selected], device),
            labels[selected].to(device),
            optimizer,
            freeze_batch_norm=freeze_batch_norm,
        )
        losses.append(loss)
        gradient_norms.append(gradient_norm)
        if update == 0:
            first_update_loss = loss
            first_update_gradient_norm = gradient_norm
            first_update_parameter_changed = any(
                not torch.equal(before, parameter.detach())
                for before, parameter in zip(first_update_before, parameters, strict=True)
            )
            first_update_finite = (
                math.isfinite(loss) and math.isfinite(gradient_norm) and gradient_norm > 0.0
            )
    elapsed = time.monotonic() - started
    first_parameter_changed = not torch.equal(initial, parameters[-1].detach())
    train_accuracy, train_loss, train_recalls = _evaluate(
        model,
        clips,
        labels,
        device,
        batch_size,
        training_mode=True,
        freeze_batch_norm=freeze_batch_norm,
    )
    eval_accuracy, eval_loss, eval_recalls = _evaluate(
        model,
        clips,
        labels,
        device,
        batch_size,
        training_mode=False,
        freeze_batch_norm=freeze_batch_norm,
    )
    passed = (
        first_parameter_changed
        and first_update_parameter_changed
        and first_update_finite
        and eval_accuracy >= minimum_accuracy
        and eval_loss <= maximum_loss
        and min(eval_recalls) > 0.0
    )
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        checkpoint_path,
        metadata={
            "purpose": "diagnostic_only",
            "dataset_sha256": _sha256(dataset_path),
            "representation_sha256": representation_sha256,
        },
    )
    report: dict[str, object] = {
        "status": "PASSED" if passed else "FAILED",
        "schema_version": "movement-mvp-stage-b-overfit32-report-v0",
        "full_training_called": False,
        "diagnostic_checkpoint_only": True,
        "dataset_sha256": _sha256(dataset_path),
        "representation_sha256": representation_sha256,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "device": str(device),
        "normalization_mode": (
            "group_norm"
            if architecture == "task-specific"
            else "frozen_batch_norm"
            if freeze_batch_norm
            else "train_batch_norm"
        ),
        "architecture": architecture,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "updates": maximum_updates,
        "elapsed_seconds": elapsed,
        "first_parameter_changed": first_parameter_changed,
        "first_update_loss": first_update_loss,
        "first_update_gradient_norm": first_update_gradient_norm,
        "first_update_parameter_changed": first_update_parameter_changed,
        "first_update_finite": first_update_finite,
        "initial_loss": losses[0],
        "final_update_loss": losses[-1],
        "maximum_gradient_norm": max(gradient_norms),
        "train_mode_accuracy": train_accuracy,
        "train_mode_loss": train_loss,
        "eval_accuracy": eval_accuracy,
        "eval_loss": eval_loss,
        "eval_recall": dict(zip(MOVEMENT_ACTIONS, eval_recalls, strict=True)),
        "minimum_eval_accuracy": minimum_accuracy,
        "maximum_eval_loss": maximum_loss,
        "passed": passed,
        "next_stage_allowed": passed,
        "diagnostic_attempt_limit": attempt_limit,
        "diagnostic_attempts_used_before_run": attempts_used,
        "diagnostic_checkpoint_reusable_for_formal_training": False,
        "formal_training_initialization": "fresh-seed-0",
        "input_commands_sent": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_stage_c_manifest(dataset_root: Path) -> dict[str, object]:
    payload = cast(
        dict[str, object], json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    )
    supplied = str(payload.pop("manifest_sha256"))
    if payload.get(
        "schema_version"
    ) != "movement-mvp-stage-c-trajectories-v0" or supplied != _canonical_sha256(payload):
        raise ValueError("stage C manifest binding differs")
    payload["manifest_sha256"] = supplied
    return payload


class TrajectoryWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset_root: Path, split: str) -> None:
        manifest = load_stage_c_manifest(dataset_root)
        sequence_frames = int(cast(int, manifest["sequence_frames"]))
        rows = cast(list[dict[str, object]], manifest["episodes"])
        self.sequence_frames = sequence_frames
        self.episodes: list[tuple[np.ndarray, np.ndarray]] = []
        self.references: list[tuple[int, int]] = []
        for row in rows:
            if row["split"] != split:
                continue
            path = dataset_root / "episodes" / str(row["basename"])
            if _sha256(path) != row["artifact_sha256"]:
                raise ValueError("stage C episode artifact binding differs")
            with np.load(path, allow_pickle=False) as data:
                frames = data["frames"].copy()
                labels = data["labels"].astype(np.int64)
                timestamps = data["frame_timestamps_ms"]
                window_end = data["window_end"]
            if (
                len(frames) != len(labels)
                or window_end.ndim != 1
                or len(window_end) == 0
                or window_end[0] < 0
                or window_end[-1] >= len(labels)
                or np.any(np.diff(window_end) <= 0)
                or not np.array_equal(
                    timestamps,
                    np.arange(len(labels)) * int(cast(int, manifest["step_duration_ms"])),
                )
            ):
                raise ValueError("stage C causal episode arrays differ")
            episode_index = len(self.episodes)
            self.episodes.append((frames, labels))
            self.references.extend((episode_index, int(end)) for end in window_end)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode_index, end = self.references[index]
        frames, labels = self.episodes[episode_index]
        start = max(0, end - self.sequence_frames + 1)
        clip = frames[start : end + 1]
        if len(clip) < self.sequence_frames:
            padding = np.repeat(clip[:1], self.sequence_frames - len(clip), axis=0)
            clip = np.concatenate((padding, clip), axis=0)
        return torch.from_numpy(clip.copy()), torch.tensor(labels[end], dtype=torch.int64)


def train_stage_c_candidate(
    config_path: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    device_name: str,
    sampling: str = "uniform",
) -> dict[str, object]:
    if sampling not in ("uniform", "class-balanced"):
        raise ValueError("unknown stage C training sampler")
    raw = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
    stage = cast(dict[str, object], raw["stage_c"])
    if cast(dict[str, object], raw["stage_b"])["selected_architecture"] != "task-specific":
        raise ValueError("stage C requires the selected task-specific architecture")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "train-report.json"
    if report_path.exists() or any(output_dir.glob("*.safetensors")):
        raise ValueError("stage C training output already exists")
    manifest = load_stage_c_manifest(dataset_root)
    if bool(stage.get("navigation_only", False)) != bool(manifest.get("navigation_only", False)):
        raise ValueError("stage C training and dataset environments differ")
    train_rows = [
        row
        for row in cast(list[dict[str, object]], manifest["episodes"])
        if row["split"] == "train"
    ]
    if len(train_rows) != int(cast(int, stage["train_episodes"])):
        raise ValueError("stage C train episode count differs")

    config_hash = _sha256(config_path)
    contract_path = output_dir / "training-contract.json"
    if contract_path.exists():
        raise ValueError("stage C training contract already exists")
    contract = {
        "config": raw,
        "config_sha256": config_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": _sha256(Path(__file__)),
        "movement_source_sha256": _sha256(Path(__file__).with_name("movement_mvp.py")),
        "normalization": "uint8-rgb/127.5-1-fp32",
        "action_order": list(MOVEMENT_ACTIONS),
        "fresh_initialization": True,
        "sampling": sampling,
    }
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    seed = int(cast(int, raw["seed"]))
    epochs = int(cast(int, stage["epochs"]))
    batch_size = int(cast(int, stage["batch_size"]))
    checkpoint_epochs = tuple(int(value) for value in cast(list[int], stage["checkpoint_epochs"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = TaskSpecificMovement().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cast(float, stage["learning_rate"])),
        weight_decay=float(cast(float, stage["weight_decay"])),
    )
    dataset = TrajectoryWindowDataset(dataset_root, "train")
    dataset_labels = torch.tensor(
        [int(dataset.episodes[episode][1][end]) for episode, end in dataset.references]
    )
    label_counts = torch.bincount(dataset_labels, minlength=len(MOVEMENT_ACTIONS))
    sampler = (
        WeightedRandomSampler(
            1.0 / label_counts[dataset_labels].double(),
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        if sampling == "class-balanced"
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    first_before = [parameter.detach().clone() for parameter in model.parameters()]
    first_update: dict[str, object] | None = None
    epoch_losses: list[float] = []
    checkpoints: list[dict[str, object]] = []
    sampled_label_counts = torch.zeros(len(MOVEMENT_ACTIONS), dtype=torch.int64)
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        losses = []
        for clips, labels in loader:
            sampled_label_counts += torch.bincount(labels, minlength=len(MOVEMENT_ACTIONS))
            loss, gradient_norm = train_step(
                model, _batch(clips, device), labels.to(device), optimizer
            )
            losses.append(loss)
            if first_update is None:
                changed = any(
                    not torch.equal(before, parameter.detach())
                    for before, parameter in zip(first_before, model.parameters(), strict=True)
                )
                first_update = {
                    "loss": loss,
                    "gradient_norm": gradient_norm,
                    "finite": math.isfinite(loss)
                    and math.isfinite(gradient_norm)
                    and gradient_norm > 0.0,
                    "parameter_changed": changed,
                }
        epoch_losses.append(sum(losses) / len(losses))
        if epoch in checkpoint_epochs:
            path = output_dir / f"epoch-{epoch:03d}.safetensors"
            save_file(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                path,
                metadata={
                    "purpose": "formal-simulator-bc-candidate",
                    "fresh_initialization": "true",
                    "manifest_sha256": str(manifest["manifest_sha256"]),
                    "config_sha256": config_hash,
                    "training_contract_sha256": _sha256(contract_path),
                    "sampling": sampling,
                    "epoch": str(epoch),
                },
            )
            checkpoints.append({"epoch": epoch, "basename": path.name, "sha256": _sha256(path)})
    elapsed = time.monotonic() - started
    if first_update is None or not first_update["finite"] or not first_update["parameter_changed"]:
        raise ValueError("stage C first update gate failed")
    report: dict[str, object] = {
        "status": "TRAINED_NOT_EVALUATED",
        "schema_version": "movement-mvp-stage-c-train-report-v0",
        "architecture": "task-specific-groupnorm-gru",
        "fresh_initialization": True,
        "diagnostic_checkpoint_loaded": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": config_hash,
        "training_contract_sha256": _sha256(contract_path),
        "train_episodes": len(train_rows),
        "train_windows": len(dataset),
        "sampling": sampling,
        "source_label_counts": dict(zip(MOVEMENT_ACTIONS, label_counts.tolist(), strict=True)),
        "sampled_label_counts": dict(
            zip(MOVEMENT_ACTIONS, sampled_label_counts.tolist(), strict=True)
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "epoch_losses": epoch_losses,
        "first_update": first_update,
        "checkpoints": checkpoints,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "dev_rollout_called_during_training": False,
        "simulator_only": True,
        "input_commands_sent": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


_OPPOSITE = {
    "N": "S",
    "S": "N",
    "W": "E",
    "E": "W",
    "NW": "SE",
    "SE": "NW",
    "NE": "SW",
    "SW": "NE",
}


def _window(frames: Sequence[np.ndarray], length: int) -> torch.Tensor:
    selected = list(frames[-length:])
    if len(selected) < length:
        selected = [selected[0]] * (length - len(selected)) + selected
    return torch.from_numpy(np.stack(selected)[None])


def _integer(value: object) -> int:
    return int(cast(int, value))


def _number(value: object) -> float:
    return float(cast(float, value))


def _aggregate_rollouts(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    movement_requests = sum(_integer(row["movement_requests"]) for row in rows)
    comparable_pairs = sum(_integer(row["comparable_pairs"]) for row in rows)
    successes = sum(row["status"] == "success" for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "timeouts": sum(row["status"] == "timeout" for row in rows),
        "runtime_errors": sum(row["status"] == "runtime_error" for row in rows),
        "collision_fraction": (
            sum(_integer(row["collisions"]) for row in rows) / movement_requests
            if movement_requests
            else 0.0
        ),
        "oscillation_fraction": (
            sum(_integer(row["oscillations"]) for row in rows) / comparable_pairs
            if comparable_pairs
            else 0.0
        ),
        "mean_steps": sum(_integer(row["steps"]) for row in rows) / len(rows),
        "episode_results": list(rows),
    }


def _rollout(
    scenario: dict[str, object],
    policy: str,
    maximum_steps: int,
    sequence_frames: int,
    random_seed: int,
    model: TaskSpecificMovement | None = None,
    device: torch.device | None = None,
    *,
    stop_confirmation_steps: int = 1,
    record_trace: bool = False,
    navigation_only: bool = False,
) -> dict[str, object]:
    goal = cast(tuple[int, int], tuple(cast(list[int], scenario["goal"])))
    render_seed = int(cast(int, scenario["render_seed"]))
    arena = stage_c_arena(scenario, maximum_steps, navigation_only)
    rng = Random(random_seed + render_seed)
    frames: list[np.ndarray] = []
    previous: StageAMovement = "STOP"
    collisions = oscillations = comparable_pairs = movement_requests = 0
    path_length = 0.0
    stop_streak = 0
    trace: list[dict[str, object]] = []
    error_reason: str | None = None
    try:
        for step in range(maximum_steps):
            observation = arena.observe("blue")
            marked = mark_visible_target(render(observation, render_seed), "opponent_hero")
            frames.append(marked)
            position_raw = cast(dict[str, int], observation["self_position"])
            before = (position_raw["x"], position_raw["y"])
            legal = arena.legal_actions("blue")
            legal_names = tuple(
                action for action in MOVEMENT_ACTIONS if to_arena_action(action) in legal
            )
            requested: StageAMovement | None = None
            if policy == "teacher":
                action = rule_movement_in_range(before, goal)
            elif policy == "geometry":
                action = rgb_geometry_movement(marked)
            elif policy == "fixed_east":
                action = "E" if "E" in legal_names else "STOP"
            elif policy == "random":
                action = rng.choice(legal_names)
            elif policy == "learned" and model is not None and device is not None:
                with torch.no_grad():
                    logits = model(_batch(_window(frames, sequence_frames), device))[0]
                requested = MOVEMENT_ACTIONS[int(logits.argmax())]
                mask = torch.full_like(logits, -torch.inf)
                for name in legal_names:
                    mask[MOVEMENT_ACTIONS.index(name)] = logits[MOVEMENT_ACTIONS.index(name)]
                action = MOVEMENT_ACTIONS[int(mask.argmax())]
            else:
                raise ValueError("unknown stage C rollout policy")
            arena.step(to_arena_action(action), wait_action())
            after_raw = cast(dict[str, int], arena.observe("blue")["self_position"])
            after = (after_raw["x"], after_raw["y"])
            if record_trace:
                trace.append(
                    {
                        "step": step,
                        "before": list(before),
                        "after": list(after),
                        "requested_action": requested if requested is not None else action,
                        "executed_action": action,
                        "teacher_action": rule_movement_in_range(before, goal),
                        "self_health": observation["self_health"],
                    }
                )
            if action != "STOP":
                movement_requests += 1
                collisions += int(after == before)
            delta_x, delta_y = after[0] - before[0], after[1] - before[1]
            path_length += math.sqrt(delta_x**2 + delta_y**2)
            if previous != "STOP" and action != "STOP":
                comparable_pairs += 1
                oscillations += int(_OPPOSITE.get(previous) == action)
            previous = action
            stopped_at_goal = (
                action == "STOP"
                and abs(goal[0] - before[0]) + abs(goal[1] - before[1]) <= 1
                and before == after
                and (stop_confirmation_steps == 1 or _integer(observation["self_health"]) > 0)
            )
            stop_streak = stop_streak + 1 if stopped_at_goal else 0
            if stop_streak >= stop_confirmation_steps:
                return {
                    "scenario_id": scenario["scenario_id"],
                    "status": "success",
                    "steps": step + 1,
                    "collisions": collisions,
                    "movement_requests": movement_requests,
                    "oscillations": oscillations,
                    "comparable_pairs": comparable_pairs,
                    "path_length": path_length,
                    "stop_streak": stop_streak,
                    **({"trace": trace} if record_trace else {}),
                }
        status = "timeout"
    except ValueError as exc:
        status = "runtime_error"
        error_reason = str(exc)
    return {
        "scenario_id": scenario["scenario_id"],
        "status": status,
        "steps": maximum_steps,
        "collisions": collisions,
        "movement_requests": movement_requests,
        "oscillations": oscillations,
        "comparable_pairs": comparable_pairs,
        "path_length": path_length,
        "stop_streak": stop_streak,
        "error_reason": error_reason,
        **({"trace": trace} if record_trace else {}),
    }


def stage_c_dev_gates(
    stage: dict[str, object],
    baselines: dict[str, dict[str, object]],
    selected: dict[str, object],
) -> dict[str, bool]:
    better_simple = max(_integer(baselines[name]["successes"]) for name in ("fixed_east", "random"))
    gates = {
        "teacher_success": _integer(baselines["teacher"]["successes"])
        >= _integer(stage["minimum_teacher_successes"]),
        "learned_success": _integer(selected["successes"])
        >= _integer(stage["minimum_learned_successes"]),
        "collision": _number(selected["collision_fraction"])
        <= _number(stage["maximum_collision_fraction"]),
        "oscillation": _number(selected["oscillation_fraction"])
        <= _number(stage["maximum_oscillation_fraction"]),
    }
    if stage.get("dev_contract_version") == "movement-mvp-dev-v2":
        best_simple_cost = min(
            _number(baselines[name]["mean_steps"]) for name in ("fixed_east", "random")
        )
        gates["simple_success_noninferior"] = _integer(selected["successes"]) >= better_simple
        gates["failure_inclusive_step_cost"] = _number(selected["mean_steps"]) <= (
            best_simple_cost * _number(stage["maximum_step_cost_ratio_vs_simple"])
        )
    else:
        gates["gain_over_simple"] = _integer(selected["successes"]) - better_simple >= _integer(
            stage["minimum_gain_over_random_or_fixed"]
        )
    return gates


def evaluate_stage_c_dev(
    config_path: Path,
    dataset_root: Path,
    checkpoints: Sequence[Path],
    output_dir: Path,
    *,
    device_name: str,
    reference_only: bool = False,
) -> dict[str, object]:
    raw = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
    stage = cast(dict[str, object], raw["stage_c"])
    manifest = load_stage_c_manifest(dataset_root)
    scenarios = tuple(
        row for row in cast(list[dict[str, object]], manifest["episodes"]) if row["split"] == "dev"
    )
    if len(scenarios) != int(cast(int, stage["dev_episodes"])) or not checkpoints:
        raise ValueError("stage C dev evaluation inputs differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dev-report.json"
    if report_path.exists():
        raise ValueError("stage C dev evaluation output already exists")
    maximum_steps = int(cast(int, stage["maximum_episode_steps"]))
    sequence_frames = int(cast(int, stage["sequence_frames"]))
    random_seed = int(cast(int, stage["random_baseline_seed"]))
    v2 = stage.get("dev_contract_version") == "movement-mvp-dev-v2"
    stop_confirmation_steps = _integer(stage.get("stop_confirmation_steps", 1))
    navigation_only = bool(stage.get("navigation_only", False))
    config_hash = _sha256(config_path)
    device = torch.device(device_name)
    started = time.monotonic()
    baselines = {
        name: _aggregate_rollouts(
            [
                _rollout(
                    scenario,
                    name,
                    maximum_steps,
                    sequence_frames,
                    random_seed,
                    stop_confirmation_steps=stop_confirmation_steps,
                    navigation_only=navigation_only,
                )
                for scenario in scenarios
            ]
        )
        for name in ("teacher", "geometry", "fixed_east", "random")
    }
    candidates: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        if v2 and not reference_only:
            with safe_open(checkpoint, framework="pt", device="cpu") as saved:
                metadata = saved.metadata() or {}
            if (
                metadata.get("manifest_sha256") != manifest["manifest_sha256"]
                or metadata.get("config_sha256") != config_hash
            ):
                raise ValueError("stage C checkpoint contract binding differs")
        model = TaskSpecificMovement().to(device)
        model.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
        model.eval()
        metrics = _aggregate_rollouts(
            [
                _rollout(
                    scenario,
                    "learned",
                    maximum_steps,
                    sequence_frames,
                    random_seed,
                    model,
                    device,
                    stop_confirmation_steps=stop_confirmation_steps,
                    record_trace=v2,
                    navigation_only=navigation_only,
                )
                for scenario in scenarios
            ]
        )
        candidates.append({"basename": checkpoint.name, "sha256": _sha256(checkpoint), **metrics})
    selected = max(
        candidates,
        key=lambda row: (
            _integer(row["successes"]),
            -_number(row["collision_fraction"]),
            -_number(row["mean_steps"]),
        ),
    )
    gates = stage_c_dev_gates(stage, baselines, selected)
    passed = all(gates.values()) and not reference_only
    report: dict[str, object] = {
        "status": "PASSED" if passed else "FAILED",
        "schema_version": "movement-mvp-stage-c-dev-report-v0",
        "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": config_hash,
        "evaluation_contract": stage,
        "reference_only": reference_only,
        "gate_results": gates,
        "elapsed_seconds": time.monotonic() - started,
        "dev_episodes": len(scenarios),
        "baselines": baselines,
        "candidates": candidates,
        "selected_checkpoint": selected["basename"],
        "selected_checkpoint_sha256": selected["sha256"],
        "passed": passed,
        "holdout_opened": False,
        "teacher_fallback_used": False,
        "simulator_only": True,
        "input_commands_sent": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
