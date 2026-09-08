from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
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
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

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
    def __init__(self, output_actions: int = len(MOVEMENT_ACTIONS)) -> None:
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
        self.head = nn.Linear(128, output_actions)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels, height, width = clips.shape
        features = self.spatial(clips.reshape(batch * sequence, channels, height, width))
        features = self.project(features.flatten(1)).reshape(batch, sequence, 128)
        _output, hidden = self.temporal(features)
        return cast(torch.Tensor, self.head(hidden[-1]))


class RelationalMovement(nn.Module):
    """Learn two spatial slots and classify their relative motion with no coordinate labels."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.attention = nn.Conv2d(32, 2, 1)
        self.project = nn.Sequential(nn.Linear(40, 64), nn.GELU())
        self.temporal = nn.GRU(64, 128, batch_first=True)
        self.head = nn.Linear(128, len(MOVEMENT_ACTIONS))

    def forward_with_slots(
        self, clips: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, channels, height, width = clips.shape
        features = self.spatial(clips.reshape(batch * sequence, channels, height, width))
        attention_logits = self.attention(features).flatten(2)
        attention = torch.softmax(attention_logits, dim=-1)
        axis_y = torch.linspace(-1.0, 1.0, features.shape[-2], device=features.device)
        axis_x = torch.linspace(-1.0, 1.0, features.shape[-1], device=features.device)
        grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
        grid = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
        coordinates = attention @ grid
        relation = coordinates[:, 0] - coordinates[:, 1]
        confidence = attention.max(dim=-1).values
        pooled = features.mean(dim=(2, 3))
        frame_features = torch.cat((pooled, coordinates.flatten(1), relation, confidence), dim=-1)
        encoded = self.project(frame_features).reshape(batch, sequence, 64)
        _output, hidden = self.temporal(encoded)
        return (
            cast(torch.Tensor, self.head(hidden[-1])),
            attention_logits.reshape(batch, sequence, 2, -1),
            coordinates.reshape(batch, sequence, 2, 2),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        logits, _attention, _coordinates = self.forward_with_slots(clips)
        return logits


MovementModel = MovementBranch | TaskSpecificMovement | RelationalMovement


def _movement_model(architecture: str, device: torch.device) -> MovementModel:
    if architecture == "task-specific":
        return TaskSpecificMovement().to(device)
    if architecture == "relational":
        return RelationalMovement().to(device)
    raise ValueError("unknown task-specific Movement architecture")


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
    model: MovementModel,
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
    real_counterfactual = (
        raw.get("schema_version") == "movement-real-counterfactual-overfit32-train-contract-v1"
    )
    if real_counterfactual:
        unsigned = {key: value for key, value in raw.items() if key != "contract_sha256"}
        if (
            raw.get("contract_sha256") != _canonical_sha256(unsigned)
            or raw.get("overfit_dataset_sha256") != _sha256(dataset_path)
            or raw.get("test_allowed") is not False
            or raw.get("formal_training_allowed") is not False
            or raw.get("r2_allowed") is not False
            or raw.get("device_input_allowed") is not False
            or raw.get("action_order") != list(MOVEMENT_ACTIONS)
            or stage.get("selected_architecture") != architecture
            or raw.get("labels_are_geometric_counterfactuals") is not True
            or raw.get("labels_are_executed_actions") is not False
        ):
            raise ValueError("real counterfactual training contract differs")
    if architecture == "relational":
        unsigned = {key: value for key, value in raw.items() if key != "contract_sha256"}
        if (
            raw.get("schema_version") != "movement-goal-canvas-relational-contract-v1"
            or raw.get("contract_sha256") != _canonical_sha256(unsigned)
            or raw.get("overfit_dataset_sha256") != _sha256(dataset_path)
            or raw.get("data_changed") is not False
            or raw.get("epochs_changed") is not False
            or raw.get("sampling_changed") is not False
        ):
            raise ValueError("relational Movement contract binding differs")
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
        model: MovementModel = MovementBranch(
            load_file(representation_path, device="cpu"), output_actions=len(MOVEMENT_ACTIONS)
        ).to(device)
    elif architecture == "task-specific":
        representation_sha256 = "not_used"
        model = TaskSpecificMovement().to(device)
    elif architecture == "relational":
        representation_sha256 = "not_used"
        model = RelationalMovement().to(device)
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
            "architecture": architecture,
        },
    )
    report: dict[str, object] = {
        "status": "PASSED" if passed else "FAILED",
        "schema_version": "movement-mvp-stage-b-overfit32-report-v0",
        "full_training_called": False,
        "diagnostic_checkpoint_only": True,
        "dataset_sha256": _sha256(dataset_path),
        "config_sha256": _sha256(config_path),
        "representation_sha256": representation_sha256,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "device": str(device),
        "normalization_mode": (
            "group_norm"
            if architecture in {"task-specific", "relational"}
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
        "next_stage_allowed": passed and not real_counterfactual,
        "real_counterfactual_diagnostic": real_counterfactual,
        "source_window_generalization_verified": False if real_counterfactual else None,
        "formal_training_allowed": False if real_counterfactual else None,
        "diagnostic_attempt_limit": attempt_limit,
        "diagnostic_attempts_used_before_run": attempts_used,
        "diagnostic_checkpoint_reusable_for_formal_training": False,
        "formal_training_initialization": "fresh-seed-0",
        "input_commands_sent": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_joystick_overfit32(
    config_path: Path, dataset_dir: Path, output_dir: Path, *, device_name: str,
) -> dict[str, object]:
    from hok_agent.movement_real_rgb import validate_joystick_overfit32

    config = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
    unsigned = {key: value for key, value in config.items() if key != "contract_sha256"}
    training = cast(dict[str, object], config["training"])
    dataset_path = dataset_dir / "joystick-overfit32.npz"
    manifest_path = dataset_dir / "manifest.json"
    conclusion_path = dataset_dir / "conclusion.json"
    conclusion = cast(dict[str, object], json.loads(conclusion_path.read_text(encoding="utf-8")))
    expected_training = {
        "architecture": "task_specific_groupnorm_gru_686281",
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0,
        "batch_size": 8, "maximum_updates": 200, "precision": "fp32",
        "minimum_eval_accuracy": 0.95, "maximum_eval_loss": 0.05,
    }
    if (
        config.get("schema_version") != "joystick-overfit32-train-contract-v1"
        or config.get("contract_sha256") != _canonical_sha256(unsigned)
        or config.get("dataset_sha256") != _sha256(dataset_path)
        or config.get("manifest_file_sha256") != _sha256(manifest_path)
        or config.get("conclusion_file_sha256") != _sha256(conclusion_path)
        or config.get("action_order") != list(MOVEMENT_ACTIONS)
        or training != expected_training
        or config.get("attempt_limit") != 1
        or config.get("diagnostic_only") is not True
        or config.get("formal_training_allowed") is not False
        or config.get("checkpoint_promotion_allowed") is not False
        or config.get("dev_allowed") is not False
        or config.get("test_allowed") is not False
        or config.get("device_input_allowed") is not False
        or conclusion.get("diagnostic_overfit_allowed") is not True
        or conclusion.get("formal_training_allowed") is not False
    ):
        raise ValueError("joystick overfit32 contract differs")
    validate_joystick_overfit32(dataset_dir)
    if output_dir.exists():
        raise ValueError("joystick overfit32 output exists")
    with np.load(dataset_path, allow_pickle=False) as data:
        clips = torch.from_numpy(data["rgb"].copy())
        labels = torch.from_numpy(data["label"].astype(np.int64))
    if clips.shape != (32, 16, 128, 128, 3) or labels.shape != (32,):
        raise ValueError("joystick overfit32 arrays differ")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    seed = cast(int, config["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = TaskSpecificMovement().to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=cast(float, training["learning_rate"]),
        weight_decay=cast(float, training["weight_decay"]),
    )
    batch_size = cast(int, training["batch_size"])
    maximum_updates = cast(int, training["maximum_updates"])
    generator = torch.Generator().manual_seed(seed)
    first_before = [parameter.detach().clone() for parameter in parameters]
    first_loss = math.nan
    first_gradient = math.nan
    first_changed = False
    losses: list[float] = []
    gradients: list[float] = []
    started = time.monotonic()
    for update in range(maximum_updates):
        if update % (len(clips) // batch_size) == 0:
            order = torch.randperm(len(clips), generator=generator)
        offset = update % (len(clips) // batch_size) * batch_size
        selected = order[offset:offset + batch_size]
        loss, gradient = train_step(
            model, _batch(clips[selected], device), labels[selected].to(device), optimizer
        )
        losses.append(loss)
        gradients.append(gradient)
        if update == 0:
            first_loss, first_gradient = loss, gradient
            first_changed = any(
                not torch.equal(before, parameter.detach())
                for before, parameter in zip(first_before, parameters, strict=True)
            )
    elapsed = time.monotonic() - started
    accuracy, loss, recalls = _evaluate(
        model, clips, labels, device, batch_size,
        training_mode=False, freeze_batch_norm=False,
    )
    minimum_accuracy = cast(float, training["minimum_eval_accuracy"])
    maximum_loss = cast(float, training["maximum_eval_loss"])
    passed = (
        first_changed and math.isfinite(first_loss) and first_gradient > 0
        and accuracy >= minimum_accuracy and loss <= maximum_loss and min(recalls) > 0
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-overfit32-", dir=output_dir.parent))
    checkpoint = staging / "diagnostic-last.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint,
        metadata={"purpose": "diagnostic_only", "dataset_sha256": _sha256(dataset_path),
                  "contract_sha256": cast(str, config["contract_sha256"]),
                  "architecture": "task-specific"},
    )
    report: dict[str, object] = {
        "schema_version": "joystick-overfit32-report-v1",
        "status": "PASSED" if passed else "FAILED", "passed": passed,
        "contract_sha256": config["contract_sha256"],
        "config_file_sha256": _sha256(config_path),
        "dataset_sha256": _sha256(dataset_path),
        "manifest_file_sha256": _sha256(manifest_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "diagnostic_checkpoint_only": True, "checkpoint_promotion_allowed": False,
        "formal_training_allowed": False, "next_stage_allowed": False,
        "semantic_accuracy_verified": False, "generalization_verified": False,
        "architecture": "TaskSpecificMovement-GroupNorm-GRU-686281",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "device": str(device), "precision": "fp32", "seed": seed,
        "updates": maximum_updates, "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "first_update_loss": first_loss, "first_update_gradient_norm": first_gradient,
        "first_update_parameter_changed": first_changed,
        "first_update_finite": math.isfinite(first_loss) and math.isfinite(first_gradient),
        "initial_loss": losses[0], "final_update_loss": losses[-1],
        "maximum_gradient_norm": max(gradients),
        "eval_accuracy": accuracy, "eval_loss": loss,
        "eval_recall": dict(zip(MOVEMENT_ACTIONS, recalls, strict=True)),
        "minimum_eval_accuracy": minimum_accuracy, "maximum_eval_loss": maximum_loss,
        "attempt_limit": 1, "attempts_used": 1,
        "full_training_called": False, "input_commands_sent": 0,
        "dev_frames_opened": 0, "test_frames_opened": 0,
    }
    report["report_sha256"] = _canonical_sha256(report)
    (staging / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(output_dir)
    return report


def _joystick_pilot_metrics(
    model: TaskSpecificMovement, clips: torch.Tensor, labels: torch.Tensor,
    sources: np.ndarray, device: torch.device, batch_size: int,
    action_order: tuple[str, ...] = MOVEMENT_ACTIONS,
) -> dict[str, object]:
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(clips), batch_size):
            logits.append(model(_batch(clips[start:start + batch_size], device)).cpu())
    scores = torch.cat(logits)
    predicted = scores.argmax(1).numpy()
    expected = labels.numpy()
    metrics = _classification_metrics(predicted, expected, len(action_order), action_order)
    metrics["loss"] = float(nn.functional.cross_entropy(scores, labels))
    metrics["support"] = dict(zip(
        action_order, np.bincount(expected, minlength=len(action_order)).tolist(),
        strict=True,
    ))
    metrics["nonzero_recalls"] = sum(
        float(value) > 0 for value in cast(dict[str, float], metrics["recall"]).values()
    )
    metrics["source_accuracy"] = {
        source: float(np.mean(predicted[sources == source] == expected[sources == source]))
        for source in sorted(set(map(str, sources.tolist())))
    }
    return metrics


def run_joystick_grouped_pilot(
    config_path: Path, dataset_dir: Path, output_dir: Path, *, device_name: str,
) -> dict[str, object]:
    from hok_agent.movement_real_rgb import (
        JOYSTICK_CONTINUATION_ACTIONS,
        validate_joystick_continuation_dataset,
        validate_joystick_grouped_pilot,
        validate_joystick_scale21_dataset,
    )

    config = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
    scale21 = config.get("schema_version") == "joystick-scale21-pilot-train-contract-v1"
    continuation = (
        config.get("schema_version") == "joystick-continuation-pilot-train-contract-v1"
    )
    action_order = JOYSTICK_CONTINUATION_ACTIONS if continuation else MOVEMENT_ACTIONS
    unsigned = {key: value for key, value in config.items() if key != "contract_sha256"}
    training = cast(dict[str, object], config["training"])
    gates = cast(dict[str, object], config["gates"])
    expected_training = {
        "architecture": (
            "task_specific_groupnorm_gru_686152_8head"
            if continuation else "task_specific_groupnorm_gru_686281"
        ),
        "optimizer": "AdamW",
        "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 8, "epochs": 30,
        "evaluation_epochs": [5, 10, 15, 20, 25, 30],
        "sampling": "class_balanced_replacement",
        "samples_per_epoch": 203 if continuation else 201 if scale21 else 73,
        "precision": "fp32",
        "selection": "highest_dev_macro_f1_then_lower_loss_then_earlier_epoch",
    }
    expected_gates = {
        "minimum_train_accuracy": 0.85,
        "minimum_dev_accuracy": 0.35 if continuation else 0.32,
        "minimum_dev_macro_f1": 0.35,
        "minimum_nonzero_dev_recalls": 6 if continuation else 5,
        "minimum_gain_over_majority_macro_f1": 0.20,
        "minimum_each_dev_source_accuracy": 0.20,
    }
    dataset_path = dataset_dir / (
        "joystick-continuation-grouped.npz" if continuation
        else "joystick-scale21-grouped.npz" if scale21
        else "joystick-grouped-pilot.npz"
    )
    manifest_path = dataset_dir / "manifest.json"
    conclusion_path = dataset_dir / "conclusion.json"
    conclusion = cast(dict[str, object], json.loads(conclusion_path.read_text()))
    if (
        config.get("schema_version") not in {
            "joystick-grouped-pilot-train-contract-v1",
            "joystick-scale21-pilot-train-contract-v1",
            "joystick-continuation-pilot-train-contract-v1",
        }
        or config.get("contract_sha256") != _canonical_sha256(unsigned)
        or config.get("dataset_sha256") != _sha256(dataset_path)
        or config.get("manifest_file_sha256") != _sha256(manifest_path)
        or config.get("conclusion_file_sha256") != _sha256(conclusion_path)
        or config.get("action_order") != list(action_order)
        or training != expected_training or gates != expected_gates
        or config.get("attempt_limit") != 1
        or config.get("fresh_initialization") is not True
        or (not continuation and config.get("overfit_checkpoint_allowed") is not False)
        or (scale21 and config.get("prior_failed_checkpoint_allowed") is not False)
        or (scale21 and config.get("prior_failed_report_file_sha256") !=
            "2d9d917df08255cad30d90a8d3cb9e141610289724b4c5a8615e3c333669282b")
        or (continuation and config.get("previous_checkpoint_allowed") is not False)
        or (continuation and config.get("prior_failed_report_file_sha256") !=
            "056f299bd6e8e9816622294eeb9d3fdfcc532b28a4ec123b124a93be2105dd93")
        or (continuation and config.get("stop_owner") != "deterministic_router")
        or config.get("formal_training_allowed") is not False
        or config.get("checkpoint_promotion_allowed") is not False
        or config.get("video_dev_allowed") is not False
        or config.get("video_test_allowed") is not False
        or config.get("device_input_allowed") is not False
        or (not continuation and conclusion.get("pilot_training_allowed") is not True)
        or (continuation and conclusion.get("continuation_training_allowed") is not True)
        or conclusion.get("checkpoint_promotion_allowed") is not False
    ):
        raise ValueError("joystick grouped pilot contract differs")
    if continuation:
        validate_joystick_continuation_dataset(dataset_dir)
    elif scale21:
        validate_joystick_scale21_dataset(dataset_dir)
    else:
        validate_joystick_grouped_pilot(dataset_dir)
    if output_dir.exists():
        raise ValueError("joystick grouped pilot output exists")
    with np.load(dataset_path, allow_pickle=False) as data:
        train_clips = torch.from_numpy(data["train_rgb"].copy())
        train_labels = torch.from_numpy(data["train_label"].astype(np.int64))
        train_sources = data["train_source_id"].copy()
        dev_clips = torch.from_numpy(data["dev_rgb"].copy())
        dev_labels = torch.from_numpy(data["dev_label"].astype(np.int64))
        dev_sources = data["dev_source_id"].copy()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    seed = cast(int, config["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = TaskSpecificMovement(output_actions=len(action_order)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cast(float, training["learning_rate"]),
        weight_decay=cast(float, training["weight_decay"]),
    )
    counts = torch.bincount(train_labels, minlength=len(action_order))
    sampler = WeightedRandomSampler(
        1.0 / counts[train_labels].double(), num_samples=cast(int, training["samples_per_epoch"]),
        replacement=True, generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TensorDataset(train_clips, train_labels), batch_size=cast(int, training["batch_size"]),
        sampler=sampler, num_workers=0, generator=torch.Generator().manual_seed(seed),
    )
    initial_hash = hashlib.sha256(b"".join(
        value.detach().cpu().numpy().tobytes() for value in model.state_dict().values()
    )).hexdigest()
    first_before = [parameter.detach().clone() for parameter in model.parameters()]
    first_update: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-math.inf, -math.inf, -math.inf)
    best_epoch = 0
    sampled_counts = torch.zeros(len(action_order), dtype=torch.int64)
    started = time.monotonic()
    evaluation_epochs = set(cast(list[int], training["evaluation_epochs"]))
    for epoch in range(1, cast(int, training["epochs"]) + 1):
        losses: list[float] = []
        for clips, labels in loader:
            sampled_counts += torch.bincount(labels, minlength=len(action_order))
            loss, gradient = train_step(
                model, _batch(clips, device), labels.to(device), optimizer
            )
            losses.append(loss)
            if first_update is None:
                first_update = {
                    "loss": loss, "gradient_norm": gradient,
                    "parameter_changed": any(
                        not torch.equal(before, parameter.detach())
                        for before, parameter in zip(first_before, model.parameters(), strict=True)
                    ),
                    "finite": math.isfinite(loss) and math.isfinite(gradient) and gradient > 0,
                }
        if epoch in evaluation_epochs:
            train_metrics = _joystick_pilot_metrics(
                model, train_clips, train_labels, train_sources, device,
                cast(int, training["batch_size"]), action_order,
            )
            dev_metrics = _joystick_pilot_metrics(
                model, dev_clips, dev_labels, dev_sources, device,
                cast(int, training["batch_size"]), action_order,
            )
            history.append({"epoch": epoch, "mean_train_update_loss": float(np.mean(losses)),
                            "train": train_metrics, "dev": dev_metrics})
            key = (cast(float, dev_metrics["macro_f1"]),
                   -cast(float, dev_metrics["loss"]), -float(epoch))
            if key > best_key:
                best_key, best_epoch = key, epoch
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
    if first_update is None or best_state is None:
        raise ValueError("joystick grouped pilot performed no updates or evaluation")
    model.load_state_dict(best_state)
    train_metrics = _joystick_pilot_metrics(
        model, train_clips, train_labels, train_sources, device,
        cast(int, training["batch_size"]), action_order,
    )
    dev_metrics = _joystick_pilot_metrics(
        model, dev_clips, dev_labels, dev_sources, device,
        cast(int, training["batch_size"]), action_order,
    )
    majority = int(torch.bincount(train_labels).argmax())
    majority_metrics = _classification_metrics(
        np.full(len(dev_labels), majority), dev_labels.numpy(), len(action_order), action_order
    )
    source_accuracy = cast(dict[str, float], dev_metrics["source_accuracy"])
    gate_results = {
        "train_accuracy": cast(float, train_metrics["accuracy"])
        >= cast(float, gates["minimum_train_accuracy"]),
        "dev_accuracy": cast(float, dev_metrics["accuracy"])
        >= cast(float, gates["minimum_dev_accuracy"]),
        "dev_macro_f1": cast(float, dev_metrics["macro_f1"])
        >= cast(float, gates["minimum_dev_macro_f1"]),
        "nonzero_dev_recalls": cast(int, dev_metrics["nonzero_recalls"])
        >= cast(int, gates["minimum_nonzero_dev_recalls"]),
        "majority_gain": cast(float, dev_metrics["macro_f1"])
        - cast(float, majority_metrics["macro_f1"])
        >= cast(float, gates["minimum_gain_over_majority_macro_f1"]),
        "each_dev_source_accuracy": min(source_accuracy.values())
        >= cast(float, gates["minimum_each_dev_source_accuracy"]),
        "first_update": bool(first_update["finite"] and first_update["parameter_changed"]),
    }
    passed = all(gate_results.values())
    elapsed = time.monotonic() - started
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joystick-grouped-pilot-", dir=output_dir.parent))
    checkpoint = staging / "best-internal-dev.safetensors"
    save_file(
        best_state, checkpoint,
        metadata={"purpose": "continuation_pilot_only" if continuation else "grouped_pilot_only",
                  "dataset_sha256": _sha256(dataset_path),
                  "contract_sha256": cast(str, config["contract_sha256"]),
                  "fresh_initialization": "true", "overfit_checkpoint_loaded": "false",
                  "best_epoch": str(best_epoch)},
    )
    report: dict[str, object] = {
        "schema_version": (
            "joystick-continuation-pilot-report-v1" if continuation
            else "joystick-scale21-pilot-report-v1" if scale21
            else "joystick-grouped-pilot-report-v1"
        ),
        "status": "PASSED" if passed else "FAILED", "passed": passed,
        "contract_sha256": config["contract_sha256"],
        "config_file_sha256": _sha256(config_path), "dataset_sha256": _sha256(dataset_path),
        "manifest_file_sha256": _sha256(manifest_path),
        "checkpoint_sha256": _sha256(checkpoint), "checkpoint_promotion_allowed": False,
        "formal_training_allowed": False, "next_stage_allowed": False,
        "semantic_accuracy_verified": False,
        "generalization_scope": "five internal train-cohort sources" if scale21 or continuation
        else "two internal train-cohort sources",
        "architecture": (
            "TaskSpecificMovement-GroupNorm-GRU-686152-8head"
            if continuation else "TaskSpecificMovement-GroupNorm-GRU-686281"
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fresh_initialization": True, "initial_state_sha256": initial_hash,
        "overfit_checkpoint_loaded": False, "seed": seed, "device": str(device),
        "prior_failed_checkpoint_loaded": False,
        "comparison_prior_report_file_sha256": (
            config.get("prior_failed_report_file_sha256") if scale21 or continuation else None
        ),
        "epochs": training["epochs"], "best_epoch": best_epoch,
        "sampled_train_label_counts": dict(zip(
            action_order, sampled_counts.tolist(), strict=True
        )),
        "first_update": first_update, "history": history,
        "best_train": train_metrics, "best_dev": dev_metrics,
        "majority_baseline": {"predicted_action": action_order[majority], **majority_metrics},
        "stop_owner": "deterministic_router" if continuation else None,
        "gate_results": gate_results, "gates": gates,
        "elapsed_seconds": elapsed,
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "attempt_limit": 1, "attempts_used": 1, "input_commands_sent": 0,
        "video_dev_opened": False, "video_test_opened": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    (staging / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(output_dir)
    return report


def _classification_metrics(
    predicted: np.ndarray, labels: np.ndarray, classes: int,
    action_order: Sequence[str] = MOVEMENT_ACTIONS,
) -> dict[str, object]:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for expected, actual in zip(labels, predicted, strict=True):
        confusion[int(expected), int(actual)] += 1
    recalls: list[float] = []
    f1_values: list[float] = []
    for label in range(classes):
        true_positive = int(confusion[label, label])
        actual_support = int(confusion[:, label].sum())
        expected_support = int(confusion[label].sum())
        recall = true_positive / expected_support if expected_support else 0.0
        precision = true_positive / actual_support if actual_support else 0.0
        recalls.append(recall)
        f1_values.append(
            2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "macro_f1": float(np.mean(f1_values)),
        "recall": dict(zip(action_order, recalls, strict=True)),
        "confusion": confusion.tolist(),
    }


def _train_group_fold(
    clips: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    holdout_group: int,
    stage: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    seed = int(cast(int, stage["seed"]))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TaskSpecificMovement().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cast(float, stage["learning_rate"])),
        weight_decay=float(cast(float, stage["weight_decay"])),
    )
    train_indices = np.flatnonzero(groups != holdout_group)
    dev_indices = np.flatnonzero(groups == holdout_group)
    source_group_leakage = bool(
        set(groups[train_indices].tolist()) & set(groups[dev_indices].tolist())
    )
    batch_size = int(cast(int, stage["batch_size"]))
    updates = int(cast(int, stage["updates"]))
    generator = torch.Generator().manual_seed(seed)
    order = torch.empty(0, dtype=torch.int64)
    offset = 0
    losses: list[float] = []
    started = time.monotonic()
    clip_tensor = torch.from_numpy(clips)
    label_tensor = torch.from_numpy(labels)
    for _update in range(updates):
        if offset >= len(order):
            order = torch.from_numpy(train_indices)[
                torch.randperm(len(train_indices), generator=generator)
            ]
            offset = 0
        selected = order[offset : offset + batch_size]
        offset += len(selected)
        loss, _gradient = train_step(
            model,
            _batch(clip_tensor[selected], device),
            label_tensor[selected].to(device),
            optimizer,
        )
        losses.append(loss)
    model.eval()
    with torch.no_grad():
        logits = model(_batch(clip_tensor[dev_indices], device)).cpu()
    predicted = logits.argmax(1).numpy()
    expected = labels[dev_indices]
    metrics = _classification_metrics(predicted, expected, len(MOVEMENT_ACTIONS))
    metrics.update(
        {
            "holdout_group": holdout_group,
            "train_samples": len(train_indices),
            "dev_samples": len(dev_indices),
            "source_group_leakage": source_group_leakage,
            "initial_loss": losses[0],
            "final_update_loss": losses[-1],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return metrics, predicted, expected


def run_real_counterfactual_grouped_eval(
    contract_path: Path,
    localization_report_path: Path,
    session_root: Path,
    old_dataset_path: Path,
    old_data_report_path: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    from hok_agent.movement_real_rgb import (
        _file_sha256,
        _load_bound_json,
        _object_sha256,
        prepare_real_counterfactual_grouped_data,
    )

    raw = cast(dict[str, object], json.loads(contract_path.read_text(encoding="utf-8")))
    supplied_hash = str(raw.pop("contract_sha256", ""))
    raw["contract_sha256"] = supplied_hash
    if (
        raw.get("schema_version") != "movement-real-counterfactual-grouped-eval-v1"
        or supplied_hash
        != _canonical_sha256({key: value for key, value in raw.items() if key != "contract_sha256"})
        or raw.get("test_allowed") is not False
        or raw.get("formal_training_allowed") is not False
        or raw.get("r2_allowed") is not False
        or raw.get("device_input_allowed") is not False
        or raw.get("variants") != ["full", "player_masked", "goal_only"]
        or raw.get("folds") != 5
        or raw.get("expected_model_runs") != 15
        or raw.get("checkpoints_persisted") != 0
        or cast(dict[str, object], raw["training"]).get("model") != "task_specific_groupnorm_gru"
    ):
        raise ValueError("real grouped evaluation contract differs")
    localization = _load_bound_json(localization_report_path, "report_sha256")
    old_data_report = _load_bound_json(old_data_report_path, "report_sha256")
    if (
        localization.get("status") != "PLAYER_CUE_PARTIAL_SESSION002_ONLY"
        or localization.get("semantic_identity_scope")
        != "session002_partial_action_response_supported"
        or _file_sha256(localization_report_path) != raw.get("localization_report_file_sha256")
        or localization.get("report_sha256") != raw.get("localization_report_sha256")
        or _file_sha256(old_dataset_path) != raw.get("old_dataset_sha256")
        or _file_sha256(old_data_report_path) != raw.get("old_data_report_file_sha256")
        or old_data_report.get("report_sha256") != raw.get("old_data_report_sha256")
        or old_data_report.get("dataset_sha256") != raw.get("old_dataset_sha256")
    ):
        raise ValueError("real grouped evaluation lineage differs")
    if session_root.is_symlink() or not session_root.is_dir():
        raise ValueError("real grouped evaluation session root differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("real grouped evaluation output already exists")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")

    with np.load(old_dataset_path, allow_pickle=False) as old_dataset:
        old_clips = old_dataset["rgb_sequence"]
        old_unique_clips = len({hashlib.sha256(clip.tobytes()).hexdigest() for clip in old_clips})
        old_unique_windows = len(np.unique(old_dataset["end_timestamp_ms"]))
    old_duplicates = len(old_clips) - old_unique_clips
    expected_old = cast(dict[str, object], raw["old_dataset_expected"])
    if (
        len(old_clips) != expected_old["samples"]
        or old_unique_clips != expected_old["unique_clips"]
        or old_unique_windows != expected_old["unique_source_windows"]
        or old_duplicates != expected_old["duplicate_clips"]
    ):
        raise ValueError("real grouped old duplicate audit differs")

    variants, labels, groups, data_metadata = prepare_real_counterfactual_grouped_data(
        raw, session_root
    )
    if (
        labels.shape != (45,)
        or groups.shape != (45,)
        or sorted(np.bincount(groups, minlength=5).tolist()) != [9] * 5
        or any(clips.shape != (45, 16, 128, 128, 3) for clips in variants.values())
    ):
        raise ValueError("real grouped generated data differs")
    stage = cast(dict[str, object], raw["training"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_started = time.monotonic()
    fold_rows: list[dict[str, object]] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = {
        name: [] for name in cast(list[str], raw["variants"])
    }
    aggregate_expected: dict[str, list[np.ndarray]] = {
        name: [] for name in cast(list[str], raw["variants"])
    }
    for variant in cast(list[str], raw["variants"]):
        for holdout_group in range(int(cast(int, raw["folds"]))):
            metrics, predicted, expected = _train_group_fold(
                variants[variant], labels, groups, holdout_group, stage, device
            )
            metrics["variant"] = variant
            fold_rows.append(metrics)
            aggregate_predictions[variant].append(predicted)
            aggregate_expected[variant].append(expected)
    elapsed = time.monotonic() - run_started
    aggregate: dict[str, dict[str, object]] = {}
    for variant in cast(list[str], raw["variants"]):
        metrics = _classification_metrics(
            np.concatenate(aggregate_predictions[variant]),
            np.concatenate(aggregate_expected[variant]),
            len(MOVEMENT_ACTIONS),
        )
        variant_folds = [row for row in fold_rows if row["variant"] == variant]
        metrics["mean_fold_accuracy"] = float(
            np.mean([float(cast(float, row["accuracy"])) for row in variant_folds])
        )
        metrics["mean_fold_macro_f1"] = float(
            np.mean([float(cast(float, row["macro_f1"])) for row in variant_folds])
        )
        metrics["worst_fold_accuracy"] = min(
            float(cast(float, row["accuracy"])) for row in variant_folds
        )
        aggregate[variant] = metrics
    full = aggregate["full"]
    controls = {
        name: float(cast(float, full["mean_fold_accuracy"]))
        - float(cast(float, aggregate[name]["mean_fold_accuracy"]))
        for name in ("player_masked", "goal_only")
    }
    gates = cast(dict[str, object], raw["gates"])
    gate_results = {
        "mean_accuracy": float(cast(float, full["mean_fold_accuracy"]))
        >= float(cast(float, gates["minimum_full_mean_accuracy"])),
        "mean_macro_f1": float(cast(float, full["mean_fold_macro_f1"]))
        >= float(cast(float, gates["minimum_full_mean_macro_f1"])),
        "worst_fold_accuracy": float(cast(float, full["worst_fold_accuracy"]))
        >= float(cast(float, gates["minimum_full_worst_fold_accuracy"])),
        "per_class_recall": min(cast(dict[str, float], full["recall"]).values())
        >= float(cast(float, gates["minimum_full_per_class_recall"])),
        "player_mask_gain": controls["player_masked"]
        >= float(cast(float, gates["minimum_control_accuracy_gain"])),
        "goal_only_gain": controls["goal_only"]
        >= float(cast(float, gates["minimum_control_accuracy_gain"])),
        "group_isolation": all(row["source_group_leakage"] is False for row in fold_rows),
    }
    passed = all(gate_results.values())
    report: dict[str, object] = {
        "schema_version": "movement-real-counterfactual-grouped-eval-report-v1",
        "status": "SESSION002_RELATION_SIGNAL_PASSED"
        if passed
        else "REAL_COUNTERFACTUAL_MODEL_SHORTCUT_OR_NO_GENERALIZATION",
        "contract_sha256": supplied_hash,
        "localization_report_sha256": localization["report_sha256"],
        "old_data_report_sha256": old_data_report["report_sha256"],
        "old_dataset_duplicate_audit": {
            "samples": len(old_clips),
            "unique_clips": old_unique_clips,
            "duplicate_clips": old_duplicates,
            "unique_source_windows": old_unique_windows,
        },
        "generated_data": data_metadata,
        "folds": fold_rows,
        "aggregate": aggregate,
        "full_accuracy_gain_over_controls": controls,
        "gate_results": gate_results,
        "model_runs": len(fold_rows),
        "expected_model_runs": raw["expected_model_runs"],
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "checkpoints_persisted": 0,
        "source_window_generalization_verified": passed,
        "session_generalization_verified": False,
        "semantic_accuracy_verified": False,
        "formal_training_allowed": False,
        "test_frames_read": 0,
        "device_input_commands_sent": 0,
        "next_action": "expand_automatic_player_localization_coverage"
        if passed
        else "stop_model_tuning_and_repair_player_localization",
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        with (staging / "report.json").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        raise
    return report


def _native_relation_variants(
    dataset_path: Path,
    *,
    render_after_resize: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, object]]:
    from hok_agent.movement_real_rgb import _mask_player_patch, mark_pixel_goal

    with np.load(dataset_path, allow_pickle=False) as arrays:
        source = arrays["source_clips"].copy()
        group_split = arrays["split"].copy()
        anchors = arrays["anchor_yx"].copy()
        sample_group = arrays["sample_group_index"].astype(np.int64)
        labels = arrays["sample_label"].astype(np.int64)
        targets = arrays["sample_target_yx"].copy()
    if (
        source.ndim != 5
        or source.shape[1:] != (16, 256, 256, 3)
        or source.dtype != np.uint8
        or group_split.shape != (len(source),)
        or anchors.shape != (len(source), 2)
        or sample_group.shape != labels.shape
        or targets.shape != (len(labels), 2)
        or np.any(sample_group < 0)
        or np.any(sample_group >= len(source))
    ):
        raise ValueError("native relation dataset arrays differ")
    variant_rows: dict[str, list[np.ndarray]] = {
        "full": [],
        "anchor_masked": [],
        "goal_only": [],
    }
    for group_index, target in zip(sample_group, targets, strict=True):
        clip = source[group_index]
        anchor = (int(anchors[group_index, 0]), int(anchors[group_index, 1]))
        goal = (int(target[0]), int(target[1]))
        if render_after_resize:
            resized = np.ascontiguousarray(clip[:, ::2, ::2])
            small_anchor = (round(anchor[0] / 2), round(anchor[1] / 2))
            small_goal = (round(goal[0] / 2), round(goal[1] / 2))
            masked = np.stack([_mask_player_patch(frame, small_anchor, 9) for frame in resized])
            neutral = np.full_like(resized, 32)
            bases = (("full", resized), ("anchor_masked", masked), ("goal_only", neutral))
            for name, base in bases:
                variant_rows[name].append(
                    np.stack([mark_pixel_goal(frame, small_goal) for frame in base])
                )
        else:
            masked = np.stack([_mask_player_patch(frame, anchor, 18) for frame in clip])
            neutral = np.full_like(clip, 32)
            bases = (("full", clip), ("anchor_masked", masked), ("goal_only", neutral))
            for name, base in bases:
                marked = np.stack([mark_pixel_goal(frame, goal) for frame in base])
                variant_rows[name].append(np.ascontiguousarray(marked[:, ::2, ::2]))
    variants = {name: np.stack(rows) for name, rows in variant_rows.items()}
    sample_split = group_split[sample_group]
    metadata: dict[str, object] = {
        "source_sha256_before": hashlib.sha256(source.tobytes()).hexdigest(),
        "variant_sha256": {
            name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in variants.items()
        },
        "samples": len(labels),
        "groups": len(source),
        "render_order": (
            "resize_128_then_mark_radius7"
            if render_after_resize
            else "mark_radius7_at_256_then_resize_128"
        ),
    }
    return variants, labels, sample_split, metadata


def _train_native_relation_variant(
    clips: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    dev_indices: np.ndarray,
    device: torch.device,
    *,
    updates: int,
) -> tuple[TaskSpecificMovement, dict[str, object]]:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model = TaskSpecificMovement().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(0)
    order = torch.empty(0, dtype=torch.int64)
    offset = 0
    first_before = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    first_loss = 0.0
    first_gradient = 0.0
    losses: list[float] = []
    clip_tensor = torch.from_numpy(clips)
    label_tensor = torch.from_numpy(labels)
    started = time.monotonic()
    for update in range(updates):
        if offset + 8 > len(order):
            order = torch.from_numpy(train_indices)[
                torch.randperm(len(train_indices), generator=generator)
            ]
            offset = 0
        selected = order[offset : offset + 8]
        offset += len(selected)
        loss, gradient = train_step(
            model,
            _batch(clip_tensor[selected], device),
            label_tensor[selected].to(device),
            optimizer,
        )
        losses.append(loss)
        if update == 0:
            first_loss, first_gradient = loss, gradient
    first_changed = any(
        not torch.equal(before, parameter.detach().cpu())
        for before, parameter in zip(first_before, model.parameters(), strict=True)
    )
    model.eval()
    with torch.no_grad():
        logits = model(_batch(clip_tensor[dev_indices], device)).cpu()
    predicted = logits.argmax(1).numpy()
    metrics = _classification_metrics(predicted, labels[dev_indices], len(MOVEMENT_ACTIONS))
    metrics["cross_entropy"] = float(
        nn.functional.cross_entropy(logits, label_tensor[dev_indices]).item()
    )
    metrics.update(
        {
            "updates": updates,
            "train_samples": len(train_indices),
            "dev_samples": len(dev_indices),
            "first_update_loss": first_loss,
            "first_update_gradient_norm": first_gradient,
            "first_update_parameter_changed": first_changed,
            "final_update_loss": losses[-1],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return model, metrics


def run_native_anchor_relation_diagnostic(
    dataset_path: Path,
    dataset_report_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    repair_report_path: Path | None = None,
    updates: int = 400,
    overfit_updates: int = 400,
) -> dict[str, object]:
    from hok_agent.movement_real_rgb import _file_sha256, _load_bound_json, _object_sha256

    if output_dir.exists():
        raise ValueError("native anchor relation output already exists")
    dataset_report = _load_bound_json(dataset_report_path, "report_sha256")
    if (
        dataset_report.get("schema_version") != "native-weak-anchor-counterfactual-dataset-v1"
        or dataset_report.get("status") != "WEAK_ANCHOR_COUNTERFACTUAL_DATASET_READY"
        or dataset_report.get("relation_diagnostic_training_allowed") is not True
        or dataset_report.get("movement_policy_training_allowed") is not False
        or dataset_report.get("dataset_sha256") != _file_sha256(dataset_path)
        or updates != 400
        or overfit_updates != 400
    ):
        raise ValueError("native anchor relation training contract differs")
    repair: dict[str, object] | None = None
    if repair_report_path is not None:
        repair = _load_bound_json(repair_report_path, "report_sha256")
        repair_checks = cast(dict[str, object], repair.get("checks"))
        if (
            repair.get("schema_version") != "native-weak-anchor-relation-diagnostic-v1"
            or repair.get("status") != "WEAK_ANCHOR_RELATION_DIAGNOSTIC_FAILED"
            or repair.get("dataset_sha256") != _file_sha256(dataset_path)
            or repair.get("model_runs") != 1
            or repair.get("results") != {}
            or repair_checks.get("overfit36") is not False
        ):
            raise ValueError("native anchor relation repair evidence differs")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable for native anchor relation diagnostic")
    variants, labels, splits, variant_metadata = _native_relation_variants(
        dataset_path, render_after_resize=repair is not None
    )
    train_indices = np.flatnonzero(splits == "train")
    dev_indices = np.flatnonzero(splits == "dev")
    overfit_groups = np.unique(train_indices // len(MOVEMENT_ACTIONS))[:4]
    overfit_indices = np.flatnonzero(
        np.isin(np.arange(len(labels)) // len(MOVEMENT_ACTIONS), overfit_groups)
    )
    _overfit_model, overfit = _train_native_relation_variant(
        variants["full"],
        labels,
        overfit_indices,
        overfit_indices,
        device,
        updates=overfit_updates,
    )
    overfit_passed = bool(
        cast(float, overfit["accuracy"]) >= 0.95 and cast(float, overfit["cross_entropy"]) <= 0.05
    )
    results: dict[str, dict[str, object]] = {}
    if overfit_passed:
        for name in ("full", "anchor_masked", "goal_only"):
            _model, result = _train_native_relation_variant(
                variants[name], labels, train_indices, dev_indices, device, updates=updates
            )
            results[name] = result
    full = results.get("full", {})
    masked = results.get("anchor_masked", {})
    goal = results.get("goal_only", {})
    update_evidence = [overfit, *results.values()]
    checks = {
        "overfit36": overfit_passed,
        "first_update_evidence": all(
            row.get("first_update_parameter_changed") is True
            and math.isfinite(cast(float, row["first_update_loss"]))
            and cast(float, row["first_update_loss"]) > 0
            and math.isfinite(cast(float, row["first_update_gradient_norm"]))
            and cast(float, row["first_update_gradient_norm"]) > 0
            for row in update_evidence
        ),
        "full_accuracy": bool(full) and cast(float, full["accuracy"]) >= 0.80,
        "full_macro_f1": bool(full) and cast(float, full["macro_f1"]) >= 0.80,
        "full_per_class_recall": bool(full)
        and min(cast(dict[str, float], full["recall"]).values()) >= 0.60,
        "full_over_anchor_masked": bool(full and masked)
        and cast(float, full["accuracy"]) - cast(float, masked["accuracy"]) >= 0.15,
        "full_over_goal_only": bool(full and goal)
        and cast(float, full["accuracy"]) - cast(float, goal["accuracy"]) >= 0.15,
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": "native-weak-anchor-relation-diagnostic-v1",
        "status": "WEAK_ANCHOR_RELATION_SIGNAL_PASSED"
        if passed
        else "WEAK_ANCHOR_RELATION_DIAGNOSTIC_FAILED",
        "dataset_report_sha256": dataset_report["report_sha256"],
        "dataset_report_file_sha256": _file_sha256(dataset_report_path),
        "dataset_sha256": _file_sha256(dataset_path),
        "repair_of_report_file_sha256": (
            _file_sha256(repair_report_path) if repair_report_path is not None else None
        ),
        "repair_of_report_sha256": repair["report_sha256"] if repair is not None else None,
        "implementation_sha256": _sha256(Path(__file__)),
        "architecture": "TaskSpecificMovement-GroupNorm-GRU-686281",
        "seed": 0,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 8,
        "updates": updates,
        "overfit_updates": overfit_updates,
        "overfit36": overfit,
        "results": results,
        "checks": checks,
        "variant_metadata": variant_metadata,
        "model_runs": 1 + len(results),
        "checkpoint_saved": False,
        "relation_signal_only": True,
        "movement_policy_training_allowed": False,
        "controlled_player_identity_verified": False,
        "executed_action_labels_used": False,
        "test_frames_read": 0,
        "video_frames_decoded": 0,
        "input_commands_sent": 0,
        "device": device.type,
    }
    report["report_sha256"] = _object_sha256(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )
    return report


def _slot_cell_targets(coordinates: torch.Tensor, grid_size: int) -> torch.Tensor:
    scaled = torch.round(coordinates / 127.0 * (grid_size - 1)).long()
    scaled = scaled.clamp(0, grid_size - 1)
    return scaled[..., 1] * grid_size + scaled[..., 0]


def localized_train_step(
    model: RelationalMovement,
    clips: torch.Tensor,
    labels: torch.Tensor,
    coordinates: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    action_weight: float,
    localization_weight: float,
    grid_size: int,
) -> tuple[float, float, float, float]:
    model.train()
    action_logits, attention_logits, _slot_coordinates = model.forward_with_slots(clips)
    action_loss = nn.functional.cross_entropy(action_logits, labels)
    cell_targets = _slot_cell_targets(coordinates, grid_size)
    localization_loss = nn.functional.cross_entropy(
        attention_logits.reshape(-1, grid_size * grid_size), cell_targets.flatten()
    )
    loss = action_weight * action_loss + localization_weight * localization_loss
    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = [
        torch.sum(parameter.grad.detach() ** 2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = float(torch.sqrt(torch.stack(gradients).sum()))
    optimizer.step()
    return (
        float(loss.detach()),
        float(action_loss.detach()),
        float(localization_loss.detach()),
        gradient_norm,
    )


def _evaluate_localized(
    model: RelationalMovement,
    clips: torch.Tensor,
    labels: torch.Tensor,
    coordinates: torch.Tensor,
    device: torch.device,
    grid_size: int,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        action_logits, attention_logits, predicted_coordinates = model.forward_with_slots(
            _batch(clips, device)
        )
    labels_device = labels.to(device)
    coordinates_device = coordinates.to(device)
    action_loss = float(nn.functional.cross_entropy(action_logits, labels_device))
    predicted = action_logits.argmax(dim=1)
    recalls = []
    for label in range(len(MOVEMENT_ACTIONS)):
        selected = labels_device == label
        recalls.append(float((predicted[selected] == label).float().mean()))
    targets = _slot_cell_targets(coordinates_device, grid_size)
    predicted_cells = attention_logits.argmax(dim=-1)
    cell_accuracy = float((predicted_cells == targets).float().mean())
    target_x, target_y = targets % grid_size, targets // grid_size
    predicted_x, predicted_y = predicted_cells % grid_size, predicted_cells // grid_size
    cell_error = torch.maximum(torch.abs(predicted_x - target_x), torch.abs(predicted_y - target_y))
    predicted_pixels = (predicted_coordinates + 1.0) * 63.5
    slot_error = torch.linalg.vector_norm(predicted_pixels - coordinates_device, dim=-1)
    return {
        "action_accuracy": float((predicted == labels_device).float().mean()),
        "action_loss": action_loss,
        "action_recall": dict(zip(MOVEMENT_ACTIONS, recalls, strict=True)),
        "slot_cell_accuracy": cell_accuracy,
        "slot_within_one_cell_accuracy": float((cell_error <= 1).float().mean()),
        "maximum_slot_cell_error": int(cell_error.max()),
        "mean_slot_error_pixels": float(slot_error.mean()),
        "player_slot_error_pixels": float(slot_error[:, :, 0].mean()),
        "goal_slot_error_pixels": float(slot_error[:, :, 1].mean()),
    }


def run_localized_overfit32(
    config_path: Path,
    dataset_path: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    raw = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
    unsigned = {key: value for key, value in raw.items() if key != "contract_sha256"}
    training = cast(dict[str, object], raw["training"])
    if (
        raw.get("schema_version") != "movement-goal-canvas-localized-contract-v1"
        or raw.get("contract_sha256") != _canonical_sha256(unsigned)
        or int(cast(int, training["diagnostic_attempts_used"]))
        >= int(cast(int, training["diagnostic_attempt_limit"]))
        or raw.get("formal_training_allowed") is not False
        or raw.get("test_allowed") is not False
        or raw.get("r2_allowed") is not False
        or raw.get("device_input_allowed") is not False
    ):
        raise ValueError("localized overfit32 training contract differs")
    with np.load(dataset_path, allow_pickle=False) as data:
        clips = torch.from_numpy(data["rgb_sequence"].copy())
        labels = torch.from_numpy(data["label"].astype(np.int64))
        player = torch.from_numpy(data["player_xy_sequence"].astype(np.float32))
        goal = torch.from_numpy(data["goal_xy_sequence"].astype(np.float32))
        dataset_contract = str(data["contract_sha256"][0])
    coordinates = torch.stack((player, goal), dim=2)
    if (
        clips.shape != (32, 16, 128, 128, 3)
        or coordinates.shape != (32, 16, 2, 2)
        or dataset_contract != raw["contract_sha256"]
        or sorted(labels.tolist())
        != [*([0] * 8), *[label for label in range(1, 9) for _ in range(3)]]
    ):
        raise ValueError("localized overfit32 dataset differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("localized overfit32 training output already exists")
    output_dir.mkdir(parents=True)
    seed = int(cast(int, raw["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = RelationalMovement().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cast(float, training["learning_rate"])),
        weight_decay=float(cast(float, training["weight_decay"])),
    )
    batch_size = int(cast(int, training["batch_size"]))
    updates = int(cast(int, training["maximum_updates"]))
    grid_size = int(cast(int, training["attention_grid_size"]))
    generator = torch.Generator().manual_seed(seed)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    first_update: dict[str, object] | None = None
    final_losses: tuple[float, float, float, float] | None = None
    started = time.monotonic()
    for update in range(updates):
        if update % (len(clips) // batch_size) == 0:
            order = torch.randperm(len(clips), generator=generator)
        start = update % (len(clips) // batch_size) * batch_size
        selected = order[start : start + batch_size]
        final_losses = localized_train_step(
            model,
            _batch(clips[selected], device),
            labels[selected].to(device),
            coordinates[selected].to(device),
            optimizer,
            action_weight=float(cast(float, training["action_loss_weight"])),
            localization_weight=float(cast(float, training["localization_loss_weight"])),
            grid_size=grid_size,
        )
        if update == 0:
            first_update = {
                "total_loss": final_losses[0],
                "action_loss": final_losses[1],
                "localization_loss": final_losses[2],
                "gradient_norm": final_losses[3],
                "parameter_changed": any(
                    not torch.equal(previous, parameter.detach())
                    for previous, parameter in zip(before, model.parameters(), strict=True)
                ),
                "finite": all(math.isfinite(value) for value in final_losses),
            }
    elapsed = time.monotonic() - started
    if first_update is None or final_losses is None:
        raise ValueError("localized overfit32 performed no updates")
    evaluation = _evaluate_localized(model, clips, labels, coordinates, device, grid_size)
    passed = (
        first_update["parameter_changed"] is True
        and first_update["finite"] is True
        and float(cast(float, evaluation["action_accuracy"]))
        >= float(cast(float, training["minimum_action_accuracy"]))
        and float(cast(float, evaluation["action_loss"]))
        <= float(cast(float, training["maximum_action_loss"]))
        and float(cast(float, evaluation["slot_cell_accuracy"]))
        >= float(cast(float, training["minimum_slot_cell_accuracy"]))
        and float(cast(float, evaluation["mean_slot_error_pixels"]))
        <= float(cast(float, training["maximum_slot_error_pixels"]))
    )
    checkpoint_path = output_dir / "diagnostic-last.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        checkpoint_path,
        metadata={
            "purpose": "localized_diagnostic_only",
            "dataset_sha256": _sha256(dataset_path),
            "config_sha256": _sha256(config_path),
            "coordinate_labels_in_actor_input": "false",
        },
    )
    report: dict[str, object] = {
        "schema_version": "movement-goal-canvas-localized-overfit32-report-v1",
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "contract_sha256": raw["contract_sha256"],
        "config_sha256": _sha256(config_path),
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "architecture": "relational-spatial-slots-gru",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "updates": updates,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "first_update": first_update,
        "final_update": {
            "total_loss": final_losses[0],
            "action_loss": final_losses[1],
            "localization_loss": final_losses[2],
            "gradient_norm": final_losses[3],
        },
        "evaluation": evaluation,
        "gates": {
            "minimum_action_accuracy": training["minimum_action_accuracy"],
            "maximum_action_loss": training["maximum_action_loss"],
            "minimum_slot_cell_accuracy": training["minimum_slot_cell_accuracy"],
            "maximum_slot_error_pixels": training["maximum_slot_error_pixels"],
        },
        "automatic_localization_targets": True,
        "coordinate_labels_in_actor_input": False,
        "diagnostic_checkpoint_only": True,
        "diagnostic_checkpoint_reusable_for_formal_training": False,
        "full_training_called": False,
        "next_stage_allowed": passed,
        "real_rgb_training_frames": 0,
        "test_frames_read": 0,
        "holdout_opened": False,
        "r2_allowed": False,
        "device_input_commands_sent": 0,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    report["report_sha256"] = _canonical_sha256(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _load_localized_dataset(
    dataset_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(dataset_path, allow_pickle=False) as data:
        clips = torch.from_numpy(data["rgb_sequence"].copy())
        labels = torch.from_numpy(data["label"].astype(np.int64))
        player = torch.from_numpy(data["player_xy_sequence"].astype(np.float32))
        goal = torch.from_numpy(data["goal_xy_sequence"].astype(np.float32))
    coordinates = torch.stack((player, goal), dim=2)
    if clips.shape != (32, 16, 128, 128, 3) or coordinates.shape != (32, 16, 2, 2):
        raise ValueError("two-stage localized dataset differs")
    return clips, labels, coordinates


def _localization_step(
    model: RelationalMovement,
    clips: torch.Tensor,
    coordinates: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grid_size: int,
) -> tuple[float, float]:
    model.train()
    _actions, attention_logits, _predicted = model.forward_with_slots(clips)
    targets = _slot_cell_targets(coordinates, grid_size)
    loss = nn.functional.cross_entropy(
        attention_logits.reshape(-1, grid_size * grid_size), targets.flatten()
    )
    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = [
        torch.sum(parameter.grad.detach() ** 2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = float(torch.sqrt(torch.stack(gradients).sum()))
    optimizer.step()
    return float(loss.detach()), gradient_norm


def _action_step(
    model: RelationalMovement,
    clips: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    loss = nn.functional.cross_entropy(model(clips), labels)
    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = [
        torch.sum(parameter.grad.detach() ** 2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = float(torch.sqrt(torch.stack(gradients).sum()))
    optimizer.step()
    return float(loss.detach()), gradient_norm


def run_two_stage_overfit32(
    config_path: Path,
    failed_joint_report_path: Path,
    dataset_path: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    raw = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
    unsigned = {key: value for key, value in raw.items() if key != "contract_sha256"}
    localization = cast(dict[str, object], raw["localization_stage"])
    action = cast(dict[str, object], raw["action_stage"])
    schema = raw.get("schema_version")
    prior_hash_field = (
        "failed_v1_report_sha256"
        if schema == "movement-goal-canvas-two-stage-contract-v2"
        else "failed_joint_report_sha256"
    )
    if (
        schema
        not in {
            "movement-goal-canvas-two-stage-contract-v1",
            "movement-goal-canvas-two-stage-contract-v2",
        }
        or raw.get("contract_sha256") != _canonical_sha256(unsigned)
        or _sha256(failed_joint_report_path) != raw.get(prior_hash_field)
        or _sha256(dataset_path) != raw.get("localized_dataset_sha256")
        or int(cast(int, raw["diagnostic_attempts_used"]))
        >= int(cast(int, raw["diagnostic_attempt_limit"]))
        or raw.get("fresh_initialization") is not True
        or raw.get("formal_training_allowed") is not False
        or raw.get("test_allowed") is not False
        or raw.get("r2_allowed") is not False
        or raw.get("device_input_allowed") is not False
    ):
        raise ValueError("two-stage Movement contract binding differs")
    try:
        failed = json.loads(failed_joint_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("two-stage prior report is invalid") from exc
    if failed.get("status") != "FAILED" or failed.get("next_stage_allowed") is not False:
        raise ValueError("two-stage prior result differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("two-stage Movement output already exists")
    clips, labels, coordinates = _load_localized_dataset(dataset_path)
    output_dir.mkdir(parents=True)
    seed = int(cast(int, raw["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = RelationalMovement().to(device)
    grid_size = int(cast(int, localization["attention_grid_size"]))
    localization_parameters = [
        *model.spatial.parameters(),
        *model.attention.parameters(),
    ]
    localization_optimizer = torch.optim.AdamW(
        localization_parameters,
        lr=float(cast(float, localization["learning_rate"])),
        weight_decay=float(cast(float, localization["weight_decay"])),
    )
    generator = torch.Generator().manual_seed(seed)
    localization_updates = int(cast(int, localization["maximum_updates"]))
    localization_batch = int(cast(int, localization["batch_size"]))
    localization_first: tuple[float, float] | None = None
    localization_final: tuple[float, float] | None = None
    started = time.monotonic()
    for update in range(localization_updates):
        if update % (len(clips) // localization_batch) == 0:
            order = torch.randperm(len(clips), generator=generator)
        start = update % (len(clips) // localization_batch) * localization_batch
        selected = order[start : start + localization_batch]
        localization_final = _localization_step(
            model,
            _batch(clips[selected], device),
            coordinates[selected].to(device),
            localization_optimizer,
            grid_size,
        )
        if update == 0:
            localization_first = localization_final
    localization_evaluation = _evaluate_localized(
        model, clips, labels, coordinates, device, grid_size
    )
    accuracy_name = (
        "slot_within_one_cell_accuracy"
        if schema == "movement-goal-canvas-two-stage-contract-v2"
        else "slot_cell_accuracy"
    )
    minimum_accuracy_name = (
        "minimum_slot_within_one_cell_accuracy"
        if schema == "movement-goal-canvas-two-stage-contract-v2"
        else "minimum_slot_cell_accuracy"
    )
    localization_passed = float(cast(float, localization_evaluation[accuracy_name])) >= float(
        cast(float, localization[minimum_accuracy_name])
    ) and float(cast(float, localization_evaluation["mean_slot_error_pixels"])) <= float(
        cast(float, localization["maximum_slot_error_pixels"])
    )
    frozen_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("spatial.", "attention."))
    }
    action_first: tuple[float, float] | None = None
    action_final: tuple[float, float] | None = None
    if localization_passed:
        for parameter in localization_parameters:
            parameter.requires_grad = False
        action_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        action_optimizer = torch.optim.AdamW(
            action_parameters,
            lr=float(cast(float, action["learning_rate"])),
            weight_decay=float(cast(float, action["weight_decay"])),
        )
        action_updates = int(cast(int, action["maximum_updates"]))
        action_batch = int(cast(int, action["batch_size"]))
        for update in range(action_updates):
            if update % (len(clips) // action_batch) == 0:
                order = torch.randperm(len(clips), generator=generator)
            start = update % (len(clips) // action_batch) * action_batch
            selected = order[start : start + action_batch]
            action_final = _action_step(
                model,
                _batch(clips[selected], device),
                labels[selected].to(device),
                action_optimizer,
            )
            if update == 0:
                action_first = action_final
    evaluation = _evaluate_localized(model, clips, labels, coordinates, device, grid_size)
    frozen_unchanged = all(
        torch.equal(value, model.state_dict()[name].detach())
        for name, value in frozen_before.items()
    )
    action_passed = (
        localization_passed
        and action_first is not None
        and action_final is not None
        and float(cast(float, evaluation["action_accuracy"]))
        >= float(cast(float, action["minimum_action_accuracy"]))
        and float(cast(float, evaluation["action_loss"]))
        <= float(cast(float, action["maximum_action_loss"]))
        and min(cast(dict[str, float], evaluation["action_recall"]).values()) > 0.0
        and frozen_unchanged
    )
    passed = localization_passed and action_passed
    checkpoint_path = output_dir / "diagnostic-last.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        checkpoint_path,
        metadata={
            "purpose": "two_stage_diagnostic_only",
            "dataset_sha256": _sha256(dataset_path),
            "config_sha256": _sha256(config_path),
            "fresh_initialization": "true",
        },
    )
    report: dict[str, object] = {
        "schema_version": (
            "movement-goal-canvas-two-stage-overfit32-report-v2"
            if schema == "movement-goal-canvas-two-stage-contract-v2"
            else "movement-goal-canvas-two-stage-overfit32-report-v1"
        ),
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "contract_sha256": raw["contract_sha256"],
        "config_sha256": _sha256(config_path),
        "dataset_sha256": _sha256(dataset_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "architecture": "relational-spatial-slots-gru",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "localization_stage": {
            "updates": localization_updates,
            "first": localization_first,
            "final": localization_final,
            "evaluation": {
                key: localization_evaluation[key]
                for key in (
                    "slot_cell_accuracy",
                    "slot_within_one_cell_accuracy",
                    "maximum_slot_cell_error",
                    "mean_slot_error_pixels",
                    "player_slot_error_pixels",
                    "goal_slot_error_pixels",
                )
            },
            "passed": localization_passed,
            "accuracy_metric": accuracy_name,
        },
        "action_stage": {
            "called": localization_passed,
            "updates": int(cast(int, action["maximum_updates"])) if localization_passed else 0,
            "first": action_first,
            "final": action_final,
            "evaluation": {
                key: evaluation[key] for key in ("action_accuracy", "action_loss", "action_recall")
            },
            "frozen_localizer_unchanged": frozen_unchanged,
            "passed": action_passed,
        },
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "coordinate_labels_in_actor_input": False,
        "diagnostic_checkpoint_only": True,
        "diagnostic_checkpoint_reusable_for_formal_training": False,
        "full_training_called": False,
        "next_stage_allowed": passed,
        "real_rgb_training_frames": 0,
        "test_frames_read": 0,
        "holdout_opened": False,
        "r2_allowed": False,
        "device_input_commands_sent": 0,
    }
    report["report_sha256"] = _canonical_sha256(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    architecture = str(stage.get("architecture", "task-specific"))
    if cast(dict[str, object], raw["stage_b"])["selected_architecture"] != architecture:
        raise ValueError("stage C architecture selection differs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "train-report.json"
    if report_path.exists() or any(output_dir.glob("*.safetensors")):
        raise ValueError("stage C training output already exists")
    manifest = load_stage_c_manifest(dataset_root)
    if architecture == "relational" and (
        raw.get("schema_version") != "movement-goal-canvas-relational-contract-v1"
        or raw.get("trajectory_manifest_sha256") != manifest["manifest_sha256"]
        or raw.get("data_changed") is not False
        or raw.get("epochs_changed") is not False
        or raw.get("sampling_changed") is not False
    ):
        raise ValueError("relational Stage C contract binding differs")
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
    model = _movement_model(architecture, device)
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
                    "architecture": architecture,
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
        "architecture": architecture,
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
    model: MovementModel | None = None,
    device: torch.device | None = None,
    *,
    stop_confirmation_steps: int = 1,
    record_trace: bool = False,
    navigation_only: bool = False,
    goal_canvas: bool = False,
    marker: dict[str, object] | None = None,
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
            position_raw = cast(dict[str, int], observation["self_position"])
            before = (position_raw["x"], position_raw["y"])
            if goal_canvas:
                if marker is None:
                    raise ValueError("goal canvas rollout requires marker geometry")
                from hok_agent.movement_goal_canvas import render_goal_minimap

                marked = render_goal_minimap(before, goal, render_seed, marker)
            else:
                marked = mark_visible_target(render(observation, render_seed), "opponent_hero")
            frames.append(marked)
            legal = arena.legal_actions("blue")
            legal_names = tuple(
                action for action in MOVEMENT_ACTIONS if to_arena_action(action) in legal
            )
            requested: StageAMovement | None = None
            if policy == "teacher":
                action = rule_movement_in_range(before, goal)
            elif policy == "geometry":
                if goal_canvas:
                    from hok_agent.movement_goal_canvas import goal_canvas_geometry_movement

                    action = goal_canvas_geometry_movement(marked)
                else:
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
    architecture = str(stage.get("architecture", "task-specific"))
    if (
        architecture == "relational"
        and raw.get("trajectory_manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise ValueError("relational dev manifest binding differs")
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
    goal_canvas = raw.get("actor_input") == "synthetic_minimap_rgb_with_hollow_goal_ring"
    marker = cast(dict[str, object] | None, raw.get("marker"))
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
                    goal_canvas=goal_canvas,
                    marker=marker,
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
        model = _movement_model(architecture, device)
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
                    goal_canvas=goal_canvas,
                    marker=marker,
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
        "actor_input": raw.get("actor_input", "goal_marked_rgb_only"),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
