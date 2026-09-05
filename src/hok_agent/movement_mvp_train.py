from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from hok_agent.hierarchical_p1v2_movement_branch import MovementBranch
from hok_agent.movement_mvp import MOVEMENT_ACTIONS


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
