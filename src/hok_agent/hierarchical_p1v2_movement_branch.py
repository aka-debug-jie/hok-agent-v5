from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hok_agent.hierarchical_p0_ssl import TemporalSSL
from hok_agent.hierarchical_p0_ssl_v2 import verify_representation
from hok_agent.hierarchical_p1v2 import DIRECTIONS

SCHEMA = "hok-agent-hierarchical-p1v2-movement-branch-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p1v2-movement-branch-report-v1"


class MovementBranchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BranchConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    overfit_steps: int
    overfit_batch_size: int
    overfit_accuracy: float
    overfit_maximum_loss: float
    minimum_dev_macro_f1: float
    minimum_recall: float
    minimum_shuffle_gain: float
    required_unique_predictions: int
    minimum_trainable_parameters: int
    maximum_trainable_parameters: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[BranchConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    architecture = cast(dict[str, object], payload["architecture"])
    training = cast(dict[str, object], payload["training"])
    overfit = cast(dict[str, object], payload["overfit32"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_seed0_visible_target_movement_branch_repair1"
        or architecture.get("directions") != list(DIRECTIONS)
        or training.get("device") != "cuda"
        or training.get("cublas_workspace_config") != ":4096:8"
        or training.get("checkpoint_selection")
        != "fixed_last_epoch_without_dev_selection"
        or claim.get("capability") != "local_visible_target_approach_only"
        or claim.get("lane_strategy_verified") is not False
        or claim.get("real_video_semantics_verified") is not False
        or claim.get("video_test_allowed") is not False
        or claim.get("macro_branch_training_allowed") is not False
        or claim.get("combat_branch_training_allowed") is not False
        or claim.get("policy_bundle_assembly_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("online_rl_allowed") is not False
        or claim.get("device_input_allowed") is not False
    ):
        raise MovementBranchError("P1v2 Movement branch boundary differs")
    repair = cast(dict[str, object], payload["repair_history"])
    if (
        repair.get("repairs_allowed") != 1
        or repair.get("repairs_used") != 1
        or repair.get("repair") != "deterministically_shuffle_overfit32_minibatches"
        or repair.get("model_data_optimizer_epoch_or_gate_changed") is not False
    ):
        raise MovementBranchError("P1v2 Movement branch repair differs")
    config = BranchConfig(
        int(cast(int, training["seed"])),
        int(cast(int, training["epochs"])),
        int(cast(int, training["batch_size"])),
        float(cast(float, training["learning_rate"])),
        float(cast(float, training["weight_decay"])),
        int(cast(int, overfit["steps"])),
        int(cast(int, overfit["batch_size"])),
        float(cast(float, overfit["minimum_accuracy"])),
        float(cast(float, overfit["maximum_loss"])),
        float(cast(float, gate["minimum_dev_macro_f1"])),
        float(cast(float, gate["minimum_per_direction_recall"])),
        float(cast(float, gate["minimum_gain_over_label_shuffle"])),
        int(cast(int, gate["required_unique_predictions"])),
        int(cast(int, architecture["minimum_trainable_parameters"])),
        int(cast(int, architecture["maximum_trainable_parameters"])),
    )
    return config, payload, _sha(_canonical(payload))


class MovementBranch(nn.Module):
    def __init__(self, representation_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        base = TemporalSSL()
        missing, unexpected = base.load_state_dict(representation_state, strict=False)
        if set(missing) != {"classifier.weight", "classifier.bias"} or unexpected:
            raise MovementBranchError("P1v2 Movement P0 tensor schema differs")
        self.encoder = base.encoder
        self.temporal = base.temporal
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, len(DIRECTIONS)),
        )
        for parameter in self.parameters():
            parameter.requires_grad = True
        for module in (
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.layer1,
            self.encoder.layer2,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.freeze_shared_trunk()

    def freeze_shared_trunk(self) -> None:
        for module in (
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
        ):
            module.eval()

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels, height, width = clips.shape
        features = cast(
            torch.Tensor,
            self.encoder(clips.reshape(batch * sequence, channels, height, width)),
        ).reshape(batch, sequence, 512)
        _output, hidden = self.temporal(features)
        return cast(torch.Tensor, self.head(hidden[-1]))


def _load_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        clips = torch.from_numpy(data["rgb_sequence"].copy())
        labels = torch.from_numpy(data["label"].astype(np.int64))
    return clips, labels


def _batch(clips: torch.Tensor, device: torch.device) -> torch.Tensor:
    return clips.to(device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 127.5 - 1.0


def _metrics(
    model: MovementBranch,
    clips: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(clips), batch_size):
            logits = model(_batch(clips[start : start + batch_size], device))
            predictions.append(logits.argmax(1).cpu())
    predicted = torch.cat(predictions)
    recalls, f1_scores = [], []
    for label in range(len(DIRECTIONS)):
        tp = int(((labels == label) & (predicted == label)).sum())
        fp = int(((labels != label) & (predicted == label)).sum())
        fn = int(((labels == label) & (predicted != label)).sum())
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_direction_recall": dict(zip(DIRECTIONS, recalls, strict=True)),
        "unique_predictions": len(set(predicted.tolist())),
    }


def _new_model(
    representation_path: Path, seed: int, device: torch.device
) -> MovementBranch:
    torch.manual_seed(seed)
    return MovementBranch(load_file(representation_path, device="cpu")).to(device)


def overfit_batch_order(length: int, batch_size: int, seed: int) -> torch.Tensor:
    if length % batch_size:
        raise MovementBranchError("overfit32 batch size must divide the sample count")
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed))


def _overfit32(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    representation_path: Path,
    config: BranchConfig,
    device: torch.device,
) -> dict[str, float | bool]:
    indices = torch.cat(
        [torch.nonzero(train_y == label).flatten()[:4] for label in range(len(DIRECTIONS))]
    )
    clips, labels = train_x[indices], train_y[indices]
    model = _new_model(representation_path, config.seed, device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.001
    )
    batches_per_epoch = len(clips) // config.overfit_batch_size
    order = overfit_batch_order(len(clips), config.overfit_batch_size, config.seed)
    for step in range(config.overfit_steps):
        if step and step % batches_per_epoch == 0:
            order = overfit_batch_order(
                len(clips), config.overfit_batch_size, config.seed + step // batches_per_epoch
            )
        start = step % batches_per_epoch * config.overfit_batch_size
        selected = order[start : start + config.overfit_batch_size]
        model.train()
        model.freeze_shared_trunk()
        logits = model(_batch(clips[selected], device))
        loss = nn.functional.cross_entropy(logits, labels[selected].to(device))
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    values = _metrics(model, clips, labels, device, config.batch_size)
    model.eval()
    with torch.no_grad():
        loss_value = float(
            nn.functional.cross_entropy(model(_batch(clips, device)), labels.to(device))
        )
    recalls = cast(dict[str, float], values["per_direction_recall"])
    accuracy = sum(recalls.values()) / len(DIRECTIONS)
    passed = accuracy >= config.overfit_accuracy and loss_value <= config.overfit_maximum_loss
    del model
    torch.cuda.empty_cache()
    return {"accuracy": accuracy, "loss": loss_value, "passed": passed}


def _fit(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    representation_path: Path,
    config: BranchConfig,
    device: torch.device,
    *,
    shuffle_labels: bool,
) -> tuple[MovementBranch, dict[str, object]]:
    model = _new_model(representation_path, config.seed, device)
    labels = train_y.clone()
    if shuffle_labels:
        labels = labels[torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(train_x, labels),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    for _epoch in range(config.epochs):
        model.train()
        model.freeze_shared_trunk()
        for clips, target in loader:
            loss = nn.functional.cross_entropy(model(_batch(clips, device)), target.to(device))
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    return model, _metrics(model, dev_x, dev_y, device, config.batch_size)


def _trainable_parameters(model: MovementBranch) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_branch(
    contract_path: Path,
    source_dir: Path,
    p0_run_dir: Path,
    p0_contract_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    source_report = cast(dict[str, object], json.loads((source_dir / "report.json").read_text()))
    supplied = source_report.pop("report_sha256", None)
    p0_report = verify_representation(p0_run_dir, p0_contract_path)
    if (
        supplied != _sha(_canonical(source_report))
        or supplied != contract["movement_source_report_sha256"]
        or source_report.get("movement_branch_training_allowed") is not True
        or p0_report.get("representation_sha256") != contract["p0_representation_sha256"]
    ):
        raise MovementBranchError("P1v2 Movement evidence binding differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available():
        raise MovementBranchError("P1v2 Movement deterministic host CUDA is unavailable")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    train_x, train_y = _load_dataset(source_dir / "train.npz")
    dev_x, dev_y = _load_dataset(source_dir / "dev.npz")
    representation_path = p0_run_dir / "representation.safetensors"
    overfit = _overfit32(train_x, train_y, representation_path, config, device)
    model, metrics = _fit(
        train_x, train_y, dev_x, dev_y, representation_path, config, device, shuffle_labels=False
    )
    _shuffle_model, shuffle = _fit(
        train_x, train_y, dev_x, dev_y, representation_path, config, device, shuffle_labels=True
    )
    trainable = _trainable_parameters(model)
    macro_f1 = cast(float, metrics["macro_f1"])
    shuffle_f1 = cast(float, shuffle["macro_f1"])
    recalls = cast(dict[str, float], metrics["per_direction_recall"])
    checks = {
        "overfit32": bool(overfit["passed"]),
        "dev_macro_f1": macro_f1 >= config.minimum_dev_macro_f1,
        "per_direction_recall": min(recalls.values()) >= config.minimum_recall,
        "label_shuffle_gain": macro_f1 - shuffle_f1 >= config.minimum_shuffle_gain,
        "unique_predictions": metrics["unique_predictions"] == config.required_unique_predictions,
        "parameter_budget": config.minimum_trainable_parameters
        <= trainable
        <= config.maximum_trainable_parameters,
    }
    passed = all(checks.values())
    if output_dir.exists() or output_dir.is_symlink():
        raise MovementBranchError("P1v2 Movement output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        checkpoint_sha: str | None = None
        if passed:
            checkpoint = staging / "movement-branch.safetensors"
            state = {
                key: value.detach().cpu().contiguous()
                for key, value in model.state_dict().items()
                if key.startswith(("encoder.layer3.", "encoder.layer4.", "temporal.", "head."))
            }
            save_file(
                state,
                checkpoint,
                metadata={
                    "schema_version": "hok-agent-hierarchical-p1v2-movement-branch-v1",
                    "contract_sha256": contract_sha,
                    "p0_representation_sha256": cast(str, p0_report["representation_sha256"]),
                    "capability": "local_visible_target_approach_only",
                },
            )
            checkpoint_sha = _sha(checkpoint.read_bytes())
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "status": "P1V2_MOVEMENT_BRANCH_PASSED" if passed else "P1V2_MOVEMENT_BRANCH_FAILED",
            "contract_sha256": contract_sha,
            "movement_source_report_sha256": supplied,
            "p0_representation_sha256": p0_report["representation_sha256"],
            "train_sequences": len(train_y),
            "dev_sequences": len(dev_y),
            "trainable_parameters": trainable,
            "overfit32": overfit,
            "metrics": metrics,
            "label_shuffle": shuffle,
            "checks": checks,
            "movement_branch_sha256": checkpoint_sha,
            "capability": "local_visible_target_approach_only",
            "lane_strategy_verified": False,
            "real_video_semantics_verified": False,
            "video_test_opened": False,
            "movement_branch_available": passed,
            "macro_branch_available": False,
            "combat_branch_available": False,
            "policy_bundle_assembly_allowed": False,
            "reward_allowed": False,
            "online_rl_allowed": False,
            "device_input_allowed": False,
        }
        report["report_sha256"] = _sha(_canonical(report))
        (staging / "report.json").write_bytes(_canonical(report) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1v2 task-specific Movement branch")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--p0-run-dir", type=Path, required=True)
    parser.add_argument("--p0-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_branch(
                args.contract,
                args.source_dir,
                args.p0_run_dir,
                args.p0_contract,
                args.output_dir,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
