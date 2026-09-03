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

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from hok_agent.global_policy import GlobalWindowDataset, load_global_manifest
from hok_agent.hierarchical_p0_ssl import TemporalSSL
from hok_agent.hierarchical_p0_ssl_v2 import verify_representation

SCHEMA = "hok-agent-hierarchical-p1-macro-head-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p1-macro-head-report-v1"
ACTIVE_INTENTS = ("FARM_LANE", "PUSH_STRUCTURE", "ENGAGE")


class P1MacroHeadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MacroHeadConfig:
    seed: int
    epochs: int
    batch_size: int
    feature_batch_size: int
    learning_rate: float
    weight_decay: float
    overfit_steps: int
    overfit_accuracy: float
    overfit_maximum_loss: float
    minimum_dev_macro_f1: float
    minimum_recall: float
    minimum_time_gain: float
    time_only_macro_f1: float
    minimum_shuffle_gain: float
    required_unique_predictions: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[MacroHeadConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    architecture = cast(dict[str, object], payload["architecture"])
    training = cast(dict[str, object], payload["training"])
    overfit = cast(dict[str, object], payload["overfit32"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_seed0_simulator_macro_head"
        or architecture.get("active_intents") != list(ACTIVE_INTENTS)
        or architecture.get("router_owned_intents") != ["DISENGAGE", "RECALL"]
        or training.get("device") != "cuda"
        or training.get("cublas_workspace_config") != ":4096:8"
        or training.get("checkpoint_selection")
        != "fixed_last_epoch_without_dev_selection"
        or claim.get("p0_representation_frozen") is not True
        or claim.get("simulator_only") is not True
        or claim.get("real_video_semantics_verified") is not False
        or claim.get("video_test_allowed") is not False
        or claim.get("movement_head_training_allowed") is not False
        or claim.get("combat_head_training_allowed") is not False
        or claim.get("policy_bundle_assembly_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("device_input_allowed") is not False
    ):
        raise P1MacroHeadError("P1 Macro Head boundary differs")
    config = MacroHeadConfig(
        int(cast(int, training["seed"])),
        int(cast(int, training["epochs"])),
        int(cast(int, training["batch_size"])),
        int(cast(int, training["feature_batch_size"])),
        float(cast(float, training["learning_rate"])),
        float(cast(float, training["weight_decay"])),
        int(cast(int, overfit["steps"])),
        float(cast(float, overfit["minimum_accuracy"])),
        float(cast(float, overfit["maximum_loss"])),
        float(cast(float, gate["minimum_dev_macro_f1"])),
        float(cast(float, gate["minimum_per_class_recall"])),
        float(cast(float, gate["minimum_gain_over_time_only"])),
        float(cast(float, gate["time_only_macro_f1"])),
        float(cast(float, gate["minimum_gain_over_label_shuffle"])),
        int(cast(int, gate["required_unique_predictions"])),
    )
    return config, payload, _sha(_canonical(payload))


class MacroHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, len(ACTIVE_INTENTS)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(state))


def compose_policy_canvas(
    main: torch.Tensor, minimap: torch.Tensor, hud: torch.Tensor
) -> torch.Tensor:
    if main.ndim != 5 or main.shape[-2:] != (128, 128):
        raise P1MacroHeadError("P1 Macro main-view shape differs")
    batch, sequence = main.shape[:2]

    def resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        flat = value.reshape(batch * sequence, *value.shape[2:])
        resized = nn.functional.interpolate(flat, size=size, mode="bilinear", align_corners=False)
        return cast(torch.Tensor, resized.reshape(batch, sequence, 3, *size))

    canvas = main.clone()
    canvas[:, :, :, :48, :48] = resize(minimap, (48, 48))
    canvas[:, :, :, 104:128, 16:112] = resize(hud, (24, 96))
    return canvas


def _active_subset(dataset: GlobalWindowDataset) -> Subset[tuple[torch.Tensor, ...]]:
    indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if int(sample[3]) < len(ACTIVE_INTENTS)
    ]
    return Subset(dataset, indices)


def _load_representation(path: Path, device: torch.device) -> TemporalSSL:
    model = TemporalSSL()
    missing, unexpected = model.load_state_dict(load_file(path, device="cpu"), strict=False)
    if set(missing) != {"classifier.weight", "classifier.bias"} or unexpected:
        raise P1MacroHeadError("P1 Macro P0 tensor schema differs")
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device).eval()


def _features(
    dataset: GlobalWindowDataset,
    representation: TemporalSSL,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states, labels = [], []
    loader = DataLoader(_active_subset(dataset), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for main, minimap, hud, intent, _zone, _scene, _tick in loader:
            canvas = compose_policy_canvas(main, minimap, hud).to(device)
            _logits, state, _projection = representation(canvas * 2.0 - 1.0)
            states.append(state.cpu())
            labels.append(intent)
    return torch.cat(states), torch.cat(labels)


def _fit_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    config: MacroHeadConfig,
    device: torch.device,
    *,
    shuffle_labels: bool,
) -> tuple[MacroHead, dict[str, object]]:
    torch.manual_seed(config.seed)
    model = MacroHead().to(device)
    labels = train_y.clone()
    if shuffle_labels:
        labels = labels[torch.randperm(len(labels), generator=torch.Generator().manual_seed(0))]
    counts = torch.bincount(labels, minlength=3).float()
    weights = (counts.sum() / counts.clamp_min(1)).to(device)
    weights /= weights.mean()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = DataLoader(
        TensorDataset(train_x, labels),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    for _epoch in range(config.epochs):
        model.train()
        for features, target in loader:
            logits = model(features.to(device))
            loss = nn.functional.cross_entropy(logits, target.to(device), weight=weights)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(dev_x.to(device)).argmax(1).cpu()
    recalls = []
    f1_scores = []
    for label in range(3):
        tp = int(((dev_y == label) & (predictions == label)).sum())
        fp = int(((dev_y != label) & (predictions == label)).sum())
        fn = int(((dev_y == label) & (predictions != label)).sum())
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return model, {
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class_recall": dict(zip(ACTIVE_INTENTS, recalls, strict=True)),
        "unique_predictions": len(set(predictions.tolist())),
    }


def _overfit32(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    config: MacroHeadConfig,
    device: torch.device,
) -> dict[str, float | bool]:
    selected = torch.cat(
        [torch.nonzero(train_y == label).flatten()[:10] for label in range(3)]
        + [torch.nonzero(train_y == label).flatten()[10:11] for label in range(2)]
    )
    features, labels = train_x[selected].to(device), train_y[selected].to(device)
    torch.manual_seed(config.seed)
    model = MacroHead().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _step in range(config.overfit_steps):
        loss = nn.functional.cross_entropy(model(features), labels)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        logits = model(features)
        loss_value = float(nn.functional.cross_entropy(logits, labels))
        accuracy = float((logits.argmax(1) == labels).float().mean())
    passed = accuracy >= config.overfit_accuracy and loss_value <= config.overfit_maximum_loss
    return {"accuracy": accuracy, "loss": loss_value, "passed": passed}


def run_macro_head(
    contract_path: Path,
    macro_data_report_path: Path,
    dataset_root: Path,
    p0_run_dir: Path,
    p0_contract_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available():
        raise P1MacroHeadError("P1 Macro deterministic host CUDA is unavailable")
    macro_data = cast(dict[str, object], json.loads(macro_data_report_path.read_text()))
    supplied = macro_data.pop("report_sha256", None)
    macro_data["report_sha256"] = supplied
    manifest = load_global_manifest(dataset_root)
    p0_report = verify_representation(p0_run_dir, p0_contract_path)
    if (
        supplied != _sha(_canonical({k: v for k, v in macro_data.items() if k != "report_sha256"}))
        or supplied != contract["macro_data_report_sha256"]
        or macro_data.get("macro_head_training_allowed") is not True
        or manifest.get("manifest_sha256") != contract["global_dataset_manifest_sha256"]
        or p0_report.get("representation_sha256") != contract["p0_representation_sha256"]
    ):
        raise P1MacroHeadError("P1 Macro training evidence differs")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    representation = _load_representation(p0_run_dir / "representation.safetensors", device)
    train_x, train_y = _features(
        GlobalWindowDataset(dataset_root, "train"),
        representation,
        device,
        config.feature_batch_size,
    )
    dev_x, dev_y = _features(
        GlobalWindowDataset(dataset_root, "dev"),
        representation,
        device,
        config.feature_batch_size,
    )
    overfit = _overfit32(train_x, train_y, config, device)
    model, metrics = _fit_head(
        train_x, train_y, dev_x, dev_y, config, device, shuffle_labels=False
    )
    _shuffle_model, shuffle = _fit_head(
        train_x, train_y, dev_x, dev_y, config, device, shuffle_labels=True
    )
    macro_f1 = cast(float, metrics["macro_f1"])
    recalls = cast(dict[str, float], metrics["per_class_recall"])
    shuffle_f1 = cast(float, shuffle["macro_f1"])
    checks = {
        "overfit32": bool(overfit["passed"]),
        "dev_macro_f1": macro_f1 >= config.minimum_dev_macro_f1,
        "per_class_recall": min(recalls.values()) >= config.minimum_recall,
        "time_only_gain": macro_f1 - config.time_only_macro_f1 >= config.minimum_time_gain,
        "label_shuffle_gain": macro_f1 - shuffle_f1 >= config.minimum_shuffle_gain,
        "unique_predictions": metrics["unique_predictions"] == config.required_unique_predictions,
    }
    passed = all(checks.values())
    if output_dir.exists() or output_dir.is_symlink():
        raise P1MacroHeadError("P1 Macro output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        checkpoint_sha: str | None = None
        if passed:
            checkpoint = staging / "macro-head.safetensors"
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in model.state_dict().items()
                },
                checkpoint,
                metadata={
                    "schema_version": "hok-agent-hierarchical-p1-macro-head-v1",
                    "contract_sha256": contract_sha,
                    "p0_representation_sha256": cast(str, p0_report["representation_sha256"]),
                    "simulator_only": "true",
                },
            )
            checkpoint_sha = _sha(checkpoint.read_bytes())
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "status": "P1_MACRO_HEAD_PASSED" if passed else "P1_MACRO_HEAD_FAILED",
            "contract_sha256": contract_sha,
            "macro_data_report_sha256": supplied,
            "global_dataset_manifest_sha256": manifest["manifest_sha256"],
            "p0_representation_sha256": p0_report["representation_sha256"],
            "train_windows": len(train_y),
            "dev_windows": len(dev_y),
            "overfit32": overfit,
            "metrics": metrics,
            "label_shuffle": shuffle,
            "checks": checks,
            "macro_head_sha256": checkpoint_sha,
            "p0_representation_frozen": True,
            "simulator_only": True,
            "real_video_semantics_verified": False,
            "video_test_opened": False,
            "macro_head_training_allowed": passed,
            "movement_head_training_allowed": False,
            "combat_head_training_allowed": False,
            "policy_bundle_assembly_allowed": False,
            "reward_allowed": False,
            "promotion_allowed": False,
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
    parser = argparse.ArgumentParser(description="P1 frozen-P0 simulator Macro Head")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--macro-data-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--p0-run-dir", type=Path, required=True)
    parser.add_argument("--p0-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_macro_head(
                args.contract,
                args.macro_data_report,
                args.dataset_root,
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
