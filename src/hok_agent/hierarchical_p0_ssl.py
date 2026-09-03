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
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18  # type: ignore[import-untyped]

SCHEMA = "hok-agent-hierarchical-p0-temporal-ssl-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p0-temporal-ssl-report-v1"


class P0SSLError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SSLConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    invariance_weight: float
    brightness_scales: tuple[float, float]
    overfit_steps: int
    overfit_accuracy: float
    overfit_loss_fraction: float
    minimum_dev_macro_f1: float
    minimum_baseline_gain: float
    frozen_baseline_macro_f1: float
    minimum_feature_std: float
    minimum_effective_rank: float
    minimum_augmentation_cosine: float


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path) -> tuple[SSLConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    training = cast(dict[str, object], payload["training"])
    overfit = cast(dict[str, object], payload["overfit32"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    repair = cast(dict[str, object], payload["repair_history"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_seed0_temporal_ssl_pilot"
        or training.get("device") != "cuda"
        or training.get("cublas_workspace_config") != ":4096:8"
        or repair.get("repairs_used") != 1
        or repair.get("model_data_or_gate_changed") is not False
        or claim.get("old_adapter_loaded") is not False
        or claim.get("video_test_allowed") is not False
        or claim.get("policy_training_allowed") is not False
        or claim.get("reward_allowed") is not False
    ):
        raise P0SSLError("P0 SSL boundary differs")
    scales = cast(list[float], training["brightness_scales"])
    config = SSLConfig(
        int(cast(int, training["seed"])),
        int(cast(int, training["epochs"])),
        int(cast(int, training["batch_size"])),
        float(cast(float, training["learning_rate"])),
        float(cast(float, training["weight_decay"])),
        float(cast(float, training["invariance_loss_weight"])),
        (float(scales[0]), float(scales[1])),
        int(cast(int, overfit["steps"])),
        float(cast(float, overfit["minimum_accuracy"])),
        float(cast(float, overfit["maximum_final_loss_fraction"])),
        float(cast(float, gate["minimum_dev_macro_f1"])),
        float(cast(float, gate["minimum_gain_over_best_frozen_baseline"])),
        float(cast(float, gate["best_frozen_baseline_macro_f1"])),
        float(cast(float, gate["minimum_feature_std"])),
        float(cast(float, gate["minimum_effective_rank"])),
        float(cast(float, gate["minimum_augmentation_cosine"])),
    )
    return config, payload, _sha(_canonical(payload))


class TemporalSSL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.temporal = nn.GRU(512, 128, batch_first=True)
        self.projector = nn.Linear(128, 64)
        self.classifier = nn.Linear(128, 2)

    def forward(self, clips: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, channels, height, width = clips.shape
        frame_features = cast(
            torch.Tensor,
            self.encoder(clips.reshape(batch * sequence, channels, height, width)),
        ).reshape(batch, sequence, 512)
        _output, hidden = self.temporal(frame_features)
        state = hidden[-1]
        projection = nn.functional.normalize(self.projector(state), dim=1)
        return cast(torch.Tensor, self.classifier(state)), state, projection


def _load(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        frames = data["rgb_sequence"]
        labels = data["label"].astype(np.int64)
    tensor = torch.from_numpy(frames.copy()).permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0
    return tensor, torch.from_numpy(labels)


def _macro_f1(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    scores = []
    for label in (0, 1):
        tp = int(((labels == label) & (predictions == label)).sum())
        fp = int(((labels != label) & (predictions == label)).sum())
        fn = int(((labels == label) & (predictions != label)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / 2


def _overfit32(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    config: SSLConfig,
    device: torch.device,
) -> dict[str, float | bool]:
    indices = torch.cat([torch.nonzero(train_y == label).flatten()[:16] for label in (0, 1)])
    clips, labels = train_x[indices], train_y[indices]
    torch.manual_seed(config.seed)
    model = TemporalSSL().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with torch.no_grad():
        initial = float(
            nn.functional.cross_entropy(model(clips.to(device))[0], labels.to(device))
        )
    for step in range(config.overfit_steps):
        start = (step * 8) % len(clips)
        selected = torch.arange(start, start + 8) % len(clips)
        batch_x, batch_y = clips[selected].to(device), labels[selected].to(device)
        loss = nn.functional.cross_entropy(model(batch_x)[0], batch_y)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        logits = model(clips.to(device))[0]
        final = float(nn.functional.cross_entropy(logits, labels.to(device)))
        accuracy = float((logits.argmax(1).cpu() == labels).float().mean())
    passed = accuracy >= config.overfit_accuracy and final <= initial * config.overfit_loss_fraction
    del model, clips, labels
    torch.cuda.empty_cache()
    return {"initial_loss": initial, "final_loss": final, "accuracy": accuracy, "passed": passed}


def _effective_rank(features: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(features - features.mean(dim=0, keepdim=True))
    probabilities = singular / singular.sum().clamp_min(1e-12)
    return float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))


def run_ssl(
    contract_path: Path,
    dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    dataset_report = cast(dict[str, object], json.loads((dataset / "report.json").read_text()))
    if dataset_report.get("report_sha256") != contract["dataset_report_sha256"]:
        raise P0SSLError("P0 SSL dataset binding differs")
    if not torch.cuda.is_available():
        raise P0SSLError("P0 SSL requires host CUDA")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    train_x, train_y = _load(dataset / "train.npz")
    dev_x, dev_y = _load(dataset / "dev.npz")
    overfit = _overfit32(train_x, train_y, config, device)
    metrics: dict[str, float] = {}
    model: TemporalSSL | None = None
    if bool(overfit["passed"]):
        torch.manual_seed(config.seed)
        model = TemporalSSL().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        model.train()
        for _epoch in range(config.epochs):
            for clips, labels in loader:
                clips, labels = clips.to(device), labels.to(device)
                logits, _state, first = model((clips * config.brightness_scales[0]).clamp(-1, 1))
                _logits, _state, second = model((clips * config.brightness_scales[1]).clamp(-1, 1))
                order_loss = nn.functional.cross_entropy(logits, labels)
                invariance = 1.0 - (first * second).sum(dim=1).mean()
                loss = order_loss + config.invariance_weight * invariance
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, states, first = model(dev_x.to(device))
            _other_logits, _other_states, second = model(
                (dev_x.to(device) * config.brightness_scales[1]).clamp(-1, 1)
            )
            predictions = logits.argmax(dim=1).cpu()
            metrics = {
                "dev_accuracy": float((predictions == dev_y).float().mean()),
                "dev_macro_f1": _macro_f1(dev_y, predictions),
                "feature_std": float(states.std()),
                "effective_rank": _effective_rank(states),
                "augmentation_cosine": float((first * second).sum(dim=1).mean()),
            }
    checks = {
        "overfit32": bool(overfit["passed"]),
        "dev_macro_f1": metrics.get("dev_macro_f1", 0.0) >= config.minimum_dev_macro_f1,
        "baseline_margin": metrics.get("dev_macro_f1", 0.0) - config.frozen_baseline_macro_f1
        >= config.minimum_baseline_gain,
        "feature_std": metrics.get("feature_std", 0.0) >= config.minimum_feature_std,
        "effective_rank": metrics.get("effective_rank", 0.0) >= config.minimum_effective_rank,
        "augmentation_cosine": metrics.get("augmentation_cosine", 0.0)
        >= config.minimum_augmentation_cosine,
    }
    passed = all(checks.values()) and model is not None
    if output_dir.exists() or output_dir.is_symlink():
        raise P0SSLError("P0 SSL output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        encoder_sha: str | None = None
        if passed and model is not None:
            path = staging / "encoder.safetensors"
            save_file(
                {
                    key.removeprefix("encoder."): value.detach().cpu().contiguous()
                    for key, value in model.state_dict().items()
                    if key.startswith("encoder.")
                },
                path,
                metadata={"architecture": "resnet18-fc-identity", "seed": str(config.seed)},
            )
            encoder_sha = _sha(path.read_bytes())
        payload: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "status": "P0_TEMPORAL_SSL_PASSED" if passed else "P0_TEMPORAL_SSL_FAILED",
            "contract_sha256": contract_sha,
            "dataset_report_sha256": dataset_report["report_sha256"],
            "overfit32": overfit,
            "metrics": metrics,
            "checks": checks,
            "encoder_sha256": encoder_sha,
            "p0_initialization_allowed": passed,
            "old_adapter_loaded": False,
            "video_test_opened": False,
            "policy_training_allowed": False,
            "reward_allowed": False,
            "promotion_allowed": False,
            "gpu": torch.cuda.get_device_name(0),
        }
        payload["report_sha256"] = _sha(_canonical(payload))
        (staging / "report.json").write_bytes(_canonical(payload) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0 temporal SSL pilot")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_ssl(args.contract, args.dataset, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
