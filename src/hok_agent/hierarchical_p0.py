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

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

SCHEMA = "hok-agent-hierarchical-p0-adapter-value-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p0-adapter-value-report-v1"


class P0Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class P0Config:
    seed: int
    steps: int
    learning_rate: float
    weight_decay: float
    minimum_macro_f1: float
    minimum_source_gain: float
    minimum_random_gain: float
    minimum_feature_std: float


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path) -> tuple[P0Config, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    probe = cast(dict[str, object], payload["probe"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_existing_adapter_value_probe"
        or probe.get("device") != "cuda"
        or claim.get("existing_adapter_immutable") is not True
        or claim.get("encoder_frozen") is not True
        or claim.get("test_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
    ):
        raise P0Error("P0 adapter-value contract boundary differs")
    config = P0Config(
        int(cast(int, probe["seed"])),
        int(cast(int, probe["steps"])),
        float(cast(float, probe["learning_rate"])),
        float(cast(float, probe["weight_decay"])),
        float(cast(float, gate["minimum_adapter_dev_macro_f1"])),
        float(cast(float, gate["minimum_gain_over_source"])),
        float(cast(float, gate["minimum_gain_over_random"])),
        float(cast(float, gate["minimum_feature_std"])),
    )
    return config, payload, _sha(_canonical(payload))


class FrozenResNet(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor] | None, *, adapter: bool = False) -> None:
        super().__init__()
        network = resnet18(weights=None, num_classes=6)
        if state is not None:
            if adapter:
                adapted = {
                    key.removeprefix("encoder."): value
                    for key, value in state.items()
                    if key.startswith("encoder.")
                }
                network.load_state_dict(adapted, strict=False)
            else:
                network.load_state_dict(state, strict=True)
        network.fc = nn.Identity()
        self.network = network.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(frames))


def _load_clips(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["rgb_sequence"], data["label"].astype(np.int64)


def _features(
    model: FrozenResNet,
    clips: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    model = model.to(device)
    descriptors = []
    with torch.no_grad():
        for start in range(0, len(clips), 8):
            batch = torch.from_numpy(clips[start : start + 8].copy())
            batch = batch.permute(0, 1, 4, 2, 3).float().to(device) / 127.5 - 1.0
            count, sequence, channels, height, width = batch.shape
            encoded = model(batch.reshape(count * sequence, channels, height, width))
            encoded = encoded.reshape(count, sequence, 512)
            delta = (encoded[:, 1:] - encoded[:, :-1]).abs().mean(dim=1)
            descriptors.append(torch.cat([encoded[:, -1], encoded.mean(dim=1), delta], dim=1).cpu())
    return torch.cat(descriptors)


class ProbeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1536, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(features))


def _macro_f1(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    values = []
    for label in (0, 1):
        tp = int(((labels == label) & (predictions == label)).sum())
        fp = int(((labels != label) & (predictions == label)).sum())
        fn = int(((labels == label) & (predictions != label)).sum())
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(values) / 2


def _probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    config: P0Config,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(config.seed)
    head = ProbeHead().to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    x, y = train_x.to(device), train_y.to(device)
    for _step in range(config.steps):
        loss = nn.functional.cross_entropy(head(x), y)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        predictions = head(dev_x.to(device)).argmax(dim=1).cpu()
    return {
        "accuracy": float((predictions == dev_y).float().mean()),
        "macro_f1": _macro_f1(dev_y, predictions),
        "feature_std": float(dev_x.std()),
    }


def run_adapter_value_probe(
    contract_path: Path,
    source_checkpoint: Path,
    adapter_checkpoint: Path,
    adapter_report: Path,
    dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    if not torch.cuda.is_available():
        raise P0Error("P0 adapter-value probe requires host CUDA")
    if _sha(source_checkpoint.read_bytes()) != contract["source_checkpoint_sha256"]:
        raise P0Error("P0 source checkpoint hash differs")
    if _sha(adapter_checkpoint.read_bytes()) != contract["adapter_checkpoint_sha256"]:
        raise P0Error("P0 adapter checkpoint hash differs")
    adapter_meta = cast(dict[str, object], json.loads(adapter_report.read_text()))
    selected = cast(dict[str, object], adapter_meta["selected"])
    dataset_meta = cast(dict[str, object], json.loads((dataset / "report.json").read_text()))
    if (
        selected.get("epoch") != contract["adapter_report_selected_epoch"]
        or selected.get("sha256") != contract["adapter_checkpoint_sha256"]
        or dataset_meta.get("report_sha256") != contract["dataset_report_sha256"]
        or adapter_meta.get("video_test_accessed") is not False
    ):
        raise P0Error("P0 adapter evidence binding differs")
    device = torch.device("cuda")
    train_frames, train_labels = _load_clips(dataset / "train.npz")
    dev_frames, dev_labels = _load_clips(dataset / "dev.npz")
    source_state = load_file(source_checkpoint, device="cpu")
    adapter_state = load_file(adapter_checkpoint, device="cpu")
    torch.manual_seed(config.seed)
    encoders = {
        "source": FrozenResNet(source_state),
        "adapter": FrozenResNet(adapter_state, adapter=True),
        "random": FrozenResNet(None),
    }
    train_y = torch.from_numpy(train_labels)
    dev_y = torch.from_numpy(dev_labels)
    metrics = {}
    for name, encoder in encoders.items():
        train_x = _features(encoder, train_frames, device)
        dev_x = _features(encoder, dev_frames, device)
        metrics[name] = _probe(train_x, train_y, dev_x, dev_y, config, device)
        del encoder
        torch.cuda.empty_cache()
    adapter_f1 = metrics["adapter"]["macro_f1"]
    checks = {
        "adapter_macro_f1": adapter_f1 >= config.minimum_macro_f1,
        "source_margin": adapter_f1 - metrics["source"]["macro_f1"] >= config.minimum_source_gain,
        "random_margin": adapter_f1 - metrics["random"]["macro_f1"] >= config.minimum_random_gain,
        "feature_not_collapsed": metrics["adapter"]["feature_std"] >= config.minimum_feature_std,
    }
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "P0_ADAPTER_VALUE_PASSED" if all(checks.values()) else "P0_ADAPTER_VALUE_FAILED",
        "contract_sha256": contract_sha,
        "metrics": metrics,
        "checks": checks,
        "selected_adapter_sha256": contract["adapter_checkpoint_sha256"]
        if all(checks.values())
        else None,
        "p0_initialization_allowed": all(checks.values()),
        "video_test_opened": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "gpu": torch.cuda.get_device_name(0),
    }
    payload["report_sha256"] = _sha(_canonical(payload))
    if output_dir.exists() or output_dir.is_symlink():
        raise P0Error("P0 output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / "report.json").write_bytes(_canonical(payload) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0 frozen adapter-value probe")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_adapter_value_probe(
        args.contract,
        args.source_checkpoint,
        args.adapter_checkpoint,
        args.adapter_report,
        args.dataset,
        args.output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
