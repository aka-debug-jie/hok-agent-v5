from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

from hok_agent.hierarchical_p0 import FrozenResNet

SCHEMA = "hok-agent-hierarchical-p0-temporal-probe-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p0-temporal-probe-report-v1"
HeadMode = Literal["temporal", "last_frame"]


class P0TemporalProbeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    seed: int
    steps: int
    learning_rate: float
    weight_decay: float
    hidden_size: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path) -> tuple[ProbeConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    probe = cast(dict[str, object], payload["probe"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_encoder_temporal_order_probe"
        or probe.get("device") != "cuda"
        or claim.get("encoders_frozen") is not True
        or claim.get("test_allowed") is not False
        or claim.get("reward_allowed") is not False
    ):
        raise P0TemporalProbeError("P0 temporal probe boundary differs")
    config = ProbeConfig(
        int(cast(int, probe["seed"])),
        int(cast(int, probe["steps"])),
        float(cast(float, probe["learning_rate"])),
        float(cast(float, probe["weight_decay"])),
        int(cast(int, probe["hidden_size"])),
    )
    return config, payload, _sha(_canonical(payload))


def _load(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        return data["rgb_sequence"], torch.from_numpy(data["label"].astype(np.int64))


def _extract(model: FrozenResNet, clips: np.ndarray, device: torch.device) -> torch.Tensor:
    model = model.to(device)
    output = []
    with torch.no_grad():
        for start in range(0, len(clips), 8):
            batch = torch.from_numpy(clips[start : start + 8].copy())
            batch = batch.permute(0, 1, 4, 2, 3).float().to(device) / 127.5 - 1.0
            count, sequence, channels, height, width = batch.shape
            features = model(batch.reshape(count * sequence, channels, height, width))
            output.append(features.reshape(count, sequence, 512).cpu())
    return torch.cat(output)


class TemporalHead(nn.Module):
    def __init__(self, hidden_size: int, mode: HeadMode) -> None:
        super().__init__()
        self.mode = mode
        self.gru = nn.GRU(512, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size if mode == "temporal" else 512, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.mode == "last_frame":
            state = features[:, -1]
        else:
            _output, hidden = self.gru(features)
            state = hidden[-1]
        return cast(torch.Tensor, self.head(state))


def _macro_f1(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    scores = []
    for label in (0, 1):
        tp = int(((labels == label) & (predictions == label)).sum())
        fp = int(((labels != label) & (predictions == label)).sum())
        fn = int(((labels == label) & (predictions != label)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / 2


def _fit(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    config: ProbeConfig,
    device: torch.device,
    mode: HeadMode,
) -> dict[str, float]:
    torch.manual_seed(config.seed)
    model = TemporalHead(config.hidden_size, mode).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    x, y = train_x.to(device), train_y.to(device)
    for _step in range(config.steps):
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        predictions = model(dev_x.to(device)).argmax(dim=1).cpu()
    return {
        "accuracy": float((predictions == dev_y).float().mean()),
        "macro_f1": _macro_f1(dev_y, predictions),
    }


def run_probe(
    contract_path: Path,
    source_checkpoint: Path,
    adapter_checkpoint: Path,
    dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    dataset_report = cast(dict[str, object], json.loads((dataset / "report.json").read_text()))
    if (
        dataset_report.get("report_sha256") != contract["dataset_report_sha256"]
        or _sha(source_checkpoint.read_bytes()) != contract["source_checkpoint_sha256"]
        or _sha(adapter_checkpoint.read_bytes()) != contract["adapter_checkpoint_sha256"]
    ):
        raise P0TemporalProbeError("P0 temporal evidence binding differs")
    if not torch.cuda.is_available():
        raise P0TemporalProbeError("P0 temporal probe requires host CUDA")
    device = torch.device("cuda")
    train_frames, train_y = _load(dataset / "train.npz")
    dev_frames, dev_y = _load(dataset / "dev.npz")
    source_state = load_file(source_checkpoint, device="cpu")
    adapter_state = load_file(adapter_checkpoint, device="cpu")
    torch.manual_seed(config.seed)
    encoders = {
        "source": FrozenResNet(source_state),
        "adapter": FrozenResNet(adapter_state, adapter=True),
        "random": FrozenResNet(None),
    }
    features = {}
    for name, encoder in encoders.items():
        features[name] = (
            _extract(encoder, train_frames, device),
            _extract(encoder, dev_frames, device),
        )
        del encoder
        torch.cuda.empty_cache()
    metrics = {
        name: _fit(train_x, train_y, dev_x, dev_y, config, device, "temporal")
        for name, (train_x, dev_x) in features.items()
    }
    metrics["adapter_last_frame"] = _fit(
        features["adapter"][0],
        train_y,
        features["adapter"][1],
        dev_y,
        config,
        device,
        "last_frame",
    )
    gate = cast(dict[str, object], contract["gate"])
    adapter_f1 = metrics["adapter"]["macro_f1"]
    checks = {
        "adapter_temporal": adapter_f1
        >= float(cast(float, gate["minimum_adapter_temporal_macro_f1"])),
        "source_margin": adapter_f1 - metrics["source"]["macro_f1"]
        >= float(cast(float, gate["minimum_gain_over_source"])),
        "random_margin": adapter_f1 - metrics["random"]["macro_f1"]
        >= float(cast(float, gate["minimum_gain_over_random"])),
        "last_frame_margin": adapter_f1 - metrics["adapter_last_frame"]["macro_f1"]
        >= float(cast(float, gate["minimum_gain_over_adapter_last_frame"])),
        "last_frame_ceiling": metrics["adapter_last_frame"]["macro_f1"]
        <= float(cast(float, gate["maximum_adapter_last_frame_macro_f1"])),
    }
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "P0_TEMPORAL_ADAPTER_PASSED" if all(checks.values()) else "P0_TEMPORAL_ADAPTER_FAILED"
        ),
        "contract_sha256": contract_sha,
        "metrics": metrics,
        "checks": checks,
        "p0_initialization_allowed": all(checks.values()),
        "selected_adapter_sha256": contract["adapter_checkpoint_sha256"]
        if all(checks.values())
        else None,
        "video_test_opened": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "gpu": torch.cuda.get_device_name(0),
    }
    payload["report_sha256"] = _sha(_canonical(payload))
    if output_dir.exists() or output_dir.is_symlink():
        raise P0TemporalProbeError("P0 temporal output exists")
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
    parser = argparse.ArgumentParser(description="P0 temporal frozen-encoder value probe")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_probe(
        args.contract,
        args.source_checkpoint,
        args.adapter_checkpoint,
        args.dataset,
        args.output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
