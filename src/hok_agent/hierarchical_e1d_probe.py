from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCHEMA = "hok-agent-hierarchical-event-e1d-probe-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1d-probe-report-v1"
Mode = Literal["temporal", "last_frame", "shuffled"]


class E1dProbeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    sequence_frames: int
    input_size: int
    overfit_steps: int
    overfit_minimum_accuracy: float
    overfit_maximum_loss_fraction: float
    minimum_temporal_dev_macro_f1: float
    minimum_gain_over_last: float
    minimum_gain_over_shuffle: float


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[ProbeConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    training = cast(dict[str, object], payload["training"])
    overfit = cast(dict[str, object], payload["overfit32"])
    gate = cast(dict[str, object], payload["gate"])
    controls = cast(dict[str, object], payload["controls"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_seed0_temporal_value_probe"
        or training.get("device") != "cpu"
        or controls.get("time_or_offset_input") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
        or claim.get("video_test_allowed") is not False
    ):
        raise E1dProbeError("E1d probe boundary differs")
    config = ProbeConfig(
        int(cast(int, training["seed"])),
        int(cast(int, training["epochs"])),
        int(cast(int, training["batch_size"])),
        float(cast(float, training["learning_rate"])),
        float(cast(float, training["weight_decay"])),
        int(cast(int, training["sequence_frames"])),
        int(cast(int, training["input_size"])),
        int(cast(int, overfit["steps"])),
        float(cast(float, overfit["minimum_accuracy"])),
        float(cast(float, overfit["maximum_final_loss_fraction"])),
        float(cast(float, gate["minimum_temporal_dev_macro_f1"])),
        float(cast(float, gate["minimum_gain_over_last_frame"])),
        float(cast(float, gate["minimum_gain_over_shuffled_frames"])),
    )
    return config, payload, _sha(_canonical(payload))


class FrameEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 4, 64),
            nn.ReLU(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(frames))


class TerminalProbe(nn.Module):
    def __init__(self, mode: Mode) -> None:
        super().__init__()
        self.mode = mode
        self.encoder = FrameEncoder()
        self.temporal = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Linear(64, 2)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels, height, width = clips.shape
        features = self.encoder(clips.reshape(batch * sequence, channels, height, width))
        features = features.reshape(batch, sequence, 64)
        if self.mode == "last_frame":
            state = features[:, -1]
        else:
            _output, hidden = self.temporal(features)
            state = hidden[-1]
        return cast(torch.Tensor, self.head(state))


def _load_split(path: Path, input_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        frames = data["rgb_sequence"]
        labels = data["label"].astype(np.int64)
    stride = 128 // input_size
    frames = frames[:, :, ::stride, ::stride, :]
    tensor = torch.from_numpy(frames.copy()).permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0
    return tensor, torch.from_numpy(labels)


def _macro_f1(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    scores = []
    for label in (0, 1):
        tp = int(((labels == label) & (predictions == label)).sum())
        fp = int(((labels != label) & (predictions == label)).sum())
        fn = int(((labels == label) & (predictions != label)).sum())
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / 2


def _permute(clips: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(917)
    order = torch.randperm(clips.shape[1], generator=generator)
    return clips[:, order]


def _fit(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    config: ProbeConfig,
    mode: Mode,
) -> dict[str, float]:
    torch.manual_seed(config.seed)
    model = TerminalProbe(mode)
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
            if mode == "shuffled":
                clips = _permute(clips)
            loss = nn.functional.cross_entropy(model(clips), labels)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    model.eval()
    evaluated = _permute(dev_x) if mode == "shuffled" else dev_x
    with torch.no_grad():
        predictions = model(evaluated).argmax(dim=1)
    return {
        "accuracy": float((predictions == dev_y).float().mean()),
        "macro_f1": _macro_f1(dev_y, predictions),
    }


def _overfit32(
    train_x: torch.Tensor, train_y: torch.Tensor, config: ProbeConfig
) -> dict[str, float | bool]:
    selected = torch.cat([torch.nonzero(train_y == label).flatten()[:16] for label in (0, 1)])
    clips, labels = train_x[selected], train_y[selected]
    torch.manual_seed(config.seed)
    model = TerminalProbe("temporal")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = float(nn.functional.cross_entropy(model(clips), labels))
    for _step in range(config.overfit_steps):
        loss = nn.functional.cross_entropy(model(clips), labels)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        final = float(nn.functional.cross_entropy(model(clips), labels))
        accuracy = float((model(clips).argmax(1) == labels).float().mean())
    passed = (
        accuracy >= config.overfit_minimum_accuracy
        and final <= initial * config.overfit_maximum_loss_fraction
    )
    return {"initial_loss": initial, "final_loss": final, "accuracy": accuracy, "passed": passed}


def run_probe(contract_path: Path, dataset: Path, output_dir: Path) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    report = cast(dict[str, object], json.loads((dataset / "report.json").read_text()))
    if contract["clip_report_sha256"] != report.get("report_sha256"):
        raise E1dProbeError("E1d probe dataset binding differs")
    random.seed(config.seed)
    np.random.seed(config.seed)
    train_x, train_y = _load_split(dataset / "train.npz", config.input_size)
    dev_x, dev_y = _load_split(dataset / "dev.npz", config.input_size)
    overfit = _overfit32(train_x, train_y, config)
    metrics: dict[str, dict[str, float]] = (
        {
            mode: _fit(train_x, train_y, dev_x, dev_y, config, mode)
            for mode in ("temporal", "last_frame", "shuffled")
        }
        if bool(overfit["passed"])
        else {}
    )
    temporal = metrics.get("temporal", {}).get("macro_f1", 0.0)
    checks = {
        "overfit32": bool(overfit["passed"]),
        "temporal_dev": temporal >= config.minimum_temporal_dev_macro_f1,
        "last_frame_margin": temporal
        - metrics.get("last_frame", {}).get("macro_f1", 0.0)
        >= config.minimum_gain_over_last,
        "shuffle_margin": temporal
        - metrics.get("shuffled", {}).get("macro_f1", 0.0)
        >= config.minimum_gain_over_shuffle,
    }
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "E1D_TEMPORAL_PROBE_PASSED"
        if all(checks.values())
        else "E1D_TEMPORAL_PROBE_FAILED",
        "contract_sha256": contract_sha,
        "clip_report_sha256": report["report_sha256"],
        "overfit32": overfit,
        "metrics": metrics,
        "checks": checks,
        "device": "cpu",
        "semantic_accuracy_verified": False,
        "win_loss_verified": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "video_test_opened": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload))
    if output_dir.exists():
        raise E1dProbeError("E1d probe output exists")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_probe(args.contract, args.dataset, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
