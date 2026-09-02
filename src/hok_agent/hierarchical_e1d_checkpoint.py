from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hok_agent.hierarchical_e1d_probe import (
    Mode,
    ProbeConfig,
    TerminalProbe,
    _load_split,
    _macro_f1,
    _overfit32,
    _permute,
)
from hok_agent.hierarchical_e1d_probe import (
    load_contract as load_probe_contract,
)

SCHEMA = "hok-agent-hierarchical-event-e1d-checkpoint-config-v1"
BUNDLE_SCHEMA = "hok-agent-hierarchical-event-e1d-checkpoint-bundle-v1"


class CheckpointError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def load_checkpoint_contract(path: Path) -> tuple[dict[str, object], str]:
    payload = _load_json(path)
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_pre_test_checkpoint_contract"
        or payload.get("modes") != ["temporal", "last_frame", "shuffled"]
        or payload.get("selected_mode") != "temporal"
        or claim.get("test_allowed_before_bundle_freeze") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
    ):
        raise CheckpointError("checkpoint contract boundary differs")
    return payload, _sha(_canonical(payload))


def _train_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    config: ProbeConfig,
    mode: Mode,
) -> tuple[TerminalProbe, dict[str, float]]:
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
    return model, {
        "accuracy": float((predictions == dev_y).float().mean()),
        "macro_f1": _macro_f1(dev_y, predictions),
    }


def _state(model: TerminalProbe) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}


def verify_checkpoint_bundle(bundle_dir: Path, contract_path: Path) -> dict[str, object]:
    contract, contract_sha = load_checkpoint_contract(contract_path)
    bundle_path = bundle_dir / "bundle.json"
    data = bundle_path.read_bytes()
    payload = cast(dict[str, object], json.loads(data))
    supplied = payload.get("bundle_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "bundle_sha256"}
    if (
        payload.get("schema_version") != BUNDLE_SCHEMA
        or payload.get("contract_sha256") != contract_sha
        or supplied != _sha(_canonical(unsigned))
        or data != _canonical(payload) + b"\n"
        or payload.get("test_opened") is not False
        or payload.get("reward_allowed") is not False
    ):
        raise CheckpointError("checkpoint bundle metadata is invalid")
    checkpoints = cast(dict[str, dict[str, object]], payload["checkpoints"])
    for mode in cast(list[str], contract["modes"]):
        row = checkpoints[mode]
        path = bundle_dir / cast(str, row["filename"])
        if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != row["sha256"]:
            raise CheckpointError("checkpoint weight hash differs")
        tensors = load_file(path, device="cpu")
        expected = TerminalProbe(cast(Mode, mode)).state_dict()
        if set(tensors) != set(expected) or any(
            tensors[name].shape != expected[name].shape for name in expected
        ):
            raise CheckpointError("checkpoint tensor contract differs")
    return payload


def freeze_checkpoints(
    contract_path: Path,
    probe_contract_path: Path,
    probe_report_path: Path,
    dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract, contract_sha = load_checkpoint_contract(contract_path)
    probe_config, _probe_contract, probe_contract_sha = load_probe_contract(probe_contract_path)
    probe_report = _load_json(probe_report_path)
    dataset_report = _load_json(dataset / "report.json")
    if (
        contract["probe_contract_sha256"] != probe_contract_sha
        or contract["probe_report_sha256"] != probe_report.get("report_sha256")
        or contract["clip_report_sha256"] != dataset_report.get("report_sha256")
        or probe_report.get("status") != "E1D_TEMPORAL_PROBE_PASSED"
    ):
        raise CheckpointError("checkpoint evidence binding differs")
    train_x, train_y = _load_split(dataset / "train.npz", probe_config.input_size)
    dev_x, dev_y = _load_split(dataset / "dev.npz", probe_config.input_size)
    overfit = _overfit32(train_x, train_y, probe_config)
    if not bool(overfit["passed"]):
        raise CheckpointError("checkpoint overfit32 did not reproduce")
    required = cast(dict[str, float], contract["required_dev_macro_f1"])
    tolerance = float(cast(float, contract["metric_absolute_tolerance"]))
    if output_dir.exists() or output_dir.is_symlink():
        raise CheckpointError("checkpoint output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        checkpoints: dict[str, dict[str, object]] = {}
        metrics: dict[str, dict[str, float]] = {}
        for mode in ("temporal", "last_frame", "shuffled"):
            model, metric = _train_model(train_x, train_y, dev_x, dev_y, probe_config, mode)
            if not math.isclose(metric["macro_f1"], required[mode], abs_tol=tolerance):
                raise CheckpointError(f"{mode} metric did not reproduce")
            filename = f"{mode}.safetensors"
            path = staging / filename
            save_file(
                _state(model),
                path,
                metadata={
                    "architecture": cast(str, contract["architecture"]),
                    "mode": mode,
                    "normalization": cast(str, contract["normalization"]),
                },
            )
            checkpoints[mode] = {"filename": filename, "sha256": _sha(path.read_bytes())}
            metrics[mode] = metric
        payload: dict[str, object] = {
            "schema_version": BUNDLE_SCHEMA,
            "contract_sha256": contract_sha,
            "probe_contract_sha256": probe_contract_sha,
            "probe_report_sha256": probe_report["report_sha256"],
            "clip_report_sha256": dataset_report["report_sha256"],
            "architecture": contract["architecture"],
            "normalization": contract["normalization"],
            "seed": probe_config.seed,
            "sequence_frames": probe_config.sequence_frames,
            "input_size": probe_config.input_size,
            "selected_mode": "temporal",
            "metrics": metrics,
            "overfit32": overfit,
            "checkpoints": checkpoints,
            "test_opened": False,
            "semantic_accuracy_verified": False,
            "reward_allowed": False,
            "promotion_allowed": False,
        }
        payload["bundle_sha256"] = _sha(_canonical(payload))
        (staging / "bundle.json").write_bytes(_canonical(payload) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_checkpoint_bundle(output_dir, contract_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze pre-test E1d checkpoint bundle")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe-contract", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_checkpoints(
        args.contract,
        args.probe_contract,
        args.probe_report,
        args.dataset,
        args.output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
