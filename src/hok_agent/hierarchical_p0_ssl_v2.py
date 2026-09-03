from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hok_agent.hierarchical_p0_ssl import TemporalSSL, _effective_rank, _macro_f1

SCHEMA = "hok-agent-hierarchical-p0-temporal-ssl-v2-config-v1"
INDEX_SCHEMA = "hok-agent-hierarchical-p0-temporal-ssl-v2-index-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p0-temporal-ssl-v2-report-v1"


class P0SSLV2Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SSLV2Config:
    seed: int
    epochs: int
    pair_batch_size: int
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


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[SSLV2Config, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    training = cast(dict[str, object], payload["training"])
    overfit = cast(dict[str, object], payload["overfit32"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_seed0_train_only_temporal_ssl"
        or training.get("device") != "cuda"
        or training.get("cublas_workspace_config") != ":4096:8"
        or training.get("checkpoint_selection")
        != "fixed_last_epoch_without_dev_selection"
        or claim.get("v1_checkpoint_loaded") is not False
        or claim.get("old_adapter_loaded") is not False
        or claim.get("training_video_dev_allowed") is not False
        or claim.get("video_test_allowed") is not False
        or claim.get("policy_training_allowed") is not False
        or claim.get("reward_allowed") is not False
    ):
        raise P0SSLV2Error("P0 SSL v2 boundary differs")
    scales = cast(list[float], training["brightness_scales"])
    config = SSLV2Config(
        int(cast(int, training["seed"])),
        int(cast(int, training["epochs"])),
        int(cast(int, training["pair_batch_size"])),
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


def _self_hashed(path: Path, field: str, schema: str) -> dict[str, object]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    supplied = payload.pop(field, None)
    if payload.get("schema_version") != schema or supplied != _sha(_canonical(payload)):
        raise P0SSLV2Error(f"P0 SSL v2 {path.name} binding differs")
    payload[field] = supplied
    return payload


def _load_train_clips(
    index_path: Path, target_dataset: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = _self_hashed(index_path, "index_sha256", INDEX_SCHEMA)
    rows = cast(list[dict[str, object]], index["windows"])
    by_shard: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for position, row in enumerate(rows):
        by_shard[cast(str, row["shard"])].append((position, row))
    clips = np.empty((len(rows), 16, 128, 128, 3), dtype=np.uint8)
    permutations = np.empty((len(rows), 16), dtype=np.int64)
    reversed_pairs = np.empty(len(rows), dtype=np.bool_)
    for shard_name, selected in by_shard.items():
        path = target_dataset / "shards" / shard_name
        data = path.read_bytes()
        expected = cast(str, selected[0][1]["shard_sha256"])
        if _sha(data) != expected or any(row["shard_sha256"] != expected for _, row in selected):
            raise P0SSLV2Error("P0 SSL v2 indexed shard hash differs")
        with np.load(io.BytesIO(data), allow_pickle=False) as shard:
            frames = shard["frames"]
            for position, row in selected:
                indices = np.asarray(row["frame_indices"], dtype=np.int64)
                clips[position] = frames[indices]
                permutations[position] = np.asarray(row["middle_permutation"], dtype=np.int64)
                reversed_pairs[position] = bool(row["pair_reversed"])
    return (
        torch.from_numpy(clips),
        torch.from_numpy(permutations),
        torch.from_numpy(reversed_pairs),
    )


def _pair_batch(
    clips: torch.Tensor,
    permutations: torch.Tensor,
    reversed_pairs: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chronological = (
        clips.to(device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 127.5 - 1.0
    )
    gather = permutations.to(device).view(len(clips), 16, 1, 1, 1).expand_as(chronological)
    shuffled = torch.gather(chronological, 1, gather)
    samples = torch.stack((chronological, shuffled), dim=1)
    labels = torch.tensor((0, 1), device=device).expand(len(clips), 2).clone()
    reverse = reversed_pairs.to(device)
    samples[reverse] = samples[reverse].flip(1)
    labels[reverse] = labels[reverse].flip(1)
    return samples.flatten(0, 1), labels.flatten()


def _overfit32(
    clips: torch.Tensor,
    permutations: torch.Tensor,
    reversed_pairs: torch.Tensor,
    config: SSLV2Config,
    device: torch.device,
) -> dict[str, float | bool]:
    clips, permutations, reversed_pairs = clips[:16], permutations[:16], reversed_pairs[:16]
    samples, labels = _pair_batch(clips, permutations, reversed_pairs, device)
    torch.manual_seed(config.seed)
    model = TemporalSSL().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with torch.no_grad():
        initial = float(nn.functional.cross_entropy(model(samples)[0], labels))
    for step in range(config.overfit_steps):
        start = (step * 8) % len(samples)
        selected = torch.arange(start, start + 8, device=device) % len(samples)
        loss = nn.functional.cross_entropy(model(samples[selected])[0], labels[selected])
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        logits = model(samples)[0]
        final = float(nn.functional.cross_entropy(logits, labels))
        accuracy = float((logits.argmax(1) == labels).float().mean())
    passed = accuracy >= config.overfit_accuracy and final <= initial * config.overfit_loss_fraction
    del model, samples, labels
    torch.cuda.empty_cache()
    return {"initial_loss": initial, "final_loss": final, "accuracy": accuracy, "passed": passed}


def _evaluate(
    model: TemporalSSL,
    dataset: Path,
    config: SSLV2Config,
    device: torch.device,
) -> dict[str, float]:
    with np.load(dataset / "dev.npz", allow_pickle=False) as data:
        clips = torch.from_numpy(data["rgb_sequence"].copy())
        labels = torch.from_numpy(data["label"].astype(np.int64))
    predictions, states, first, second = [], [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(clips), 8):
            batch = clips[start : start + 8].to(device=device, dtype=torch.float32)
            batch = batch.permute(0, 1, 4, 2, 3) / 127.5 - 1.0
            logits, state, projection = model(batch)
            _logits, _state, augmented = model(
                (batch * config.brightness_scales[1]).clamp(-1, 1)
            )
            predictions.append(logits.argmax(1).cpu())
            states.append(state.cpu())
            first.append(projection.cpu())
            second.append(augmented.cpu())
    predicted = torch.cat(predictions)
    state_values = torch.cat(states)
    first_values, second_values = torch.cat(first), torch.cat(second)
    return {
        "dev_accuracy": float((predicted == labels).float().mean()),
        "dev_macro_f1": _macro_f1(labels, predicted),
        "feature_std": float(state_values.std()),
        "effective_rank": _effective_rank(state_values),
        "augmentation_cosine": float((first_values * second_values).sum(dim=1).mean()),
    }


def run_ssl_v2(
    contract_path: Path,
    data_dir: Path,
    target_dataset: Path,
    evaluation_dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise P0SSLV2Error("P0 SSL v2 deterministic CuBLAS workspace differs")
    data_report = _self_hashed(
        data_dir / "report.json",
        "report_sha256",
        "hok-agent-hierarchical-p0-temporal-ssl-v2-data-report-v1",
    )
    index = _self_hashed(data_dir / "index.json", "index_sha256", INDEX_SCHEMA)
    evaluation_report = cast(
        dict[str, object], json.loads((evaluation_dataset / "report.json").read_text())
    )
    if (
        data_report.get("report_sha256") != contract["data_report_sha256"]
        or data_report.get("ssl_training_allowed") is not True
        or index.get("index_sha256") != contract["index_sha256"]
        or evaluation_report.get("report_sha256")
        != contract["evaluation_dataset_report_sha256"]
    ):
        raise P0SSLV2Error("P0 SSL v2 evidence binding differs")
    if not torch.cuda.is_available():
        raise P0SSLV2Error("P0 SSL v2 requires host CUDA")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    clips, permutations, reversed_pairs = _load_train_clips(data_dir / "index.json", target_dataset)
    overfit = _overfit32(clips, permutations, reversed_pairs, config, device)
    metrics: dict[str, float] = {}
    model: TemporalSSL | None = None
    if bool(overfit["passed"]):
        torch.manual_seed(config.seed)
        model = TemporalSSL().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loader = DataLoader(
            TensorDataset(clips, permutations, reversed_pairs),
            batch_size=config.pair_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        model.train()
        for _epoch in range(config.epochs):
            for batch_clips, batch_permutations, batch_reversed in loader:
                samples, labels = _pair_batch(
                    batch_clips, batch_permutations, batch_reversed, device
                )
                logits, _state, first = model(
                    (samples * config.brightness_scales[0]).clamp(-1, 1)
                )
                _logits, _state, second = model(
                    (samples * config.brightness_scales[1]).clamp(-1, 1)
                )
                order_loss = nn.functional.cross_entropy(logits, labels)
                invariance = 1.0 - (first * second).sum(dim=1).mean()
                loss = order_loss + config.invariance_weight * invariance
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        metrics = _evaluate(model, evaluation_dataset, config, device)
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
        raise P0SSLV2Error("P0 SSL v2 output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        checkpoint_sha: str | None = None
        if passed and model is not None:
            path = staging / "representation.safetensors"
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in model.state_dict().items()
                    if not key.startswith("classifier.")
                },
                path,
                metadata={
                    "architecture": "resnet18-gru128-project64",
                    "contract_sha256": contract_sha,
                    "seed": str(config.seed),
                },
            )
            checkpoint_sha = _sha(path.read_bytes())
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "status": "P0_TEMPORAL_SSL_V2_PASSED" if passed else "P0_TEMPORAL_SSL_V2_FAILED",
            "contract_sha256": contract_sha,
            "data_report_sha256": data_report["report_sha256"],
            "index_sha256": index["index_sha256"],
            "evaluation_dataset_report_sha256": evaluation_report["report_sha256"],
            "train_pairs": len(clips),
            "overfit32": overfit,
            "metrics": metrics,
            "checks": checks,
            "representation_sha256": checkpoint_sha,
            "gpu": torch.cuda.get_device_name(0),
            "v1_checkpoint_loaded": False,
            "old_adapter_loaded": False,
            "training_video_dev_opened": False,
            "evaluation_dev_opened": bool(overfit["passed"]),
            "video_test_opened": False,
            "p0_initialization_allowed": passed,
            "policy_training_allowed": False,
            "reward_allowed": False,
            "promotion_allowed": False,
        }
        report["report_sha256"] = _sha(_canonical(report))
        (staging / "report.json").write_bytes(_canonical(report) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def verify_representation(output_dir: Path, contract_path: Path) -> dict[str, object]:
    _config, _contract, contract_sha = load_contract(contract_path)
    report = _self_hashed(output_dir / "report.json", "report_sha256", REPORT_SCHEMA)
    checkpoint = output_dir / "representation.safetensors"
    if (
        report.get("status") != "P0_TEMPORAL_SSL_V2_PASSED"
        or report.get("contract_sha256") != contract_sha
        or report.get("p0_initialization_allowed") is not True
        or report.get("video_test_opened") is not False
        or report.get("policy_training_allowed") is not False
        or report.get("reward_allowed") is not False
        or not checkpoint.is_file()
        or _sha(checkpoint.read_bytes()) != report.get("representation_sha256")
    ):
        raise P0SSLV2Error("P0 SSL v2 representation binding differs")
    state = load_file(checkpoint, device="cpu")
    prefixes = {key.split(".", 1)[0] for key in state}
    if not state or prefixes != {"encoder", "temporal", "projector"}:
        raise P0SSLV2Error("P0 SSL v2 representation tensors differ")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0 temporal SSL v2")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_ssl_v2(
                args.contract,
                args.data_dir,
                args.target_dataset,
                args.evaluation_dataset,
                args.output_dir,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
