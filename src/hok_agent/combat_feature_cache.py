"""Materialize frozen visual features once for fast causal combat training."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.t8 import (
    V2_ADAPTER_SCHEMA,
    _canonical,
    _head_metrics,
    _large_existing,
    _large_new,
    _load_v2_adapter,
    _sha,
    _V2ResidualBlock,
    _v25_rows,
    _V25FrameCache,
)

FEATURE_CACHE_SCHEMA = "hok-agent-global-combat-feature-cache-v1"
WINDOW_FRAMES = 32
FEATURE_SIZE = 1024
FEATURE_MODEL_SCHEMA = "hok-agent-global-combat-feature-model-v1"


class CombatFeatureCacheError(ValueError):
    pass


def _encode_views(
    encoder: torch.nn.Module, views: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    if views.ndim != 5 or views.shape[1:] != (2, 128, 128, 3) or views.dtype != np.uint8:
        raise CombatFeatureCacheError("combat views are invalid")
    encoded: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(views), batch_size):
            batch = views[start : start + batch_size]
            tensor = (
                torch.from_numpy(batch)
                .to(device)
                .permute(0, 1, 4, 2, 3)
                .reshape(-1, 3, 128, 128)
                .float()
                .div(255.0)
            )
            values = cast(torch.Tensor, encoder(tensor)).reshape(len(batch), FEATURE_SIZE)
            encoded.append(values.cpu().numpy().astype(np.float16))
    return np.concatenate(encoded)


def materialize_global_combat_features(
    *,
    dataset_root: Path,
    split_path: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 128,
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise CombatFeatureCacheError("feature-cache runtime settings are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise CombatFeatureCacheError("CUDA is unavailable")
    target = torch.device(device)
    root, train_rows, split = _v25_rows(dataset_root, split_path, "train")
    dev_root, dev_rows, dev_split = _v25_rows(dataset_root, split_path, "dev")
    if root != dev_root or split.get("split_sha256") != dev_split.get("split_sha256"):
        raise CombatFeatureCacheError("combat split changed during feature materialization")
    adapter = _large_existing(adapter_checkpoint)
    state, metadata = _load_v2_adapter(adapter, target)
    if metadata.get("schema") != V2_ADAPTER_SCHEMA:
        raise CombatFeatureCacheError("video adapter schema differs")
    encoder = resnet18(weights=None)
    encoder.fc = torch.nn.Identity()
    encoder.load_state_dict(state, strict=True)
    encoder.to(target).eval()
    output = _large_new(output_dir)
    cache = _V25FrameCache()
    rows_out: list[dict[str, object]] = []
    counts = {split_name: [0, 0, 0, 0] for split_name in ("train", "dev")}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shards = staging / "shards"
        shards.mkdir()
        session_features: dict[str, np.ndarray] = {}
        for split_name, rows in (("train", train_rows), ("dev", dev_rows)):
            for ordinal, row in enumerate(rows):
                session = cast(str, row["session"])
                if session not in session_features:
                    session_features[session] = _encode_views(
                        encoder, cache.views(root, row), target, batch_size
                    )
                decision_path = root / session / cast(str, row["path"])
                with np.load(decision_path, allow_pickle=False) as decision:
                    observation = decision["observation_index"].astype(np.int64)
                    shifted = decision["shifted_observation_index"].astype(np.int64)
                    labels = decision["combat_id"].astype(np.int8)
                features = session_features[session]
                windows = np.stack([features[index - 31 : index + 1] for index in observation])
                shifted_windows = np.stack([features[index - 31 : index + 1] for index in shifted])
                name = f"{split_name}-{ordinal:04d}.npz"
                path = shards / name
                np.savez_compressed(
                    path,
                    features=windows,
                    shifted_features=shifted_windows,
                    combat_id=labels,
                )
                class_counts = np.bincount(labels, minlength=4).tolist()
                counts[split_name] = [
                    current + incoming
                    for current, incoming in zip(counts[split_name], class_counts, strict=True)
                ]
                rows_out.append(
                    {
                        "path": name,
                        "sha256": _sha(path),
                        "split": split_name,
                        "session": session,
                        "rows": len(labels),
                        "class_counts": class_counts,
                    }
                )
        manifest: dict[str, object] = {
            "schema_version": FEATURE_CACHE_SCHEMA,
            "status": "COMPLETED",
            "adapter_sha256": _sha(adapter),
            "adapter_source_sha256": metadata.get("v5_source_model_sha256"),
            "split_sha256": split["split_sha256"],
            "window_frames": WINDOW_FRAMES,
            "feature_shape": [WINDOW_FRAMES, FEATURE_SIZE],
            "feature_dtype": "float16",
            "counts": counts,
            "shards": rows_out,
            "raw_rgb_persisted": False,
            "source_paths_persisted": False,
            "test_opened": False,
            "control_output": False,
            "device_input_allowed": False,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return manifest


def _cache_split(
    root: Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if split not in {"train", "dev"}:
        raise CombatFeatureCacheError("feature cache may open train or dev only")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != FEATURE_CACHE_SCHEMA:
        raise CombatFeatureCacheError("feature cache manifest is invalid")
    supplied = manifest.get("manifest_sha256")
    unsigned = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    if (
        supplied != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or manifest.get("test_opened") is not False
    ):
        raise CombatFeatureCacheError("feature cache manifest identity differs")
    rows = [
        cast(dict[str, object], row)
        for row in cast(list[object], manifest["shards"])
        if cast(dict[str, object], row).get("split") == split
    ]
    features: list[np.ndarray] = []
    shifted: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for row in rows:
        name = cast(str, row["path"])
        path = root / "shards" / name
        if _sha(path) != row["sha256"]:
            raise CombatFeatureCacheError("feature cache shard hash differs")
        with np.load(path, allow_pickle=False) as shard:
            features.append(shard["features"])
            shifted.append(shard["shifted_features"])
            labels.append(shard["combat_id"])
    return np.concatenate(features), np.concatenate(shifted), np.concatenate(labels), manifest


class _CachedCombatModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Conv1d(FEATURE_SIZE, 256, 1)
        self.temporal = nn.Sequential(*(_V2ResidualBlock(value) for value in (1, 2, 4, 8)))
        self.gate = nn.Linear(256, 2)
        self.action = nn.Linear(256, 3)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or tuple(values.shape[1:]) != (WINDOW_FRAMES, FEATURE_SIZE):
            raise CombatFeatureCacheError("cached combat model input is invalid")
        hidden = self.temporal(self.mix(values.transpose(1, 2)))[..., -1]
        return self.gate(hidden), self.action(hidden)


def _predict(
    model: _CachedCombatModel, values: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            gate, action = model(
                torch.from_numpy(values[start : start + batch_size]).to(device).float()
            )
            active = gate.softmax(1)[:, 1] >= 0.65
            output.append(
                torch.where(active, action.argmax(1) + 1, torch.zeros_like(active.long()))
                .cpu()
                .numpy()
            )
    return np.concatenate(output)


def _fit(
    features: np.ndarray,
    labels: np.ndarray,
    dev_features: np.ndarray,
    dev_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    shuffled: bool,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    rng = np.random.default_rng(0)
    target_labels = labels.copy()
    if shuffled:
        rng.shuffle(target_labels)
    model = _CachedCombatModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    gate_counts = np.bincount((target_labels > 0).astype(np.int64), minlength=2)
    action_counts = np.bincount(target_labels[target_labels > 0] - 1, minlength=3)
    gate_weight = torch.tensor(
        gate_counts.sum() / (2 * np.maximum(gate_counts, 1)), device=device, dtype=torch.float32
    )
    action_weight = torch.tensor(
        action_counts.sum() / (3 * np.maximum(action_counts, 1)), device=device, dtype=torch.float32
    )
    gate_loss = nn.CrossEntropyLoss(weight=gate_weight)
    action_loss = nn.CrossEntropyLoss(weight=action_weight)
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = 0
    best_metrics: dict[str, object] = {}
    for epoch in range(1, 9):
        order = rng.permutation(len(features))
        model.train()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            gate, action = model(torch.from_numpy(features[rows]).to(device).float())
            targets = torch.from_numpy(target_labels[rows]).to(device).long()
            loss = gate_loss(gate, (targets > 0).long())
            active = targets > 0
            if bool(active.any()):
                loss = loss + action_loss(action[active], targets[active] - 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = _head_metrics(_predict(model, dev_features, device, batch_size), dev_labels, 4)
        score = cast(float, metrics["macro_f1"])
        if score > best_f1:
            best_f1, best_epoch, best_metrics = score, epoch, metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    return {"best_epoch": best_epoch, "metrics": best_metrics, "shuffled": shuffled}, best_state


def train_global_combat_feature_head(
    *, feature_root: Path, output_dir: Path, device: str, batch_size: int = 128
) -> dict[str, object]:
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise CombatFeatureCacheError("cached combat training settings are invalid")
    target = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise CombatFeatureCacheError("CUDA is unavailable")
    root = _large_existing(feature_root)
    train_x, _train_shifted, train_y, manifest = _cache_split(root, "train")
    dev_x, dev_shifted, dev_y, dev_manifest = _cache_split(root, "dev")
    if manifest["manifest_sha256"] != dev_manifest["manifest_sha256"]:
        raise CombatFeatureCacheError("feature cache changed between splits")
    normal, state = _fit(train_x, train_y, dev_x, dev_y, target, batch_size, False)
    shuffled, _ = _fit(train_x, train_y, dev_x, dev_y, target, batch_size, True)
    model = _CachedCombatModel().to(target)
    model.load_state_dict(state, strict=True)
    static = np.repeat(dev_x[:, -1:, :], WINDOW_FRAMES, axis=1)
    metrics = cast(dict[str, object], normal["metrics"])
    report: dict[str, object] = {
        "schema_version": FEATURE_MODEL_SCHEMA,
        "status": "COMPLETED",
        "feature_manifest_sha256": manifest["manifest_sha256"],
        "normal": normal,
        "shuffled": shuffled,
        "static": _head_metrics(_predict(model, static, target, batch_size), dev_y, 4),
        "time_shift": _head_metrics(_predict(model, dev_shifted, target, batch_size), dev_y, 4),
        "normal_minus_shuffled_macro_f1": cast(float, metrics["macro_f1"])
        - cast(float, cast(dict[str, object], shuffled["metrics"])["macro_f1"]),
        "test_opened": False,
        "control_output": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        staging = Path(temp)
        model_path = staging / "model-seed-0.safetensors"
        save_file(
            state,
            model_path,
            metadata={
                "schema": FEATURE_MODEL_SCHEMA,
                "feature_manifest_sha256": cast(str, manifest["manifest_sha256"]),
            },
        )
        report["model_sha256"] = _sha(model_path)
        report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report
