"""T8-v5 ROI-isolated zero-label diagnostics over the frozen T8-v4 repair lineage."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.mobile_testbed import ABILITIES, load_layout
from hok_agent.t8 import (
    _canonical,
    _head_metrics,
    _large_existing,
    _large_new,
    _load_v2_adapter,
    _read_object,
    _retrospective_content_box,
    _retrospective_load_session,
    _retrospective_target_index,
    _sha,
)
from hok_agent.t8_v4 import (
    PSEUDOLABEL_SCHEMA,
    STATE_NAMES,
    _normalize_teacher_frame,
    _self_hash,
    _verified_pseudolabel_manifest,
)

CONTRACT_SCHEMA: Final = "hok-agent-t8-v5-roi-experiment-contract-v1"
DATASET_SCHEMA: Final = "hok-agent-t8-v5-roi-feature-dataset-v1"
DIAGNOSIS_SCHEMA: Final = "hok-agent-t8-v5-roi-seed0-diagnosis-v1"
DECISION_SCHEMA: Final = "hok-agent-t8-v5-roi-decision-v1"
FORMAL_HEADS: Final = STATE_NAMES[:3]
DIAGNOSTIC_HEADS: Final = STATE_NAMES[3:]
FEATURE_SIZE: Final = 512


class T8V5Error(ValueError):
    pass


def verify_t8_v5_contract(path: Path) -> dict[str, object]:
    value = _read_object(path, "T8-v5 ROI contract is unreadable")
    if value.get("schema_version") != CONTRACT_SCHEMA or value.get(
        "experiment_sha256"
    ) != _self_hash(value, "experiment_sha256"):
        raise T8V5Error("T8-v5 ROI contract identity is invalid")
    if (
        value.get("source_lineage") != PSEUDOLABEL_SCHEMA
        or value.get("required_rule_repairs_used") != 1
        or value.get("train_sessions") != 103
        or value.get("dev_sessions") != 23
        or value.get("state_names") != list(STATE_NAMES)
        or value.get("formal_gate_heads") != list(FORMAL_HEADS)
        or value.get("diagnostic_only_heads") != list(DIAGNOSTIC_HEADS)
        or value.get("feature_size") != FEATURE_SIZE
        or value.get("seed") != 0
        or value.get("epochs") != 8
        or value.get("models")
        != [
            "class_prior",
            "time_only",
            "correct_roi_linear",
            "wrong_roi_linear",
            "label_shuffle",
        ]
        or value.get("enemy_correct_roi") != [0.0, 0.0, 0.71875, 1.0]
        or value.get("enemy_wrong_roi") != [0.5234375, 0.296875, 1.0, 1.0]
        or value.get("button_roi_half_size_fraction") != 0.08
        or value.get("minimum_gain_over_time_only") != 0.1
        or value.get("minimum_gain_over_wrong_roi") != 0.15
        or value.get("minimum_gain_over_shuffle") != 0.15
        or value.get("video_test_access_allowed") is not False
        or value.get("human_labels_used") is not False
        or value.get("semantic_accuracy_verified") is not False
        or value.get("promotion_allowed") is not False
        or value.get("control_output") is not False
        or value.get("device_input_allowed") is not False
    ):
        raise T8V5Error("T8-v5 ROI frozen contract values differ")
    return {
        "schema_version": "hok-agent-t8-v5-roi-contract-check-v1",
        "status": "PASSED",
        "experiment_sha256": value["experiment_sha256"],
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "video_test_accessed": False,
        "promotion_allowed": False,
        "control_output": False,
        "device_input_allowed": False,
    }


def _fractional_crop(frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
    if frame.shape != (128, 128, 3) or frame.dtype != np.uint8 or len(box) != 4:
        raise T8V5Error("T8-v5 ROI crop input is invalid")
    x0, y0, x1, y1 = (float(value) for value in box)
    left = max(0, min(127, int(round(x0 * 128))))
    top = max(0, min(127, int(round(y0 * 128))))
    right = max(left + 1, min(128, int(round(x1 * 128))))
    bottom = max(top + 1, min(128, int(round(y1 * 128))))
    rows = np.linspace(top, bottom - 1, 128).astype(np.int64)
    columns = np.linspace(left, right - 1, 128).astype(np.int64)
    return frame[rows[:, None], columns[None, :], :]


def _button_box(point: tuple[float, float], half_size: float) -> tuple[float, ...]:
    return (
        max(0.0, point[0] - half_size),
        max(0.0, point[1] - half_size),
        min(1.0, point[0] + half_size),
        min(1.0, point[1] + half_size),
    )


def roi_views(
    frames: np.ndarray,
    points: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if frames.ndim != 4 or frames.shape[1:] != (128, 128, 3) or len(points) != 3:
        raise T8V5Error("T8-v5 ROI frames or layout points are invalid")
    correct_rows: list[np.ndarray] = []
    wrong_rows: list[np.ndarray] = []
    for frame in frames:
        correct = [_fractional_crop(frame, (0.0, 0.0, 0.71875, 1.0))]
        wrong = [_fractional_crop(frame, (0.5234375, 0.296875, 1.0, 1.0))]
        for index, point in enumerate(points):
            correct.append(_fractional_crop(frame, _button_box(point, 0.08)))
            wrong.append(_fractional_crop(frame, _button_box(points[(index + 1) % 3], 0.08)))
        correct_rows.append(np.stack(correct))
        wrong_rows.append(np.stack(wrong))
    return np.stack(correct_rows), np.stack(wrong_rows)


def _encode_rois(
    encoder: nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    flat = values.reshape(-1, 128, 128, 3)
    output: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            tensor = (
                torch.from_numpy(flat[start : start + batch_size])
                .to(device)
                .permute(0, 3, 1, 2)
                .float()
                .div(255.0)
            )
            output.append(cast(torch.Tensor, encoder(tensor)).cpu().numpy())
    return np.concatenate(output).reshape(len(values), len(STATE_NAMES), FEATURE_SIZE)


def materialize_t8_v5_roi_features(
    *,
    pseudolabel_root: Path,
    target_root: Path,
    adapter_checkpoint: Path,
    layout_path: Path,
    experiment_contract: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, object]:
    contract = verify_t8_v5_contract(experiment_contract)
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise T8V5Error("T8-v5 ROI materialization device is unavailable")
    if batch_size < 1:
        raise T8V5Error("T8-v5 ROI materialization batch size is invalid")
    pseudo_root = _large_existing(pseudolabel_root)
    pseudo = _verified_pseudolabel_manifest(pseudo_root)
    if (
        pseudo.get("rule_repairs_used") != 1
        or pseudo.get("teacher_input_normalization") != "detected_content_box_to_128_nearest_v1"
    ):
        raise T8V5Error("T8-v5 requires the frozen T8-v4 repair-1 lineage")
    adapter = _large_existing(adapter_checkpoint)
    if _sha(adapter) != pseudo.get("adapter_sha256"):
        raise T8V5Error("T8-v5 ROI adapter differs from the weak-target lineage")
    target_device = torch.device(device)
    encoder_state, _adapter_meta = _load_v2_adapter(adapter, target_device)
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(target_device).eval()
    layout, layout_sha = load_layout(layout_path)
    raw_points = [layout.buttons[name] for name in ABILITIES[1:4]]
    if any(point is None for point in raw_points):
        raise T8V5Error("T8-v5 ROI layout lacks combat buttons")
    points = cast(list[tuple[float, float]], raw_points)
    source_rows = pseudo.get("shards")
    if not isinstance(source_rows, list):
        raise T8V5Error("T8-v5 pseudolabel shard index is invalid")
    output = _large_new(output_dir)
    manifest_rows: list[dict[str, object]] = []
    split_counts: dict[str, dict[str, object]] = {}
    target_manifest_sha: str | None = None
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        shards = staging / "shards"
        shards.mkdir()
        for split, expected in (("train", 103), ("dev", 23)):
            target_base, target_sha, sessions = _retrospective_target_index(target_root, split)
            if len(sessions) != expected:
                raise T8V5Error("T8-v5 frozen session count differs")
            if target_manifest_sha is None:
                target_manifest_sha = target_sha
            elif target_sha != target_manifest_sha:
                raise T8V5Error("T8-v5 target manifest changed between splits")
            if target_sha != pseudo.get("target_manifest_sha256"):
                raise T8V5Error("T8-v5 target and pseudolabel lineages differ")
            indexed = dict(sessions)
            selected = [
                cast(dict[str, object], row)
                for row in source_rows
                if isinstance(row, dict) and row.get("split") == split
            ]
            if len(selected) != expected:
                raise T8V5Error("T8-v5 pseudolabel session count differs")
            rows_total = 0
            class_counts = {name: {"positive": 0, "negative": 0} for name in STATE_NAMES}
            for ordinal, row in enumerate(selected):
                identity = row.get("session_hash")
                source_name = row.get("path")
                if (
                    not isinstance(identity, str)
                    or identity not in indexed
                    or not isinstance(source_name, str)
                    or Path(source_name).name != source_name
                ):
                    raise T8V5Error("T8-v5 pseudolabel session identity is invalid")
                source_path = pseudo_root / "shards" / source_name
                if _sha(source_path) != row.get("sha256"):
                    raise T8V5Error("T8-v5 pseudolabel shard hash differs")
                with np.load(source_path, allow_pickle=False) as values:
                    labels = values["weak_labels"].copy()
                    mask = values["training_mask"].copy()
                    observation_end = values["observation_end_timestamp_ms"].copy()
                source_frames, timestamps, _hashes = _retrospective_load_session(
                    target_base, split, identity, indexed[identity]
                )
                canonical, _orientation, content_box = _retrospective_content_box(source_frames)
                indices = np.searchsorted(timestamps, observation_end)
                if np.any(indices >= len(timestamps)) or not np.array_equal(
                    timestamps[indices], observation_end
                ):
                    raise T8V5Error("T8-v5 ROI timestamps do not bind RGB")
                normalized = np.stack(
                    [
                        _normalize_teacher_frame(canonical[int(index)], content_box)
                        for index in indices
                    ]
                )
                correct_rgb, wrong_rgb = roi_views(normalized, points)
                correct = _encode_rois(encoder, correct_rgb, target_device, batch_size)
                wrong = _encode_rois(encoder, wrong_rgb, target_device, batch_size)
                name = f"{split}-{ordinal:04d}.npz"
                path = shards / name
                np.savez_compressed(
                    path,
                    correct_roi_features=correct.astype(np.float16),
                    wrong_roi_features=wrong.astype(np.float16),
                    weak_labels=labels,
                    training_mask=mask,
                    observation_end_timestamp_ms=observation_end,
                )
                for index, state_name in enumerate(STATE_NAMES):
                    class_counts[state_name]["positive"] += int(
                        ((labels[:, index] == 1) & (mask[:, index] == 1)).sum()
                    )
                    class_counts[state_name]["negative"] += int(
                        ((labels[:, index] == 0) & (mask[:, index] == 1)).sum()
                    )
                rows_total += len(labels)
                manifest_rows.append(
                    {
                        "path": name,
                        "sha256": _sha(path),
                        "split": split,
                        "session_hash": identity,
                        "rows": len(labels),
                    }
                )
            split_counts[split] = {
                "sessions": len(selected),
                "rows": rows_total,
                "accepted_class_counts": class_counts,
            }
        manifest: dict[str, object] = {
            "schema_version": DATASET_SCHEMA,
            "status": "COMPLETED",
            "state_names": list(STATE_NAMES),
            "formal_gate_heads": list(FORMAL_HEADS),
            "diagnostic_only_heads": list(DIAGNOSTIC_HEADS),
            "feature_shape": [len(STATE_NAMES), FEATURE_SIZE],
            "roi_encoder": "frozen_t8_v2_video_adapter_resnet18",
            "experiment_sha256": contract["experiment_sha256"],
            "source_pseudolabel_manifest_sha256": pseudo["manifest_sha256"],
            "adapter_sha256": _sha(adapter),
            "target_manifest_sha256": target_manifest_sha,
            "layout_sha256": layout_sha,
            "splits": split_counts,
            "shards": manifest_rows,
            "human_labels_used": False,
            "semantic_accuracy_verified": False,
            "raw_rgb_persisted": False,
            "raw_video_or_source_paths_persisted": False,
            "video_test_accessed": False,
            "promotion_allowed": False,
            "control_output": False,
            "device_input_allowed": False,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return manifest


def _verified_dataset(root: Path, experiment_sha256: str) -> dict[str, object]:
    value = _read_object(root / "manifest.json", "T8-v5 ROI manifest is unreadable")
    if (
        value.get("schema_version") != DATASET_SCHEMA
        or value.get("status") != "COMPLETED"
        or value.get("state_names") != list(STATE_NAMES)
        or value.get("formal_gate_heads") != list(FORMAL_HEADS)
        or value.get("diagnostic_only_heads") != list(DIAGNOSTIC_HEADS)
        or value.get("feature_shape") != [len(STATE_NAMES), FEATURE_SIZE]
        or value.get("experiment_sha256") != experiment_sha256
        or value.get("human_labels_used") is not False
        or value.get("semantic_accuracy_verified") is not False
        or value.get("raw_rgb_persisted") is not False
        or value.get("raw_video_or_source_paths_persisted") is not False
        or value.get("video_test_accessed") is not False
        or value.get("promotion_allowed") is not False
        or value.get("control_output") is not False
        or value.get("device_input_allowed") is not False
        or value.get("manifest_sha256") != _self_hash(value, "manifest_sha256")
    ):
        raise T8V5Error("T8-v5 ROI dataset identity is invalid")
    splits = value.get("splits")
    if (
        not isinstance(splits, dict)
        or not isinstance(splits.get("train"), dict)
        or not isinstance(splits.get("dev"), dict)
        or cast(dict[str, object], splits["train"]).get("sessions") != 103
        or cast(dict[str, object], splits["dev"]).get("sessions") != 23
    ):
        raise T8V5Error("T8-v5 ROI dataset split differs")
    return value


def _load_split(
    root: Path, manifest: Mapping[str, object], split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = manifest.get("shards")
    if split not in {"train", "dev"} or not isinstance(rows, list):
        raise T8V5Error("T8-v5 ROI split is invalid")
    correct_values: list[np.ndarray] = []
    wrong_values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    sessions: list[np.ndarray] = []
    session_index = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != split:
            continue
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise T8V5Error("T8-v5 ROI shard name is invalid")
        path = root / "shards" / name
        if _sha(path) != row.get("sha256"):
            raise T8V5Error("T8-v5 ROI shard hash differs")
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != {
                "correct_roi_features",
                "wrong_roi_features",
                "weak_labels",
                "training_mask",
                "observation_end_timestamp_ms",
            }:
                raise T8V5Error("T8-v5 ROI shard fields differ")
            correct = values["correct_roi_features"].astype(np.float32)
            wrong = values["wrong_roi_features"].astype(np.float32)
            current_labels = values["weak_labels"].astype(np.int8)
            mask = values["training_mask"].astype(np.uint8)
        expected = (len(correct), len(STATE_NAMES), FEATURE_SIZE)
        if correct.shape != expected or wrong.shape != expected:
            raise T8V5Error("T8-v5 ROI feature shape differs")
        correct_values.append(correct)
        wrong_values.append(wrong)
        labels.append(current_labels)
        masks.append(mask)
        sessions.append(np.full(len(correct), session_index, dtype=np.int32))
        session_index += 1
    if not correct_values:
        raise T8V5Error(f"T8-v5 ROI {split} split is empty")
    return (
        np.concatenate(correct_values),
        np.concatenate(wrong_values),
        np.concatenate(labels),
        np.concatenate(masks),
        np.concatenate(sessions),
    )


class _IndependentHeads(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(feature_size, 1) for _ in STATE_NAMES)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != len(STATE_NAMES):
            raise T8V5Error("T8-v5 independent-head input is invalid")
        return torch.stack(
            [head(values[:, index]).squeeze(1) for index, head in enumerate(self.heads)], dim=1
        )


def _time_features(sessions: np.ndarray) -> np.ndarray:
    result = np.zeros((len(sessions), len(STATE_NAMES), 3), dtype=np.float32)
    for session in np.unique(sessions):
        selected = np.flatnonzero(sessions == session)
        progress = np.linspace(0.0, 1.0, len(selected), dtype=np.float32)
        base = np.column_stack(
            (
                progress,
                np.sin(progress * np.float32(2 * np.pi)),
                np.cos(progress * np.float32(2 * np.pi)),
            )
        )
        result[selected] = np.repeat(base[:, None, :], len(STATE_NAMES), axis=1)
    return result


def _metrics(probabilities: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    heads: dict[str, object] = {}
    for index, name in enumerate(STATE_NAMES):
        selected = mask[:, index].astype(bool)
        if not selected.any():
            raise T8V5Error("T8-v5 ROI dev head has no accepted targets")
        heads[name] = _head_metrics(
            (probabilities[selected, index] >= 0.5).astype(np.int64),
            labels[selected, index].astype(np.int64),
            2,
        )
    return {
        "heads": heads,
        "formal_mean_macro_f1": float(
            np.mean(
                [
                    cast(float, cast(dict[str, object], heads[name])["macro_f1"])
                    for name in FORMAL_HEADS
                ]
            )
        ),
        "skill2_diagnostic_macro_f1": cast(
            float, cast(dict[str, object], heads[DIAGNOSTIC_HEADS[0]])["macro_f1"]
        ),
    }


def _predict(
    model: _IndependentHeads, values: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            logits = model(torch.from_numpy(values[start : start + batch_size]).to(device))
            rows.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(rows)


def _fit(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    dev_mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    shuffled: bool,
) -> tuple[dict[str, object], dict[str, torch.Tensor], np.ndarray]:
    torch.manual_seed(0)
    labels = train_y.copy()
    if shuffled:
        rng = np.random.default_rng(0)
        for index in range(len(STATE_NAMES)):
            selected = np.flatnonzero(train_mask[:, index])
            labels[selected, index] = labels[rng.permutation(selected), index]
    model = _IndependentHeads(train_x.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    positive = ((labels == 1) & (train_mask == 1)).sum(axis=0)
    negative = ((labels == 0) & (train_mask == 1)).sum(axis=0)
    pos_weight = torch.from_numpy(negative / np.maximum(positive, 1)).float().to(device)
    best_score = -1.0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(8):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            current_x = torch.from_numpy(train_x[selected]).to(device)
            current_y = torch.from_numpy(labels[selected]).float().to(device)
            current_mask = torch.from_numpy(train_mask[selected]).float().to(device)
            logits = model(current_x)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, current_y.clamp_min(0), pos_weight=pos_weight, reduction="none"
            )
            loss = (losses * current_mask).sum() / current_mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        probabilities = _predict(model, dev_x, device, batch_size)
        metrics = _metrics(probabilities, dev_y, dev_mask)
        score = cast(float, metrics["formal_mean_macro_f1"])
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state, strict=True)
    probabilities = _predict(model, dev_x, device, batch_size)
    return _metrics(probabilities, dev_y, dev_mask), best_state, probabilities


def _prior_metrics(
    train_y: np.ndarray,
    train_mask: np.ndarray,
    dev_rows: int,
    dev_y: np.ndarray,
    dev_mask: np.ndarray,
) -> dict[str, object]:
    prevalence = []
    for index in range(len(STATE_NAMES)):
        selected = train_mask[:, index].astype(bool)
        prevalence.append(float(train_y[selected, index].mean()))
    probabilities = np.repeat(np.asarray(prevalence, dtype=np.float32)[None], dev_rows, axis=0)
    return _metrics(probabilities, dev_y, dev_mask)


def diagnose_t8_v5_roi_seed0(
    *,
    dataset_root: Path,
    experiment_contract: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, object]:
    contract = verify_t8_v5_contract(experiment_contract)
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise T8V5Error("T8-v5 ROI diagnostic device is unavailable")
    if batch_size < 1:
        raise T8V5Error("T8-v5 ROI diagnostic batch size is invalid")
    root = _large_existing(dataset_root)
    manifest = _verified_dataset(root, cast(str, contract["experiment_sha256"]))
    train_correct, train_wrong, train_y, train_mask, train_sessions = _load_split(
        root, manifest, "train"
    )
    dev_correct, dev_wrong, dev_y, dev_mask, dev_sessions = _load_split(root, manifest, "dev")
    target = torch.device(device)
    time_metrics, _time_state, _time_probability = _fit(
        train_x=_time_features(train_sessions),
        train_y=train_y,
        train_mask=train_mask,
        dev_x=_time_features(dev_sessions),
        dev_y=dev_y,
        dev_mask=dev_mask,
        device=target,
        batch_size=batch_size,
        shuffled=False,
    )
    correct_metrics, correct_state, correct_probability = _fit(
        train_x=train_correct,
        train_y=train_y,
        train_mask=train_mask,
        dev_x=dev_correct,
        dev_y=dev_y,
        dev_mask=dev_mask,
        device=target,
        batch_size=batch_size,
        shuffled=False,
    )
    wrong_metrics, _wrong_state, _wrong_probability = _fit(
        train_x=train_wrong,
        train_y=train_y,
        train_mask=train_mask,
        dev_x=dev_wrong,
        dev_y=dev_y,
        dev_mask=dev_mask,
        device=target,
        batch_size=batch_size,
        shuffled=False,
    )
    shuffle_metrics, _shuffle_state, _shuffle_probability = _fit(
        train_x=train_correct,
        train_y=train_y,
        train_mask=train_mask,
        dev_x=dev_correct,
        dev_y=dev_y,
        dev_mask=dev_mask,
        device=target,
        batch_size=batch_size,
        shuffled=True,
    )
    prior = _prior_metrics(train_y, train_mask, len(dev_y), dev_y, dev_mask)
    correct_heads = cast(dict[str, object], correct_metrics["heads"])
    time_heads = cast(dict[str, object], time_metrics["heads"])
    wrong_heads = cast(dict[str, object], wrong_metrics["heads"])
    shuffle_heads = cast(dict[str, object], shuffle_metrics["heads"])
    gates: dict[str, object] = {}
    all_passed = True
    for name in FORMAL_HEADS:
        correct_score = cast(float, cast(dict[str, object], correct_heads[name])["macro_f1"])
        time_score = cast(float, cast(dict[str, object], time_heads[name])["macro_f1"])
        wrong_score = cast(float, cast(dict[str, object], wrong_heads[name])["macro_f1"])
        shuffle_score = cast(float, cast(dict[str, object], shuffle_heads[name])["macro_f1"])
        passed = (
            correct_score - time_score >= 0.1
            and correct_score - wrong_score >= 0.15
            and correct_score - shuffle_score >= 0.15
        )
        all_passed &= passed
        gates[name] = {
            "correct_roi_macro_f1": correct_score,
            "gain_over_time_only": correct_score - time_score,
            "gain_over_wrong_roi": correct_score - wrong_score,
            "gain_over_shuffle": correct_score - shuffle_score,
            "passed": passed,
        }
    decision: dict[str, object] = {
        "schema_version": DECISION_SCHEMA,
        "human_labels_used": False,
        "roi_signal_demonstrated": all_passed,
        "formal_heads_passed": all_passed,
        "skill2_diagnostic_only": True,
        "semantic_accuracy_verified": False,
        "promotion_allowed": False,
        "control_output": False,
        "device_input_allowed": False,
        "next_required_action": (
            "run_roi_tcn_value_test" if all_passed else "freeze_t8_v5_roi_evidence_insufficient"
        ),
    }
    decision["decision_sha256"] = hashlib.sha256(_canonical(decision)).hexdigest()
    report: dict[str, object] = {
        "schema_version": DIAGNOSIS_SCHEMA,
        "status": "COMPLETED",
        "seed": 0,
        "epochs": 8,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "experiment_sha256": contract["experiment_sha256"],
        "models": {
            "class_prior": prior,
            "time_only": time_metrics,
            "correct_roi_linear": correct_metrics,
            "wrong_roi_linear": wrong_metrics,
            "label_shuffle": shuffle_metrics,
        },
        "formal_head_gates": gates,
        "decision": decision,
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "video_test_accessed": False,
        "promotion_allowed": False,
        "control_output": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        model_path = staging / "correct-roi-linear-seed-0.safetensors"
        save_file(correct_state, model_path)
        report["model_sha256"] = _sha(model_path)
        report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "decision.json").write_text(
            json.dumps(decision, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report
