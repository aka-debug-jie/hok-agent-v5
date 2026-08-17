"""T8-v4 zero-human-label weak supervision and visual-causality diagnostics."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torchvision.models import resnet18  # type: ignore[import-untyped]

from hok_agent.mobile_testbed import ABILITIES, load_layout, load_rgb_teacher_calibration
from hok_agent.rich_arena import RichPixelArena
from hok_agent.rich_renderer import RENDERER_HASH
from hok_agent.rich_renderer import render as render_rich
from hok_agent.t8 import (
    CAUSAL_VIDEO_DATASET_SCHEMA,
    _canonical,
    _causal_pixel_views,
    _head_metrics,
    _large_existing,
    _large_new,
    _load_v2_adapter,
    _read_object,
    _retrospective_content_box,
    _retrospective_load_session,
    _retrospective_target_index,
    _sha,
    _V2ResidualBlock,
    _visual_teacher_features,
)
from hok_agent.t8_v3 import _v3_feature_manifest

OBSERVATION_SCHEMA: Final = "hok-agent-t8-v4-observation-contract-v2"
CANDIDATE_SCHEMA: Final = "hok-agent-t8-v4-candidate-action-contract-v1"
WEAK_SCHEMA: Final = "hok-agent-t8-v4-weak-supervision-contract-v1"
EXPERIMENT_SCHEMA: Final = "hok-agent-t8-v4-experiment-contract-v1"
SOURCE_TEACHER_SCHEMA: Final = "hok-agent-t8-v4-pixelarena-source-teacher-v1"
PSEUDOLABEL_SCHEMA: Final = "hok-agent-t8-v4-consensus-pseudolabel-dataset-v1"
WEAK_AUDIT_SCHEMA: Final = "hok-agent-t8-v4-weak-audit-v1"
DIAGNOSIS_SCHEMA: Final = "hok-agent-t8-v4-seed0-diagnosis-v1"
DECISION_SCHEMA: Final = "hok-agent-t8-v4-decision-v1"
STATE_NAMES: Final = (
    "main_view_enemy_cue_visible",
    "basic_attack_button_visual_enabled",
    "skill1_button_visual_ready",
    "skill2_button_visual_ready",
)
WINDOW_FRAMES: Final = 16
FEATURE_SIZE: Final = 512
VIEW_NAMES: Final = (
    "original",
    "brightness_plus_8pct",
    "brightness_minus_8pct",
    "contrast_0_9",
    "contrast_1_1",
)
CONTRACT_FILES: Final = (
    "observation_contract_v2.json",
    "candidate_action_contract_v1.json",
    "t8_v4_weak_supervision_v1.json",
    "t8_v4_experiment_plan_v1.json",
)


class T8V4Error(ValueError):
    pass


@dataclass(frozen=True)
class V4Contracts:
    observation: dict[str, object]
    candidate: dict[str, object]
    weak: dict[str, object]
    experiment: dict[str, object]
    contract_set_sha256: str


def _self_hash(value: Mapping[str, object], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _contract(path: Path, schema: str, hash_field: str) -> dict[str, object]:
    value = _read_object(path, f"T8-v4 contract is unreadable: {path.name}")
    if value.get("schema_version") != schema or value.get(hash_field) != _self_hash(
        value, hash_field
    ):
        raise T8V4Error(f"T8-v4 contract identity is invalid: {path.name}")
    return value


def verify_t8_v4_contracts(
    *,
    observation_contract: Path,
    candidate_contract: Path,
    weak_supervision_contract: Path,
    experiment_contract: Path,
) -> dict[str, object]:
    observation = _contract(observation_contract, OBSERVATION_SCHEMA, "observation_sha256")
    candidate = _contract(candidate_contract, CANDIDATE_SCHEMA, "candidate_action_sha256")
    weak = _contract(weak_supervision_contract, WEAK_SCHEMA, "weak_supervision_sha256")
    experiment = _contract(experiment_contract, EXPERIMENT_SCHEMA, "experiment_sha256")
    if (
        observation.get("state_names") != list(STATE_NAMES)
        or observation.get("human_labels_used") is not False
        or observation.get("uncertain_enters_loss") is not False
        or observation.get("final_frame_labels_window_frames") != WINDOW_FRAMES
        or candidate.get("learned_actions") != []
        or candidate.get("candidate_actions")
        != ["candidate_basic_attack", "candidate_skill1", "candidate_skill2", "none"]
        or candidate.get("control_output") is not False
        or candidate.get("promotion_allowed") is not False
        or candidate.get("device_input_allowed") is not False
        or candidate.get("offline_log_only") is not True
        or weak.get("teachers") != ["frozen_rule_teacher_v1", "pixelarena_transfer_teacher_v1"]
        or weak.get("fusion_rule") != "intersection_only"
        or weak.get("teacher_confidence_threshold") != 0.8
        or weak.get("minimum_head_coverage") != 0.15
        or weak.get("minimum_accepted_stability") != 0.9
        or weak.get("maximum_negative_to_positive_ratio") != 3
        or weak.get("views") != list(VIEW_NAMES)
        or weak.get("maximum_rule_repairs") != 1
        or weak.get("human_labels_used") is not False
        or weak.get("semantic_accuracy_verified") is not False
        or experiment.get("train_sessions") != 103
        or experiment.get("dev_sessions") != 23
        or experiment.get("window_frames") != WINDOW_FRAMES
        or experiment.get("feature_shape") != [WINDOW_FRAMES, FEATURE_SIZE]
        or experiment.get("seed") != 0
        or experiment.get("epochs") != 8
        or experiment.get("model_ladder")
        != [
            "class_prior",
            "time_only",
            "last_frame_linear",
            "pool_mlp",
            "causal_tcn",
            "label_shuffle",
        ]
        or experiment.get("interventions")
        != [
            "gameplay_mask",
            "hud_mask",
            "gameplay_swap",
            "hud_swap",
            "sham_swap",
            "static",
            "reverse",
            "time_shift_2000ms",
        ]
        or experiment.get("minimum_source_head_macro_f1") != 0.9
        or experiment.get("minimum_rgb_gain_over_time_only") != 0.1
        or experiment.get("minimum_normal_gain_over_shuffle") != 0.15
        or experiment.get("minimum_relevant_mask_confidence_drop") != 0.15
        or experiment.get("maximum_irrelevant_mask_confidence_drop") != 0.05
        or experiment.get("minimum_tcn_gain") != 0.05
        or experiment.get("video_test_access_allowed") is not False
        or experiment.get("human_labels_used") is not False
        or experiment.get("promotion_allowed") is not False
        or experiment.get("control_output") is not False
        or experiment.get("device_input_allowed") is not False
    ):
        raise T8V4Error("T8-v4 frozen contract values differ")
    bindings = {
        "observation_sha256": observation["observation_sha256"],
        "candidate_action_sha256": candidate["candidate_action_sha256"],
        "weak_supervision_sha256": weak["weak_supervision_sha256"],
        "experiment_sha256": experiment["experiment_sha256"],
    }
    contract_set_sha256 = hashlib.sha256(_canonical(bindings)).hexdigest()
    return {
        "schema_version": "hok-agent-t8-v4-contract-check-v1",
        "status": "PASSED",
        **bindings,
        "contract_set_sha256": contract_set_sha256,
        "human_labels_used": False,
        "video_test_accessed": False,
        "promotion_allowed": False,
        "control_output": False,
        "device_input_allowed": False,
    }


def _load_contracts(
    observation_contract: Path,
    candidate_contract: Path,
    weak_supervision_contract: Path,
    experiment_contract: Path,
) -> V4Contracts:
    checked = verify_t8_v4_contracts(
        observation_contract=observation_contract,
        candidate_contract=candidate_contract,
        weak_supervision_contract=weak_supervision_contract,
        experiment_contract=experiment_contract,
    )
    return V4Contracts(
        _read_object(observation_contract, "observation contract is unreadable"),
        _read_object(candidate_contract, "candidate contract is unreadable"),
        _read_object(weak_supervision_contract, "weak contract is unreadable"),
        _read_object(experiment_contract, "experiment contract is unreadable"),
        cast(str, checked["contract_set_sha256"]),
    )


def deterministic_photometric_views(frames: np.ndarray) -> np.ndarray:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise T8V4Error("T8-v4 views require uint8 NHWC RGB")
    value = frames.astype(np.float32)
    mean = value.mean(axis=(1, 2, 3), keepdims=True)
    variants = (
        value,
        np.clip(value * 1.08, 0, 255),
        np.clip(value * 0.92, 0, 255),
        np.clip((value - mean) * 0.9 + mean, 0, 255),
        np.clip((value - mean) * 1.1 + mean, 0, 255),
    )
    return np.stack([item.round().astype(np.uint8) for item in variants], axis=1)


def consensus_labels(
    rule_probabilities: np.ndarray,
    source_probabilities: np.ndarray,
    threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    expected_tail = (len(VIEW_NAMES), len(STATE_NAMES))
    if (
        rule_probabilities.shape != source_probabilities.shape
        or rule_probabilities.ndim != 3
        or rule_probabilities.shape[1:] != expected_tail
        or not np.isfinite(rule_probabilities).all()
        or not np.isfinite(source_probabilities).all()
        or np.any((rule_probabilities < 0) | (rule_probabilities > 1))
        or np.any((source_probabilities < 0) | (source_probabilities > 1))
        or threshold != 0.8
    ):
        raise T8V4Error("T8-v4 teacher probability contract is invalid")
    rule_binary = rule_probabilities >= 0.5
    source_binary = source_probabilities >= 0.5
    rule_stable = np.all(rule_binary == rule_binary[:, :1], axis=1)
    source_stable = np.all(source_binary == source_binary[:, :1], axis=1)
    agreement = rule_binary[:, 0] == source_binary[:, 0]
    rule_confidence = np.minimum(rule_probabilities[:, 0], 1.0 - rule_probabilities[:, 0])
    source_confidence = np.minimum(source_probabilities[:, 0], 1.0 - source_probabilities[:, 0])
    rule_certainty = 1.0 - rule_confidence
    source_certainty = 1.0 - source_confidence
    accepted = (
        agreement
        & rule_stable
        & source_stable
        & (rule_certainty >= threshold)
        & (source_certainty >= threshold)
    )
    labels = np.full(accepted.shape, -1, dtype=np.int8)
    labels[accepted] = rule_binary[:, 0][accepted].astype(np.int8)
    rows = max(len(labels), 1)
    qc: dict[str, object] = {
        "coverage_by_head": {
            name: float(accepted[:, index].sum() / rows) for index, name in enumerate(STATE_NAMES)
        },
        "disagreement_by_head": {
            name: float((~agreement[:, index]).sum() / rows)
            for index, name in enumerate(STATE_NAMES)
        },
        "rule_flip_rate_by_head": {
            name: float((~rule_stable[:, index]).sum() / rows)
            for index, name in enumerate(STATE_NAMES)
        },
        "source_flip_rate_by_head": {
            name: float((~source_stable[:, index]).sum() / rows)
            for index, name in enumerate(STATE_NAMES)
        },
    }
    return labels, accepted.astype(np.uint8), qc


def spatial_mask(frames: np.ndarray, region: str) -> np.ndarray:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (128, 128, 3):
        raise T8V4Error("T8-v4 intervention requires Nx128x128x3 RGB")
    if region not in {"gameplay", "hud"}:
        raise T8V4Error("T8-v4 intervention region is invalid")
    result = frames.copy()
    if region == "gameplay":
        result[:] = 0
        result[:, 38:128, 66:128] = frames[:, 38:128, 66:128]
    else:
        result[:, 38:128, 66:128] = 0
    return result


def spatial_swap(recipients: np.ndarray, donors: np.ndarray, region: str) -> np.ndarray:
    if recipients.shape != donors.shape:
        raise T8V4Error("T8-v4 swap tensors must align")
    result = recipients.copy()
    if region == "gameplay":
        result = donors.copy()
        result[:, 38:128, 66:128] = recipients[:, 38:128, 66:128]
    elif region == "hud":
        result[:, 38:128, 66:128] = donors[:, 38:128, 66:128]
    else:
        raise T8V4Error("T8-v4 swap region is invalid")
    return result


class PixelArenaStateTeacher(nn.Module):
    def __init__(self, encoder_state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        try:
            encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            raise T8V4Error("T8-v4 adapter encoder is incompatible") from exc
        self.encoder = encoder
        self.head = nn.Linear(FEATURE_SIZE, len(STATE_NAMES))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (3, 128, 128):
            raise T8V4Error("T8-v4 source teacher requires Bx3x128x128 RGB")
        return cast(torch.Tensor, self.head(cast(torch.Tensor, self.encoder(frames))))


def _rgb_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div(255.0)


def _disc(frame: np.ndarray, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
    yy, xx = np.ogrid[:128, :128]
    frame[(xx - x) ** 2 + (yy - y) ** 2 <= radius * radius] = color


def _synthetic_state_data(
    *, points: Sequence[tuple[float, float]], rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if rows < 32 or len(points) != 3:
        raise T8V4Error("T8-v4 synthetic source settings are invalid")
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    labels: list[list[int]] = []
    for index in range(rows):
        arena = RichPixelArena()
        arena.reset(seed * 100_000 + index)
        observation = arena.observe("blue")
        frame = render_rich(observation, render_seed=seed * 1_000_003 + index).copy()
        background = np.asarray(
            [16 + index % 7, 22 + (index * 3) % 9, 25 + (index * 5) % 11],
            dtype=np.uint8,
        )
        frame[20:108, 8:120] = background
        lane_y = 36 + index % 54
        frame[lane_y : lane_y + 2, 8:120] = np.asarray((45, 56, 38), dtype=np.uint8)
        state = [(index >> bit) & 1 for bit in range(4)]
        own_x, own_y = 26 + index % 22, 45 + (index * 7) % 48
        _disc(frame, own_x, own_y, 5, (55, 195, 235))
        if state[0]:
            enemy_x, enemy_y = 75 + (index * 11) % 35, 43 + (index * 13) % 50
            _disc(frame, enemy_x, enemy_y, 5, (225, 70, 65))
            frame[max(20, enemy_y - 9) : enemy_y - 7, enemy_x - 6 : enemy_x + 7] = (
                205,
                48,
                45,
            )
        for button_index, (x_value, y_value) in enumerate(points):
            x = int(round(x_value * 127))
            y = int(round(y_value * 127))
            enabled = bool(state[button_index + 1])
            _disc(frame, x, y, 6, (230, 205, 100) if enabled else (48, 48, 52))
            if not enabled:
                frame[max(0, y - 1) : min(128, y + 2), max(0, x - 5) : min(128, x + 6)] = (
                    22,
                    22,
                    24,
                )
        noise = rng.integers(-4, 5, size=frame.shape, dtype=np.int16)
        frames.append(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
        labels.append(state)
    return np.stack(frames), np.asarray(labels, dtype=np.float32)


def _binary_head_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    predicted = (probabilities >= 0.5).astype(np.int64)
    heads = {
        name: _head_metrics(predicted[:, index], labels[:, index].astype(np.int64), 2)
        for index, name in enumerate(STATE_NAMES)
    }
    return {
        "heads": heads,
        "minimum_head_macro_f1": min(cast(float, value["macro_f1"]) for value in heads.values()),
        "mean_head_macro_f1": float(
            np.mean([cast(float, value["macro_f1"]) for value in heads.values()])
        ),
    }


def _source_predict(
    model: PixelArenaStateTeacher,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            logits = model(_rgb_tensor(frames[start : start + batch_size], device))
            values.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(values)


def train_t8_v4_source_teacher(
    *,
    adapter_checkpoint: Path,
    layout_path: Path,
    observation_contract: Path,
    candidate_contract: Path,
    weak_supervision_contract: Path,
    experiment_contract: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 64,
    epochs: int = 8,
) -> dict[str, object]:
    contracts = _load_contracts(
        observation_contract,
        candidate_contract,
        weak_supervision_contract,
        experiment_contract,
    )
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise T8V4Error("T8-v4 source teacher device is unavailable")
    if batch_size < 1 or epochs != 8:
        raise T8V4Error("T8-v4 source teacher training settings differ")
    target = torch.device(device)
    adapter = _large_existing(adapter_checkpoint)
    encoder_state, adapter_meta = _load_v2_adapter(adapter, target)
    layout, layout_sha = load_layout(layout_path)
    raw_points = [layout.buttons[name] for name in ABILITIES[1:4]]
    if any(point is None for point in raw_points):
        raise T8V4Error("T8-v4 source teacher requires three combat button points")
    points = cast(list[tuple[float, float]], raw_points)
    train_x, train_y = _synthetic_state_data(points=points, rows=1_024, seed=0)
    dev_x, dev_y = _synthetic_state_data(points=points, rows=256, seed=1)
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    model = PixelArenaStateTeacher(encoder_state).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            logits = model(_rgb_tensor(train_x[selected], target))
            labels = torch.from_numpy(train_y[selected]).to(target)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        probabilities = _source_predict(model, dev_x, target, batch_size)
        metrics = _binary_head_metrics(probabilities, dev_y)
        score = cast(float, metrics["minimum_head_macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if not best_state:
        raise T8V4Error("T8-v4 source teacher produced no selected model")
    model.load_state_dict(best_state, strict=True)
    metrics = _binary_head_metrics(_source_predict(model, dev_x, target, batch_size), dev_y)
    passed = cast(float, metrics["minimum_head_macro_f1"]) >= 0.9
    report: dict[str, object] = {
        "schema_version": SOURCE_TEACHER_SCHEMA,
        "status": "PASSED" if passed else "SOURCE_TEACHER_FAILED",
        "strict_passed": passed,
        "human_labels_used": False,
        "teacher_source": "pixelarena_counterfactual_rgb_truth_v1",
        "teacher_renderer_sha256": RENDERER_HASH,
        "rule_teacher_artifacts_read": False,
        "state_names": list(STATE_NAMES),
        "train_rows": len(train_x),
        "dev_rows": len(dev_x),
        "seed": 0,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "minimum_head_macro_f1_required": 0.9,
        "adapter_sha256": _sha(adapter),
        "adapter_source_sha256": adapter_meta.get("v5_source_model_sha256"),
        "layout_sha256": layout_sha,
        "contract_set_sha256": contracts.contract_set_sha256,
        "video_test_accessed": False,
        "promotion_allowed": False,
        "control_output": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        model_path = staging / "source-teacher-seed-0.safetensors"
        metadata = {
            "schema": SOURCE_TEACHER_SCHEMA,
            "contract_set_sha256": contracts.contract_set_sha256,
            "adapter_sha256": _sha(adapter),
            "layout_sha256": layout_sha,
            "state_names": ",".join(STATE_NAMES),
            "human_labels_used": "false",
            "rule_teacher_artifacts_read": "false",
        }
        save_file(best_state, str(model_path), metadata=metadata)
        report["model_sha256"] = _sha(model_path)
        report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def _load_source_teacher(
    model_path: Path, contracts: V4Contracts, layout_sha256: str, device: torch.device
) -> PixelArenaStateTeacher:
    source = _large_existing(model_path)
    try:
        with safe_open(source, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except OSError as exc:
        raise T8V4Error("T8-v4 source teacher is unreadable") from exc
    if (
        metadata is None
        or metadata.get("schema") != SOURCE_TEACHER_SCHEMA
        or metadata.get("contract_set_sha256") != contracts.contract_set_sha256
        or metadata.get("layout_sha256") != layout_sha256
        or metadata.get("state_names") != ",".join(STATE_NAMES)
        or metadata.get("human_labels_used") != "false"
        or metadata.get("rule_teacher_artifacts_read") != "false"
    ):
        raise T8V4Error("T8-v4 source teacher metadata differs")
    state = load_file(source, device=str(device))
    encoder = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    model = PixelArenaStateTeacher(encoder).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise T8V4Error("T8-v4 source teacher tensors differ") from exc
    return model


def _rule_view_probabilities(
    frames: np.ndarray,
    histories: np.ndarray,
    points: Sequence[tuple[float, float]],
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    views = deterministic_photometric_views(frames)
    history_views = deterministic_photometric_views(histories)
    outputs: list[np.ndarray] = []
    for view_index in range(len(VIEW_NAMES)):
        current_pixels = np.stack(
            [_causal_pixel_views(frame, (0, 0, 128, 128)) for frame in views[:, view_index]]
        )
        history_pixels = np.stack(
            [_causal_pixel_views(frame, (0, 0, 128, 128)) for frame in history_views[:, view_index]]
        )
        _activity, scores = _visual_teacher_features(current_pixels, history_pixels, points)
        normalized = (scores - medians) / scales
        ready_probability = 1.0 / (1.0 + np.exp(-4.0 * normalized))
        scene = current_pixels[:, 0, 8:108, 15:108].astype(np.int16)
        red, green, blue = (scene[..., index] for index in range(3))
        red_mask = (red > 140) & (red - green > 45) & (red - blue > 25)
        red_pixels = red_mask.sum(axis=(1, 2))
        red_row_max = red_mask.sum(axis=2).max(axis=1)
        red_strength = np.maximum(red_pixels / 400.0, red_row_max / 11.0)
        enemy_probability = 1.0 / (1.0 + np.exp(-4.0 * (red_strength - 1.0)))
        outputs.append(np.column_stack((enemy_probability, ready_probability)))
    return np.stack(outputs, axis=1).astype(np.float32)


def _source_view_probabilities(
    model: PixelArenaStateTeacher,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    views = deterministic_photometric_views(frames)
    return np.stack(
        [
            _source_predict(model, views[:, index], device, batch_size)
            for index in range(len(VIEW_NAMES))
        ],
        axis=1,
    ).astype(np.float32)


def _normalize_teacher_frame(
    frame: np.ndarray, content_box: tuple[int, int, int, int]
) -> np.ndarray:
    if frame.shape != (128, 128, 3) or frame.dtype != np.uint8:
        raise T8V4Error("T8-v4 teacher frame is invalid")
    x0, y0, x1, y1 = content_box
    if not (0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 128):
        raise T8V4Error("T8-v4 teacher content box is invalid")
    rows = np.linspace(y0, y1 - 1, 128).astype(np.int64)
    columns = np.linspace(x0, x1 - 1, 128).astype(np.int64)
    return frame[rows[:, None], columns[None, :], :]


def _restore_teacher_frame(
    normalized: np.ndarray,
    canonical: np.ndarray,
    content_box: tuple[int, int, int, int],
    orientation: str,
) -> np.ndarray:
    if normalized.shape != (128, 128, 3) or canonical.shape != normalized.shape:
        raise T8V4Error("T8-v4 restored teacher frame is invalid")
    x0, y0, x1, y1 = content_box
    rows = np.linspace(0, 127, y1 - y0).astype(np.int64)
    columns = np.linspace(0, 127, x1 - x0).astype(np.int64)
    restored = canonical.copy()
    restored[y0:y1, x0:x1] = normalized[rows[:, None], columns[None, :], :]
    if orientation == "counter_clockwise_90":
        return np.rot90(restored, -1).copy()
    if orientation != "stored":
        raise T8V4Error("T8-v4 restored teacher orientation is invalid")
    return restored


def _balanced_training_mask(
    labels: np.ndarray, accepted: np.ndarray, timestamps: np.ndarray, session_hash: str
) -> np.ndarray:
    mask = accepted.astype(bool).copy()
    for head_index, state_name in enumerate(STATE_NAMES):
        positive = np.flatnonzero(mask[:, head_index] & (labels[:, head_index] == 1))
        negative = np.flatnonzero(mask[:, head_index] & (labels[:, head_index] == 0))
        limit = 3 * len(positive)
        if len(negative) <= limit:
            continue
        ranked = sorted(
            negative,
            key=lambda index: hashlib.sha256(
                f"{session_hash}:{int(timestamps[index])}:{state_name}".encode()
            ).hexdigest(),
        )
        mask[ranked[limit:], head_index] = False
    return mask.astype(np.uint8)


def materialize_t8_v4_pseudolabels(
    *,
    feature_root: Path,
    target_root: Path,
    rule_teacher_report: Path,
    source_teacher_model: Path,
    source_teacher_report: Path,
    layout_path: Path,
    observation_contract: Path,
    candidate_contract: Path,
    weak_supervision_contract: Path,
    experiment_contract: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, object]:
    contracts = _load_contracts(
        observation_contract,
        candidate_contract,
        weak_supervision_contract,
        experiment_contract,
    )
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise T8V4Error("T8-v4 pseudolabel device is unavailable")
    if batch_size < 1:
        raise T8V4Error("T8-v4 pseudolabel batch size is invalid")
    features_root = _large_existing(feature_root)
    feature_manifest = _v3_feature_manifest(features_root)
    if feature_manifest.get("schema_version") != CAUSAL_VIDEO_DATASET_SCHEMA:
        raise T8V4Error("T8-v4 feature lineage differs")
    layout, layout_sha = load_layout(layout_path)
    calibration = load_rgb_teacher_calibration(rule_teacher_report, layout_sha)
    raw_points = [layout.buttons[name] for name in ABILITIES[1:4]]
    if any(point is None for point in raw_points):
        raise T8V4Error("T8-v4 rule teacher requires three combat button points")
    points = cast(list[tuple[float, float]], raw_points)
    medians = np.asarray(calibration.medians, dtype=np.float32)
    scales = np.asarray(calibration.scales, dtype=np.float32)
    source_report_path = _large_existing(source_teacher_report)
    source_report = _read_object(source_report_path, "T8-v4 source teacher report is unreadable")
    model_path = _large_existing(source_teacher_model)
    source_metrics = source_report.get("metrics")
    minimum_macro_f1 = (
        source_metrics.get("minimum_head_macro_f1") if isinstance(source_metrics, dict) else None
    )
    if (
        source_report.get("schema_version") != SOURCE_TEACHER_SCHEMA
        or source_report.get("strict_passed") is not True
        or source_report.get("rule_teacher_artifacts_read") is not False
        or source_report.get("contract_set_sha256") != contracts.contract_set_sha256
        or source_report.get("layout_sha256") != layout_sha
        or source_report.get("model_sha256") != _sha(model_path)
        or not isinstance(minimum_macro_f1, (int, float))
        or float(minimum_macro_f1) < 0.9
    ):
        raise T8V4Error("T8-v4 source teacher did not pass independently")
    target_device = torch.device(device)
    source_model = _load_source_teacher(model_path, contracts, layout_sha, target_device)
    source_rows = feature_manifest.get("shards")
    if not isinstance(source_rows, list):
        raise T8V4Error("T8-v4 feature shard index is invalid")
    output = _large_new(output_dir)
    manifest_rows: list[dict[str, object]] = []
    split_qc: dict[str, dict[str, object]] = {}
    target_manifest_sha: str | None = None
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        shard_dir = staging / "shards"
        shard_dir.mkdir()
        for split, expected_sessions in (("train", 103), ("dev", 23)):
            current_root, current_sha, sessions = _retrospective_target_index(target_root, split)
            if len(sessions) != expected_sessions:
                raise T8V4Error("T8-v4 frozen session count differs")
            if target_manifest_sha is None:
                target_manifest_sha = current_sha
            elif current_sha != target_manifest_sha:
                raise T8V4Error("T8-v4 target manifest changed between splits")
            if current_sha != feature_manifest.get("target_manifest_sha256"):
                raise T8V4Error("T8-v4 RGB and feature lineages differ")
            indexed = dict(sessions)
            selected = [
                cast(dict[str, object], row)
                for row in cast(list[object], source_rows)
                if isinstance(row, dict) and row.get("split") == split and row.get("lag_ms") == 100
            ]
            if {str(row.get("session_hash")) for row in selected} != set(indexed):
                raise T8V4Error("T8-v4 feature sessions differ from target sessions")
            coverage_total = np.zeros(len(STATE_NAMES), dtype=np.int64)
            accepted_total = np.zeros(len(STATE_NAMES), dtype=np.int64)
            positive_total = np.zeros(len(STATE_NAMES), dtype=np.int64)
            negative_total = np.zeros(len(STATE_NAMES), dtype=np.int64)
            session_reports: list[dict[str, object]] = []
            for ordinal, row in enumerate(selected):
                identity = str(row["session_hash"])
                source_name = row.get("path")
                if not isinstance(source_name, str) or Path(source_name).name != source_name:
                    raise T8V4Error("T8-v4 feature shard name is invalid")
                feature_path = features_root / "shards" / source_name
                if _sha(feature_path) != row.get("sha256"):
                    raise T8V4Error("T8-v4 feature shard hash differs")
                with np.load(feature_path, allow_pickle=False) as values:
                    features = values["features"].copy()
                    observation_end = values["observation_end_timestamp_ms"].copy()
                if (
                    features.dtype != np.float16
                    or features.shape[1:] != (WINDOW_FRAMES, FEATURE_SIZE)
                    or observation_end.dtype != np.int64
                    or observation_end.shape != (len(features),)
                ):
                    raise T8V4Error("T8-v4 feature tensor contract differs")
                frames, timestamps, _hashes = _retrospective_load_session(
                    current_root, split, identity, indexed[identity]
                )
                canonical, _orientation, content_box = _retrospective_content_box(frames)
                indices = np.searchsorted(timestamps, observation_end)
                if np.any(indices >= len(timestamps)) or not np.array_equal(
                    timestamps[indices], observation_end
                ):
                    raise T8V4Error("T8-v4 feature timestamps do not bind RGB")
                history_indices = np.maximum(
                    np.searchsorted(timestamps, observation_end - 2_000, side="right") - 1, 0
                )
                current_frames = np.stack(
                    [
                        _normalize_teacher_frame(canonical[int(index)], content_box)
                        for index in indices
                    ]
                )
                history_frames = np.stack(
                    [
                        _normalize_teacher_frame(canonical[int(index)], content_box)
                        for index in history_indices
                    ]
                )
                rule_probability = _rule_view_probabilities(
                    current_frames, history_frames, points, medians, scales
                )
                source_probability = _source_view_probabilities(
                    source_model, current_frames, target_device, batch_size
                )
                labels, accepted, qc = consensus_labels(rule_probability, source_probability)
                training_mask = _balanced_training_mask(labels, accepted, observation_end, identity)
                accepted_total += accepted.sum(axis=0, dtype=np.int64)
                positive_total += ((labels == 1) & (accepted == 1)).sum(axis=0, dtype=np.int64)
                negative_total += ((labels == 0) & (accepted == 1)).sum(axis=0, dtype=np.int64)
                coverage_total += len(accepted)
                name = f"{split}-{ordinal:04d}.npz"
                path = shard_dir / name
                np.savez_compressed(
                    path,
                    features=features,
                    weak_labels=labels,
                    accepted=accepted,
                    training_mask=training_mask,
                    rule_probability=rule_probability[:, 0],
                    source_probability=source_probability[:, 0],
                    observation_end_timestamp_ms=observation_end,
                )
                session_qc = {
                    "session_hash": identity,
                    "rows": len(features),
                    "accepted_positive_by_head": {
                        name: int(((labels[:, index] == 1) & (accepted[:, index] == 1)).sum())
                        for index, name in enumerate(STATE_NAMES)
                    },
                    "accepted_negative_by_head": {
                        name: int(((labels[:, index] == 0) & (accepted[:, index] == 1)).sum())
                        for index, name in enumerate(STATE_NAMES)
                    },
                    **qc,
                }
                session_reports.append(session_qc)
                manifest_rows.append(
                    {
                        "path": name,
                        "sha256": _sha(path),
                        "split": split,
                        "session_hash": identity,
                        "rows": len(features),
                    }
                )
            split_qc[split] = {
                "sessions": len(selected),
                "rows": int(coverage_total[0]),
                "coverage_by_head": {
                    name: float(accepted_total[index] / max(coverage_total[index], 1))
                    for index, name in enumerate(STATE_NAMES)
                },
                "accepted_positive_by_head": {
                    name: int(positive_total[index]) for index, name in enumerate(STATE_NAMES)
                },
                "accepted_negative_by_head": {
                    name: int(negative_total[index]) for index, name in enumerate(STATE_NAMES)
                },
                "sessions_qc": session_reports,
            }
        manifest: dict[str, object] = {
            "schema_version": PSEUDOLABEL_SCHEMA,
            "status": "COMPLETED",
            "state_names": list(STATE_NAMES),
            "contract_set_sha256": contracts.contract_set_sha256,
            "source_feature_manifest_sha256": feature_manifest["manifest_sha256"],
            "adapter_sha256": feature_manifest["adapter_sha256"],
            "target_manifest_sha256": target_manifest_sha,
            "rule_teacher_report_sha256": calibration.report_sha256,
            "source_teacher_model_sha256": _sha(model_path),
            "source_teacher_report_sha256": _sha(source_report_path),
            "layout_sha256": layout_sha,
            "feature_shape": [WINDOW_FRAMES, FEATURE_SIZE],
            "views": list(VIEW_NAMES),
            "teacher_confidence_threshold": 0.8,
            "teacher_input_normalization": "detected_content_box_to_128_nearest_v1",
            "rule_repairs_used": 1,
            "splits": split_qc,
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


def _verified_pseudolabel_manifest(root: Path) -> dict[str, object]:
    dataset = _large_existing(root)
    manifest = _read_object(dataset / "manifest.json", "T8-v4 manifest is unreadable")
    claimed = manifest.get("manifest_sha256")
    if (
        manifest.get("schema_version") != PSEUDOLABEL_SCHEMA
        or claimed != _self_hash(manifest, "manifest_sha256")
        or manifest.get("human_labels_used") is not False
        or manifest.get("semantic_accuracy_verified") is not False
        or manifest.get("video_test_accessed") is not False
        or manifest.get("promotion_allowed") is not False
        or manifest.get("control_output") is not False
        or manifest.get("device_input_allowed") is not False
        or manifest.get("raw_rgb_persisted") is not False
        or manifest.get("raw_video_or_source_paths_persisted") is not False
    ):
        raise T8V4Error("T8-v4 pseudolabel manifest identity differs")
    forbidden_key = {"source_path", "source_root", "device_path", "video_test_path"}

    def forbidden_provenance(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).lower() in forbidden_key or forbidden_provenance(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(forbidden_provenance(item) for item in value)
        if isinstance(value, str):
            return value.startswith(("/", "file://")) or "video-test" in value.lower()
        return False

    if forbidden_provenance(manifest):
        raise T8V4Error("T8-v4 pseudolabel manifest exposes forbidden provenance")
    return manifest


def audit_t8_v4_weak_supervision(
    *,
    dataset_root: Path,
    observation_contract: Path,
    candidate_contract: Path,
    weak_supervision_contract: Path,
    experiment_contract: Path,
    output_path: Path,
) -> dict[str, object]:
    contracts = _load_contracts(
        observation_contract,
        candidate_contract,
        weak_supervision_contract,
        experiment_contract,
    )
    root = _large_existing(dataset_root)
    manifest = _verified_pseudolabel_manifest(root)
    if (
        manifest.get("rule_repairs_used") != 1
        or manifest.get("teacher_input_normalization") != "detected_content_box_to_128_nearest_v1"
    ):
        raise T8V4Error("T8-v4 audit requires the single frozen coordinate repair")
    if manifest.get("contract_set_sha256") != contracts.contract_set_sha256:
        raise T8V4Error("T8-v4 pseudolabel contract binding differs")
    rows = manifest.get("shards")
    splits = manifest.get("splits")
    if not isinstance(rows, list) or not isinstance(splits, dict):
        raise T8V4Error("T8-v4 pseudolabel index is invalid")
    split_names = {row.get("split") for row in rows if isinstance(row, dict)}
    if split_names != {"train", "dev"}:
        raise T8V4Error("T8-v4 pseudolabel split differs")
    for row in rows:
        if not isinstance(row, dict):
            raise T8V4Error("T8-v4 pseudolabel row is invalid")
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise T8V4Error("T8-v4 pseudolabel shard name is invalid")
        path = root / "shards" / name
        if _sha(path) != row.get("sha256"):
            raise T8V4Error("T8-v4 pseudolabel shard hash differs")
        with np.load(path, allow_pickle=False) as values:
            required = {
                "features",
                "weak_labels",
                "accepted",
                "training_mask",
                "rule_probability",
                "source_probability",
                "observation_end_timestamp_ms",
            }
            if set(values.files) != required:
                raise T8V4Error("T8-v4 pseudolabel shard fields differ")
            labels = values["weak_labels"]
            accepted = values["accepted"]
            if labels.shape != accepted.shape or labels.shape[1:] != (len(STATE_NAMES),):
                raise T8V4Error("T8-v4 pseudolabel tensor shape differs")
            if np.any(labels[accepted == 0] != -1):
                raise T8V4Error("T8-v4 uncertain labels entered the accepted set")
    train = splits.get("train")
    dev = splits.get("dev")
    if not isinstance(train, dict) or not isinstance(dev, dict):
        raise T8V4Error("T8-v4 split QC is invalid")
    if train.get("sessions") != 103 or dev.get("sessions") != 23:
        raise T8V4Error("T8-v4 frozen session count differs")
    coverage_values: list[float] = []
    class_coverage_complete = True
    for split in (train, dev):
        coverage = split.get("coverage_by_head")
        positive = split.get("accepted_positive_by_head")
        negative = split.get("accepted_negative_by_head")
        if (
            not isinstance(coverage, dict)
            or not isinstance(positive, dict)
            or not isinstance(negative, dict)
            or set(coverage) != set(STATE_NAMES)
            or set(positive) != set(STATE_NAMES)
            or set(negative) != set(STATE_NAMES)
        ):
            raise T8V4Error("T8-v4 coverage QC differs")
        class_coverage_complete &= all(
            int(positive[name]) >= 1 and int(negative[name]) >= 1 for name in STATE_NAMES
        )
        coverage_values.extend(float(coverage[name]) for name in STATE_NAMES)
    coverage_passed = min(coverage_values) >= 0.15 and class_coverage_complete
    report: dict[str, object] = {
        "schema_version": WEAK_AUDIT_SCHEMA,
        "status": "PASSED" if coverage_passed else "COVERAGE_FAILED",
        "contract_set_sha256": contracts.contract_set_sha256,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "coverage_by_split": {"train": train["coverage_by_head"], "dev": dev["coverage_by_head"]},
        "minimum_coverage": min(coverage_values),
        "minimum_coverage_required": 0.15,
        "accepted_class_coverage_complete": class_coverage_complete,
        "accepted_stability": 1.0,
        "minimum_accepted_stability_required": 0.9,
        "teacher_consensus_usable": coverage_passed,
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "video_test_accessed": False,
        "promotion_allowed": False,
        "control_output": False,
        "device_input_allowed": False,
    }
    decision: dict[str, object] | None = None
    if not coverage_passed:
        decision = {
            "schema_version": DECISION_SCHEMA,
            "human_labels_used": False,
            "synthetic_teacher_passed": True,
            "teacher_consensus_usable": False,
            "rgb_signal_against_weak_targets_demonstrated": False,
            "spatial_selectivity_demonstrated": False,
            "temporal_order_adds_value": False,
            "semantic_accuracy_verified": False,
            "promotion_allowed": False,
            "control_output": False,
            "next_required_action": "repair_rule_teacher_once",
        }
        decision["decision_sha256"] = hashlib.sha256(_canonical(decision)).hexdigest()
        report["decision_sha256"] = decision["decision_sha256"]
    report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    destination = _large_new(output_path)
    decision_path = (
        _large_new(destination.parent / "decision.json") if decision is not None else None
    )
    destination.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if decision is not None and decision_path is not None:
        decision_path.write_text(
            json.dumps(decision, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    return report


class _V4LastLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(FEATURE_SIZE, len(STATE_NAMES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(features[:, -1]))


class _V4PoolMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_SIZE * 2, 256), nn.ReLU(), nn.Linear(256, len(STATE_NAMES))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.net(torch.cat((features[:, -1], features.mean(dim=1)), dim=1)),
        )


class _V4CausalTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(FEATURE_SIZE, 256, 1)
        self.temporal = nn.Sequential(
            _V2ResidualBlock(1),
            _V2ResidualBlock(2),
            _V2ResidualBlock(4),
        )
        self.head = nn.Linear(256, len(STATE_NAMES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.temporal(self.input(features.transpose(1, 2)))
        return cast(torch.Tensor, self.head(encoded[:, :, -1]))


def _masked_metrics(
    probabilities: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    heads: dict[str, object] = {}
    scores: list[float] = []
    for index, name in enumerate(STATE_NAMES):
        selected = mask[:, index].astype(bool)
        if not selected.any():
            raise T8V4Error("T8-v4 dev head has no accepted weak targets")
        metrics = _head_metrics(
            (probabilities[selected, index] >= 0.5).astype(np.int64),
            labels[selected, index].astype(np.int64),
            2,
        )
        heads[name] = metrics
        scores.append(cast(float, metrics["macro_f1"]))
    return {"heads": heads, "mean_head_macro_f1": float(np.mean(scores))}


def _fit_v4_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    dev_mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    shuffled: bool = False,
) -> tuple[dict[str, object], dict[str, torch.Tensor], np.ndarray]:
    torch.manual_seed(0)
    labels = train_y.copy()
    if shuffled:
        rng = np.random.default_rng(0)
        for index in range(len(STATE_NAMES)):
            selected = np.flatnonzero(train_mask[:, index])
            labels[selected, index] = labels[rng.permutation(selected), index]
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_score = -1.0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(8):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            logits = model(torch.from_numpy(train_x[selected]).float().to(device))
            targets = torch.from_numpy(labels[selected]).float().to(device)
            masks = torch.from_numpy(train_mask[selected]).float().to(device)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, targets.clamp_min(0), reduction="none"
            )
            loss = (losses * masks).sum() / masks.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        probabilities = _predict_v4(model, dev_x, device, batch_size)
        metrics = _masked_metrics(probabilities, dev_y, dev_mask)
        score = cast(float, metrics["mean_head_macro_f1"])
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state, strict=True)
    probabilities = _predict_v4(model, dev_x, device, batch_size)
    return _masked_metrics(probabilities, dev_y, dev_mask), best_state, probabilities


def _predict_v4(
    model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            logits = model(
                torch.from_numpy(features[start : start + batch_size]).float().to(device)
            )
            values.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(values)


def _load_v4_split(
    root: Path, manifest: Mapping[str, object], split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise T8V4Error("T8-v4 shard index is invalid")
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    sessions: list[np.ndarray] = []
    for session_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("split") != split:
            continue
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise T8V4Error("T8-v4 shard name is invalid")
        path = root / "shards" / name
        if _sha(path) != row.get("sha256"):
            raise T8V4Error("T8-v4 shard hash differs")
        with np.load(path, allow_pickle=False) as values:
            current = values["features"].astype(np.float32)
            weak = values["weak_labels"].astype(np.int8)
            mask = values["training_mask"].astype(np.uint8)
        features.append(current)
        labels.append(weak)
        masks.append(mask)
        sessions.append(np.full(len(current), session_index, dtype=np.int32))
    if not features:
        raise T8V4Error(f"T8-v4 {split} split is empty")
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(masks),
        np.concatenate(sessions),
    )


def _prior_probabilities(labels: np.ndarray, mask: np.ndarray, rows: int) -> np.ndarray:
    prevalence = []
    for index in range(len(STATE_NAMES)):
        selected = mask[:, index].astype(bool)
        prevalence.append(float(labels[selected, index].mean()) if selected.any() else 0.5)
    return np.repeat(np.asarray(prevalence, dtype=np.float32)[None], rows, axis=0)


def _time_only_features(sessions: np.ndarray) -> np.ndarray:
    result = np.zeros((len(sessions), WINDOW_FRAMES, FEATURE_SIZE), dtype=np.float32)
    for session in np.unique(sessions):
        selected = np.flatnonzero(sessions == session)
        progress = np.linspace(0.0, 1.0, len(selected), dtype=np.float32)
        result[selected, :, 0] = progress[:, None]
        result[selected, :, 1] = np.sin(progress * np.float32(2 * np.pi))[:, None]
        result[selected, :, 2] = np.cos(progress * np.float32(2 * np.pi))[:, None]
    return result


def _session_bootstrap(
    probabilities: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, float]:
    unique = np.unique(sessions)
    scores: list[float] = []
    for session in unique:
        selected = sessions == session
        head_scores: list[float] = []
        for index in range(len(STATE_NAMES)):
            usable = selected & mask[:, index].astype(bool)
            if not usable.any():
                continue
            metrics = _head_metrics(
                (probabilities[usable, index] >= 0.5).astype(np.int64),
                labels[usable, index].astype(np.int64),
                2,
            )
            head_scores.append(cast(float, metrics["macro_f1"]))
        if head_scores:
            scores.append(float(np.mean(head_scores)))
    if not scores:
        raise T8V4Error("T8-v4 session bootstrap has no accepted weak targets")
    rng = np.random.default_rng(0)
    draws = np.asarray(
        [np.mean(rng.choice(scores, size=len(scores), replace=True)) for _ in range(1_000)]
    )
    return {
        "session_mean": float(np.mean(scores)),
        "bootstrap_low_95": float(np.quantile(draws, 0.025)),
        "bootstrap_high_95": float(np.quantile(draws, 0.975)),
        "sessions": float(len(scores)),
    }


def _encode_v4_frames(
    encoder: nn.Module, frames: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    values: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            encoded = encoder(_rgb_tensor(frames[start : start + batch_size], device))
            values.append(cast(torch.Tensor, encoded).cpu().numpy().astype(np.float32))
    return np.concatenate(values)


def _assigned_confidence_drop(
    original: np.ndarray,
    changed: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    head: int,
) -> float:
    selected = mask[:, head].astype(bool)
    if not selected.any():
        raise T8V4Error("T8-v4 spatial head has no accepted dev targets")
    positive = labels[selected, head] == 1
    original_assigned = np.where(positive, original[selected, head], 1.0 - original[selected, head])
    changed_assigned = np.where(positive, changed[selected, head], 1.0 - changed[selected, head])
    return float(np.mean(original_assigned - changed_assigned))


def _donor_indices(
    labels: np.ndarray, mask: np.ndarray, sessions: np.ndarray, head: int, opposite: bool
) -> np.ndarray:
    result = np.arange(len(labels))
    for session in np.unique(sessions):
        members = np.flatnonzero(sessions == session)
        for recipient in members:
            accepted = members[mask[members, head].astype(bool)]
            if len(accepted) == 0:
                continue
            target = labels[recipient, head]
            candidates = accepted[labels[accepted, head] != target]
            if not opposite:
                candidates = accepted[labels[accepted, head] == target]
                candidates = candidates[candidates != recipient]
            if len(candidates):
                result[recipient] = int(candidates[recipient % len(candidates)])
    return result


def _student_spatial_audit(
    *,
    selected_model: nn.Module,
    selected_probability: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    dev_mask: np.ndarray,
    dev_sessions: np.ndarray,
    dataset_manifest: Mapping[str, object],
    dataset_root: Path,
    target_root: Path,
    adapter_checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    adapter = _large_existing(adapter_checkpoint)
    if _sha(adapter) != dataset_manifest.get("adapter_sha256"):
        raise T8V4Error("T8-v4 spatial adapter differs from feature lineage")
    encoder_state, _adapter_meta = _load_v2_adapter(adapter, device)
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(device).eval()
    target_base, target_sha, target_sessions = _retrospective_target_index(target_root, "dev")
    if target_sha != dataset_manifest.get("target_manifest_sha256"):
        raise T8V4Error("T8-v4 spatial RGB lineage differs")
    indexed = dict(target_sessions)
    rows = dataset_manifest.get("shards")
    if not isinstance(rows, list):
        raise T8V4Error("T8-v4 spatial shard index is invalid")
    normalized_by_session: list[np.ndarray] = []
    canonical_by_session: list[np.ndarray] = []
    orientations: list[str] = []
    content_boxes: list[tuple[int, int, int, int]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != "dev":
            continue
        identity = row.get("session_hash")
        name = row.get("path")
        if not isinstance(identity, str) or identity not in indexed or not isinstance(name, str):
            raise T8V4Error("T8-v4 spatial session identity differs")
        with np.load(dataset_root / "shards" / name, allow_pickle=False) as values:
            observation_end = values["observation_end_timestamp_ms"].copy()
        source_frames, timestamps, _hashes = _retrospective_load_session(
            target_base, "dev", identity, indexed[identity]
        )
        canonical, orientation, content_box = _retrospective_content_box(source_frames)
        indices = np.searchsorted(timestamps, observation_end)
        if np.any(indices >= len(timestamps)) or not np.array_equal(
            timestamps[indices], observation_end
        ):
            raise T8V4Error("T8-v4 spatial timestamps do not bind RGB")
        selected_canonical = canonical[indices]
        canonical_by_session.append(selected_canonical)
        normalized_by_session.append(
            np.stack([_normalize_teacher_frame(frame, content_box) for frame in selected_canonical])
        )
        orientations.extend([orientation] * len(indices))
        content_boxes.extend([content_box] * len(indices))
    current_frames = np.concatenate(normalized_by_session)
    canonical_frames = np.concatenate(canonical_by_session)
    if len(current_frames) != len(dev_x):
        raise T8V4Error("T8-v4 spatial RGB rows differ from feature rows")

    def predict_changed(frames: np.ndarray) -> np.ndarray:
        stored = np.stack(
            [
                _restore_teacher_frame(
                    frame, canonical_frames[index], content_boxes[index], orientations[index]
                )
                for index, frame in enumerate(frames)
            ]
        )
        changed = dev_x.copy()
        changed[:, -1] = _encode_v4_frames(encoder, stored, device, batch_size)
        return _predict_v4(selected_model, changed, device, batch_size)

    gameplay_probability = predict_changed(spatial_mask(current_frames, "gameplay"))
    hud_probability = predict_changed(spatial_mask(current_frames, "hud"))
    gameplay_drops = [
        _assigned_confidence_drop(
            selected_probability, gameplay_probability, dev_y, dev_mask, index
        )
        for index in range(len(STATE_NAMES))
    ]
    hud_drops = [
        _assigned_confidence_drop(selected_probability, hud_probability, dev_y, dev_mask, index)
        for index in range(len(STATE_NAMES))
    ]
    swaps: dict[str, object] = {}
    for index, name in enumerate(STATE_NAMES):
        region = "gameplay" if index == 0 else "hud"
        opposite = _donor_indices(dev_y, dev_mask, dev_sessions, index, True)
        sham = _donor_indices(dev_y, dev_mask, dev_sessions, index, False)
        opposite_probability = predict_changed(
            spatial_swap(current_frames, current_frames[opposite], region)
        )
        sham_probability = predict_changed(
            spatial_swap(current_frames, current_frames[sham], region)
        )
        swaps[name] = {
            "region": region,
            "opposite_label_assigned_confidence_drop": _assigned_confidence_drop(
                selected_probability, opposite_probability, dev_y, dev_mask, index
            ),
            "sham_assigned_confidence_drop": _assigned_confidence_drop(
                selected_probability, sham_probability, dev_y, dev_mask, index
            ),
        }
    relevant = [gameplay_drops[0], *hud_drops[1:]]
    irrelevant = [hud_drops[0], *gameplay_drops[1:]]
    passed = min(relevant) >= 0.15 and max(irrelevant) <= 0.05
    return {
        "schema_version": "hok-agent-t8-v4-student-spatial-audit-v1",
        "available": True,
        "gameplay_mask_confidence_drop_by_head": dict(
            zip(STATE_NAMES, gameplay_drops, strict=True)
        ),
        "hud_mask_confidence_drop_by_head": dict(zip(STATE_NAMES, hud_drops, strict=True)),
        "swaps": swaps,
        "minimum_relevant_confidence_drop": min(relevant),
        "maximum_irrelevant_confidence_drop": max(irrelevant),
        "spatial_selectivity_demonstrated": passed,
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "adapter_sha256": _sha(adapter),
        "target_manifest_sha256": target_sha,
        "raw_rgb_persisted": False,
        "video_test_accessed": False,
        "control_output": False,
    }


def diagnose_t8_v4_seed0(
    *,
    dataset_root: Path,
    target_root: Path,
    adapter_checkpoint: Path,
    weak_audit_report: Path,
    observation_contract: Path,
    candidate_contract: Path,
    weak_supervision_contract: Path,
    experiment_contract: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, object]:
    contracts = _load_contracts(
        observation_contract,
        candidate_contract,
        weak_supervision_contract,
        experiment_contract,
    )
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise T8V4Error("T8-v4 diagnostic device is unavailable")
    if batch_size < 1:
        raise T8V4Error("T8-v4 diagnostic batch size is invalid")
    root = _large_existing(dataset_root)
    manifest = _verified_pseudolabel_manifest(root)
    if (
        manifest.get("rule_repairs_used") != 1
        or manifest.get("teacher_input_normalization") != "detected_content_box_to_128_nearest_v1"
    ):
        raise T8V4Error("T8-v4 diagnostic requires the single frozen coordinate repair")
    audit_path = _large_existing(weak_audit_report)
    audit = _read_object(audit_path, "T8-v4 weak audit is unreadable")
    if (
        audit.get("schema_version") != WEAK_AUDIT_SCHEMA
        or audit.get("status") != "PASSED"
        or audit.get("contract_set_sha256") != contracts.contract_set_sha256
        or audit.get("dataset_manifest_sha256") != manifest.get("manifest_sha256")
        or audit.get("report_sha256") != _self_hash(audit, "report_sha256")
    ):
        raise T8V4Error("T8-v4 weak audit did not pass")
    train_x, train_y, train_mask, train_sessions = _load_v4_split(root, manifest, "train")
    dev_x, dev_y, dev_mask, dev_sessions = _load_v4_split(root, manifest, "dev")
    target = torch.device(device)
    prior_probability = _prior_probabilities(train_y, train_mask, len(dev_y))
    prior_metrics = _masked_metrics(prior_probability, dev_y, dev_mask)
    torch.manual_seed(0)
    time_metrics, _time_state, time_probability = _fit_v4_model(
        _V4LastLinear(),
        _time_only_features(train_sessions),
        train_y,
        train_mask,
        _time_only_features(dev_sessions),
        dev_y,
        dev_mask,
        target,
        batch_size,
    )
    torch.manual_seed(0)
    last_metrics, last_state, last_probability = _fit_v4_model(
        _V4LastLinear(), train_x, train_y, train_mask, dev_x, dev_y, dev_mask, target, batch_size
    )
    torch.manual_seed(0)
    pool_metrics, pool_state, pool_probability = _fit_v4_model(
        _V4PoolMLP(), train_x, train_y, train_mask, dev_x, dev_y, dev_mask, target, batch_size
    )
    torch.manual_seed(0)
    tcn_model = _V4CausalTCN()
    tcn_metrics, tcn_state, tcn_probability = _fit_v4_model(
        tcn_model, train_x, train_y, train_mask, dev_x, dev_y, dev_mask, target, batch_size
    )
    torch.manual_seed(0)
    shuffle_metrics, _shuffle_state, _shuffle_probability = _fit_v4_model(
        _V4CausalTCN(),
        train_x,
        train_y,
        train_mask,
        dev_x,
        dev_y,
        dev_mask,
        target,
        batch_size,
        shuffled=True,
    )
    scores = {
        "last_frame_linear": cast(float, last_metrics["mean_head_macro_f1"]),
        "pool_mlp": cast(float, pool_metrics["mean_head_macro_f1"]),
        "causal_tcn": cast(float, tcn_metrics["mean_head_macro_f1"]),
    }
    simpler_score = max(scores["last_frame_linear"], scores["pool_mlp"])
    temporal_adds_value = scores["causal_tcn"] - simpler_score >= 0.05
    if temporal_adds_value:
        selected_name, selected_state, selected_probability = (
            "causal_tcn",
            tcn_state,
            tcn_probability,
        )
        selected_model: nn.Module = tcn_model
    elif scores["pool_mlp"] >= scores["last_frame_linear"]:
        selected_name, selected_state, selected_probability = (
            "pool_mlp",
            pool_state,
            pool_probability,
        )
        selected_model = _V4PoolMLP()
    else:
        selected_name, selected_state, selected_probability = (
            "last_frame_linear",
            last_state,
            last_probability,
        )
        selected_model = _V4LastLinear()
    selected_model.to(target).load_state_dict(selected_state, strict=True)
    static_probability = _predict_v4(
        selected_model, np.repeat(dev_x[:, -1:], WINDOW_FRAMES, axis=1), target, batch_size
    )
    reverse_probability = _predict_v4(selected_model, dev_x[:, ::-1].copy(), target, batch_size)
    shifted = np.roll(dev_x, 20, axis=0)
    for session in np.unique(dev_sessions):
        selected = np.flatnonzero(dev_sessions == session)
        shifted[selected[: min(20, len(selected))]] = dev_x[selected[0]]
    shift_probability = _predict_v4(selected_model, shifted, target, batch_size)
    spatial = _student_spatial_audit(
        selected_model=selected_model,
        selected_probability=selected_probability,
        dev_x=dev_x,
        dev_y=dev_y,
        dev_mask=dev_mask,
        dev_sessions=dev_sessions,
        dataset_manifest=manifest,
        dataset_root=root,
        target_root=target_root,
        adapter_checkpoint=adapter_checkpoint,
        device=target,
        batch_size=batch_size,
    )
    time_score = cast(float, time_metrics["mean_head_macro_f1"])
    selected_score = scores[selected_name]
    shuffle_score = cast(float, shuffle_metrics["mean_head_macro_f1"])
    rgb_signal = selected_score - time_score >= 0.10 and selected_score - shuffle_score >= 0.15
    spatial_passed = spatial.get("spatial_selectivity_demonstrated") is True
    decision: dict[str, object] = {
        "schema_version": DECISION_SCHEMA,
        "human_labels_used": False,
        "synthetic_teacher_passed": True,
        "teacher_consensus_usable": True,
        "rgb_signal_against_weak_targets_demonstrated": rgb_signal,
        "spatial_selectivity_demonstrated": spatial_passed,
        "temporal_order_adds_value": temporal_adds_value,
        "semantic_accuracy_verified": False,
        "promotion_allowed": False,
        "control_output": False,
        "next_required_action": (
            "record_weak_supervision_evidence_insufficient"
            if not rgb_signal or not spatial_passed
            else "freeze_t8_v4_gate"
        ),
    }
    decision["decision_sha256"] = hashlib.sha256(_canonical(decision)).hexdigest()
    report: dict[str, object] = {
        "schema_version": DIAGNOSIS_SCHEMA,
        "status": "COMPLETED",
        "seed": 0,
        "epochs": 8,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "weak_audit_report_sha256": _sha(audit_path),
        "contract_set_sha256": contracts.contract_set_sha256,
        "models": {
            "class_prior": prior_metrics,
            "time_only": time_metrics,
            "last_frame_linear": last_metrics,
            "pool_mlp": pool_metrics,
            "causal_tcn": tcn_metrics,
            "label_shuffle": shuffle_metrics,
        },
        "selected_model": selected_name,
        "selected_session_bootstrap": _session_bootstrap(
            selected_probability, dev_y, dev_mask, dev_sessions
        ),
        "interventions": {
            "static": _masked_metrics(static_probability, dev_y, dev_mask),
            "reverse": _masked_metrics(reverse_probability, dev_y, dev_mask),
            "time_shift_2s": _masked_metrics(shift_probability, dev_y, dev_mask),
            "student_spatial": spatial,
        },
        "decision": decision,
        "semantic_accuracy_verified": False,
        "video_test_accessed": False,
        "promotion_allowed": False,
        "control_output": False,
        "device_input_allowed": False,
    }
    output = _large_new(output_dir)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as raw:
        staging = Path(raw)
        save_file(selected_state, staging / "selected-model-seed-0.safetensors")
        report["model_sha256"] = _sha(staging / "selected-model-seed-0.safetensors")
        report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
        (staging / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "decision.json").write_text(
            json.dumps(decision, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report
