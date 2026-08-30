from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from hok_agent.global_agent import ENABLED_INTENTS, ENABLED_ZONES
from hok_agent.global_policy import (
    GlobalMacroPolicy,
    GlobalWindowDataset,
    _macro_f1,
    _selected_video_shards,
    _sha,
    _under_large_root,
    _write_json,
    real_video_views,
)
from hok_agent.human_ifo import _auc, _canonical, _sequence_hidden, _sim_episode_groups

ACCEPTANCE_SCHEMA: Final = "hok-agent-human-ifo-broad-acceptance-v1"
GATE_B_ACCEPTANCE_SCHEMA: Final = "hok-agent-human-ifo-gate-b-acceptance-v1"
HORIZONS: Final = {"delta_0_4s": 2, "delta_0_8s": 4, "delta_1_2s": 6}


class HumanInverseError(ValueError):
    pass


class InverseMacroModel(nn.Module):
    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_features, 128), nn.ReLU())
        self.intent = nn.Linear(128, len(ENABLED_INTENTS))
        self.zone = nn.Linear(128, len(ENABLED_ZONES))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(values)
        return self.intent(hidden), self.zone(hidden)


class TransitionStyleDiscriminator(nn.Module):
    def __init__(self, input_features: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_features, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(values).squeeze(1))


def load_broad_acceptance(path: Path) -> tuple[dict[str, object], str]:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    claimed = raw.pop("contract_sha256", None)
    digest = _sha(_canonical(raw).encode())
    if claimed != digest or raw.get("schema_version") != ACCEPTANCE_SCHEMA:
        raise HumanInverseError("invalid broad Human IfO acceptance contract")
    if raw.get("gate_b_simulator_only_allowed") is not True:
        raise HumanInverseError("Gate B simulator-only acceptance is closed")
    if any(
        raw.get(key) is not False
        for key in ("semantic_identity_verified", "gate_c_allowed", "device_input_allowed")
    ):
        raise HumanInverseError("broad Human IfO acceptance exceeded its authority")
    if cast(float, raw["observed_human_auc"]) < cast(float, raw["engineering_human_auc_threshold"]):
        raise HumanInverseError("broad Human IfO engineering threshold is not met")
    return raw, digest


def _load_shared_encoder(
    path: Path, acceptance: dict[str, object], device: torch.device
) -> GlobalMacroPolicy:
    if _sha(path.read_bytes()) != acceptance["shared_checkpoint_sha256"]:
        raise HumanInverseError("shared representation checkpoint differs from acceptance")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata is None or metadata.get("schema_version") != (
        "hok-agent-human-ifo-shared-representation-v1"
    ):
        raise HumanInverseError("invalid shared representation checkpoint")
    model = GlobalMacroPolicy("tcn")
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    model.to(device).eval()
    return model


def _materialize_features(
    dataset_root: Path, split: str, model: GlobalMacroPolicy, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = GlobalWindowDataset(dataset_root, split)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features: list[torch.Tensor] = []
    intents: list[torch.Tensor] = []
    zones: list[torch.Tensor] = []
    with torch.no_grad():
        for main, minimap, hud, intent, zone, _scene, _tick in loader:
            features.append(
                _sequence_hidden(model, main.to(device), minimap.to(device), hud.to(device)).cpu()
            )
            intents.append(intent)
            zones.append(zone)
    values = torch.cat(features)
    return (
        values,
        torch.cat(intents),
        torch.cat(zones),
        torch.from_numpy(_sim_episode_groups(dataset_root, split, len(values))),
    )


def _pairs(
    values: torch.Tensor,
    intent: torch.Tensor,
    zone: torch.Tensor,
    groups: torch.Tensor,
    delta: int,
    *,
    state_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = torch.arange(0, len(values) - delta)
    keep = groups[left] == groups[left + delta]
    left = left[keep]
    right = left + delta
    inputs = values[left] if state_only else torch.cat((values[left], values[right]), dim=1)
    return inputs, intent[left], zone[left]


def _train_head(
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dev: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    shuffle_labels: bool = False,
) -> tuple[InverseMacroModel, dict[str, float]]:
    torch.manual_seed(0)
    train_x, train_intent, train_zone = train
    if shuffle_labels:
        order = torch.randperm(len(train_intent), generator=torch.Generator().manual_seed(0))
        train_intent, train_zone = train_intent[order], train_zone[order]
    model = InverseMacroModel(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(train_x, train_intent, train_zone),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    for _epoch in range(20):
        model.train()
        for values, intent, zone in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values.to(device))
            loss = F.cross_entropy(logits[0], intent.to(device)) + F.cross_entropy(
                logits[1], zone.to(device)
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    model.eval()
    dev_x, dev_intent, dev_zone = dev
    with torch.no_grad():
        intent_logits, zone_logits = model(dev_x.to(device))
    intent_prediction = intent_logits.argmax(dim=1).cpu().tolist()
    zone_prediction = zone_logits.argmax(dim=1).cpu().tolist()
    return model, {
        "intent_macro_f1": _macro_f1(dev_intent.tolist(), intent_prediction, len(ENABLED_INTENTS)),
        "zone_macro_f1": _macro_f1(dev_zone.tolist(), zone_prediction, len(ENABLED_ZONES)),
    }


def train_inverse_macro(
    dataset_root: Path,
    shared_checkpoint: Path,
    acceptance_path: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HumanInverseError("CUDA requested but unavailable")
    acceptance, acceptance_sha256 = load_broad_acceptance(acceptance_path)
    model = _load_shared_encoder(shared_checkpoint, acceptance, device)
    train_values, train_intent, train_zone, train_groups = _materialize_features(
        dataset_root, "train", model, device
    )
    dev_values, dev_intent, dev_zone, dev_groups = _materialize_features(
        dataset_root, "dev", model, device
    )
    results: dict[str, dict[str, float]] = {}
    trained: dict[str, InverseMacroModel] = {}
    state_train = _pairs(train_values, train_intent, train_zone, train_groups, 2, state_only=True)
    state_dev = _pairs(dev_values, dev_intent, dev_zone, dev_groups, 2, state_only=True)
    trained["state_only"], results["state_only"] = _train_head(state_train, state_dev, device)
    for name, delta in HORIZONS.items():
        train = _pairs(
            train_values, train_intent, train_zone, train_groups, delta, state_only=False
        )
        dev = _pairs(dev_values, dev_intent, dev_zone, dev_groups, delta, state_only=False)
        trained[name], results[name] = _train_head(train, dev, device)
    _shuffle, results["label_shuffle"] = _train_head(
        _pairs(train_values, train_intent, train_zone, train_groups, 4, state_only=False),
        _pairs(dev_values, dev_intent, dev_zone, dev_groups, 4, state_only=False),
        device,
        shuffle_labels=True,
    )
    selected = max(
        HORIZONS,
        key=lambda name: results[name]["intent_macro_f1"] + results[name]["zone_macro_f1"],
    )
    selected_mean = (results[selected]["intent_macro_f1"] + results[selected]["zone_macro_f1"]) / 2
    state_mean = (
        results["state_only"]["intent_macro_f1"] + results["state_only"]["zone_macro_f1"]
    ) / 2
    shuffle_mean = (
        results["label_shuffle"]["intent_macro_f1"] + results["label_shuffle"]["zone_macro_f1"]
    ) / 2
    transition_gain = selected_mean - state_mean
    shuffle_margin = selected_mean - shuffle_mean
    passed = (
        results[selected]["intent_macro_f1"] >= 0.80
        and results[selected]["zone_macro_f1"] >= 0.80
        and transition_gain >= 0.05
        and shuffle_margin >= 0.20
    )
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    inverse_path = output / "inverse.safetensors"
    save_file(
        trained[selected].state_dict(),
        inverse_path,
        {
            "schema_version": "hok-agent-human-inverse-macro-v1",
            "horizon": selected,
            "shared_checkpoint_sha256": str(acceptance["shared_checkpoint_sha256"]),
            "device_input_allowed": "false",
        },
    )
    payload: dict[str, object] = {
        "schema_version": "hok-agent-human-inverse-gate-b-v1",
        "status": "PASSED" if passed else "FAILED",
        "gate_b_passed": passed,
        "acceptance_sha256": acceptance_sha256,
        "shared_checkpoint_sha256": acceptance["shared_checkpoint_sha256"],
        "selected_horizon": selected,
        "metrics": results,
        "transition_gain_over_state_only": transition_gain,
        "margin_over_label_shuffle": shuffle_margin,
        "inverse_checkpoint_sha256": _sha(inverse_path.read_bytes()),
        "human_observations_used": False,
        "simulator_truth_only": True,
        "gate_c_allowed": passed,
        "video_test_opened": False,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload


def load_gate_b_acceptance(path: Path) -> tuple[dict[str, object], str]:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    claimed = raw.pop("contract_sha256", None)
    digest = _sha(_canonical(raw).encode())
    if claimed != digest or raw.get("schema_version") != GATE_B_ACCEPTANCE_SCHEMA:
        raise HumanInverseError("invalid Gate B engineering acceptance")
    if raw.get("gate_c_pseudolabel_audit_allowed") is not True:
        raise HumanInverseError("Gate C pseudolabel audit is closed")
    if raw.get("human_bc_allowed") is not False or raw.get("device_input_allowed") is not False:
        raise HumanInverseError("Gate B acceptance exceeded pseudolabel audit authority")
    if cast(float, raw["observed_transition_gain"]) < cast(
        float, raw["engineering_transition_gain_threshold"]
    ):
        raise HumanInverseError("Gate B engineering transition threshold is not met")
    return raw, digest


def _load_inverse(
    path: Path, acceptance: dict[str, object], device: torch.device
) -> InverseMacroModel:
    if _sha(path.read_bytes()) != acceptance["inverse_checkpoint_sha256"]:
        raise HumanInverseError("inverse checkpoint differs from Gate B acceptance")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata is None or metadata.get("schema_version") != "hok-agent-human-inverse-macro-v1":
        raise HumanInverseError("invalid inverse macro checkpoint")
    if metadata.get("horizon") != acceptance["selected_horizon"]:
        raise HumanInverseError("inverse horizon differs from Gate B acceptance")
    model = InverseMacroModel(256)
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    model.to(device).eval()
    return model


def _video_session_features(
    video_root: Path,
    split: str,
    encoder: GlobalMacroPolicy,
    device: torch.device,
) -> list[dict[str, object]]:
    shards, _count = _selected_video_shards(video_root, split)
    first_shard: dict[str, dict[str, object]] = {}
    for shard in shards:
        hashes = cast(list[object], shard["session_hashes"])
        if len(hashes) == 1:
            first_shard.setdefault(str(hashes[0]), shard)
    rows: list[dict[str, object]] = []
    for session_index, (session, shard) in enumerate(sorted(first_shard.items())):
        with np.load(video_root / "shards" / str(shard["path"]), allow_pickle=False) as archive:
            frames = np.asarray(archive["frames"])
            timestamps = np.asarray(archive["timestamp_ms"])
        endings = list(range(30, len(frames), 4))
        main: list[np.ndarray] = []
        minimap: list[np.ndarray] = []
        hud: list[np.ndarray] = []
        for end in endings:
            views = [real_video_views(frame) for frame in frames[end - 30 : end + 1 : 2]]
            main.append(np.stack([view[0] for view in views]))
            minimap.append(np.stack([view[1] for view in views]))
            hud.append(np.stack([view[2] for view in views]))
        tensors = tuple(
            torch.from_numpy(np.stack(values)).permute(0, 1, 4, 2, 3).float().div(255.0)
            for values in (main, minimap, hud)
        )
        features: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(endings), 16):
                batch = tuple(value[start : start + 16].to(device) for value in tensors)
                features.append(_sequence_hidden(encoder, *batch).cpu())
        values = torch.cat(features)
        for index in range(len(values) - 2):
            rows.append(
                {
                    "session_hash": session,
                    "session_index": session_index,
                    "timestamp_ms": int(timestamps[endings[index]]),
                    "current": values[index],
                    "future": values[index + 2],
                }
            )
    return rows


def _predict_pseudolabels(
    rows: list[dict[str, object]],
    model: InverseMacroModel,
    device: torch.device,
    *,
    temperature: float = 1.0,
) -> list[dict[str, object]]:
    inputs = torch.stack(
        [
            torch.cat((cast(torch.Tensor, row["current"]), cast(torch.Tensor, row["future"])))
            for row in rows
        ]
    )
    intent_predictions: list[int] = []
    zone_predictions: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for start in range(0, len(inputs), 256):
            intent, zone = model(inputs[start : start + 256].to(device))
            intent, zone = intent / temperature, zone / temperature
            intent_probability = intent.softmax(dim=1)
            zone_probability = zone.softmax(dim=1)
            intent_predictions.extend(intent_probability.argmax(dim=1).cpu().tolist())
            zone_predictions.extend(zone_probability.argmax(dim=1).cpu().tolist())
            confidences.extend(
                torch.minimum(
                    intent_probability.max(dim=1).values, zone_probability.max(dim=1).values
                )
                .cpu()
                .tolist()
            )
    output: list[dict[str, object]] = []
    for row, intent, zone, confidence in zip(
        rows, intent_predictions, zone_predictions, confidences, strict=True
    ):
        output.append({**row, "intent": intent, "zone": zone, "confidence": confidence})
    return output


def _apply_segment_hold(rows: list[dict[str, object]], minimum_samples: int = 4) -> list[bool]:
    accepted = [False] * len(rows)
    start = 0
    while start < len(rows):
        identity = (
            rows[start]["session_hash"],
            rows[start]["intent"],
            rows[start]["zone"],
        )
        end = start + 1
        while (
            end < len(rows)
            and (rows[end]["session_hash"], rows[end]["intent"], rows[end]["zone"]) == identity
        ):
            end += 1
        if end - start >= minimum_samples:
            for index in range(start, end):
                accepted[index] = cast(float, rows[index]["confidence"]) >= 0.75
        start = end
    return accepted


def _pseudolabel_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted = _apply_segment_hold(rows)
    accepted_rows = [row for row, keep in zip(rows, accepted, strict=True) if keep]
    intent_counts = Counter(cast(int, row["intent"]) for row in accepted_rows)
    zone_counts = Counter(cast(int, row["zone"]) for row in accepted_rows)
    coverage = len(accepted_rows) / max(1, len(rows))
    maximum_intent_fraction = max(intent_counts.values(), default=0) / max(1, len(accepted_rows))
    return {
        "samples": len(rows),
        "accepted_samples": len(accepted_rows),
        "high_confidence_coverage": coverage,
        "intent_counts": dict(sorted(intent_counts.items())),
        "zone_counts": dict(sorted(zone_counts.items())),
        "accepted_intents": len(intent_counts),
        "maximum_intent_fraction": maximum_intent_fraction,
        "passed": coverage >= 0.40 and len(intent_counts) >= 4 and maximum_intent_fraction < 0.70,
    }


def materialize_human_pseudolabels(
    video_root: Path,
    shared_checkpoint: Path,
    broad_acceptance_path: Path,
    inverse_checkpoint: Path,
    gate_b_acceptance_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    temperature_path: Path | None = None,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HumanInverseError("CUDA requested but unavailable")
    broad_acceptance, broad_sha256 = load_broad_acceptance(broad_acceptance_path)
    gate_b_acceptance, gate_b_sha256 = load_gate_b_acceptance(gate_b_acceptance_path)
    encoder = _load_shared_encoder(shared_checkpoint, broad_acceptance, device)
    inverse = _load_inverse(inverse_checkpoint, gate_b_acceptance, device)
    temperature = 1.0
    temperature_sha256: str | None = None
    if temperature_path is not None:
        calibration = cast(
            dict[str, object], json.loads(temperature_path.read_text(encoding="utf-8"))
        )
        claimed = calibration.pop("report_sha256", None)
        if (
            claimed != _sha(_canonical(calibration).encode())
            or calibration.get("schema_version") != "hok-agent-human-inverse-temperature-v1"
        ):
            raise HumanInverseError("invalid inverse temperature calibration")
        if calibration.get("gate_b_acceptance_sha256") != gate_b_sha256:
            raise HumanInverseError("temperature calibration Gate B identity differs")
        temperature = cast(float, calibration["selected_temperature"])
        temperature_sha256 = claimed
    train_rows = _predict_pseudolabels(
        _video_session_features(video_root, "train", encoder, device),
        inverse,
        device,
        temperature=temperature,
    )
    dev_rows = _predict_pseudolabels(
        _video_session_features(video_root, "dev", encoder, device),
        inverse,
        device,
        temperature=temperature,
    )
    train_summary = _pseudolabel_summary(train_rows)
    dev_summary = _pseudolabel_summary(dev_rows)
    train_keep = _apply_segment_hold(train_rows)
    accepted = [row for row, keep in zip(train_rows, train_keep, strict=True) if keep]
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    feature_path = output / "train-pseudolabel-features.npz"
    current = (
        np.stack([cast(torch.Tensor, row["current"]).numpy() for row in accepted])
        if accepted
        else np.empty((0, 128), dtype=np.float32)
    )
    future = (
        np.stack([cast(torch.Tensor, row["future"]).numpy() for row in accepted])
        if accepted
        else np.empty((0, 128), dtype=np.float32)
    )
    with feature_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            current=current,
            future=future,
            intent=np.asarray([row["intent"] for row in accepted], dtype=np.int16),
            zone=np.asarray([row["zone"] for row in accepted], dtype=np.int16),
            confidence=np.asarray([row["confidence"] for row in accepted], dtype=np.float32),
            session_index=np.asarray([row["session_index"] for row in accepted], dtype=np.int16),
            timestamp_ms=np.asarray([row["timestamp_ms"] for row in accepted], dtype=np.int64),
        )
    passed = bool(train_summary["passed"] and dev_summary["passed"])
    payload: dict[str, object] = {
        "schema_version": "hok-agent-human-ifo-gate-c-v1",
        "status": "PASSED" if passed else "FAILED",
        "gate_c_passed": passed,
        "broad_acceptance_sha256": broad_sha256,
        "gate_b_acceptance_sha256": gate_b_sha256,
        "temperature": temperature,
        "temperature_report_sha256": temperature_sha256,
        "train": train_summary,
        "dev_diagnostic": dev_summary,
        "feature_sha256": _sha(feature_path.read_bytes()),
        "human_dev_used_for_training": False,
        "human_labels_used": False,
        "human_bc_allowed": passed,
        "video_test_opened": False,
        "source_paths_persisted": False,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload


def calibrate_inverse_temperature(
    dataset_root: Path,
    shared_checkpoint: Path,
    broad_acceptance_path: Path,
    inverse_checkpoint: Path,
    gate_b_acceptance_path: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    device = torch.device(device_name)
    broad_acceptance, _broad_sha256 = load_broad_acceptance(broad_acceptance_path)
    gate_b_acceptance, gate_b_sha256 = load_gate_b_acceptance(gate_b_acceptance_path)
    encoder = _load_shared_encoder(shared_checkpoint, broad_acceptance, device)
    inverse = _load_inverse(inverse_checkpoint, gate_b_acceptance, device)
    values, intent, zone, groups = _materialize_features(dataset_root, "dev", encoder, device)
    inputs, target_intent, target_zone = _pairs(values, intent, zone, groups, 4, state_only=False)
    with torch.no_grad():
        intent_logits, zone_logits = inverse(inputs.to(device))
    candidates: list[dict[str, float]] = []
    for temperature in np.linspace(0.5, 3.0, 26):
        loss = F.cross_entropy(intent_logits / temperature, target_intent.to(device)) + (
            F.cross_entropy(zone_logits / temperature, target_zone.to(device))
        )
        candidates.append({"temperature": float(temperature), "dev_nll": float(loss)})
    selected = min(candidates, key=lambda row: row["dev_nll"])
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-human-inverse-temperature-v1",
        "status": "COMPLETED",
        "gate_b_acceptance_sha256": gate_b_sha256,
        "selected_temperature": selected["temperature"],
        "selected_dev_nll": selected["dev_nll"],
        "candidates": candidates,
        "calibration_attempt": 1,
        "additional_calibration_allowed": False,
        "human_observations_used": False,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "temperature.json", payload)
    return payload


def _load_style_contract(path: Path) -> tuple[dict[str, object], str]:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    claimed = raw.pop("contract_sha256", None)
    digest = _sha(_canonical(raw).encode())
    if claimed != digest or raw.get("schema_version") != (
        "hok-agent-human-ifo-transition-style-v1"
    ):
        raise HumanInverseError("invalid transition style contract")
    if raw.get("mobile_input_allowed") is not False:
        raise HumanInverseError("transition style contract must remain simulator-only")
    return raw, digest


def _human_transition_tensor(rows: list[dict[str, object]]) -> torch.Tensor:
    return torch.stack(
        [
            torch.cat((cast(torch.Tensor, row["current"]), cast(torch.Tensor, row["future"])))
            for row in rows
        ]
    )


def train_transition_style_discriminator(
    dataset_root: Path,
    video_root: Path,
    shared_checkpoint: Path,
    broad_acceptance_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    contract_path: Path = Path("configs/human_ifo_transition_style_v1.json"),
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HumanInverseError("CUDA requested but unavailable")
    contract, contract_sha256 = _load_style_contract(contract_path)
    broad_acceptance, broad_sha256 = load_broad_acceptance(broad_acceptance_path)
    encoder = _load_shared_encoder(shared_checkpoint, broad_acceptance, device)
    human_train = _human_transition_tensor(
        _video_session_features(video_root, "train", encoder, device)
    )
    human_dev = _human_transition_tensor(
        _video_session_features(video_root, "dev", encoder, device)
    )
    sim_train_values, sim_train_intent, sim_train_zone, sim_train_groups = _materialize_features(
        dataset_root, "train", encoder, device
    )
    sim_dev_values, sim_dev_intent, sim_dev_zone, sim_dev_groups = _materialize_features(
        dataset_root, "dev", encoder, device
    )
    sim_train, _intent, _zone = _pairs(
        sim_train_values,
        sim_train_intent,
        sim_train_zone,
        sim_train_groups,
        4,
        state_only=False,
    )
    sim_dev, _dev_intent, _dev_zone = _pairs(
        sim_dev_values,
        sim_dev_intent,
        sim_dev_zone,
        sim_dev_groups,
        4,
        state_only=False,
    )
    count = min(len(human_train), len(sim_train))
    input_mode = str(contract.get("transition_input", "absolute_pair"))
    normalization: dict[str, torch.Tensor] = {}
    if input_mode == "domain_normalized_delta":
        human_train = human_train[:, 128:] - human_train[:, :128]
        human_dev = human_dev[:, 128:] - human_dev[:, :128]
        sim_train = sim_train[:, 128:] - sim_train[:, :128]
        sim_dev = sim_dev[:, 128:] - sim_dev[:, :128]
        human_mean = human_train.mean(dim=0)
        human_std = human_train.std(dim=0).clamp_min(1e-5)
        sim_mean = sim_train.mean(dim=0)
        sim_std = sim_train.std(dim=0).clamp_min(1e-5)
        normalization = {
            "human_mean": human_mean,
            "human_std": human_std,
            "simulator_mean": sim_mean,
            "simulator_std": sim_std,
        }
        human_train = (human_train - human_mean) / human_std
        reversed_human_dev = (-human_dev - human_mean) / human_std
        human_dev = (human_dev - human_mean) / human_std
        sim_train = (sim_train - sim_mean) / sim_std
        sim_dev = (sim_dev - sim_mean) / sim_std
    else:
        reversed_human_dev = torch.cat((human_dev[:, 128:], human_dev[:, :128]), dim=1)
    values = torch.cat((human_train[:count], sim_train[:count]))
    labels = torch.cat((torch.ones(count), torch.zeros(count)))
    torch.manual_seed(cast(int, contract["seed"]))
    discriminator = TransitionStyleDiscriminator(values.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=cast(float, contract["learning_rate"])
    )
    loader = DataLoader(
        TensorDataset(values, labels),
        batch_size=cast(int, contract["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(cast(int, contract["seed"])),
    )
    for _epoch in range(cast(int, contract["epochs"])):
        discriminator.train()
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                discriminator(batch.to(device)), target.to(device)
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    discriminator.eval()
    dev_count = min(len(human_dev), len(sim_dev))
    with torch.no_grad():
        human_score = discriminator(human_dev[:dev_count].to(device)).sigmoid().cpu().numpy()
        sim_score = discriminator(sim_dev[:dev_count].to(device)).sigmoid().cpu().numpy()
        reversed_score = (
            discriminator(reversed_human_dev[:dev_count].to(device)).sigmoid().mean().item()
        )
    dev_auc = _auc(human_score, sim_score)
    human_score_mean = float(human_score.mean())
    sim_score_mean = float(sim_score.mean())
    order_margin = human_score_mean - reversed_score
    passed = (
        dev_auc >= cast(float, contract["minimum_dev_auc"])
        and dev_auc <= cast(float, contract["maximum_dev_auc"])
        and order_margin >= cast(float, contract["minimum_order_margin"])
    )
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    checkpoint = output / "style-discriminator.safetensors"
    checkpoint_tensors = dict(discriminator.state_dict())
    checkpoint_tensors.update(normalization)
    save_file(
        checkpoint_tensors,
        checkpoint,
        {
            "schema_version": "hok-agent-human-transition-style-v1",
            "contract_sha256": contract_sha256,
            "shared_checkpoint_sha256": str(broad_acceptance["shared_checkpoint_sha256"]),
            "device_input_allowed": "false",
            "transition_input": input_mode,
        },
    )
    payload: dict[str, object] = {
        "schema_version": "hok-agent-human-transition-style-report-v1",
        "status": "PASSED" if passed else "FAILED",
        "style_constraint_usable": passed,
        "contract_sha256": contract_sha256,
        "broad_acceptance_sha256": broad_sha256,
        "dev_auc": dev_auc,
        "transition_input": input_mode,
        "human_score_mean": human_score_mean,
        "simulator_score_mean": sim_score_mean,
        "reversed_human_score_mean": reversed_score,
        "temporal_order_margin": order_margin,
        "human_train_transitions": len(human_train),
        "human_dev_transitions": len(human_dev),
        "simulator_train_transitions": len(sim_train),
        "simulator_dev_transitions": len(sim_dev),
        "reward_contract": {
            "human_transition": contract["human_transition_reward_weight"],
            "terminal": contract["terminal_reward_weight"],
            "structure": contract["structure_reward_weight"],
            "stuck": -cast(float, contract["stuck_penalty_weight"]),
            "dagger_distillation": contract["dagger_distillation_weight"],
        },
        "style_checkpoint_sha256": _sha(checkpoint.read_bytes()),
        "policy_improvement_allowed": passed,
        "human_labels_used": False,
        "video_test_opened": False,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload
