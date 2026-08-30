from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from copy import copy, deepcopy
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from hok_agent.global_agent import GlobalArena, GlobalRuleTeacher, load_global_config
from hok_agent.global_policy import (
    HOLDOUT_SEEDS,
    SCENES,
    WINDOW_FRAMES,
    GlobalMacroPolicy,
    GlobalWindowDataset,
    _challenge_arena,
    _class_weights,
    _evaluate_model,
    _macro_f1,
    _predict_command,
    _rollout_summary,
    _save_model,
    _selected_video_shards,
    _sha,
    _student_rollout,
    _under_large_root,
    _video_manifest,
    _window_sample,
    _write_json,
    load_global_manifest,
    load_global_model,
    real_video_views,
    render_views,
)

COHORT_SCHEMA: Final = "hok-agent-human-ifo-cohort-v1"
CONTRACT_SCHEMA: Final = "hok-agent-human-ifo-v1"
BROAD_CONTRACT_SCHEMA: Final = "hok-agent-human-ifo-broad-v1"
Views = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OBSERVABLE_FACTORS: Final = {
    "health_bucket": 3,
    "at_own_base": 2,
    "enemy_distance_bucket": 3,
    "push_condition": 2,
    "ordinary_lane_advance": 2,
}
REQUIRED_COHORT_FIELDS: Final = (
    "session_hash",
    "split",
    "hero_id",
    "role_id",
    "mode_id",
    "human_controlled",
    "complete_match",
    "hud_stable",
    "overlay_free",
    "rois_usable",
)


class HumanIfoError(ValueError):
    pass


class ObservableFactorProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(128, classes) for name, classes in OBSERVABLE_FACTORS.items()}
        )

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(values) for name, head in self.heads.items()}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _contract_payload(
    path: Path = Path("configs/human_ifo_v1.json"),
) -> tuple[dict[str, object], str]:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    claimed = raw.pop("contract_sha256", None)
    digest = _sha(_canonical(raw).encode())
    if claimed != digest:
        raise HumanIfoError("Human IfO contract hash mismatch")
    if raw.get("schema_version") not in {CONTRACT_SCHEMA, BROAD_CONTRACT_SCHEMA}:
        raise HumanIfoError("invalid Human IfO contract schema")
    for key in (
        "video_test_allowed",
        "human_dev_training_allowed",
        "human_labels_allowed",
        "device_input_allowed",
    ):
        if raw.get(key) is not False:
            raise HumanIfoError(f"Human IfO contract must keep {key}=false")
    if raw.get("window_frames") != 16 or raw.get("sample_hz") != 5:
        raise HumanIfoError("Human IfO temporal contract drift")
    return raw, digest


def cohort_template() -> dict[str, object]:
    return {
        "schema_version": COHORT_SCHEMA,
        "sessions": [
            {
                "session_hash": "replace-with-anonymous-64-hex-session-hash",
                "split": "train",
                "hero_id": "replace-with-one-local-hero-id",
                "role_id": "replace-with-one-local-role-id",
                "mode_id": "replace-with-one-local-mode-id",
                "human_controlled": True,
                "complete_match": True,
                "hud_stable": True,
                "overlay_free": True,
                "rois_usable": True,
            }
        ],
    }


def write_cohort_template(path: Path) -> dict[str, object]:
    if path.exists():
        raise HumanIfoError("refusing to overwrite local Human IfO cohort")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cohort_template()
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")
    return {"status": "COMPLETED", "schema_version": COHORT_SCHEMA, "template_written": True}


def _manifest_sessions(video_root: Path) -> dict[str, str]:
    manifest, _digest = _video_manifest(video_root)
    return {
        str(row["session_hash"]): str(row["split"])
        for row in cast(list[dict[str, object]], manifest.get("sessions"))
    }


def load_human_cohort(path: Path, video_root: Path) -> tuple[list[dict[str, object]], str]:
    if not path.is_file():
        raise HumanIfoError("local Human IfO cohort is required")
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if raw.get("schema_version") != COHORT_SCHEMA:
        raise HumanIfoError("invalid Human IfO cohort schema")
    rows = cast(list[dict[str, object]], raw.get("sessions"))
    if not isinstance(rows, list):
        raise HumanIfoError("Human IfO cohort must contain sessions")
    known = _manifest_sessions(video_root)
    seen: set[str] = set()
    values: set[tuple[str, str, str]] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        if set(row) != set(REQUIRED_COHORT_FIELDS):
            raise HumanIfoError("Human IfO cohort fields differ from contract")
        session = row["session_hash"]
        split = row["split"]
        if not isinstance(session, str) or len(session) != 64 or session in seen:
            raise HumanIfoError("Human IfO cohort session identity is invalid")
        if split not in {"train", "dev"} or known.get(session) != split:
            raise HumanIfoError("Human IfO cohort split is invalid or drifts from manifest")
        if not all(row[key] is True for key in REQUIRED_COHORT_FIELDS[-5:]):
            raise HumanIfoError("Human IfO cohort qualification is incomplete")
        identity = cast(
            tuple[str, str, str],
            tuple(str(row[key]) for key in ("hero_id", "role_id", "mode_id")),
        )
        if any(not value or value.startswith("replace-with-") for value in identity):
            raise HumanIfoError("Human IfO cohort identity is incomplete")
        seen.add(session)
        values.add(identity)
        counts[cast(str, split)] += 1
    if counts != Counter(train=20, dev=5):
        raise HumanIfoError("Human IfO cohort requires exactly 20 train and 5 dev sessions")
    if len(values) != 1:
        raise HumanIfoError("Human IfO cohort must keep one hero, role, and mode")
    normalized = {
        "schema_version": COHORT_SCHEMA,
        "sessions": sorted(rows, key=lambda row: str(row["session_hash"])),
    }
    return rows, _sha(_canonical(normalized).encode())


def gate_a_contract_check(
    dataset_root: Path,
    checkpoint: Path,
    video_root: Path,
    cohort_path: Path,
    *,
    contract_path: Path = Path("configs/human_ifo_v1.json"),
) -> dict[str, object]:
    contract, contract_sha256 = _contract_payload(contract_path)
    manifest = load_global_manifest(dataset_root)
    device = torch.device("cpu")
    _model, checkpoint_metadata = load_global_model(checkpoint, device)
    if checkpoint_metadata.get("manifest_sha256") != str(manifest["manifest_sha256"]):
        raise HumanIfoError("frozen DAgger checkpoint and simulator manifest differ")
    rows, cohort_sha256 = load_human_cohort(cohort_path, video_root)
    _shards, train_count = _selected_video_shards(video_root, "train")
    _dev_shards, dev_count = _selected_video_shards(video_root, "dev")
    return {
        "schema_version": "hok-agent-human-ifo-gate-a-preflight-v1",
        "status": "PASSED",
        "contract_sha256": contract_sha256,
        "cohort_sha256": cohort_sha256,
        "checkpoint_sha256": _sha(checkpoint.read_bytes()),
        "simulator_manifest_sha256": manifest["manifest_sha256"],
        "human_train_sessions": sum(row["split"] == "train" for row in rows),
        "human_dev_sessions": sum(row["split"] == "dev" for row in rows),
        "available_video_train_sessions": train_count,
        "available_video_dev_sessions": dev_count,
        "video_test_opened": False,
        "human_dev_training_allowed": contract["human_dev_training_allowed"],
        "human_labels_used": False,
        "device_input_allowed": False,
        "source_paths_persisted": False,
    }


def _human_windows(
    video_root: Path,
    rows: list[dict[str, object]],
    per_session: int,
    split: str,
    *,
    max_shards_per_session: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    wanted = {str(row["session_hash"]) for row in rows}
    by_session: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    accepted_shards: Counter[str] = Counter()
    shards, _count = _selected_video_shards(video_root, split)
    for shard in shards:
        session_hashes = cast(list[object], shard["session_hashes"])
        if len(session_hashes) != 1 or str(session_hashes[0]) not in wanted:
            continue
        session = str(session_hashes[0])
        if (
            max_shards_per_session is not None
            and accepted_shards[session] >= max_shards_per_session
        ):
            continue
        path = video_root / "shards" / str(shard["path"])
        if _sha(path.read_bytes()) != shard["sha256"]:
            raise HumanIfoError("Human IfO video shard hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            shard_frames = np.asarray(archive["frames"])
            timestamps = np.asarray(archive["timestamp_ms"])
        accepted_shards[session] += 1
        by_session[session].extend(
            (int(timestamp), frame)
            for timestamp, frame in zip(timestamps, shard_frames, strict=True)
        )
    if set(by_session) != wanted:
        raise HumanIfoError("Human IfO selected training session has no video frames")
    main_rows: list[np.ndarray] = []
    minimap_rows: list[np.ndarray] = []
    hud_rows: list[np.ndarray] = []
    group_ids: list[int] = []
    sessions: list[str] = []
    for group, session in enumerate(sorted(wanted)):
        session_frames = [frame for _timestamp, frame in sorted(by_session[session])]
        if len(session_frames) < 16:
            raise HumanIfoError("Human IfO session is shorter than one causal window")
        ends = np.linspace(15, len(session_frames) - 1, per_session).round().astype(np.int64)
        for end in ends:
            views = [
                real_video_views(frame) for frame in session_frames[int(end) - 15 : int(end) + 1]
            ]
            main_rows.append(np.stack([view[0] for view in views]))
            minimap_rows.append(np.stack([view[1] for view in views]))
            hud_rows.append(np.stack([view[2] for view in views]))
            group_ids.append(group)
            sessions.append(session)
    return (
        np.stack(main_rows),
        np.stack(minimap_rows),
        np.stack(hud_rows),
        np.asarray(group_ids),
        sessions,
    )


def _tensor_views(main: np.ndarray, minimap: np.ndarray, hud: np.ndarray) -> Views:
    def convert(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value).permute(0, 1, 4, 2, 3).float().div(255.0)

    return convert(main), convert(minimap), convert(hud)


def _technical_signature(frames: np.ndarray) -> tuple[np.ndarray, bool]:
    if frames.ndim != 4 or frames.shape[0] < 16:
        return np.zeros(12, dtype=np.float64), False
    indices = np.linspace(0, len(frames) - 1, 4).round().astype(np.int64)
    sampled = [real_video_views(frames[index]) for index in indices]
    values: list[float] = []
    for view_index in range(3):
        sequence = np.stack([view[view_index] for view in sampled]).astype(np.float64) / 255.0
        values.extend((float(sequence.mean()), float(sequence.std())))
        values.append(float(np.abs(np.diff(sequence, axis=0)).mean()))
        values.append(float(sequence[-1].mean() - sequence[0].mean()))
    signature = np.asarray(values, dtype=np.float64)
    return signature, bool(signature.std() > 1e-6 and signature[2] > 1e-5)


def _select_unsupervised_cohort(
    rows: list[dict[str, object]], train_count: int = 20, dev_count: int = 5
) -> list[dict[str, object]]:
    valid = [row for row in rows if row["technical_qc_passed"] is True]
    vectors = np.asarray([cast(list[float], row["signature"]) for row in valid])
    if len(vectors) < train_count + dev_count:
        raise HumanIfoError("insufficient technically usable sessions for unsupervised cohort")
    median = np.median(vectors, axis=0)
    scale = np.maximum(np.std(vectors, axis=0), 1e-6)
    for row, vector in zip(valid, vectors, strict=True):
        row["technical_distance"] = float(np.linalg.norm((vector - median) / scale))
    selected: list[dict[str, object]] = []
    for split, count in (("train", train_count), ("dev", dev_count)):
        candidates = sorted(
            (row for row in valid if row["split"] == split),
            key=lambda row: (cast(float, row["technical_distance"]), str(row["session_hash"])),
        )
        if len(candidates) < count:
            raise HumanIfoError(f"insufficient {split} sessions for unsupervised cohort")
        selected.extend(candidates[:count])
    return selected


def propose_unsupervised_cohort(video_root: Path, output_dir: Path) -> dict[str, object]:
    output = _under_large_root(output_dir, output=True)
    rows: list[dict[str, object]] = []
    for split in ("train", "dev"):
        shards, _count = _selected_video_shards(video_root, split)
        first_shard: dict[str, dict[str, object]] = {}
        for shard in shards:
            hashes = cast(list[object], shard["session_hashes"])
            if len(hashes) == 1:
                first_shard.setdefault(str(hashes[0]), shard)
        for session, shard in sorted(first_shard.items()):
            path = video_root / "shards" / str(shard["path"])
            with np.load(path, allow_pickle=False) as archive:
                signature, qc_passed = _technical_signature(np.asarray(archive["frames"]))
            rows.append(
                {
                    "session_hash": session,
                    "split": split,
                    "technical_qc_passed": qc_passed,
                    "signature": signature.round(8).tolist(),
                }
            )
    selected = _select_unsupervised_cohort(rows)
    candidates = [
        {
            "session_hash": row["session_hash"],
            "split": row["split"],
            "technical_qc_passed": row["technical_qc_passed"],
            "technical_distance": row["technical_distance"],
        }
        for row in selected
    ]
    output.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-human-ifo-unsupervised-cohort-v1",
        "status": "COMPLETED",
        "candidates": candidates,
        "candidate_train_sessions": 20,
        "candidate_dev_sessions": 5,
        "semantic_identity_verified": False,
        "human_control_verified": False,
        "human_labels_used": False,
        "promotion_allowed": False,
        "gate_a_allowed": False,
        "video_test_opened": False,
        "device_input_allowed": False,
        "source_paths_persisted": False,
    }
    payload["proposal_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "proposal.json", payload)
    return payload


def _load_unsupervised_proposal(path: Path) -> tuple[list[dict[str, object]], str]:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    claimed = raw.pop("proposal_sha256", None)
    if claimed != _sha(_canonical(raw).encode()):
        raise HumanIfoError("unsupervised cohort proposal hash mismatch")
    if raw.get("schema_version") != "hok-agent-human-ifo-unsupervised-cohort-v1":
        raise HumanIfoError("invalid unsupervised cohort proposal schema")
    if any(
        raw.get(key) is not expected
        for key, expected in (
            ("semantic_identity_verified", False),
            ("human_control_verified", False),
            ("promotion_allowed", False),
            ("gate_a_allowed", False),
            ("video_test_opened", False),
            ("device_input_allowed", False),
        )
    ):
        raise HumanIfoError("unsupervised cohort proposal attempted promotion")
    rows = cast(list[dict[str, object]], raw.get("candidates"))
    if not isinstance(rows, list) or Counter(str(row.get("split")) for row in rows) != Counter(
        train=20, dev=5
    ):
        raise HumanIfoError("unsupervised cohort proposal does not bind 20 train and 5 dev")
    if any(
        not isinstance(row.get("session_hash"), str) or len(str(row["session_hash"])) != 64
        for row in rows
    ):
        raise HumanIfoError("unsupervised cohort proposal session identity is invalid")
    return rows, claimed


def _all_unlabeled_video_rows(video_root: Path) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    for split, expected in (("train", 103), ("dev", 23)):
        shards, count = _selected_video_shards(video_root, split)
        if count != expected:
            raise HumanIfoError(f"broad Human IfO {split} count differs")
        sessions = sorted(
            {
                str(session)
                for shard in shards
                for session in cast(list[object], shard["session_hashes"])
            }
        )
        rows.extend({"session_hash": session, "split": split} for session in sessions)
    digest = _sha(_canonical({"schema_version": BROAD_CONTRACT_SCHEMA, "sessions": rows}).encode())
    return rows, digest


def run_unsupervised_precheck(
    dataset_root: Path,
    checkpoint: Path,
    video_root: Path,
    proposal_path: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HumanIfoError("CUDA requested but unavailable")
    proposal_rows, proposal_sha256 = _load_unsupervised_proposal(proposal_path)
    manifest = load_global_manifest(dataset_root)
    model, metadata = load_global_model(checkpoint, device)
    if metadata.get("manifest_sha256") != str(manifest["manifest_sha256"]):
        raise HumanIfoError("frozen DAgger checkpoint and simulator manifest differ")
    output = _under_large_root(output_dir, output=True)
    train_rows = [row for row in proposal_rows if row["split"] == "train"]
    dev_rows = [row for row in proposal_rows if row["split"] == "dev"]
    contract, contract_sha256 = _contract_payload()
    per_session = cast(int, contract["windows_per_session"])
    train_main, train_minimap, train_hud, train_groups, _train_sessions = _human_windows(
        video_root, train_rows, per_session, "train", max_shards_per_session=1
    )
    dev_main, dev_minimap, dev_hud, dev_groups, _dev_sessions = _human_windows(
        video_root, dev_rows, per_session, "dev", max_shards_per_session=1
    )
    human_train = _tensor_views(train_main, train_minimap, train_hud)
    human_dev = _tensor_views(dev_main, dev_minimap, dev_hud)
    sim = GlobalWindowDataset(dataset_root, "train")
    sim_values = [sim[index] for index in range(min(160, len(sim)))]
    sim_tensors = cast(
        Views,
        tuple(torch.stack([row[field] for row in sim_values]) for field in range(3)),
    )
    sim_groups = _sim_episode_groups(dataset_root, "train", len(sim_values))
    human_auc = _neighbor_auc(model, human_dev, dev_groups, device, shuffle_time=False)
    human_shuffle_auc = _neighbor_auc(model, human_dev, dev_groups, device, shuffle_time=True)
    sim_auc = _neighbor_auc(model, sim_tensors, sim_groups, device, shuffle_time=False)
    sim_shuffle_auc = _neighbor_auc(model, sim_tensors, sim_groups, device, shuffle_time=True)
    support = _nearest_scene_support(model, human_train, sim, device)
    output.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-human-ifo-unsupervised-precheck-v1",
        "status": "COMPLETED",
        "proposal_sha256": proposal_sha256,
        "contract_sha256": contract_sha256,
        "checkpoint_sha256": _sha(checkpoint.read_bytes()),
        "simulator_manifest_sha256": manifest["manifest_sha256"],
        "human_temporal_neighbor_auc": human_auc,
        "human_shuffled_temporal_neighbor_auc": human_shuffle_auc,
        "simulator_temporal_neighbor_auc": sim_auc,
        "simulator_shuffled_temporal_neighbor_auc": sim_shuffle_auc,
        "human_to_sim_nearest_scene_support": support,
        "temporal_signal_present": human_auc > human_shuffle_auc and sim_auc > sim_shuffle_auc,
        "shared_representation_trained": False,
        "semantic_identity_verified": False,
        "human_labels_used": False,
        "promotion_allowed": False,
        "gate_a_allowed": False,
        "inverse_macro_allowed": False,
        "video_test_opened": False,
        "device_input_allowed": False,
        "source_paths_persisted": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload


def _sequence_hidden(
    model: GlobalMacroPolicy, main: torch.Tensor, minimap: torch.Tensor, hud: torch.Tensor
) -> torch.Tensor:
    sequence = model.encode_views(main, minimap, hud).transpose(1, 2)
    for layer in model.temporal:
        sequence = F.relu(layer(F.pad(sequence, (2, 0))))
    return sequence[:, :, -1]


def _hidden_batched(
    model: GlobalMacroPolicy, values: Views, device: torch.device, batch_size: int = 16
) -> torch.Tensor:
    model.eval()
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(values[0]), batch_size):
            batch = tuple(value[start : start + batch_size].to(device) for value in values)
            rows.append(_sequence_hidden(model, *batch).cpu())
    return torch.cat(rows)


def _temporal_loss(sequence: torch.Tensor) -> torch.Tensor:
    current = F.normalize(sequence[:, :, -1], dim=1)
    previous = F.normalize(sequence[:, :, -2], dim=1)
    negative = previous.roll(1, dims=0)
    return F.softplus(-(current * previous).sum(dim=1) + (current * negative).sum(dim=1)).mean()


def _cross_domain_alignment_loss(human: torch.Tensor, simulator: torch.Tensor) -> torch.Tensor:
    human = human[:, -1]
    simulator = simulator[:, -1]
    mean_loss = (human.mean(dim=0) - simulator.mean(dim=0)).square().mean()
    human = human - human.mean(dim=0, keepdim=True)
    simulator = simulator - simulator.mean(dim=0, keepdim=True)
    human_covariance = human.T @ human / max(1, len(human) - 1)
    simulator_covariance = simulator.T @ simulator / max(1, len(simulator) - 1)
    return mean_loss + (human_covariance - simulator_covariance).square().mean()


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if not len(positive) or not len(negative):
        return 0.5
    return float(
        ((positive[:, None] > negative[None, :]).mean())
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )


def _neighbor_auc(
    model: GlobalMacroPolicy,
    values: Views,
    groups: np.ndarray,
    device: torch.device,
    *,
    shuffle_time: bool,
) -> float:
    if shuffle_time:
        order = torch.arange(values[0].shape[1] - 1, -1, -1)
        main, minimap, hud = (value[:, order] for value in values)
        values = main, minimap, hud
    vectors = F.normalize(_hidden_batched(model, values, device), dim=1).numpy()
    positive: list[float] = []
    negative: list[float] = []
    for index in range(len(vectors)):
        same = [j for j in range(len(vectors)) if j != index and groups[j] == groups[index]]
        other = [j for j in range(len(vectors)) if groups[j] != groups[index]]
        if same and other:
            neighbor = min(same, key=lambda candidate: abs(candidate - index))
            positive.append(float(vectors[index] @ vectors[neighbor]))
            negative.append(float(vectors[index] @ vectors[other[0]]))
    return _auc(np.asarray(positive), np.asarray(negative))


def _sim_auxiliary_metrics(
    model: GlobalMacroPolicy, dataset_root: Path, device: torch.device
) -> dict[str, object]:
    dev = GlobalWindowDataset(dataset_root, "dev")
    return _evaluate_model(model, DataLoader(dev, batch_size=8, shuffle=False), device)


def _sim_episode_groups(dataset_root: Path, split: str, limit: int) -> np.ndarray:
    manifest = load_global_manifest(dataset_root)
    groups: list[int] = []
    group = 0
    for row in cast(list[dict[str, object]], manifest["episodes"]):
        if row["split"] != split:
            continue
        groups.extend([group] * max(0, cast(int, row["rows"]) - 15))
        group += 1
        if len(groups) >= limit:
            break
    if len(groups) < limit:
        raise HumanIfoError("simulator episode groups are incomplete")
    return np.asarray(groups[:limit])


def _nearest_scene_support(
    model: GlobalMacroPolicy,
    human: Views,
    sim: GlobalWindowDataset,
    device: torch.device,
) -> dict[str, object]:
    human_z = F.normalize(_hidden_batched(model, human, device), dim=1)
    sim_rows = [sim[index] for index in range(min(160, len(sim)))]
    sim_views = cast(
        Views,
        tuple(torch.stack([row[field] for row in sim_rows]) for field in range(3)),
    )
    scenes = [int(row[5]) for row in sim_rows]
    sim_z = F.normalize(_hidden_batched(model, sim_views, device), dim=1)
    nearest = (human_z @ sim_z.T).argmax(dim=1).tolist()
    labels = [SCENES[scenes[index]] for index in nearest]
    counts = Counter(labels)
    return {
        "counts": dict(sorted(counts.items())),
        "unique_scenes": len(counts),
        "maximum_scene_fraction": max(counts.values()) / len(labels),
    }


def _domain_probe(
    model: GlobalMacroPolicy,
    human: Views,
    sim: GlobalWindowDataset,
    device: torch.device,
) -> float:
    human_z = _hidden_batched(model, human, device)
    sim_rows = [sim[index] for index in range(min(len(human_z), len(sim)))]
    sim_views = cast(
        Views,
        tuple(torch.stack([row[field] for row in sim_rows]) for field in range(3)),
    )
    sim_z = _hidden_batched(model, sim_views, device)
    values = torch.cat((human_z, sim_z))
    labels = torch.cat((torch.ones(len(human_z)), torch.zeros(len(sim_z))))
    probe = nn.Linear(values.shape[1], 1)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.05)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(probe(values).squeeze(1), labels).backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    return float(((probe(values).squeeze(1) >= 0) == labels.bool()).float().mean())


def run_gate_a(
    dataset_root: Path,
    checkpoint: Path,
    video_root: Path,
    cohort_path: Path | None,
    output_dir: Path,
    *,
    device_name: str,
    contract_path: Path = Path("configs/human_ifo_v1.json"),
    proposal_path: Path | None = None,
    broad_unlabeled: bool = False,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HumanIfoError("CUDA requested but unavailable")
    contract, _contract_sha = _contract_payload(contract_path)
    diagnostic_only = proposal_path is not None or broad_unlabeled
    if diagnostic_only:
        if broad_unlabeled:
            rows, diagnostic_cohort_sha256 = _all_unlabeled_video_rows(video_root)
        else:
            rows, diagnostic_cohort_sha256 = _load_unsupervised_proposal(cast(Path, proposal_path))
        manifest = load_global_manifest(dataset_root)
        _frozen, metadata = load_global_model(checkpoint, torch.device("cpu"))
        if metadata.get("manifest_sha256") != str(manifest["manifest_sha256"]):
            raise HumanIfoError("frozen DAgger checkpoint and simulator manifest differ")
        preflight: dict[str, object] = {
            "contract_sha256": _contract_sha,
            "cohort_sha256": diagnostic_cohort_sha256,
            "cohort_mode": "full_train_dev_unlabeled"
            if broad_unlabeled
            else "technical_20_5_unlabeled",
            "checkpoint_sha256": _sha(checkpoint.read_bytes()),
            "simulator_manifest_sha256": manifest["manifest_sha256"],
            "semantic_identity_verified": False,
            "human_labels_used": False,
            "video_test_opened": False,
            "device_input_allowed": False,
        }
    else:
        if cohort_path is None:
            raise HumanIfoError("local Human IfO cohort is required")
        preflight = gate_a_contract_check(
            dataset_root, checkpoint, video_root, cohort_path, contract_path=contract_path
        )
        rows, _cohort_sha = load_human_cohort(cohort_path, video_root)
    output = _under_large_root(output_dir, output=True)
    train_rows = [row for row in rows if row["split"] == "train"]
    per_session = cast(int, contract["windows_per_session"])
    main, minimap, hud, groups, _sessions = _human_windows(
        video_root,
        train_rows,
        per_session,
        "train",
        max_shards_per_session=1 if diagnostic_only else None,
    )
    dev_rows = [row for row in rows if row["split"] == "dev"]
    dev_main, dev_minimap, dev_hud, dev_groups, _dev_sessions = _human_windows(
        video_root,
        dev_rows,
        per_session,
        "dev",
        max_shards_per_session=1 if diagnostic_only else None,
    )
    human = _tensor_views(main, minimap, hud)
    human_dev = _tensor_views(dev_main, dev_minimap, dev_hud)
    sim = GlobalWindowDataset(dataset_root, "train")
    model, _metadata = load_global_model(checkpoint, device)
    before = _sim_auxiliary_metrics(model, dataset_root, device)
    np.random.seed(0)
    torch.manual_seed(0)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cast(float, contract["learning_rate"]))
    indices = torch.arange(len(human[0]))
    epochs = cast(int, contract["epochs"])
    batch_size = cast(int, contract["batch_size"])
    for _epoch in range(epochs):
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = tuple(value[selected].to(device) for value in human)
            sim_rows = [sim[int(index) % len(sim)] for index in selected.tolist()]
            sim_batch = tuple(
                torch.stack([row[field] for row in sim_rows]).to(device) for field in range(7)
            )
            optimizer.zero_grad(set_to_none=True)
            human_sequence = model.encode_views(*batch)
            sim_sequence = model.encode_views(sim_batch[0], sim_batch[1], sim_batch[2])
            human_loss = _temporal_loss(human_sequence)
            sim_loss = _temporal_loss(sim_sequence)
            alignment_loss = _cross_domain_alignment_loss(human_sequence, sim_sequence)
            logits = model(sim_batch[0], sim_batch[1], sim_batch[2])
            auxiliary = F.cross_entropy(logits[0], sim_batch[3]) + F.cross_entropy(
                logits[1], sim_batch[4]
            )
            loss = (
                cast(float, contract["contrast_weight"]) * (human_loss + sim_loss)
                + cast(float, contract["simulator_auxiliary_weight"]) * auxiliary
                + cast(float, contract["cross_domain_alignment_weight"]) * alignment_loss
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    after = _sim_auxiliary_metrics(model, dataset_root, device)
    human_auc = _neighbor_auc(model, human_dev, dev_groups, device, shuffle_time=False)
    human_shuffle_auc = _neighbor_auc(model, human_dev, dev_groups, device, shuffle_time=True)
    sim_values = [sim[index] for index in range(min(160, len(sim)))]
    sim_tensors = cast(
        Views,
        tuple(torch.stack([row[field] for row in sim_values]) for field in range(3)),
    )
    sim_groups = _sim_episode_groups(dataset_root, "train", len(sim_values))
    sim_auc = _neighbor_auc(model, sim_tensors, sim_groups, device, shuffle_time=False)
    sim_shuffle_auc = _neighbor_auc(model, sim_tensors, sim_groups, device, shuffle_time=True)
    support = _nearest_scene_support(model, human, sim, device)
    probe_accuracy = _domain_probe(model, human, sim, device)
    degradation = max(
        cast(float, before["intent_macro_f1"]) - cast(float, after["intent_macro_f1"]),
        cast(float, before["zone_macro_f1"]) - cast(float, after["zone_macro_f1"]),
    )
    failures: list[str] = []
    if min(human_auc, sim_auc) < cast(float, contract["minimum_temporal_neighbor_auc"]):
        failures.append("temporal_neighbor_auc")
    if not (human_auc > human_shuffle_auc and sim_auc > sim_shuffle_auc):
        failures.append("temporal_order_not_better_than_shuffle")
    if degradation > cast(float, contract["maximum_simulator_macro_f1_degradation"]):
        failures.append("simulator_macro_regression")
    if cast(int, support["unique_scenes"]) < cast(
        int, contract["minimum_nearest_scene_count"]
    ) or cast(float, support["maximum_scene_fraction"]) > cast(
        float, contract["maximum_nearest_scene_fraction"]
    ):
        failures.append("human_to_sim_neighborhood_collapsed")
    criteria_met = not failures
    gate_a_passed = criteria_met and not diagnostic_only
    output.mkdir(parents=True)
    checkpoint_out = output / "shared.safetensors"
    save_file(
        model.state_dict(),
        checkpoint_out,
        {
            "schema_version": "hok-agent-human-ifo-shared-representation-v1",
            "source_checkpoint_sha256": str(preflight["checkpoint_sha256"]),
            "contract_sha256": str(preflight["contract_sha256"]),
            "cohort_sha256": str(preflight["cohort_sha256"]),
            "device_input_allowed": "false",
        },
    )
    payload: dict[str, object] = {
        **preflight,
        "schema_version": "hok-agent-human-ifo-gate-a-report-v1",
        "status": "PASSED"
        if gate_a_passed
        else "DIAGNOSTIC_COMPLETED"
        if diagnostic_only
        else "FAILED",
        "gate_a_passed": gate_a_passed,
        "diagnostic_gate_criteria_met": criteria_met,
        "failure_reasons": failures,
        "human_temporal_neighbor_auc": human_auc,
        "human_shuffled_temporal_neighbor_auc": human_shuffle_auc,
        "simulator_temporal_neighbor_auc": sim_auc,
        "simulator_shuffled_temporal_neighbor_auc": sim_shuffle_auc,
        "simulator_macro_before": before,
        "simulator_macro_after": after,
        "simulator_macro_f1_degradation": degradation,
        "human_to_sim_nearest_scene_support": support,
        "linear_domain_probe_accuracy_diagnostic": probe_accuracy,
        "human_train_windows": len(human[0]),
        "human_dev_used_for_training": False,
        "shared_checkpoint_sha256": _sha(checkpoint_out.read_bytes()),
        "representation_repairs_remaining": 0 if diagnostic_only else 1 if failures else 0,
        "semantic_identity_verified": not diagnostic_only,
        "promotion_allowed": gate_a_passed,
        "inverse_macro_allowed": gate_a_passed,
        "device_input_allowed": False,
        "source_paths_persisted": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload


def run_unsupervised_repair(
    dataset_root: Path,
    checkpoint: Path,
    video_root: Path,
    proposal_path: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    return run_gate_a(
        dataset_root,
        checkpoint,
        video_root,
        None,
        output_dir,
        device_name=device_name,
        proposal_path=proposal_path,
    )


def run_broad_unlabeled_representation(
    dataset_root: Path,
    checkpoint: Path,
    video_root: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, object]:
    return run_gate_a(
        dataset_root,
        checkpoint,
        video_root,
        None,
        output_dir,
        device_name=device_name,
        contract_path=Path("configs/human_ifo_broad_v1.json"),
        broad_unlabeled=True,
    )


def _load_rebind_contract(path: Path) -> tuple[dict[str, object], str]:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    claimed = raw.pop("contract_sha256", None)
    digest = _sha(_canonical(raw).encode())
    if claimed != digest or raw.get("schema_version") != ("hok-agent-human-ifo-encoder-rebind-v1"):
        raise HumanIfoError("invalid encoder rebind contract")
    if raw.get("human_labels_allowed") is not False or raw.get("device_input_allowed") is not False:
        raise HumanIfoError("encoder rebind contract exceeded offline simulator authority")
    return raw, digest


def _load_broad_shared_for_rebind(
    checkpoint: Path, acceptance_path: Path, device: torch.device
) -> tuple[GlobalMacroPolicy, str]:
    acceptance = cast(dict[str, object], json.loads(acceptance_path.read_text(encoding="utf-8")))
    claimed = acceptance.pop("contract_sha256", None)
    if claimed != _sha(_canonical(acceptance).encode()):
        raise HumanIfoError("broad acceptance hash mismatch")
    if _sha(checkpoint.read_bytes()) != acceptance.get("shared_checkpoint_sha256"):
        raise HumanIfoError("broad shared checkpoint differs from acceptance")
    model = GlobalMacroPolicy("tcn")
    model.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
    model.to(device).eval()
    return model, claimed


def _challenge_score(model: GlobalMacroPolicy, device: torch.device) -> dict[str, object]:
    config, _digest = load_global_config()
    scenarios = cast(list[str], config["challenge_scenarios"])
    passed: list[str] = []
    for index, name in enumerate(scenarios):
        arena, expected_intent, expected_zone = _challenge_arena(name)
        observation = arena.observe("blue")
        frame = render_views(observation, 70_000 + index)
        frames: deque[tuple[np.ndarray, ...]] = deque([frame] * WINDOW_FRAMES, maxlen=WINDOW_FRAMES)
        command = _predict_command(model, frames, device)
        if (command.intent, command.target_zone) == (expected_intent, expected_zone):
            passed.append(name)
    return {"passed": len(passed), "total": len(scenarios), "passed_scenarios": passed}


def rebind_human_encoder_to_simulator_policy(
    dataset_root: Path,
    shared_checkpoint: Path,
    broad_acceptance_path: Path,
    dagger_checkpoint: Path,
    output_dir: Path,
    *,
    device_name: str,
    contract_path: Path = Path("configs/human_ifo_encoder_rebind_v1.json"),
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HumanIfoError("CUDA requested but unavailable")
    contract, contract_sha256 = _load_rebind_contract(contract_path)
    candidate, broad_acceptance_sha256 = _load_broad_shared_for_rebind(
        shared_checkpoint, broad_acceptance_path, device
    )
    dagger, dagger_metadata = load_global_model(dagger_checkpoint, device)
    manifest = load_global_manifest(dataset_root)
    if dagger_metadata.get("manifest_sha256") != str(manifest["manifest_sha256"]):
        raise HumanIfoError("Dagger checkpoint and simulator dataset differ")
    for module in (candidate.main, candidate.minimap, candidate.hud, candidate.project):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    train = GlobalWindowDataset(dataset_root, "train")
    dev = GlobalWindowDataset(dataset_root, "dev")
    weights = (
        _class_weights(train, 3, 5).to(device),
        _class_weights(train, 4, 4).to(device),
        _class_weights(train, 5, len(SCENES)).to(device),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=cast(float, contract["learning_rate"]),
        weight_decay=1e-4,
    )
    loader = DataLoader(
        train,
        batch_size=cast(int, contract["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(cast(int, contract["seed"])),
    )
    torch.manual_seed(cast(int, contract["seed"]))
    dagger.eval()
    for _epoch in range(cast(int, contract["epochs"])):
        candidate.train()
        for module in (candidate.main, candidate.minimap, candidate.hud, candidate.project):
            module.eval()
        for main, minimap, hud, intent, zone, scene, _tick in loader:
            main, minimap, hud = main.to(device), minimap.to(device), hud.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = candidate(main, minimap, hud)
            with torch.no_grad():
                teacher_logits = dagger(main, minimap, hud)
            supervised = (
                F.cross_entropy(logits[0], intent.to(device), weight=weights[0])
                + F.cross_entropy(logits[1], zone.to(device), weight=weights[1])
                + cast(float, contract["scene_loss_weight"])
                * F.cross_entropy(logits[2], scene.to(device), weight=weights[2])
            )
            distillation = F.kl_div(
                F.log_softmax(logits[0], dim=1),
                F.softmax(teacher_logits[0], dim=1),
                reduction="batchmean",
            ) + F.kl_div(
                F.log_softmax(logits[1], dim=1),
                F.softmax(teacher_logits[1], dim=1),
                reduction="batchmean",
            )
            loss = supervised + cast(float, contract["dagger_distillation_weight"]) * distillation
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    dev_metrics = _evaluate_model(candidate, DataLoader(dev, batch_size=32, shuffle=False), device)
    baseline = _rollout_summary(
        [
            _student_rollout(dagger, seed, device, authority_fraction=1.0, collect=False)[0]
            for seed in HOLDOUT_SEEDS
        ]
    )
    rebound = _rollout_summary(
        [
            _student_rollout(candidate, seed, device, authority_fraction=1.0, collect=False)[0]
            for seed in HOLDOUT_SEEDS
        ]
    )
    baseline_challenge = _challenge_score(dagger, device)
    rebound_challenge = _challenge_score(candidate, device)
    promotion_allowed = (
        cast(int, rebound["safety_violations"]) == 0
        and cast(int, rebound["invalid_actions"]) == 0
        and cast(int, rebound["non_timeout_terminals"])
        >= cast(int, baseline["non_timeout_terminals"])
        and cast(int, rebound["tower_progress_episodes"])
        >= cast(int, baseline["tower_progress_episodes"])
        and cast(float, rebound["mean_stuck_time_ratio"])
        <= cast(float, baseline["mean_stuck_time_ratio"])
        + cast(float, contract["maximum_stuck_regression"])
        and cast(int, rebound_challenge["passed"])
        >= cast(int, contract["minimum_challenge_passes"])
    )
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    rebound_path = output / "rebound.safetensors"
    rebound_sha256 = _save_model(rebound_path, candidate, "tcn", str(manifest["manifest_sha256"]))
    payload: dict[str, object] = {
        "schema_version": "hok-agent-human-ifo-encoder-rebind-report-v1",
        "status": "PASSED" if promotion_allowed else "FAILED",
        "promotion_allowed": promotion_allowed,
        "selected_model": "rebound" if promotion_allowed else "dagger",
        "contract_sha256": contract_sha256,
        "broad_acceptance_sha256": broad_acceptance_sha256,
        "dagger_checkpoint_sha256": _sha(dagger_checkpoint.read_bytes()),
        "rebound_checkpoint_sha256": rebound_sha256,
        "dev_metrics": dev_metrics,
        "baseline_holdout": baseline,
        "rebound_holdout": rebound,
        "baseline_challenge": baseline_challenge,
        "rebound_challenge": rebound_challenge,
        "human_video_role": "visual_encoder_adaptation_only",
        "strategy_supervision": "GlobalArena_rule_teacher_only",
        "dagger_fallback_preserved": True,
        "human_labels_used": False,
        "video_test_opened": False,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload


def _load_curriculum_contract(path: Path) -> tuple[dict[str, object], str]:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    claimed = raw.pop("contract_sha256", None)
    digest = _sha(_canonical(raw).encode())
    if claimed != digest or raw.get("schema_version") != (
        "hok-agent-global-challenge-curriculum-v1"
    ):
        raise HumanIfoError("invalid challenge curriculum contract")
    if raw.get("human_video_used") is not False or raw.get("device_input_allowed") is not False:
        raise HumanIfoError("challenge curriculum must remain simulator-only")
    return raw, digest


def _parameterized_challenge_arena(name: str, variant: int, split: str) -> GlobalArena:
    arena, _intent, _zone = _challenge_arena(name)
    holdout = split == "holdout"
    index = variant
    if name == "low_health_far_from_base":
        arena.state.blue.x = (5, 11)[index % 2] if holdout else (6, 7, 9, 10)[index % 4]
        arena.state.blue.health = 2 if holdout else (1, 3)[index % 2]
    elif name == "low_health_at_base":
        arena.state.blue.x = 0 if holdout else 1
        arena.state.blue.health = 2 if holdout else (1, 3)[index % 2]
        arena.state.blue.cooldowns["skill1"] = (index % 2) + 1
    elif name == "wave_in_tower_range":
        arena.state.blue.x = 12 if holdout else (11, 13)[index % 2]
        arena.state.red.x = (7, 9)[index % 2] if holdout else (6, 10)[index % 2]
    elif name == "enemy_hero_contact":
        arena.state.blue.x = (5, 10)[index % 2] if holdout else (6, 7, 8, 9)[index % 4]
        arena.state.red.x = arena.state.blue.x + (-1 if arena.state.blue.x >= 10 else 1)
    elif name == "ordinary_lane_advance":
        arena.state.blue.x = 2 if holdout else (1, 3, 4)[index % 3]
        arena.state.red.x = (10, 13)[index % 2] if holdout else (11, 12)[index % 2]
        arena.state.blue.cooldowns["skill2"] = 1 + (index % 2)
    elif name == "tower_destroyed_crystal_range":
        arena.state.blue.x = 13 if holdout else (12, 14)[index % 2]
        arena.state.red.x = (8, 10)[index % 2] if holdout else 9
    else:
        raise HumanIfoError(f"unknown curriculum family: {name}")
    return arena


def _curriculum_samples(split: str, variants_per_family: int) -> list[tuple[np.ndarray, ...]]:
    config, _digest = load_global_config()
    families = cast(list[str], config["challenge_scenarios"])
    teacher = GlobalRuleTeacher()
    rows: list[tuple[np.ndarray, ...]] = []
    for family_index, name in enumerate(families):
        for variant in range(variants_per_family):
            arena = _parameterized_challenge_arena(name, variant, split)
            observation = arena.observe("blue")
            decision = teacher.decide("blue", arena.legal_actions("blue"), observation)
            render_seed = (81_000 if split == "train" else 91_000) + family_index * 100 + variant
            frame = render_views(observation, render_seed)
            frames: deque[tuple[np.ndarray, ...]] = deque(
                [frame] * WINDOW_FRAMES, maxlen=WINDOW_FRAMES
            )
            rows.append(
                _window_sample(
                    frames,
                    decision.command.intent,
                    decision.command.target_zone,
                    decision.scene_id,
                    arena.state.tick,
                )
            )
    return rows


def _parameterized_challenge_score(
    model: GlobalMacroPolicy, device: torch.device, variants_per_family: int
) -> dict[str, object]:
    config, _digest = load_global_config()
    families = cast(list[str], config["challenge_scenarios"])
    teacher = GlobalRuleTeacher()
    passed = 0
    family_passes: Counter[str] = Counter()
    for family_index, name in enumerate(families):
        for variant in range(variants_per_family):
            arena = _parameterized_challenge_arena(name, variant, "holdout")
            observation = arena.observe("blue")
            expected = teacher.decide("blue", arena.legal_actions("blue"), observation).command
            frame = render_views(observation, 91_000 + family_index * 100 + variant)
            frames: deque[tuple[np.ndarray, ...]] = deque(
                [frame] * WINDOW_FRAMES, maxlen=WINDOW_FRAMES
            )
            command = _predict_command(model, frames, device)
            if (command.intent, command.target_zone) == (
                expected.intent,
                expected.target_zone,
            ):
                passed += 1
                family_passes[name] += 1
    total = len(families) * variants_per_family
    return {
        "passed": passed,
        "total": total,
        "fraction": passed / total,
        "family_passes": dict(sorted(family_passes.items())),
    }


def train_parameterized_challenge_curriculum(
    dataset_root: Path,
    dagger_checkpoint: Path,
    output_dir: Path,
    *,
    device_name: str,
    contract_path: Path = Path("configs/global_challenge_curriculum_v1.json"),
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HumanIfoError("CUDA requested but unavailable")
    contract, contract_sha256 = _load_curriculum_contract(contract_path)
    dagger, metadata = load_global_model(dagger_checkpoint, device)
    candidate = deepcopy(dagger)
    manifest = load_global_manifest(dataset_root)
    if metadata.get("manifest_sha256") != str(manifest["manifest_sha256"]):
        raise HumanIfoError("Dagger checkpoint and simulator dataset differ")
    train = GlobalWindowDataset(dataset_root, "train")
    augmented = copy(train)
    curriculum = _curriculum_samples("train", cast(int, contract["train_variants_per_family"]))
    augmented.samples = [
        *train.samples,
        *(curriculum * cast(int, contract["curriculum_repeat"])),
    ]
    for module in (candidate.main, candidate.minimap, candidate.hud, candidate.project):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    weights = (
        _class_weights(augmented, 3, 5).to(device),
        _class_weights(augmented, 4, 4).to(device),
        _class_weights(augmented, 5, len(SCENES)).to(device),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=cast(float, contract["learning_rate"]),
        weight_decay=1e-4,
    )
    loader = DataLoader(
        augmented,
        batch_size=cast(int, contract["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(cast(int, contract["seed"])),
    )
    torch.manual_seed(cast(int, contract["seed"]))
    dagger.eval()
    for _epoch in range(cast(int, contract["epochs"])):
        candidate.train()
        for module in (candidate.main, candidate.minimap, candidate.hud, candidate.project):
            module.eval()
        for main, minimap, hud, intent, zone, scene, _tick in loader:
            main, minimap, hud = main.to(device), minimap.to(device), hud.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = candidate(main, minimap, hud)
            with torch.no_grad():
                teacher_logits = dagger(main, minimap, hud)
            supervised = (
                F.cross_entropy(logits[0], intent.to(device), weight=weights[0])
                + F.cross_entropy(logits[1], zone.to(device), weight=weights[1])
                + cast(float, contract["scene_loss_weight"])
                * F.cross_entropy(logits[2], scene.to(device), weight=weights[2])
            )
            distillation = F.kl_div(
                F.log_softmax(logits[0], dim=1),
                F.softmax(teacher_logits[0], dim=1),
                reduction="batchmean",
            ) + F.kl_div(
                F.log_softmax(logits[1], dim=1),
                F.softmax(teacher_logits[1], dim=1),
                reduction="batchmean",
            )
            loss = supervised + cast(float, contract["dagger_distillation_weight"]) * distillation
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    baseline_holdout = _rollout_summary(
        [
            _student_rollout(dagger, seed, device, authority_fraction=1.0, collect=False)[0]
            for seed in HOLDOUT_SEEDS
        ]
    )
    candidate_holdout = _rollout_summary(
        [
            _student_rollout(candidate, seed, device, authority_fraction=1.0, collect=False)[0]
            for seed in HOLDOUT_SEEDS
        ]
    )
    baseline_challenge = _challenge_score(dagger, device)
    candidate_challenge = _challenge_score(candidate, device)
    parameterized = _parameterized_challenge_score(
        candidate, device, cast(int, contract["holdout_variants_per_family"])
    )
    promotion_allowed = (
        cast(int, candidate_holdout["safety_violations"]) == 0
        and cast(int, candidate_holdout["invalid_actions"]) == 0
        and cast(int, candidate_holdout["non_timeout_terminals"])
        >= cast(int, baseline_holdout["non_timeout_terminals"])
        and cast(int, candidate_holdout["tower_progress_episodes"])
        >= cast(int, baseline_holdout["tower_progress_episodes"])
        and cast(float, candidate_holdout["mean_stuck_time_ratio"])
        <= cast(float, baseline_holdout["mean_stuck_time_ratio"])
        + cast(float, contract["maximum_stuck_regression"])
        and cast(int, candidate_challenge["passed"])
        >= cast(int, contract["minimum_canonical_challenge_passes"])
        and cast(float, parameterized["fraction"])
        >= cast(float, contract["minimum_parameterized_holdout_fraction"])
    )
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    checkpoint = output / "curriculum.safetensors"
    checkpoint_sha256 = _save_model(checkpoint, candidate, "tcn", str(manifest["manifest_sha256"]))
    payload: dict[str, object] = {
        "schema_version": "hok-agent-global-challenge-curriculum-report-v1",
        "status": "PASSED" if promotion_allowed else "FAILED",
        "promotion_allowed": promotion_allowed,
        "selected_model": "curriculum" if promotion_allowed else "dagger",
        "contract_sha256": contract_sha256,
        "dagger_checkpoint_sha256": _sha(dagger_checkpoint.read_bytes()),
        "curriculum_checkpoint_sha256": checkpoint_sha256,
        "train_curriculum_samples": len(curriculum),
        "train_holdout_rgb_overlap": False,
        "baseline_holdout": baseline_holdout,
        "candidate_holdout": candidate_holdout,
        "baseline_canonical_challenge": baseline_challenge,
        "candidate_canonical_challenge": candidate_challenge,
        "candidate_parameterized_holdout": parameterized,
        "human_video_used": False,
        "dagger_fallback_preserved": True,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload


def _observable_factor_labels(observation: dict[str, object]) -> dict[str, int]:
    own = cast(dict[str, int], observation["self_position"])
    opponent = cast(dict[str, int], observation["opponent_position"])
    health = cast(int, observation["self_health"])
    distance = abs(own["x"] - opponent["x"]) + abs(own["y"] - opponent["y"])
    tower_health = cast(int, observation["enemy_tower_health"])
    health_bucket = 0 if health <= 2 else 1 if health <= 5 else 2
    distance_bucket = 0 if distance <= 1 else 1 if distance <= 4 else 2
    push_condition = int(tower_health == 0 or own["x"] >= 11)
    ordinary_lane = int(health > 2 and tower_health > 0 and 3 <= own["x"] < 11 and distance > 4)
    return {
        "health_bucket": health_bucket,
        "at_own_base": int(own["x"] <= 2),
        "enemy_distance_bucket": distance_bucket,
        "push_condition": push_condition,
        "ordinary_lane_advance": ordinary_lane,
    }


def _observable_probe_data(split: str) -> tuple[Views, dict[str, torch.Tensor]]:
    health_values: tuple[int, ...]
    x_values: tuple[int, ...]
    distance_values: tuple[int, ...]
    tower_values: tuple[int, ...]
    if split == "train":
        health_values = (1, 2, 4, 7, 10)
        x_values = (0, 2, 5, 8, 11, 13)
        distance_values = (0, 1, 3, 6)
        tower_values = (0, 12)
        render_base = 120_000
    elif split == "dev":
        health_values = (3, 5, 8, 9)
        x_values = (1, 3, 6, 9, 12, 14)
        distance_values = (0, 2, 4, 7)
        tower_values = (0, 6, 12)
        render_base = 130_000
    else:
        raise HumanIfoError("observable probe split must be train or dev")
    main: list[np.ndarray] = []
    minimap: list[np.ndarray] = []
    hud: list[np.ndarray] = []
    labels: dict[str, list[int]] = {name: [] for name in OBSERVABLE_FACTORS}
    index = 0
    for health in health_values:
        for own_x in x_values:
            for requested_distance in distance_values:
                for tower_health in tower_values:
                    arena = GlobalArena()
                    arena.reset(index)
                    arena.state.blue.health = health
                    arena.state.blue.x, arena.state.blue.y = own_x, 3
                    direction = -1 if index % 2 else 1
                    arena.state.red.x = min(14, max(0, own_x + direction * requested_distance))
                    arena.state.red.y = 3
                    arena.state.red_tower_health = tower_health
                    observation = arena.observe("blue")
                    frame = render_views(observation, render_base + index)
                    main.append(np.stack([frame[0]] * WINDOW_FRAMES))
                    minimap.append(np.stack([frame[1]] * WINDOW_FRAMES))
                    hud.append(np.stack([frame[2]] * WINDOW_FRAMES))
                    for name, value in _observable_factor_labels(observation).items():
                        labels[name].append(value)
                    index += 1

    def convert(values: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(values)).permute(0, 1, 4, 2, 3).float().div(255.0)

    return (
        (convert(main), convert(minimap), convert(hud)),
        {name: torch.tensor(values, dtype=torch.long) for name, values in labels.items()},
    )


def _train_factor_probe(
    train_features: torch.Tensor,
    train_labels: dict[str, torch.Tensor],
    dev_features: torch.Tensor,
    dev_labels: dict[str, torch.Tensor],
    contract: dict[str, object],
    *,
    shuffle_labels: bool,
) -> dict[str, float]:
    torch.manual_seed(cast(int, contract["seed"]))
    probe = ObservableFactorProbe()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=cast(float, contract["learning_rate"]))
    labels = train_labels
    if shuffle_labels:
        generator = torch.Generator().manual_seed(cast(int, contract["seed"]))
        labels = {
            name: values[torch.randperm(len(values), generator=generator)]
            for name, values in train_labels.items()
        }
    indices = torch.arange(len(train_features))
    generator = torch.Generator().manual_seed(cast(int, contract["seed"]))
    loader = DataLoader(
        TensorDataset(indices),
        batch_size=cast(int, contract["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    weights = {
        name: (len(values) / torch.bincount(values, minlength=classes).float().clamp_min(1.0))
        for (name, classes), values in zip(OBSERVABLE_FACTORS.items(), labels.values(), strict=True)
    }
    for _epoch in range(cast(int, contract["epochs"])):
        probe.train()
        for (selected,) in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(train_features[selected])
            loss = torch.stack(
                [
                    F.cross_entropy(logits[name], labels[name][selected], weight=weights[name])
                    for name in OBSERVABLE_FACTORS
                ]
            ).sum()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    probe.eval()
    with torch.no_grad():
        logits = probe(dev_features)
    return {
        name: _macro_f1(
            dev_labels[name].tolist(),
            logits[name].argmax(dim=1).tolist(),
            classes,
        )
        for name, classes in OBSERVABLE_FACTORS.items()
    }


def run_observable_factor_probe(
    dagger_checkpoint: Path,
    output_dir: Path,
    *,
    device_name: str,
    contract_path: Path = Path("configs/global_observable_factor_probe_v1.json"),
) -> dict[str, object]:
    device = torch.device(device_name)
    contract = cast(dict[str, object], json.loads(contract_path.read_text(encoding="utf-8")))
    claimed = contract.pop("contract_sha256", None)
    contract_sha256 = _sha(_canonical(contract).encode())
    if claimed != contract_sha256 or contract.get("schema_version") != (
        "hok-agent-global-observable-factor-probe-v1"
    ):
        raise HumanIfoError("invalid observable factor probe contract")
    if contract.get("device_input_allowed") is not False:
        raise HumanIfoError("observable factor probe must remain offline")
    dagger, _metadata = load_global_model(dagger_checkpoint, device)
    train_views, train_labels = _observable_probe_data("train")
    dev_views, dev_labels = _observable_probe_data("dev")
    train_features = _hidden_batched(dagger, train_views, device)
    dev_features = _hidden_batched(dagger, dev_views, device)
    normal = _train_factor_probe(
        train_features, train_labels, dev_features, dev_labels, contract, shuffle_labels=False
    )
    shuffled = _train_factor_probe(
        train_features, train_labels, dev_features, dev_labels, contract, shuffle_labels=True
    )
    threshold = cast(float, contract["minimum_macro_f1"])
    margin = cast(float, contract["minimum_shuffle_margin"])
    passed_factors = [
        name
        for name in OBSERVABLE_FACTORS
        if normal[name] >= threshold and normal[name] - shuffled[name] >= margin
    ]
    very_low = [name for name in OBSERVABLE_FACTORS if normal[name] < 0.70]
    representation_sufficient = (
        len(passed_factors) >= cast(int, contract["minimum_passing_factors"]) and not very_low
    )
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-global-observable-factor-probe-report-v1",
        "status": "PASSED" if representation_sufficient else "FAILED",
        "representation_sufficient": representation_sufficient,
        "auxiliary_representation_training_allowed": not representation_sufficient,
        "contract_sha256": contract_sha256,
        "dagger_checkpoint_sha256": _sha(dagger_checkpoint.read_bytes()),
        "normal_macro_f1": normal,
        "label_shuffle_macro_f1": shuffled,
        "passed_factors": passed_factors,
        "very_low_factors": very_low,
        "train_samples": len(train_features),
        "dev_samples": len(dev_features),
        "structured_truth_actor_input": False,
        "human_video_used": False,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload


def _configure_auxiliary_trainable(model: GlobalMacroPolicy) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.main.layer4, model.project):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def train_observable_auxiliary_representation(
    dataset_root: Path,
    dagger_checkpoint: Path,
    probe_report: Path,
    output_dir: Path,
    *,
    device_name: str,
    contract_path: Path = Path("configs/global_observable_aux_training_v1.json"),
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HumanIfoError("CUDA requested but unavailable")
    contract = cast(dict[str, object], json.loads(contract_path.read_text(encoding="utf-8")))
    claimed = contract.pop("contract_sha256", None)
    contract_sha256 = _sha(_canonical(contract).encode())
    if claimed != contract_sha256 or contract.get("schema_version") != (
        "hok-agent-global-observable-aux-training-v1"
    ):
        raise HumanIfoError("invalid observable auxiliary training contract")
    if contract.get("device_input_allowed") is not False:
        raise HumanIfoError("observable auxiliary training must remain offline")
    probe = cast(dict[str, object], json.loads(probe_report.read_text(encoding="utf-8")))
    probe_sha256 = probe.pop("report_sha256", None)
    if (
        probe_sha256 != _sha(_canonical(probe).encode())
        or probe.get("auxiliary_representation_training_allowed") is not True
    ):
        raise HumanIfoError("observable factor probe does not authorize auxiliary training")
    dagger, metadata = load_global_model(dagger_checkpoint, device)
    candidate = deepcopy(dagger)
    manifest = load_global_manifest(dataset_root)
    if metadata.get("manifest_sha256") != str(manifest["manifest_sha256"]):
        raise HumanIfoError("Dagger checkpoint and simulator dataset differ")
    trainable_names = _configure_auxiliary_trainable(candidate)
    expected_prefixes = ("main.layer4.", "project.")
    if any(not name.startswith(expected_prefixes) for name in trainable_names):
        raise HumanIfoError("auxiliary training attempted to unfreeze an unapproved module")
    auxiliary = ObservableFactorProbe().to(device)
    train = GlobalWindowDataset(dataset_root, "train")
    macro_weights = (
        _class_weights(train, 3, 5).to(device),
        _class_weights(train, 4, 4).to(device),
        _class_weights(train, 5, len(SCENES)).to(device),
    )
    factor_views, factor_labels = _observable_probe_data("train")
    factor_values = (*factor_views, *(factor_labels[name] for name in OBSERVABLE_FACTORS))
    factor_loader = DataLoader(
        TensorDataset(*factor_values),
        batch_size=cast(int, contract["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(cast(int, contract["seed"])),
    )
    factor_weights = {
        name: (
            len(factor_labels[name])
            / torch.bincount(factor_labels[name], minlength=classes).float().clamp_min(1.0)
        ).to(device)
        for name, classes in OBSERVABLE_FACTORS.items()
    }
    macro_loader = DataLoader(
        train,
        batch_size=cast(int, contract["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(cast(int, contract["seed"])),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad]
        + list(auxiliary.parameters()),
        lr=cast(float, contract["learning_rate"]),
        weight_decay=1e-4,
    )
    dagger.eval()
    torch.manual_seed(cast(int, contract["seed"]))
    for _epoch in range(cast(int, contract["epochs"])):
        candidate.eval()
        candidate.main.layer4.train()
        candidate.project.train()
        auxiliary.train()
        factor_iterator = iter(factor_loader)
        for main, minimap, hud, intent, zone, scene, _tick in macro_loader:
            try:
                factor_batch = next(factor_iterator)
            except StopIteration:
                factor_iterator = iter(factor_loader)
                factor_batch = next(factor_iterator)
            main, minimap, hud = main.to(device), minimap.to(device), hud.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = candidate(main, minimap, hud)
            with torch.no_grad():
                teacher_logits = dagger(main, minimap, hud)
            macro_loss = (
                F.cross_entropy(logits[0], intent.to(device), weight=macro_weights[0])
                + F.cross_entropy(logits[1], zone.to(device), weight=macro_weights[1])
                + cast(float, contract["scene_loss_weight"])
                * F.cross_entropy(logits[2], scene.to(device), weight=macro_weights[2])
            )
            distillation = F.kl_div(
                F.log_softmax(logits[0], dim=1),
                F.softmax(teacher_logits[0], dim=1),
                reduction="batchmean",
            ) + F.kl_div(
                F.log_softmax(logits[1], dim=1),
                F.softmax(teacher_logits[1], dim=1),
                reduction="batchmean",
            )
            factor_main, factor_minimap, factor_hud = (
                factor_batch[0].to(device),
                factor_batch[1].to(device),
                factor_batch[2].to(device),
            )
            factor_logits = auxiliary(
                _sequence_hidden(candidate, factor_main, factor_minimap, factor_hud)
            )
            auxiliary_loss = torch.stack(
                [
                    F.cross_entropy(
                        factor_logits[name],
                        factor_batch[3 + index].to(device),
                        weight=factor_weights[name],
                    )
                    for index, name in enumerate(OBSERVABLE_FACTORS)
                ]
            ).sum()
            loss = (
                macro_loss
                + cast(float, contract["dagger_distillation_weight"]) * distillation
                + cast(float, contract["auxiliary_loss_weight"]) * auxiliary_loss
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    dev_views, dev_labels = _observable_probe_data("dev")
    dev_features = _hidden_batched(candidate, dev_views, device)
    auxiliary.eval()
    with torch.no_grad():
        factor_logits = auxiliary(dev_features.to(device))
    factor_f1 = {
        name: _macro_f1(
            dev_labels[name].tolist(),
            factor_logits[name].argmax(dim=1).cpu().tolist(),
            classes,
        )
        for name, classes in OBSERVABLE_FACTORS.items()
    }
    baseline_holdout = _rollout_summary(
        [
            _student_rollout(dagger, seed, device, authority_fraction=1.0, collect=False)[0]
            for seed in HOLDOUT_SEEDS
        ]
    )
    candidate_holdout = _rollout_summary(
        [
            _student_rollout(candidate, seed, device, authority_fraction=1.0, collect=False)[0]
            for seed in HOLDOUT_SEEDS
        ]
    )
    canonical = _challenge_score(candidate, device)
    parameterized = _parameterized_challenge_score(candidate, device, 4)
    promotion_allowed = (
        cast(int, candidate_holdout["safety_violations"]) == 0
        and cast(int, candidate_holdout["invalid_actions"]) == 0
        and cast(int, candidate_holdout["non_timeout_terminals"])
        >= cast(int, contract["minimum_non_timeout_terminals"])
        and cast(int, candidate_holdout["tower_progress_episodes"])
        >= cast(int, contract["minimum_tower_progress_episodes"])
        and cast(float, candidate_holdout["mean_stuck_time_ratio"])
        <= cast(float, contract["maximum_stuck_time_ratio"])
        and cast(float, candidate_holdout["mean_tower_damage"])
        >= cast(float, contract["minimum_mean_tower_damage"])
        and cast(int, canonical["passed"])
        >= cast(int, contract["minimum_canonical_challenge_passes"])
        and cast(int, parameterized["passed"])
        >= cast(int, contract["minimum_parameterized_holdout_passes"])
    )
    output = _under_large_root(output_dir, output=True)
    output.mkdir(parents=True)
    checkpoint = output / "observable-aux.safetensors"
    checkpoint_sha256 = _save_model(checkpoint, candidate, "tcn", str(manifest["manifest_sha256"]))
    auxiliary_path = output / "auxiliary-heads.safetensors"
    save_file(auxiliary.state_dict(), auxiliary_path)
    payload: dict[str, object] = {
        "schema_version": "hok-agent-global-observable-aux-report-v1",
        "status": "PASSED" if promotion_allowed else "FAILED",
        "promotion_allowed": promotion_allowed,
        "selected_model": "observable_aux" if promotion_allowed else "dagger",
        "contract_sha256": contract_sha256,
        "probe_report_sha256": probe_sha256,
        "dagger_checkpoint_sha256": _sha(dagger_checkpoint.read_bytes()),
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "auxiliary_heads_sha256": _sha(auxiliary_path.read_bytes()),
        "trainable_parameter_names": trainable_names,
        "factor_dev_macro_f1": factor_f1,
        "baseline_holdout": baseline_holdout,
        "candidate_holdout": candidate_holdout,
        "candidate_canonical_challenge": canonical,
        "candidate_parameterized_holdout": parameterized,
        "auxiliary_training_attempts_remaining": 0,
        "human_video_used": False,
        "structured_truth_actor_input": False,
        "dagger_fallback_preserved": True,
        "device_input_allowed": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload).encode())
    _write_json(output / "report.json", payload)
    return payload
