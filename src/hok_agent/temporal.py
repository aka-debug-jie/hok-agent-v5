# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F

from hok_agent import alignment

ACTION_NAMES: tuple[str, ...] = alignment.ACTION_TYPES
ABSTAIN = "ABSTAIN"
V6_SCHEMA, CHECKPOINT_SCHEMA, TRACKING_SCHEMA, TRACKING_SPLIT_SCHEMA, AUDIT_SCHEMA = (
    "hok-agent-v6-release-v1",
    "hok-agent-v6-temporal-model-v1",
    "hok-agent-v6-tracking-evidence-v1",
    "hok-agent-v6-tracking-split-v1",
    "hok-agent-v6-temporal-audit-v1",
)
ARCHITECTURE = "rgb-dual-hero-hud-causal-depthwise-tcn-v1"
CLAIM_SCOPE = "audited_abstract_host_advice"
OOD_THRESHOLD, TRACKING_QUALITY_THRESHOLD = 0.05, 0.80


@dataclass(frozen=True)
class _Checkpoint:
    sha256: str
    metadata: dict[str, str]
    model: TemporalModel


@dataclass(frozen=True)
class _Evidence:
    sha256: str
    metrics: dict[str, float]
    allowed_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _V6Binding:
    release_sha256: str
    allowed_classes: tuple[str, ...]
    class_thresholds: dict[str, float]


@dataclass(frozen=True)
class _V5Binding:
    release_sha256: str
    model_sha256: str
    manifest_sha256: str
    split_binding_sha256: str
    pre_ingest_sha256: str
    session_splits: dict[str, str]
    allowed_classes: tuple[str, ...]
    class_thresholds: dict[str, float]


@dataclass(frozen=True)
class _TrackingSplitBinding:
    sha256: str
    session_splits: dict[str, str]
    row_identities: tuple[str, ...]
    manifest_sha256: str
    split_binding_sha256: str

    @property
    def test_sessions(self) -> set[str]:
        return {session for session, split in self.session_splits.items() if split == "test"}


class TemporalError(ValueError):
    pass


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(alignment._read_regular(path))


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


ACTION_HASH = _sha(_json(ACTION_NAMES).encode())
TRAINING_CONTRACT_HASH = _sha(_json({"architecture": ARCHITECTURE, "rgb_only": True, "sequence_length": 8, "ood_max": OOD_THRESHOLD, "tracking_quality_min": TRACKING_QUALITY_THRESHOLD, "stable_frames": 5}).encode())


def _load_json(path: Path, fields: set[str], schema: str) -> dict[str, object]:
    try:
        value = alignment._strict_json_bytes(alignment._read_regular(path, ".json"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TemporalError("invalid JSON evidence") from exc
    if type(value) is not dict or set(value) != fields or value.get("schema_version") != schema:
        raise TemporalError("evidence schema/fields are not exact")
    return value


def _number(value: object, *, low: float = 0.0, high: float = math.inf) -> float:
    number = float(cast(Any, value)) if type(value) in {int, float} else math.nan
    if not math.isfinite(number) or not low <= number <= high:
        raise TemporalError("numeric evidence value is invalid")
    return number


def _binary_f1(predicted: list[bool], truth: list[bool]) -> float:
    tp = sum(a and b for a, b in zip(predicted, truth, strict=True))
    fp = sum(a and not b for a, b in zip(predicted, truth, strict=True))
    fn = sum(not a and b for a, b in zip(predicted, truth, strict=True))
    return 2 * tp / max(1, 2 * tp + fp + fn)


def _p95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def _load_v5(
    release_path: Path, model_path: Path, manifest_path: Path, pre_ingest_path: Path, privacy_context_path: Path,
    owner_attestation_path: Path, owner_component_confirmation_path: Path, shard_paths: Sequence[Path]
) -> _V5Binding:
    loader = getattr(alignment, "load_bound_v5_release", None)
    if loader is None:
        raise TemporalError("V5 path binding API is unavailable")
    try:
        manifest = alignment.load_v5_manifest(manifest_path, pre_ingest_path, privacy_context_path, owner_attestation_path, owner_component_confirmation_path, shard_paths)
        value = loader(release_path, model_path)
        release_sha = value.release_sha256
        model_sha = value.model_sha256
        bound_manifest_sha = value.manifest_sha256
        split_binding_sha = value.split_binding_sha256
        manifest_sha = manifest.manifest_sha256
        actual_split_sha = manifest.split_binding_sha256
        pre_ingest_sha = manifest.pre_ingest_sha256
        session_splits = manifest.session_splits
        allowed = value.allowed_classes
        thresholds = value.class_thresholds
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise TemporalError("invalid V5 release/manifest/model binding") from exc
    if (
        release_sha != _file_sha(release_path)
        or model_sha != _file_sha(model_path)
        or bound_manifest_sha != manifest_sha
        or split_binding_sha != actual_split_sha
        or not all(_valid_sha(item) for item in (manifest_sha, split_binding_sha, pre_ingest_sha))
        or type(session_splits) is not dict
    ):
        raise TemporalError("V5 binding does not name the actual files")
    if type(allowed) is not tuple or not allowed or len(set(allowed)) != len(allowed):
        raise TemporalError("V5 classes are invalid")
    if type(thresholds) is not dict or set(thresholds) != set(allowed):
        raise TemporalError("V5 thresholds are invalid")
    parsed = {name: _number(thresholds[name], low=0.75, high=1.0) for name in allowed}
    if any(type(name) is not str or name not in ACTION_NAMES for name in allowed):
        raise TemporalError("V5 class is outside the V6 vocabulary")
    splits = dict(session_splits)
    if not splits or any(not _valid_sha(key) or value not in alignment.SPLITS for key, value in splits.items()):
        raise TemporalError("V5 manifest session split mapping is invalid")
    return _V5Binding(release_sha, model_sha, manifest_sha, split_binding_sha, pre_ingest_sha, splits, allowed, parsed)


def _checkpoint_metadata(v5: _V5Binding, training_artifact_path: Path, tracking_split_path: Path, seed: int) -> dict[str, str]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "v5_release_sha256": v5.release_sha256,
        "v5_model_sha256": v5.model_sha256,
        "v5_manifest_sha256": v5.manifest_sha256,
        "v5_split_binding_sha256": v5.split_binding_sha256,
        "v5_pre_ingest_sha256": v5.pre_ingest_sha256,
        "training_artifact_sha256": _file_sha(training_artifact_path),
        "tracking_split_sha256": _file_sha(tracking_split_path),
        "training_contract_hash": TRAINING_CONTRACT_HASH,
        "action_vocabulary_hash": ACTION_HASH,
        "architecture": ARCHITECTURE,
        "training_seed": str(seed),
        "sequence_length": "8",
        "claim_scope": CLAIM_SCOPE,
        "rgb_only": "true",
        "control_output": "false",
    }


def _load_checkpoint(path: Path, v5: _V5Binding, training_artifact_path: Path, tracking_split_path: Path) -> _Checkpoint:
    if path.is_symlink() or not path.is_file() or path.suffix != ".safetensors":
        raise TemporalError("V6 checkpoint path is invalid")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except (OSError, SafetensorError) as exc:
        raise TemporalError("invalid V6 safetensors checkpoint") from exc
    if metadata is None or set(metadata) != set(_checkpoint_metadata(v5, training_artifact_path, tracking_split_path, 0)):
        raise TemporalError("V6 checkpoint metadata fields are not exact")
    artifact, split_hash, seed_text = (
        metadata.get("training_artifact_sha256", ""),
        metadata.get("tracking_split_sha256", ""),
        metadata.get("training_seed", ""),
    )
    if (
        not _valid_sha(artifact)
        or not _valid_sha(split_hash)
        or not seed_text.isdigit()
        or str(int(seed_text)) != seed_text
    ):
        raise TemporalError("V6 checkpoint provenance is invalid")
    expected = _checkpoint_metadata(v5, training_artifact_path, tracking_split_path, int(seed_text))
    if split_hash != expected["tracking_split_sha256"]:
        raise TemporalError("V6 checkpoint split binding is unbound")
    if metadata != expected:
        raise TemporalError("V6 checkpoint contract mismatch")
    model = TemporalModel()
    try:
        state = load_file(path, device="cpu")
        if not state or any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
            raise TemporalError("V6 checkpoint tensors must be finite")
        model.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, SafetensorError) as exc:
        raise TemporalError("V6 checkpoint tensor contract mismatch") from exc
    model.eval()
    return _Checkpoint(_file_sha(path), dict(metadata), model)


def _bool_pair(value: object) -> list[bool]:
    if type(value) is not list or len(value) != 2 or any(type(item) is not bool for item in value):
        raise TemporalError("tracking boolean pair is invalid")
    return value


def _float_pair(value: object) -> list[float]:
    if type(value) is not list or len(value) != 2:
        raise TemporalError("tracking numeric pair is invalid")
    return [_number(item, high=1.0) for item in value]


def _centers(value: object) -> list[list[float]]:
    if type(value) is not list or len(value) != 2:
        raise TemporalError("tracking centers are invalid")
    return [_float_pair(item) for item in value]


def _load_tracking_split(path: Path, v5: _V5Binding) -> _TrackingSplitBinding:
    payload = _load_json(path, _TRACKING_SPLIT_FIELDS, TRACKING_SPLIT_SCHEMA)
    row_ids = payload["tracking_row_identities"]
    sessions = payload["session_splits"]
    if (
        payload["v5_manifest_sha256"] != v5.manifest_sha256
        or payload["v5_split_binding_sha256"] != v5.split_binding_sha256
        or type(row_ids) is not list
        or type(sessions) is not dict
        or len(row_ids) != 300
    ):
        raise TemporalError("tracking split manifest must be exact and bounded")
    identities: list[str] = []
    counts: dict[str, int] = {name: 0 for name in ("train", "dev", "test")}
    session_splits: dict[str, str] = {}
    for key, value in sessions.items():
        if type(key) is not str or not _valid_sha(key):
            raise TemporalError("tracking split session must be session hash")
        if type(value) is not str or value not in counts:
            raise TemporalError("tracking split session split must be train/dev/test")
        session_splits[key] = value

    for item in row_ids:
        if type(item) is not str:
            raise TemporalError("tracking split identities must be string")
        session, _, frame_id = item.partition(":")
        if not _valid_sha(session) or not frame_id:
            raise TemporalError("tracking split identities must be session:frame")
        if session_splits.get(session) is None:
            raise TemporalError("tracking split identities must map to declared sessions")
        counts[session_splits[session]] += 1
        identities.append(item)
    if counts != {"train": 180, "dev": 60, "test": 60} or session_splits != v5.session_splits:
        raise TemporalError("tracking split must bind exactly 300 session identities in 180/60/60")
    return _TrackingSplitBinding(_file_sha(path), session_splits, tuple(identities), v5.manifest_sha256, v5.split_binding_sha256)


def _load_tracking(path: Path, checkpoint_sha256: str, split_binding: _TrackingSplitBinding) -> _Evidence:
    payload = _load_json(path, {"schema_version", "checkpoint_sha256", "rows"}, TRACKING_SCHEMA)
    rows = payload["rows"]
    if payload["checkpoint_sha256"] != checkpoint_sha256 or type(rows) is not list or len(rows) != 300:
        raise TemporalError("tracking evidence must bind exactly 300 rows")
    expected = {"frame_id", "session_hash", "split", "predicted_centers", "truth_centers", "predicted_visibility", "truth_visibility", "predicted_hp", "truth_hp", "predicted_skill_ready", "truth_skill_ready"}
    ids: set[str] = set()
    evidence_ids = split_binding.row_identities
    split_counts = {name: 0 for name in ("train", "dev", "test")}
    split_sessions: dict[str, set[str]] = {name: set() for name in split_counts}
    distances: list[float] = []
    pred_vis: list[bool] = []
    true_vis: list[bool] = []
    pred_hp: list[float] = []
    true_hp: list[float] = []
    pred_skill: list[bool] = []
    true_skill: list[bool] = []
    for row in rows:
        split, session = row.get("split"), row.get("session_hash")
        if type(row) is not dict or set(row) != expected or type(row["frame_id"]) is not str or not row["frame_id"] or not _valid_sha(session) or split not in split_counts:
            raise TemporalError("tracking row fields are invalid")
        identity = f"{session}:{row['frame_id']}"
        if identity not in evidence_ids:
            raise TemporalError("tracking row is not bound to split manifest")
        row_split = split_binding.session_splits.get(cast(str, session))
        if row_split is None or row_split != split:
            raise TemporalError("tracking row split does not match bound sessions")
        if identity in ids:
            raise TemporalError("tracking rows are not unique")
        ids.add(identity)
        split_counts[cast(str, split)] += 1
        split_sessions[cast(str, split)].add(cast(str, session))
        pc, tc = _centers(row["predicted_centers"]), _centers(row["truth_centers"])
        pv, tv = _bool_pair(row["predicted_visibility"]), _bool_pair(row["truth_visibility"])
        ph, th = _float_pair(row["predicted_hp"]), _float_pair(row["truth_hp"])
        ps, ts = _bool_pair(row["predicted_skill_ready"]), _bool_pair(row["truth_skill_ready"])
        if split == "test":
            distances.extend(math.dist(a, b) for a, b, visible in zip(pc, tc, tv, strict=True) if visible)
            pred_vis.extend(pv)
            true_vis.extend(tv)
            pred_hp.extend(ph)
            true_hp.extend(th)
            pred_skill.extend(ps)
            true_skill.extend(ts)
    if split_counts != {"train": 180, "dev": 60, "test": 60} or any(split_sessions[a] & split_sessions[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise TemporalError("tracking evidence requires session-isolated 180/60/60 splits")
    if ids != set(evidence_ids):
        raise TemporalError("tracking rows must match bound identities exactly")
    if not distances:
        raise TemporalError("tracking PCK has no visible ground truth")
    metrics = {
        "pck": sum(value <= 0.1 for value in distances) / len(distances),
        "visibility_f1": _binary_f1(pred_vis, true_vis),
        "hp_mae": sum(abs(a - b) for a, b in zip(pred_hp, true_hp, strict=True)) / len(pred_hp),
        "skill_ready_f1": _binary_f1(pred_skill, true_skill),
    }
    if not (metrics["pck"] >= 0.85 and metrics["visibility_f1"] >= 0.90 and metrics["hp_mae"] <= 0.10 and metrics["skill_ready_f1"] >= 0.90):
        raise TemporalError("tracking evidence fails the frozen V6 gate")
    return _Evidence(_file_sha(path), metrics)


def _runtime_advice(action: str, confidence: float, ood: float, quality: float, stable: bool, allowed: tuple[str, ...], thresholds: dict[str, float]) -> str:
    return action if stable and action in allowed and confidence >= thresholds[action] and ood <= OOD_THRESHOLD and quality >= TRACKING_QUALITY_THRESHOLD else ABSTAIN


def _switches(values: list[str]) -> int:
    return sum(left != right for left, right in zip(values, values[1:], strict=False))


def _load_audit(
    path: Path, checkpoint_sha256: str, allowed: tuple[str, ...], thresholds: dict[str, float], test_sessions: set[str]
) -> _Evidence:
    payload = _load_json(path, {"schema_version", "checkpoint_sha256", "rows"}, AUDIT_SCHEMA)
    rows = payload["rows"]
    if payload["checkpoint_sha256"] != checkpoint_sha256 or type(rows) is not list or len(rows) != 200:
        raise TemporalError("temporal audit must bind exactly 200 clips")
    row_fields = {"clip_id", "session_hash", "annotations", "transition", "ood", "reference_event_ms", "baseline_actions", "predictions"}
    annotation_fields = {"reviewer", "observed_action", "validity"}
    pred_fields = {"timestamp_ms", "action", "confidence", "ood_score", "tracking_quality", "stable", "latency_ms"}
    ids: set[str] = set()
    final: list[tuple[str, str, bool, bool]] = []
    delays: list[float] = []
    intervals: list[float] = []
    latencies: list[float] = []
    baseline_switches = advice_switches = 0
    reviewer_pairs: set[tuple[str, str]] = set()
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != row_fields
            or type(row["clip_id"]) is not str
            or not row["clip_id"]
            or not _valid_sha(row["session_hash"])
            or row["session_hash"] not in test_sessions
        ):
            raise TemporalError("temporal audit row fields are invalid")
        identity = f"{row['session_hash']}:{row['clip_id']}"
        if identity in ids or type(row["transition"]) is not bool or type(row["ood"]) is not bool:
            raise TemporalError("temporal audit rows are not unique/typed")
        ids.add(identity)
        annotations = row["annotations"]
        if type(annotations) is not list or len(annotations) != 2:
            raise TemporalError("temporal audit requires two blind annotations per clip")
        reviewers: set[str] = set()
        references: list[str] = []
        for annotation in annotations:
            if type(annotation) is not dict or set(annotation) != annotation_fields or type(annotation["reviewer"]) is not str or not annotation["reviewer"] or annotation["observed_action"] not in ACTION_NAMES or type(annotation["validity"]) is not bool:
                raise TemporalError("temporal audit annotation is invalid")
            reviewers.add(annotation["reviewer"])
            references.append(annotation["observed_action"] if annotation["validity"] else ABSTAIN)
        if len(reviewers) != 2 or references[0] != references[1]:
            raise TemporalError("temporal audit annotations require two-reviewer consensus")
        reviewer_pairs.add(cast(tuple[str, str], tuple(sorted(reviewers))))
        reference = references[0]
        if (row["ood"] and reference != ABSTAIN) or (not row["ood"] and reference not in ACTION_NAMES):
            raise TemporalError("temporal audit reference is invalid")
        baseline, predictions = row["baseline_actions"], row["predictions"]
        if type(baseline) is not list or type(predictions) is not list or len(baseline) != len(predictions) or len(baseline) < 2 or any(value not in (*ACTION_NAMES, ABSTAIN) for value in baseline):
            raise TemporalError("temporal audit sequences are invalid")
        emitted: list[str] = []
        timestamps: list[int] = []
        for prediction in predictions:
            if type(prediction) is not dict or set(prediction) != pred_fields or type(prediction["timestamp_ms"]) is not int or prediction["timestamp_ms"] < 0 or prediction["action"] not in ACTION_NAMES or type(prediction["stable"]) is not bool:
                raise TemporalError("temporal raw prediction is invalid")
            confidence = _number(prediction["confidence"], high=1.0)
            ood_score = _number(prediction["ood_score"], high=1.0)
            quality = _number(prediction["tracking_quality"], high=1.0)
            latencies.append(_number(prediction["latency_ms"]))
            timestamps.append(prediction["timestamp_ms"])
            emitted.append(_runtime_advice(prediction["action"], confidence, ood_score, quality, prediction["stable"], allowed, thresholds))
        if any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise TemporalError("temporal timestamps must strictly increase")
        event = row["reference_event_ms"]
        if type(event) is not int or not timestamps[0] <= event <= timestamps[-1]:
            raise TemporalError("temporal reference event is invalid")
        intervals.extend(
            float(right - left) for left, right in zip(timestamps, timestamps[1:], strict=False)
        )
        baseline_switches += _switches(baseline)
        advice_switches += _switches(emitted)
        final.append((emitted[-1], reference, row["transition"], row["ood"]))
        if emitted[-1] != ABSTAIN:
            delays.append(float(timestamps[-1] - event))
    if len(reviewer_pairs) != 1:
        raise TemporalError("temporal audit must use the same two blinded reviewers")
    accepted = [row for row in final if row[0] != ABSTAIN]
    correct = [row for row in accepted if not row[3] and row[0] == row[1]]
    transitions = [row for row in final if row[2]]
    ood_rows = [row for row in final if row[3]]
    class_precision = {name: sum(row[1] == name and not row[3] for row in accepted if row[0] == name) / max(1, sum(row[0] == name for row in accepted)) for name in allowed}
    metrics = {
        "overall_precision": len(correct) / max(1, len(accepted)),
        "coverage": len(accepted) / 200,
        "transition_false_advice": sum(row[0] != ABSTAIN and (row[3] or row[0] != row[1]) for row in transitions) / max(1, len(transitions)),
        "ood_false_advice": sum(row[0] != ABSTAIN for row in ood_rows) / max(1, len(ood_rows)),
        "switch_reduction": 1.0 - advice_switches / max(1, baseline_switches),
        "median_delay_ms": float(torch.tensor(delays).median().item()) if delays else math.inf,
        "p95_delay_ms": _p95(delays) if delays else math.inf,
        "live_hz": 1000.0 / (sum(intervals) / len(intervals)),
        "live_p95_ms": _p95(latencies),
    }
    eligible = tuple(name for name in allowed if any(row[0] == name for row in accepted) and class_precision[name] >= 0.75)
    passed = bool(eligible) and bool(transitions) and bool(ood_rows) and baseline_switches > 0 and metrics["overall_precision"] >= 0.85 and metrics["coverage"] >= 0.20 and metrics["transition_false_advice"] <= 0.05 and metrics["ood_false_advice"] <= 0.05 and metrics["switch_reduction"] >= 0.50 and metrics["median_delay_ms"] <= 300.0 and metrics["p95_delay_ms"] <= 500.0 and metrics["live_hz"] >= 10.0 and metrics["live_p95_ms"] <= 100.0
    if not passed:
        raise TemporalError("temporal audit fails the frozen V6 gate")
    return _Evidence(_file_sha(path), metrics, eligible)


_RELEASE_FIELDS = {"schema_version", "v5_release_sha256", "v5_model_sha256", "v5_manifest_sha256", "v5_split_binding_sha256", "v5_pre_ingest_sha256", "checkpoint_sha256", "tracking_evidence_sha256", "tracking_split_sha256", "temporal_audit_sha256", "training_artifact_sha256", "training_contract_hash", "action_vocabulary_hash", "architecture", "training_seed", "sequence_length", "claim_scope", "control_output", "overall_pass", "allowed_classes", "class_thresholds", "release_sha256"}
_TRACKING_SPLIT_FIELDS = {"schema_version", "v5_manifest_sha256", "v5_split_binding_sha256", "session_splits", "tracking_row_identities"}


def _release_payload(v5: _V5Binding, checkpoint: _Checkpoint, tracking_split: _TrackingSplitBinding, tracking: _Evidence, audit: _Evidence) -> dict[str, object]:
    return {
        "schema_version": V6_SCHEMA,
        "v5_release_sha256": v5.release_sha256,
        "v5_model_sha256": v5.model_sha256,
        "v5_manifest_sha256": v5.manifest_sha256,
        "v5_split_binding_sha256": v5.split_binding_sha256,
        "v5_pre_ingest_sha256": v5.pre_ingest_sha256,
        "checkpoint_sha256": checkpoint.sha256,
        "tracking_evidence_sha256": tracking.sha256,
        "tracking_split_sha256": tracking_split.sha256,
        "temporal_audit_sha256": audit.sha256,
        "training_artifact_sha256": checkpoint.metadata["training_artifact_sha256"],
        "training_contract_hash": TRAINING_CONTRACT_HASH,
        "action_vocabulary_hash": ACTION_HASH,
        "architecture": ARCHITECTURE,
        "training_seed": checkpoint.metadata["training_seed"],
        "sequence_length": 8,
        "claim_scope": CLAIM_SCOPE,
        "control_output": False,
        "overall_pass": False,
        "allowed_classes": [],
        "class_thresholds": {},
    }


def create_v6_release(*, v5_release_path: Path, v5_model_path: Path, v5_manifest_path: Path, v5_shard_paths: Sequence[Path], v5_pre_ingest_path: Path, v5_privacy_context_path: Path, v5_owner_attestation_path: Path, v5_owner_component_confirmation_path: Path, training_artifact_path: Path, checkpoint_path: Path, tracking_split_path: Path, tracking_evidence_path: Path, temporal_audit_path: Path, release_path: Path) -> dict[str, object]:
    paths = (v5_release_path, v5_model_path, v5_manifest_path, v5_pre_ingest_path, v5_privacy_context_path, v5_owner_attestation_path, v5_owner_component_confirmation_path, training_artifact_path, checkpoint_path, tracking_split_path, tracking_evidence_path, temporal_audit_path, release_path)
    if any(not isinstance(path, Path) for path in paths) or not isinstance(v5_shard_paths, Sequence) or not v5_shard_paths or any(not isinstance(path, Path) for path in v5_shard_paths):
        raise TypeError("V6 release inputs must be pathlib.Path")
    v5 = _load_v5(v5_release_path, v5_model_path, v5_manifest_path, v5_pre_ingest_path, v5_privacy_context_path, v5_owner_attestation_path, v5_owner_component_confirmation_path, v5_shard_paths)
    checkpoint = _load_checkpoint(checkpoint_path, v5, training_artifact_path, tracking_split_path)
    split_binding = _load_tracking_split(tracking_split_path, v5)
    tracking = _load_tracking(tracking_evidence_path, checkpoint.sha256, split_binding)
    first = _load_audit(temporal_audit_path, checkpoint.sha256, v5.allowed_classes, v5.class_thresholds, split_binding.test_sessions)
    thresholds = {name: v5.class_thresholds[name] for name in first.allowed_classes}
    audit = _load_audit(temporal_audit_path, checkpoint.sha256, first.allowed_classes, thresholds, split_binding.test_sessions)
    payload = _release_payload(v5, checkpoint, split_binding, tracking, audit)
    payload["release_sha256"] = _sha(_json(payload).encode())
    try:
        with release_path.open("x", encoding="utf-8") as handle:
            handle.write(_json(payload) + "\n")
    except FileExistsError as exc:
        raise TemporalError("V6 release already exists") from exc
    return payload


def _load_v6_release(
    path: Path, v5: _V5Binding, checkpoint: _Checkpoint, tracking_path: Path, split_path: Path, audit_path: Path
) -> _V6Binding:
    payload = _load_json(path, _RELEASE_FIELDS, V6_SCHEMA)
    signature = payload.pop("release_sha256")
    if not _valid_sha(signature) or _sha(_json(payload).encode()) != signature:
        raise TemporalError("V6 release self-hash mismatch")
    fixed = {
        "v5_release_sha256": v5.release_sha256,
        "v5_model_sha256": v5.model_sha256,
        "v5_manifest_sha256": v5.manifest_sha256,
        "v5_split_binding_sha256": v5.split_binding_sha256,
        "v5_pre_ingest_sha256": v5.pre_ingest_sha256,
        "checkpoint_sha256": checkpoint.sha256,
        "tracking_evidence_sha256": _file_sha(tracking_path),
        "tracking_split_sha256": _file_sha(split_path),
        "temporal_audit_sha256": _file_sha(audit_path),
        "training_artifact_sha256": checkpoint.metadata["training_artifact_sha256"],
        "training_contract_hash": TRAINING_CONTRACT_HASH,
        "action_vocabulary_hash": ACTION_HASH,
        "architecture": ARCHITECTURE,
        "training_seed": checkpoint.metadata["training_seed"],
        "sequence_length": 8,
        "claim_scope": CLAIM_SCOPE,
        "control_output": False,
        "overall_pass": False,
    }
    if any(type(payload.get(key)) is not type(value) or payload.get(key) != value for key, value in fixed.items()):
        raise TemporalError("V6 release artifact binding is invalid")
    allowed, thresholds = payload.get("allowed_classes"), payload.get("class_thresholds")
    if allowed != [] or thresholds != {}:
        raise TemporalError("V6 released classes are invalid")
    return _V6Binding(signature, (), {})


@dataclass
class _Runtime:
    classes: deque[int]
    timestamps: deque[int]
    features: deque[Tensor]
    last_timestamp_ms: int | None = None
    cooldown_until_ms: int = -1


class _RGBEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(),
        )
        self.heatmap_head = nn.Conv2d(32, 2, 1)
        self.hud_head = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 4), nn.Sigmoid())

    def forward(self, rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stem = self.stem(rgb)
        heatmaps = torch.sigmoid(self.heatmap_head(F.adaptive_avg_pool2d(stem, (8, 8))))
        feature = F.adaptive_avg_pool2d(self.body(stem), 1).flatten(1)
        return feature, heatmaps, self.hud_head(feature)


class _CausalDepthwiseTCN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv1d(channels, channels, 3, groups=channels)
        self.second = nn.Conv1d(channels, channels, 3, dilation=2, groups=channels)

    def forward(self, sequence: Tensor) -> Tensor:
        value = sequence.transpose(1, 2)
        value = torch.relu(self.first(F.pad(value, (2, 0))))
        value = torch.relu(self.second(F.pad(value, (4, 0))))
        return value.transpose(1, 2)


class TemporalModel(nn.Module):
    def __init__(
        self,
        *,
        class_threshold: float = 0.75,
        quality_threshold: float = 0.80,
        stable_frames: int = 5,
        frame_interval_ms: int = 100,
        gap_reset_ms: int = 250,
        cooldown_ms: int = 500,
    ) -> None:
        super().__init__()
        self.seq_len = 8
        self.class_threshold = class_threshold
        self.quality_threshold = quality_threshold
        self.stable_frames = stable_frames
        self.frame_interval_ms = frame_interval_ms
        self.gap_reset_ms = gap_reset_ms
        self.cooldown_ms = cooldown_ms
        self.visual_encoder = _RGBEncoder()
        self.visual_projection = nn.Linear(96, 64)
        self.track_projection = nn.Linear(10, 64)
        self.tcn = _CausalDepthwiseTCN(64)
        self.logit_head = nn.Linear(64, len(ACTION_NAMES))
        self.ood_head = nn.Linear(64, 1)
        self._runtime: list[_Runtime] | None = None
        self._track_pos: Tensor | None = None
        self._track_vel: Tensor | None = None
        self._track_conf: Tensor | None = None
        self._last_frame: Tensor | None = None
        self._reset_generation = 0

    def reset(self, reason: str = "new_segment") -> None:
        self._runtime = None
        self._track_pos = None
        self._track_vel = None
        self._track_conf = None
        self._last_frame = None
        self._reset_generation += 1

    def _ensure_runtime(self, batch: int, device: torch.device) -> None:
        self._runtime = [
            _Runtime(
                deque(maxlen=self.stable_frames), deque(maxlen=self.stable_frames), deque(maxlen=8)
            )
            for _ in range(batch)
        ]
        self._track_pos = torch.zeros(batch, 2, 2, device=device)
        self._track_vel = torch.zeros(batch, 2, 2, device=device)
        self._track_conf = torch.zeros(batch, 2, device=device)

    def _reset_sample(self, index: int, reason: str) -> None:
        if (
            self._runtime is None
            or self._track_pos is None
            or self._track_vel is None
            or self._track_conf is None
        ):
            raise RuntimeError("temporal state is not initialized")
        self._runtime[index] = _Runtime(
            deque(maxlen=self.stable_frames), deque(maxlen=self.stable_frames), deque(maxlen=8)
        )
        self._track_pos[index].zero_()
        self._track_vel[index].zero_()
        self._track_conf[index].zero_()
        self._reset_generation += 1

    @staticmethod
    def _observations(heatmaps: Tensor) -> tuple[Tensor, Tensor]:
        flat = heatmaps.flatten(2)
        probs = flat / flat.sum(-1, keepdim=True).clamp_min(1e-6)
        axis = torch.linspace(0.0, 1.0, 8, device=heatmaps.device)
        ys, xs = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack([(probs * xs.flatten()).sum(-1), (probs * ys.flatten()).sum(-1)], dim=-1)
        return xy, heatmaps.mean(dim=(2, 3))

    def _tracking_update(self, observed: Tensor, visibility: Tensor, dt_ms: Tensor) -> None:
        if self._track_pos is None or self._track_vel is None or self._track_conf is None:
            raise RuntimeError("tracking state is not initialized")
        dt = dt_ms.to(self._track_pos.dtype).clamp_min(1.0)[:, None, None] / 1000.0
        predicted = self._track_pos + self._track_vel * dt
        gain = (0.65 * visibility).clamp(0.0, 0.9).unsqueeze(-1)
        innovation = observed - predicted
        self._track_pos = predicted + gain * innovation
        self._track_vel = self._track_vel + 0.35 * gain * innovation / dt
        self._track_conf = (visibility * (1.0 - 0.2 * innovation.abs().mean(-1))).clamp(0.0, 1.0)

    def _timestamps(
        self, value: Tensor | int | float | None, frames: int, device: torch.device
    ) -> Tensor:
        if value is None:
            base = 0
            if self._runtime and self._runtime[0].last_timestamp_ms is not None:
                base = self._runtime[0].last_timestamp_ms + self.frame_interval_ms
            return (
                torch.arange(frames, device=device, dtype=torch.long) * self.frame_interval_ms
                + base
            )
        if isinstance(value, Tensor):
            if value.ndim == 1 and value.numel() == frames:
                return value.to(device=device, dtype=torch.long)
            if value.ndim == 0:
                value = int(value.item())
            else:
                raise ValueError("timestamps_ms must be scalar or length-T")
        start = int(value)
        return (
            torch.arange(frames, device=device, dtype=torch.long) * self.frame_interval_ms + start
        )

    def _window(self, device: torch.device) -> Tensor:
        if self._runtime is None:
            raise RuntimeError("temporal state is not initialized")
        rows: list[Tensor] = []
        for state in self._runtime:
            history = list(state.features)
            padding = [torch.zeros(64, device=device) for _ in range(8 - len(history))]
            rows.append(torch.stack([*padding, *history]))
        return torch.stack(rows)

    def forward(
        self, rgb: Tensor, timestamps_ms: Tensor | int | float | None = None
    ) -> dict[str, Any]:
        if rgb.ndim == 4:
            rgb = rgb[:, None]
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("rgb must be [B,3,H,W] or [B,T,3,H,W]")
        if not rgb.is_floating_point():
            rgb = rgb.float()
        batch, frames = rgb.shape[:2]
        if self._runtime is None or len(self._runtime) != batch:
            self._ensure_runtime(batch, rgb.device)
        assert self._runtime is not None
        timestamps = self._timestamps(timestamps_ms, frames, rgb.device)
        logits_rows: list[Tensor] = []
        ood_rows: list[Tensor] = []
        quality_rows: list[Tensor] = []
        dt_rows: list[Tensor] = []
        advisory = [ABSTAIN] * batch
        reasons = ["STABILITY"] * batch
        last_heatmaps = torch.empty(batch, 2, 8, 8, device=rgb.device)
        last_visibility = torch.empty(batch, 2, device=rgb.device)
        last_hud = torch.empty(batch, 4, device=rgb.device)
        reset_count = 0

        for frame_index in range(frames):
            timestamp = int(timestamps[frame_index].item())
            current = rgb[:, frame_index]
            boundary: list[str | None] = [None] * batch
            deltas = torch.full((batch,), float(self.frame_interval_ms), device=rgb.device)
            for sample, state in enumerate(self._runtime):
                previous = state.last_timestamp_ms
                if previous is not None:
                    delta = timestamp - previous
                    if delta < 0:
                        boundary[sample] = "PTS_BACKWARD"
                    elif delta > self.gap_reset_ms:
                        boundary[sample] = "PTS_GAP"
                    else:
                        deltas[sample] = float(delta)
                previous_frame = (
                    rgb[sample, frame_index - 1]
                    if frame_index
                    else (self._last_frame[sample] if self._last_frame is not None else None)
                )
                if (
                    previous_frame is not None
                    and (current[sample] - previous_frame).abs().mean() > 0.55
                ):
                    boundary[sample] = "SCENE_CUT"
                if boundary[sample] is not None:
                    self._reset_sample(sample, boundary[sample] or "SEGMENT")
                    reset_count += 1

            feature, heatmaps, hud = self.visual_encoder(current)
            observed, visibility = self._observations(heatmaps)
            with torch.no_grad():
                self._tracking_update(observed, visibility, deltas)
            assert (
                self._track_pos is not None
                and self._track_vel is not None
                and self._track_conf is not None
            )
            track = torch.cat(
                (self._track_pos.flatten(1), self._track_vel.flatten(1), self._track_conf), dim=1
            )
            encoded = self.visual_projection(feature) + self.track_projection(track)
            for sample, state in enumerate(self._runtime):
                state.features.append(encoded[sample])
            temporal = self.tcn(self._window(rgb.device))[:, -1]
            logits = self.logit_head(temporal)
            ood = torch.sigmoid(self.ood_head(temporal).squeeze(-1))
            quality = self._track_conf.mean(1)
            classes = logits.argmax(1)
            confidence = torch.softmax(logits, 1).max(1).values

            for sample, state in enumerate(self._runtime):
                cls = int(classes[sample].item())
                state.classes.append(cls)
                state.timestamps.append(timestamp)
                reason = boundary[sample]
                stable = (
                    len(state.classes) >= self.stable_frames
                    and sum(value == cls for value in state.classes) >= self.stable_frames - 1
                    and max(state.timestamps) - min(state.timestamps) >= 300
                )
                if reason is None and confidence[sample] < self.class_threshold:
                    reason = "LOW_SCORE"
                elif reason is None and quality[sample] < self.quality_threshold:
                    reason = "LOW_QUALITY"
                elif reason is None and not stable:
                    reason = "STABILITY"
                elif reason is None and timestamp < state.cooldown_until_ms:
                    reason = "COOLDOWN"
                if reason is None:
                    advisory[sample] = ACTION_NAMES[cls]
                    reasons[sample] = ""
                    state.cooldown_until_ms = timestamp + self.cooldown_ms
                else:
                    advisory[sample] = ABSTAIN
                    reasons[sample] = reason
                state.last_timestamp_ms = timestamp
            self._last_frame = current.detach()
            logits_rows.append(logits)
            ood_rows.append(ood)
            quality_rows.append(quality)
            dt_rows.append(deltas)
            last_heatmaps, last_visibility, last_hud = heatmaps, visibility, hud

        frame_logits = torch.stack(logits_rows, 1)[:, -8:]
        frame_ood = torch.stack(ood_rows, 1)[:, -8:]
        frame_quality = torch.stack(quality_rows, 1)[:, -8:]
        frame_dt = torch.stack(dt_rows, 1)[:, -8:]
        assert self._track_pos is not None
        assert self._track_vel is not None
        assert self._track_conf is not None
        self._track_pos = self._track_pos.detach()
        self._track_vel = self._track_vel.detach()
        self._track_conf = self._track_conf.detach()
        return {
            "logits": frame_logits[:, -1],
            "ood": frame_ood[:, -1],
            "tracking_quality": frame_quality[:, -1],
            "hero_heatmaps": last_heatmaps,
            "hero_visibility": last_visibility,
            "hud": last_hud,
            "frame_logits": frame_logits,
            "frame_ood": frame_ood,
            "frame_tracking_quality": frame_quality,
            "frame_dt_ms": frame_dt,
            "advisory": advisory,
            "abstain_reason": reasons,
            "metrics": {"label_available": False, "label_based_gate_passed": False},
            "reset_generation": self._reset_generation,
            "reset_count": reset_count,
        }


class TemporalCoach:
    def __init__(self, *, v5_release_path: Path | None = None, v5_model_path: Path | None = None, v5_manifest_path: Path | None = None, v5_shard_paths: Sequence[Path] | None = None, v5_pre_ingest_path: Path | None = None, v5_privacy_context_path: Path | None = None, v5_owner_attestation_path: Path | None = None, v5_owner_component_confirmation_path: Path | None = None, training_artifact_path: Path | None = None, checkpoint_path: Path | None = None, tracking_split_path: Path | None = None, tracking_evidence_path: Path | None = None, temporal_audit_path: Path | None = None, temporal_release_path: Path | None = None, device: str = "cpu") -> None:
        self._paths = (
            v5_release_path,
            v5_model_path,
            v5_manifest_path,
            v5_pre_ingest_path,
            v5_privacy_context_path,
            v5_owner_attestation_path,
            v5_owner_component_confirmation_path,
            training_artifact_path,
            checkpoint_path,
            tracking_split_path,
            tracking_evidence_path,
            temporal_audit_path,
            temporal_release_path,
        )
        self._v5_shard_paths = tuple(v5_shard_paths) if v5_shard_paths is not None else None
        if any(path is not None and not isinstance(path, Path) for path in self._paths) or (self._v5_shard_paths is not None and (not self._v5_shard_paths or any(not isinstance(path, Path) for path in self._v5_shard_paths))):
            raise TypeError("TemporalCoach artifacts must be pathlib.Path")
        if type(device) is not str or device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self.device = device
        self.model: TemporalModel | None = None
        self._binding: _V6Binding | None = None
        self._tracking: _Evidence | None = None
        self._audit: _Evidence | None = None
        self._fingerprint: tuple[str, ...] | None = None
        self._reset_generation = 0

    def _clear(self, reason: str) -> None:
        if self.model is not None:
            self.model.reset(reason)
        self.model = None
        self._binding = self._tracking = self._audit = None
        self._fingerprint = None
        self._reset_generation += 1

    def _preflight(self) -> None:
        v5_release, v5_model, manifest_path, pre_ingest_path, privacy_context_path, owner_attestation_path, owner_confirmation_path, training_artifact_path, checkpoint_path, split_path, tracking_path, audit_path, release_path = self._paths
        shard_paths = self._v5_shard_paths
        if v5_release is None or v5_model is None or manifest_path is None or pre_ingest_path is None or privacy_context_path is None or owner_attestation_path is None or owner_confirmation_path is None or training_artifact_path is None or checkpoint_path is None or split_path is None or tracking_path is None or audit_path is None or release_path is None or shard_paths is None:
            raise TemporalError("MISSING_V6_EVIDENCE")
        required_paths = (v5_release, v5_model, manifest_path, pre_ingest_path, privacy_context_path, owner_attestation_path, owner_confirmation_path, training_artifact_path, checkpoint_path, split_path, tracking_path, audit_path, release_path, *shard_paths)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise TemporalError("DEVICE_UNAVAILABLE")
        v5 = _load_v5(v5_release, v5_model, manifest_path, pre_ingest_path, privacy_context_path, owner_attestation_path, owner_confirmation_path, shard_paths)
        checkpoint = _load_checkpoint(checkpoint_path, v5, training_artifact_path, split_path)
        split_binding = _load_tracking_split(split_path, v5)
        binding = _load_v6_release(release_path, v5, checkpoint, tracking_path, split_path, audit_path)
        tracking = _load_tracking(tracking_path, checkpoint.sha256, split_binding)
        audit = _load_audit(audit_path, checkpoint.sha256, v5.allowed_classes, v5.class_thresholds, split_binding.test_sessions)
        if binding.allowed_classes or binding.class_thresholds:
            raise TemporalError("V6 must remain fail-closed while V5 is blocked")
        fingerprint = tuple(_file_sha(path) for path in required_paths)
        if self._fingerprint is not None and fingerprint != self._fingerprint:
            self._clear("ARTIFACT_CHANGED")
            raise TemporalError("ARTIFACT_CHANGED")
        if self.model is not None:
            return
        self.model = checkpoint.model.to(torch.device(self.device)).eval()
        self._binding, self._tracking, self._audit = binding, tracking, audit
        self._fingerprint = fingerprint

    def _failure(self, batch: int, reason: str) -> dict[str, Any]:
        self._clear(reason)
        return {
            "advisory": [ABSTAIN] * batch,
            "abstain_reason": [reason] * batch,
            "control_output": False,
            "metrics": {"release_binding_passed": False, "v6_release_binding_passed": False},
            "reset_generation": self._reset_generation,
        }

    @staticmethod
    def _abstain_output(
        batch: int,
        reason: str,
        *,
        passed: bool,
        binding: _V6Binding | None,
        tracking: _Evidence | None,
        audit: _Evidence | None,
        reset_generation: int,
    ) -> dict[str, Any]:
        return {
            "advisory": [ABSTAIN] * batch,
            "abstain_reason": [reason] * batch,
            "control_output": False,
            "metrics": {
                "release_binding_passed": passed,
                "v6_release_binding_passed": passed,
                "release_sha256": binding.release_sha256 if binding is not None else "",
                "tracking": tracking.metrics if tracking is not None else {},
                "temporal_audit": audit.metrics if audit is not None else {},
            },
            "reset_generation": reset_generation,
        }

    def __call__(
        self, rgb: Tensor, timestamps_ms: Tensor | int | float | None = None
    ) -> dict[str, Any]:
        if rgb.ndim not in {4, 5} or rgb.shape[1 if rgb.ndim == 4 else 2] != 3:
            raise ValueError("rgb must be [B,3,H,W] or [B,T,3,H,W]")
        batch = int(rgb.shape[0])
        try:
            self._preflight()
            assert self.model is not None and self._binding is not None
            assert self._tracking is not None and self._audit is not None
            with torch.inference_mode():
                cast(dict[str, Any], self.model(rgb.to(self.device), timestamps_ms))
        except (TemporalError, alignment.AlignmentError, OSError, RuntimeError, SafetensorError) as exc:
            return self._failure(batch, str(exc))
        return self._abstain_output(
            batch,
            alignment.COLLAPSE_BLOCK,
            passed=True,
            binding=self._binding,
            tracking=self._tracking,
            audit=self._audit,
            reset_generation=self._reset_generation,
        )


def cpu_smoke() -> dict[str, object]:
    output = TemporalCoach()(torch.zeros((1, 8, 3, 64, 64)))
    return {
        "status": "PASSED",
        "disposition": "NON_PROMOTING_FAIL_CLOSED_SMOKE",
        "advisory": output["advisory"],
        "release_binding_passed": output["metrics"]["release_binding_passed"],
        "expected_advisory": ABSTAIN,
    }
