from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import cast

import numpy as np

from hok_agent.hierarchical_e1c_clip import load_clip_report

CONTRACT_SCHEMA = "hok-agent-hierarchical-event-e1c-probe-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1c-probe-report-v1"


class E1cProbeError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_probe_contract(path: Path) -> tuple[dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    baseline = cast(dict[str, object], payload["time_only_baseline"])
    stop = cast(dict[str, object], payload["stop_policy"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != CONTRACT_SCHEMA
        or payload.get("status") != "frozen_time_confound_preflight"
        or baseline.get("feature") != "within_session_materialization_ordinal"
        or baseline.get("labels_visible_to_predictor") is not False
        or stop.get("block_visual_training_on_time_confound") is not True
        or stop.get("do_not_run_temporal_model_after_failure") is not True
        or claim.get("visual_training_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("video_test_allowed") is not False
    ):
        raise E1cProbeError("E1c probe contract boundary differs")
    return payload, _sha(_canonical(payload))


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores: list[float] = []
    for label in (0, 1, 2):
        true_positive = int(((labels == label) & (predictions == label)).sum())
        false_positive = int(((labels != label) & (predictions == label)).sum())
        false_negative = int(((labels == label) & (predictions != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def ordinal_time_only_score(shard: Path) -> dict[str, object]:
    with np.load(shard, allow_pickle=False) as data:
        labels = data["label"].astype(np.int64)
        sessions = data["session_id"]
    predictions = np.empty_like(labels)
    for session in sorted(set(sessions.tolist())):
        indices = np.flatnonzero(sessions == session)
        if len(indices) != 3:
            raise E1cProbeError("E1c probe requires exactly three clips per session")
        predictions[indices] = np.arange(3, dtype=np.int64)
    return {
        "rows": len(labels),
        "accuracy": float((labels == predictions).mean()),
        "macro_f1": _macro_f1(labels, predictions),
        "labels_visible_to_predictor": False,
        "feature": "within_session_materialization_ordinal",
    }


def run_time_confound_preflight(
    contract_path: Path,
    clip_dataset: Path,
    clip_contract_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    contract, contract_sha256 = load_probe_contract(contract_path)
    clip_report = load_clip_report(clip_dataset / "report.json", clip_contract_path)
    if contract["clip_report_sha256"] != clip_report["report_sha256"]:
        raise E1cProbeError("E1c probe clip report binding differs")
    train = ordinal_time_only_score(clip_dataset / "train.npz")
    dev = ordinal_time_only_score(clip_dataset / "dev.npz")
    maximum = float(
        cast(
            float,
            cast(dict[str, object], contract["time_only_baseline"])[
                "maximum_dev_macro_f1_to_allow_visual_training"
            ],
        )
    )
    time_confound = float(cast(float, dev["macro_f1"])) > maximum
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "E1C_PROBE_BLOCKED_TIME_CONFOUND"
            if time_confound
            else "E1C_PROBE_VISUAL_TRAINING_ALLOWED"
        ),
        "contract_sha256": contract_sha256,
        "clip_report_sha256": clip_report["report_sha256"],
        "train_time_only": train,
        "dev_time_only": dev,
        "time_confound_detected": time_confound,
        "visual_training_started": False,
        "visual_training_allowed": not time_confound,
        "semantic_accuracy_verified": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "video_test_opened": False,
        "gpu_used": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload))
    if output_dir.exists() or output_dir.is_symlink():
        raise E1cProbeError("E1c probe output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / "report.json").write_bytes(_canonical(payload) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def load_probe_report(report_path: Path, contract_path: Path) -> dict[str, object]:
    if report_path.is_symlink() or not report_path.is_file():
        raise E1cProbeError("E1c probe report must be a regular file")
    data = report_path.read_bytes()
    payload = cast(dict[str, object], json.loads(data))
    _contract, contract_sha256 = load_probe_contract(contract_path)
    supplied = payload.get("report_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    if (
        payload.get("schema_version") != REPORT_SCHEMA
        or payload.get("contract_sha256") != contract_sha256
        or supplied != _sha(_canonical(unsigned))
        or data != _canonical(payload) + b"\n"
        or payload.get("visual_training_started") is not False
        or payload.get("reward_allowed") is not False
        or payload.get("video_test_opened") is not False
    ):
        raise E1cProbeError("E1c probe report is invalid")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E1c time-confound preflight")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--clip-dataset", type=Path, required=True)
    parser.add_argument("--clip-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_time_confound_preflight(
        args.contract,
        args.clip_dataset,
        args.clip_contract,
        args.output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
