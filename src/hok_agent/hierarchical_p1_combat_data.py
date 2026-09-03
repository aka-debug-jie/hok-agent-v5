from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from hok_agent.operation_policy import _checked_npz, _shard_rows, _verified_summary

SCHEMA = "hok-agent-hierarchical-p1-combat-data-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p1-combat-data-report-v1"
LABELS = ("none", "basic_attack", "skill1", "skill2", "skill3")


class P1CombatDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CombatDataConfig:
    required_sessions: int
    train_sessions: int
    dev_sessions: int
    sample_period_ms: int
    minimum_samples: tuple[int, int]
    maximum_time_only_macro_f1: float
    minimum_unique_sequences: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[CombatDataConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    selection = cast(dict[str, object], payload["selection"])
    labels = cast(dict[str, object], payload["labels"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_visual_combat_execution_audit"
        or selection.get("split") != [6, 2]
        or selection.get("test_allowed") is not False
        or labels.get("vocabulary") != list(LABELS)
        or labels.get("source") != "executed_cooldown_round_robin_v1"
        or labels.get("meaning") != "executed_button_not_tactical_choice"
        or labels.get("hero_identity_artifact_verified") is not False
        or labels.get("semantic_accuracy_verified") is not False
        or claim.get("macro_head_training_allowed") is not False
        or claim.get("movement_head_training_allowed") is not False
        or claim.get("policy_bundle_assembly_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("device_input_allowed") is not False
    ):
        raise P1CombatDataError("P1 Combat data boundary differs")
    split = cast(list[int], selection["split"])
    config = CombatDataConfig(
        int(cast(int, selection["required_sessions"])),
        split[0],
        split[1],
        int(cast(int, selection["sample_period_ms"])),
        (
            int(cast(int, gate["minimum_train_samples_per_positive_class"])),
            int(cast(int, gate["minimum_dev_samples_per_positive_class"])),
        ),
        float(cast(float, gate["maximum_time_only_macro_f1"])),
        int(cast(int, gate["minimum_unique_positive_sequences"])),
    )
    return config, payload, _sha(_canonical(payload))


def macro_f1(labels: list[int], predictions: list[int]) -> float:
    scores = []
    for label in range(len(LABELS)):
        tp = sum(a == label and b == label for a, b in zip(labels, predictions, strict=True))
        fp = sum(a != label and b == label for a, b in zip(labels, predictions, strict=True))
        fn = sum(a == label and b != label for a, b in zip(labels, predictions, strict=True))
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def _session(path: Path) -> tuple[list[int], list[int], str]:
    summary = _verified_summary(path / "summary.json", "hok-agent-visual-combat-arbiter-v1")
    if summary.get("status") != "PASSED" or summary.get("training_candidate") is not True:
        raise P1CombatDataError("P1 Combat session is not a training candidate")
    timestamps: list[int] = []
    labels: list[int] = []
    for row in _shard_rows(summary, "frame_shards"):
        with _checked_npz(path, row) as shard:
            required = {"scheduled_elapsed_ms", "action_id", "input_sent"}
            if not required.issubset(shard.files):
                raise P1CombatDataError("P1 Combat shard fields differ")
            timestamps.extend(map(int, shard["scheduled_elapsed_ms"]))
            actions = np.where(shard["input_sent"].astype(bool), shard["action_id"], 0)
            labels.extend(map(int, actions))
    if len(timestamps) != len(labels) or not timestamps:
        raise P1CombatDataError("P1 Combat rows differ")
    positive = np.asarray([label for label in labels if label > 0], dtype=np.int8)
    return timestamps, labels, _sha(positive.tobytes())


def audit(contract_path: Path, dataset_root: Path, output_dir: Path) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    prefix = cast(str, cast(dict[str, object], contract["selection"])["session_prefix"])
    paths = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )
    if len(paths) != config.required_sessions:
        raise P1CombatDataError("P1 Combat session count differs")
    split_paths = {
        "train": paths[: config.train_sessions],
        "dev": paths[config.train_sessions :],
    }
    rows: dict[str, list[tuple[int, int]]] = {"train": [], "dev": []}
    counts = {split: Counter({index: 0 for index in range(5)}) for split in rows}
    sequence_hashes: set[str] = set()
    sessions: list[dict[str, object]] = []
    for split, selected in split_paths.items():
        for path in selected:
            timestamps, labels, sequence_sha = _session(path)
            rows[split].extend(zip(timestamps, labels, strict=True))
            counts[split].update(labels)
            sequence_hashes.add(sequence_sha)
            sessions.append(
                {
                    "session_id": _sha(f"p1-combat:{contract_sha}:{path.name}".encode()),
                    "split": split,
                    "rows": len(labels),
                    "class_counts": {LABELS[index]: labels.count(index) for index in range(5)},
                    "positive_sequence_sha256": sequence_sha,
                }
            )
    buckets: dict[int, Counter[int]] = defaultdict(Counter)
    for timestamp, label in rows["train"]:
        buckets[round(timestamp / config.sample_period_ms)][label] += 1
    dev_labels = [label for _timestamp, label in rows["dev"]]
    time_predictions = [
        buckets[round(timestamp / config.sample_period_ms)].most_common(1)[0][0]
        if buckets[round(timestamp / config.sample_period_ms)]
        else 0
        for timestamp, _label in rows["dev"]
    ]
    time_f1 = macro_f1(dev_labels, time_predictions)
    labels_contract = cast(dict[str, object], contract["labels"])
    train_ids = {row["session_id"] for row in sessions if row["split"] == "train"}
    dev_ids = {row["session_id"] for row in sessions if row["split"] == "dev"}
    checks = {
        "positive_class_support": all(
            counts[split][label] >= config.minimum_samples[index]
            for index, split in enumerate(("train", "dev"))
            for label in range(1, 5)
        ),
        "time_only_control": time_f1 <= config.maximum_time_only_macro_f1,
        "positive_sequence_diversity": len(sequence_hashes) >= config.minimum_unique_sequences,
        "hero_identity_artifact": labels_contract["hero_identity_artifact_verified"] is True,
        "split_disjoint": not (train_ids & dev_ids),
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "P1_COMBAT_DATA_PASSED" if passed else "P1_COMBAT_DATA_FAILED",
        "contract_sha256": contract_sha,
        "p0_representation_sha256": contract["p0_representation_sha256"],
        "session_counts": {split: len(paths) for split, paths in split_paths.items()},
        "row_counts": {split: len(values) for split, values in rows.items()},
        "class_counts": {
            split: {LABELS[index]: counts[split][index] for index in range(5)}
            for split in ("train", "dev")
        },
        "time_only_macro_f1": time_f1,
        "unique_positive_sequences": len(sequence_hashes),
        "sessions": sessions,
        "checks": checks,
        "label_source": "executed_cooldown_round_robin_v1",
        "tactical_choice_verified": False,
        "hero_identity_artifact_verified": False,
        "human_labels_used": False,
        "semantic_accuracy_verified": False,
        "test_opened": False,
        "combat_head_training_allowed": passed,
        "macro_head_training_allowed": False,
        "movement_head_training_allowed": False,
        "policy_bundle_assembly_allowed": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "device_input_allowed": False,
    }
    report["report_sha256"] = _sha(_canonical(report))
    if output_dir.exists() or output_dir.is_symlink():
        raise P1CombatDataError("P1 Combat output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / "report.json").write_bytes(_canonical(report) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 Combat execution-data audit")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.contract, args.dataset_root, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
