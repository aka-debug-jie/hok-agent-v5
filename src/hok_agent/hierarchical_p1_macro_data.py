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

from hok_agent.global_policy import load_global_manifest
from hok_agent.hierarchical_p0_ssl_v2 import verify_representation

SCHEMA = "hok-agent-hierarchical-p1-macro-data-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p1-macro-data-report-v1"
ACTIVE_INTENTS = ("FARM_LANE", "PUSH_STRUCTURE", "ENGAGE")


class P1MacroDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MacroDataConfig:
    window_frames: int
    minimum_windows: tuple[int, int]
    minimum_episodes: tuple[int, int]
    maximum_class_prior_macro_f1: float
    maximum_time_only_macro_f1: float


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[MacroDataConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    selection = cast(dict[str, object], payload["selection"])
    labels = cast(dict[str, object], payload["labels"])
    model = cast(dict[str, object], payload["model_input"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_simulator_macro_data_audit"
        or selection.get("splits") != ["train", "dev"]
        or selection.get("active_intents") != list(ACTIVE_INTENTS)
        or selection.get("router_owned_intents") != ["DISENGAGE", "RECALL"]
        or selection.get("test_allowed") is not False
        or labels.get("source") != "pixelarena_structured_rule_teacher_v1"
        or labels.get("simulator_semantics_verified") is not True
        or labels.get("real_video_semantics_verified") is not False
        or model.get("fields") != ["main_rgb", "minimap_rgb", "hud_rgb"]
        or any(
            model.get(field) is not False
            for field in ("structured_state_allowed", "tick_allowed", "label_allowed")
        )
        or claim.get("movement_head_training_allowed") is not False
        or claim.get("combat_head_training_allowed") is not False
        or claim.get("policy_bundle_assembly_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("device_input_allowed") is not False
    ):
        raise P1MacroDataError("P1 Macro data boundary differs")
    config = MacroDataConfig(
        int(cast(int, selection["window_frames"])),
        (
            int(cast(int, gate["minimum_train_windows_per_intent"])),
            int(cast(int, gate["minimum_dev_windows_per_intent"])),
        ),
        (
            int(cast(int, gate["minimum_train_episodes_per_intent"])),
            int(cast(int, gate["minimum_dev_episodes_per_intent"])),
        ),
        float(cast(float, gate["maximum_class_prior_macro_f1"])),
        float(cast(float, gate["maximum_time_only_macro_f1"])),
    )
    return config, payload, _sha(_canonical(payload))


def macro_f1(labels: list[int], predictions: list[int]) -> float:
    scores = []
    for label in range(len(ACTIVE_INTENTS)):
        tp = sum(a == label and b == label for a, b in zip(labels, predictions, strict=True))
        fp = sum(a != label and b == label for a, b in zip(labels, predictions, strict=True))
        fn = sum(a == label and b != label for a, b in zip(labels, predictions, strict=True))
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def _controls(train: list[tuple[int, int]], dev: list[tuple[int, int]]) -> dict[str, float]:
    prior = Counter(label for _tick, label in train).most_common(1)[0][0]
    class_prior = macro_f1(
        [label for _tick, label in dev], [prior] * len(dev)
    )
    buckets: dict[int, Counter[int]] = defaultdict(Counter)
    for tick, label in train:
        buckets[tick // 8][label] += 1
    predictions = [
        buckets[tick // 8].most_common(1)[0][0] if buckets[tick // 8] else prior
        for tick, _label in dev
    ]
    return {
        "class_prior_macro_f1": class_prior,
        "time_only_macro_f1": macro_f1([label for _tick, label in dev], predictions),
    }


def audit(
    contract_path: Path,
    dataset_root: Path,
    p0_run_dir: Path,
    p0_contract_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    manifest = load_global_manifest(dataset_root)
    p0_report = verify_representation(p0_run_dir, p0_contract_path)
    if (
        manifest.get("manifest_sha256") != contract["global_dataset_manifest_sha256"]
        or p0_report.get("representation_sha256") != contract["p0_representation_sha256"]
    ):
        raise P1MacroDataError("P1 Macro evidence binding differs")
    windows: dict[str, list[tuple[int, int]]] = {
        split: [] for split in ("train", "dev")
    }
    counts = {split: Counter({index: 0 for index in range(3)}) for split in ("train", "dev")}
    support = {split: Counter({index: 0 for index in range(3)}) for split in ("train", "dev")}
    for row in cast(list[dict[str, object]], manifest["episodes"]):
        split = cast(str, row["split"])
        with np.load(dataset_root / cast(str, row["shard"]), allow_pickle=False) as shard:
            intent = shard["intent"].astype(np.int64)
            ticks = shard["tick"].astype(np.int64)
        present: set[int] = set()
        for end in range(config.window_frames - 1, len(intent)):
            label = int(intent[end])
            if label >= len(ACTIVE_INTENTS):
                continue
            windows[split].append((int(ticks[end]), label))
            counts[split][label] += 1
            present.add(label)
        support[split].update(present)
    controls = _controls(windows["train"], windows["dev"])
    checks = {
        "window_support": all(
            counts[split][label] >= config.minimum_windows[index]
            for index, split in enumerate(("train", "dev"))
            for label in range(3)
        ),
        "episode_support": all(
            support[split][label] >= config.minimum_episodes[index]
            for index, split in enumerate(("train", "dev"))
            for label in range(3)
        ),
        "class_prior_control": controls["class_prior_macro_f1"]
        <= config.maximum_class_prior_macro_f1,
        "time_only_control": controls["time_only_macro_f1"]
        <= config.maximum_time_only_macro_f1,
        "split_disjoint": not (
            set(cast(list[int], manifest["train_seeds"]))
            & set(cast(list[int], manifest["dev_seeds"]))
        ),
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "P1_MACRO_DATA_PASSED" if passed else "P1_MACRO_DATA_FAILED",
        "contract_sha256": contract_sha,
        "global_dataset_manifest_sha256": manifest["manifest_sha256"],
        "p0_representation_sha256": p0_report["representation_sha256"],
        "window_counts": {
            split: {ACTIVE_INTENTS[label]: counts[split][label] for label in range(3)}
            for split in ("train", "dev")
        },
        "episode_support": {
            split: {ACTIVE_INTENTS[label]: support[split][label] for label in range(3)}
            for split in ("train", "dev")
        },
        "controls": controls,
        "checks": checks,
        "label_source": "pixelarena_structured_rule_teacher_v1",
        "simulator_semantics_verified": True,
        "real_video_semantics_verified": False,
        "human_labels_used": False,
        "test_opened": False,
        "macro_head_training_allowed": passed,
        "movement_head_training_allowed": False,
        "combat_head_training_allowed": False,
        "policy_bundle_assembly_allowed": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "device_input_allowed": False,
    }
    report["report_sha256"] = _sha(_canonical(report))
    if output_dir.exists() or output_dir.is_symlink():
        raise P1MacroDataError("P1 Macro output exists")
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
    parser = argparse.ArgumentParser(description="P1 Macro simulator-data audit")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--p0-run-dir", type=Path, required=True)
    parser.add_argument("--p0-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            audit(
                args.contract,
                args.dataset_root,
                args.p0_run_dir,
                args.p0_contract,
                args.output_dir,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
