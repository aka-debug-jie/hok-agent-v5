from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from hok_agent.global_policy import load_global_manifest
from hok_agent.hierarchical_p1v2 import DIRECTIONS, load_architecture

SCHEMA = "hok-agent-hierarchical-p1v2-movement-data-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p1v2-movement-data-report-v1"
NORMALIZE = {
    "north": "north",
    "northeast": "north_east",
    "east": "east",
    "southeast": "south_east",
    "south": "south",
    "southwest": "south_west",
    "west": "west",
    "northwest": "north_west",
}


class P1V2MovementDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MovementDataConfig:
    window_frames: int
    minimum_samples: tuple[int, int]
    minimum_episodes: tuple[int, int]
    maximum_dominant_fraction: float
    required_direction_count: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[MovementDataConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    selection = cast(dict[str, object], payload["selection"])
    labels = cast(dict[str, object], payload["labels"])
    gate = cast(dict[str, object], payload["gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_pixelarena_router_movement_audit"
        or selection.get("splits") != ["train", "dev"]
        or selection.get("required_action_type") != "move"
        or selection.get("skill_aim_as_movement_allowed") is not False
        or selection.get("wait_as_direction_allowed") is not False
        or selection.get("test_allowed") is not False
        or labels.get("vocabulary") != list(DIRECTIONS)
        or labels.get("source") != "pixelarena_global_router_executed_move_v1"
        or labels.get("real_video_semantics_verified") is not False
        or claim.get("macro_branch_training_allowed") is not False
        or claim.get("combat_branch_training_allowed") is not False
        or claim.get("policy_bundle_assembly_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("online_rl_allowed") is not False
        or claim.get("device_input_allowed") is not False
    ):
        raise P1V2MovementDataError("P1v2 Movement data boundary differs")
    config = MovementDataConfig(
        int(cast(int, selection["window_frames"])),
        (
            int(cast(int, gate["minimum_train_samples_per_direction"])),
            int(cast(int, gate["minimum_dev_samples_per_direction"])),
        ),
        (
            int(cast(int, gate["minimum_train_episodes_per_direction"])),
            int(cast(int, gate["minimum_dev_episodes_per_direction"])),
        ),
        float(cast(float, gate["maximum_dominant_direction_fraction"])),
        int(cast(int, gate["required_direction_count"])),
    )
    return config, payload, _sha(_canonical(payload))


def normalize_move(action: dict[str, object]) -> str | None:
    if action.get("action_type") != "move":
        return None
    direction = action.get("direction")
    return NORMALIZE.get(cast(str, direction))


def audit(
    contract_path: Path,
    architecture_path: Path,
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    _schedule, _architecture, architecture_sha = load_architecture(architecture_path)
    manifest = load_global_manifest(dataset_root)
    if (
        architecture_sha != contract["p1v2_architecture_sha256"]
        or manifest.get("manifest_sha256") != contract["global_dataset_manifest_sha256"]
    ):
        raise P1V2MovementDataError("P1v2 Movement evidence binding differs")
    counts = {split: Counter({name: 0 for name in DIRECTIONS}) for split in ("train", "dev")}
    support = {split: Counter({name: 0 for name in DIRECTIONS}) for split in ("train", "dev")}
    action_types: dict[str, Counter[str]] = {
        split: Counter() for split in ("train", "dev")
    }
    rows = {"train": 0, "dev": 0}
    for episode in cast(list[dict[str, object]], manifest["episodes"]):
        split = cast(str, episode["split"])
        present: set[str] = set()
        with np.load(dataset_root / cast(str, episode["shard"]), allow_pickle=False) as shard:
            actions = shard["executed_action"][config.window_frames - 1 :]
        for raw in actions:
            action = cast(dict[str, object], json.loads(str(raw)))
            action_types[split][cast(str, action["action_type"])] += 1
            direction = normalize_move(action)
            if direction is None:
                continue
            rows[split] += 1
            counts[split][direction] += 1
            present.add(direction)
        support[split].update(present)
    dominant = {
        split: max(counts[split].values()) / rows[split] if rows[split] else 1.0
        for split in ("train", "dev")
    }
    observed = {
        split: sum(counts[split][direction] > 0 for direction in DIRECTIONS)
        for split in ("train", "dev")
    }
    checks = {
        "direction_sample_support": all(
            counts[split][direction] >= config.minimum_samples[index]
            for index, split in enumerate(("train", "dev"))
            for direction in DIRECTIONS
        ),
        "direction_episode_support": all(
            support[split][direction] >= config.minimum_episodes[index]
            for index, split in enumerate(("train", "dev"))
            for direction in DIRECTIONS
        ),
        "direction_count": all(
            observed[split] == config.required_direction_count for split in ("train", "dev")
        ),
        "dominant_direction": all(
            dominant[split] <= config.maximum_dominant_fraction for split in ("train", "dev")
        ),
        "split_disjoint": not (
            set(cast(list[int], manifest["train_seeds"]))
            & set(cast(list[int], manifest["dev_seeds"]))
        ),
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "P1V2_MOVEMENT_DATA_PASSED" if passed else "P1V2_MOVEMENT_DATA_FAILED",
        "contract_sha256": contract_sha,
        "p1v2_architecture_sha256": architecture_sha,
        "global_dataset_manifest_sha256": manifest["manifest_sha256"],
        "movement_rows": rows,
        "direction_counts": {split: dict(counts[split]) for split in ("train", "dev")},
        "direction_episode_support": {
            split: dict(support[split]) for split in ("train", "dev")
        },
        "observed_direction_count": observed,
        "dominant_direction_fraction": dominant,
        "excluded_action_type_counts": {
            split: dict(action_types[split]) for split in ("train", "dev")
        },
        "checks": checks,
        "label_source": "pixelarena_global_router_executed_move_v1",
        "skill_aim_used": False,
        "wait_used": False,
        "real_video_semantics_verified": False,
        "human_labels_used": False,
        "test_opened": False,
        "movement_branch_training_allowed": passed,
        "macro_branch_training_allowed": False,
        "combat_branch_training_allowed": False,
        "policy_bundle_assembly_allowed": False,
        "reward_allowed": False,
        "online_rl_allowed": False,
        "device_input_allowed": False,
    }
    report["report_sha256"] = _sha(_canonical(report))
    if output_dir.exists() or output_dir.is_symlink():
        raise P1V2MovementDataError("P1v2 Movement output exists")
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
    parser = argparse.ArgumentParser(description="P1v2 Movement simulator-data audit")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            audit(args.contract, args.architecture, args.dataset_root, args.output_dir),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
