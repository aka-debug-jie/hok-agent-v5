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
from typing import Literal, cast

import numpy as np

from hok_agent.hierarchical_p1v2 import DIRECTIONS, load_architecture
from hok_agent.rich_arena import RichPixelArena
from hok_agent.rich_renderer import RENDERER_HASH, render

SCHEMA = "hok-agent-hierarchical-p1v2-movement-2d-source-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-p1v2-movement-2d-source-report-v1"
Side = Literal["blue", "red"]
VECTORS = {
    "north": (0, -1),
    "north_east": (1, -1),
    "east": (1, 0),
    "south_east": (1, 1),
    "south": (0, 1),
    "south_west": (-1, 1),
    "west": (-1, 0),
    "north_west": (-1, -1),
}


class Movement2DSourceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceConfig:
    sequence_frames: int
    train_groups_per_direction: int
    dev_groups_per_direction: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> tuple[SourceConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    curriculum = cast(dict[str, object], payload["curriculum"])
    model = cast(dict[str, object], payload["model_input"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_balanced_visible_target_curriculum"
        or curriculum.get("sides") != ["blue", "red"]
        or curriculum.get("directions") != list(DIRECTIONS)
        or curriculum.get("target") != "visible_enemy_hero_one_step_relative"
        or curriculum.get("render_variation_only_within_sequence") is not True
        or curriculum.get("test_allowed") is not False
        or model.get("fields") != ["rgb_sequence"]
        or any(
            model.get(field) is not False
            for field in (
                "label_allowed",
                "group_id_allowed",
                "side_allowed",
                "structured_state_allowed",
            )
        )
        or claim.get("capability") != "local_visible_target_approach_only"
        or claim.get("lane_strategy_verified") is not False
        or claim.get("real_video_semantics_verified") is not False
        or claim.get("policy_bundle_assembly_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("online_rl_allowed") is not False
        or claim.get("device_input_allowed") is not False
    ):
        raise Movement2DSourceError("P1v2 Movement 2D source boundary differs")
    config = SourceConfig(
        int(cast(int, curriculum["sequence_frames"])),
        int(cast(int, curriculum["train_groups_per_direction"])),
        int(cast(int, curriculum["dev_groups_per_direction"])),
    )
    return config, payload, _sha(_canonical(payload))


def world_delta(direction: str, side: Side) -> tuple[int, int]:
    dx, dy = VECTORS[direction]
    return (dx, dy) if side == "blue" else (-dx, -dy)


def _sequence(direction: str, side: Side, group_seed: int, frames: int) -> np.ndarray:
    arena = RichPixelArena()
    arena.reset(group_seed)
    own = arena.state.blue if side == "blue" else arena.state.red
    other = arena.state.red if side == "blue" else arena.state.blue
    own.x, own.y = 7, 3
    dx, dy = world_delta(direction, side)
    other.x, other.y = own.x + dx, own.y + dy
    own.health = 6 + group_seed % 5
    other.health = 5 + (group_seed // 3) % 6
    output = []
    for index in range(frames):
        arena.state.tick = (group_seed + index) % arena.config.max_ticks
        own.cooldowns["skill1"] = (group_seed + index) % 5
        own.cooldowns["skill2"] = (group_seed // 2 + index) % 4
        own.cooldowns["skill3"] = (group_seed // 3 + index) % 7
        output.append(render(arena.observe(side), group_seed * 101 + index))
    return np.stack(output)


def materialize(
    contract_path: Path,
    architecture_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha = load_contract(contract_path)
    _schedule, _architecture, architecture_sha = load_architecture(architecture_path)
    if architecture_sha != contract["p1v2_architecture_sha256"]:
        raise Movement2DSourceError("P1v2 Movement architecture binding differs")
    if output_dir.exists() or output_dir.is_symlink():
        raise Movement2DSourceError("P1v2 Movement 2D output exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        shard_hashes = {}
        counts: dict[str, Counter[str]] = {}
        groups: dict[str, set[str]] = {}
        for split, per_direction, base in (
            ("train", config.train_groups_per_direction, 10_000),
            ("dev", config.dev_groups_per_direction, 20_000),
        ):
            sequences, labels, identities = [], [], []
            counts[split] = Counter()
            groups[split] = set()
            for direction_index, direction in enumerate(DIRECTIONS):
                for ordinal in range(per_direction):
                    seed = base + direction_index * per_direction + ordinal
                    side: Side = "blue" if ordinal % 2 == 0 else "red"
                    identity = _sha(f"p1v2-movement-2d:{contract_sha}:{split}:{seed}".encode())
                    sequences.append(_sequence(direction, side, seed, config.sequence_frames))
                    labels.append(direction_index)
                    identities.append(identity)
                    counts[split][direction] += 1
                    groups[split].add(identity)
            path = staging / f"{split}.npz"
            np.savez_compressed(
                path,
                rgb_sequence=np.stack(sequences),
                label=np.asarray(labels, dtype=np.int8),
                group_id=np.asarray(identities, dtype="<U64"),
            )
            shard_hashes[split] = _sha(path.read_bytes())
        gate = cast(dict[str, object], contract["gate"])
        imbalance = {
            split: max(counts[split].values()) - min(counts[split].values())
            for split in ("train", "dev")
        }
        overlap = len(groups["train"] & groups["dev"])
        checks = {
            "train_sequences": sum(counts["train"].values())
            == gate["required_train_sequences"],
            "dev_sequences": sum(counts["dev"].values()) == gate["required_dev_sequences"],
            "train_direction_groups": min(counts["train"].values())
            == gate["required_train_groups_per_direction"],
            "dev_direction_groups": min(counts["dev"].values())
            == gate["required_dev_groups_per_direction"],
            "direction_balance": max(imbalance.values())
            <= cast(int, gate["maximum_direction_imbalance"]),
            "group_disjoint": overlap <= cast(int, gate["maximum_group_overlap"]),
        }
        passed = all(checks.values())
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "status": (
                "P1V2_MOVEMENT_2D_SOURCE_PASSED"
                if passed
                else "P1V2_MOVEMENT_2D_SOURCE_FAILED"
            ),
            "contract_sha256": contract_sha,
            "p1v2_architecture_sha256": architecture_sha,
            "previous_movement_audit_report_sha256": contract[
                "previous_movement_audit_report_sha256"
            ],
            "renderer_sha256": RENDERER_HASH,
            "sequence_counts": {split: sum(value.values()) for split, value in counts.items()},
            "direction_counts": {split: dict(value) for split, value in counts.items()},
            "direction_imbalance": imbalance,
            "group_overlap": overlap,
            "shard_sha256": shard_hashes,
            "checks": checks,
            "model_input_fields": ["rgb_sequence"],
            "capability": "local_visible_target_approach_only",
            "lane_strategy_verified": False,
            "real_video_semantics_verified": False,
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
        (staging / "report.json").write_bytes(_canonical(report) + b"\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1v2 balanced visible-target Movement source")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize(args.contract, args.architecture, args.output_dir), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
