from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

SCHEMA = "hok-agent-hierarchical-p1v2-architecture-v1"
DIRECTIONS = (
    "north",
    "north_east",
    "east",
    "south_east",
    "south",
    "south_west",
    "west",
    "north_west",
)


class P1V2Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BranchSchedule:
    macro_hz: int
    movement_hz: int
    combat_hz: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_architecture(path: Path) -> tuple[BranchSchedule, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text()))
    trunk = cast(dict[str, object], payload["shared_trunk"])
    branches = cast(dict[str, dict[str, object]], payload["branches"])
    macro, movement, combat = branches["macro"], branches["movement"], branches["combat"]
    router = cast(dict[str, object], payload["router"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "frozen_task_specific_adapter_decision"
        or trunk.get("frozen_during_p1") is not True
        or trunk.get("single_framebus_version_required") is not True
        or macro.get("trainable") is not True
        or macro.get("training_independent") is not True
        or movement.get("trainable") is not True
        or movement.get("training_independent") is not True
        or movement.get("outputs") != list(DIRECTIONS)
        or movement.get("persistent_joystick_execution") is not True
        or combat.get("trainable") is not False
        or combat.get("implementation") != "deterministic_visual_cooldown_arbiter_v1"
        or router.get("learned") is not False
        or router.get("proposal_only_models") is not True
        or router.get("movement_and_combat_may_execute_concurrently") is not True
        or router.get("pointer_ownership_remains_deterministic") is not True
        or router.get("hard_stop_and_death_override_all") is not True
        or claim.get("old_p1_lineages_reopened") is not False
        or claim.get("architecture_decision_only") is not True
        or claim.get("branch_training_allowed") is not False
        or claim.get("policy_bundle_assembly_allowed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("online_rl_allowed") is not False
        or claim.get("device_input_allowed") is not False
    ):
        raise P1V2Error("P1v2 architecture boundary differs")
    schedule = BranchSchedule(
        int(cast(int, macro["frequency_hz"])),
        int(cast(int, movement["frequency_hz"])),
        int(cast(int, combat["frequency_hz"])),
    )
    if schedule != BranchSchedule(2, 10, 10):
        raise P1V2Error("P1v2 branch schedule differs")
    return schedule, payload, hashlib.sha256(_canonical(payload)).hexdigest()
