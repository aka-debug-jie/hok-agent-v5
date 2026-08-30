from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, cast

from hok_agent.global_agent import MacroIntent, TargetZone

SCHEMA: Final = "hok-agent-global-scenario-cards-v1"
DEFAULT_CONTRACT: Final = (
    Path(__file__).resolve().parents[2] / "configs/global_agent_scenario_cards_v1.json"
)
EXPECTED_CARDS: Final = (
    ("farm_lane", MacroIntent.FARM_LANE, TargetZone.MID_LANE),
    ("push_structure", MacroIntent.PUSH_STRUCTURE, TargetZone.ENEMY_BASE),
    ("engage", MacroIntent.ENGAGE, TargetZone.HOLD_CURRENT_ZONE),
    ("disengage", MacroIntent.DISENGAGE, TargetZone.OWN_BASE),
    ("recall", MacroIntent.RECALL, TargetZone.OWN_BASE),
)


class GlobalScenarioCardError(ValueError):
    pass


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def verify_scenario_card_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GlobalScenarioCardError("scenario-card contract is unavailable") from exc
    if not isinstance(raw, dict):
        raise GlobalScenarioCardError("scenario-card contract is invalid")
    payload = cast(dict[str, object], raw.copy())
    digest = payload.pop("contract_sha256", None)
    if not isinstance(digest, str) or digest != _sha(payload):
        raise GlobalScenarioCardError("scenario-card contract hash mismatch")
    expected_cards = [
        {
            "card_id": card_id,
            "intent": intent.value,
            "target_zone": zone.value,
            "minimum_stable_seconds": 8,
        }
        for card_id, intent, zone in EXPECTED_CARDS
    ]
    expected = {
        "schema_version": SCHEMA,
        "cards": expected_cards,
        "episodes_per_card": 20,
        "train_episodes_per_card": 16,
        "dev_episodes_per_card": 4,
        "window_frames": 16,
        "sample_hz": 5,
        "human_frame_labels_used": False,
        "runtime_app_state_input": False,
        "device_input_allowed": False,
        "raw_video_persisted": False,
        "source_paths_persisted": False,
    }
    if payload != expected:
        raise GlobalScenarioCardError("scenario-card contract differs from the admitted scope")
    return {
        "status": "PASSED",
        "schema_version": SCHEMA,
        "contract_sha256": digest,
        "card_count": len(EXPECTED_CARDS),
        "episodes_required": len(EXPECTED_CARDS) * 20,
        "train_episodes_required": len(EXPECTED_CARDS) * 16,
        "dev_episodes_required": len(EXPECTED_CARDS) * 4,
        "human_frame_labels_used": False,
        "runtime_app_state_input": False,
        "device_input_allowed": False,
    }
