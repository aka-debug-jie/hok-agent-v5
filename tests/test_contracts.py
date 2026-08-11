from __future__ import annotations

import pytest

from hok_agent.contracts import FactorizedAction, PublicObservation
from hok_agent.contracts.types import (
    ActionType,
    ContractValidationError,
    TargetType,
)


def _move_action() -> FactorizedAction:
    return FactorizedAction(
        schema_version=1,
        macro_intent="advance",
        action_type=ActionType.MOVE,
        target_type=TargetType.NONE,
        target_key=None,
        direction=3,
        skill_slot=None,
        item_slot=None,
        auxiliary_parameter=None,
    )


def test_factorized_action_round_trip_is_strict() -> None:
    action = _move_action()
    assert FactorizedAction.from_dict(action.to_dict()) == action
    invalid = action.to_dict()
    invalid["unexpected"] = True
    with pytest.raises(ContractValidationError, match="unexpected keys"):
        FactorizedAction.from_dict(invalid)


def test_public_observation_rejects_actor_denylisted_field() -> None:
    with pytest.raises(ContractValidationError, match="not permitted"):
        PublicObservation(
            schema_version=1,
            tick=0,
            self_state={"reward": 1.0},
            visible_entities=(),
            global_features={"phase": 0.0},
            previous_action=None,
        )
