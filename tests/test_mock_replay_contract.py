from __future__ import annotations

from copy import deepcopy

import pytest

from hok_agent.artifacts.hashing import sha256_json
from hok_agent.contracts import (
    FactorizedAction,
    MockPublicReplay,
    MockReplayValidationError,
)
from hok_agent.contracts.mock_replay import parse_mock_public_transition_replay
from hok_agent.contracts.types import ActionType, TargetType


def _action() -> dict[str, object]:
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
    ).to_dict()


def _outcome(result: str = "win") -> dict[str, str]:
    return {"result": result}


def _transition(
    *,
    sequence: int,
    previous_row_hash: str | None,
    input_tick: int,
    terminal: bool = False,
    truncated: bool = False,
    outcome: dict[str, str] | None = None,
    seed: int = 0,
) -> dict[str, object]:
    row = {
        "sequence": sequence,
        "previous_row_hash": previous_row_hash,
        "input_tick": input_tick,
        "pre_public_observation_hash": sha256_json({"obs": f"pre-{seed}-{sequence}"}),
        "action": _action(),
        "output_tick": input_tick + 1,
        "post_public_observation_hash": sha256_json({"obs": f"post-{seed}-{sequence}"}),
        "terminal": terminal,
        "truncated": truncated,
        "outcome": outcome,
    }
    row["row_hash"] = sha256_json({key: value for key, value in row.items() if key != "row_hash"})
    return row


def _artifact(seed: int = 42, transitions_count: int = 1, *, terminal: bool = True) -> dict[str, object]:
    transition0 = _transition(
        sequence=0,
        previous_row_hash=None,
        input_tick=0,
        terminal=False,
        truncated=False,
        outcome=None,
        seed=seed,
    )
    transition1 = None
    transitions: list[dict[str, object]] = [transition0]
    if transitions_count >= 2:
        transition1 = _transition(
            sequence=1,
            previous_row_hash=transition0["row_hash"],
            input_tick=1,
            terminal=terminal,
            truncated=not terminal,
            outcome=_outcome("win" if terminal else "draw"),
            seed=seed,
        )
        transitions.append(transition1)

    if transition1 is None and terminal:
        transition0["terminal"] = True
        transition0["outcome"] = _outcome("loss")
        transition0["row_hash"] = sha256_json({key: value for key, value in transition0.items() if key != "row_hash"})

    artifact = {
        "schema_version": 1,
        "artifact_kind": "mock_public_transition_replay",
        "environment_kind": "mock",
        "formal": False,
        "capability_claim": "none",
        "service_identity_hash": sha256_json({"service": seed}),
        "protocol_identity_hash": sha256_json({"protocol": seed}),
        "mapping_identity_hash": sha256_json({"mapping": seed}),
        "mock_profile": {
            "version": "mock-public-profile-v1",
            "max_steps_per_episode": 12,
        },
        "seed": seed,
        "side": "blue",
        "mode": "1v1",
        "episode_id": f"mock-{seed}-blue-1",
        "transition_count": len(transitions),
        "transitions": transitions,
        "chain_head": transitions[-1]["row_hash"],
        "artifact_hash": "",
    }
    artifact["artifact_hash"] = sha256_json({key: value for key, value in artifact.items() if key != "artifact_hash"})
    return artifact


def test_mock_public_replay_roundtrip_is_strict() -> None:
    artifact = _artifact(seed=7, transitions_count=2, terminal=True)
    parsed = MockPublicReplay.from_dict(artifact)
    assert parsed.to_dict() == artifact


def test_mock_public_replay_verifies_chain_and_artifact_hash() -> None:
    artifact = _artifact(seed=11, transitions_count=1, terminal=True)
    parsed = parse_mock_public_transition_replay(artifact)
    assert parsed.artifact_hash == artifact["artifact_hash"]

    bad_row = deepcopy(artifact)
    bad_row["transitions"][0]["row_hash"] = "sha256:" + "0" * 64
    with pytest.raises(MockReplayValidationError, match="row_hash"):
        MockPublicReplay.from_dict(bad_row)

    bad_chain = deepcopy(artifact)
    bad_chain["chain_head"] = "sha256:" + "1" * 64
    with pytest.raises(MockReplayValidationError, match="chain_head"):
        MockPublicReplay.from_dict(bad_chain)


def test_output_tick_must_be_input_tick_plus_one() -> None:
    artifact = _artifact(seed=13, transitions_count=1, terminal=True)
    artifact["transitions"][0]["output_tick"] = artifact["transitions"][0]["input_tick"] + 2
    artifact["transitions"][0]["row_hash"] = sha256_json(
        {key: value for key, value in artifact["transitions"][0].items() if key != "row_hash"}
    )
    artifact["artifact_hash"] = sha256_json(
        {key: value for key, value in artifact.items() if key != "artifact_hash"}
    )

    with pytest.raises(MockReplayValidationError, match="output_tick must equal input_tick \\+ 1"):
        MockPublicReplay.from_dict(artifact)


def test_partial_episode_without_terminal_or_truncated_is_rejected() -> None:
    artifact = _artifact(seed=17, transitions_count=2, terminal=True)
    artifact["transitions"][-1]["terminal"] = False
    artifact["transitions"][-1]["truncated"] = False
    artifact["transitions"][-1]["outcome"] = None
    artifact["transitions"][-1]["row_hash"] = sha256_json(
        {key: value for key, value in artifact["transitions"][-1].items() if key != "row_hash"}
    )
    artifact["transition_count"] = len(artifact["transitions"])
    artifact["chain_head"] = artifact["transitions"][-1]["row_hash"]
    artifact["artifact_hash"] = sha256_json(
        {key: value for key, value in artifact.items() if key != "artifact_hash"}
    )

    with pytest.raises(MockReplayValidationError, match="last transition must be terminal or truncated"):
        MockPublicReplay.from_dict(artifact)


def test_transition_after_terminal_is_rejected() -> None:
    artifact = _artifact(seed=18, transitions_count=2, terminal=True)
    artifact["transitions"][0]["terminal"] = True
    artifact["transitions"][0]["outcome"] = _outcome("win")
    artifact["transitions"][0]["row_hash"] = sha256_json(
        {key: value for key, value in artifact["transitions"][0].items() if key != "row_hash"}
    )
    artifact["transitions"][1]["previous_row_hash"] = artifact["transitions"][0]["row_hash"]
    artifact["transitions"][1]["row_hash"] = sha256_json(
        {key: value for key, value in artifact["transitions"][1].items() if key != "row_hash"}
    )
    artifact["chain_head"] = artifact["transitions"][1]["row_hash"]
    artifact["artifact_hash"] = sha256_json({key: value for key, value in artifact.items() if key != "artifact_hash"})

    with pytest.raises(MockReplayValidationError, match="only the last transition"):
        MockPublicReplay.from_dict(artifact)


def test_partial_chain_head_is_last_transition_hash() -> None:
    artifact = _artifact(seed=19, transitions_count=2, terminal=True)
    first = artifact["transitions"][0]["row_hash"]
    with pytest.raises(MockReplayValidationError, match="chain_head must equal last transition row_hash"):
        MockPublicReplay.from_dict(
            {**artifact, "chain_head": first}
        )


def test_fixed_mock_profile_identity_is_enforced() -> None:
    artifact = _artifact(seed=29, transitions_count=1, terminal=True)
    artifact["side"] = "green"
    artifact["episode_id"] = "mock-29-green-1"
    artifact["artifact_hash"] = sha256_json({key: value for key, value in artifact.items() if key != "artifact_hash"})
    with pytest.raises(MockReplayValidationError, match="side must be blue or red"):
        MockPublicReplay.from_dict(artifact)

    profile = _artifact(seed=29, transitions_count=1, terminal=True)
    profile["mock_profile"]["max_steps_per_episode"] = 1
    profile["artifact_hash"] = sha256_json({key: value for key, value in profile.items() if key != "artifact_hash"})
    with pytest.raises(MockReplayValidationError, match="max_steps_per_episode must be >= 2"):
        MockPublicReplay.from_dict(profile)


def test_terminal_transition_requires_public_outcome() -> None:
    artifact = _artifact(seed=23, transitions_count=1)
    artifact["transitions"][0]["terminal"] = True
    artifact["transitions"][0]["outcome"] = None
    artifact["transition_count"] = 1
    artifact["transitions"][0]["row_hash"] = sha256_json(
        {key: value for key, value in artifact["transitions"][0].items() if key != "row_hash"}
    )
    artifact["artifact_hash"] = sha256_json({key: value for key, value in artifact.items() if key != "artifact_hash"})

    with pytest.raises(MockReplayValidationError, match="requires outcome"):
        MockPublicReplay.from_dict(artifact)


def test_forbidden_and_unknown_fields_are_rejected() -> None:
    artifact = _artifact(seed=31, transitions_count=0)
    artifact_with_unknown = deepcopy(artifact)
    artifact_with_unknown["forbidden_root"] = "x"
    with pytest.raises(MockReplayValidationError, match="unexpected keys"):
        MockPublicReplay.from_dict(artifact_with_unknown)

    artifact_with_nested_forbidden = deepcopy(artifact)
    artifact_with_nested_forbidden["transitions"][0]["outcome"] = {"result": "win", "reward": 1}
    artifact_with_nested_forbidden["transitions"][0]["row_hash"] = sha256_json(
        {key: value for key, value in artifact_with_nested_forbidden["transitions"][0].items() if key != "row_hash"}
    )
    artifact_with_nested_forbidden["artifact_hash"] = sha256_json(
        {key: value for key, value in artifact_with_nested_forbidden.items() if key != "artifact_hash"}
    )
    with pytest.raises(MockReplayValidationError, match="not a public-only key"):
        MockPublicReplay.from_dict(artifact_with_nested_forbidden)

    artifact_with_prefix_variant = deepcopy(artifact)
    artifact_with_prefix_variant["transitions"][0]["reward_components"] = {"total": 1}
    artifact_with_prefix_variant["transitions"][0]["row_hash"] = sha256_json(
        {key: value for key, value in artifact_with_prefix_variant["transitions"][0].items() if key != "row_hash"}
    )
    artifact_with_prefix_variant["artifact_hash"] = sha256_json(
        {key: value for key, value in artifact_with_prefix_variant.items() if key != "artifact_hash"}
    )
    with pytest.raises(MockReplayValidationError, match="not a public-only key"):
        MockPublicReplay.from_dict(artifact_with_prefix_variant)
