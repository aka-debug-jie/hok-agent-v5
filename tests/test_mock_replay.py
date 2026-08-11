from __future__ import annotations

import json
from pathlib import Path

import pytest

from hok_agent.artifacts.hashing import sha256_file, sha256_json
from hok_agent.artifacts.verification import verify_artifact
from hok_agent.cli import main
from hok_agent.contracts import FactorizedAction
from hok_agent.contracts.types import ActionType, TargetType
from hok_agent.control_plane import ExternalAccessGate
from hok_agent.evaluation.mock_replay import (
    MockReplayExecutionError,
    record_mock_public_replay,
    verify_mock_public_replay,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "mock_public_replay.schema.json"
PROGRAM_CONFIG = ROOT / "configs" / "program_v1.yaml"


def _load(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _rehash(document: dict[str, object]) -> None:
    transitions = document["transitions"]
    assert isinstance(transitions, list)
    previous: str | None = None
    for transition in transitions:
        assert isinstance(transition, dict)
        transition["previous_row_hash"] = previous
        transition["row_hash"] = sha256_json({key: value for key, value in transition.items() if key != "row_hash"})
        row_hash = transition["row_hash"]
        assert isinstance(row_hash, str)
        previous = row_hash
    document["chain_head"] = previous
    document["artifact_hash"] = sha256_json({key: value for key, value in document.items() if key != "artifact_hash"})


def test_record_and_fresh_process_verify_are_exact(tmp_path: Path) -> None:
    path = tmp_path / "mock_public_replay.json"

    recorded = record_mock_public_replay(path, seed=101, side="blue", max_steps_per_episode=12)
    before_verification_hash = sha256_file(path)
    verified = verify_mock_public_replay(path)

    assert recorded.transition_count == 2
    assert verified.transition_count == recorded.transition_count
    assert verified.artifact_hash == recorded.artifact_hash
    assert recorded.service_pid != verified.service_pid
    assert verified.input_sha256 == before_verification_hash == sha256_file(path)
    assert verify_artifact(path, SCHEMA).passed


def test_same_seed_profile_is_byte_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    record_mock_public_replay(first_path, seed=102, side="red", max_steps_per_episode=12)
    record_mock_public_replay(second_path, seed=102, side="red", max_steps_per_episode=12)

    assert first_path.read_bytes() == second_path.read_bytes()
    with pytest.raises(MockReplayExecutionError, match="refusing to overwrite"):
        record_mock_public_replay(first_path, seed=102, side="red", max_steps_per_episode=12)


def test_rehashed_action_tamper_fails_fresh_transition_replay(tmp_path: Path) -> None:
    path = tmp_path / "tampered_action.json"
    record_mock_public_replay(path, seed=101, side="blue", max_steps_per_episode=12)
    document = _load(path)
    transitions = document["transitions"]
    assert isinstance(transitions, list)
    assert isinstance(transitions[0], dict)
    transitions[0]["action"] = FactorizedAction(
        schema_version=1,
        macro_intent="hold",
        action_type=ActionType.WAIT,
        target_type=TargetType.NONE,
        target_key=None,
        direction=None,
        skill_slot=None,
        item_slot=None,
        auxiliary_parameter=None,
    ).to_dict()
    _rehash(document)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert verify_artifact(path, SCHEMA).passed
    with pytest.raises(MockReplayExecutionError, match="transition.post_public_observation_hash"):
        verify_mock_public_replay(path)


def test_private_field_and_nonpublic_target_are_rejected(tmp_path: Path) -> None:
    private_path = tmp_path / "private.json"
    record_mock_public_replay(private_path, seed=101, side="blue", max_steps_per_episode=12)
    private_document = _load(private_path)
    transitions = private_document["transitions"]
    assert isinstance(transitions, list)
    assert isinstance(transitions[0], dict)
    transitions[0]["reward_components"] = {"total": 1.0}
    _rehash(private_document)
    private_path.write_text(
        json.dumps(private_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MockReplayExecutionError, match="schema/self-hash"):
        verify_mock_public_replay(private_path)

    target_path = tmp_path / "nonpublic_target.json"
    record_mock_public_replay(target_path, seed=101, side="blue", max_steps_per_episode=12)
    target_document = _load(target_path)
    target_transitions = target_document["transitions"]
    assert isinstance(target_transitions, list)
    assert isinstance(target_transitions[0], dict)
    action = target_transitions[0]["action"]
    assert isinstance(action, dict)
    action["target_key"] = "internal_target_17"
    _rehash(target_document)
    target_path.write_text(
        json.dumps(target_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert verify_artifact(target_path, SCHEMA).passed
    with pytest.raises(MockReplayExecutionError, match="not present in the fresh public observation"):
        verify_mock_public_replay(target_path)


def test_record_and_verify_do_not_use_control_plane_or_change_program_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_control_plane_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("mock replay must not reach the external control plane")

    monkeypatch.setattr(ExternalAccessGate, "require_local", unexpected_control_plane_call)
    monkeypatch.setattr(ExternalAccessGate, "require_runtime_license", unexpected_control_plane_call)
    before = sha256_file(PROGRAM_CONFIG)
    path = tmp_path / "isolated.json"

    record_mock_public_replay(path, seed=101, side="blue", max_steps_per_episode=12)
    verify_mock_public_replay(path)

    assert sha256_file(PROGRAM_CONFIG) == before


def test_cli_record_verify_and_generic_artifact_verification(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "mock_public_replay.json"

    assert main(["mock-replay-record", "--output", str(path), "--seed", "101"]) == 0
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["environment_kind"] == "mock"
    assert recorded["formal"] is False
    assert main(["mock-replay-verify", str(path)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["fresh_process_exact"] is True
    assert main(["verify-artifact", str(path), "--root", str(ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
