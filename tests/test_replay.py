# ruff: noqa: E501, E702
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hok_agent.replay import ReplayError, accept_minimal_v1, record_episode, verify_trace


def test_public_jsonl_replays_in_a_fresh_process_and_rejects_tamper(tmp_path: Path) -> None:
    trace = tmp_path / "episode.jsonl"; recorded = record_episode(trace, "scripted", "null", 101)
    verified = verify_trace(trace)
    assert recorded["outcome"] == "blue_win_crystal_destroyed"
    assert recorded["process_id"] != verified["process_id"]
    documents = [json.loads(line) for line in trace.read_text().splitlines()]
    assert documents[0]["claim_scope"] == "pixelarena_engineering"
    assert documents[0]["hok_capability_claim"] is False
    persisted_keys = set().union(*(row.keys() for row in documents))
    assert not persisted_keys & {"legal_actions", "reward", "teacher", "truth", "entity_id"}
    documents[1]["observation_hash"] = "0" * 64
    trace.write_text("".join(json.dumps(row) + "\n" for row in documents))
    with pytest.raises(ReplayError, match="transition mismatch"):
        verify_trace(trace)


@pytest.mark.parametrize(("location", "key", "value", "message"), [(0, "hok_capability_claim", True, "claim metadata"), (0, "teacher", "hidden", "header fields"), (1, "reward", 1.0, "transition fields")])
def test_replay_rejects_claim_changes_and_non_public_fields(
    tmp_path: Path, location: int, key: str, value: object, message: str) -> None:
    trace = tmp_path / "episode.jsonl"
    record_episode(trace, "scripted", "null", 101); documents = [json.loads(line) for line in trace.read_text().splitlines()]; documents[location][key] = value
    trace.write_text("".join(json.dumps(row) + "\n" for row in documents))
    with pytest.raises(ReplayError, match=message):
        verify_trace(trace)


@pytest.mark.parametrize(("location", "key", "value", "message"), [
    (0, "seed", [101], "header types"),
    (0, "seed", 101.9, "header types"),
    (0, "hok_capability_claim", 0, "header types"),
    (1, "blue_action", list("abcdefg"), "action fields"), (1, "events", "not-a-list", "transition types"),
])
def test_replay_rejects_wrong_json_types(
    tmp_path: Path, location: int, key: str, value: object, message: str) -> None:
    trace = tmp_path / "episode.jsonl"
    record_episode(trace, "scripted", "null", 101); documents = [json.loads(line) for line in trace.read_text().splitlines()]; documents[location][key] = value
    trace.write_text("".join(json.dumps(row) + "\n" for row in documents))
    with pytest.raises(ReplayError, match=message):
        verify_trace(trace)


def test_replay_rejects_wrong_action_value_type_and_policy_metadata(tmp_path: Path) -> None:
    trace = tmp_path / "episode.jsonl"
    record_episode(trace, "scripted", "null", 101)
    documents = [json.loads(line) for line in trace.read_text().splitlines()]; documents[1]["blue_action"]["auxiliary"] = 0.9
    trace.write_text("".join(json.dumps(row) + "\n" for row in documents))
    with pytest.raises(ReplayError, match="action types"):
        verify_trace(trace)
    documents[1]["blue_action"]["auxiliary"] = 0; documents[0]["blue_policy"] = "null"
    trace.write_text("".join(json.dumps(row) + "\n" for row in documents))
    with pytest.raises(ReplayError, match="policy mismatch"):
        verify_trace(trace)


def test_default_acceptance_does_not_report_deleted_paths() -> None:
    report = accept_minimal_v1(101)
    for scenario in report["scenarios"]:
        assert scenario["record"]["trace_retained"] is False
        assert "path" not in scenario["record"]


def test_complete_acceptance_gate(tmp_path: Path) -> None:
    report = accept_minimal_v1(101, tmp_path)
    assert report["acceptance"] == "PASSED"
    assert report["tamper_rejected"] is True
    outcomes = {scenario["record"]["outcome"] for scenario in report["scenarios"]}; assert outcomes >= {"blue_win_crystal_destroyed", "red_win_crystal_destroyed"}



def test_seeded_random_recording_is_identical_across_runs(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    record_episode(first, "random", "random", 19)
    record_episode(second, "random", "random", 19)
    assert first.read_bytes() == second.read_bytes()
