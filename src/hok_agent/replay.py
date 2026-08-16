from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import cast

from hok_agent.arena import FactorizedAction, Side, observation_hash
from hok_agent.policies import make_policy
from hok_agent.service import PixelArenaService


class ReplayError(RuntimeError): pass  # noqa: E701


HEADER_KEYS = {
    "kind", "claim_scope", "hok_capability_claim", "gamecore_equivalence_claim",
    "environment_identity", "config_hash", "seed", "blue_policy", "red_policy",
    "initial_observation_hash",
}
ROW_KEYS = {"tick", "blue_action", "red_action", "observation_hash", "events", "terminal",
            "outcome"}
ACTION_KEYS = {"macro", "action_type", "target", "direction", "skill", "upgrade", "auxiliary"}


def _legal(response: dict[str, object], side: Side) -> tuple[FactorizedAction, ...]:
    by_side = cast(dict[str, object], response["legal_actions"])
    rows = cast(list[dict[str, object]], by_side[side])
    return tuple(FactorizedAction.from_dict(row) for row in rows)


def _state_hash(response: dict[str, object]) -> str:
    return observation_hash(cast(dict[str, object], response["observation"]))


def _write_jsonl(path: Path, documents: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                   for row in documents)
    path.write_text(text, encoding="utf-8")


def record_episode(path: Path, blue_policy: str, red_policy: str,
                   seed: int) -> dict[str, object]:
    blue = make_policy(blue_policy, seed, "blue"); red = make_policy(red_policy, seed, "red")  # noqa: E702
    with PixelArenaService() as service:
        health = service.health()
        response = service.reset(seed)
        header: dict[str, object] = {
            "kind": "pixelarena_public_trace_v1", "claim_scope": "pixelarena_engineering",
            "hok_capability_claim": False, "gamecore_equivalence_claim": False,
            "environment_identity": health["identity"], "config_hash": health["config_hash"],
            "seed": seed, "blue_policy": blue_policy, "red_policy": red_policy,
            "initial_observation_hash": _state_hash(response),
        }
        documents = [header]; moves = attacks = structure_damage = 0  # noqa: E702
        while not bool(response["terminal"]):
            blue_action = blue.select("blue", _legal(response, "blue"))
            red_action = red.select("red", _legal(response, "red"))
            response = service.step(blue_action, red_action)
            events = cast(list[str], response["events"])
            moves += sum(":move:" in event for event in events)
            attacks += sum(":attack:" in event for event in events)
            structure_damage += sum(
                "_tower_health:" in event or "_crystal_health:" in event for event in events
            )
            documents.append(
                {
                    "tick": cast(dict[str, object],
                                 cast(dict[str, object], response["observation"])["blue"])["tick"],
                    "blue_action": blue_action.to_dict(), "red_action": red_action.to_dict(),
                    "observation_hash": _state_hash(response), "events": events,
                    "terminal": response["terminal"],
                    "outcome": response["outcome"],
                }
            )
        blue_observation = cast(dict[str, object],
                                cast(dict[str, object], response["observation"])["blue"])
        _write_jsonl(path, documents)
        return {
            "path": str(path), "ticks": blue_observation["tick"],
            "outcome": response["outcome"], "terminal": response["terminal"],
            "moves": moves, "attacks": attacks,
            "structure_damage_events": structure_damage,
            "process_id": health["process_id"],
        }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        document = json.loads(line)
        if not isinstance(document, dict): raise ReplayError("trace rows must be objects")  # noqa: E701
        documents.append(cast(dict[str, object], document))
    return documents


def _valid_action_document(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != ACTION_KEYS:
        return False
    string_keys = ACTION_KEYS - {"auxiliary"}
    return (all(type(value[key]) is str for key in string_keys)
            and type(value["auxiliary"]) is int)


def _validate_trace_types(header: dict[str, object], rows: list[dict[str, object]]) -> None:
    string_header = {
        "kind", "claim_scope", "environment_identity", "config_hash", "blue_policy", "red_policy",
        "initial_observation_hash",
    }
    if not all(type(header[key]) is str for key in string_header):
        raise ReplayError("invalid trace header types")
    if type(header["seed"]) is not int or any(
        type(header[key]) is not bool
        for key in ("hok_capability_claim", "gamecore_equivalence_claim")
    ): raise ReplayError("invalid trace header types")  # noqa: E701
    for row in rows:
        if type(row["tick"]) is not int: raise ReplayError("invalid transition types")  # noqa: E701
        if not all(_valid_action_document(row[f"{side}_action"]) for side in ("blue", "red")):
            raise ReplayError("invalid action types")
        if (
            type(row["observation_hash"]) is not str
            or type(row["terminal"]) is not bool
            or type(row["outcome"]) is not str
            or not isinstance(row["events"], list)
            or not all(type(event) is str for event in row["events"])
        ): raise ReplayError("invalid transition types")  # noqa: E701


def verify_trace(path: Path) -> dict[str, object]:
    documents = _read_jsonl(path)
    if len(documents) < 2: raise ReplayError("invalid trace header")  # noqa: E701
    header, rows = documents[0], documents[1:]
    if set(header) != HEADER_KEYS: raise ReplayError("invalid trace header fields")  # noqa: E701
    required_header = {
        "kind": "pixelarena_public_trace_v1", "claim_scope": "pixelarena_engineering",
        "hok_capability_claim": False, "gamecore_equivalence_claim": False,
    }
    if any(header.get(key) != value for key, value in required_header.items()):
        raise ReplayError("invalid trace claim metadata")
    for row in rows:
        if set(row) != ROW_KEYS: raise ReplayError("invalid transition fields")  # noqa: E701
        for side in ("blue", "red"):
            action = row[f"{side}_action"]
            if not isinstance(action, dict) or set(action) != ACTION_KEYS:
                raise ReplayError("invalid action fields")
    _validate_trace_types(header, rows)
    with PixelArenaService() as service:
        health = service.health()
        for key, actual in (("environment_identity", health["identity"]),
                            ("config_hash", health["config_hash"])):
            if header.get(key) != actual: raise ReplayError(f"header mismatch: {key}")  # noqa: E701
        seed = int(cast(int, header["seed"]))
        try:
            blue_policy = make_policy(str(header["blue_policy"]), seed, "blue")
            red_policy = make_policy(str(header["red_policy"]), seed, "red")
        except ValueError as exc:
            raise ReplayError("invalid policy metadata") from exc
        response = service.reset(seed)
        if header.get("initial_observation_hash") != _state_hash(response):
            raise ReplayError("initial observation mismatch")
        for expected_tick, row in enumerate(rows, start=1):
            blue_action = FactorizedAction.from_dict(cast(dict[str, object], row["blue_action"]))
            red_action = FactorizedAction.from_dict(cast(dict[str, object], row["red_action"]))
            if blue_action != blue_policy.select("blue", _legal(response, "blue")):
                raise ReplayError(f"blue policy mismatch at tick {expected_tick}")
            if red_action != red_policy.select("red", _legal(response, "red")):
                raise ReplayError(f"red policy mismatch at tick {expected_tick}")
            response = service.step(blue_action, red_action)
            expected = {
                "tick": expected_tick, "observation_hash": _state_hash(response),
                "events": response["events"], "terminal": response["terminal"],
                "outcome": response["outcome"],
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ReplayError(f"transition mismatch at tick {expected_tick}")
        if not bool(response["terminal"]): raise ReplayError("trace does not end at terminal")  # noqa: E701
        return {
            "verified": True, "ticks": len(rows), "outcome": response["outcome"],
            "process_id": health["process_id"],
        }


def accept_minimal_v1(seed: int, output_dir: Path | None = None) -> dict[str, object]:
    temporary_output = output_dir is None
    temporary = tempfile.TemporaryDirectory() if temporary_output else None
    root = Path(temporary.name) if temporary else output_dir
    assert root is not None; root.mkdir(parents=True, exist_ok=True)  # noqa: E702
    definitions = (
        ("blue-scripted", "scripted", "null", "blue_win_crystal_destroyed"),
        ("red-scripted", "null", "scripted", "red_win_crystal_destroyed"),
        ("seeded-random", "random", "null", None),
    )
    scenarios: list[dict[str, object]] = []
    for name, blue, red, expected_outcome in definitions:
        path = root / f"{name}.jsonl"
        recorded = record_episode(path, blue, red, seed); verified = verify_trace(path)  # noqa: E702
        if expected_outcome is not None and recorded["outcome"] != expected_outcome:
            raise ReplayError(f"unexpected outcome for {name}")
        if recorded["process_id"] == verified["process_id"]:
            raise ReplayError("replay did not use a fresh process")
        if name.endswith("scripted") and not all(
            int(cast(int, recorded[key])) > 0
            for key in ("moves", "attacks", "structure_damage_events")
        ):
            raise ReplayError(f"missing causal events for {name}")
        recorded["trace_retained"] = not temporary_output
        if temporary_output: del recorded["path"]  # noqa: E701
        scenarios.append({"name": name, "record": recorded, "replay": verified})
    original = _read_jsonl(root / "blue-scripted.jsonl")
    original[1]["observation_hash"] = "0" * 64
    tampered = root / "tampered.jsonl"
    _write_jsonl(tampered, original)
    tamper_rejected = False
    try:
        verify_trace(tampered)
    except ReplayError:
        tamper_rejected = True
    if temporary: temporary.cleanup()  # noqa: E701
    if not tamper_rejected: raise ReplayError("tampered trace was accepted")  # noqa: E701
    return {
        "acceptance": "PASSED", "claim_scope": "pixelarena_engineering",
        "hok_capability_claim": False, "gamecore_equivalence_claim": False,
        "policies_are_learned": False,
        "scenarios": scenarios,
        "tamper_rejected": True,
    }
