from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file

from hok_agent.hierarchical_e1d_checkpoint import (
    BUNDLE_SCHEMA,
    CheckpointError,
    load_checkpoint_contract,
    verify_checkpoint_bundle,
)
from hok_agent.hierarchical_e1d_probe import TerminalProbe

CONTRACT = Path("configs/hierarchical_event_e1d_checkpoint.json")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_checkpoint_bundle_verifies_weights_and_rejects_tamper(tmp_path: Path) -> None:
    _contract, contract_sha = load_checkpoint_contract(CONTRACT)
    checkpoints = {}
    for mode in ("temporal", "last_frame", "shuffled"):
        path = tmp_path / f"{mode}.safetensors"
        save_file(TerminalProbe(mode).state_dict(), path)
        checkpoints[mode] = {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    payload: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA,
        "contract_sha256": contract_sha,
        "checkpoints": checkpoints,
        "test_opened": False,
        "reward_allowed": False,
    }
    payload["bundle_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    (tmp_path / "bundle.json").write_bytes(_canonical(payload) + b"\n")
    assert verify_checkpoint_bundle(tmp_path, CONTRACT) == payload
    (tmp_path / "temporal.safetensors").write_bytes(b"tampered")
    with pytest.raises(CheckpointError, match="weight hash differs"):
        verify_checkpoint_bundle(tmp_path, CONTRACT)
