import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from hok_agent.hierarchical_p0_ssl import TemporalSSL
from hok_agent.hierarchical_p0_ssl_v2 import (
    P0SSLV2Error,
    _pair_batch,
    load_contract,
    verify_representation,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_v2_contract_and_pair_batch_preserve_labels() -> None:
    config, payload, digest = load_contract(Path("configs/hierarchical_p0_temporal_ssl_v2.json"))
    clips = torch.arange(2 * 16 * 2 * 3, dtype=torch.uint8).reshape(2, 16, 2, 3, 1)
    permutations = torch.tensor([list(range(16)), [0, *range(14, 0, -1), 15]])
    reversed_pairs = torch.tensor([False, True])
    samples, labels = _pair_batch(clips, permutations, reversed_pairs, torch.device("cpu"))
    assert samples.shape == (4, 16, 1, 2, 3)
    assert labels.tolist() == [0, 1, 1, 0]
    assert config.seed == 0 and config.epochs == 12 and len(digest) == 64
    assert payload["training"]["checkpoint_selection"] == "fixed_last_epoch_without_dev_selection"
    assert payload["claim_boundary"]["training_video_dev_allowed"] is False
    assert payload["claim_boundary"]["video_test_allowed"] is False


def test_v2_representation_verifier_rejects_weight_tamper(tmp_path: Path) -> None:
    _config, _payload, contract_sha = load_contract(
        Path("configs/hierarchical_p0_temporal_ssl_v2.json")
    )
    checkpoint = tmp_path / "representation.safetensors"
    state = {
        key: value
        for key, value in TemporalSSL().state_dict().items()
        if not key.startswith("classifier.")
    }
    save_file(state, checkpoint)
    report: dict[str, object] = {
        "schema_version": "hok-agent-hierarchical-p0-temporal-ssl-v2-report-v1",
        "status": "P0_TEMPORAL_SSL_V2_PASSED",
        "contract_sha256": contract_sha,
        "representation_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "p0_initialization_allowed": True,
        "video_test_opened": False,
        "policy_training_allowed": False,
        "reward_allowed": False,
    }
    report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    (tmp_path / "report.json").write_bytes(_canonical(report) + b"\n")
    assert verify_representation(tmp_path, Path("configs/hierarchical_p0_temporal_ssl_v2.json"))
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(P0SSLV2Error, match="representation binding differs"):
        verify_representation(tmp_path, Path("configs/hierarchical_p0_temporal_ssl_v2.json"))
