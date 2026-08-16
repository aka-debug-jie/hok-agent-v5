# ruff: noqa: E501, E702
from __future__ import annotations

import copy
import inspect
import json
from collections import defaultdict
from pathlib import Path

import pytest
import torch

import hok_agent.bc as bc


@pytest.fixture(scope="module")
def dataset() -> bc.Dataset:
    return bc.collect_dataset()


def _write(path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_collect_dataset_contract(dataset: bc.Dataset, tmp_path: Path) -> None:
    assert dataset.episode_count == 256; assert len(dataset.samples) >= 400; assert dataset.conflict_count == 0
    split_hashes: dict[str, set[str]] = defaultdict(set)
    by_action: dict[int, set[str]] = defaultdict(set)
    for sample in dataset.samples:
        split_hashes[sample.split].add(sample.digest)
        by_action[sample.action].add(sample.split)
    assert set(split_hashes) == {"train", "validation", "test"}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert split_hashes[left].isdisjoint(split_hashes[right])
    assert all(len(splits) == 3 for splits in by_action.values())
    path = tmp_path / "dataset.jsonl"; bc.write_dataset(path, dataset)
    rows = [json.loads(line) for line in path.read_text().splitlines()]; assert rows
    for row in rows[1:]:
        assert set(row) == {"sample_hash", "split", "observation", "action"}
        assert not {"legal_actions", "reward", "teacher", "truth"} & set(row)
        assert row["split"] in split_hashes


def test_forward_signature_train0_roundtrip_and_load_model_validation(
    dataset: bc.Dataset, tmp_path: Path
) -> None:
    signature = inspect.signature(bc.StructuredActor.forward)
    assert list(signature.parameters) == ["self", "observation"]
    actor = bc.StructuredActor()
    assert sum(param.numel() for param in actor.parameters()) == 550
    trained, metrics = bc.train_one(dataset, 0)
    assert bool(metrics["passed"])
    document = copy.deepcopy(bc._model_document(trained, dataset, 0))
    payload = json.loads(json.dumps(document))
    path = tmp_path / "model.json"
    _write(path, payload); loaded, seed = bc.load_model(path, dataset.config)
    assert seed == 0
    features = bc._feature_vector(dataset.samples[0].observation, dataset.config)
    x = torch.tensor([features], dtype=torch.float32)
    torch.testing.assert_close(loaded(x), trained(x))

    invalid = copy.deepcopy(payload)
    invalid["unexpected"] = True
    _write(tmp_path / "bad-extra.json", invalid)
    with pytest.raises(bc.BCError):
        bc.load_model(tmp_path / "bad-extra.json", dataset.config)

    bad_shape = copy.deepcopy(payload)
    bad_shape["weights"]["hidden.weight"] = [[0.0]]
    _write(tmp_path / "bad-shape.json", bad_shape)
    with pytest.raises(bc.BCError):
        bc.load_model(tmp_path / "bad-shape.json", dataset.config)

    bad_config = copy.deepcopy(payload)
    bad_config["config_hash"] = "0" * 64
    _write(tmp_path / "bad-config.json", bad_config)
    with pytest.raises(bc.BCError):
        bc.load_model(tmp_path / "bad-config.json", dataset.config)

    nonfinite = copy.deepcopy(payload)
    nonfinite["weights"]["hidden.bias"][0] = float("nan")
    _write(tmp_path / "nonfinite.json", nonfinite)
    with pytest.raises(bc.BCError):
        bc.load_model(tmp_path / "nonfinite.json", dataset.config)

    wrong_nested_type = copy.deepcopy(payload)
    wrong_nested_type["architecture"]["input"] = 10.0
    _write(tmp_path / "wrong-type.json", wrong_nested_type)
    with pytest.raises(bc.BCError):
        bc.load_model(tmp_path / "wrong-type.json", dataset.config)

    duplicate = json.dumps(payload).replace('"kind":', '"kind":"duplicate","kind":', 1)
    (tmp_path / "duplicate.json").write_text(duplicate, encoding="utf-8")
    with pytest.raises(bc.BCError):
        bc.load_model(tmp_path / "duplicate.json", dataset.config)


def test_accept_minimal_v2_produces_full_report(tmp_path: Path) -> None:
    output = tmp_path / "accept"
    report = bc.accept_minimal_v2(output)
    expected = {"dataset.jsonl", "model-seed-0.json", "model-seed-1.json", "model-seed-2.json", "report.json"}
    produced = {file.name for file in output.iterdir()}
    assert produced == expected
    assert report["status"] == "PASSED"
    best = min(report["training_runs"], key=lambda run: run["validation_cross_entropy"])
    assert report["best_model"] == f"model-seed-{best['seed']}.json"
    assert len(report["closed_loop"]) == 2
    assert all(
        loop["raw_illegal_actions"] == 0
        and loop["mask_corrections"] == 0
        and loop["passed"]
        for loop in report["closed_loop"]
    )
    for name, digest in report["files"].items():
        assert bc._sha(output / name) == digest

    existing = tmp_path / "existing"; existing.mkdir()
    (existing / "marker.txt").write_text("KEEP", encoding="utf-8")
    with pytest.raises(bc.BCError):
        bc.accept_minimal_v2(existing)
    assert (existing / "marker.txt").read_text(encoding="utf-8") == "KEEP"
    assert not (existing / "report.json").exists()


def test_accept_minimal_v2_train_one_failure_only_keeps_failed_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(*_args: object, **_kwargs: object) -> tuple[bc.StructuredActor, dict[str, object]]:
        raise RuntimeError("boom")

    monkeypatch.setattr(bc, "train_one", broken)
    output = tmp_path / "failed"
    with pytest.raises(bc.BCError):
        bc.accept_minimal_v2(output)
    files = {path.name for path in output.iterdir()}
    assert files == {"report.json"}
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED"
    assert report["stage"] == "train"
    assert report["error_type"] == "RuntimeError"
    assert "boom" in report["error"]
