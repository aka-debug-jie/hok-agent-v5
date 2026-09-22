# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent import global_policy as gp


def _write_shard(path: Path, frames: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        main_rgb=rng.integers(0, 255, (frames, 8, 8, 3), dtype=np.uint8),
        minimap_rgb=rng.integers(0, 255, (frames, 4, 4, 3), dtype=np.uint8),
        hud_rgb=rng.integers(0, 255, (frames, 2, 4, 3), dtype=np.uint8),
        intent=np.zeros(frames, dtype=np.int64),
        zone=np.zeros(frames, dtype=np.int64),
        scene=np.zeros(frames, dtype=np.int64),
        tick=np.arange(frames, dtype=np.int64),
    )
    return gp._sha(path.read_bytes())


def _write_dataset(root: Path, episodes: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(episodes):
        name = f"episode-{index}.npz"
        digest = _write_shard(root / name, gp.WINDOW_FRAMES, index)
        rows.append(
            {"split": "dev", "shard": name, "shard_sha256": digest, "episode": f"ep-{index}"}
        )
    payload: dict[str, object] = {"test_present": False, "episodes": rows}
    payload["manifest_sha256"] = gp._sha(gp._canonical(payload).encode())
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_parameter_groups_sum_to_total_and_the_main_view_dominates() -> None:
    model = gp.GlobalMacroPolicy("tcn")
    groups = gp._parameter_groups(model)
    assert groups["main"] + groups["minimap"] + groups["hud"] + groups["temporal_and_heads"] == (
        groups["total"]
    )
    assert groups["main"] > 0.9 * groups["total"]


def _report(intent_f1: float, median: float, p95: float, parameters: int) -> dict:
    return {
        "schema_version": gp.GLOBAL_SPEED_SCHEMA,
        "behaviour": {"intent_macro_f1": intent_f1, "zone_macro_f1": 0.7},
        "latency_ms": {"median": median, "p95": p95},
        "parameters": {"total": parameters, "main": parameters, "minimap": 1, "hud": 1, "temporal_and_heads": 1},
    }


def test_speed_verdict_passes_only_when_behaviour_holds_and_both_latencies_drop() -> None:
    baseline = _report(0.889, 2790.0, 4573.0, 11_383_694)
    candidate = _report(0.889, 900.0, 1500.0, 2_000_000)
    verdict = gp.global_speed_verdict(baseline=baseline, candidate=candidate)
    assert verdict["passed"] is True
    assert verdict["reasons"] == []


def test_speed_verdict_rejects_a_faster_model_that_loses_behaviour() -> None:
    baseline = _report(0.889, 2790.0, 4573.0, 11_383_694)
    candidate = _report(0.500, 500.0, 900.0, 1_000_000)
    verdict = gp.global_speed_verdict(baseline=baseline, candidate=candidate)
    assert verdict["passed"] is False
    assert "intent_macro_f1_regressed" in verdict["reasons"]


def test_speed_verdict_rejects_a_p95_regression_even_when_the_median_improves() -> None:
    baseline = _report(0.889, 2790.0, 4573.0, 11_383_694)
    candidate = _report(0.889, 1000.0, 4500.0, 2_000_000)
    verdict = gp.global_speed_verdict(baseline=baseline, candidate=candidate)
    assert verdict["passed"] is False
    assert "p95_latency_not_reduced" in verdict["reasons"]


def test_speed_verdict_never_passes_on_parameters_alone() -> None:
    """A smaller model with no latency win must fail: parameters are not a substitute."""
    baseline = _report(0.889, 2790.0, 4573.0, 11_383_694)
    candidate = _report(0.889, 3000.0, 5000.0, 1_000_000)
    verdict = gp.global_speed_verdict(baseline=baseline, candidate=candidate)
    assert verdict["passed"] is False
    assert verdict["parameters_delta"] < 0
    assert verdict["reasons"] == ["median_latency_not_reduced", "p95_latency_not_reduced"]


def test_measure_global_speed_refuses_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(gp.GlobalPolicyError):
        gp.measure_global_speed(
            checkpoint_path=tmp_path / "missing.safetensors",
            dataset_root=tmp_path,
        )


def test_measure_global_speed_reports_parameters_latency_and_behaviour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    root = _write_dataset(tmp_path / "dataset", episodes=3)
    model = gp.GlobalMacroPolicy("pool_mlp")
    checkpoint = tmp_path / "selected.safetensors"
    gp._save_model(checkpoint, model, "pool_mlp", "0" * 64)

    report = gp.measure_global_speed(
        checkpoint_path=checkpoint,
        dataset_root=root,
        split="dev",
        batch_size=2,
        repetitions=3,
        warmup=1,
    )

    assert report["schema_version"] == gp.GLOBAL_SPEED_SCHEMA
    assert report["windows"] == 3
    assert report["parameters"]["total"] == sum(p.numel() for p in model.parameters())
    assert report["parameters"]["main"] > 0
    assert report["latency_ms"]["median"] > 0
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["minimum"]
    assert 0.0 <= report["behaviour"]["intent_macro_f1"] <= 1.0
    assert report["checkpoint_sha256"] == gp._sha(checkpoint.read_bytes())
    assert report["variant"] == "pool_mlp"


def test_measure_global_speed_refuses_a_symlinked_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    root = _write_dataset(tmp_path / "dataset", episodes=2)
    real = tmp_path / "real.safetensors"
    gp._save_model(real, gp.GlobalMacroPolicy("pool_mlp"), "pool_mlp", "0" * 64)
    link = tmp_path / "link.safetensors"
    link.symlink_to(real)
    with pytest.raises(gp.GlobalPolicyError):
        gp.measure_global_speed(
            checkpoint_path=link, dataset_root=root, batch_size=1, repetitions=1, warmup=0
        )


def test_compare_global_speed_interleaves_and_returns_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOK_LARGE_ROOT", str(tmp_path))
    root = _write_dataset(tmp_path / "dataset", episodes=3)
    baseline = tmp_path / "baseline.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    gp._save_model(baseline, gp.GlobalMacroPolicy("pool_mlp"), "pool_mlp", "0" * 64)
    # Same weights on both sides, so any behaviour difference is measurement noise, not a model.
    candidate.write_bytes(baseline.read_bytes())

    comparison = gp.compare_global_speed(
        baseline_checkpoint=baseline,
        candidate_checkpoint=candidate,
        dataset_root=root,
        split="dev",
        batch_size=2,
        repetitions=3,
        warmup=1,
        minimum_latency_reduction_fraction=0.0,
    )

    assert comparison["measurement"] == "interleaved"
    assert comparison["baseline"]["measurement"] == "interleaved"
    assert comparison["candidate"]["measurement"] == "interleaved"
    assert set(comparison["verdict"]) >= {"passed", "reasons", "parameters_delta"}
    # Identical models: behaviour holds, so with a zero latency threshold only the latency half can
    # fail, and the parameter delta is exactly zero.
    assert comparison["verdict"]["parameters_delta"] == 0
    assert "intent_macro_f1_regressed" not in comparison["verdict"]["reasons"]


def test_compare_global_speed_refuses_a_missing_baseline(tmp_path: Path) -> None:
    with pytest.raises(gp.GlobalPolicyError):
        gp.compare_global_speed(
            baseline_checkpoint=tmp_path / "missing.safetensors",
            candidate_checkpoint=tmp_path / "also-missing.safetensors",
            dataset_root=tmp_path,
        )
