from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from hok_agent.cli import main
from hok_agent.movement_delivery import (
    _object_sha256,
    create_r0_package,
    verify_r0_package,
)
from hok_agent.movement_mvp import run_rule_batch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "movement_mvp.json"


@pytest.fixture(scope="module")
def delivery_evidence(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("movement-delivery")
    interrupted, control, package = root / "interrupted", root / "control", root / "package"
    run_rule_batch(CONFIG, interrupted, 10, step_budget=4)
    run_rule_batch(CONFIG, interrupted, 10, resume=True)
    run_rule_batch(CONFIG, control, 10)
    create_r0_package(interrupted, control, package)
    return interrupted, control, package


def _rewrite_self_bound(path: Path, field: str, updates: dict[str, object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    payload.pop(field)
    payload[field] = _object_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_r0_package_create_and_verify(
    delivery_evidence: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    interrupted, control, package = delivery_evidence
    result = verify_r0_package(package)
    assert result["status"] == "PASSED"
    assert result["delivery_grade"] == "R0_RULE_OFFLINE"
    assert result["transitions"] == 90
    assert result["terminal_transitions"] == 10
    assert result["frame_bundles"] == 100
    assert result["reward_total"] == 0.0
    assert result["input_commands_sent"] == 0
    assert result["promoted_checkpoint"] is None
    assert not tuple(package.glob("replay.sqlite3-*"))
    assert not tuple(package.rglob("*.safetensors"))
    with pytest.raises(ValueError, match="already exists"):
        create_r0_package(interrupted, control, package)
    with pytest.raises(ValueError, match="accepted pair"):
        create_r0_package(control, interrupted, tmp_path / "reversed")

    fresh = tmp_path / "fresh"
    created = create_r0_package(interrupted, control, fresh)
    assert created["transition_content_sha256"] == result["transition_content_sha256"]


@pytest.mark.parametrize(
    ("relative", "content"),
    (
        ("summary.json", b"{}"),
        ("manifest.json", b"{}"),
        ("frames/movement-rule-seed-0-episode-000-000.npz", b"corrupt"),
        ("replay.sqlite3", b"corrupt"),
    ),
)
def test_r0_package_rejects_tampered_files(
    delivery_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
    relative: str,
    content: bytes,
) -> None:
    package = delivery_evidence[2]
    tampered = tmp_path / relative.replace("/", "-")
    shutil.copytree(package, tampered)
    (tampered / relative).write_bytes(content)
    with pytest.raises(ValueError):
        verify_r0_package(tampered)


@pytest.mark.parametrize(
    "updates",
    (
        {"completed_episodes": 9},
        {"reward_total": 1.0},
        {"input_commands_sent": 1},
    ),
)
def test_r0_package_rejects_ineligible_source(
    delivery_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    source, control, _package = delivery_evidence
    changed = tmp_path / next(iter(updates))
    shutil.copytree(source, changed)
    _rewrite_self_bound(changed / "batch-summary.json", "summary_sha256", updates)
    with pytest.raises(ValueError, match="admission fields"):
        create_r0_package(changed, control, tmp_path / f"output-{next(iter(updates))}")


def test_package_cli_is_lazy_and_verify_only_is_read_only(
    delivery_evidence: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    package = delivery_evidence[2]
    forbidden = (
        "hok_agent.mobile_testbed",
        "hok_agent.shadow",
        "hok_agent.movement_mvp_train",
    )
    for name in forbidden:
        sys.modules.pop(name, None)
    before = (package / "manifest.json").read_bytes()
    assert (
        main(
            [
                "movement-mvp",
                "--mode",
                "package",
                "--verify-only",
                "--output-dir",
                str(package),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (package / "manifest.json").read_bytes() == before
    assert all(name not in sys.modules for name in forbidden)
