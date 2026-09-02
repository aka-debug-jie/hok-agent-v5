from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.hierarchical_e1 import (
    AuditSession,
    E1Error,
    HealthTemporalEventEngine,
    audit_health_sessions,
    detect_center_health_bar,
    load_health_contract,
    load_health_report,
)
from hok_agent.visual_events import EventEngineIdentity, LifeState, VisualEventType

CONTRACT = Path("configs/hierarchical_event_e1_health.json")


def _frame(run_length: int, *, distractor_length: int = 0) -> np.ndarray:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    if run_length:
        frame[30, 58 : 58 + run_length] = (40, 180, 50)
    if distractor_length:
        frame[12, 45 : 45 + distractor_length] = (40, 180, 50)
    return frame


def test_center_health_bar_detects_ratio_and_ignores_oversized_distractor() -> None:
    config, _contract, _sha = load_health_contract(CONTRACT)
    full = detect_center_health_bar(_frame(12), config)
    half = detect_center_health_bar(_frame(6, distractor_length=30), config)
    missing = detect_center_health_bar(_frame(0), config)
    assert full.visible is True and full.hp_ratio == 1.0
    assert half.visible is True and half.hp_ratio == 0.5 and half.run_length_px == 6
    assert missing.visible is False and missing.hp_ratio is None


def test_health_temporal_engine_emits_hp_death_and_respawn_once() -> None:
    config, _contract, contract_sha = load_health_contract(CONTRACT)
    engine = HealthTemporalEventEngine(
        "episode-001",
        EventEngineIdentity("health-v1", contract_sha),
        config,
    )
    events = []
    timestamp = 0
    for run_length in [12, 12, 12, 6, 6, 0, 0, 0, 0, 12, 12, 12]:
        timestamp += 200_000_000
        update = engine.update(
            f"obs-{timestamp}",
            timestamp,
            detect_center_health_bar(_frame(run_length), config),
        )
        events.extend(update.events)
    assert [event.event_type for event in events] == [
        VisualEventType.SELF_HP_DELTA,
        VisualEventType.DEATH,
        VisualEventType.RESPAWN,
    ]
    assert events[0].delta == -0.5
    assert update.state.life_state == LifeState.ALIVE
    assert len(engine.fusion.accepted) == 3


def _write_session(path: Path, runs: list[int], hard_stops: list[int]) -> None:
    path.mkdir()
    frames = np.stack([_frame(run) for run in runs])
    np.savez_compressed(
        path / "observations-0000.npz",
        main_rgb=frames,
        scheduled_elapsed_ms=np.arange(len(frames), dtype=np.int64) * 200,
        hard_stop=np.asarray(hard_stops, dtype=np.uint8),
    )
    (path / "summary.json").write_text('{"status":"SYNTHETIC"}\n', encoding="utf-8")


def test_health_audit_is_path_free_transactional_and_non_promoting(tmp_path: Path) -> None:
    train_a = tmp_path / "train-a"
    train_b = tmp_path / "train-b"
    dev = tmp_path / "dev-death"
    challenge = tmp_path / "challenge"
    _write_session(train_a, [12] * 10, [0] * 10)
    _write_session(train_b, [12] * 10, [0] * 10)
    _write_session(dev, [12, 12, 12, 0, 0, 0, 0, 12, 12, 12], [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    _write_session(challenge, [12] * 10, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    output = tmp_path / "report"
    payload = audit_health_sessions(
        CONTRACT,
        (
            AuditSession("train-a", "train_live", train_a),
            AuditSession("train-b", "train_live", train_b),
            AuditSession("dev-death", "dev_death", dev),
            AuditSession("challenge", "challenge_false_positive", challenge),
        ),
        output,
    )
    assert payload["status"] == "E1A_ENGINEERING_DIAGNOSTIC_PASSED"
    assert payload["reward_allowed"] is False
    assert payload["promotion_allowed"] is False
    assert payload["video_test_opened"] is False
    persisted = json.loads((output / "report.json").read_text())
    assert persisted == payload
    assert load_health_report(output / "report.json", CONTRACT) == payload
    assert str(tmp_path) not in (output / "report.json").read_text()
    tampered = tmp_path / "tampered.json"
    tampered.write_text((output / "report.json").read_text().replace("PASSED", "FAILED"))
    with pytest.raises(E1Error, match="report is invalid"):
        load_health_report(tampered, CONTRACT)
    with pytest.raises(E1Error, match="already exists"):
        audit_health_sessions(CONTRACT, (), output)
