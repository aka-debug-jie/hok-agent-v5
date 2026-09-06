from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hok_agent.hierarchical_e1 import (
    AuditSession,
    E1Error,
    HealthTemporalEventEngine,
    audit_existing_health_candidates,
    audit_health_sessions,
    detect_center_health_bar,
    load_health_contract,
    load_health_report,
    replay_health_event_transitions,
)
from hok_agent.transition_store import UnifiedTransitionStore
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


def _write_bound_session(path: Path, runs: list[int], hard_stops: list[int]) -> Path:
    shard_dir = path / "shards"
    shard_dir.mkdir(parents=True)
    frames = np.stack([_frame(run) for run in runs])
    shard = shard_dir / "observations-0000.npz"
    np.savez_compressed(
        shard,
        main_rgb=frames,
        minimap_rgb=frames,
        hud_rgb=frames,
        scheduled_elapsed_ms=np.arange(len(frames), dtype=np.int64) * 200,
        hard_stop=np.asarray(hard_stops, dtype=np.uint8),
    )
    import hashlib

    summary = {
        "status": "PASSED",
        "derived_roi_rgb_persisted": True,
        "raw_frames_persisted": False,
        "observation_shards": [
            {
                "path": shard.name,
                "rows": len(frames),
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    summary_path = path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    return shard


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


def test_health_event_replay_writes_causal_zero_reward_nontraining_chain(tmp_path: Path) -> None:
    train_a = tmp_path / "train-a"
    train_b = tmp_path / "train-b"
    dev = tmp_path / "dev-death"
    challenge = tmp_path / "challenge"
    _write_session(train_a, [12] * 10, [0] * 10)
    _write_session(train_b, [12] * 10, [0] * 10)
    shard = _write_bound_session(
        dev,
        [12, 12, 12, 0, 0, 0, 0, 12, 12, 12],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    )
    _write_session(challenge, [12] * 10, [0] * 10)
    audit_dir = tmp_path / "audit"
    audit_health_sessions(
        CONTRACT,
        (
            AuditSession("train-a", "train_live", train_a),
            AuditSession("train-b", "train_live", train_b),
            AuditSession("dev-death", "dev_death", dev),
            AuditSession("challenge", "challenge_false_positive", challenge),
        ),
        audit_dir,
    )
    output = tmp_path / "replay"
    report = replay_health_event_transitions(CONTRACT, audit_dir / "report.json", dev, output)
    assert report["status"] == "DEATH_RESPAWN_EVENT_TRANSITION_REPLAY_PASSED"
    assert report["frames"] == 10 and report["transitions"] == 9
    assert report["training_eligible_transitions"] == 0
    assert report["event_counts"]["DEATH"] == report["event_counts"]["RESPAWN"] == 1
    assert report["reward_total"] == 0.0 and all(report["checks"].values())
    assert len(list((output / "frames").glob("*.npz"))) == 10
    with UnifiedTransitionStore(output / "replay.sqlite3") as store:
        rows = store.load_episode("offline-death-respawn-001")
    assert all(row["causal_order_valid"] and not row["training_eligible"] for row in rows)
    assert all(row["reward"]["total"] == 0 and row["reward"]["event_ids"] == [] for row in rows)
    assert [row["done"] for row in rows] == [False] * 8 + [True]
    assert rows[-1]["terminal_reason"] == "VIDEO_EOF"
    with pytest.raises(E1Error, match="already exists"):
        replay_health_event_transitions(CONTRACT, audit_dir / "report.json", dev, output)
    shard.write_bytes(b"tampered")
    with pytest.raises(E1Error, match="shard differs"):
        replay_health_event_transitions(
            CONTRACT, audit_dir / "report.json", dev, tmp_path / "tampered-output"
        )


def test_existing_health_candidate_audit_freezes_insufficient_positive_sessions(
    tmp_path: Path,
) -> None:
    positive = tmp_path / "positive"
    negative_a = tmp_path / "negative-a"
    negative_b = tmp_path / "negative-b"
    negative_c = tmp_path / "negative-c"
    _write_session(
        positive,
        [12, 12, 12, 0, 0, 0, 0, 12, 12, 12],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    )
    for path in (negative_a, negative_b, negative_c):
        _write_session(path, [12] * 10, [0] * 10)
    output = tmp_path / "candidate-audit"
    report = audit_existing_health_candidates(
        CONTRACT,
        tuple(
            AuditSession(path.name, "challenge_false_positive", path)
            for path in (positive, negative_a, negative_b, negative_c)
        ),
        output,
    )
    assert report["status"] == "DEATH_RESPAWN_CANDIDATES_INSUFFICIENT"
    assert report["positive_sessions"] == 1
    assert report["negative_sessions"] == 3
    assert report["checks"] == {
        "minimum_positive_sessions": False,
        "minimum_negative_sessions": True,
        "no_unpaired_sessions": True,
        "no_death_on_zero_hard_stop_sessions": True,
    }
    assert report["reward_allowed"] is report["training_allowed"] is False
    assert str(tmp_path) not in (output / "report.json").read_text()
    with pytest.raises(E1Error, match="already exists"):
        audit_existing_health_candidates(CONTRACT, (), output)
