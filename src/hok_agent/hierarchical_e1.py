from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from hok_agent.visual_events import (
    EventEngineIdentity,
    EventSource,
    ExactOnceEventFusion,
    LifeState,
    MatchState,
    VisualEvent,
    VisualEventType,
    VisualState,
)

CONTRACT_SCHEMA = "hok-agent-hierarchical-event-e1-health-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1-health-audit-v1"

SessionRole = Literal["train_live", "dev_death", "challenge_false_positive"]


class E1Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HealthBarConfig:
    minimum_green: int
    minimum_green_minus_red: int
    minimum_green_minus_blue: int
    minimum_y: int
    maximum_y_exclusive: int
    minimum_run_midpoint_x: float
    maximum_run_midpoint_x: float
    minimum_bar_run_px: int
    maximum_bar_run_px: int
    full_health_run_px: int
    alive_confirmation_frames: int
    death_confirmation_frames: int
    respawn_confirmation_frames: int
    minimum_hp_delta: float


@dataclass(frozen=True, slots=True)
class HealthBarEvidence:
    visible: bool
    hp_ratio: float | None
    run_length_px: int
    confidence: float


@dataclass(frozen=True, slots=True)
class HealthUpdate:
    state: VisualState
    events: tuple[VisualEvent, ...]


@dataclass(frozen=True, slots=True)
class AuditSession:
    session_id: str
    role: SessionRole
    directory: Path


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_health_contract(path: Path) -> tuple[HealthBarConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise E1Error("E1 health contract schema differs")
    green = cast(dict[str, object], payload["green_mask"])
    search = cast(dict[str, object], payload["search"])
    temporal = cast(dict[str, object], payload["temporal"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    repair = cast(dict[str, object], payload["repair_history"])
    if (
        payload.get("status") != "exploratory_dev_seen_engineering_diagnostic"
        or repair.get("repairs_allowed") != 1
        or repair.get("repairs_used") != 1
        or repair.get("gate_thresholds_changed") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
        or claim.get("video_test_allowed") is not False
    ):
        raise E1Error("E1 health diagnostic claim boundary differs")
    config = HealthBarConfig(
        minimum_green=int(cast(int, green["minimum_green"])),
        minimum_green_minus_red=int(cast(int, green["minimum_green_minus_red"])),
        minimum_green_minus_blue=int(cast(int, green["minimum_green_minus_blue"])),
        minimum_y=int(cast(int, search["minimum_y"])),
        maximum_y_exclusive=int(cast(int, search["maximum_y_exclusive"])),
        minimum_run_midpoint_x=float(cast(int, search["minimum_run_midpoint_x"])),
        maximum_run_midpoint_x=float(cast(int, search["maximum_run_midpoint_x"])),
        minimum_bar_run_px=int(cast(int, search["minimum_bar_run_px"])),
        maximum_bar_run_px=int(cast(int, search["maximum_bar_run_px"])),
        full_health_run_px=int(cast(int, search["full_health_run_px"])),
        alive_confirmation_frames=int(cast(int, temporal["alive_confirmation_frames"])),
        death_confirmation_frames=int(cast(int, temporal["death_confirmation_frames"])),
        respawn_confirmation_frames=int(cast(int, temporal["respawn_confirmation_frames"])),
        minimum_hp_delta=float(cast(float, temporal["minimum_hp_delta"])),
    )
    if (
        config.minimum_y < 0
        or config.maximum_y_exclusive > 128
        or config.minimum_y >= config.maximum_y_exclusive
        or config.minimum_bar_run_px <= 0
        or config.minimum_bar_run_px > config.maximum_bar_run_px
        or config.full_health_run_px <= 0
        or min(
            config.alive_confirmation_frames,
            config.death_confirmation_frames,
            config.respawn_confirmation_frames,
        )
        <= 0
    ):
        raise E1Error("E1 health contract values are invalid")
    return config, payload, _sha(_canonical(payload))


def load_health_report(report_path: Path, contract_path: Path) -> dict[str, object]:
    if report_path.is_symlink() or not report_path.is_file():
        raise E1Error("E1 health report must be a regular file")
    data = report_path.read_bytes()
    payload = cast(dict[str, object], json.loads(data))
    _config, _contract, contract_sha256 = load_health_contract(contract_path)
    supplied = payload.get("report_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    if (
        payload.get("schema_version") != REPORT_SCHEMA
        or payload.get("contract_sha256") != contract_sha256
        or supplied != _sha(_canonical(unsigned))
        or data != _canonical(payload) + b"\n"
        or payload.get("reward_allowed") is not False
        or payload.get("promotion_allowed") is not False
        or payload.get("video_test_opened") is not False
    ):
        raise E1Error("E1 health report is invalid")
    return payload


def detect_center_health_bar(frame: np.ndarray, config: HealthBarConfig) -> HealthBarEvidence:
    if frame.shape != (128, 128, 3) or frame.dtype != np.uint8:
        raise E1Error("health detector requires uint8 128x128x3 RGB")
    rgb = frame.astype(np.int16)
    green = (
        (rgb[:, :, 1] >= config.minimum_green)
        & (rgb[:, :, 1] - rgb[:, :, 0] >= config.minimum_green_minus_red)
        & (rgb[:, :, 1] - rgb[:, :, 2] >= config.minimum_green_minus_blue)
    )
    candidates: list[tuple[int, float]] = []
    for row in green[config.minimum_y : config.maximum_y_exclusive]:
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends, strict=True):
            length = int(end - start)
            midpoint = float(start + end) / 2.0
            if (
                config.minimum_bar_run_px <= length <= config.maximum_bar_run_px
                and config.minimum_run_midpoint_x <= midpoint <= config.maximum_run_midpoint_x
            ):
                candidates.append((length, midpoint))
    if not candidates:
        return HealthBarEvidence(False, None, 0, 0.8)
    length, midpoint = max(
        candidates,
        key=lambda item: (item[0], -abs(item[1] - 64.0)),
    )
    ratio = min(1.0, length / config.full_health_run_px)
    center_confidence = max(0.0, 1.0 - abs(midpoint - 64.0) / 32.0)
    confidence = min(1.0, 0.5 + 0.5 * center_confidence)
    return HealthBarEvidence(True, ratio, length, confidence)


class HealthTemporalEventEngine:
    def __init__(
        self,
        episode_id: str,
        identity: EventEngineIdentity,
        config: HealthBarConfig,
    ) -> None:
        if not episode_id:
            raise E1Error("episode_id must not be empty")
        self.episode_id = episode_id
        self.config = config
        self.fusion = ExactOnceEventFusion(identity)
        self._life_state = LifeState.UNKNOWN
        self._present_streak = 0
        self._absent_streak = 0
        self._pending_start_ns: int | None = None
        self._previous_ratio: float | None = None
        self._previous_timestamp_ns: int | None = None
        self._event_sequence = 0

    def _event(
        self,
        event_type: VisualEventType,
        start_ns: int,
        end_ns: int,
        old_value: float | str | bool | None,
        new_value: float | str | bool | None,
        delta: float | None,
        confidence: float,
    ) -> VisualEvent:
        self._event_sequence += 1
        event_id = f"{self.episode_id}:{self._event_sequence:06d}:{event_type.value}"
        return VisualEvent(
            event_id=event_id,
            event_type=event_type,
            start_ns=start_ns,
            end_ns=end_ns,
            old_value=old_value,
            new_value=new_value,
            delta=delta,
            confidence=confidence,
            roi_id="main_center_health_bar",
            dedup_key=event_id,
            source=EventSource.RGB_FUSION,
        )

    def update(
        self,
        observation_id: str,
        timestamp_ns: int,
        evidence: HealthBarEvidence,
    ) -> HealthUpdate:
        emitted: list[VisualEvent] = []
        previous_life_state = self._life_state
        if evidence.visible:
            self._present_streak += 1
            self._absent_streak = 0
            if self._present_streak == 1:
                self._pending_start_ns = timestamp_ns
            if self._life_state == LifeState.UNKNOWN:
                if self._present_streak >= self.config.alive_confirmation_frames:
                    self._life_state = LifeState.ALIVE
            elif (
                self._life_state == LifeState.DEAD
                and self._present_streak >= self.config.respawn_confirmation_frames
            ):
                event = self._event(
                    VisualEventType.RESPAWN,
                    cast(int, self._pending_start_ns),
                    timestamp_ns,
                    LifeState.DEAD.value,
                    LifeState.ALIVE.value,
                    None,
                    evidence.confidence,
                )
                if self.fusion.accept(event):
                    emitted.append(event)
                self._life_state = LifeState.ALIVE
                self._previous_ratio = None
        else:
            self._absent_streak += 1
            self._present_streak = 0
            if self._absent_streak == 1:
                self._pending_start_ns = timestamp_ns
            if (
                self._life_state == LifeState.ALIVE
                and self._absent_streak >= self.config.death_confirmation_frames
            ):
                event = self._event(
                    VisualEventType.DEATH,
                    cast(int, self._pending_start_ns),
                    timestamp_ns,
                    LifeState.ALIVE.value,
                    LifeState.DEAD.value,
                    None,
                    evidence.confidence,
                )
                if self.fusion.accept(event):
                    emitted.append(event)
                self._life_state = LifeState.DEAD
                self._previous_ratio = None

        if (
            evidence.visible
            and evidence.hp_ratio is not None
            and self._life_state == LifeState.ALIVE
            and previous_life_state != LifeState.DEAD
        ):
            if (
                self._previous_ratio is not None
                and abs(evidence.hp_ratio - self._previous_ratio) >= self.config.minimum_hp_delta
            ):
                event = self._event(
                    VisualEventType.SELF_HP_DELTA,
                    cast(int, self._previous_timestamp_ns),
                    timestamp_ns,
                    self._previous_ratio,
                    evidence.hp_ratio,
                    evidence.hp_ratio - self._previous_ratio,
                    evidence.confidence,
                )
                if self.fusion.accept(event):
                    emitted.append(event)
            self._previous_ratio = evidence.hp_ratio
            self._previous_timestamp_ns = timestamp_ns

        reported_life = self._life_state
        if self._life_state == LifeState.DEAD and evidence.visible:
            reported_life = LifeState.RESPAWNING
        elif self._life_state == LifeState.ALIVE and not evidence.visible:
            reported_life = LifeState.UNKNOWN
        state = VisualState(
            observation_id=observation_id,
            timestamp_ns=timestamp_ns,
            match_state=MatchState.UNKNOWN,
            life_state=reported_life,
            self_hp_ratio=evidence.hp_ratio,
            match_confidence=0.0,
            life_confidence=evidence.confidence,
            self_hp_confidence=evidence.confidence if evidence.visible else 0.0,
        )
        return HealthUpdate(state, tuple(emitted))


def _load_session_frames(session: AuditSession) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if session.directory.is_symlink() or not session.directory.is_dir():
        raise E1Error("audit session must be a regular directory")
    frames: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    hard_stops: list[np.ndarray] = []
    shards = sorted(session.directory.rglob("observations-*.npz"))
    if not shards:
        raise E1Error("audit session has no observation shards")
    for shard in shards:
        if shard.is_symlink() or not shard.is_file():
            raise E1Error("audit observation shard must be a regular file")
        with np.load(shard, allow_pickle=False) as data:
            frames.append(data["main_rgb"])
            timestamps.append(data["scheduled_elapsed_ms"])
            hard_stops.append(data["hard_stop"])
    return (
        np.concatenate(frames),
        np.concatenate(timestamps),
        np.concatenate(hard_stops).astype(bool),
    )


def _session_report(
    session: AuditSession,
    config: HealthBarConfig,
    identity: EventEngineIdentity,
) -> dict[str, object]:
    frames, timestamps_ms, hard_stops = _load_session_frames(session)
    engine = HealthTemporalEventEngine(session.session_id, identity, config)
    visible = 0
    visible_on_legacy_stop = 0
    events: list[VisualEvent] = []
    for sequence, (frame, timestamp_ms, hard_stop) in enumerate(
        zip(frames, timestamps_ms, hard_stops, strict=True)
    ):
        evidence = detect_center_health_bar(frame, config)
        visible += int(evidence.visible)
        visible_on_legacy_stop += int(evidence.visible and hard_stop)
        update = engine.update(
            f"{session.session_id}:{sequence:06d}",
            int(timestamp_ms) * 1_000_000,
            evidence,
        )
        events.extend(update.events)
    event_counts = {
        event_type.value: sum(event.event_type == event_type for event in events)
        for event_type in (
            VisualEventType.DEATH,
            VisualEventType.RESPAWN,
            VisualEventType.SELF_HP_DELTA,
        )
    }
    summary_path = session.directory / "summary.json"
    summary_sha256 = _sha(summary_path.read_bytes()) if summary_path.is_file() else None
    return {
        "session_id": session.session_id,
        "role": session.role,
        "summary_sha256": summary_sha256,
        "frames": len(frames),
        "health_visible_frames": visible,
        "health_visibility_coverage": visible / len(frames),
        "legacy_hard_stop_frames": int(hard_stops.sum()),
        "health_visible_on_legacy_hard_stop_frames": visible_on_legacy_stop,
        "event_counts": event_counts,
        "source_locator_persisted": False,
    }


def audit_health_sessions(
    contract_path: Path,
    sessions: tuple[AuditSession, ...],
    output_dir: Path,
) -> dict[str, object]:
    config, contract, contract_sha256 = load_health_contract(contract_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise E1Error("audit output directory already exists")
    roles = [session.role for session in sessions]
    if roles.count("train_live") < 2 or roles.count("dev_death") < 1 or roles.count(
        "challenge_false_positive"
    ) < 1:
        raise E1Error("audit requires two train-live, one dev-death, and one challenge session")
    identity = EventEngineIdentity("hierarchical-e1-health-v1", contract_sha256)
    rows = [_session_report(session, config, identity) for session in sessions]
    train_rows = [row for row in rows if row["role"] == "train_live"]
    dev_rows = [row for row in rows if row["role"] == "dev_death"]
    challenge_rows = [row for row in rows if row["role"] == "challenge_false_positive"]
    train_frames = sum(cast(int, row["frames"]) for row in train_rows)
    train_visible = sum(cast(int, row["health_visible_frames"]) for row in train_rows)
    train_false_deaths = sum(
        cast(dict[str, int], row["event_counts"])[VisualEventType.DEATH.value]
        for row in train_rows
    )
    dev_deaths = sum(
        cast(dict[str, int], row["event_counts"])[VisualEventType.DEATH.value]
        for row in dev_rows
    )
    dev_respawns = sum(
        cast(dict[str, int], row["event_counts"])[VisualEventType.RESPAWN.value]
        for row in dev_rows
    )
    challenge_false_deaths = sum(
        cast(dict[str, int], row["event_counts"])[VisualEventType.DEATH.value]
        for row in challenge_rows
    )
    gate = cast(dict[str, object], contract["diagnostic_gate"])
    checks = {
        "train_live_visibility": train_visible / train_frames
        >= float(cast(float, gate["minimum_train_live_visibility_coverage"])),
        "train_false_deaths": train_false_deaths
        <= int(cast(int, gate["maximum_train_false_death_events"])),
        "dev_death_events": dev_deaths >= int(cast(int, gate["minimum_dev_death_events"])),
        "dev_respawn_events": dev_respawns
        >= int(cast(int, gate["minimum_dev_respawn_events"])),
        "challenge_false_deaths": challenge_false_deaths
        <= int(cast(int, gate["maximum_challenge_false_death_events"])),
    }
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "E1A_ENGINEERING_DIAGNOSTIC_PASSED"
            if all(checks.values())
            else "E1A_ENGINEERING_DIAGNOSTIC_FAILED"
        ),
        "contract_sha256": contract_sha256,
        "sessions": rows,
        "metrics": {
            "train_live_visibility_coverage": train_visible / train_frames,
            "train_false_death_events": train_false_deaths,
            "dev_death_events": dev_deaths,
            "dev_respawn_events": dev_respawns,
            "challenge_false_death_events": challenge_false_deaths,
        },
        "checks": checks,
        "semantic_accuracy_verified": False,
        "self_hp_numeric_accuracy_verified": False,
        "reward_allowed": False,
        "promotion_allowed": False,
        "video_test_opened": False,
        "mobile_capture_used": False,
        "device_input_used": False,
    }
    payload["report_sha256"] = _sha(_canonical(payload))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        report = staging / "report.json"
        report.write_bytes(_canonical(payload) + b"\n")
        with report.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def _session(raw: str, role: SessionRole) -> AuditSession:
    session_id, separator, path = raw.partition("=")
    if not separator or not session_id or "/" in session_id:
        raise E1Error("session argument must be ID=/absolute/directory")
    return AuditSession(session_id, role, Path(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Hierarchical Event E1 health audit")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--train-live", action="append", default=[])
    parser.add_argument("--dev-death", action="append", default=[])
    parser.add_argument("--challenge", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sessions = tuple(
        [*(_session(value, "train_live") for value in args.train_live)]
        + [*(_session(value, "dev_death") for value in args.dev_death)]
        + [*(_session(value, "challenge_false_positive") for value in args.challenge)]
    )
    payload = audit_health_sessions(args.contract, sessions, args.output_dir)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
