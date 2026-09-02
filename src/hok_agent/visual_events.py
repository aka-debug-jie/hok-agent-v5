from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, TypedDict

EventValue: TypeAlias = float | str | bool | None


class MatchState(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    WIN = "WIN"
    LOSS = "LOSS"
    UNKNOWN = "UNKNOWN"


class LifeState(StrEnum):
    ALIVE = "ALIVE"
    DEAD = "DEAD"
    RESPAWNING = "RESPAWNING"
    UNKNOWN = "UNKNOWN"


class VisualEventType(StrEnum):
    GAME_END = "GAME_END"
    WIN = "WIN"
    LOSS = "LOSS"
    DEATH = "DEATH"
    RESPAWN = "RESPAWN"
    SELF_HP_DELTA = "SELF_HP_DELTA"
    TOWER_DAMAGE = "TOWER_DAMAGE"
    TOWER_DESTROYED = "TOWER_DESTROYED"


class EventSource(StrEnum):
    RGB_RULE = "rgb_rule"
    RGB_MODEL = "rgb_model"
    RGB_OCR = "rgb_ocr"
    RGB_FUSION = "rgb_fusion"


class VisualEventRecord(TypedDict):
    event_id: str
    event_type: str
    start_ns: int
    end_ns: int
    old_value: EventValue
    new_value: EventValue
    delta: float | None
    confidence: float
    roi_id: str
    dedup_key: str
    source: str


@dataclass(frozen=True, slots=True)
class EventEngineIdentity:
    version: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("EventEngine version must not be empty")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("EventEngine sha256 must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class VisualState:
    observation_id: str
    timestamp_ns: int
    match_state: MatchState
    life_state: LifeState
    self_hp_ratio: float | None
    match_confidence: float
    life_confidence: float
    self_hp_confidence: float

    def __post_init__(self) -> None:
        if not self.observation_id or self.timestamp_ns < 0:
            raise ValueError("VisualState identity and timestamp are required")
        for confidence in (
            self.match_confidence,
            self.life_confidence,
            self.self_hp_confidence,
        ):
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("VisualState confidence must be within [0, 1]")
        if self.self_hp_ratio is not None and not 0.0 <= self.self_hp_ratio <= 1.0:
            raise ValueError("self_hp_ratio must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class VisualEvent:
    event_id: str
    event_type: VisualEventType
    start_ns: int
    end_ns: int
    old_value: EventValue
    new_value: EventValue
    delta: float | None
    confidence: float
    roi_id: str
    dedup_key: str
    source: EventSource

    def __post_init__(self) -> None:
        if not self.event_id or not self.roi_id or not self.dedup_key:
            raise ValueError("VisualEvent identity, ROI, and dedup key are required")
        if self.start_ns < 0 or self.end_ns < self.start_ns:
            raise ValueError("VisualEvent timestamps are invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("VisualEvent confidence must be within [0, 1]")

    def to_record(self) -> VisualEventRecord:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "delta": self.delta,
            "confidence": self.confidence,
            "roi_id": self.roi_id,
            "dedup_key": self.dedup_key,
            "source": self.source.value,
        }


class ExactOnceEventFusion:
    """Episode-local event deduplication without detector or reward logic."""

    def __init__(self, identity: EventEngineIdentity) -> None:
        self.identity = identity
        self._event_ids: set[str] = set()
        self._dedup_keys: set[str] = set()
        self._accepted: list[VisualEvent] = []

    def accept(self, event: VisualEvent) -> bool:
        if event.event_id in self._event_ids or event.dedup_key in self._dedup_keys:
            return False
        self._event_ids.add(event.event_id)
        self._dedup_keys.add(event.dedup_key)
        self._accepted.append(event)
        return True

    @property
    def accepted(self) -> tuple[VisualEvent, ...]:
        return tuple(self._accepted)

    def reset_episode(self) -> None:
        self._event_ids.clear()
        self._dedup_keys.clear()
        self._accepted.clear()

