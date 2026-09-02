from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import av
import numpy as np

from hok_agent import pre_ingest

CONTRACT_SCHEMA = "hok-agent-hierarchical-event-e1-terminal-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1-terminal-audit-v1"
COHORT_SCHEMA = "hok-agent-v5-automatic-cohort-v1"


class TerminalAuditError(ValueError):
    pass


class TerminalLabel(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    GAME_END_UNKNOWN = "GAME_END_UNKNOWN"
    WIN = "WIN"
    LOSS = "LOSS"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class OcrToken:
    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TerminalFrameEvidence:
    label: TerminalLabel
    allowlisted_hits: tuple[str, ...]
    result_fragment_count: int


@dataclass(frozen=True, slots=True)
class TerminalAuditConfig:
    train_sessions: int
    dev_sessions: int
    tail_seconds: int
    sample_hz: int
    minimum_confidence: float
    win_tokens: tuple[str, ...]
    loss_tokens: tuple[str, ...]
    result_fragments: tuple[str, ...]
    minimum_result_fragments_per_frame: int
    minimum_train_game_end_coverage: float
    minimum_dev_game_end_coverage: float
    minimum_train_outcome_coverage: float
    minimum_dev_outcome_coverage: float
    maximum_outcome_conflicts: int
    require_both_outcome_classes: bool


class OcrResult(Protocol):
    ocr_text: str
    score: float


class OcrSystem(Protocol):
    def detect_and_ocr(self, image: np.ndarray) -> list[OcrResult]: ...


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_terminal_contract(path: Path) -> tuple[TerminalAuditConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise TerminalAuditError("terminal contract schema differs")
    selection = cast(dict[str, object], payload["selection"])
    sampling = cast(dict[str, object], payload["sampling"])
    ocr = cast(dict[str, object], payload["ocr"])
    gate = cast(dict[str, object], payload["coverage_gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("status") != "frozen_weak_label_coverage_audit"
        or selection.get("test_allowed") is not False
        or sampling.get("time_is_not_a_label") is not True
        or ocr.get("arbitrary_ocr_text_persisted") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
        or claim.get("video_test_allowed") is not False
    ):
        raise TerminalAuditError("terminal contract boundary differs")
    config = TerminalAuditConfig(
        train_sessions=int(cast(int, selection["train_sessions"])),
        dev_sessions=int(cast(int, selection["dev_sessions"])),
        tail_seconds=int(cast(int, sampling["tail_seconds"])),
        sample_hz=int(cast(int, sampling["sample_hz"])),
        minimum_confidence=float(cast(float, ocr["minimum_confidence"])),
        win_tokens=tuple(cast(list[str], ocr["win_tokens"])),
        loss_tokens=tuple(cast(list[str], ocr["loss_tokens"])),
        result_fragments=tuple(cast(list[str], ocr["result_fragments"])),
        minimum_result_fragments_per_frame=int(
            cast(int, ocr["minimum_result_fragments_per_frame"])
        ),
        minimum_train_game_end_coverage=float(
            cast(float, gate["minimum_train_game_end_coverage"])
        ),
        minimum_dev_game_end_coverage=float(
            cast(float, gate["minimum_dev_game_end_coverage"])
        ),
        minimum_train_outcome_coverage=float(
            cast(float, gate["minimum_train_outcome_coverage"])
        ),
        minimum_dev_outcome_coverage=float(cast(float, gate["minimum_dev_outcome_coverage"])),
        maximum_outcome_conflicts=int(cast(int, gate["maximum_outcome_conflicts"])),
        require_both_outcome_classes=bool(gate["require_both_outcome_classes"]),
    )
    if (
        min(config.train_sessions, config.dev_sessions, config.tail_seconds, config.sample_hz) <= 0
        or not 0.0 <= config.minimum_confidence <= 1.0
        or config.minimum_result_fragments_per_frame <= 0
    ):
        raise TerminalAuditError("terminal contract values are invalid")
    return config, payload, _sha(_canonical(payload))


def classify_ocr_tokens(
    tokens: tuple[OcrToken, ...],
    config: TerminalAuditConfig,
) -> TerminalFrameEvidence:
    accepted = [
        token.text.strip().upper()
        for token in tokens
        if token.confidence >= config.minimum_confidence
    ]
    win_hits = {token for token in accepted if token in config.win_tokens}
    loss_hits = {token for token in accepted if token in config.loss_tokens}
    fragment_hits = {
        fragment
        for fragment in config.result_fragments
        if any(fragment.upper() in text for text in accepted)
    }
    allowlisted = tuple(sorted({*win_hits, *loss_hits, *fragment_hits}))
    if win_hits and loss_hits:
        label = TerminalLabel.CONFLICT
    elif win_hits:
        label = TerminalLabel.WIN
    elif loss_hits:
        label = TerminalLabel.LOSS
    elif len(fragment_hits) >= config.minimum_result_fragments_per_frame:
        label = TerminalLabel.GAME_END_UNKNOWN
    else:
        label = TerminalLabel.IN_PROGRESS
    return TerminalFrameEvidence(label, allowlisted, len(fragment_hits))


def _load_cohort(path: Path, evidence: pre_ingest.PreIngestEvidence) -> dict[str, str]:
    data = path.read_bytes()
    payload = cast(dict[str, object], json.loads(data))
    supplied = payload.get("cohort_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "cohort_sha256"}
    session_splits = cast(dict[str, str], payload.get("session_splits"))
    if (
        payload.get("schema_version") != COHORT_SCHEMA
        or payload.get("pre_ingest_sha256") != evidence.pre_ingest_sha256
        or supplied != _sha(_canonical(unsigned))
        or data != _canonical(payload) + b"\n"
        or set(session_splits) != set(evidence.component_of)
        or any(split not in {"train", "dev", "test"} for split in session_splits.values())
    ):
        raise TerminalAuditError("terminal cohort is invalid")
    return session_splits


def _candidate_paths(
    raw_root: Path,
    evidence: pre_ingest.PreIngestEvidence,
) -> dict[str, Path]:
    paths = pre_ingest._scan(raw_root)
    candidates = pre_ingest._candidate_list(paths, raw_root)
    mapped = {
        candidate.candidate_id: path
        for path, candidate in zip(paths, candidates, strict=True)
    }
    if set(mapped) != set(evidence.component_of):
        raise TerminalAuditError("raw videos differ from pre-ingest evidence")
    return mapped


def _select_sessions(
    session_splits: dict[str, str],
    contract_sha256: str,
    train_count: int,
    dev_count: int,
) -> dict[str, tuple[str, ...]]:
    selected: dict[str, tuple[str, ...]] = {}
    for split, count in (("train", train_count), ("dev", dev_count)):
        candidates = [candidate for candidate, value in session_splits.items() if value == split]
        ordered = sorted(
            candidates,
            key=lambda candidate: _sha(f"{contract_sha256}:{candidate}".encode()),
        )
        if len(ordered) < count:
            raise TerminalAuditError(f"terminal cohort has too few {split} sessions")
        selected[split] = tuple(ordered[:count])
    return selected


def _tail_frames(path: Path, config: TerminalAuditConfig) -> tuple[np.ndarray, ...]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is None or stream.time_base is None:
            raise TerminalAuditError("terminal audit video has no duration")
        target = max(0, stream.duration - int(config.tail_seconds / float(stream.time_base)))
        container.seek(target, stream=stream, backward=True, any_frame=False)
        next_sample_s: float | None = None
        interval = 1.0 / config.sample_hz
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            timestamp_s = float(frame.pts * frame.time_base)
            if next_sample_s is None or timestamp_s >= next_sample_s:
                frames.append(frame.to_ndarray(format="bgr24"))
                next_sample_s = timestamp_s + interval
    if not frames:
        raise TerminalAuditError("terminal audit decoded no frames")
    return tuple(frames[-config.tail_seconds * config.sample_hz :])


def _ocr_system() -> OcrSystem:
    try:
        module = importlib.import_module("ppocronnx")
    except ImportError as exc:
        raise TerminalAuditError("ppocr-onnx optional dependency is unavailable") from exc
    factory = cast(Callable[[], object], module.TextSystem)
    return cast(OcrSystem, factory())


def _audit_session(
    candidate_id: str,
    split: str,
    path: Path,
    config: TerminalAuditConfig,
    contract_sha256: str,
    ocr: OcrSystem,
) -> dict[str, object]:
    frames = _tail_frames(path, config)
    labels: list[TerminalLabel] = []
    token_counts: Counter[str] = Counter()
    for frame in frames:
        results = ocr.detect_and_ocr(frame)
        evidence = classify_ocr_tokens(
            tuple(OcrToken(item.ocr_text, float(item.score)) for item in results),
            config,
        )
        labels.append(evidence.label)
        token_counts.update(evidence.allowlisted_hits)
    wins = labels.count(TerminalLabel.WIN)
    losses = labels.count(TerminalLabel.LOSS)
    conflicts = labels.count(TerminalLabel.CONFLICT)
    result_frames = labels.count(TerminalLabel.GAME_END_UNKNOWN)
    if conflicts or (wins and losses):
        outcome = TerminalLabel.CONFLICT
    elif wins:
        outcome = TerminalLabel.WIN
    elif losses:
        outcome = TerminalLabel.LOSS
    elif result_frames:
        outcome = TerminalLabel.GAME_END_UNKNOWN
    else:
        outcome = TerminalLabel.IN_PROGRESS
    game_end_detected = bool(wins or losses or result_frames)
    return {
        "session_id": _sha(f"e1-terminal:{contract_sha256}:{candidate_id}".encode()),
        "split": split,
        "sampled_frames": len(frames),
        "game_end_detected": game_end_detected,
        "outcome": outcome.value,
        "win_frames": wins,
        "loss_frames": losses,
        "conflict_frames": conflicts,
        "result_screen_frames": result_frames,
        "allowlisted_token_counts": dict(sorted(token_counts.items())),
        "source_locator_persisted": False,
        "arbitrary_ocr_text_persisted": False,
    }


def _coverage(rows: list[dict[str, object]], split: str) -> tuple[float, float]:
    selected = [row for row in rows if row["split"] == split]
    game_end = sum(bool(row["game_end_detected"]) for row in selected)
    outcome = sum(row["outcome"] in {"WIN", "LOSS"} for row in selected)
    return game_end / len(selected), outcome / len(selected)


def audit_terminal_coverage(
    contract_path: Path,
    raw_root: Path,
    pre_ingest_path: Path,
    cohort_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, _contract, contract_sha256 = load_terminal_contract(contract_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise TerminalAuditError("terminal output directory already exists")
    evidence = pre_ingest.load_pre_ingest(pre_ingest_path)
    session_splits = _load_cohort(cohort_path, evidence)
    paths = _candidate_paths(raw_root, evidence)
    selected = _select_sessions(
        session_splits,
        contract_sha256,
        config.train_sessions,
        config.dev_sessions,
    )
    ocr = _ocr_system()
    rows: list[dict[str, object]] = []
    for split in ("train", "dev"):
        for candidate_id in selected[split]:
            row = _audit_session(
                candidate_id,
                split,
                paths[candidate_id],
                config,
                contract_sha256,
                ocr,
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "terminal_session_complete": row["session_id"],
                        "split": split,
                        "game_end_detected": row["game_end_detected"],
                        "outcome": row["outcome"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    train_game_end, train_outcome = _coverage(rows, "train")
    dev_game_end, dev_outcome = _coverage(rows, "dev")
    conflicts = sum(row["outcome"] == TerminalLabel.CONFLICT.value for row in rows)
    outcomes = {cast(str, row["outcome"]) for row in rows}
    checks = {
        "train_game_end_coverage": train_game_end >= config.minimum_train_game_end_coverage,
        "dev_game_end_coverage": dev_game_end >= config.minimum_dev_game_end_coverage,
        "train_outcome_coverage": train_outcome >= config.minimum_train_outcome_coverage,
        "dev_outcome_coverage": dev_outcome >= config.minimum_dev_outcome_coverage,
        "outcome_conflicts": conflicts <= config.maximum_outcome_conflicts,
        "both_outcome_classes": (
            {"WIN", "LOSS"}.issubset(outcomes) if config.require_both_outcome_classes else True
        ),
    }
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "E1B_WEAK_LABEL_COVERAGE_PASSED"
            if all(checks.values())
            else "E1B_WEAK_LABEL_COVERAGE_FAILED"
        ),
        "contract_sha256": contract_sha256,
        "pre_ingest_sha256": evidence.pre_ingest_sha256,
        "selected_train_sessions": config.train_sessions,
        "selected_dev_sessions": config.dev_sessions,
        "rows": rows,
        "metrics": {
            "train_game_end_coverage": train_game_end,
            "dev_game_end_coverage": dev_game_end,
            "train_outcome_coverage": train_outcome,
            "dev_outcome_coverage": dev_outcome,
            "outcome_conflicts": conflicts,
        },
        "checks": checks,
        "weak_label_accuracy_verified": False,
        "semantic_accuracy_verified": False,
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


def load_terminal_report(report_path: Path, contract_path: Path) -> dict[str, object]:
    if report_path.is_symlink() or not report_path.is_file():
        raise TerminalAuditError("terminal report must be a regular file")
    data = report_path.read_bytes()
    payload = cast(dict[str, object], json.loads(data))
    _config, _contract, contract_sha256 = load_terminal_contract(contract_path)
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
        raise TerminalAuditError("terminal report is invalid")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Hierarchical Event E1 terminal audit")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--pre-ingest", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit_terminal_coverage(
        args.contract,
        args.raw_root,
        args.pre_ingest,
        args.cohort,
        args.output_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
