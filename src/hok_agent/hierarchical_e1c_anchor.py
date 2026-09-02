from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import av
import numpy as np

from hok_agent import pre_ingest
from hok_agent.hierarchical_e1_terminal import (
    OcrSystem,
    OcrToken,
    TerminalLabel,
    _candidate_paths,
    _load_cohort,
    _ocr_system,
)

CONTRACT_SCHEMA = "hok-agent-hierarchical-event-e1c-anchor-config-v1"
REPORT_SCHEMA = "hok-agent-hierarchical-event-e1c-anchor-report-v1"


class E1cAnchorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AnchorConfig:
    train_sessions_expected: int
    dev_sessions_expected: int
    minimum_confidence: float
    win_tokens: tuple[str, ...]
    loss_tokens: tuple[str, ...]
    result_fragments: tuple[str, ...]
    minimum_result_fragments: int
    minimum_train_anchor_sessions: int
    minimum_dev_anchor_sessions: int
    maximum_outcome_conflicts: int


@dataclass(frozen=True, slots=True)
class AnchorEvidence:
    anchored: bool
    outcome: TerminalLabel
    allowlisted_hits: tuple[str, ...]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_anchor_contract(path: Path) -> tuple[AnchorConfig, dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise E1cAnchorError("E1c anchor contract schema differs")
    scope = cast(dict[str, object], payload["scope"])
    selection = cast(dict[str, object], payload["frame_selection"])
    ocr = cast(dict[str, object], payload["ocr"])
    gate = cast(dict[str, object], payload["preflight_gate"])
    claim = cast(dict[str, object], payload["claim_boundary"])
    if (
        payload.get("status") != "frozen_result_page_anchor_preflight"
        or scope.get("test_allowed") is not False
        or scope.get("all_train_dev_sessions_required") is not True
        or selection.get("video_end_is_label") is not False
        or selection.get("anchor_frame_allowed_in_future_model_input") is not False
        or ocr.get("arbitrary_ocr_text_persisted") is not False
        or claim.get("anchor_is_win_loss_truth") is not False
        or claim.get("reward_allowed") is not False
        or claim.get("promotion_allowed") is not False
        or claim.get("video_test_allowed") is not False
    ):
        raise E1cAnchorError("E1c anchor boundary differs")
    config = AnchorConfig(
        train_sessions_expected=int(cast(int, scope["train_sessions_expected"])),
        dev_sessions_expected=int(cast(int, scope["dev_sessions_expected"])),
        minimum_confidence=float(cast(float, ocr["minimum_confidence"])),
        win_tokens=tuple(cast(list[str], ocr["win_tokens"])),
        loss_tokens=tuple(cast(list[str], ocr["loss_tokens"])),
        result_fragments=tuple(cast(list[str], ocr["result_fragments"])),
        minimum_result_fragments=int(cast(int, ocr["minimum_result_fragments"])),
        minimum_train_anchor_sessions=int(cast(int, gate["minimum_train_anchor_sessions"])),
        minimum_dev_anchor_sessions=int(cast(int, gate["minimum_dev_anchor_sessions"])),
        maximum_outcome_conflicts=int(cast(int, gate["maximum_outcome_conflicts"])),
    )
    if (
        min(
            config.train_sessions_expected,
            config.dev_sessions_expected,
            config.minimum_result_fragments,
            config.minimum_train_anchor_sessions,
            config.minimum_dev_anchor_sessions,
        )
        <= 0
        or not 0.0 <= config.minimum_confidence <= 1.0
    ):
        raise E1cAnchorError("E1c anchor values are invalid")
    return config, payload, _sha(_canonical(payload))


def classify_anchor(tokens: tuple[OcrToken, ...], config: AnchorConfig) -> AnchorEvidence:
    accepted = [
        token.text.strip().upper()
        for token in tokens
        if token.confidence >= config.minimum_confidence
    ]
    wins = {text for text in accepted if text in config.win_tokens}
    losses = {text for text in accepted if text in config.loss_tokens}
    fragments = {
        fragment
        for fragment in config.result_fragments
        if any(fragment.upper() in text for text in accepted)
    }
    if wins and losses:
        outcome = TerminalLabel.CONFLICT
    elif wins:
        outcome = TerminalLabel.WIN
    elif losses:
        outcome = TerminalLabel.LOSS
    else:
        outcome = TerminalLabel.GAME_END_UNKNOWN
    anchored = bool(wins or losses or len(fragments) >= config.minimum_result_fragments)
    return AnchorEvidence(anchored, outcome, tuple(sorted({*wins, *losses, *fragments})))


def _final_frame(path: Path) -> np.ndarray:
    final: np.ndarray | None = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is None or stream.time_base is None:
            raise E1cAnchorError("E1c anchor video has no duration")
        target = max(0, stream.duration - int(2 / float(stream.time_base)))
        container.seek(target, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            final = frame.to_ndarray(format="bgr24")
    if final is None:
        raise E1cAnchorError("E1c anchor video decoded no frame")
    return final


def _session_row(
    candidate_id: str,
    split: str,
    path: Path,
    config: AnchorConfig,
    contract_sha256: str,
    ocr: OcrSystem,
) -> dict[str, object]:
    frame = _final_frame(path)
    results = ocr.detect_and_ocr(frame)
    evidence = classify_anchor(
        tuple(OcrToken(item.ocr_text, float(item.score)) for item in results),
        config,
    )
    return {
        "session_id": _sha(f"e1c-anchor:{contract_sha256}:{candidate_id}".encode()),
        "split": split,
        "anchored": evidence.anchored,
        "outcome": evidence.outcome.value,
        "allowlisted_hits": list(evidence.allowlisted_hits),
        "source_locator_persisted": False,
        "arbitrary_ocr_text_persisted": False,
        "anchor_frame_persisted": False,
    }


def audit_result_page_anchors(
    contract_path: Path,
    raw_root: Path,
    pre_ingest_path: Path,
    cohort_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config, _contract, contract_sha256 = load_anchor_contract(contract_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise E1cAnchorError("E1c anchor output directory already exists")
    evidence = pre_ingest.load_pre_ingest(pre_ingest_path)
    session_splits = _load_cohort(cohort_path, evidence)
    counts = Counter(session_splits.values())
    if (
        counts["train"] != config.train_sessions_expected
        or counts["dev"] != config.dev_sessions_expected
    ):
        raise E1cAnchorError("E1c anchor train/dev counts differ")
    paths = _candidate_paths(raw_root, evidence)
    ocr = _ocr_system()
    candidates = sorted(
        (candidate for candidate, split in session_splits.items() if split in {"train", "dev"}),
        key=lambda candidate: _sha(f"{contract_sha256}:{candidate}".encode()),
    )
    rows: list[dict[str, object]] = []
    for index, candidate_id in enumerate(candidates, 1):
        split = session_splits[candidate_id]
        row = _session_row(
            candidate_id,
            split,
            paths[candidate_id],
            config,
            contract_sha256,
            ocr,
        )
        rows.append(row)
        if bool(row["anchored"]) or index % 10 == 0 or index == len(candidates):
            print(
                json.dumps(
                    {
                        "anchor_scan_complete": index,
                        "anchor": row["anchored"],
                        "split": split,
                        "total": len(candidates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    train_anchors = sum(row["split"] == "train" and bool(row["anchored"]) for row in rows)
    dev_anchors = sum(row["split"] == "dev" and bool(row["anchored"]) for row in rows)
    conflicts = sum(row["outcome"] == TerminalLabel.CONFLICT.value for row in rows)
    checks = {
        "train_anchor_support": train_anchors >= config.minimum_train_anchor_sessions,
        "dev_anchor_support": dev_anchors >= config.minimum_dev_anchor_sessions,
        "outcome_conflicts": conflicts <= config.maximum_outcome_conflicts,
    }
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "E1C_RESULT_PAGE_ANCHOR_PREFLIGHT_PASSED"
            if all(checks.values())
            else "E1C_RESULT_PAGE_ANCHOR_PREFLIGHT_FAILED"
        ),
        "contract_sha256": contract_sha256,
        "pre_ingest_sha256": evidence.pre_ingest_sha256,
        "scanned_train_sessions": counts["train"],
        "scanned_dev_sessions": counts["dev"],
        "train_anchor_sessions": train_anchors,
        "dev_anchor_sessions": dev_anchors,
        "outcome_conflicts": conflicts,
        "checks": checks,
        "rows": rows,
        "video_end_used_as_label": False,
        "anchor_frame_allowed_in_future_model_input": False,
        "dynamic_transition_accuracy_verified": False,
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


def load_anchor_report(report_path: Path, contract_path: Path) -> dict[str, object]:
    if report_path.is_symlink() or not report_path.is_file():
        raise E1cAnchorError("E1c anchor report must be a regular file")
    data = report_path.read_bytes()
    payload = cast(dict[str, object], json.loads(data))
    _config, _contract, contract_sha256 = load_anchor_contract(contract_path)
    supplied = payload.get("report_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    if (
        payload.get("schema_version") != REPORT_SCHEMA
        or payload.get("contract_sha256") != contract_sha256
        or supplied != _sha(_canonical(unsigned))
        or data != _canonical(payload) + b"\n"
        or payload.get("video_end_used_as_label") is not False
        or payload.get("anchor_frame_allowed_in_future_model_input") is not False
        or payload.get("reward_allowed") is not False
        or payload.get("promotion_allowed") is not False
        or payload.get("video_test_opened") is not False
    ):
        raise E1cAnchorError("E1c anchor report is invalid")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline E1c result-page anchor preflight")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--pre-ingest", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit_result_page_anchors(
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
