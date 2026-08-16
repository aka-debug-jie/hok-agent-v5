# ruff: noqa: E501
"""Hash-identified, zero-label V5 component cohort and target-frame ingestion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from . import alignment, pre_ingest

COHORT_SCHEMA = "hok-agent-v5-automatic-cohort-v1"
TARGET_SCHEMA = "hok-agent-v5-zero-label-target-v1"
FRAME_PERIOD_MS = 100
MAX_PENDING_DERIVED_BYTES = 16 * 1024 * 1024
MAX_ZERO_LABEL_WORKERS = 64
_HEX = frozenset("0123456789abcdef")


class V5DataError(ValueError):
    """The zero-label target-data chain is malformed or incomplete."""


@dataclass(frozen=True)
class Cohort:
    directory: Path
    pre_ingest_path: Path
    owner_attestation_path: Path
    privacy_context_path: Path
    component_cohort_path: Path
    cohort_sha256: str
    session_splits: dict[str, str]
    components: tuple[str, ...]


@dataclass(frozen=True)
class TargetData:
    directory: Path
    manifest_path: Path
    manifest_sha256: str
    session_splits: dict[str, str]
    shard_paths: tuple[Path, ...]


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _self_hash(payload: Mapping[str, object], field: str) -> str:
    supplied = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    if not _is_sha(supplied) or _sha(_json(unsigned).encode()) != supplied:
        raise V5DataError(f"{field} mismatch")
    return supplied


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _large_directory(path: Path) -> Path:
    root_text = os.environ.get("HOK_LARGE_ROOT")
    if not root_text:
        raise V5DataError("HOK_LARGE_ROOT is required")
    try:
        root = Path(root_text).resolve(strict=True)
    except OSError as exc:
        raise V5DataError("HOK_LARGE_ROOT is unavailable") from exc
    if root.is_symlink() or not root.is_dir():
        raise V5DataError("HOK_LARGE_ROOT must be a real directory")
    target = path.resolve()
    if target == root or root not in target.parents or os.path.lexists(target):
        raise V5DataError("output must be a new directory below HOK_LARGE_ROOT")
    return target


def _atomic_directory(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))


def _clear_staging(directory: Path) -> None:
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        else:
            path.rmdir()
    directory.rmdir()


def _split_components(components: Sequence[str]) -> dict[str, str]:
    values = tuple(sorted(components))
    if (
        len(values) < 12
        or len(set(values)) != len(values)
        or any(not _is_sha(value) for value in values)
    ):
        raise V5DataError("automatic cohort requires at least 12 unique components")
    held_out = math.ceil(0.15 * len(values))
    train_count = len(values) - 2 * held_out
    if train_count < 8 or held_out < 2:
        raise V5DataError("automatic cohort cannot satisfy 8/2/2")
    return {
        component: "train"
        if index < train_count
        else "dev"
        if index < train_count + held_out
        else "test"
        for index, component in enumerate(values)
    }


def _artifact(payload: dict[str, object], field: str) -> bytes:
    payload[field] = _sha(_json(payload).encode())
    return (_json(payload) + "\n").encode()


def build_automatic_cohort(
    *,
    pre_ingest_path: Path,
    output_dir: Path,
    recording_owner_confirmed: bool,
    local_research_confirmed: bool,
    zero_redaction_confirmed: bool,
    redistribution_confirmed: bool = False,
) -> dict[str, object]:
    """Bind every clean component to a deterministic all-component 8/2/2 split."""
    if (
        not recording_owner_confirmed
        or not local_research_confirmed
        or not zero_redaction_confirmed
        or redistribution_confirmed
    ):
        raise V5DataError("owner attestation flags do not authorize local zero-redaction research")
    try:
        evidence = pre_ingest.load_pre_ingest(pre_ingest_path)
    except pre_ingest.PreIngestError as exc:
        raise V5DataError("pre-ingest artifact is invalid") from exc
    if evidence.disposition != pre_ingest.READY:
        raise V5DataError("pre-ingest evidence is not clean and ready")
    components = tuple(sorted(set(evidence.component_of.values())))
    component_splits = _split_components(components)
    session_splits = {
        session: component_splits[component]
        for session, component in sorted(evidence.component_of.items())
    }
    target = _large_directory(output_dir)
    staging = _atomic_directory(target)
    try:
        owner: dict[str, object] = {
            "schema_version": alignment.OWNER_ATTESTATION_SCHEMA,
            "recording_owner": True,
            "local_research_only": True,
            "zero_redaction_authorized": True,
            "redistribution": False,
        }
        owner_data = _artifact(owner, "owner_attestation_sha256")
        owner_hash = cast(str, owner["owner_attestation_sha256"])
        reviews: list[dict[str, object]] = []
        for session, component in sorted(evidence.component_of.items()):
            row: dict[str, object] = {
                "session_hash": session,
                "component_hash": component,
                "zero_redaction_authorized": True,
            }
            _artifact(row, "privacy_review_sha256")
            reviews.append(row)
        privacy: dict[str, object] = {
            "schema_version": alignment.PRIVACY_CONTEXT_SCHEMA,
            "transform": alignment.PRIVACY_SPEC,
            "privacy_transform_sha256": alignment.privacy_transform_hash(),
            "owner_attestation_sha256": owner_hash,
            "reviews": reviews,
        }
        privacy_data = _artifact(privacy, "privacy_context_sha256")
        confirmation: dict[str, object] = {
            "schema_version": alignment.COMPONENT_COHORT_SCHEMA,
            "pre_ingest_sha256": evidence.pre_ingest_sha256,
            "owner_attestation_sha256": owner_hash,
            "component_hashes": list(components),
        }
        confirmation_data = _artifact(confirmation, "component_cohort_sha256")
        cohort: dict[str, object] = {
            "schema_version": COHORT_SCHEMA,
            "pre_ingest_sha256": evidence.pre_ingest_sha256,
            "owner_attestation_sha256": owner_hash,
            "privacy_context_sha256": privacy["privacy_context_sha256"],
            "component_cohort_sha256": confirmation["component_cohort_sha256"],
            "selection_rule": "all_clean_components_lexicographic_ceil15pct_v1",
            "components": list(components),
            "session_splits": session_splits,
        }
        cohort_data = _artifact(cohort, "cohort_sha256")
        for name, data in (
            ("owner-attestation.json", owner_data),
            ("privacy-context.json", privacy_data),
            ("component-cohort.json", confirmation_data),
            ("cohort.json", cohort_data),
        ):
            _write_new(staging / name, data)
        staging.rename(target)
    except BaseException:
        _clear_staging(staging)
        raise
    return {
        "status": "PASSED",
        "disposition": "AUTOMATIC_COHORT_READY_FOR_ZERO_LABEL_SHARDS",
        "component_count": len(components),
        "split_components": {
            split: sum(value == split for value in component_splits.values())
            for split in ("train", "dev", "test")
        },
        "session_count": len(session_splits),
        "cohort_sha256": cohort["cohort_sha256"],
        "output_dir": str(target),
        "raw_source_locators_persisted": False,
    }


def _load_json(path: Path) -> dict[str, object]:
    data = alignment._read_regular(path, ".json")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V5DataError("invalid cohort JSON") from exc
    if type(value) is not dict or data != (_json(value) + "\n").encode():
        raise V5DataError("cohort JSON is not canonical")
    return cast(dict[str, object], value)


def load_automatic_cohort(directory: Path, pre_ingest_path: Path) -> Cohort:
    owner_path = directory / "owner-attestation.json"
    privacy_path = directory / "privacy-context.json"
    confirmation_path = directory / "component-cohort.json"
    cohort_path = directory / "cohort.json"
    try:
        evidence = pre_ingest.load_pre_ingest(pre_ingest_path)
    except pre_ingest.PreIngestError as exc:
        raise V5DataError("pre-ingest artifact is invalid") from exc
    if evidence.disposition != pre_ingest.READY:
        raise V5DataError("pre-ingest evidence is not clean and ready")
    try:
        owner_hash = alignment.load_owner_attestation(owner_path)
        privacy = alignment.load_privacy_context(privacy_path, owner_path)
        confirmation = alignment.load_owner_component_confirmation(
            confirmation_path, evidence.pre_ingest_sha256, owner_hash
        )
    except alignment.AlignmentError as exc:
        raise V5DataError("cohort owner/privacy artifacts are invalid") from exc
    payload = _load_json(cohort_path)
    fields = {
        "schema_version",
        "pre_ingest_sha256",
        "owner_attestation_sha256",
        "privacy_context_sha256",
        "component_cohort_sha256",
        "selection_rule",
        "components",
        "session_splits",
        "cohort_sha256",
    }
    if set(payload) != fields or payload.get("schema_version") != COHORT_SCHEMA:
        raise V5DataError("cohort fields are invalid")
    cohort_hash = _self_hash(payload, "cohort_sha256")
    if (
        payload.get("pre_ingest_sha256") != evidence.pre_ingest_sha256
        or payload.get("owner_attestation_sha256") != owner_hash
        or payload.get("privacy_context_sha256") != privacy.privacy_context_sha256
        or payload.get("component_cohort_sha256")
        != confirmation.component_cohort_sha256
        or payload.get("selection_rule") != "all_clean_components_lexicographic_ceil15pct_v1"
    ):
        raise V5DataError("cohort evidence binding mismatch")
    components = payload.get("components")
    session_splits = payload.get("session_splits")
    if (
        not isinstance(components, list)
        or components != sorted(components)
        or tuple(components) != tuple(sorted(set(evidence.component_of.values())))
        or set(components) != set(confirmation.component_hashes)
        or not isinstance(session_splits, dict)
        or set(session_splits) != set(evidence.component_of)
        or any(value not in alignment.SPLITS for value in session_splits.values())
    ):
        raise V5DataError("cohort components or session split are invalid")
    expected_components = _split_components(cast(list[str], components))
    expected_sessions = {
        session: expected_components[component]
        for session, component in sorted(evidence.component_of.items())
    }
    if session_splits != expected_sessions:
        raise V5DataError("cohort split differs from frozen automatic rule")
    return Cohort(
        directory,
        pre_ingest_path,
        owner_path,
        privacy_path,
        confirmation_path,
        cohort_hash,
        cast(dict[str, str], session_splits),
        tuple(cast(list[str], components)),
    )


def _candidate_paths(input_root: Path, evidence: pre_ingest.PreIngestEvidence) -> dict[str, Path]:
    """Reconstruct anonymous candidate identities in memory; never persist source locators."""
    try:
        paths = pre_ingest._scan(input_root)
        candidates = pre_ingest._candidate_list(paths, input_root)
    except pre_ingest.PreIngestError as exc:
        raise V5DataError("raw candidate reconstruction failed") from exc
    if any(candidate.blocker is not None for candidate in candidates):
        raise V5DataError("raw candidate reconstruction contains decode blockers")
    mapping = {
        candidate.candidate_id: path for candidate, path in zip(candidates, paths, strict=True)
    }
    if set(mapping) != set(evidence.component_of):
        raise V5DataError("raw candidate set differs from pre-ingest evidence")
    return mapping


def _video_records(path: Path, session_hash: str, split: str) -> Iterator[dict[str, object]]:
    import av

    descriptor, opened = pre_ingest._open_regular(path)
    count = 0
    try:
        with os.fdopen(descriptor, "rb") as handle:
            with av.open(handle, mode="r") as container:
                streams = list(container.streams.video)
                if len(streams) != 1:
                    raise V5DataError("raw candidate has unsupported video streams")
                stream = streams[0]
                try:
                    rotation = pre_ingest._rotation(stream)
                except pre_ingest.PreIngestError as exc:
                    raise V5DataError("raw candidate rotation is unsupported") from exc
                previous: int | None = None
                first: int | None = None
                next_target: int | None = None
                for frame in container.decode(stream):
                    if frame.pts is None or frame.time_base is None:
                        raise V5DataError("raw frame is missing PTS/time-base")
                    numerator, denominator = (
                        int(frame.time_base.numerator),
                        int(frame.time_base.denominator),
                    )
                    if denominator <= 0:
                        raise V5DataError("raw frame has invalid time-base")
                    pts_us = int(frame.pts) * numerator * 1_000_000 // denominator
                    if previous is not None and pts_us <= previous:
                        raise V5DataError("raw frame PTS is not strictly ordered")
                    previous = pts_us
                    first = pts_us if first is None else first
                    next_target = pts_us if next_target is None else next_target
                    if pts_us < next_target:
                        continue
                    yield {
                        "frame": alignment.zero_redaction_letterbox_rgb(
                            frame.to_ndarray(format="rgb24"), rotation
                        ),
                        "session_hash": session_hash,
                        "timestamp_ms": count * FRAME_PERIOD_MS,
                        "pts": int(frame.pts),
                        "time_base": (numerator, denominator),
                        "rotation_degrees": rotation,
                        "split": split,
                        "source": "target",
                    }
                    count += 1
                    assert first is not None
                    next_target = first + count * FRAME_PERIOD_MS * 1_000
                if count == 0:
                    raise V5DataError("raw candidate has no sampled RGB frames")
            pre_ingest._assert_unchanged(descriptor, opened)
    except (OSError, pre_ingest.PreIngestError) as exc:
        raise V5DataError("raw candidate changed or became unreadable") from exc


def _derive_zero_label_worker(
    *,
    session: str,
    raw_path: Path,
    split: str,
    session_index: int,
    shard_root: Path,
    pre_ingest_path: Path,
    privacy_context_path: Path,
    owner_attestation_path: Path,
    owner_component_confirmation_path: Path,
    shard_size: int,
    pending_byte_limit: int,
) -> tuple[str, int, tuple[Path, ...]]:
    pending: list[dict[str, object]] = []
    pending_bytes = 0
    shard_paths: list[Path] = []
    row_count = 0
    start_index = 0
    name_prefix = f"{session_index:06d}-"

    def flush() -> None:
        nonlocal start_index, pending_bytes
        if not pending:
            return
        paths = alignment.write_npz_shards(
            pending,
            shard_root,
            shard_size=shard_size,
            pre_ingest_path=pre_ingest_path,
            privacy_context_path=privacy_context_path,
            owner_attestation_path=owner_attestation_path,
            owner_component_confirmation_path=owner_component_confirmation_path,
            name_prefix=name_prefix,
            start_index=start_index,
            frames_already_normalized=True,
        )
        start_index += len(paths)
        shard_paths.extend(paths)
        pending.clear()
        pending_bytes = 0

    for row in _video_records(raw_path, session, split):
        frame = row.get("frame")
        frame_array = np.asarray(frame)
        rotation = cast(int, row["rotation_degrees"])
        if (
            not isinstance(frame_array, np.ndarray)
            or frame_array.shape != (128, 128, 3)
            or frame_array.dtype != np.uint8
        ):
            frame_array = alignment.zero_redaction_letterbox_rgb(frame_array, rotation)
        row["frame"] = frame_array
        row_bytes = int(frame_array.nbytes)
        if pending_bytes + row_bytes > pending_byte_limit and pending:
            flush()
        pending.append(row)
        pending_bytes += row_bytes
        row_count += 1
        if len(pending) >= shard_size:
            flush()
    flush()
    return split, row_count, tuple(shard_paths)


def _manifest_payload(
    *,
    cohort: Cohort,
    shard_paths: Sequence[Path],
) -> dict[str, object]:
    session_components = pre_ingest.load_pre_ingest(cohort.pre_ingest_path).component_of
    privacy = alignment.load_privacy_context(
        cohort.privacy_context_path, cohort.owner_attestation_path
    )
    sessions: list[dict[str, object]] = [
        {
            "session_hash": session,
            "component_hash": session_components[session],
            "parent_hash": None,
            "near_duplicate_hashes": [],
            "split": cohort.session_splits[session],
            "privacy_review_sha256": privacy.privacy_reviews[session],
        }
        for session in sorted(cohort.session_splits)
    ]
    shards: list[dict[str, object]] = []
    for path in shard_paths:
        try:
            shard_data = alignment._read_regular(path, ".npz")
            shard = alignment._load_npz_bytes(shard_data)
        except alignment.AlignmentError as exc:
            raise V5DataError("target shard is invalid") from exc
        splits = {str(value) for value in shard["split"]}
        sources = {str(value) for value in shard["source"]}
        sessions_in_shard = sorted({str(value) for value in shard["session_hash"]})
        if len(splits) != 1 or sources != {"target"} or not sessions_in_shard:
            raise V5DataError("target shard must contain one split and source=target")
        shards.append(
            {
                "path": path.name,
                "sha256": _sha(shard_data),
                "row_count": len(shard["frames"]),
                "session_hashes": sessions_in_shard,
                "split": next(iter(splits)),
                "source": "target",
            }
        )
    payload: dict[str, object] = {
        "schema_version": alignment.MANIFEST_SCHEMA,
        "pre_ingest_sha256": pre_ingest.load_pre_ingest(cohort.pre_ingest_path).pre_ingest_sha256,
        "component_cohort_sha256": alignment.load_owner_component_confirmation(
            cohort.component_cohort_path,
            pre_ingest.load_pre_ingest(cohort.pre_ingest_path).pre_ingest_sha256,
            privacy.owner_attestation_sha256,
        ).component_cohort_sha256,
        "privacy_context_sha256": privacy.privacy_context_sha256,
        "privacy_transform_sha256": privacy.privacy_transform_sha256,
        "owner_attestation_sha256": privacy.owner_attestation_sha256,
        "split_binding_sha256": alignment.split_binding_hash(cohort.session_splits),
        "sessions": sessions,
        "shards": sorted(shards, key=lambda value: cast(str, value["path"])),
    }
    payload["manifest_sha256"] = _sha(_json(payload).encode())
    return payload


def load_zero_label_target(directory: Path, cohort_dir: Path, pre_ingest_path: Path) -> TargetData:
    """Stream-verify the target manifest without retaining RGB frames across shards."""
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    payload = _load_json(directory / "manifest.json")
    fields = {
        "schema_version",
        "pre_ingest_sha256",
        "component_cohort_sha256",
        "privacy_context_sha256",
        "privacy_transform_sha256",
        "owner_attestation_sha256",
        "split_binding_sha256",
        "sessions",
        "shards",
        "manifest_sha256",
    }
    if set(payload) != fields or payload.get("schema_version") != alignment.MANIFEST_SCHEMA:
        raise V5DataError("target manifest fields are invalid")
    manifest_sha256 = _self_hash(payload, "manifest_sha256")
    evidence = pre_ingest.load_pre_ingest(pre_ingest_path)
    privacy = alignment.load_privacy_context(
        cohort.privacy_context_path, cohort.owner_attestation_path
    )
    confirmation = alignment.load_owner_component_confirmation(
        cohort.component_cohort_path, evidence.pre_ingest_sha256, privacy.owner_attestation_sha256
    )
    bindings = (
        ("pre_ingest_sha256", evidence.pre_ingest_sha256),
        ("component_cohort_sha256", confirmation.component_cohort_sha256),
        ("privacy_context_sha256", privacy.privacy_context_sha256),
        ("privacy_transform_sha256", privacy.privacy_transform_sha256),
        ("owner_attestation_sha256", privacy.owner_attestation_sha256),
        ("split_binding_sha256", alignment.split_binding_hash(cohort.session_splits)),
    )
    if any(payload.get(key) != value for key, value in bindings):
        raise V5DataError("target manifest evidence binding mismatch")
    sessions = payload.get("sessions")
    shards = payload.get("shards")
    if not isinstance(sessions, list) or not isinstance(shards, list):
        raise V5DataError("target manifest sessions/shards are invalid")
    declared: dict[str, str] = {}
    for row in sessions:
        if not isinstance(row, dict) or set(row) != {
            "session_hash",
            "component_hash",
            "parent_hash",
            "near_duplicate_hashes",
            "split",
            "privacy_review_sha256",
        }:
            raise V5DataError("target manifest session row is invalid")
        session, component, split, review = (
            row["session_hash"],
            row["component_hash"],
            row["split"],
            row["privacy_review_sha256"],
        )
        if (
            not _is_sha(session)
            or not _is_sha(component)
            or not _is_sha(review)
            or split not in alignment.SPLITS
            or evidence.component_of.get(session) != component
            or privacy.privacy_reviews.get(session) != review
            or cohort.session_splits.get(session) != split
            or row["parent_hash"] is not None
            or row["near_duplicate_hashes"] != []
            or session in declared
        ):
            raise V5DataError("target manifest session binding is invalid")
        declared[session] = split
    if declared != cohort.session_splits:
        raise V5DataError("target manifest sessions do not match automatic cohort")
    supplied = {path.name: path for path in sorted((directory / "shards").glob("*.npz"))}
    if not supplied:
        raise V5DataError("target manifest has no shards")
    expected_names: set[str] = set()
    seen_sessions: set[str] = set()
    session_rank = {
        split: {
            session: index
            for index, session in enumerate(
                sorted(session for session, value in declared.items() if value == split)
            )
        }
        for split in alignment.SPLITS
    }
    last_rank = {split: -1 for split in alignment.SPLITS}
    last_timestamp: dict[tuple[str, str], int] = {}
    for row in shards:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "sha256",
            "row_count",
            "session_hashes",
            "split",
            "source",
        }:
            raise V5DataError("target manifest shard row is invalid")
        name, digest, row_count, row_sessions, split, source = (
            row["path"],
            row["sha256"],
            row["row_count"],
            row["session_hashes"],
            row["split"],
            row["source"],
        )
        row_session_texts = [str(value) for value in row_sessions]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".npz")
            or name in expected_names
            or not _is_sha(digest)
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
            or not isinstance(row_sessions, list)
            or not row_sessions
            or len(set(row_session_texts)) != len(row_session_texts)
            or any(session not in declared for session in row_session_texts)
            or split not in alignment.SPLITS
            or source != "target"
        ):
            raise V5DataError("target manifest shard declaration is invalid")
        row_session_set = set(row_session_texts)
        path = supplied.get(name)
        if path is None:
            raise V5DataError("target manifest shard is missing")
        try:
            shard_data = alignment._read_regular(path, ".npz")
            shard = alignment._load_npz_bytes(shard_data)
        except alignment.AlignmentError as exc:
            raise V5DataError("target manifest shard is invalid") from exc
        if (
            _sha(shard_data) != digest
            or len(shard["frames"]) != row_count
            or any(str(value) != split for value in shard["split"])
            or any(str(value) != "target" for value in shard["source"])
            or any(str(value) != privacy.privacy_transform_sha256 for value in shard["privacy_transform_sha256"])
            or any(str(value) != privacy.owner_attestation_sha256 for value in shard["owner_attestation_sha256"])
        ):
            raise V5DataError("target manifest shard binding mismatch")
        observed_sessions: set[str] = set()
        observed_rows = 0
        for session, timestamp, review in zip(
            shard["session_hash"],
            shard["timestamp_ms"],
            shard["privacy_review_sha256"],
            strict=True,
        ):
            session_text = str(session)
            if session_text not in row_session_set:
                raise V5DataError("target manifest shard binding mismatch")
            rank = session_rank[split][session_text]
            key = (split, session_text)
            timestamp_value = int(timestamp)
            if (
                rank < last_rank[split]
                or (rank == last_rank[split] and timestamp_value <= last_timestamp[key])
                or privacy.privacy_reviews[session_text] != str(review)
            ):
                raise V5DataError("target frames are not ordered or privacy-bound")
            last_rank[split] = rank
            last_timestamp[key] = timestamp_value
            observed_sessions.add(session_text)
            observed_rows += 1
        if observed_sessions != row_session_set or observed_rows != row_count:
            raise V5DataError("target manifest shard binding mismatch")
        seen_sessions.update(observed_sessions)
        expected_names.add(name)
    if set(supplied) != expected_names or seen_sessions != set(declared):
        raise V5DataError("target manifest shard coverage is incomplete")
    return TargetData(
        directory,
        directory / "manifest.json",
        manifest_sha256,
        declared,
        tuple(supplied[name] for name in sorted(supplied)),
    )


def ingest_zero_label_target(
    *,
    input_root: Path,
    cohort_dir: Path,
    pre_ingest_path: Path,
    output_dir: Path,
    shard_size: int = 256,
) -> dict[str, object]:
    """Decode cohort MP4s into hash-identified, rotation-normalized target shards."""
    if shard_size < 1:
        raise V5DataError("shard_size must be positive")
    cohort = load_automatic_cohort(cohort_dir, pre_ingest_path)
    try:
        evidence = pre_ingest.load_pre_ingest(pre_ingest_path)
    except pre_ingest.PreIngestError as exc:
        raise V5DataError("pre-ingest artifact is invalid") from exc
    candidates = _candidate_paths(input_root, evidence)
    target = _large_directory(output_dir)
    staging = _atomic_directory(target)
    try:
        shard_root = staging / "shards"
        shard_root.mkdir()
        shard_paths: list[Path] = []
        rows_per_split = {name: 0 for name in ("train", "dev", "test")}
        jobs = [
            (index, session, raw_path)
            for index, (session, raw_path) in enumerate(sorted(candidates.items()))
        ]
        for session, _raw_path in sorted(candidates.items()):
            if cohort.session_splits.get(session) is None:
                raise V5DataError("raw candidate is absent from automatic cohort")
        with ThreadPoolExecutor(max_workers=MAX_ZERO_LABEL_WORKERS) as executor:
            futures = [
                executor.submit(
                    _derive_zero_label_worker,
                    session=session,
                    raw_path=raw_path,
                    split=cohort.session_splits[session],
                    session_index=session_index,
                    shard_root=shard_root,
                    pre_ingest_path=cohort.pre_ingest_path,
                    privacy_context_path=cohort.privacy_context_path,
                    owner_attestation_path=cohort.owner_attestation_path,
                    owner_component_confirmation_path=cohort.component_cohort_path,
                    shard_size=shard_size,
                    pending_byte_limit=MAX_PENDING_DERIVED_BYTES,
                )
                for session_index, session, raw_path in jobs
            ]
            for future in as_completed(futures):
                split, rows, paths = future.result()
                rows_per_split[split] += rows
                shard_paths.extend(paths)
        shard_paths.sort(key=lambda path: path.name)
        if any(count == 0 for count in rows_per_split.values()):
            raise V5DataError("one automatic cohort split has no derived frames")
        payload = _manifest_payload(cohort=cohort, shard_paths=shard_paths)
        manifest_path = staging / "manifest.json"
        _write_new(manifest_path, (_json(payload) + "\n").encode())
        try:
            load_zero_label_target(staging, cohort_dir, pre_ingest_path)
        except (OSError, V5DataError, alignment.AlignmentError) as exc:
            raise V5DataError("target manifest self-validation failed") from exc
        report: dict[str, object] = {
            "schema_version": TARGET_SCHEMA,
            "manifest_sha256": payload["manifest_sha256"],
            "cohort_sha256": cohort.cohort_sha256,
            "frame_period_ms": FRAME_PERIOD_MS,
            "rows_per_split": rows_per_split,
            "shard_count": len(shard_paths),
            "raw_source_locators_persisted": False,
            "human_labels_consumed": False,
        }
        _write_new(staging / "target-report.json", _artifact(report, "target_report_sha256"))
        staging.rename(target)
    except BaseException:
        _clear_staging(staging)
        raise
    return {
        "status": "PASSED",
        "disposition": "ZERO_LABEL_TARGET_SHARDS_READY",
        "manifest_sha256": payload["manifest_sha256"],
        "rows_per_split": rows_per_split,
        "shard_count": len(shard_paths),
        "output_dir": str(target),
        "raw_source_locators_persisted": False,
        "human_labels_consumed": False,
    }
