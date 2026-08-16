# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from hok_agent import alignment, pre_ingest, v5_data


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_pre_ingest(path: Path, candidate_count: int = 12) -> tuple[Path, dict[str, str]]:
    ids = [_sha(f"session-{index}") for index in range(candidate_count)]
    mapping = pre_ingest._component_map(ids, [])
    payload: dict[str, object] = {
        "schema_version": pre_ingest.SCHEMA,
        "algorithm_spec": pre_ingest.ALGORITHM_SPEC,
        "relationship_mode": pre_ingest.RELATIONSHIP_MODE,
        "candidate_count": candidate_count,
        "component_count": len(set(mapping.values())),
        "candidates": [
            {"candidate_id": session, "component_id": component, "pts_range_us": [0, 1_000_000]}
            for session, component in sorted(mapping.items())
        ],
        "relations": [],
        "component_of": mapping,
        "uncertain_relation_count": 0,
        "blockers": [],
        "review_status": pre_ingest.REVIEW_STATUS,
        "disposition": "BLOCKED_LT_12_COMPONENTS" if candidate_count < 12 else pre_ingest.READY,
    }
    payload["pre_ingest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return path, mapping


def _rewrite_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(_json(payload) + "\n")


def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "large"
    root.mkdir()
    monkeypatch.setenv("HOK_LARGE_ROOT", str(root))
    return root


def _build_manifest_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path, list[Path]]:
    root = _root(tmp_path, monkeypatch)
    pre_path, _ = _write_pre_ingest(tmp_path / "pre.json")
    cohort_dir = root / "audit" / "cohort"
    v5_data.build_automatic_cohort(
        pre_ingest_path=pre_path,
        output_dir=cohort_dir,
        recording_owner_confirmed=True,
        local_research_confirmed=True,
        zero_redaction_confirmed=True,
    )
    evidence = pre_ingest.load_pre_ingest(pre_path)
    private_root = tmp_path / "private-raw"
    private_root.mkdir()
    private = {session: private_root / f"{index}.mp4" for index, session in enumerate(sorted(evidence.component_of))}
    for path in private.values():
        path.write_bytes(b"private")

    monkeypatch.setattr(v5_data, "_candidate_paths", lambda _root, _evidence: private)

    def rows(_path: Path, session: str, split: str) -> object:
        yield {
            "frame": np.full((128, 128, 3), 0, dtype=np.uint8),
            "session_hash": session,
            "timestamp_ms": 0,
            "pts": 0,
            "time_base": (1, 10),
            "rotation_degrees": 90,
            "split": split,
            "source": "target",
        }

    monkeypatch.setattr(v5_data, "_video_records", rows)
    target = root / "datasets" / "target"
    result = v5_data.ingest_zero_label_target(
        input_root=private_root,
        cohort_dir=cohort_dir,
        pre_ingest_path=pre_path,
        output_dir=target,
        shard_size=2,
    )
    assert result["status"] == "PASSED"
    return (
        target / "manifest.json",
        pre_path,
        cohort_dir / "privacy-context.json",
        cohort_dir / "owner-attestation.json",
        cohort_dir / "component-cohort.json",
        sorted((target / "shards").glob("*.npz")),
    )


def test_automatic_cohort_uses_all_clean_components_and_never_writes_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    pre_path, mapping = _write_pre_ingest(tmp_path / "pre.json")
    result = v5_data.build_automatic_cohort(
        pre_ingest_path=pre_path,
        output_dir=root / "audit" / "cohort",
        recording_owner_confirmed=True,
        local_research_confirmed=True,
        zero_redaction_confirmed=True,
    )
    assert result["component_count"] == 12
    assert result["split_components"] == {"train": 8, "dev": 2, "test": 2}
    cohort = v5_data.load_automatic_cohort(root / "audit" / "cohort", pre_path)
    assert set(cohort.session_splits) == set(mapping)
    payload = (root / "audit" / "cohort" / "cohort.json").read_text()
    assert "raw" not in payload and "/" not in payload and "\\" not in payload


def test_automatic_cohort_fails_closed_for_too_few_independent_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    pre_path, _ = _write_pre_ingest(tmp_path / "pre.json", 11)
    with pytest.raises(v5_data.V5DataError):
        v5_data.build_automatic_cohort(
            pre_ingest_path=pre_path,
            output_dir=root / "audit" / "cohort",
            recording_owner_confirmed=True,
            local_research_confirmed=True,
            zero_redaction_confirmed=True,
        )


def test_candidate_path_reconstruction_matches_file_atomic_pre_ingest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, contents in (("b.mp4", b"b"), ("a.mp4", b"a")):
        (raw / name).write_bytes(contents)
    evidence_path = tmp_path / "pre-ingest.json"
    pre_ingest.pre_ingest(raw, evidence_path)
    evidence = pre_ingest.load_pre_ingest(evidence_path)
    resolved = v5_data._candidate_paths(raw, evidence)
    assert set(resolved) == set(evidence.component_of)
    assert [path.name for path in resolved.values()] == ["a.mp4", "b.mp4"]


def test_automatic_cohort_load_rejects_omitted_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    pre_path, _ = _write_pre_ingest(tmp_path / "pre.json")
    cohort_dir = root / "audit" / "cohort"
    result = v5_data.build_automatic_cohort(
        pre_ingest_path=pre_path,
        output_dir=cohort_dir,
        recording_owner_confirmed=True,
        local_research_confirmed=True,
        zero_redaction_confirmed=True,
    )
    assert result["status"] == "PASSED"
    payload = json.loads((cohort_dir / "cohort.json").read_text())
    payload["components"] = list(payload["components"])[:-1]
    payload["cohort_sha256"] = _sha(_json({key: value for key, value in payload.items() if key != "cohort_sha256"}))
    _rewrite_json(payload, cohort_dir / "cohort.json")
    with pytest.raises(v5_data.V5DataError):
        v5_data.load_automatic_cohort(cohort_dir, pre_path)


def test_automatic_cohort_load_rejects_reassigned_valid_count_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    pre_path, _ = _write_pre_ingest(tmp_path / "pre.json")
    cohort_dir = root / "audit" / "cohort"
    v5_data.build_automatic_cohort(
        pre_ingest_path=pre_path,
        output_dir=cohort_dir,
        recording_owner_confirmed=True,
        local_research_confirmed=True,
        zero_redaction_confirmed=True,
    )
    payload = json.loads((cohort_dir / "cohort.json").read_text())
    session_splits = cast(dict[str, str], payload["session_splits"])
    sessions = sorted(session_splits)
    first_train = next(session for session in sessions if session_splits[session] == "train")
    first_dev = next(session for session in sessions if session_splits[session] == "dev")
    payload["session_splits"][first_train], payload["session_splits"][first_dev] = (
        "dev",
        "train",
    )
    payload["cohort_sha256"] = _sha(_json({key: value for key, value in payload.items() if key != "cohort_sha256"}))
    _rewrite_json(payload, cohort_dir / "cohort.json")
    with pytest.raises(v5_data.V5DataError):
        v5_data.load_automatic_cohort(cohort_dir, pre_path)


def test_automatic_cohort_load_rejects_recipe_field_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    pre_path, _ = _write_pre_ingest(tmp_path / "pre.json")
    cohort_dir = root / "audit" / "cohort"
    v5_data.build_automatic_cohort(
        pre_ingest_path=pre_path,
        output_dir=cohort_dir,
        recording_owner_confirmed=True,
        local_research_confirmed=True,
        zero_redaction_confirmed=True,
    )
    cohort = json.loads((cohort_dir / "cohort.json").read_text())
    privacy = json.loads((cohort_dir / "privacy-context.json").read_text())
    privacy["transform"]["privacy_recipe"]["crop"] = True
    privacy["privacy_context_sha256"] = _sha(
        _json({key: value for key, value in privacy.items() if key != "privacy_context_sha256"})
    )
    _rewrite_json(privacy, cohort_dir / "privacy-context.json")
    cohort["privacy_context_sha256"] = privacy["privacy_context_sha256"]
    cohort["cohort_sha256"] = _sha(_json({key: value for key, value in cohort.items() if key != "cohort_sha256"}))
    _rewrite_json(cohort, cohort_dir / "cohort.json")
    with pytest.raises(v5_data.V5DataError):
        v5_data.load_automatic_cohort(cohort_dir, pre_path)


def test_alignment_manifest_load_rejects_omitted_component_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, pre_path, privacy_path, owner_path, confirmation_path, shards = _build_manifest_bundle(
        tmp_path, monkeypatch
    )
    confirmation = json.loads(confirmation_path.read_text())
    confirmation["component_hashes"][-1] = "f" * 64
    confirmation["component_cohort_sha256"] = _sha(
        _json({key: value for key, value in confirmation.items() if key != "component_cohort_sha256"})
    )
    _rewrite_json(confirmation, confirmation_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["component_cohort_sha256"] = confirmation["component_cohort_sha256"]
    manifest["manifest_sha256"] = _sha(
        _json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )
    _rewrite_json(manifest, manifest_path)
    with pytest.raises(alignment.AlignmentError, match="absent from pre-ingest"):
        alignment.load_v5_manifest(
            manifest_path, pre_path, privacy_path, owner_path, confirmation_path, shards
        )


def test_alignment_manifest_load_rejects_reassigned_valid_count_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, pre_path, privacy_path, owner_path, confirmation_path, shards = _build_manifest_bundle(
        tmp_path, monkeypatch
    )
    manifest = json.loads(manifest_path.read_text())
    rows = {row["session_hash"]: row for row in manifest["sessions"]}
    by_split: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for session, row in rows.items():
        by_split[cast(str, row["split"])].append(session)
    first_train = by_split["train"][0]
    first_dev = by_split["dev"][0]
    rows[first_train]["split"] = "dev"
    rows[first_dev]["split"] = "train"
    manifest["sessions"] = list(rows.values())
    manifest["split_binding_sha256"] = alignment.split_binding_hash(
        {row["session_hash"]: row["split"] for row in manifest["sessions"]}
    )
    manifest["manifest_sha256"] = _sha(
        _json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )
    _rewrite_json(manifest, manifest_path)
    with pytest.raises(
        alignment.AlignmentError,
        match="manifest split allocation does not match all-clean lexicographic rule",
    ):
        alignment.load_v5_manifest(
            manifest_path, pre_path, privacy_path, owner_path, confirmation_path, shards
        )


def test_alignment_manifest_load_rejects_recipe_field_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, pre_path, privacy_path, owner_path, confirmation_path, shards = _build_manifest_bundle(
        tmp_path, monkeypatch
    )
    privacy = json.loads(privacy_path.read_text())
    privacy["transform"]["privacy_recipe"]["crop"] = True
    privacy["privacy_context_sha256"] = _sha(
        _json({key: value for key, value in privacy.items() if key != "privacy_context_sha256"})
    )
    _rewrite_json(privacy, privacy_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["privacy_context_sha256"] = privacy["privacy_context_sha256"]
    manifest["manifest_sha256"] = _sha(
        _json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )
    _rewrite_json(manifest, manifest_path)
    with pytest.raises(alignment.AlignmentError, match="privacy context binding is invalid"):
        alignment.load_v5_manifest(
            manifest_path, pre_path, privacy_path, owner_path, confirmation_path, shards
        )


def test_pre_ingest_matches_reencode_by_pts_when_sampling_strides_differ() -> None:
    def sample(seconds: int) -> pre_ingest._Fingerprint:
        return pre_ingest._Fingerprint(seconds * 1_000_000, seconds, (seconds % 256,) * 16)

    left = pre_ingest._Candidate(
        "a" * 64, "b" * 64, (0, 99_000_000), tuple(sample(value) for value in range(100)), None
    )
    right = pre_ingest._Candidate(
        "c" * 64, "d" * 64, (0, 98_000_000), tuple(sample(value) for value in range(0, 100, 2)), None
    )
    relation, metrics = pre_ingest._classify(left, right)
    assert relation == "reencode"
    assert metrics["matched_samples"] == 50
    assert metrics["shorter_coverage_ppm"] == 1_000_000


def test_pre_ingest_groups_partial_pts_overlap_without_cross_split_leakage() -> None:
    def sample(pts: int, content: int) -> pre_ingest._Fingerprint:
        digest = hashlib.sha256(str(content).encode()).digest()
        return pre_ingest._Fingerprint(
            pts * 1_000_000, int.from_bytes(digest[:8], "big"), tuple(digest[8:24])
        )

    left = pre_ingest._Candidate(
        "a" * 64, "b" * 64, (0, 9_000_000), tuple(sample(value, value) for value in range(10)), None
    )
    right = pre_ingest._Candidate(
        "c" * 64, "d" * 64, (0, 9_000_000), tuple(sample(value - 5, value) for value in range(5, 15)), None
    )
    relation, metrics = pre_ingest._classify(left, right)
    assert relation == "overlap"
    assert metrics["matched_samples"] == 5
    assert metrics["match_fraction_ppm"] == 1_000_000
    assert metrics["shorter_coverage_ppm"] == 500_000


def test_pre_ingest_file_atomic_mode_never_decodes_and_emits_one_component_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    files = [
        (root / "b.mp4", b"video-b"),
        (root / "a.mp4", b"video-a"),
        (root / "c.mp4", b"video-c"),
    ]
    for path, value in files:
        path.write_bytes(value)

    output = tmp_path / "pre.json"
    monkeypatch.setattr(pre_ingest, "_decode", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decode invoked")))
    first = pre_ingest.pre_ingest(root, output)

    assert first["relationship_mode"] == pre_ingest.RELATIONSHIP_MODE
    assert first["schema_version"] == pre_ingest.SCHEMA
    assert first["algorithm_spec"]["input_integrity"]["content_hashing"] is False
    assert first["relations"] == []
    assert first["candidate_count"] == 3
    assert first["component_count"] == 3
    assert first["uncertain_relation_count"] == 0
    assert len(first["candidates"]) == 3
    component_ids = [row["component_id"] for row in first["candidates"]]
    candidate_ids = [row["candidate_id"] for row in first["candidates"]]
    assert len(set(component_ids)) == 3
    assert set(component_ids) == set(candidate_ids)


def test_pre_ingest_file_atomic_deterministic_lexicographic_ordering(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    for index, name in enumerate(("z.mp4", "a.mp4", "m.mp4")):
        (root / name).write_bytes(f"video-{index}".encode())
    output = tmp_path / "pre.json"
    first = pre_ingest.pre_ingest(root, output)
    second = pre_ingest.pre_ingest(root, tmp_path / "pre2.json")
    assert [row["candidate_id"] for row in first["candidates"]] == [row["candidate_id"] for row in second["candidates"]]
    assert first["candidate_count"] == 3 == first["component_count"]
    assert second["candidate_count"] == 3 == second["component_count"]


def test_pre_ingest_file_atomic_rejects_non_regular_and_mutating_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    target = root / "target.mp4"
    target.write_bytes(b"video")
    link = root / "link.mp4"
    link.symlink_to(target)
    monkeypatch.setattr(pre_ingest, "_scan", lambda _root: [link])
    output = tmp_path / "pre.json"
    with pytest.raises(pre_ingest.PreIngestError):
        pre_ingest.pre_ingest(root, output)

    def patched_fstat(fd: int) -> os.stat_result:
        stat_result = os.fstat(fd)
        if patched_fstat.__dict__.setdefault("count", 0) == 0:
            patched_fstat.__dict__["count"] += 1
            return stat_result
        return stat_result._replace(st_size=stat_result.st_size + 1)

    root = tmp_path / "videos2"
    root.mkdir()
    (root / "a.mp4").write_bytes(b"video")
    output = tmp_path / "pre2.json"
    monkeypatch.setattr(pre_ingest.os, "fstat", patched_fstat)
    with pytest.raises(pre_ingest.PreIngestError):
        pre_ingest.pre_ingest(root, output)


def test_load_pre_ingest_file_atomic_mode_rejects_mode_or_relation_tamper(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (root / name).write_bytes(name.encode())
    payload_path = tmp_path / "pre.json"
    pre = pre_ingest.pre_ingest(root, payload_path)
    payload = json.loads(payload_path.read_text())
    payload["relationship_mode"] = "bad_mode"
    payload["pre_ingest_sha256"] = _sha(_json(payload))
    payload_path.write_text(_json(payload) + "\n")
    with pytest.raises(pre_ingest.PreIngestError):
        pre_ingest.load_pre_ingest(payload_path)

    payload = json.loads(payload_path.read_text())
    payload["relationship_mode"] = pre["relationship_mode"]
    payload["relations"] = [
        {
            "left_candidate_id": pre["candidates"][0]["candidate_id"],
            "right_candidate_id": pre["candidates"][1]["candidate_id"],
            "relation": "exact",
            "evidence": {
                "whole_file_sha256_equal": True,
                "compared_samples": 0,
                "matched_samples": 0,
                "shorter_coverage_ppm": 0,
                "match_fraction_ppm": 0,
                "offset_samples": 0,
                "median_dhash_hamming": 0,
                "median_luma_delta": 0,
                "evidence_sha256": "0" * 64,
            },
        }
    ]
    payload["pre_ingest_sha256"] = _sha(_json(payload))
    payload_path.write_text(_json(payload) + "\n")
    with pytest.raises(pre_ingest.PreIngestError):
        pre_ingest.load_pre_ingest(payload_path)


def test_ingest_writes_anonymous_target_manifest_and_rotation_bound_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    pre_path, mapping = _write_pre_ingest(tmp_path / "pre.json")
    cohort_dir = root / "audit" / "cohort"
    v5_data.build_automatic_cohort(
        pre_ingest_path=pre_path,
        output_dir=cohort_dir,
        recording_owner_confirmed=True,
        local_research_confirmed=True,
        zero_redaction_confirmed=True,
    )
    raw_root = tmp_path / "private-raw"
    raw_root.mkdir()
    private = {session: raw_root / f"{index}.mp4" for index, session in enumerate(sorted(mapping))}
    for path in private.values():
        path.write_bytes(b"private")
    monkeypatch.setattr(v5_data, "_candidate_paths", lambda _root, _evidence: private)

    def rows(path: Path, session: str, split: str) -> object:
        del path
        for tick in range(3):
            yield {
                "frame": np.full((128, 128, 3), tick, dtype=np.uint8),
                "session_hash": session,
                "timestamp_ms": tick * 100,
                "pts": tick,
                "time_base": (1, 10),
                "rotation_degrees": 90,
                "split": split,
                "source": "target",
            }

    monkeypatch.setattr(v5_data, "_video_records", rows)
    result = v5_data.ingest_zero_label_target(
        input_root=raw_root,
        cohort_dir=cohort_dir,
        pre_ingest_path=pre_path,
        output_dir=root / "datasets" / "target",
        shard_size=2,
    )
    target = root / "datasets" / "target"
    shards = sorted((target / "shards").glob("*.npz"))
    manifest = target / "manifest.json"
    assert result["status"] == "PASSED" and len(shards) > 3
    loaded = alignment.load_v5_manifest(
        manifest,
        pre_path,
        cohort_dir / "privacy-context.json",
        cohort_dir / "owner-attestation.json",
        cohort_dir / "component-cohort.json",
        shards,
    )
    assert set(loaded.session_splits) == set(mapping)
    streamed = v5_data.load_zero_label_target(target, cohort_dir, pre_path)
    assert streamed.session_splits == loaded.session_splits
    text = manifest.read_text() + (target / "target-report.json").read_text()
    assert str(raw_root) not in text and ".mp4" not in text and "audio" not in text

    shard = shards[0]
    shard.write_bytes(b"tampered")
    with pytest.raises(v5_data.V5DataError):
        v5_data.load_zero_label_target(target, cohort_dir, pre_path)


def test_ingest_zero_label_parallel_workers_use_fixed_pool_and_session_stable_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    pre_path, mapping = _write_pre_ingest(tmp_path / "pre.json")
    cohort_dir = root / "audit" / "cohort"
    v5_data.build_automatic_cohort(
        pre_ingest_path=pre_path,
        output_dir=cohort_dir,
        recording_owner_confirmed=True,
        local_research_confirmed=True,
        zero_redaction_confirmed=True,
    )
    raw_root = tmp_path / "private-raw"
    raw_root.mkdir()
    private = {session: raw_root / f"{index}.mp4" for index, session in enumerate(sorted(mapping))}
    for path in private.values():
        path.write_bytes(b"private")
    monkeypatch.setattr(v5_data, "_candidate_paths", lambda _root, _evidence: private)

    def rows(_path: Path, session: str, split: str) -> object:
        for tick in range(2):
            yield {
                "frame": np.full((128, 128, 3), tick, dtype=np.uint8),
                "session_hash": session,
                "timestamp_ms": tick * 100,
                "pts": tick,
                "time_base": (1, 10),
                "rotation_degrees": 90,
                "source": "target",
                "split": split,
            }

    monkeypatch.setattr(v5_data, "_video_records", rows)

    real_pool = ThreadPoolExecutor
    worker_sizes: list[int] = []

    class RecordingPool:
        def __init__(self, max_workers: int) -> None:
            worker_sizes.append(max_workers)
            self._inner = real_pool(max_workers=max_workers)

        def submit(self, fn: object, *args: object, **kwargs: object) -> object:
            return self._inner.submit(fn, *args, **kwargs)

        def __enter__(self) -> RecordingPool:
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
            return self._inner.__exit__(exc_type, exc, tb)

    expected_prefixes = {
        session: f"{index:06d}-"
        for index, (session, _) in enumerate(sorted(private.items()))
    }
    observed_prefixes: dict[str, set[str]] = {session: set() for session in mapping}

    real_write_npz_shards = alignment.write_npz_shards
    observed_rows_per_session: dict[str, int] = {session: 0 for session in mapping}

    def spy_write_npz_shards(rows: Sequence[dict[str, object]], *args: object, **kwargs: object) -> tuple[Path, ...]:
        kwargs_prefix = cast(str, kwargs["name_prefix"])
        row_session = cast(str, rows[0]["session_hash"])
        assert len({cast(str, row["session_hash"]) for row in rows}) == 1
        observed_prefixes[row_session].add(kwargs_prefix)
        observed_rows_per_session[row_session] += len(rows)
        return real_write_npz_shards(
            rows,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(v5_data, "ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(alignment, "write_npz_shards", spy_write_npz_shards)
    result = v5_data.ingest_zero_label_target(
        input_root=raw_root,
        cohort_dir=cohort_dir,
        pre_ingest_path=pre_path,
        output_dir=root / "datasets" / "target",
        shard_size=99,
    )
    assert worker_sizes == [v5_data.MAX_ZERO_LABEL_WORKERS]
    for session, expected_prefix in expected_prefixes.items():
        assert observed_prefixes[session] == {expected_prefix}
    cohort = v5_data.load_automatic_cohort(cohort_dir, pre_path)
    loaded = v5_data.load_zero_label_target(root / "datasets" / "target", cohort_dir, pre_path)
    assert loaded.session_splits == cohort.session_splits
    assert sum(observed_rows_per_session.values()) == sum(result["rows_per_split"].values())
    assert result["status"] == "PASSED"


def test_ingest_flushes_zero_label_shards_by_byte_budget_and_normalizes_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    pre_path, _ = _write_pre_ingest(tmp_path / "pre.json")
    evidence = pre_ingest.load_pre_ingest(pre_path)
    cohort_dir = root / "audit" / "cohort"
    v5_data.build_automatic_cohort(
        pre_ingest_path=pre_path,
        output_dir=cohort_dir,
        recording_owner_confirmed=True,
        local_research_confirmed=True,
        zero_redaction_confirmed=True,
    )
    raw_root = tmp_path / "private-raw"
    raw_root.mkdir()
    private = {session: raw_root / f"{index}.mp4" for index, session in enumerate(sorted(evidence.component_of))}
    for path in private.values():
        path.write_bytes(b"private")
    monkeypatch.setattr(v5_data, "_candidate_paths", lambda _root, _evidence: private)

    def rows(_path: Path, session: str, split: str) -> object:
        for tick in range(7):
            yield {
                "frame": np.full((512, 512, 3), tick, dtype=np.uint8),
                "session_hash": session,
                "timestamp_ms": tick * 100,
                "pts": tick,
                "time_base": (1, 10),
                "rotation_degrees": 90,
                "split": split,
                "source": "target",
            }

    monkeypatch.setattr(v5_data, "_video_records", rows)
    monkeypatch.setattr(v5_data, "MAX_PENDING_DERIVED_BYTES", 128 * 128 * 3 + 1)
    calls: list[int] = []
    real_write = alignment.write_npz_shards

    def spy_rows(rows: Sequence[dict[str, object]], *args: object, **kwargs: object) -> tuple[Path, ...]:
        for row in rows:
            frame = row["frame"]
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (128, 128, 3)
            assert frame.dtype == np.uint8
        calls.append(len(rows))
        return real_write(rows, *args, **kwargs)

    monkeypatch.setattr(alignment, "write_npz_shards", spy_rows)
    result = v5_data.ingest_zero_label_target(
        input_root=raw_root,
        cohort_dir=cohort_dir,
        pre_ingest_path=pre_path,
        output_dir=root / "datasets" / "target",
        shard_size=99,
    )
    assert result["status"] == "PASSED"
    assert max(calls) == 1


def test_load_zero_label_target_session_order_validation_is_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    pre_path, _ = _write_pre_ingest(tmp_path / "pre.json")
    cohort_dir = root / "audit" / "cohort"
    v5_data.build_automatic_cohort(
        pre_ingest_path=pre_path,
        output_dir=cohort_dir,
        recording_owner_confirmed=True,
        local_research_confirmed=True,
        zero_redaction_confirmed=True,
    )
    cohort = v5_data.load_automatic_cohort(cohort_dir, pre_path)
    evidence = pre_ingest.load_pre_ingest(pre_path)
    privacy = alignment.load_privacy_context(
        cohort_dir / "privacy-context.json", cohort_dir / "owner-attestation.json"
    )

    target = root / "datasets" / "target"
    (target / "shards").mkdir(parents=True)

    split_sessions: dict[str, list[str]] = {
        split: [session for session, value in sorted(cohort.session_splits.items()) if value == split]
        for split in alignment.SPLITS
    }
    rows: list[dict[str, object]] = []
    payloads: dict[bytes, dict[str, object]] = {}

    class OnePassSequence:
        def __init__(self, values: list[str]) -> None:
            self.values = tuple(values)
            self.seen = 0

        def __iter__(self) -> Iterator[str]:
            self.seen += 1
            if self.seen > 1:
                raise AssertionError("session_hash iterated more than once")
            return iter(self.values)

        def __len__(self) -> int:
            return len(self.values)

    for split in alignment.SPLITS:
        sessions = split_sessions[split]
        sequence: list[str] = []
        review: list[str] = []
        timestamps: list[int] = []
        for index, session in enumerate(sessions):
            for repeat in range(2):
                sequence.append(session)
                review.append(privacy.privacy_reviews[session])
                timestamps.append(index * 10 + repeat)
        path = target / "shards" / f"{split}.npz"
        path.touch()
        session_iter = OnePassSequence(sequence) if split == "train" else sequence
        shard_payload = {
            "frames": sequence,
            "session_hash": session_iter,
            "timestamp_ms": timestamps,
            "privacy_review_sha256": review,
            "split": [split] * len(sequence),
            "source": ["target"] * len(sequence),
            "privacy_transform_sha256": [privacy.privacy_transform_sha256] * len(sequence),
            "owner_attestation_sha256": [privacy.owner_attestation_sha256] * len(sequence),
            "frame_hash": ["0" * 64] * len(sequence),
            "alignment_hash": ["0" * 64] * len(sequence),
            "time_base": [(1, 10)] * len(sequence),
            "rotation_degrees": [0] * len(sequence),
            "frame": [0] * len(sequence),
        }
        payloads[path.name.encode()] = shard_payload
        rows.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.name.encode()).hexdigest(),
                "row_count": len(sequence),
                "session_hashes": sorted(sessions),
                "split": split,
                "source": "target",
            }
        )

    sessions_payload = [
        {
            "session_hash": session,
            "component_hash": evidence.component_of[session],
            "parent_hash": None,
            "near_duplicate_hashes": [],
            "split": split,
            "privacy_review_sha256": privacy.privacy_reviews[session],
        }
        for session, split in cohort.session_splits.items()
    ]
    manifest = {
        "schema_version": alignment.MANIFEST_SCHEMA,
        "pre_ingest_sha256": evidence.pre_ingest_sha256,
        "component_cohort_sha256": alignment.load_owner_component_confirmation(
            cohort_dir / "component-cohort.json",
            pre_ingest.load_pre_ingest(pre_path).pre_ingest_sha256,
            privacy.owner_attestation_sha256,
        ).component_cohort_sha256,
        "privacy_context_sha256": privacy.privacy_context_sha256,
        "privacy_transform_sha256": privacy.privacy_transform_sha256,
        "owner_attestation_sha256": privacy.owner_attestation_sha256,
        "split_binding_sha256": alignment.split_binding_hash(cohort.session_splits),
        "sessions": sessions_payload,
        "shards": rows,
    }
    manifest["manifest_sha256"] = hashlib.sha256(v5_data._json(manifest).encode()).hexdigest()
    (target / "manifest.json").write_text(v5_data._json(manifest) + "\n")

    original_read_regular = alignment._read_regular
    def fake_read_regular(path: Path, suffix: str) -> bytes:
        if suffix == ".npz":
            return path.name.encode()
        return original_read_regular(path, suffix)

    def fake_load_npz_bytes(data: bytes) -> dict[str, object]:
        payload = payloads.get(data)
        if payload is None:
            raise alignment.AlignmentError("missing shard payload")
        return payload

    monkeypatch.setattr(alignment, "_read_regular", fake_read_regular)
    monkeypatch.setattr(alignment, "_load_npz_bytes", fake_load_npz_bytes)

    loaded = v5_data.load_zero_label_target(target, cohort_dir, pre_path)
    assert loaded.session_splits == cohort.session_splits
