from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from hok_agent.transition_store import HierarchicalTransitionRecord, validate_transition

PACKAGE_SCHEMA = "movement-mvp-r0-package-v1"
SUMMARY_SCHEMA = "movement-mvp-r0-delivery-summary-v1"
RESOLVED_CONFIG_SCHEMA = "movement-mvp-r0-resolved-config-v1"
MOVEMENT_ACTIONS = ("STOP", "N", "S", "W", "E", "NW", "NE", "SW", "SE")


@dataclass(frozen=True, slots=True)
class RuntimeAudit:
    rows: tuple[HierarchicalTransitionRecord, ...]
    frame_paths: tuple[Path, ...]
    transition_sha256: str
    frame_view_sha256: str
    terminal_transitions: int
    reward_total: float
    sqlite_integrity: str


@dataclass(frozen=True, slots=True)
class D0Evidence:
    root: Path
    summary: dict[str, object]
    contract: dict[str, object]
    audit: RuntimeAudit


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required JSON is not a regular file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path.name}")
    return cast(dict[str, object], value)


def _read_self_bound(path: Path, field: str) -> dict[str, object]:
    payload = _read_json(path)
    supplied = str(payload.pop(field, ""))
    calculated = _object_sha256(payload)
    payload[field] = supplied
    if supplied != calculated:
        raise ValueError(f"{path.name} self hash differs")
    return payload


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sqlite_rows(database: Path) -> tuple[str, tuple[HierarchicalTransitionRecord, ...]]:
    if database.is_symlink() or not database.is_file():
        raise ValueError("replay.sqlite3 is not a regular file")
    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            integrity_row = connection.execute("PRAGMA integrity_check").fetchone()
            encoded_rows = connection.execute(
                "SELECT payload_json FROM transitions ORDER BY episode_id, step_id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError("replay.sqlite3 cannot be audited") from exc
    integrity = str(integrity_row[0]) if integrity_row else "missing"
    try:
        rows = tuple(
            cast(HierarchicalTransitionRecord, json.loads(str(encoded)))
            for (encoded,) in encoded_rows
        )
    except json.JSONDecodeError as exc:
        raise ValueError("transition JSON is invalid") from exc
    return integrity, rows


def _audit_runtime(database: Path, frame_dir: Path) -> RuntimeAudit:
    integrity, rows = _sqlite_rows(database)
    if integrity != "ok" or len(rows) != 90:
        raise ValueError("R0 runtime requires an intact 90-transition SQLite store")
    episodes: dict[str, list[HierarchicalTransitionRecord]] = {}
    frames: dict[str, dict[str, object]] = {}
    for row in rows:
        if not validate_transition(row).valid:
            raise ValueError("R0 runtime contains an invalid transition")
        episode = episodes.setdefault(row["episode_id"], [])
        if row["step_id"] != len(episode):
            raise ValueError("R0 transition steps are not contiguous")
        if episode and row["observation"]["observation_id"] != episode[-1]["next_observation"][
            "observation_id"
        ]:
            raise ValueError("R0 observation chain differs")
        if episode and episode[-1]["done"]:
            raise ValueError("R0 episode continues after terminal")
        episode.append(row)
        for key in ("observation", "next_observation"):
            frame = cast(dict[str, object], row[key])
            identity = str(frame["observation_id"])
            previous = frames.setdefault(identity, frame)
            if previous != frame:
                raise ValueError("R0 duplicate observation metadata differs")
    if len(episodes) != 10 or any(
        len(episode) != 9
        or not episode[-1]["done"]
        or [row["executed_action"]["applied_movement"] for row in episode[-3:]]
        != ["STOP", "STOP", "STOP"]
        for episode in episodes.values()
    ):
        raise ValueError("R0 episode completion contract differs")

    frame_paths: list[Path] = []
    frame_views: dict[str, object] = {}
    for identity, frame in sorted(frames.items()):
        basename = str(frame["frame_bundle_ref"])
        if Path(basename).name != basename:
            raise ValueError("R0 frame reference is not an anonymous basename")
        path = frame_dir / basename
        if path.is_symlink() or not path.is_file():
            raise ValueError("R0 frame bundle is missing")
        try:
            with np.load(path, allow_pickle=False) as bundle:
                if set(bundle.files) != {"main", "minimap", "hud"}:
                    raise ValueError("R0 frame fields differ")
                hashes = {
                    name: hashlib.sha256(np.ascontiguousarray(bundle[name]).tobytes()).hexdigest()
                    for name in bundle.files
                }
        except (OSError, ValueError) as exc:
            raise ValueError("R0 frame bundle cannot be audited") from exc
        if hashes != frame["view_sha256"]:
            raise ValueError("R0 frame view hash differs")
        frame_paths.append(path)
        frame_views[identity] = hashes
    actual_npz = {
        path.name for path in frame_dir.iterdir() if path.is_file() and path.suffix == ".npz"
    }
    if len(frame_paths) != 100 or actual_npz != {path.name for path in frame_paths}:
        raise ValueError("R0 frame bundle set differs")
    return RuntimeAudit(
        rows,
        tuple(frame_paths),
        _object_sha256(rows),
        _object_sha256(frame_views),
        sum(bool(row["done"]) for row in rows),
        sum(float(row["reward"]["total"]) for row in rows),
        integrity,
    )


def _load_d0_evidence(root: Path) -> D0Evidence:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("D0 source run is not a regular directory")
    summary = _read_self_bound(root / "batch-summary.json", "summary_sha256")
    contract = _read_self_bound(root / "run-contract.json", "run_contract_sha256")
    audit = _audit_runtime(root / "replay.sqlite3", root)
    required = {
        "status": "PASSED",
        "delivery_grade": "R0_RULE_OFFLINE",
        "completed_episodes": 10,
        "transitions": 90,
        "terminal_transitions": 10,
        "reward_total": 0.0,
        "input_commands_sent": 0,
        "sqlite_integrity": "ok",
        "learned_navigation": False,
        "model_checkpoint_loaded": False,
    }
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError("D0 summary admission fields differ")
    if (
        summary.get("run_contract_sha256") != contract.get("run_contract_sha256")
        or summary.get("transition_content_sha256") != audit.transition_sha256
        or summary.get("frame_view_manifest_sha256") != audit.frame_view_sha256
        or audit.terminal_transitions != 10
        or audit.reward_total != 0.0
    ):
        raise ValueError("D0 summary does not bind its runtime evidence")
    resolved = cast(dict[str, object], contract.get("resolved_config"))
    if (
        resolved.get("hero") != "houyi"
        or resolved.get("role") != "marksman"
        or resolved.get("side") != "blue"
        or resolved.get("lane") != "bottom"
        or contract.get("step_duration_ms") != 100
        or contract.get("movement_actions") != list(MOVEMENT_ACTIONS)
    ):
        raise ValueError("D0 resolved navigation contract differs")
    return D0Evidence(root, summary, contract, audit)


def _backup_sqlite(source: Path, target: Path) -> None:
    with (
        sqlite3.connect(f"file:{source}?mode=ro", uri=True) as source_connection,
        sqlite3.connect(target) as target_connection,
    ):
        source_connection.backup(target_connection)
        target_connection.execute("PRAGMA journal_mode=DELETE")
        target_connection.commit()
    _fsync_file(target)


def _manifest_files(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return files


def create_r0_package(source_run: Path, control_run: Path, output_dir: Path) -> dict[str, object]:
    source = _load_d0_evidence(source_run)
    control = _load_d0_evidence(control_run)
    if (
        source.summary.get("recovered_transition_count") != 4
        or int(cast(int, source.summary.get("resume_count", 0))) < 1
        or control.summary.get("recovered_transition_count") != 0
        or control.summary.get("resume_count") != 0
        or source.audit.transition_sha256 != control.audit.transition_sha256
        or source.audit.frame_view_sha256 != control.audit.frame_view_sha256
    ):
        raise ValueError("D0 interrupted and control evidence do not form the accepted pair")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("R0 package output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        frame_dir = staging / "frames"
        frame_dir.mkdir()
        resolved = {
            "schema_version": RESOLVED_CONFIG_SCHEMA,
            "navigation": source.contract["resolved_config"],
            "movement_actions": source.contract["movement_actions"],
            "step_duration_ms": source.contract["step_duration_ms"],
            "policy": source.contract["policy"],
            "source_run_contract_sha256": source.contract["run_contract_sha256"],
        }
        _write_json(staging / "resolved-config.json", resolved)
        _backup_sqlite(source.root / "replay.sqlite3", staging / "replay.sqlite3")
        for frame in source.audit.frame_paths:
            target = frame_dir / frame.name
            shutil.copyfile(frame, target)
            _fsync_file(target)
        packaged_runtime_bytes = sum(
            path.stat().st_size
            for path in (
                staging / "resolved-config.json",
                staging / "replay.sqlite3",
                *frame_dir.iterdir(),
            )
        )
        summary: dict[str, object] = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "PASSED",
            "delivery_grade": "R0_RULE_OFFLINE",
            "selected_runtime_policy": "structured_simulator_rule",
            "promoted_checkpoint": None,
            "learned_navigation": False,
            "mid_episode_resume": True,
            "real_rgb": False,
            "mobile": False,
            "reward_enabled": False,
            "reinforcement_learning": False,
            "holdout_opened": False,
            "episodes": 10,
            "transitions": 90,
            "terminal_transitions": 10,
            "frame_bundles": 100,
            "reward_total": 0.0,
            "input_commands_sent": 0,
            "transition_content_sha256": source.audit.transition_sha256,
            "frame_view_manifest_sha256": source.audit.frame_view_sha256,
            "source_evidence": {
                "interrupted_summary_file_sha256": _file_sha256(
                    source.root / "batch-summary.json"
                ),
                "continuous_summary_file_sha256": _file_sha256(
                    control.root / "batch-summary.json"
                ),
                "run_contract_file_sha256": _file_sha256(source.root / "run-contract.json"),
            },
            "learning_evidence": {
                "initial_candidate_successes": 15,
                "initial_candidate_episodes": 24,
                "initial_candidate_collision_fraction": 0.27075812274368233,
                "recovery_candidate_successes": 0,
                "balanced_candidate_successes": 0,
                "promoted": False,
            },
            "budget": {
                "stage_e_gpu_hours": 0.0,
                "exact_engineering_hours": None,
                "packaged_runtime_bytes": packaged_runtime_bytes,
            },
        }
        summary["summary_sha256"] = _object_sha256(summary)
        _write_json(staging / "summary.json", summary)
        files = _manifest_files(staging)
        manifest: dict[str, object] = {
            "schema_version": PACKAGE_SCHEMA,
            "status": "COMPLETE",
            "files": files,
            "file_count": len(files),
            "package_payload_bytes": sum(int(cast(int, row["bytes"])) for row in files),
        }
        manifest["manifest_sha256"] = _object_sha256(manifest)
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_r0_package(output_dir)


def verify_r0_package(output_dir: Path) -> dict[str, object]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("R0 package is not a regular directory")
    manifest_path = output_dir / "manifest.json"
    manifest = _read_self_bound(manifest_path, "manifest_sha256")
    if manifest.get("schema_version") != PACKAGE_SCHEMA or manifest.get("status") != "COMPLETE":
        raise ValueError("R0 package manifest identity differs")
    rows = cast(list[dict[str, object]], manifest.get("files"))
    declared = {str(row["path"]): row for row in rows}
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if (
        len(rows) != int(cast(int, manifest.get("file_count", -1)))
        or len(declared) != len(rows)
        or actual != set(declared)
        or any(path.is_symlink() for path in output_dir.rglob("*"))
    ):
        raise ValueError("R0 package file set differs")
    for relative, row in declared.items():
        path = output_dir / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.stat().st_size != int(cast(int, row["bytes"]))
            or _file_sha256(path) != row["sha256"]
        ):
            raise ValueError("R0 package file binding differs")
    if sum(int(cast(int, row["bytes"])) for row in rows) != manifest.get(
        "package_payload_bytes"
    ):
        raise ValueError("R0 package payload size differs")
    summary = _read_self_bound(output_dir / "summary.json", "summary_sha256")
    resolved = _read_json(output_dir / "resolved-config.json")
    audit = _audit_runtime(output_dir / "replay.sqlite3", output_dir / "frames")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != "PASSED"
        or summary.get("delivery_grade") != "R0_RULE_OFFLINE"
        or summary.get("promoted_checkpoint") is not None
        or summary.get("transition_content_sha256") != audit.transition_sha256
        or summary.get("frame_view_manifest_sha256") != audit.frame_view_sha256
        or summary.get("reward_total") != audit.reward_total
        or resolved.get("schema_version") != RESOLVED_CONFIG_SCHEMA
        or resolved.get("movement_actions") != list(MOVEMENT_ACTIONS)
    ):
        raise ValueError("R0 package summary or resolved config differs")
    if any(output_dir.rglob("*.safetensors")) or any(output_dir.rglob("replay.sqlite3-*")):
        raise ValueError("R0 package contains a checkpoint or SQLite sidecar")
    package_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    return {
        "status": "PASSED",
        "delivery_grade": "R0_RULE_OFFLINE",
        "episodes": 10,
        "transitions": len(audit.rows),
        "terminal_transitions": audit.terminal_transitions,
        "frame_bundles": len(audit.frame_paths),
        "reward_total": audit.reward_total,
        "input_commands_sent": 0,
        "sqlite_integrity": audit.sqlite_integrity,
        "transition_content_sha256": audit.transition_sha256,
        "frame_view_manifest_sha256": audit.frame_view_sha256,
        "manifest_file_sha256": _file_sha256(manifest_path),
        "summary_file_sha256": _file_sha256(output_dir / "summary.json"),
        "package_bytes": package_bytes,
        "offline_only": True,
        "promoted_checkpoint": None,
    }
