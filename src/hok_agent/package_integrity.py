"""Source-only package manifest generation and verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hok_agent.artifacts.hashing import sha256_file, sha256_json

MANIFEST_FILE = "PACKAGE_MANIFEST.json"
MANIFEST_VERSION = "1"
PACKAGE_ID = "hok-agent-v5"
MANIFEST_VERSION_FIELD = "manifest_version"
PACKAGE_FIELD = "package"
SELF_HASH_FIELD = "self_hash"

_DIR_EXCLUDES = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "build",
    "datasets",
    "checkpoints",
    "dist",
    "htmlcov",
    "logs",
    "replays",
    "secrets",
    "licenses",
    "gamecore",
    "keys",
}
_FILE_EXCLUDES = {
    "license.dat",
    ".env",
    ".coverage",
    ".ds_store",
}
_FILE_SUFFIX_EXCLUDES = {
    ".abs",
    ".avi",
    ".dll",
    ".engine",
    ".exe",
    ".key",
    ".mkv",
    ".mp4",
    ".onnx",
    ".pem",
    ".pt",
    ".pth",
}
_MANIFEST_KEYS = frozenset({MANIFEST_VERSION_FIELD, PACKAGE_FIELD, "files", SELF_HASH_FIELD})


class PackageIntegrityError(ValueError):
    """Raised when a package manifest is malformed or cannot be written."""


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _is_manifest_excluded(relative_path: Path) -> bool:
    parts = tuple(part.lower() for part in relative_path.parts)
    if relative_path.as_posix() == MANIFEST_FILE:
        return True
    name = relative_path.name.casefold()
    if name in _FILE_EXCLUDES or name.startswith(".env."):
        return True
    if relative_path.suffix.lower() in _FILE_SUFFIX_EXCLUDES:
        return True
    if any(part in _DIR_EXCLUDES or part.endswith(".egg-info") for part in parts):
        return True
    return any(parts[index : index + 2] == ("reports", "runtime") for index in range(len(parts) - 1))


def _recorded_files(root: Path) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root)
        if _is_manifest_excluded(relative_path):
            continue
        if path.is_symlink():
            raise PackageIntegrityError(f"controlled delivery path cannot be a symbolic link: {relative_path}")
        if not path.is_file():
            continue
        if relative_path.as_posix() == MANIFEST_FILE:
            continue
        records.append(
            {
                "path": relative_path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


@dataclass(frozen=True, slots=True)
class PackageIntegrityResult:
    root: Path
    manifest_path: Path
    manifest_version: str
    package: str
    recorded_files: int
    controlled_files: int
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "manifest_version": self.manifest_version,
            "package": self.package,
            "recorded_files": self.recorded_files,
            "controlled_files": self.controlled_files,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def _manifest_path(root: Path, manifest_path: Path | None) -> Path:
    expected = root / MANIFEST_FILE
    candidate = expected if manifest_path is None else manifest_path
    if candidate.resolve() != expected.resolve() or candidate.is_symlink() or expected.is_symlink():
        raise PackageIntegrityError("manifest_path must resolve to PACKAGE_MANIFEST.json inside root")
    return expected


def build_manifest(root: Path, *, manifest_path: Path | None = None) -> dict[str, object]:
    root = root.resolve()
    manifest_path = _manifest_path(root, manifest_path)
    files = _recorded_files(root)
    manifest: dict[str, object] = {
        MANIFEST_VERSION_FIELD: MANIFEST_VERSION,
        PACKAGE_FIELD: PACKAGE_ID,
        "files": files,
    }
    manifest[SELF_HASH_FIELD] = sha256_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_recorded_files(files: list[object]) -> tuple[dict[str, dict[str, str | int]], list[str]]:
    records: dict[str, dict[str, str | int]] = {}
    errors: list[str] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append(f"file entry {index} must be an object")
            continue
        record = cast(dict[str, object], entry)
        if set(record) != {"path", "bytes", "sha256"}:
            errors.append(f"file entry {index} must contain exactly path/bytes/sha256")
            continue
        path_value = record["path"]
        bytes_value = record["bytes"]
        sha_value = record["sha256"]
        if (
            not isinstance(path_value, str)
            or not isinstance(sha_value, str)
            or not isinstance(bytes_value, int)
            or isinstance(bytes_value, bool)
            or bytes_value < 0
        ):
            errors.append(f"file entry {index} missing or invalid path/bytes/sha256")
            continue
        relative_path = Path(path_value)
        if not path_value or relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"file entry path is not safe: {path_value}")
            continue
        if path_value in records:
            errors.append(f"file entry path is duplicated: {path_value}")
            continue
        if not _SHA256_RE.fullmatch(sha_value):
            errors.append(f"file entry sha256 is not canonical: {path_value}")
            continue
        records[path_value] = {
            "path": path_value,
            "bytes": bytes_value,
            "sha256": sha_value,
        }
    return records, errors


def check_manifest(
    root: Path,
    *,
    manifest_path: Path | None = None,
) -> PackageIntegrityResult:
    root = root.resolve()
    manifest_path = _manifest_path(root, manifest_path)

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        return PackageIntegrityResult(
            root=root,
            manifest_path=manifest_path,
            manifest_version="",
            package="",
            recorded_files=0,
            controlled_files=0,
            errors=(f"manifest not readable: {error}",),
        )
    except json.JSONDecodeError as error:
        return PackageIntegrityResult(
            root=root,
            manifest_path=manifest_path,
            manifest_version="",
            package="",
            recorded_files=0,
            controlled_files=0,
            errors=(f"manifest is not valid JSON: {error}",),
        )

    if not isinstance(document, dict):
        return PackageIntegrityResult(
            root=root,
            manifest_path=manifest_path,
            manifest_version="",
            package="",
            recorded_files=0,
            controlled_files=0,
            errors=("manifest is not a JSON object",),
        )

    doc = cast(dict[str, Any], document)
    errors: list[str] = []

    unexpected_keys = set(doc).difference(_MANIFEST_KEYS)
    missing_keys = _MANIFEST_KEYS.difference(doc)
    if unexpected_keys or missing_keys:
        errors.append(
            "manifest must contain exactly "
            f"{sorted(_MANIFEST_KEYS)}; unexpected={sorted(unexpected_keys)}, missing={sorted(missing_keys)}"
        )

    manifest_version = doc.get(MANIFEST_VERSION_FIELD)
    if manifest_version != MANIFEST_VERSION:
        errors.append(f"manifest_version must be {MANIFEST_VERSION!r}, got {manifest_version!r}")

    package = doc.get(PACKAGE_FIELD)
    if package != PACKAGE_ID:
        errors.append(f"package must be {PACKAGE_ID!r}, got {package!r}")

    files = doc.get("files")
    if not isinstance(files, list):
        errors.append("manifest files must be an array")
        return PackageIntegrityResult(
            root=root,
            manifest_path=manifest_path,
            manifest_version=str(manifest_version or ""),
            package=str(package or ""),
            recorded_files=0,
            controlled_files=0,
            errors=tuple(errors),
        )

    expected = _recorded_files(root)
    expected_by_path: dict[str, dict[str, str | int]] = {}
    for item in expected:
        expected_by_path[cast(str, item["path"])] = item

    recorded_by_path, record_errors = _validate_recorded_files(cast(list[object], files))
    errors.extend(record_errors)

    expected_self_hash = doc.get(SELF_HASH_FIELD)
    if not isinstance(expected_self_hash, str):
        errors.append("manifest missing self_hash")
    else:
        provided = dict(doc)
        provided.pop(SELF_HASH_FIELD, None)
        canonical_hash = sha256_json(provided)
        if expected_self_hash != canonical_hash:
            errors.append(
                f"self_hash mismatch: expected {expected_self_hash}, computed {canonical_hash}"
            )

    if not errors:
        manifest_paths = set(recorded_by_path)
        expected_paths = set(expected_by_path)
        for missing_path in sorted(expected_paths - manifest_paths):
            errors.append(f"missing controlled file in manifest: {missing_path}")
        for extra_path in sorted(manifest_paths - expected_paths):
            if not Path(extra_path).is_absolute():
                errors.append(f"manifest lists uncontrolled file: {extra_path}")
            else:
                errors.append(f"manifest lists unsafe path: {extra_path}")

    for path, entry in recorded_by_path.items():
        actual_path = root / path
        if not actual_path.is_file():
            errors.append(f"manifest file missing: {path}")
            continue
        bytes_expected = cast(int, entry["bytes"])
        sha_expected = cast(str, entry["sha256"])
        actual_bytes = actual_path.stat().st_size
        if actual_bytes != bytes_expected:
            errors.append(
                f"size mismatch for {path}: expected {bytes_expected}, computed {actual_bytes}"
            )
        actual_sha = sha256_file(actual_path)
        if actual_sha != sha_expected:
            errors.append(f"sha256 mismatch for {path}: expected {sha_expected}, computed {actual_sha}")

    return PackageIntegrityResult(
        root=root,
        manifest_path=manifest_path,
        manifest_version=str(manifest_version or ""),
        package=str(package or ""),
        recorded_files=len(recorded_by_path),
        controlled_files=len(expected_by_path),
        errors=tuple(errors),
    )


def write_manifest(
    root: Path,
    *,
    manifest_path: Path | None = None,
) -> Path:
    root = root.resolve()
    manifest_path = _manifest_path(root, manifest_path)
    build_manifest(root, manifest_path=manifest_path)
    return manifest_path
