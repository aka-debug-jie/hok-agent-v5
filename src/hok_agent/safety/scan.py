"""Repository-level safety and secret gates for executable project surfaces."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


class SafetyViolation(RuntimeError):
    """Raised when a safety scan identifies a prohibited executable surface."""


@dataclass(frozen=True, slots=True)
class ScanFinding:
    rule: str
    path: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "path": self.path, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class SafetyScanResult:
    root: Path
    scanned_files: int
    findings: tuple[ScanFinding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "scanned_files": self.scanned_files,
            "findings": [finding.to_dict() for finding in self.findings],
            "passed": self.passed,
        }


_SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "openai_token": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
}
_REQUIRED_GITIGNORE = {
    "secrets/",
    "licenses/",
    "license.dat",
    "gamecore/",
    "artifacts/",
    "datasets/",
    "checkpoints/",
    "replays/",
    "*.abs",
    "*.mp4",
    "*.pt",
    "*.key",
    "*.pem",
}
_EXECUTABLE_SUFFIXES = {".py", ".sh", ".yml", ".yaml"}


def _load_forbidden_terms() -> dict[str, list[str]]:
    path = Path(__file__).with_name("forbidden_terms.json")
    loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded, dict):
        raise SafetyViolation("forbidden terms configuration must be an object")
    result: dict[str, list[str]] = {}
    for category, values in cast(dict[str, Any], loaded).items():
        if not isinstance(category, str) or not isinstance(values, list):
            raise SafetyViolation("forbidden terms configuration has invalid fields")
        if not all(isinstance(value, str) and value for value in values):
            raise SafetyViolation("forbidden terms must be non-empty strings")
        result[category] = cast(list[str], values)
    return result


def _iter_executable_files(root: Path) -> Iterable[Path]:
    for relative in (Path("src"), Path("services"), Path("scripts"), Path(".github")):
        directory = root / relative
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in _EXECUTABLE_SUFFIXES:
                yield path
    for filename in ("pyproject.toml", "Makefile"):
        path = root / filename
        if path.is_file():
            yield path


def _scan_gitignore(root: Path) -> list[ScanFinding]:
    path = root / ".gitignore"
    if not path.is_file():
        return [ScanFinding("gitignore", ".gitignore", "missing .gitignore")]
    lines = set(path.read_text(encoding="utf-8").splitlines())
    return [
        ScanFinding("gitignore", ".gitignore", f"missing ignore entry {required}")
        for required in sorted(_REQUIRED_GITIGNORE.difference(lines))
    ]


def _scan_restricted_filenames(root: Path) -> list[ScanFinding]:
    restricted = {"license.dat", ".env", "id_rsa"}
    ignored_roots = {".git", ".venv", "artifacts", "datasets", "checkpoints", "replays"}
    findings: list[ScanFinding] = []
    for path in root.rglob("*"):
        if any(part in ignored_roots for part in path.relative_to(root).parts):
            continue
        restricted_suffix = path.suffix.casefold() in {".key", ".pem"}
        if path.is_file() and (path.name.casefold() in restricted or restricted_suffix):
            findings.append(ScanFinding("restricted_filename", str(path.relative_to(root)), path.name))
    return findings


def _iter_secret_scan_files(root: Path) -> Iterable[Path]:
    excluded_roots = {
        ".git",
        ".venv",
        "artifacts",
        "datasets",
        "checkpoints",
        "replays",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    for path in root.rglob("*"):
        if not path.is_file() or any(part in excluded_roots for part in path.relative_to(root).parts):
            continue
        yield path


def scan_repository(root: Path) -> SafetyScanResult:
    """Scan executable sources without exposing matched secret values."""

    root = root.resolve()
    if not root.is_dir():
        raise SafetyViolation(f"repository root does not exist: {root}")
    forbidden_terms = _load_forbidden_terms()
    excluded_self = Path(__file__).resolve()
    findings = _scan_gitignore(root) + _scan_restricted_filenames(root)
    scanned_files = 0
    executable_paths = tuple(_iter_executable_files(root))
    for path in executable_paths:
        if path.resolve() == excluded_self:
            continue
        scanned_files += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(ScanFinding("non_text_executable", str(path.relative_to(root)), "cannot decode UTF-8"))
            continue
        lowered = text.casefold()
        for category, terms in forbidden_terms.items():
            for term in terms:
                if term.casefold() in lowered:
                    findings.append(ScanFinding(category, str(path.relative_to(root)), "prohibited executable term"))
        for rule, pattern in _SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(ScanFinding(rule, str(path.relative_to(root)), "secret-shaped token detected"))
    executable_set = {path.resolve() for path in executable_paths}
    for path in _iter_secret_scan_files(root):
        if path.resolve() in executable_set:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned_files += 1
        for rule, pattern in _SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(ScanFinding(rule, str(path.relative_to(root)), "secret-shaped token detected"))
    return SafetyScanResult(root=root, scanned_files=scanned_files, findings=tuple(findings))
