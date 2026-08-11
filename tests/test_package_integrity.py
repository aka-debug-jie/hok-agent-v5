from __future__ import annotations

import json
from pathlib import Path

import pytest

from hok_agent.cli import main
from hok_agent.package_integrity import (
    MANIFEST_FILE,
    MANIFEST_VERSION,
    MANIFEST_VERSION_FIELD,
    PACKAGE_ID,
    PackageIntegrityError,
    check_manifest,
    write_manifest,
)


def _sample_project(root: Path) -> None:
    (root / ".git").mkdir()
    (root / "configs").mkdir()
    (root / "configs" / "run_smoke_v1.yaml").write_text("version: 1\n", encoding="utf-8")
    (root / "src" / "module.py").parent.mkdir(parents=True)
    (root / "src" / "module.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "services").mkdir()
    (root / "services" / "service.py").write_text("print('svc')\n", encoding="utf-8")
    (root / "schemas").mkdir()
    (root / "schemas" / "run.schema").write_text("{}", encoding="utf-8")
    (root / "reports" / "m0").mkdir(parents=True)
    (root / "reports" / "m0" / "report.md").write_text("# m0\n", encoding="utf-8")
    (root / "README.md").write_text("# README\n", encoding="utf-8")
    (root / "src" / "__pycache__").mkdir()


def test_package_integrity_roundtrip_and_cli_read_only(tmp_path: Path) -> None:
    _sample_project(tmp_path)
    assert main(["package-integrity", "--root", str(tmp_path), "--write"]) == 0
    manifest_path = tmp_path / MANIFEST_FILE
    assert manifest_path == tmp_path / MANIFEST_FILE
    result = check_manifest(tmp_path)
    assert result.passed
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload[MANIFEST_VERSION_FIELD] == MANIFEST_VERSION
    assert payload["package"] == PACKAGE_ID
    assert isinstance(payload["files"], list)
    assert payload["files"]
    exit_code = main(["package-integrity", "--root", str(tmp_path)])
    assert exit_code == 0


def test_package_integrity_detects_content_tamper(tmp_path: Path) -> None:
    _sample_project(tmp_path)
    manifest_path = write_manifest(tmp_path)
    (tmp_path / "src" / "module.py").write_text("print('tampered')\n", encoding="utf-8")
    result = check_manifest(tmp_path)
    assert not result.passed
    assert any("sha256 mismatch" in error for error in result.errors)
    assert manifest_path.exists()


def test_package_integrity_detects_added_controlled_file(tmp_path: Path) -> None:
    _sample_project(tmp_path)
    write_manifest(tmp_path)
    (tmp_path / "services" / "new_service.py").write_text("print('new')\n", encoding="utf-8")
    result = check_manifest(tmp_path)
    assert not result.passed
    assert any("missing controlled file" in error or "uncontrolled file" in error for error in result.errors)


def test_package_integrity_detects_duplicate_or_unsafe_paths(tmp_path: Path) -> None:
    _sample_project(tmp_path)
    manifest_path = write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"].append(payload["files"][0])
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    duplicate_result = check_manifest(tmp_path)
    assert not duplicate_result.passed
    assert any("duplicated" in error for error in duplicate_result.errors)

    payload["files"][0]["path"] = "/abs/path"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    unsafe_result = check_manifest(tmp_path)
    assert not unsafe_result.passed
    assert any("not safe" in error for error in unsafe_result.errors)


def test_package_integrity_detects_self_hash_tamper(tmp_path: Path) -> None:
    _sample_project(tmp_path)
    manifest_path = write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["self_hash"] = "sha256:0" * 64
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = check_manifest(tmp_path)
    assert not result.passed
    assert any("self_hash mismatch" in error for error in result.errors)


def test_package_integrity_rejects_malformed_entries_and_symbolic_links(tmp_path: Path) -> None:
    _sample_project(tmp_path)
    manifest_path = write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"][0] = "not-an-object"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = check_manifest(tmp_path)
    assert not result.passed
    assert any("must be an object" in error for error in result.errors)

    write_manifest(tmp_path)
    (tmp_path / "src" / "linked.py").symlink_to(tmp_path / "src" / "module.py")
    with pytest.raises(PackageIntegrityError, match="symbolic link"):
        check_manifest(tmp_path)


def test_package_integrity_does_not_record_secret_or_runtime_paths(tmp_path: Path) -> None:
    _sample_project(tmp_path)
    (tmp_path / ".env.local").write_text("SECRET=not-recorded\n", encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "runtime.pt").write_text("not-recorded\n", encoding="utf-8")
    manifest_path = write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in payload["files"]}
    assert ".env.local" not in paths
    assert "artifacts/runtime.pt" not in paths
    assert check_manifest(tmp_path).passed
