"""Non-invasive upstream/GameCore preflight that never reads license contents."""

from __future__ import annotations

import os
import platform
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from hok_agent.artifacts.hashing import sha256_json

UPSTREAM_REPOSITORY = "https://github.com/tencent-ailab/hok_env"
UPSTREAM_PYTHON_CONSTRAINT = ">=3.6,<3.10"


def _docker_status() -> dict[str, object]:
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"cli_found": False, "daemon_reachable": False}
    version = result.stdout.strip()
    return {
        "cli_found": True,
        "daemon_reachable": result.returncode == 0 and bool(version),
        "server_version": version if result.returncode == 0 else None,
    }


def _wsl_detected() -> bool:
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").casefold()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def _candidate_checkout_paths(repo_root: Path) -> list[Path]:
    candidates = [repo_root / "upstream" / "hok_env", repo_root.parent / "hok_env"]
    return [candidate for candidate in candidates if candidate.is_dir()]


def _optional_path_status(env_names: tuple[str, ...], explicit: Path | None) -> dict[str, object]:
    provided_names = [name for name in env_names if os.environ.get(name)]
    provided = explicit is not None or bool(provided_names)
    selected = explicit
    if selected is None and provided_names:
        selected = Path(os.environ[provided_names[0]])
    return {
        "provided": provided,
        "provided_by_env_var_names": provided_names,
        "exists": selected.exists() if selected is not None else False,
        "is_directory": selected.is_dir() if selected is not None else False,
    }


def _probe_upstream() -> dict[str, object]:
    request = urllib.request.Request(UPSTREAM_REPOSITORY, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"attempted": True, "reachable": True, "http_status": response.status}
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {"attempted": True, "reachable": False, "error_type": type(error).__name__}


def collect_preflight(
    repo_root: Path,
    *,
    gamecore_path: Path | None = None,
    license_path: Path | None = None,
    probe_upstream: bool = False,
) -> dict[str, object]:
    """Collect only path-presence metadata and public runtime facts.

    Paths and environment variable values are deliberately not placed in the report:
    they can be sensitive deployment details. Existence alone is never treated as a
    valid license or a connected GameCore service.
    """

    repo_root = repo_root.resolve()
    checkouts = _candidate_checkout_paths(repo_root)
    gamecore = _optional_path_status(("HOK_GAMECORE_PATH",), gamecore_path)
    license_status = _optional_path_status(("HOK_LICENSE_PATH", "GAMECORE_LICENSE_PATH"), license_path)
    has_required_inputs = bool(checkouts) and bool(gamecore["exists"]) and bool(license_status["exists"])
    status = "PREFLIGHT_INPUTS_PRESENT_UNVERIFIED" if has_required_inputs else "WAITING_EXTERNAL"
    reasons: list[str] = []
    if not checkouts:
        reasons.append("no hok_env checkout found in the bounded project-local search")
    if not gamecore["exists"]:
        reasons.append("no provided GameCore path exists")
    if not license_status["exists"]:
        reasons.append("no provided license path exists")
    if has_required_inputs:
        reasons.append("paths exist but no authorized service health handshake has been performed")
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "upstream_gamecore_preflight",
        "status": status,
        "waiting_reasons": reasons,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "python_constraint": UPSTREAM_PYTHON_CONSTRAINT,
            "checkout_found_count": len(checkouts),
            "remote_probe": _probe_upstream() if probe_upstream else {"attempted": False},
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "wsl_detected": _wsl_detected(),
            "docker": _docker_status(),
        },
        "gamecore": gamecore,
        "license": license_status,
        "scope": {
            "checkout_paths_checked": ["upstream/hok_env", "../hok_env"],
            "license_contents_read": False,
            "license_path_values_recorded": False,
            "gamecore_binary_downloaded": False,
            "actual_service_handshake": False,
        },
    }
    report["report_hash"] = sha256_json(report)
    return report


def write_preflight(path: Path, report: dict[str, object]) -> str:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(report["report_hash"])
