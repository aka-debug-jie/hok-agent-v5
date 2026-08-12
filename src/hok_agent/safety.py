from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "reports",
    "build",
    "dist",
}
DENIED_IMPORTS = {
    "adb",
    "ppadb",
    "pyminitouch",
    "scrcpy",
    "pynput",
    "win32api",
    "win32gui",
    "subprocess",
    "socket",
    "torch",
    "torchvision",
}
DENIED_MODULE_NAMES = {"android", "client", "device", "vision"}
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|password|private[_-]?key)\s*[:=]\s*['\"][^'\"]{8,}"
)


def project_files(root: Path = ROOT) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED or part.endswith(".egg-info") for part in path.parts)
        and not path.name.endswith((".pyc", ".pyo"))
    )


def check_project(root: Path = ROOT) -> dict[str, object]:
    files = project_files(root)
    python_files = [path for path in files if path.suffix == ".py"]
    python_lines = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in python_files)
    root_markdown = [path for path in files if path.parent == root and path.suffix == ".md"]
    findings: list[str] = []
    if len(files) > 22:
        findings.append(f"file budget exceeded: {len(files)} > 22")
    if len(python_files) > 14:
        findings.append(f"Python file budget exceeded: {len(python_files)} > 14")
    if python_lines > 1400:
        findings.append(f"Python line budget exceeded: {python_lines} > 1400")
    if len(root_markdown) > 4:
        findings.append(f"Markdown authority budget exceeded: {len(root_markdown)} > 4")
    for path in files:
        relative = path.relative_to(root)
        if any(part.lower() in DENIED_MODULE_NAMES for part in relative.parts):
            findings.append(f"denied module path: {relative}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        if SECRET_PATTERN.search(text):
            findings.append(f"secret-shaped value: {relative}")
        if path.suffix != ".py":
            continue
        tree = ast.parse(text, filename=str(relative))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in DENIED_IMPORTS:
                    findings.append(f"denied import {name}: {relative}")
    return {
        "passed": not findings,
        "files": len(files),
        "python_files": len(python_files),
        "python_lines": python_lines,
        "root_markdown": len(root_markdown),
        "findings": findings,
    }
