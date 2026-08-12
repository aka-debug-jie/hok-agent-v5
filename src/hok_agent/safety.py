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
    "runs",
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
}
ALLOWED_TORCH_PATHS = {
    Path("src/hok_agent/bc.py"),
    Path("src/hok_agent/pixel.py"),
    Path("tests/test_bc.py"),
    Path("tests/test_pixel.py"),
}
ALLOWED_VISION_PATHS = {Path("src/hok_agent/pixel.py"), Path("tests/test_pixel.py")}
DENIED_MODULE_NAMES = {"android", "client", "device"}
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
    if len(files) > 36:
        findings.append(f"file budget exceeded: {len(files)} > 36")
    if len(python_files) > 22:
        findings.append(f"Python file budget exceeded: {len(python_files)} > 22")
    if python_lines > 4000:
        findings.append(f"Python line budget exceeded: {python_lines} > 4000")
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
                if name.split(".")[0] == "torch" and relative not in ALLOWED_TORCH_PATHS:
                    findings.append(f"torch outside allowed ML modules: {relative}")
                if name.split(".")[0] in {"torchvision", "safetensors"} and relative not in (
                    ALLOWED_VISION_PATHS
                ):
                    findings.append(f"vision training import outside pixel module: {relative}")
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
