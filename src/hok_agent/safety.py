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
ALLOWED_TESTBED_PATHS = {
    Path("src/hok_agent/mobile_testbed.py"),
    Path("tests/test_mobile_testbed.py"),
}
ALLOWED_TORCH_PATHS = {
    Path("src/hok_agent/bc.py"),
    Path("src/hok_agent/pixel.py"),
    Path("src/hok_agent/alignment.py"),
    Path("src/hok_agent/temporal.py"),
    Path("src/hok_agent/v6_zero.py"),
    Path("src/hok_agent/rich_pixel.py"),
    Path("src/hok_agent/t8.py"),
    Path("src/hok_agent/t8_shadow.py"),
    Path("src/hok_agent/t8_v3.py"),
    Path("src/hok_agent/t8_v4.py"),
    Path("src/hok_agent/t8_v5.py"),
    Path("src/hok_agent/t8_basic_mvp.py"),
    Path("src/hok_agent/combat_feature_cache.py"),
    Path("src/hok_agent/operation_policy.py"),
    Path("tests/test_bc.py"),
    Path("tests/test_pixel.py"),
    Path("tests/test_alignment.py"),
    Path("tests/test_temporal.py"),
    Path("tests/test_v6_zero.py"),
    Path("tests/test_rich.py"),
    Path("tests/test_t8.py"),
    Path("tests/test_t8_shadow.py"),
    Path("tests/test_t8_v4.py"),
    Path("tests/test_t8_v5.py"),
    Path("tests/test_t8_basic_mvp.py"),
    Path("tests/test_operation_policy.py"),
}
ALLOWED_VISION_PATHS = {
    Path("src/hok_agent/pixel.py"),
    Path("src/hok_agent/alignment.py"),
    Path("src/hok_agent/temporal.py"),
    Path("src/hok_agent/v6_zero.py"),
    Path("src/hok_agent/rich_pixel.py"),
    Path("src/hok_agent/t8.py"),
    Path("src/hok_agent/t8_shadow.py"),
    Path("src/hok_agent/t8_v3.py"),
    Path("src/hok_agent/t8_v4.py"),
    Path("src/hok_agent/t8_v5.py"),
    Path("src/hok_agent/t8_basic_mvp.py"),
    Path("src/hok_agent/combat_feature_cache.py"),
    Path("src/hok_agent/operation_policy.py"),
    Path("tests/test_pixel.py"),
    Path("tests/test_alignment.py"),
    Path("tests/test_temporal.py"),
    Path("tests/test_v6_zero.py"),
    Path("tests/test_rich.py"),
    Path("tests/test_t8.py"),
    Path("tests/test_t8_shadow.py"),
    Path("tests/test_t8_v4.py"),
    Path("tests/test_t8_v5.py"),
    Path("tests/test_t8_basic_mvp.py"),
    Path("tests/test_operation_policy.py"),
}
ALLOWED_VIDEO_PATHS = {
    Path("src/hok_agent/shadow.py"),
    Path("src/hok_agent/capture.py"),
    Path("src/hok_agent/alignment.py"),
    Path("src/hok_agent/pre_ingest.py"),
    Path("src/hok_agent/v5_data.py"),
    Path("src/hok_agent/mobile_testbed.py"),
    Path("tests/test_shadow.py"),
    Path("tests/test_capture.py"),
    Path("tests/test_alignment.py"),
    Path("tests/test_mobile_testbed.py"),
}
ALLOWED_ANNOTATION_PATHS = {
    Path("src/hok_agent/alignment.py"),
    Path("src/hok_agent/mobile_testbed.py"),
    Path("tests/test_alignment.py"),
    Path("tests/test_mobile_testbed.py"),
}
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
    python_lines = sum(
        sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
        for path in python_files
    )
    root_markdown = [path for path in files if path.parent == root and path.suffix == ".md"]
    findings: list[str] = []
    if len(root_markdown) != 4:
        findings.append(f"Markdown authority count differs: {len(root_markdown)} != 4")
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
                if name.split(".")[0] == "av" and relative not in ALLOWED_VIDEO_PATHS:
                    findings.append(f"video decoder import outside Shadow modules: {relative}")
                if name.split(".")[0] in {"tkinter", "PIL"} and relative not in (
                    ALLOWED_ANNOTATION_PATHS
                ):
                    findings.append(f"annotation UI import outside alignment module: {relative}")
                if name.split(".")[0] in DENIED_IMPORTS and relative not in ALLOWED_TESTBED_PATHS:
                    findings.append(f"denied import {name}: {relative}")
    return {
        "passed": not findings,
        "files": len(files),
        "python_files": len(python_files),
        "python_lines": python_lines,
        "root_markdown": len(root_markdown),
        "findings": findings,
    }
