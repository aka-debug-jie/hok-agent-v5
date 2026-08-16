from __future__ import annotations

from pathlib import Path

from hok_agent.safety import check_project, project_files


def test_size_and_static_safety_boundaries() -> None:
    report = check_project()
    assert report["passed"], report["findings"]
    assert isinstance(report["files"], int) and report["files"] >= 0
    assert isinstance(report["python_files"], int) and report["python_files"] >= 0
    assert isinstance(report["python_lines"], int) and report["python_lines"] >= 0
    assert report["root_markdown"] == 4


def test_tree_has_no_external_runtime_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    relative = {str(path.relative_to(root)) for path in project_files(root)}
    assert not any(path.startswith(("services/", "vendor/", "third_party/")) for path in relative)
    assert not any("gamecore" in path.lower() for path in relative)


def test_generated_runs_are_outside_project_gate(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "minimal-v2"
    run.mkdir(parents=True)
    (run / "report.json").write_text("{}", encoding="utf-8")
    assert project_files(tmp_path) == []
