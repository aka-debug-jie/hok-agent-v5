from __future__ import annotations

from pathlib import Path

from hok_agent.safety import scan_repository


def test_project_safety_scan_passes() -> None:
    root = Path(__file__).parents[1]
    assert scan_repository(root).passed


def test_safety_scan_detects_a_prohibited_executable_term(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / ".gitignore").write_text(
        "\n".join(
            [
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
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "src" / "unsafe.py").write_text("command = 'adb shell input'\n", encoding="utf-8")
    result = scan_repository(tmp_path)
    assert not result.passed
    assert any(finding.rule == "real_client_input" for finding in result.findings)


def test_safety_scan_detects_secret_shaped_text_outside_source(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        "\n".join(
            [
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
            ]
        ),
        encoding="utf-8",
    )
    token = "gh" + "p_" + "A" * 36
    (tmp_path / "README.md").write_text(token, encoding="utf-8")
    result = scan_repository(tmp_path)
    assert any(finding.rule == "github_token" for finding in result.findings)
