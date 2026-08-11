from __future__ import annotations

from pathlib import Path

from hok_agent.preflight import collect_preflight


def test_preflight_without_external_inputs_is_waiting_external(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.delenv("HOK_GAMECORE_PATH", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("HOK_LICENSE_PATH", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("GAMECORE_LICENSE_PATH", raising=False)  # type: ignore[attr-defined]
    report = collect_preflight(tmp_path)
    assert report["status"] == "WAITING_EXTERNAL"
    scope = report["scope"]
    assert isinstance(scope, dict)
    assert scope["license_contents_read"] is False
