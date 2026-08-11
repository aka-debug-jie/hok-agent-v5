from __future__ import annotations

import json
import runpy
from pathlib import Path

from hok_agent.envs.rpc import PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_m0_server_stub_constructs_a_mock_server() -> None:
    namespace = runpy.run_path(str(ROOT / "services" / "hok_gamecore" / "server_stub" / "server.py"))
    build_m0_mock_server = namespace["build_m0_mock_server"]
    server = build_m0_mock_server()

    response = json.loads(
        server.handle(json.dumps({"protocol_version": PROTOCOL_VERSION, "method": "health", "payload": {}}))
    )

    assert response["ok"] is True
    assert response["result"]["environment"]["environment_kind"] == "mock"
