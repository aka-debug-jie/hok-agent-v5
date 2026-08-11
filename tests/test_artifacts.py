from __future__ import annotations

import json
from pathlib import Path

from hok_agent.artifacts.verification import verify_artifact, write_hashed_json


def test_self_hashed_json_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_report.json"
    write_hashed_json(path, {"schema_version": 1, "kind": "diagnostic"}, self_hash_field="report_hash")
    assert verify_artifact(path).passed
    document = json.loads(path.read_text(encoding="utf-8"))
    document["kind"] = "modified"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert not verify_artifact(path).passed
