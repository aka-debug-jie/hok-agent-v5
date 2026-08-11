"""Technical schema and integrity validation for JSON evidence artifacts.

Passing this verifier never establishes external GameCore authorization.  Runtime
authorization is enforced separately by :mod:`hok_agent.control_plane`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from hok_agent.artifacts.hashing import sha256_file, sha256_json

_ALLOWED_ENVIRONMENT_KINDS = frozenset({"hok_gamecore", "mock", "pixelarena"})


class ArtifactVerificationError(ValueError):
    """Raised when a required artifact integrity property is not satisfied."""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    path: Path
    sha256: str
    schema_valid: bool
    self_hash_valid: bool
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.schema_valid and self.self_hash_valid and not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "schema_valid": self.schema_valid,
            "self_hash_valid": self.self_hash_valid,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def _load_json(path: Path) -> object:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactVerificationError(f"cannot read JSON artifact {path}: {error}") from error


def _load_schema(schema_path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    if schema_path is None:
        return None, []
    try:
        schema = _load_json(schema_path)
    except ArtifactVerificationError as error:
        return None, [str(error)]
    if not isinstance(schema, dict):
        return None, [f"schema {schema_path} is not a JSON object"]
    return cast(dict[str, Any], schema), []


def _schema_errors(document: Mapping[str, Any], schema: dict[str, Any] | None) -> list[str]:
    if schema is None:
        return []
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    return [error.message for error in errors]


def _artifact_kind(document: Mapping[str, Any], schema: dict[str, Any] | None) -> str | None:
    if schema is not None:
        title = schema.get("title")
        if title == "HoK-Agent V5 Run Manifest":
            return "run_manifest"
        if title == "HoK-Agent V5 Evaluation Report":
            return "evaluation_report"
        if title == "HoK-Agent V5 Mock Public Transition Replay":
            return "mock_public_replay"

    if "artifacts" in document:
        return "run_manifest"
    if "suite" in document and "checkpoint" in document and "episodes" in document:
        return "evaluation_report"
    if document.get("artifact_kind") == "mock_public_transition_replay":
        return "mock_public_replay"
    return None


def _self_hash_field(document: Mapping[str, Any], schema: dict[str, Any] | None) -> str | None:
    kind = _artifact_kind(document, schema)
    if kind == "run_manifest":
        return "manifest_hash"
    if kind == "evaluation_report":
        return "report_hash"
    if kind == "mock_public_replay":
        return "artifact_hash"
    has_manifest = "manifest_hash" in document
    has_report = "report_hash" in document
    if has_manifest and has_report:
        return None
    if has_manifest:
        return "manifest_hash"
    if has_report:
        return "report_hash"
    if "artifact_hash" in document:
        return "artifact_hash"
    return None


def _self_hash_error(document: Mapping[str, Any], schema: dict[str, Any] | None) -> str | None:
    field = _self_hash_field(document, schema)
    if field is None:
        return "artifact does not declare a known self-hash field"
    expected = document.get(field)
    if expected is None:
        return f"artifact is missing required self-hash field {field}"
    if not isinstance(expected, str):
        return f"artifact self-hash {field} is not a string"
    actual = sha256_json(document, exclude_key=field)
    if expected != actual:
        return f"artifact self-hash mismatch: expected {expected}, computed {actual}"
    return None


def _run_manifest_artifact_checks(
    document: Mapping[str, Any],
    artifact_path: Path,
    repo_root: Path | None,
) -> list[str]:
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        return []

    candidate_roots: list[Path] = [artifact_path.parent] + list(artifact_path.parents)
    if repo_root is not None:
        candidate_roots = [repo_root] + candidate_roots

    config_rel = document.get("config")
    config_path_text = None
    if isinstance(config_rel, Mapping):
        config_path_value = config_rel.get("path")
        if isinstance(config_path_value, str):
            config_path_text = config_path_value

    if config_path_text is not None:
        resolved_config = Path(config_path_text)
        if resolved_config.is_absolute() or ".." in resolved_config.parts:
            return [f"config path is not safe: {config_path_text}"]

    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            return ["artifact entry is not a JSON object"]
        artifact_path_text = artifact.get("path")
        expected_hash = artifact.get("sha256")
        if not isinstance(artifact_path_text, str) or not isinstance(expected_hash, str):
            return ["artifact entry is missing path or sha256"]

        rel_path = Path(artifact_path_text)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            return [f"artifact path is not safe: {artifact_path_text}"]

        for root in candidate_roots:
            root = root.resolve()
            if config_path_text is not None and not (root / config_path_text).is_file():
                continue
            artifact_file = root / rel_path
            if not artifact_file.exists():
                continue
            if not artifact_file.is_file():
                return [f"artifact path is not a file: {artifact_path_text}"]
            actual = sha256_file(artifact_file)
            if actual != expected_hash:
                return [
                    f"artifact sha256 mismatch: expected {artifact_path_text}={expected_hash}, computed {actual}"
                ]
            break
        else:
            return [f"artifact path not found in repo: {artifact_path_text}"]
    return []


def _environment_kind_checks(document: Mapping[str, Any], *, evaluation: bool) -> list[str]:
    environment = document.get("environment")
    if not isinstance(environment, Mapping):
        return ["artifact environment is not an object"]
    kind = environment.get("kind")
    if not isinstance(kind, str) or kind not in _ALLOWED_ENVIRONMENT_KINDS:
        return ["artifact environment kind is not an approved typed environment kind"]
    if kind == "hok_gamecore" and environment.get("license_status") != "valid":
        return ["hok_gamecore artifact requires runtime license_status=valid"]
    if kind in {"mock", "pixelarena"}:
        if evaluation:
            if document.get("disposition") != "DIAGNOSTIC_ONLY":
                return ["mock/pixelarena evaluation must use DIAGNOSTIC_ONLY disposition"]
        elif document.get("formal") is not False:
            return ["mock/pixelarena run manifest must be non-formal"]
    return []


def _evaluation_report_gate_checks(document: Mapping[str, Any]) -> list[str]:
    environment = document.get("environment")
    gate = document.get("gate")
    if not isinstance(environment, Mapping) or not isinstance(gate, Mapping):
        return []
    if environment.get("kind") not in {"mock", "pixelarena"}:
        return []
    checks = gate.get("checks")
    if not isinstance(checks, Mapping):
        return []
    values = list(checks.values())
    if not values:
        return []
    all_checks_pass = all(isinstance(check, bool) and check for check in values)
    if not isinstance(gate.get("pass"), bool) or gate.get("pass") != all_checks_pass:
        return ["mock report gate.pass must equal all checks are true"]
    return []

def verify_artifact(
    path: Path,
    schema_path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> VerificationResult:
    """Verify JSON schema plus a canonical self-hash for a manifest or report."""

    if not path.is_file():
        raise ArtifactVerificationError(f"artifact does not exist: {path}")

    document = _load_json(path)
    if not isinstance(document, Mapping):
        return VerificationResult(
            path=path,
            sha256=sha256_file(path),
            schema_valid=False,
            self_hash_valid=False,
            errors=("artifact is not a JSON object",),
        )

    cast_document = cast(dict[str, Any], document)
    schema, schema_load_errors = _load_schema(schema_path)
    errors = list(schema_load_errors)
    errors.extend(_schema_errors(cast_document, schema))
    self_hash_error = _self_hash_error(cast_document, schema)
    if self_hash_error is not None:
        errors.append(self_hash_error)

    kind = _artifact_kind(cast_document, schema)
    if kind == "run_manifest":
        errors.extend(_environment_kind_checks(cast_document, evaluation=False))
        errors.extend(_run_manifest_artifact_checks(cast_document, path, repo_root))
    if kind == "evaluation_report":
        errors.extend(_environment_kind_checks(cast_document, evaluation=True))
        errors.extend(_evaluation_report_gate_checks(cast_document))

    return VerificationResult(
        path=path,
        sha256=sha256_file(path),
        schema_valid=not errors,
        self_hash_valid=self_hash_error is None,
        errors=tuple(errors),
    )


def write_hashed_json(
    path: Path,
    document: Mapping[str, object],
    *,
    self_hash_field: str,
    schema_path: Path | None = None,
) -> str:
    """Write canonical JSON with a self-hash calculated over all other fields."""

    output: dict[str, object] = dict(document)
    output.pop(self_hash_field, None)
    output[self_hash_field] = sha256_json(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = verify_artifact(path, schema_path)
    if not result.passed:
        raise ArtifactVerificationError("; ".join(result.errors))
    return result.sha256
