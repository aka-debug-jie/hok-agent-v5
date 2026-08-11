"""Safety-bounded command line entry points for bootstrap and control-plane checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hok_agent.artifacts.verification import ArtifactVerificationError, verify_artifact
from hok_agent.config import validate_config_tree
from hok_agent.contracts import LicenseStatus
from hok_agent.control_plane import ControlledOperation, ExternalAccessDenied, ExternalAccessGate
from hok_agent.evaluation.smoke import run_mock_benchmark, run_mock_smoke
from hok_agent.preflight import collect_preflight, write_preflight
from hok_agent.safety import SafetyViolation, scan_repository


def _print_document(document: object) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


def _default_schema(path: Path, root: Path) -> Path | None:
    if path.name == "run_manifest.json":
        return root / "schemas" / "run_manifest.schema.json"
    if path.name == "evaluation_report.json":
        return root / "schemas" / "evaluation_report.schema.json"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hok-agent", description="HoK-Agent V5 infrastructure CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    smoke = subcommands.add_parser("env-smoke", help="run the deterministic mock E0 smoke")
    smoke.add_argument("--config", type=Path, default=Path("configs/run_smoke_v1.yaml"))
    smoke.add_argument("--episodes", type=int, default=None)

    benchmark = subcommands.add_parser("env-benchmark", help="benchmark only the deterministic mock")
    benchmark.add_argument("--config", type=Path, default=Path("configs/run_smoke_v1.yaml"))
    benchmark.add_argument("--episodes", type=int, default=100)

    verify = subcommands.add_parser("verify-artifact", help="verify a self-hashed JSON artifact")
    verify.add_argument("path", type=Path)
    verify.add_argument("--schema", type=Path, default=None)
    verify.add_argument("--root", type=Path, default=Path("."))

    safety = subcommands.add_parser("safety-scan", help="scan executable surfaces and Git ignore rules")
    safety.add_argument("--root", type=Path, default=Path("."))

    preflight = subcommands.add_parser("preflight", help="perform a non-invasive upstream/GameCore preflight")
    preflight.add_argument("--root", type=Path, default=Path("."))
    preflight.add_argument("--output", type=Path, default=Path("reports/m0/upstream_gamecore_preflight.json"))
    preflight.add_argument("--gamecore-path", type=Path, default=None)
    preflight.add_argument("--license-path", type=Path, default=None)
    preflight.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/runtime_inputs_v1.yaml"),
        help="versioned YAML containing only non-secret environment-variable names",
    )
    preflight.add_argument("--probe-upstream", action="store_true")

    validate = subcommands.add_parser("validate-config", help="validate versioned YAML and JSON schemas")
    validate.add_argument("--config-dir", type=Path, default=Path("configs"))

    access_gate = subcommands.add_parser(
        "access-gate",
        help="check a sensitive GameCore operation without connecting to any service",
    )
    access_gate.add_argument("--config", type=Path, default=Path("configs/program_v1.yaml"))
    access_gate.add_argument(
        "--operation",
        required=True,
        choices=[operation.value for operation in ControlledOperation],
    )
    access_gate.add_argument(
        "--runtime-license-status",
        required=True,
        choices=[status.value for status in LicenseStatus],
        help="service-reported metadata only; this is not external authorization evidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "env-smoke":
            if args.episodes is not None and args.episodes < 1:
                raise ValueError("--episodes must be at least 1")
            result = run_mock_smoke(args.config, episodes_override=args.episodes)
            _print_document(result.to_dict())
            return 0 if result.passed else 1
        if args.command == "env-benchmark":
            if args.episodes < 1:
                raise ValueError("--episodes must be at least 1")
            _print_document(run_mock_benchmark(args.config, episodes=args.episodes))
            return 0
        if args.command == "verify-artifact":
            root = args.root.resolve()
            schema = args.schema or _default_schema(args.path, root)
            verification = verify_artifact(args.path, schema, repo_root=root)
            _print_document(verification.to_dict())
            return 0 if verification.passed else 1
        if args.command == "safety-scan":
            scan_result = scan_repository(args.root)
            _print_document(scan_result.to_dict())
            return 0 if scan_result.passed else 1
        if args.command == "preflight":
            root = args.root.resolve()
            output = args.output if args.output.is_absolute() else root / args.output
            runtime_config = (
                args.runtime_config
                if args.runtime_config.is_absolute()
                else root / args.runtime_config
            )
            report = collect_preflight(
                root,
                gamecore_path=args.gamecore_path,
                license_path=args.license_path,
                runtime_config=runtime_config,
                probe_upstream=args.probe_upstream,
            )
            write_preflight(output, report)
            _print_document({"output": str(output), **report})
            return 0
        if args.command == "validate-config":
            for message in validate_config_tree(args.config_dir.resolve()):
                print(message)
            return 0
        if args.command == "access-gate":
            gate = ExternalAccessGate.from_yaml(args.config)
            operation = ControlledOperation(args.operation)
            license_status = LicenseStatus(args.runtime_license_status)
            gate.require(operation, runtime_license_status=license_status)
            _print_document(
                {
                    "operation": operation.value,
                    "runtime_license_status": license_status.value,
                    "status": "ALLOWED_BY_LOCAL_CONTROL_PLANE",
                }
            )
            return 0
    except ExternalAccessDenied as error:
        print(f"error [{error.code}]: {error}", file=sys.stderr)
        return 2
    except (ArtifactVerificationError, SafetyViolation, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command}")
