from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hok_agent.replay import ReplayError, accept_minimal_v1, record_episode, verify_trace
from hok_agent.safety import check_project
from hok_agent.service import ServiceError

POLICIES = ("null", "random", "scripted")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hok-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="record one public PixelArena trace")
    record.add_argument("--blue", choices=POLICIES, default="scripted")
    record.add_argument("--red", choices=POLICIES, default="null")
    record.add_argument("--seed", type=int, default=101)
    record.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="replay and verify a trace")
    replay.add_argument("path", type=Path)
    accept = commands.add_parser("accept-minimal-v1", help="run the complete minimal gate")
    accept.add_argument("--seed", type=int, default=101)
    accept.add_argument("--output-dir", type=Path)
    accept_v2 = commands.add_parser(
        "accept-minimal-v2-bc", help="run the CPU structured behavior-cloning gate"
    )
    accept_v2.add_argument("--output-dir", type=Path, required=True)
    commands.add_parser("check", help="run size and static safety gates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "record":
            result = record_episode(args.output, args.blue, args.red, args.seed)
        elif args.command == "replay":
            result = verify_trace(args.path)
        elif args.command == "accept-minimal-v1":
            static = check_project()
            if not static["passed"]:
                raise ValueError(f"static checks failed: {static['findings']}")
            result = accept_minimal_v1(args.seed, args.output_dir)
            result["static_checks"] = static
        elif args.command == "accept-minimal-v2-bc":
            from hok_agent.bc import accept_minimal_v2

            result = accept_minimal_v2(args.output_dir)
        else:
            result = check_project()
            if not result["passed"]:
                print(json.dumps(result, indent=2, sort_keys=True))
                return 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, ReplayError, ServiceError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}), file=sys.stderr)
        return 2
