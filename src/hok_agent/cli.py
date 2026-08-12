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
    accept_v3 = commands.add_parser(
        "accept-pixel-v3", help="run the RGB-only PixelArena behavior-cloning gate"
    )
    accept_v3.add_argument("--device", choices=("cpu", "cuda"), required=True)
    accept_v3.add_argument("--output-dir", type=Path)
    accept_v3.add_argument("--smoke", action="store_true")
    shadow = commands.add_parser(
        "shadow-video", help="analyze one local recording without client control"
    )
    shadow.add_argument("--input", required=True)
    shadow.add_argument("--model", required=True)
    shadow.add_argument("--output-dir", type=Path, required=True)
    shadow.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    shadow.add_argument("--sample-every", type=int, default=5)
    shadow.add_argument("--max-frames", type=int, default=300)
    live = commands.add_parser(
        "shadow-live", help="read one explicit V4L2 capture node without client control"
    )
    live.add_argument("--input", required=True)
    live.add_argument("--model", required=True)
    live.add_argument("--output-dir", type=Path, required=True)
    live.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    live.add_argument("--capture-size", default="1920x1080")
    live.add_argument("--capture-fps", type=int, default=60)
    live.add_argument("--infer-hz", type=int, default=10)
    live.add_argument("--run-seconds", type=float, default=600.0)
    alignment_smoke = commands.add_parser(
        "alignment-v5-smoke", help="run non-promoting V5 contract checks"
    )
    alignment_smoke.add_argument("--release", type=Path)
    commands.add_parser("temporal-v6-smoke", help="run fail-closed RGB temporal smoke")
    rich = commands.add_parser("accept-rich-v7", help="run Rich PixelArena RGB gate")
    rich.add_argument("--device", choices=("cpu", "cuda"), required=True)
    rich.add_argument("--output-dir", type=Path)
    rich.add_argument("--smoke", action="store_true")
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
        elif args.command == "accept-pixel-v3":
            from hok_agent.pixel import accept_pixel_v3

            result = accept_pixel_v3(args.output_dir, args.device, args.smoke)
        elif args.command == "shadow-video":
            from hok_agent.shadow import analyze_video

            result = analyze_video(
                args.input,
                args.model,
                args.output_dir,
                args.device,
                args.sample_every,
                args.max_frames,
            )
        elif args.command == "shadow-live":
            from hok_agent.capture import run_shadow_live

            def emit(row: dict[str, object]) -> None:
                sequence = int(str(row["sequence"]))
                hypothesis = str(row["raw_model_hypothesis"])
                confidence = float(str(row["confidence"]))
                print(
                    f"\r{sequence:>6}  {hypothesis:<16} "
                    f"{confidence:.3f}  advisory=ABSTAIN",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

            result = run_shadow_live(
                args.input,
                args.model,
                args.output_dir,
                device=args.device,
                capture_size=args.capture_size,
                capture_fps=args.capture_fps,
                infer_hz=args.infer_hz,
                run_seconds=args.run_seconds,
                event_sink=emit,
            )
            print(file=sys.stderr)
        elif args.command == "alignment-v5-smoke":
            from hok_agent.alignment import (
                ACTION_TYPES,
                load_release,
                source_renderer_hash,
            )

            release = None if args.release is None else load_release(args.release)
            result = {
                "status": "PASSED",
                "disposition": "NON_PROMOTING_CONTRACT_SMOKE",
                "actions": ACTION_TYPES,
                "source_renderer_hash": source_renderer_hash(),
                "release_validated": release is not None,
                "real_domain_accuracy_claim": False,
            }
        elif args.command == "temporal-v6-smoke":
            from hok_agent.temporal import cpu_smoke

            result = cpu_smoke()
        elif args.command == "accept-rich-v7":
            from hok_agent.rich_pixel import accept_rich_pixel

            result = accept_rich_pixel(args.output_dir, args.device, args.smoke)
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
