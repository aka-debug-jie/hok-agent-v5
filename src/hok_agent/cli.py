from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from hok_agent.replay import ReplayError, accept_minimal_v1, record_episode, verify_trace
from hok_agent.safety import check_project
from hok_agent.service import ServiceError

POLICIES = ("null", "random", "scripted")


def _v5_source_model(source_dir: Path) -> Path:
    try:
        payload = json.loads((source_dir / "source.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("V5 source metadata is unavailable") from exc
    name = payload.get("selected_model_path") if isinstance(payload, dict) else None
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("V5 source metadata has no safe selected model name")
    return source_dir / name


def _v5_target_shards(target_dir: Path) -> tuple[Path, ...]:
    paths = tuple(sorted((target_dir / "shards").glob("*.npz")))
    if not paths:
        raise ValueError("V5 target directory has no shards")
    return paths


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
    mobile = commands.add_parser(
        "mobile-testbed", help="bounded ADB loop for the owner-authorized test app"
    )
    mobile.add_argument("--serial", required=True)
    mobile.add_argument("--model", type=Path, required=True)
    mobile.add_argument("--output-dir", type=Path, required=True)
    mobile.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    mobile.add_argument("--run-seconds", type=float, default=60.0)
    mobile.add_argument("--infer-hz", type=int, default=1)
    mobile.add_argument("--min-confidence", type=float, default=0.90)
    mobile.add_argument(
        "--layout", type=Path, default=Path("configs/mobile_testbed_layout.local.json")
    )
    mobile.add_argument("--capture-mode", choices=("adb-png", "scrcpy-v4l2"), default="adb-png")
    mobile.add_argument("--video-node", type=Path)
    mobile.add_argument("--stream-fps", type=int, default=30)
    mobile.add_argument("--enable-input", action="store_true")
    mobile.add_argument("--max-actions", type=int, default=0)
    demonstrate = commands.add_parser(
        "mobile-demonstrate",
        help="record keyboard-issued factorized actions for the owner-authorized test app",
    )
    demonstrate.add_argument("--serial", required=True)
    demonstrate.add_argument("--output-dir", type=Path, required=True)
    demonstrate.add_argument("--run-seconds", type=float, default=300.0)
    demonstrate.add_argument("--max-actions", type=int, default=300)
    demonstrate.add_argument("--shard-size", type=int, default=256)
    demonstrate.add_argument(
        "--layout", type=Path, default=Path("configs/mobile_testbed_layout.local.json")
    )
    demonstrate.add_argument(
        "--capture-mode", choices=("adb-png", "scrcpy-v4l2"), default="adb-png"
    )
    demonstrate.add_argument("--video-node", type=Path)
    demonstrate.add_argument("--stream-fps", type=int, default=30)
    demonstrate.add_argument("--scripted-seed", type=int)
    demonstrate.add_argument("--scripted-interval-seconds", type=float, default=1.5)
    demonstrate_v2 = commands.add_parser(
        "mobile-demonstrate-keyboard-v2",
        help="record 16-frame guarded keyboard actions for the T8-v2 formal lineage",
    )
    demonstrate_v2.add_argument("--serial", required=True)
    demonstrate_v2.add_argument("--output-dir", type=Path, required=True)
    demonstrate_v2.add_argument("--layout", type=Path, required=True)
    demonstrate_v2.add_argument("--video-node", type=Path, required=True)
    demonstrate_v2.add_argument("--run-seconds", type=float, default=300.0)
    demonstrate_v2.add_argument("--max-actions", type=int, default=600)
    demonstrate_v2.add_argument("--shard-size", type=int, default=256)
    demonstrate_v2.add_argument("--stream-fps", type=int, default=30)
    demonstrate_v2.add_argument("--formal-session", action="store_true")
    demonstrate_v2.add_argument("--scripted-seed", type=int)
    demonstrate_v2.add_argument("--scripted-interval-seconds", type=float, default=1.0)
    demonstrate_rgb_v25 = commands.add_parser(
        "mobile-demonstrate-rgb-teacher-v25",
        help="record RGB-conditioned combat decisions through the guarded ADB executor",
    )
    demonstrate_rgb_v25.add_argument("--serial", required=True)
    demonstrate_rgb_v25.add_argument("--output-dir", type=Path, required=True)
    demonstrate_rgb_v25.add_argument("--layout", type=Path, required=True)
    demonstrate_rgb_v25.add_argument("--video-node", type=Path, required=True)
    demonstrate_rgb_v25.add_argument("--teacher-report", type=Path, required=True)
    demonstrate_rgb_v25.add_argument("--run-seconds", type=float, required=True)
    demonstrate_rgb_v25.add_argument("--enable-input", action="store_true")
    demonstrate_rgb_v25.add_argument("--max-actions", type=int, default=0)
    demonstrate_rgb_v25.add_argument("--shard-size", type=int, default=256)
    demonstrate_rgb_v25.add_argument("--stream-fps", type=int, default=30)
    demonstrate_rgb_v25.add_argument("--formal-session", action="store_true")
    demonstrate_rgb_v25.add_argument("--warmup-basic-attack", action="store_true")
    demonstrate_rgb_v25.add_argument("--patrol", action="store_true")
    demonstrate_rgb_v25.add_argument("--balanced-actions", action="store_true")
    demonstrate_v21 = commands.add_parser(
        "mobile-demonstrate-keyboard-v2-live",
        help="record continuous keydown/keyup actions over one pinned scrcpy 1.25 session",
    )
    demonstrate_v21.add_argument("--serial", required=True)
    demonstrate_v21.add_argument("--output-dir", type=Path, required=True)
    demonstrate_v21.add_argument("--layout", type=Path, required=True)
    demonstrate_v21.add_argument("--run-seconds", type=float, required=True)
    demonstrate_v21.add_argument("--max-actions", type=int)
    demonstrate_v21.add_argument("--shard-size", type=int, default=256)
    demonstrate_v21.add_argument("--stream-fps", type=int, default=30)
    demonstrate_v21.add_argument("--countdown-seconds", type=float, default=3.0)
    demonstrate_v21.add_argument("--formal-session", action="store_true")
    demonstrate_v21.add_argument("--diagnostic-control-smoke", action="store_true")
    demonstrate_v21.add_argument("--diagnostic-inverse-probe", action="store_true")
    touch_discover = commands.add_parser(
        "mobile-touch-discover",
        help="list Type-A/Type-B touch candidates for one owner confirmation",
    )
    touch_discover.add_argument("--serial", required=True)
    touch_probe = commands.add_parser(
        "mobile-touch-probe", help="bounded read-only touch probe for one confirmed device"
    )
    touch_probe.add_argument("--serial", required=True)
    touch_probe.add_argument("--touch-device", required=True)
    touch_probe.add_argument("--touch-max-slots", type=int, required=True)
    touch_probe.add_argument("--touch-max-x", type=int, required=True)
    touch_probe.add_argument("--touch-max-y", type=int, required=True)
    touch_probe.add_argument("--touch-protocol", choices=("type_a", "type_b"), required=True)
    touch_probe.add_argument("--run-seconds", type=float, default=15.0)
    touch_calibrate = commands.add_parser(
        "mobile-touch-calibrate", help="derive one owner-confirmed raw-touch coordinate transform"
    )
    touch_calibrate.add_argument("--serial", required=True)
    touch_calibrate.add_argument("--touch-device", required=True)
    touch_calibrate.add_argument("--touch-max-slots", type=int, required=True)
    touch_calibrate.add_argument("--touch-max-x", type=int, required=True)
    touch_calibrate.add_argument("--touch-max-y", type=int, required=True)
    touch_calibrate.add_argument("--touch-protocol", choices=("type_a", "type_b"), required=True)
    touch_calibrate.add_argument("--layout", type=Path, required=True)
    touch_calibrate.add_argument("--output", type=Path, required=True)
    touch_demonstrate = commands.add_parser(
        "mobile-demonstrate-touch", help="record direct owner touches without sending input"
    )
    touch_demonstrate.add_argument("--serial", required=True)
    touch_demonstrate.add_argument("--touch-device", required=True)
    touch_demonstrate.add_argument("--touch-max-slots", type=int, required=True)
    touch_demonstrate.add_argument("--touch-max-x", type=int, required=True)
    touch_demonstrate.add_argument("--touch-max-y", type=int, required=True)
    touch_demonstrate.add_argument("--touch-protocol", choices=("type_a", "type_b"), required=True)
    touch_demonstrate.add_argument("--touch-calibration", type=Path, required=True)
    touch_demonstrate.add_argument("--output-dir", type=Path, required=True)
    touch_demonstrate.add_argument("--layout", type=Path, required=True)
    touch_demonstrate.add_argument("--video-node", type=Path, required=True)
    touch_demonstrate.add_argument("--run-seconds", type=float, default=300.0)
    touch_demonstrate.add_argument("--max-samples", type=int, default=3000)
    touch_demonstrate.add_argument("--shard-size", type=int, default=256)
    touch_demonstrate.add_argument("--stream-fps", type=int, default=30)
    touch_demonstrate.add_argument("--formal-session", action="store_true")
    touch_demonstrate.add_argument(
        "--semantic-smoke",
        action="store_true",
        help="run the fixed 20-second direct-touch semantic gate without a formal session",
    )
    calibrate = commands.add_parser(
        "mobile-layout-calibrate",
        help="guide bounded owner-confirmed calibration for the self-built test app",
    )
    calibrate.add_argument("--serial", required=True)
    calibrate.add_argument("--layout", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--video-node", type=Path, required=True)
    calibrate.add_argument("--stream-fps", type=int, default=30)
    calibrate.add_argument("--manual-points", action="store_true")
    t8_train = commands.add_parser(
        "t8-train-bc", help="train offline factorized BC from owner-operated T8 demonstrations"
    )
    t8_train.add_argument("--dataset-root", type=Path, required=True)
    t8_train.add_argument("--output-dir", type=Path, required=True)
    t8_train.add_argument("--v5-source-dir", type=Path, required=True)
    t8_train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    t8_train.add_argument("--epochs", type=int, default=20)
    t8_train.add_argument("--batch-size", type=int, default=32)
    t8_split = commands.add_parser(
        "t8-freeze-split", help="freeze the required 4/2/2 T8 demonstration split manifest"
    )
    t8_split.add_argument("--dataset-root", type=Path, required=True)
    t8_split.add_argument("--output", type=Path, required=True)
    t8_v2_adapter = commands.add_parser(
        "t8-v2-video-adapt", help="adapt a V5-initialized encoder from video-train only"
    )
    t8_v2_adapter.add_argument("--v5-source-dir", type=Path, required=True)
    t8_v2_adapter.add_argument("--target-dir", type=Path, required=True)
    t8_v2_adapter.add_argument("--output-dir", type=Path, required=True)
    t8_v2_adapter.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    t8_v2_adapter.add_argument("--batch-size", type=int, default=128)
    t8_v2_freeze = commands.add_parser("t8-v2-freeze-split", help="freeze the T8-v2 8/2/2 split")
    t8_v2_freeze.add_argument("--dataset-root", type=Path, required=True)
    t8_v2_freeze.add_argument("--output", type=Path, required=True)
    t8_v21_freeze = commands.add_parser(
        "t8-v2-live-freeze-split", help="freeze every T8-v2.1 live session deterministically"
    )
    t8_v21_freeze.add_argument("--dataset-root", type=Path, required=True)
    t8_v21_freeze.add_argument("--output", type=Path, required=True)
    t8_v21_pilot_freeze = commands.add_parser(
        "t8-v2-live-pilot-freeze", help="freeze exactly three T8-v2.1 sessions as 2 train / 1 dev"
    )
    t8_v21_pilot_freeze.add_argument("--dataset-root", type=Path, required=True)
    t8_v21_pilot_freeze.add_argument("--output", type=Path, required=True)
    t8_v2_pilot = commands.add_parser("t8-v2-pilot", help="run one seed T8-v2 train/dev pilot")
    t8_v2_pilot.add_argument("--dataset-root", type=Path, required=True)
    t8_v2_pilot.add_argument("--adapter-checkpoint", type=Path, required=True)
    t8_v2_pilot.add_argument("--output-dir", type=Path, required=True)
    t8_v2_pilot.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    t8_v2_pilot.add_argument("--batch-size", type=int, default=32)
    t8_v21_pilot = commands.add_parser(
        "t8-v2-live-pilot", help="run the paired seed-0 T8-v2.1 live-data pilot"
    )
    t8_v21_pilot.add_argument("--dataset-root", type=Path, required=True)
    t8_v21_pilot.add_argument("--split", type=Path, required=True)
    t8_v21_pilot.add_argument("--adapter-checkpoint", type=Path, required=True)
    t8_v21_pilot.add_argument("--output-dir", type=Path, required=True)
    t8_v21_pilot.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    t8_v21_pilot.add_argument("--batch-size", type=int, default=32)
    inverse_probe = commands.add_parser(
        "t8-v2-live-inverse-materialize",
        help="materialize a diagnostic before/action/after inverse-dynamics probe",
    )
    inverse_probe.add_argument("--session-dir", type=Path, required=True)
    inverse_probe.add_argument("--layout", type=Path, required=True)
    inverse_probe.add_argument("--output-dir", type=Path, required=True)
    video_three_class = commands.add_parser(
        "t8-video-three-class-materialize",
        help="filter the frozen causal video candidates to the validated three-class scope",
    )
    video_three_class.add_argument("--source-dir", type=Path, required=True)
    video_three_class.add_argument("--inverse-report", type=Path, required=True)
    video_three_class.add_argument("--output-dir", type=Path, required=True)
    video_three_class.add_argument("--retrospective", action="store_true")
    video_three_class_pilot = commands.add_parser(
        "t8-video-three-class-pilot",
        help="run the paired seed-0 strict-causal three-class learnability diagnostic",
    )
    video_three_class_pilot.add_argument("--dataset-root", type=Path, required=True)
    video_three_class_pilot.add_argument("--adapter-checkpoint", type=Path, required=True)
    video_three_class_pilot.add_argument("--output-dir", type=Path, required=True)
    video_three_class_pilot.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    video_three_class_pilot.add_argument("--batch-size", type=int, default=64)
    video_three_class_pilot.add_argument("--retrospective", action="store_true")
    retrospective_roi = commands.add_parser(
        "t8-video-retrospective-roi-evaluate",
        help="evaluate the frozen session-center retrospective action recognizer",
    )
    retrospective_roi.add_argument("--dataset-root", type=Path, required=True)
    retrospective_roi.add_argument("--probe-report", type=Path, required=True)
    retrospective_roi.add_argument("--inverse-report", type=Path, required=True)
    retrospective_roi.add_argument("--output-dir", type=Path, required=True)
    retrospective_verify = commands.add_parser(
        "t8-retrospective-baseline-verify",
        help="verify the frozen T8 retrospective-v1 evidence bundle",
    )
    retrospective_verify.add_argument("--baseline-dir", type=Path, required=True)
    retrospective_batch = commands.add_parser(
        "t8-retrospective-batch",
        help="scan derived RGB train/dev sessions into retrospective action events",
    )
    retrospective_batch.add_argument("--target-dir", type=Path, required=True)
    retrospective_batch.add_argument("--baseline-dir", type=Path, required=True)
    retrospective_batch.add_argument("--layout", type=Path, required=True)
    retrospective_batch.add_argument("--split", choices=("train", "dev"), required=True)
    retrospective_batch.add_argument("--session-hash", action="append", default=[])
    retrospective_batch.add_argument("--output-dir", type=Path, required=True)
    retrospective_batch_verify = commands.add_parser(
        "t8-retrospective-batch-verify",
        help="re-read and verify a retrospective event/QC batch",
    )
    retrospective_batch_verify.add_argument("--batch-dir", type=Path, required=True)
    retrospective_calibration = commands.add_parser(
        "t8-retrospective-calibrate-v2",
        help="select and validate the bounded basic-attack/skill3 retrospective upgrade",
    )
    retrospective_calibration.add_argument("--dataset-root", type=Path, required=True)
    retrospective_calibration.add_argument("--probe-report", type=Path, required=True)
    retrospective_calibration.add_argument("--layout", type=Path, required=True)
    retrospective_calibration.add_argument("--baseline-dir", type=Path, required=True)
    retrospective_calibration.add_argument(
        "--inverse-calibration", type=Path, action="append", required=True
    )
    retrospective_calibration.add_argument("--inverse-holdout", type=Path, required=True)
    retrospective_calibration.add_argument("--output-dir", type=Path, required=True)
    causal_materialize = commands.add_parser(
        "t8-causal-video-materialize",
        help="materialize frozen-encoder 16-frame pre-action train/dev features",
    )
    causal_materialize.add_argument("--target-dir", type=Path, required=True)
    causal_materialize.add_argument("--train-events-dir", type=Path, required=True)
    causal_materialize.add_argument("--dev-events-dir", type=Path, required=True)
    causal_materialize.add_argument("--adapter-checkpoint", type=Path, required=True)
    causal_materialize.add_argument("--output-dir", type=Path, required=True)
    causal_materialize.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    causal_materialize.add_argument("--batch-size", type=int, default=256)
    causal_pilot = commands.add_parser(
        "t8-causal-video-pilot",
        help="run the seed-0 causal four-class pilot and frozen negative controls",
    )
    causal_pilot.add_argument("--dataset-root", type=Path, required=True)
    causal_pilot.add_argument("--adapter-checkpoint", type=Path, required=True)
    causal_pilot.add_argument("--output-dir", type=Path, required=True)
    causal_pilot.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    causal_pilot.add_argument("--batch-size", type=int, default=256)
    causal_diagnostic = commands.add_parser(
        "t8-causal-video-diagnose",
        help="decompose causal learnability into fit, binary, action-only, and domain probes",
    )
    causal_diagnostic.add_argument("--dataset-root", type=Path, required=True)
    causal_diagnostic.add_argument("--pilot-dir", type=Path, required=True)
    causal_diagnostic.add_argument("--output-dir", type=Path, required=True)
    causal_diagnostic.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    causal_diagnostic.add_argument("--batch-size", type=int, default=256)
    causal_pixel_materialize = commands.add_parser(
        "t8-causal-pixel-materialize",
        help="materialize oriented pre-action RGB views with matched same-session waits",
    )
    causal_pixel_materialize.add_argument("--target-dir", type=Path, required=True)
    causal_pixel_materialize.add_argument("--train-events-dir", type=Path, required=True)
    causal_pixel_materialize.add_argument("--dev-events-dir", type=Path, required=True)
    causal_pixel_materialize.add_argument("--output-dir", type=Path, required=True)
    causal_pixel_probe = commands.add_parser(
        "t8-causal-pixel-probe",
        help="fine-tune the final ResNet block for bounded causal pixel learnability",
    )
    causal_pixel_probe.add_argument("--dataset-root", type=Path, required=True)
    causal_pixel_probe.add_argument("--adapter-checkpoint", type=Path, required=True)
    causal_pixel_probe.add_argument("--output-dir", type=Path, required=True)
    causal_pixel_probe.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    causal_pixel_probe.add_argument("--batch-size", type=int, default=64)
    visual_teacher_replay = commands.add_parser(
        "t8-visual-teacher-replay",
        help="calibrate and validate the deterministic RGB teacher without device input",
    )
    visual_teacher_replay.add_argument("--dataset-root", type=Path, required=True)
    visual_teacher_replay.add_argument("--pixel-probe-dir", type=Path, required=True)
    visual_teacher_replay.add_argument("--layout", type=Path, required=True)
    visual_teacher_replay.add_argument("--output-dir", type=Path, required=True)
    onset_audit = commands.add_parser(
        "t8-visible-onset-audit",
        help="audit the first visible combat-button onset without device input",
    )
    onset_audit.add_argument("--target-dir", type=Path, required=True)
    onset_audit.add_argument("--train-events-dir", type=Path, required=True)
    onset_audit.add_argument("--dev-events-dir", type=Path, required=True)
    onset_audit.add_argument("--layout", type=Path, required=True)
    onset_audit.add_argument("--calibration-report", type=Path, required=True)
    onset_audit.add_argument("--output-dir", type=Path, required=True)
    combat_materialize = commands.add_parser(
        "t8-combat-causal-materialize",
        help="materialize 32-frame combat-causal RGB windows after onset audit",
    )
    combat_materialize.add_argument("--target-dir", type=Path, required=True)
    combat_materialize.add_argument("--onset-audit-dir", type=Path, required=True)
    combat_materialize.add_argument("--output-dir", type=Path, required=True)
    combat_materialize.add_argument("--diagnostic-only", action="store_true")
    combat_pilot = commands.add_parser(
        "t8-combat-causal-pilot",
        help="run the bounded seed-0 temporal RGB combat-subpolicy pilot",
    )
    combat_pilot.add_argument("--dataset-root", type=Path, required=True)
    combat_pilot.add_argument("--adapter-checkpoint", type=Path, required=True)
    combat_pilot.add_argument("--output-dir", type=Path, required=True)
    combat_pilot.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    combat_pilot.add_argument("--batch-size", type=int, default=8)
    v25_freeze = commands.add_parser(
        "t8-v25-freeze-split",
        help="freeze the RGB-conditioned T8-v2.5 pilot or 8/2/2 session split",
    )
    v25_freeze.add_argument("--dataset-root", type=Path, required=True)
    v25_freeze.add_argument("--output", type=Path, required=True)
    v25_freeze.add_argument("--pilot", action="store_true")
    v25_pilot = commands.add_parser(
        "t8-v25-pilot",
        help="run one RGB-conditioned 32-frame combat learnability pilot",
    )
    v25_pilot.add_argument("--dataset-root", type=Path, required=True)
    v25_pilot.add_argument("--split", type=Path, required=True)
    v25_pilot.add_argument("--adapter-checkpoint", type=Path, required=True)
    v25_pilot.add_argument("--output-dir", type=Path, required=True)
    v25_pilot.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v25_pilot.add_argument("--batch-size", type=int, default=8)
    v25_pilot.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    v26_select = commands.add_parser(
        "t8-v26-select", help="select the final T8-v2.6 model from three dev reports"
    )
    v26_select.add_argument("--run-root", type=Path, required=True)
    v26_select.add_argument("--output", type=Path, required=True)
    v26_evaluate = commands.add_parser(
        "t8-v26-evaluate-offline",
        help="run the one-time sealed T8-v2.6 evaluation on the frozen test sessions",
    )
    v26_evaluate.add_argument("--dataset-root", type=Path, required=True)
    v26_evaluate.add_argument("--split", type=Path, required=True)
    v26_evaluate.add_argument("--run-root", type=Path, required=True)
    v26_evaluate.add_argument("--selection", type=Path, required=True)
    v26_evaluate.add_argument("--output", type=Path, required=True)
    v26_evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v26_evaluate.add_argument("--batch-size", type=int, default=8)
    v27_calibration = commands.add_parser(
        "t8-v27-calibration-pilot",
        help="fit a diagnostic head-only calibration on independent current-scene sessions",
    )
    v27_calibration.add_argument("--dataset-root", type=Path, required=True)
    v27_calibration.add_argument("--train-session", type=Path, required=True)
    v27_calibration.add_argument("--dev-session", type=Path, required=True)
    v27_calibration.add_argument("--source-model", type=Path, required=True)
    v27_calibration.add_argument("--output-dir", type=Path, required=True)
    v27_calibration.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v27_calibration.add_argument("--batch-size", type=int, default=8)
    v27_freeze = commands.add_parser(
        "t8-v27-freeze", help="freeze all failed T8-v2.7 calibration evidence"
    )
    v27_freeze.add_argument("--report", type=Path, action="append", required=True)
    v27_freeze.add_argument("--output-dir", type=Path, required=True)
    v3_materialize = commands.add_parser(
        "t8-v3-state-materialize", help="materialize RGB-observable T8-v3 video states"
    )
    v3_materialize.add_argument("--feature-root", type=Path, required=True)
    v3_materialize.add_argument("--target-root", type=Path, required=True)
    v3_materialize.add_argument("--teacher-report", type=Path, required=True)
    v3_materialize.add_argument("--layout", type=Path, required=True)
    v3_materialize.add_argument("--output-dir", type=Path, required=True)
    v3_train = commands.add_parser(
        "t8-v3-state-train", help="train the frozen single-seed T8-v3 state model"
    )
    v3_train.add_argument("--dataset-root", type=Path, required=True)
    v3_train.add_argument("--adapter-checkpoint", type=Path, required=True)
    v3_train.add_argument("--output-dir", type=Path, required=True)
    v3_train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v3_train.add_argument("--batch-size", type=int, default=256)
    v3_train.add_argument("--seed", type=int, choices=(0,), default=0)
    v3_train.add_argument("--epochs", type=int, choices=(8,), default=8)
    v3_replay = commands.add_parser(
        "t8-v3-hybrid-replay", help="run zero-control deterministic hybrid replay on video-dev"
    )
    v3_replay.add_argument("--dataset-root", type=Path, required=True)
    v3_replay.add_argument("--model", type=Path, required=True)
    v3_replay.add_argument("--training-report", type=Path, required=True)
    v3_replay.add_argument("--output-dir", type=Path, required=True)
    v3_replay.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v3_replay.add_argument("--batch-size", type=int, default=256)
    v4_contract = commands.add_parser(
        "t8-v4-contract-check", help="verify the frozen zero-label T8-v4 contracts"
    )
    v4_source = commands.add_parser(
        "t8-v4-source-teacher-train", help="train the independent PixelArena source teacher"
    )
    v4_source.add_argument("--adapter-checkpoint", type=Path, required=True)
    v4_source.add_argument("--layout", type=Path, required=True)
    v4_source.add_argument("--output-dir", type=Path, required=True)
    v4_source.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v4_source.add_argument("--batch-size", type=int, default=64)
    v4_source.add_argument("--epochs", type=int, choices=(8,), default=8)
    v4_materialize = commands.add_parser(
        "t8-v4-pseudolabel-materialize", help="materialize dual-teacher consensus targets"
    )
    v4_materialize.add_argument("--feature-root", type=Path, required=True)
    v4_materialize.add_argument("--target-root", type=Path, required=True)
    v4_materialize.add_argument("--rule-teacher-report", type=Path, required=True)
    v4_materialize.add_argument("--source-teacher-model", type=Path, required=True)
    v4_materialize.add_argument("--source-teacher-report", type=Path, required=True)
    v4_materialize.add_argument("--layout", type=Path, required=True)
    v4_materialize.add_argument("--output-dir", type=Path, required=True)
    v4_materialize.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v4_materialize.add_argument("--batch-size", type=int, default=256)
    v4_audit = commands.add_parser(
        "t8-v4-weak-audit", help="audit anonymous T8-v4 consensus shards and coverage"
    )
    v4_audit.add_argument("--dataset-root", type=Path, required=True)
    v4_audit.add_argument("--output", type=Path, required=True)
    v4_diagnose = commands.add_parser(
        "t8-v4-seed0-diagnose", help="run the frozen zero-label model ladder and controls"
    )
    v4_diagnose.add_argument("--dataset-root", type=Path, required=True)
    v4_diagnose.add_argument("--target-root", type=Path, required=True)
    v4_diagnose.add_argument("--adapter-checkpoint", type=Path, required=True)
    v4_diagnose.add_argument("--weak-audit-report", type=Path, required=True)
    v4_diagnose.add_argument("--output-dir", type=Path, required=True)
    v4_diagnose.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v4_diagnose.add_argument("--batch-size", type=int, default=256)
    for command in (v4_contract, v4_source, v4_materialize, v4_audit, v4_diagnose):
        command.add_argument(
            "--observation-contract",
            type=Path,
            default=Path("game_rules/observation_contract_v2.json"),
        )
        command.add_argument(
            "--candidate-contract",
            type=Path,
            default=Path("game_rules/candidate_action_contract_v1.json"),
        )
        command.add_argument(
            "--weak-supervision-contract",
            type=Path,
            default=Path("configs/t8_v4_weak_supervision_v1.json"),
        )
        command.add_argument(
            "--experiment-contract",
            type=Path,
            default=Path("configs/t8_v4_experiment_plan_v1.json"),
        )
    v5_roi_contract = commands.add_parser(
        "t8-v5-roi-contract-check", help="verify the frozen T8-v5 ROI-isolation contract"
    )
    v5_roi_contract.add_argument(
        "--experiment-contract",
        type=Path,
        default=Path("configs/t8_v5_roi_experiment_v1.json"),
    )
    v5_roi_materialize = commands.add_parser(
        "t8-v5-roi-materialize", help="materialize correct-ROI and wrong-ROI frozen features"
    )
    v5_roi_materialize.add_argument("--pseudolabel-root", type=Path, required=True)
    v5_roi_materialize.add_argument("--target-root", type=Path, required=True)
    v5_roi_materialize.add_argument("--adapter-checkpoint", type=Path, required=True)
    v5_roi_materialize.add_argument("--layout", type=Path, required=True)
    v5_roi_materialize.add_argument("--output-dir", type=Path, required=True)
    v5_roi_materialize.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v5_roi_materialize.add_argument("--batch-size", type=int, default=256)
    v5_roi_materialize.add_argument(
        "--experiment-contract",
        type=Path,
        default=Path("configs/t8_v5_roi_experiment_v1.json"),
    )
    v5_roi_diagnose = commands.add_parser(
        "t8-v5-roi-seed0-diagnose", help="run the single-frame T8-v5 ROI evidence ladder"
    )
    v5_roi_diagnose.add_argument("--dataset-root", type=Path, required=True)
    v5_roi_diagnose.add_argument("--output-dir", type=Path, required=True)
    v5_roi_diagnose.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v5_roi_diagnose.add_argument("--batch-size", type=int, default=256)
    v5_roi_diagnose.add_argument(
        "--experiment-contract",
        type=Path,
        default=Path("configs/t8_v5_roi_experiment_v1.json"),
    )
    basic_mvp_contract = commands.add_parser(
        "t8-basic-mvp-contract-check", help="verify the frozen Basic-only MVP contract"
    )
    basic_mvp_contract.add_argument(
        "--contract", type=Path, default=Path("configs/t8_basic_mvp_v1.json")
    )
    basic_mvp_replay = commands.add_parser(
        "t8-basic-mvp-offline-replay",
        help="run deterministic Basic-only candidates on frozen video-dev",
    )
    basic_mvp_replay.add_argument("--contract", type=Path, required=True)
    basic_mvp_replay.add_argument("--v5-contract", type=Path, required=True)
    basic_mvp_replay.add_argument("--feature-root", type=Path, required=True)
    basic_mvp_replay.add_argument("--target-root", type=Path, required=True)
    basic_mvp_replay.add_argument("--training-report", type=Path, required=True)
    basic_mvp_replay.add_argument("--model", type=Path, required=True)
    basic_mvp_replay.add_argument("--adapter-checkpoint", type=Path, required=True)
    basic_mvp_replay.add_argument("--layout", type=Path, required=True)
    basic_mvp_replay.add_argument("--output-dir", type=Path, required=True)
    basic_mvp_replay.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    basic_mvp_replay.add_argument("--batch-size", type=int, default=256)
    basic_mvp_shadow = commands.add_parser(
        "t8-basic-mvp-shadow", help="run the admitted five-minute zero-control Basic Shadow"
    )
    basic_mvp_shadow.add_argument("--serial", required=True)
    basic_mvp_shadow.add_argument("--video-node", type=Path, required=True)
    basic_mvp_shadow.add_argument("--base-contract", type=Path, required=True)
    basic_mvp_shadow.add_argument("--shadow-contract", type=Path, required=True)
    basic_mvp_shadow.add_argument("--offline-summary", type=Path, required=True)
    basic_mvp_shadow.add_argument("--v5-contract", type=Path, required=True)
    basic_mvp_shadow.add_argument("--feature-root", type=Path, required=True)
    basic_mvp_shadow.add_argument("--training-report", type=Path, required=True)
    basic_mvp_shadow.add_argument("--model", type=Path, required=True)
    basic_mvp_shadow.add_argument("--adapter-checkpoint", type=Path, required=True)
    basic_mvp_shadow.add_argument("--layout", type=Path, required=True)
    basic_mvp_shadow.add_argument("--output-dir", type=Path, required=True)
    basic_mvp_shadow.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    basic_mvp_shadow.add_argument("--batch-size", type=int, default=32)
    basic_rule_smoke = commands.add_parser(
        "basic-rule-smoke", help="run the fixed 20-second zero-input Basic ROI rule smoke"
    )
    basic_rule_smoke.add_argument("--serial", required=True)
    basic_rule_smoke.add_argument("--video-node", type=Path, required=True)
    basic_rule_smoke.add_argument("--contract", type=Path, required=True)
    basic_rule_smoke.add_argument("--teacher-report", type=Path, required=True)
    basic_rule_smoke.add_argument("--layout", type=Path, required=True)
    basic_rule_smoke.add_argument("--output-dir", type=Path, required=True)
    basic_rule_probe = commands.add_parser(
        "basic-rule-probe", help="run the admitted bounded 20-tap Basic engineering probe"
    )
    basic_rule_probe.add_argument("--serial", required=True)
    basic_rule_probe.add_argument("--video-node", type=Path, required=True)
    basic_rule_probe.add_argument("--contract", type=Path, required=True)
    basic_rule_probe.add_argument("--smoke-summary", type=Path, required=True)
    basic_rule_probe.add_argument("--teacher-report", type=Path, required=True)
    basic_rule_probe.add_argument("--layout", type=Path, required=True)
    basic_rule_probe.add_argument("--output-dir", type=Path, required=True)
    synchronous_combat = commands.add_parser(
        "synchronous-combat-probe",
        help="run the bounded acknowledged four-button combat probe",
    )
    synchronous_combat.add_argument("--serial", required=True)
    synchronous_combat.add_argument("--video-node", type=Path, required=True)
    synchronous_combat.add_argument("--contract", type=Path, required=True)
    synchronous_combat.add_argument("--teacher-report", type=Path, required=True)
    synchronous_combat.add_argument("--visual-layout", type=Path, required=True)
    synchronous_combat.add_argument("--execution-layout", type=Path, required=True)
    synchronous_combat.add_argument("--output-dir", type=Path, required=True)
    visual_arbiter = commands.add_parser(
        "visual-combat-arbiter",
        help="run the bounded visual cooldown-aware combat arbiter",
    )
    visual_arbiter.add_argument("--serial", required=True)
    visual_arbiter.add_argument("--video-node", type=Path, required=True)
    visual_arbiter.add_argument("--contract", type=Path, required=True)
    visual_arbiter.add_argument("--teacher-report", type=Path, required=True)
    visual_arbiter.add_argument("--visual-layout", type=Path, required=True)
    visual_arbiter.add_argument("--execution-layout", type=Path, required=True)
    visual_arbiter.add_argument("--output-dir", type=Path, required=True)
    visual_collect = commands.add_parser(
        "visual-combat-collect",
        help="collect timestamped single-frame RGB and synchronous combat labels",
    )
    visual_collect.add_argument("--serial", required=True)
    visual_collect.add_argument("--video-node", type=Path, required=True)
    visual_collect.add_argument("--contract", type=Path, required=True)
    visual_collect.add_argument("--teacher-report", type=Path, required=True)
    visual_collect.add_argument("--visual-layout", type=Path, required=True)
    visual_collect.add_argument("--execution-layout", type=Path, required=True)
    visual_collect.add_argument("--output-dir", type=Path, required=True)
    visual_collect.add_argument("--shard-size", type=int, default=256)
    visual_event_contract = commands.add_parser(
        "visual-combat-dataset-contract-check",
        help="verify the timestamped executed-action dataset contract",
    )
    visual_event_contract.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/visual_combat_event_dataset_v1.json"),
    )
    operation_base = commands.add_parser(
        "mobile-operation-base",
        help="run persistent movement with concurrent combat, minimap, and purchase",
    )
    operation_base.add_argument("--serial", required=True)
    operation_base.add_argument("--contract", type=Path, required=True)
    operation_base.add_argument("--teacher-report", type=Path, required=True)
    operation_base.add_argument("--visual-layout", type=Path, required=True)
    operation_base.add_argument("--execution-layout", type=Path, required=True)
    operation_base.add_argument("--observation-rois", type=Path, required=True)
    operation_base.add_argument("--output-dir", type=Path, required=True)
    operation_teacher = commands.add_parser(
        "mobile-operation-teacher",
        help="run state-conditioned minimap movement through the operation base",
    )
    operation_teacher.add_argument("--serial", required=True)
    operation_teacher.add_argument("--base-contract", type=Path, required=True)
    operation_teacher.add_argument("--movement-contract", type=Path, required=True)
    operation_teacher.add_argument("--teacher-report", type=Path, required=True)
    operation_teacher.add_argument("--visual-layout", type=Path, required=True)
    operation_teacher.add_argument("--execution-layout", type=Path, required=True)
    operation_teacher.add_argument("--observation-rois", type=Path, required=True)
    operation_teacher.add_argument("--output-dir", type=Path, required=True)
    operation_teacher.add_argument("--enable-input", action="store_true")
    movement_teacher_audit = commands.add_parser(
        "operation-minimap-teacher-audit",
        help="audit the state-conditioned minimap movement teacher offline",
    )
    movement_teacher_audit.add_argument("--session-dir", type=Path, required=True)
    movement_teacher_audit.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/operation_movement_teacher_v1.json"),
    )
    movement_teacher_audit.add_argument("--output-dir", type=Path, required=True)
    operation_contract = commands.add_parser(
        "operation-policy-contract-check",
        help="verify the offline Operation Policy v1 contract",
    )
    operation_contract.add_argument(
        "--contract", type=Path, default=Path("configs/operation_policy_v1.json")
    )
    operation_idm = commands.add_parser(
        "operation-idm-pilot",
        help="train the seed-0 movement and combat inverse-dynamics pilot",
    )
    operation_idm.add_argument("--contract", type=Path, required=True)
    operation_idm.add_argument("--adapter-checkpoint", type=Path, required=True)
    operation_idm.add_argument("--observation-rois", type=Path, required=True)
    operation_idm.add_argument("--operation-train", type=Path, required=True)
    operation_idm.add_argument("--operation-dev", type=Path, required=True)
    operation_idm.add_argument("--combat-root", type=Path, required=True)
    operation_idm.add_argument("--output-dir", type=Path, required=True)
    operation_idm.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    operation_idm.add_argument("--batch-size", type=int, default=256)
    operation_pseudo = commands.add_parser(
        "operation-video-pseudolabel",
        help="apply admitted inverse dynamics to video-train and video-dev",
    )
    operation_pseudo.add_argument("--contract", type=Path, required=True)
    operation_pseudo.add_argument("--idm-dir", type=Path, required=True)
    operation_pseudo.add_argument("--target-dir", type=Path, required=True)
    operation_pseudo.add_argument("--adapter-checkpoint", type=Path, required=True)
    operation_pseudo.add_argument("--observation-rois", type=Path, required=True)
    operation_pseudo.add_argument("--output-dir", type=Path, required=True)
    operation_pseudo.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    operation_pseudo.add_argument("--batch-size", type=int, default=512)
    operation_policy = commands.add_parser(
        "operation-policy-pilot",
        help="train the seed-0 causal movement and combat policy pilot",
    )
    operation_policy.add_argument("--contract", type=Path, required=True)
    operation_policy.add_argument("--dataset-root", type=Path, required=True)
    operation_policy.add_argument("--output-dir", type=Path, required=True)
    operation_policy.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    operation_policy.add_argument("--batch-size", type=int, default=128)
    direct_contract = commands.add_parser(
        "operation-direct-policy-contract-check",
        help="verify the executed-action direct-policy contract",
    )
    direct_contract.add_argument(
        "--contract", type=Path, default=Path("configs/operation_direct_policy_v1.json")
    )
    direct_policy = commands.add_parser(
        "operation-direct-policy-pilot",
        help="train one offline causal policy from frozen executed-action sessions",
    )
    direct_policy.add_argument("--contract", type=Path, required=True)
    direct_policy.add_argument("--adapter-checkpoint", type=Path, required=True)
    direct_policy.add_argument("--observation-rois", type=Path, required=True)
    direct_policy.add_argument("--operation-train", type=Path, required=True)
    direct_policy.add_argument("--operation-dev", type=Path, required=True)
    direct_policy.add_argument("--combat-root", type=Path, required=True)
    direct_policy.add_argument("--output-dir", type=Path, required=True)
    direct_policy.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    direct_policy.add_argument("--batch-size", type=int, default=128)
    movement_policy_contract = commands.add_parser(
        "operation-movement-policy-contract-check",
        help="verify the state-conditioned movement policy contract",
    )
    movement_policy_contract.add_argument(
        "--contract", type=Path, default=Path("configs/operation_movement_policy_v1.json")
    )
    movement_split = commands.add_parser(
        "operation-movement-freeze-split",
        help="freeze four-session pilot or twelve-session formal movement split",
    )
    movement_split.add_argument("--dataset-root", type=Path, required=True)
    movement_split.add_argument("--contract", type=Path, required=True)
    movement_split.add_argument("--output", type=Path, required=True)
    movement_split.add_argument("--pilot", action="store_true")
    movement_pilot = commands.add_parser(
        "operation-movement-pilot",
        help="train the seed-0 state-conditioned movement learnability pilot",
    )
    movement_pilot.add_argument("--dataset-root", type=Path, required=True)
    movement_pilot.add_argument("--split", type=Path, required=True)
    movement_pilot.add_argument("--contract", type=Path, required=True)
    movement_pilot.add_argument("--adapter-checkpoint", type=Path, required=True)
    movement_pilot.add_argument("--output-dir", type=Path, required=True)
    movement_pilot.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    movement_pilot.add_argument("--batch-size", type=int, default=128)
    t8_evaluate = commands.add_parser(
        "t8-evaluate-offline", help="run the sealed held-out evaluation for the selected T8 model"
    )
    t8_evaluate.add_argument("--dataset-root", type=Path, required=True)
    t8_evaluate.add_argument("--model", type=Path, required=True)
    t8_evaluate.add_argument("--training-report", type=Path, required=True)
    t8_evaluate.add_argument("--output", type=Path, required=True)
    t8_evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    t8_evaluate.add_argument("--batch-size", type=int, default=32)
    commands.add_parser("t8-smoke", help="run the non-promoting offline T8 model smoke")
    t8_shadow = commands.add_parser(
        "t8-shadow", help="run read-only live T8 diagnostics for the authorized test app"
    )
    t8_shadow.add_argument("--serial", required=True)
    t8_shadow.add_argument("--model", type=Path, required=True)
    t8_shadow.add_argument("--offline-report", type=Path, required=True)
    t8_shadow.add_argument("--layout", type=Path, required=True)
    t8_shadow.add_argument("--video-node", type=Path, required=True)
    t8_shadow.add_argument("--output-dir", type=Path, required=True)
    t8_shadow.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    t8_shadow.add_argument("--stream-fps", type=int, default=30)
    t8_shadow.add_argument("--infer-hz", type=int, default=10)
    t8_shadow.add_argument("--run-seconds", type=float, default=300.0)
    v26_shadow = commands.add_parser(
        "t8-v26-shadow", help="run the frozen T8-v2.6 model in read-only live Shadow"
    )
    v26_shadow.add_argument("--serial", required=True)
    v26_shadow.add_argument("--model", type=Path, required=True)
    v26_shadow.add_argument("--offline-report", type=Path, required=True)
    v26_shadow.add_argument("--split", type=Path, required=True)
    v26_shadow.add_argument("--layout", type=Path, required=True)
    v26_shadow.add_argument("--video-node", type=Path, required=True)
    v26_shadow.add_argument("--output-dir", type=Path, required=True)
    v26_shadow.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v26_shadow.add_argument("--stream-fps", type=int, default=30)
    v26_shadow.add_argument("--infer-hz", type=int, default=10)
    v26_shadow.add_argument("--run-seconds", type=float, default=300.0)
    v26_replay = commands.add_parser(
        "t8-v26-shadow-replay",
        help="run read-only Shadow against one sealed five-minute demonstration session",
    )
    v26_replay.add_argument("--dataset-root", type=Path, required=True)
    v26_replay.add_argument("--split", type=Path, required=True)
    v26_replay.add_argument("--run-root", type=Path, required=True)
    v26_replay.add_argument("--selection", type=Path, required=True)
    v26_replay.add_argument("--offline-report", type=Path, required=True)
    v26_replay.add_argument("--layout", type=Path, required=True)
    v26_replay.add_argument("--output-dir", type=Path, required=True)
    v26_replay.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v26_probe = commands.add_parser(
        "t8-v26-execute-probe",
        help="run the separately admitted T8-v2.6 60-second/20-action probe",
    )
    v26_probe.add_argument("--serial", required=True)
    v26_probe.add_argument("--model", type=Path, required=True)
    v26_probe.add_argument("--selection", type=Path, required=True)
    v26_probe.add_argument("--offline-report", type=Path, required=True)
    v26_probe.add_argument("--shadow-summary", type=Path, required=True)
    v26_probe.add_argument("--split", type=Path, required=True)
    v26_probe.add_argument("--layout", type=Path, required=True)
    v26_probe.add_argument("--video-node", type=Path, required=True)
    v26_probe.add_argument("--output-dir", type=Path, required=True)
    v26_probe.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    v26_probe.add_argument("--stream-fps", type=int, default=30)
    v26_probe.add_argument("--infer-hz", type=int, default=10)
    v26_probe.add_argument("--run-seconds", type=float, default=60.0)
    v26_probe.add_argument("--max-actions", type=int, default=20)
    t8_probe = commands.add_parser(
        "t8-execute-probe", help="run the strictly admitted bounded self-built-app T8 probe"
    )
    t8_probe.add_argument("--serial", required=True)
    t8_probe.add_argument("--model", type=Path, required=True)
    t8_probe.add_argument("--layout", type=Path, required=True)
    t8_probe.add_argument("--video-node", type=Path, required=True)
    t8_probe.add_argument("--training-report", type=Path, required=True)
    t8_probe.add_argument("--offline-report", type=Path, required=True)
    t8_probe.add_argument("--shadow-summary", type=Path, required=True)
    t8_probe.add_argument("--output-dir", type=Path, required=True)
    t8_probe.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    t8_probe.add_argument("--stream-fps", type=int, default=30)
    t8_probe.add_argument("--infer-hz", type=int, default=10)
    t8_probe.add_argument("--run-seconds", type=float, default=60.0)
    t8_probe.add_argument("--max-actions", type=int, default=20)
    alignment_smoke = commands.add_parser(
        "alignment-v5-smoke", help="run non-promoting V5 contract checks"
    )
    alignment_smoke.add_argument("--release", type=Path)
    pre_ingest = commands.add_parser(
        "v5-pre-ingest",
        help="emit file-atomic, hash-identified evidence for local MP4 candidates",
    )
    pre_ingest.add_argument("--input-root", type=Path, required=True)
    pre_ingest.add_argument("--output", type=Path, required=True)
    cohort = commands.add_parser(
        "v5-build-cohort", help="freeze every clean component into an 8/2/2 cohort"
    )
    cohort.add_argument("--pre-ingest", type=Path, required=True)
    cohort.add_argument("--output-dir", type=Path, required=True)
    cohort.add_argument("--recording-owner-confirmed", action="store_true")
    cohort.add_argument("--local-research-confirmed", action="store_true")
    cohort.add_argument("--zero-redaction-confirmed", action="store_true")
    ingest = commands.add_parser(
        "v5-ingest-zero-label", help="derive hash-identified zero-redaction RGB shards"
    )
    ingest.add_argument("--input-root", type=Path, required=True)
    ingest.add_argument("--pre-ingest", type=Path, required=True)
    ingest.add_argument("--cohort-dir", type=Path, required=True)
    ingest.add_argument("--output-dir", type=Path, required=True)
    validate_target = commands.add_parser(
        "v5-validate-zero-target", help="stream-validate a hash-identified V5 target dataset"
    )
    validate_target.add_argument("--target-dir", type=Path, required=True)
    validate_target.add_argument("--cohort-dir", type=Path, required=True)
    validate_target.add_argument("--pre-ingest", type=Path, required=True)
    source = commands.add_parser(
        "v5-source-produce", help="write the non-promoting PixelArena source corpus"
    )
    source.add_argument("--output-dir", type=Path, required=True)
    source.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    predict = commands.add_parser(
        "v5-model-predict", help="write non-promoting V5 model-generated train prediction evidence"
    )
    predict.add_argument("--source-dir", type=Path, required=True)
    predict.add_argument("--target-dir", type=Path, required=True)
    predict.add_argument("--cohort-dir", type=Path, required=True)
    predict.add_argument("--pre-ingest", type=Path, required=True)
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--adapted-model", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    predict.add_argument("--batch-size", type=int, default=256)
    freeze = commands.add_parser(
        "v5-freeze-training-config",
        help="write a frozen non-promoting V5 training config",
    )
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--batch-size", type=int, default=256)
    freeze.add_argument("--learning-rate", type=float, default=3e-4)
    freeze.add_argument("--weight-decay", type=float, default=1e-4)
    freeze.add_argument("--epochs", type=int, default=50)
    freeze.add_argument("--mean-teacher-epochs", type=int, default=20)
    adapt = commands.add_parser(
        "v5-train-simsiam-adapted",
        help="write one non-promoting SimSiam adapted checkpoint",
    )
    adapt.add_argument("--source-dir", type=Path, required=True)
    adapt.add_argument("--target-dir", type=Path, required=True)
    adapt.add_argument("--cohort-dir", type=Path, required=True)
    adapt.add_argument("--pre-ingest", type=Path, required=True)
    adapt.add_argument("--config", type=Path, required=True)
    adapt.add_argument("--output-checkpoint", type=Path, required=True)
    adapt.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    adapt.add_argument("--seed", type=int, default=0)
    adapt.add_argument(
        "--resume", action="store_true", help="resume from the atomic epoch checkpoint"
    )
    pseudo = commands.add_parser(
        "v5-materialize-pseudo", help="recompute and filter V5 model prediction evidence"
    )
    pseudo.add_argument("--source-dir", type=Path, required=True)
    pseudo.add_argument("--target-dir", type=Path, required=True)
    pseudo.add_argument("--cohort-dir", type=Path, required=True)
    pseudo.add_argument("--pre-ingest", type=Path, required=True)
    pseudo.add_argument("--config", type=Path, required=True)
    pseudo.add_argument("--adapted-model", type=Path, required=True)
    pseudo.add_argument("--predictions-dir", type=Path, required=True)
    pseudo.add_argument("--output", type=Path, required=True)
    pseudo.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    pseudo.add_argument("--batch-size", type=int, default=256)
    mean_teacher = commands.add_parser(
        "v5-run-mean-teacher-round",
        help="run one non-promoting Mean Teacher round",
    )
    mean_teacher.add_argument("--source-dir", type=Path, required=True)
    mean_teacher.add_argument("--target-dir", type=Path, required=True)
    mean_teacher.add_argument("--cohort-dir", type=Path, required=True)
    mean_teacher.add_argument("--pre-ingest", type=Path, required=True)
    mean_teacher.add_argument("--config", type=Path, required=True)
    mean_teacher.add_argument("--predictions", type=Path, required=True)
    mean_teacher.add_argument("--pseudo", type=Path, required=True)
    mean_teacher.add_argument("--adapted-checkpoint", type=Path, required=True)
    mean_teacher.add_argument("--ema-checkpoint", type=Path, required=True)
    mean_teacher.add_argument("--round-ledger", type=Path, required=True)
    mean_teacher.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    mean_teacher.add_argument("--seed", type=int, default=0)
    commands.add_parser("temporal-v6-smoke", help="run fail-closed RGB temporal smoke")
    commands.add_parser("v6-zero-smoke", help="run a non-promoting RGB-only V6 zero-label smoke")
    rich = commands.add_parser("accept-rich-v7", help="run Rich PixelArena RGB gate")
    rich.add_argument("--device", choices=("cpu", "cuda"), required=True)
    rich.add_argument("--output-dir", type=Path)
    rich.add_argument("--smoke", action="store_true")
    commands.add_parser("check", help="run size and static safety gates")
    adaptive_check = commands.add_parser(
        "adaptive-layout-check",
        help="verify one local adaptive device layout without opening a device",
    )
    adaptive_check.add_argument("--layout", type=Path, required=True)
    hero_check = commands.add_parser(
        "hero-profile-check",
        help="verify one explicit local hero ability behavior profile",
    )
    hero_check.add_argument("--profile", type=Path, required=True)
    combat_cache = commands.add_parser(
        "global-combat-feature-cache",
        help="materialize frozen 32-frame combat features once for fast offline training",
    )
    combat_cache.add_argument("--dataset-root", type=Path, required=True)
    combat_cache.add_argument("--split", type=Path, required=True)
    combat_cache.add_argument("--adapter-checkpoint", type=Path, required=True)
    combat_cache.add_argument("--output-dir", type=Path, required=True)
    combat_cache.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    combat_cache.add_argument("--batch-size", type=int, default=128)
    combat_cached_train = commands.add_parser(
        "global-combat-feature-train", help="train a causal combat head from cached features"
    )
    combat_cached_train.add_argument("--feature-root", type=Path, required=True)
    combat_cached_train.add_argument("--output-dir", type=Path, required=True)
    combat_cached_train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    combat_cached_train.add_argument("--batch-size", type=int, default=128)
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
                    f"\r{sequence:>6}  {hypothesis:<16} {confidence:.3f}  advisory=ABSTAIN",
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
        elif args.command == "mobile-testbed":
            from hok_agent.mobile_testbed import run_mobile_testbed

            result = run_mobile_testbed(
                serial=args.serial,
                model_path=args.model,
                output_dir=args.output_dir,
                device=args.device,
                run_seconds=args.run_seconds,
                infer_hz=args.infer_hz,
                min_confidence=args.min_confidence,
                enable_input=args.enable_input,
                max_actions=args.max_actions,
                layout_path=args.layout,
                capture_mode=args.capture_mode,
                video_node=args.video_node,
                stream_fps=args.stream_fps,
            )
        elif args.command == "mobile-demonstrate":
            from hok_agent.mobile_testbed import (
                SCRIPTED_DEMONSTRATION_SOURCE,
                TERMINAL_DEMONSTRATION_SOURCE,
                run_mobile_demonstrate,
                scripted_key_reader,
            )

            key_reader = (
                scripted_key_reader(
                    seed=args.scripted_seed,
                    interval_seconds=args.scripted_interval_seconds,
                )
                if args.scripted_seed is not None
                else None
            )

            result = run_mobile_demonstrate(
                serial=args.serial,
                output_dir=args.output_dir,
                run_seconds=args.run_seconds,
                max_actions=args.max_actions,
                shard_size=args.shard_size,
                layout_path=args.layout,
                capture_mode=args.capture_mode,
                video_node=args.video_node,
                stream_fps=args.stream_fps,
                key_reader=key_reader,
                event_source=(
                    SCRIPTED_DEMONSTRATION_SOURCE
                    if key_reader is not None
                    else TERMINAL_DEMONSTRATION_SOURCE
                ),
            )
        elif args.command == "mobile-demonstrate-keyboard-v2":
            from hok_agent.mobile_testbed import (
                run_mobile_demonstrate_keyboard_v2,
                scripted_key_reader,
            )

            result = run_mobile_demonstrate_keyboard_v2(
                serial=args.serial,
                output_dir=args.output_dir,
                layout_path=args.layout,
                video_node=args.video_node,
                run_seconds=args.run_seconds,
                max_actions=args.max_actions,
                shard_size=args.shard_size,
                stream_fps=args.stream_fps,
                key_reader=(
                    scripted_key_reader(
                        seed=args.scripted_seed,
                        interval_seconds=args.scripted_interval_seconds,
                    )
                    if args.scripted_seed is not None
                    else None
                ),
                formal_session=args.formal_session,
            )
        elif args.command == "mobile-demonstrate-rgb-teacher-v25":
            from hok_agent.mobile_testbed import run_mobile_demonstrate_rgb_teacher_v25

            result = run_mobile_demonstrate_rgb_teacher_v25(
                serial=args.serial,
                output_dir=args.output_dir,
                layout_path=args.layout,
                video_node=args.video_node,
                teacher_report_path=args.teacher_report,
                run_seconds=args.run_seconds,
                enable_input=args.enable_input,
                max_actions=args.max_actions,
                shard_size=args.shard_size,
                stream_fps=args.stream_fps,
                formal_session=args.formal_session,
                warmup_basic_attack=args.warmup_basic_attack,
                patrol=args.patrol,
                balanced_actions=args.balanced_actions,
            )
        elif args.command == "mobile-demonstrate-keyboard-v2-live":
            from hok_agent.mobile_testbed import run_mobile_demonstrate_keyboard_v21

            result = run_mobile_demonstrate_keyboard_v21(
                serial=args.serial,
                output_dir=args.output_dir,
                layout_path=args.layout,
                run_seconds=args.run_seconds,
                max_actions=args.max_actions,
                shard_size=args.shard_size,
                stream_fps=args.stream_fps,
                countdown_seconds=args.countdown_seconds,
                formal_session=args.formal_session,
                diagnostic_control_smoke=args.diagnostic_control_smoke,
                diagnostic_inverse_probe=args.diagnostic_inverse_probe,
            )
        elif args.command == "mobile-touch-discover":
            from hok_agent.mobile_testbed import discover_touch_devices

            result = {
                "status": "OWNER_CONFIRMATION_REQUIRED",
                "candidates": [
                    {
                        "path": item.path,
                        "name": item.name,
                        "protocol": item.protocol,
                        "max_slots": item.max_slots,
                        "max_x": item.max_x,
                        "max_y": item.max_y,
                        "descriptor_sha256": item.sha256,
                    }
                    for item in discover_touch_devices(args.serial)
                ],
                "raw_touch_events_persisted": False,
            }
        elif args.command == "mobile-touch-probe":
            from hok_agent.mobile_testbed import TouchDescriptor, touch_probe_report

            descriptor = TouchDescriptor(
                args.touch_device,
                "owner_confirmed",
                args.touch_max_slots,
                args.touch_max_x,
                args.touch_max_y,
                args.touch_protocol,
            )
            result = touch_probe_report(
                serial=args.serial, descriptor=descriptor, run_seconds=args.run_seconds
            )
        elif args.command == "mobile-touch-calibrate":
            from hok_agent.mobile_testbed import (
                TouchDescriptor,
                calibrate_touch_transform,
                collect_touch_calibration_points,
                load_layout,
                write_touch_calibration,
            )

            layout, layout_sha256 = load_layout(args.layout)
            descriptor = TouchDescriptor(
                args.touch_device,
                "owner_confirmed",
                args.touch_max_slots,
                args.touch_max_x,
                args.touch_max_y,
                args.touch_protocol,
            )

            def touch_prompt(name: str) -> None:
                input(
                    f"Release all phone touches. Press Enter, then touch and hold {name} "
                    "for two seconds: "
                )
                print(f"CAPTURING {name}: touch now.", flush=True)

            touch_points = collect_touch_calibration_points(
                serial=args.serial,
                descriptor=descriptor,
                prompt=touch_prompt,
            )
            result = write_touch_calibration(
                args.output,
                calibrate_touch_transform(
                    descriptor=descriptor,
                    layout=layout,
                    layout_sha256=layout_sha256,
                    raw_points=touch_points,
                ),
            )
        elif args.command == "mobile-demonstrate-touch":
            from hok_agent.mobile_testbed import TouchDescriptor, run_mobile_touch_demonstrate

            descriptor = TouchDescriptor(
                args.touch_device,
                "owner_confirmed",
                args.touch_max_slots,
                args.touch_max_x,
                args.touch_max_y,
                args.touch_protocol,
            )
            result = run_mobile_touch_demonstrate(
                serial=args.serial,
                touch_descriptor=descriptor,
                touch_calibration_path=args.touch_calibration,
                output_dir=args.output_dir,
                layout_path=args.layout,
                video_node=args.video_node,
                run_seconds=args.run_seconds,
                max_samples=args.max_samples,
                shard_size=args.shard_size,
                stream_fps=args.stream_fps,
                formal_session=args.formal_session,
                semantic_smoke=args.semantic_smoke,
            )
        elif args.command == "mobile-layout-calibrate":
            from hok_agent.mobile_testbed import pick_layout_points, run_layout_calibration

            point_provider: Callable[[str], tuple[float, float]]
            if args.manual_points:

                def manual_point_provider(name: str) -> tuple[float, float]:
                    raw = input(f"{name} normalized x,y: ").strip().split(",")
                    if len(raw) != 2:
                        raise ValueError("calibration point must be x,y")
                    return (float(raw[0]), float(raw[1]))

                point_provider = manual_point_provider
            else:
                points = pick_layout_points(
                    args.serial,
                    args.video_node,
                    args.stream_fps,
                    (
                        "joystick_center",
                        "joystick_north_endpoint",
                        "skill1",
                        "skill2",
                        "skill3",
                    ),
                )
                point_provider = points.__getitem__

            def confirmer(name: str) -> bool | None:
                answer = input(f"Did {name} act correctly? [y/N/r=retry]: ").strip().lower()
                return None if answer == "r" else answer == "y"

            result = run_layout_calibration(
                serial=args.serial,
                layout_path=args.layout,
                output_path=args.output,
                video_node=args.video_node,
                stream_fps=args.stream_fps,
                point_provider=point_provider,
                confirmer=confirmer,
            )
        elif args.command == "t8-train-bc":
            if args.device == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from hok_agent.t8 import train_t8_bc

            result = train_t8_bc(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                v5_source_dir=args.v5_source_dir,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-freeze-split":
            from hok_agent.t8 import freeze_t8_split

            result = freeze_t8_split(dataset_root=args.dataset_root, output_path=args.output)
        elif args.command == "t8-v2-video-adapt":
            if args.device == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from hok_agent.t8 import train_t8_v2_video_adapter

            result = train_t8_v2_video_adapter(
                v5_source_dir=args.v5_source_dir,
                target_dir=args.target_dir,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-v2-freeze-split":
            from hok_agent.t8 import freeze_t8_v2_split

            result = freeze_t8_v2_split(dataset_root=args.dataset_root, output_path=args.output)
        elif args.command == "t8-v2-live-freeze-split":
            from hok_agent.t8 import freeze_t8_v21_split

            result = freeze_t8_v21_split(dataset_root=args.dataset_root, output_path=args.output)
        elif args.command == "t8-v2-live-pilot-freeze":
            from hok_agent.t8 import freeze_t8_v21_pilot_split

            result = freeze_t8_v21_pilot_split(
                dataset_root=args.dataset_root, output_path=args.output
            )
        elif args.command == "t8-v2-pilot":
            if args.device == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from hok_agent.t8 import run_t8_v2_pilot_pair

            result = run_t8_v2_pilot_pair(
                dataset_root=args.dataset_root,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-v2-live-pilot":
            if args.device == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from hok_agent.t8 import run_t8_v2_pilot_pair

            result = run_t8_v2_pilot_pair(
                dataset_root=args.dataset_root,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                lineage="v2.1",
                split_path=args.split,
            )
        elif args.command == "t8-v2-live-inverse-materialize":
            from hok_agent.t8 import materialize_t8_v21_inverse_probe

            result = materialize_t8_v21_inverse_probe(
                session_dir=args.session_dir,
                layout_path=args.layout,
                output_dir=args.output_dir,
            )
        elif args.command == "t8-video-three-class-materialize":
            from hok_agent.t8 import materialize_t8_video_three_class

            result = materialize_t8_video_three_class(
                source_dir=args.source_dir,
                inverse_report_path=args.inverse_report,
                output_dir=args.output_dir,
                retrospective=args.retrospective,
            )
        elif args.command == "t8-video-three-class-pilot":
            if args.device == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from hok_agent.t8 import run_t8_video_three_class_pilot

            result = run_t8_video_three_class_pilot(
                dataset_root=args.dataset_root,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                retrospective=args.retrospective,
            )
        elif args.command == "t8-video-retrospective-roi-evaluate":
            from hok_agent.t8 import evaluate_t8_video_retrospective_roi

            result = evaluate_t8_video_retrospective_roi(
                dataset_root=args.dataset_root,
                probe_report_path=args.probe_report,
                inverse_report_path=args.inverse_report,
                output_dir=args.output_dir,
            )
        elif args.command == "t8-retrospective-baseline-verify":
            from hok_agent.t8 import verify_t8_retrospective_baseline

            result = verify_t8_retrospective_baseline(baseline_dir=args.baseline_dir)
        elif args.command == "t8-retrospective-batch":
            from hok_agent.t8 import run_t8_retrospective_batch

            result = run_t8_retrospective_batch(
                target_dir=args.target_dir,
                baseline_dir=args.baseline_dir,
                layout_path=args.layout,
                split=args.split,
                output_dir=args.output_dir,
                session_hashes=tuple(args.session_hash),
            )
        elif args.command == "t8-retrospective-batch-verify":
            from hok_agent.t8 import verify_t8_retrospective_batch

            result = verify_t8_retrospective_batch(batch_dir=args.batch_dir)
        elif args.command == "t8-retrospective-calibrate-v2":
            from hok_agent.t8 import run_t8_retrospective_calibration_v2

            result = run_t8_retrospective_calibration_v2(
                dataset_root=args.dataset_root,
                probe_report_path=args.probe_report,
                layout_path=args.layout,
                baseline_dir=args.baseline_dir,
                inverse_calibration_paths=tuple(args.inverse_calibration),
                inverse_holdout_path=args.inverse_holdout,
                output_dir=args.output_dir,
            )
        elif args.command == "t8-causal-video-materialize":
            from hok_agent.t8 import materialize_t8_causal_video_dataset

            result = materialize_t8_causal_video_dataset(
                target_dir=args.target_dir,
                train_events_dir=args.train_events_dir,
                dev_events_dir=args.dev_events_dir,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-causal-video-pilot":
            from hok_agent.t8 import run_t8_causal_video_pilot

            result = run_t8_causal_video_pilot(
                dataset_root=args.dataset_root,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-causal-video-diagnose":
            from hok_agent.t8 import run_t8_causal_video_diagnostic

            result = run_t8_causal_video_diagnostic(
                dataset_root=args.dataset_root,
                pilot_dir=args.pilot_dir,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-causal-pixel-materialize":
            from hok_agent.t8 import materialize_t8_causal_pixel_dataset

            result = materialize_t8_causal_pixel_dataset(
                target_dir=args.target_dir,
                train_events_dir=args.train_events_dir,
                dev_events_dir=args.dev_events_dir,
                output_dir=args.output_dir,
            )
        elif args.command == "t8-causal-pixel-probe":
            from hok_agent.t8 import run_t8_causal_pixel_probe

            result = run_t8_causal_pixel_probe(
                dataset_root=args.dataset_root,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-visual-teacher-replay":
            from hok_agent.t8 import run_t8_visual_teacher_replay

            result = run_t8_visual_teacher_replay(
                dataset_root=args.dataset_root,
                pixel_probe_dir=args.pixel_probe_dir,
                layout_path=args.layout,
                output_dir=args.output_dir,
            )
        elif args.command == "t8-visible-onset-audit":
            from hok_agent.t8 import run_t8_visible_onset_audit

            result = run_t8_visible_onset_audit(
                target_dir=args.target_dir,
                train_events_dir=args.train_events_dir,
                dev_events_dir=args.dev_events_dir,
                layout_path=args.layout,
                calibration_report_path=args.calibration_report,
                output_dir=args.output_dir,
            )
        elif args.command == "t8-combat-causal-materialize":
            from hok_agent.t8 import materialize_t8_combat_causal_dataset

            result = materialize_t8_combat_causal_dataset(
                target_dir=args.target_dir,
                onset_audit_dir=args.onset_audit_dir,
                output_dir=args.output_dir,
                diagnostic_only=args.diagnostic_only,
            )
        elif args.command == "t8-combat-causal-pilot":
            from hok_agent.t8 import run_t8_combat_causal_pilot

            result = run_t8_combat_causal_pilot(
                dataset_root=args.dataset_root,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-v25-freeze-split":
            from hok_agent.t8 import freeze_t8_v25_split

            result = freeze_t8_v25_split(
                dataset_root=args.dataset_root,
                output_path=args.output,
                pilot=args.pilot,
            )
        elif args.command == "t8-v25-pilot":
            from hok_agent.t8 import run_t8_v25_pilot

            result = run_t8_v25_pilot(
                dataset_root=args.dataset_root,
                split_path=args.split,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        elif args.command == "t8-v26-select":
            from hok_agent.t8 import select_t8_v26_model

            result = select_t8_v26_model(
                run_root=args.run_root,
                output_path=args.output,
            )
        elif args.command == "t8-v26-evaluate-offline":
            from hok_agent.t8 import evaluate_t8_v26_offline

            result = evaluate_t8_v26_offline(
                dataset_root=args.dataset_root,
                split_path=args.split,
                run_root=args.run_root,
                selection_path=args.selection,
                output_path=args.output,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-v27-calibration-pilot":
            from hok_agent.t8 import run_t8_v27_calibration_pilot

            result = run_t8_v27_calibration_pilot(
                dataset_root=args.dataset_root,
                train_session=args.train_session,
                dev_session=args.dev_session,
                source_model=args.source_model,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-v27-freeze":
            from hok_agent.t8_v3 import freeze_t8_v27_failures

            result = freeze_t8_v27_failures(
                report_paths=args.report,
                output_dir=args.output_dir,
            )
        elif args.command == "t8-v3-state-materialize":
            from hok_agent.t8_v3 import materialize_t8_v3_state_dataset

            result = materialize_t8_v3_state_dataset(
                feature_root=args.feature_root,
                target_root=args.target_root,
                teacher_report=args.teacher_report,
                layout_path=args.layout,
                output_dir=args.output_dir,
            )
        elif args.command == "t8-v3-state-train":
            from hok_agent.t8_v3 import train_t8_v3_state_single_seed

            result = train_t8_v3_state_single_seed(
                dataset_root=args.dataset_root,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed,
                epochs=args.epochs,
            )
        elif args.command == "t8-v3-hybrid-replay":
            from hok_agent.t8_v3 import run_t8_v3_hybrid_replay

            result = run_t8_v3_hybrid_replay(
                dataset_root=args.dataset_root,
                model_path=args.model,
                training_report=args.training_report,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-v4-contract-check":
            from hok_agent.t8_v4 import verify_t8_v4_contracts

            result = verify_t8_v4_contracts(
                observation_contract=args.observation_contract,
                candidate_contract=args.candidate_contract,
                weak_supervision_contract=args.weak_supervision_contract,
                experiment_contract=args.experiment_contract,
            )
        elif args.command == "t8-v4-source-teacher-train":
            from hok_agent.t8_v4 import train_t8_v4_source_teacher

            result = train_t8_v4_source_teacher(
                adapter_checkpoint=args.adapter_checkpoint,
                layout_path=args.layout,
                observation_contract=args.observation_contract,
                candidate_contract=args.candidate_contract,
                weak_supervision_contract=args.weak_supervision_contract,
                experiment_contract=args.experiment_contract,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
        elif args.command == "t8-v4-pseudolabel-materialize":
            from hok_agent.t8_v4 import materialize_t8_v4_pseudolabels

            result = materialize_t8_v4_pseudolabels(
                feature_root=args.feature_root,
                target_root=args.target_root,
                rule_teacher_report=args.rule_teacher_report,
                source_teacher_model=args.source_teacher_model,
                source_teacher_report=args.source_teacher_report,
                layout_path=args.layout,
                observation_contract=args.observation_contract,
                candidate_contract=args.candidate_contract,
                weak_supervision_contract=args.weak_supervision_contract,
                experiment_contract=args.experiment_contract,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-v4-weak-audit":
            from hok_agent.t8_v4 import audit_t8_v4_weak_supervision

            result = audit_t8_v4_weak_supervision(
                dataset_root=args.dataset_root,
                observation_contract=args.observation_contract,
                candidate_contract=args.candidate_contract,
                weak_supervision_contract=args.weak_supervision_contract,
                experiment_contract=args.experiment_contract,
                output_path=args.output,
            )
        elif args.command == "t8-v4-seed0-diagnose":
            from hok_agent.t8_v4 import diagnose_t8_v4_seed0

            result = diagnose_t8_v4_seed0(
                dataset_root=args.dataset_root,
                target_root=args.target_root,
                adapter_checkpoint=args.adapter_checkpoint,
                weak_audit_report=args.weak_audit_report,
                observation_contract=args.observation_contract,
                candidate_contract=args.candidate_contract,
                weak_supervision_contract=args.weak_supervision_contract,
                experiment_contract=args.experiment_contract,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-v5-roi-contract-check":
            from hok_agent.t8_v5 import verify_t8_v5_contract

            result = verify_t8_v5_contract(args.experiment_contract)
        elif args.command == "t8-v5-roi-materialize":
            from hok_agent.t8_v5 import materialize_t8_v5_roi_features

            result = materialize_t8_v5_roi_features(
                pseudolabel_root=args.pseudolabel_root,
                target_root=args.target_root,
                adapter_checkpoint=args.adapter_checkpoint,
                layout_path=args.layout,
                experiment_contract=args.experiment_contract,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-v5-roi-seed0-diagnose":
            from hok_agent.t8_v5 import diagnose_t8_v5_roi_seed0

            result = diagnose_t8_v5_roi_seed0(
                dataset_root=args.dataset_root,
                experiment_contract=args.experiment_contract,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-basic-mvp-contract-check":
            from hok_agent.t8_basic_mvp import verify_t8_basic_mvp_contract

            result = verify_t8_basic_mvp_contract(args.contract)
        elif args.command == "t8-basic-mvp-offline-replay":
            from hok_agent.t8_basic_mvp import run_t8_basic_mvp_offline_replay

            result = run_t8_basic_mvp_offline_replay(
                contract_path=args.contract,
                v5_contract=args.v5_contract,
                feature_root=args.feature_root,
                target_root=args.target_root,
                training_report=args.training_report,
                model_path=args.model,
                adapter_checkpoint=args.adapter_checkpoint,
                layout_path=args.layout,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-basic-mvp-shadow":
            from hok_agent.t8_basic_mvp import run_t8_basic_mvp_shadow

            result = run_t8_basic_mvp_shadow(
                serial=args.serial,
                video_node=args.video_node,
                base_contract_path=args.base_contract,
                shadow_contract_path=args.shadow_contract,
                offline_summary=args.offline_summary,
                v5_contract=args.v5_contract,
                feature_root=args.feature_root,
                training_report=args.training_report,
                model_path=args.model,
                adapter_checkpoint=args.adapter_checkpoint,
                layout_path=args.layout,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "basic-rule-smoke":
            from hok_agent.mobile_testbed import run_basic_rule_smoke

            result = run_basic_rule_smoke(
                serial=args.serial,
                video_node=args.video_node,
                contract_path=args.contract,
                teacher_report=args.teacher_report,
                layout_path=args.layout,
                output_dir=args.output_dir,
            )
        elif args.command == "basic-rule-probe":
            from hok_agent.mobile_testbed import run_basic_rule_probe

            result = run_basic_rule_probe(
                serial=args.serial,
                video_node=args.video_node,
                contract_path=args.contract,
                smoke_summary=args.smoke_summary,
                teacher_report=args.teacher_report,
                layout_path=args.layout,
                output_dir=args.output_dir,
            )
        elif args.command == "synchronous-combat-probe":
            from hok_agent.mobile_testbed import run_synchronous_combat_probe

            result = run_synchronous_combat_probe(
                serial=args.serial,
                video_node=args.video_node,
                contract_path=args.contract,
                teacher_report=args.teacher_report,
                visual_layout_path=args.visual_layout,
                execution_layout_path=args.execution_layout,
                output_dir=args.output_dir,
            )
        elif args.command == "visual-combat-arbiter":
            from hok_agent.mobile_testbed import run_visual_combat_arbiter

            result = run_visual_combat_arbiter(
                serial=args.serial,
                video_node=args.video_node,
                contract_path=args.contract,
                teacher_report=args.teacher_report,
                visual_layout_path=args.visual_layout,
                execution_layout_path=args.execution_layout,
                output_dir=args.output_dir,
            )
        elif args.command == "visual-combat-collect":
            from hok_agent.mobile_testbed import run_visual_combat_arbiter

            result = run_visual_combat_arbiter(
                serial=args.serial,
                video_node=args.video_node,
                contract_path=args.contract,
                teacher_report=args.teacher_report,
                visual_layout_path=args.visual_layout,
                execution_layout_path=args.execution_layout,
                output_dir=args.output_dir,
                persist_derived_rgb=True,
                shard_size=args.shard_size,
            )
        elif args.command == "visual-combat-dataset-contract-check":
            from hok_agent.mobile_testbed import verify_visual_combat_event_dataset_contract

            result = verify_visual_combat_event_dataset_contract(args.contract)
        elif args.command == "mobile-operation-base":
            from hok_agent.mobile_testbed import run_mobile_operation_base

            result = run_mobile_operation_base(
                serial=args.serial,
                contract_path=args.contract,
                teacher_report=args.teacher_report,
                visual_layout_path=args.visual_layout,
                execution_layout_path=args.execution_layout,
                observation_rois_path=args.observation_rois,
                output_dir=args.output_dir,
            )
        elif args.command == "mobile-operation-teacher":
            from hok_agent.mobile_testbed import run_mobile_operation_base

            result = run_mobile_operation_base(
                serial=args.serial,
                contract_path=args.base_contract,
                teacher_report=args.teacher_report,
                visual_layout_path=args.visual_layout,
                execution_layout_path=args.execution_layout,
                observation_rois_path=args.observation_rois,
                output_dir=args.output_dir,
                movement_teacher_contract_path=args.movement_contract,
                enable_input=args.enable_input,
            )
        elif args.command == "operation-minimap-teacher-audit":
            from hok_agent.mobile_testbed import audit_operation_movement_teacher

            result = audit_operation_movement_teacher(
                session_dir=args.session_dir,
                contract_path=args.contract,
                output_dir=args.output_dir,
            )
        elif args.command == "operation-policy-contract-check":
            from hok_agent.operation_policy import verify_operation_policy_contract

            result = verify_operation_policy_contract(args.contract)
        elif args.command == "operation-idm-pilot":
            from hok_agent.operation_policy import run_operation_idm_pilot

            result = run_operation_idm_pilot(
                contract_path=args.contract,
                adapter_checkpoint=args.adapter_checkpoint,
                observation_rois_path=args.observation_rois,
                operation_train=args.operation_train,
                operation_dev=args.operation_dev,
                combat_root=args.combat_root,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "operation-video-pseudolabel":
            from hok_agent.operation_policy import materialize_operation_video_pseudolabels

            result = materialize_operation_video_pseudolabels(
                contract_path=args.contract,
                idm_dir=args.idm_dir,
                target_dir=args.target_dir,
                adapter_checkpoint=args.adapter_checkpoint,
                observation_rois_path=args.observation_rois,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "operation-policy-pilot":
            from hok_agent.operation_policy import train_operation_policy_pilot

            result = train_operation_policy_pilot(
                contract_path=args.contract,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "operation-direct-policy-contract-check":
            from hok_agent.operation_policy import verify_operation_direct_policy_contract

            result = verify_operation_direct_policy_contract(args.contract)
        elif args.command == "operation-direct-policy-pilot":
            from hok_agent.operation_policy import run_operation_direct_policy_pilot

            result = run_operation_direct_policy_pilot(
                contract_path=args.contract,
                adapter_checkpoint=args.adapter_checkpoint,
                observation_rois_path=args.observation_rois,
                operation_train=args.operation_train,
                operation_dev=args.operation_dev,
                combat_root=args.combat_root,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "operation-movement-policy-contract-check":
            from hok_agent.operation_policy import verify_operation_movement_policy_contract

            result = verify_operation_movement_policy_contract(args.contract)
        elif args.command == "operation-movement-freeze-split":
            from hok_agent.operation_policy import freeze_operation_movement_split

            result = freeze_operation_movement_split(
                dataset_root=args.dataset_root,
                contract_path=args.contract,
                output_path=args.output,
                pilot=args.pilot,
            )
        elif args.command == "operation-movement-pilot":
            from hok_agent.operation_policy import run_operation_movement_pilot

            result = run_operation_movement_pilot(
                dataset_root=args.dataset_root,
                split_path=args.split,
                contract_path=args.contract,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-evaluate-offline":
            from hok_agent.t8 import evaluate_t8_offline

            result = evaluate_t8_offline(
                dataset_root=args.dataset_root,
                model_path=args.model,
                training_report=args.training_report,
                output_path=args.output,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "t8-smoke":
            from hok_agent.t8 import t8_smoke

            result = t8_smoke()
        elif args.command == "t8-shadow":
            from hok_agent.t8_shadow import run_t8_shadow

            result = run_t8_shadow(
                serial=args.serial,
                model_path=args.model,
                offline_report=args.offline_report,
                layout_path=args.layout,
                video_node=args.video_node,
                output_dir=args.output_dir,
                device=args.device,
                stream_fps=args.stream_fps,
                infer_hz=args.infer_hz,
                run_seconds=args.run_seconds,
            )
        elif args.command == "t8-v26-shadow":
            from hok_agent.t8_shadow import run_t8_v26_shadow

            result = run_t8_v26_shadow(
                serial=args.serial,
                model_path=args.model,
                offline_report=args.offline_report,
                split_path=args.split,
                layout_path=args.layout,
                video_node=args.video_node,
                output_dir=args.output_dir,
                device=args.device,
                stream_fps=args.stream_fps,
                infer_hz=args.infer_hz,
                run_seconds=args.run_seconds,
            )
        elif args.command == "t8-v26-shadow-replay":
            from hok_agent.t8_shadow import run_t8_v26_replay_shadow

            result = run_t8_v26_replay_shadow(
                dataset_root=args.dataset_root,
                split_path=args.split,
                run_root=args.run_root,
                selection_path=args.selection,
                offline_report=args.offline_report,
                layout_path=args.layout,
                output_dir=args.output_dir,
                device=args.device,
            )
        elif args.command == "t8-v26-execute-probe":
            from hok_agent.t8_shadow import run_t8_v26_execute_probe

            result = run_t8_v26_execute_probe(
                serial=args.serial,
                model_path=args.model,
                selection_path=args.selection,
                offline_report=args.offline_report,
                shadow_summary=args.shadow_summary,
                split_path=args.split,
                layout_path=args.layout,
                video_node=args.video_node,
                output_dir=args.output_dir,
                device=args.device,
                stream_fps=args.stream_fps,
                infer_hz=args.infer_hz,
                run_seconds=args.run_seconds,
                max_actions=args.max_actions,
            )
        elif args.command == "t8-execute-probe":
            from hok_agent.t8_shadow import run_t8_execute_probe

            result = run_t8_execute_probe(
                serial=args.serial,
                model_path=args.model,
                layout_path=args.layout,
                video_node=args.video_node,
                training_report=args.training_report,
                offline_report=args.offline_report,
                shadow_summary=args.shadow_summary,
                output_dir=args.output_dir,
                device=args.device,
                stream_fps=args.stream_fps,
                infer_hz=args.infer_hz,
                run_seconds=args.run_seconds,
                max_actions=args.max_actions,
            )
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
        elif args.command == "v5-pre-ingest":
            from hok_agent.pre_ingest import pre_ingest

            result = pre_ingest(args.input_root, args.output)
        elif args.command == "v5-build-cohort":
            from hok_agent.v5_data import build_automatic_cohort

            result = build_automatic_cohort(
                pre_ingest_path=args.pre_ingest,
                output_dir=args.output_dir,
                recording_owner_confirmed=args.recording_owner_confirmed,
                local_research_confirmed=args.local_research_confirmed,
                zero_redaction_confirmed=args.zero_redaction_confirmed,
            )
        elif args.command == "v5-ingest-zero-label":
            from hok_agent.v5_data import ingest_zero_label_target

            result = ingest_zero_label_target(
                input_root=args.input_root,
                pre_ingest_path=args.pre_ingest,
                cohort_dir=args.cohort_dir,
                output_dir=args.output_dir,
            )
        elif args.command == "v5-validate-zero-target":
            from hok_agent.v5_data import load_zero_label_target

            target = load_zero_label_target(args.target_dir, args.cohort_dir, args.pre_ingest)
            result = {
                "status": "PASSED",
                "disposition": "ZERO_LABEL_TARGET_VALIDATED",
                "manifest_sha256": target.manifest_sha256,
                "session_count": len(target.session_splits),
                "shard_count": len(target.shard_paths),
                "human_labels_consumed": False,
            }
        elif args.command == "v5-source-produce":
            if args.device == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from hok_agent.alignment import produce_v5_source

            result = produce_v5_source(
                output_dir=args.output_dir,
                device=args.device,
            )
        elif args.command == "v5-model-predict":
            if args.device == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from hok_agent.alignment import generate_v5_model_predictions

            source_model = _v5_source_model(args.source_dir)
            result = generate_v5_model_predictions(
                source_metadata_path=args.source_dir / "source.json",
                source_dataset_path=args.source_dir / "source.npz",
                manifest_path=args.target_dir / "manifest.json",
                pre_ingest_path=args.pre_ingest,
                privacy_context_path=args.cohort_dir / "privacy-context.json",
                owner_attestation_path=args.cohort_dir / "owner-attestation.json",
                owner_component_confirmation_path=args.cohort_dir / "component-cohort.json",
                target_shards=_v5_target_shards(args.target_dir),
                config_path=args.config,
                source_model_path=source_model,
                adapted_model_path=args.adapted_model,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "v5-freeze-training-config":
            from hok_agent.alignment import _json, _write_exclusive, build_training_config

            config = build_training_config(
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                mean_teacher_epochs=args.mean_teacher_epochs,
            )
            _write_exclusive(args.output, _json(config).encode())
            result = {
                "status": "PASSED",
                "disposition": "NON_PROMOTING_TRAINING_CONFIG",
                "path": str(args.output),
                "config": config,
            }
        elif args.command == "v5-train-simsiam-adapted":
            if args.device == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from hok_agent.alignment import train_shallow_simsiam

            source_model = _v5_source_model(args.source_dir)
            training = train_shallow_simsiam(
                source_metadata_path=args.source_dir / "source.json",
                source_dataset_path=args.source_dir / "source.npz",
                manifest_path=args.target_dir / "manifest.json",
                pre_ingest_path=args.pre_ingest,
                privacy_context_path=args.cohort_dir / "privacy-context.json",
                owner_attestation_path=args.cohort_dir / "owner-attestation.json",
                owner_component_confirmation_path=args.cohort_dir / "component-cohort.json",
                target_shards=_v5_target_shards(args.target_dir),
                config_path=args.config,
                source_checkpoint=source_model,
                output_checkpoint=args.output_checkpoint,
                device=args.device,
                seed=args.seed,
                resume=args.resume,
            )
            result = asdict(training)
            result["checkpoint"] = str(training.checkpoint)
            result["real_domain_advice_released"] = False
        elif args.command == "v5-run-mean-teacher-round":
            if args.device == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from hok_agent.alignment import run_mean_teacher_round

            source_model = _v5_source_model(args.source_dir)
            result = run_mean_teacher_round(
                source_metadata_path=args.source_dir / "source.json",
                source_dataset_path=args.source_dir / "source.npz",
                manifest_path=args.target_dir / "manifest.json",
                pre_ingest_path=args.pre_ingest,
                privacy_context_path=args.cohort_dir / "privacy-context.json",
                owner_attestation_path=args.cohort_dir / "owner-attestation.json",
                owner_component_confirmation_path=args.cohort_dir / "component-cohort.json",
                target_shards=_v5_target_shards(args.target_dir),
                predictions_path=args.predictions,
                pseudo_path=args.pseudo,
                source_model_path=source_model,
                adapted_checkpoint=args.adapted_checkpoint,
                config_path=args.config,
                ema_checkpoint=args.ema_checkpoint,
                round_ledger=args.round_ledger,
                device=args.device,
                seed=args.seed,
            )
            result["real_domain_advice_released"] = False
        elif args.command == "v5-materialize-pseudo":
            from hok_agent.alignment import materialize_v5_pseudo

            source_model = _v5_source_model(args.source_dir)
            pseudo, report = materialize_v5_pseudo(
                predictions_path=args.predictions_dir,
                source_metadata_path=args.source_dir / "source.json",
                source_dataset_path=args.source_dir / "source.npz",
                manifest_path=args.target_dir / "manifest.json",
                pre_ingest_path=args.pre_ingest,
                privacy_context_path=args.cohort_dir / "privacy-context.json",
                owner_attestation_path=args.cohort_dir / "owner-attestation.json",
                owner_component_confirmation_path=args.cohort_dir / "component-cohort.json",
                target_shards=_v5_target_shards(args.target_dir),
                config_path=args.config,
                source_model_path=source_model,
                adapted_model_path=args.adapted_model,
                output_path=args.output,
                prediction_device=args.device,
                prediction_batch_size=args.batch_size,
            )
            result = {
                "status": "PASSED" if report.filter_floor_met else "FAILED",
                "disposition": "NON_PROMOTING_PSEUDO_FILTER",
                "accepted": report.accepted,
                "groups": report.groups,
                "filter_floor_met": report.filter_floor_met,
                "rejected_by_reason": report.rejected_by_reason,
                "pseudo_path": str(pseudo.artifact_path),
                "pseudo_sha256": pseudo.artifact_sha256,
                "human_labels_consumed": False,
                "real_domain_advice_released": False,
            }
        elif args.command == "temporal-v6-smoke":
            from hok_agent.temporal import cpu_smoke as temporal_cpu_smoke

            result = temporal_cpu_smoke()
        elif args.command == "v6-zero-smoke":
            from hok_agent.v6_zero import cpu_smoke as zero_cpu_smoke

            result = zero_cpu_smoke()
        elif args.command == "accept-rich-v7":
            from hok_agent.rich_pixel import accept_rich_pixel

            result = accept_rich_pixel(args.output_dir, args.device, args.smoke)
        elif args.command == "adaptive-layout-check":
            from hok_agent.adaptive_layout import load_adaptive_layout

            adaptive = load_adaptive_layout(args.layout)
            result = {
                "status": "PASSED",
                "schema_version": "hok-agent-adaptive-layout-check-v1",
                "layout_sha256": adaptive.layout_sha256,
                "reference_layout_sha256": adaptive.reference_layout_sha256,
                "build_identity_sha256": adaptive.build_identity_sha256,
                "content_box_xyxy": [
                    adaptive.content_box.x0,
                    adaptive.content_box.y0,
                    adaptive.content_box.x1,
                    adaptive.content_box.y1,
                ],
                "control_output": False,
                "device_input_allowed": False,
            }
        elif args.command == "hero-profile-check":
            from hok_agent.adaptive_layout import load_hero_profile
            from hok_agent.mobile_testbed import ABILITIES

            profile = load_hero_profile(args.profile)
            result = {
                "status": "PASSED",
                "schema_version": "hok-agent-hero-profile-check-v1",
                "hero_id": profile.hero_id,
                "profile_sha256": profile.profile_sha256,
                "ability_modes": {
                    ability: profile.behavior(ability).mode for ability in ABILITIES[1:]
                },
                "control_output": False,
                "device_input_allowed": False,
            }
        elif args.command == "global-combat-feature-cache":
            from hok_agent.combat_feature_cache import materialize_global_combat_features

            result = materialize_global_combat_features(
                dataset_root=args.dataset_root,
                split_path=args.split,
                adapter_checkpoint=args.adapter_checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif args.command == "global-combat-feature-train":
            from hok_agent.combat_feature_cache import train_global_combat_feature_head

            result = train_global_combat_feature_head(
                feature_root=args.feature_root,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        else:
            result = check_project()
            if not result["passed"]:
                print(json.dumps(result, indent=2, sort_keys=True))
                return 1
        if isinstance(result, Mapping) and result.get("status") == "FAILED":
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, ReplayError, ServiceError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}), file=sys.stderr)
        return 2
