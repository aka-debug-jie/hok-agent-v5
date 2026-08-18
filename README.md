# HoK-Agent V5

[![CI](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml/badge.svg)](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

HoK-Agent V5 is an RGB-only MOBA research testbed built around project-owned PixelArena
environments, causal video models, offline Shadow evaluation, and fail-closed mobile-testbed
interfaces. It does not call a game-internal API and is not affiliated with or endorsed by a game
publisher.

The public repository intentionally contains no authorized mobile build identity, calibrated
coordinates, device serial, recording, dataset, checkpoint, or run artifact. Mobile input is
disabled unless the operator supplies local private evidence for a self-built test app and every
runtime guard passes.

## Architecture

```text
V1–V3  frozen deterministic and behavior-cloning regressions
V4     local video/V4L2 -> read-only RGB hypotheses
V5     PixelArena source teacher -> SimSiam -> pseudo labels -> Mean Teacher
V6     RGB-derived tracking and causal temporal diagnostics
V7     Rich PixelArena -> factorized ResNet-18 policy -> PixelArena-only loop
T8     mobile/video demonstrations -> causal policy diagnostics -> Shadow -> bounded gates
```

V5/V6 base training uses no human action, frame, HUD, tracking, or temporal labels. T8 is a
separate lineage and may use only its own standardized observed or actually dispatched testbed
events. Legal actions, cooldown state, and structured truth never enter an RGB Actor.

The visual-policy organization was informed by
[ResnetGPT](https://github.com/FengQuanLi/ResnetGPT),
[WZCQ](https://github.com/FengQuanLi/WZCQ), and
[wzry_ai](https://github.com/myBoris/wzry_ai). This repository does not copy their device-control
code, data, weights, coordinates, assets, or recordings.

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,bc,vision,shadow,preingest]'

make check
make accept
make accept-v2
make pixel-smoke
make shadow-live-smoke
make alignment-smoke
make temporal-smoke
make rich-smoke
```

The CI workflow uses CPU PyTorch. CUDA is required only for formal GPU acceptance and larger local
training runs; a sandbox that cannot see the GPU is not evidence that the host GPU is unavailable.

## External data storage

Large data is never committed. The public default is `.local-data/hok-agent-v5`, which is ignored
by Git:

```bash
make storage-init
make storage-show
```

Override it for another mounted data volume without changing code:

```bash
HOK_LARGE_ROOT=/absolute/private/path/hok-agent-v5 make storage-show
```

The external tree contains `datasets/`, `checkpoints/`, `runs/`, `cache/`, `audit/`, and
`staging/`. Manifests may store artifact basenames and anonymous hashes, never source-video
locators, account identifiers, raw touch dumps, or credentials.

## Read-only video

Offline input must be one non-symlink regular local recording:

```bash
python -m hok_agent shadow-video \
  --input /absolute/path/to/privacy-reviewed-recording.mp4 \
  --model /absolute/private/path/model.safetensors \
  --output-dir .local-data/hok-agent-v5/runs/shadow-offline-001
```

Live capture accepts only one explicitly selected non-symlink `/dev/videoN` V4L2 character
device. Numeric indexes, URLs, network streams, and automatic source selection are rejected.

## Mobile testbed: locked by default

The checked-in examples are intentionally invalid:

- `configs/mobile_testbed_identity.example.json`
- `configs/mobile_testbed_layout.example.json`

Create private local files only for a project-owned self-built app:

```bash
cp configs/mobile_testbed_identity.example.json configs/mobile_testbed_identity.local.json
cp configs/mobile_testbed_layout.example.json configs/mobile_testbed_layout.local.json
```

Fill the package, version, signing identity, APK SHA-256, owner attestation, date, display geometry,
and normalized control positions from your own build. Then calculate the canonical identity hash:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

path = Path("configs/mobile_testbed_identity.local.json")
payload = json.loads(path.read_text(encoding="utf-8"))
payload.pop("identity_sha256", None)
payload["identity_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
```

The default identity path may be overridden with `HOK_MOBILE_IDENTITY_PATH`. Identity and layout
files matching `*.local.json` are ignored by Git.

A read-only dry run still requires an explicit serial and capture source:

```bash
HOK_MOBILE_IDENTITY_PATH=configs/mobile_testbed_identity.local.json \
python -m hok_agent mobile-testbed \
  --serial YOUR_USB_SERIAL \
  --model /absolute/private/path/model.safetensors \
  --layout configs/mobile_testbed_layout.local.json \
  --output-dir .local-data/hok-agent-v5/runs/mobile-dry-run \
  --device cpu --run-seconds 60 --infer-hz 1
```

Device input additionally requires the explicit input flag and a finite action cap. Immediately
before every dispatched event, the runtime rechecks the serial, locally attested package,
foreground window, display identity, and guard watchdog. Missing identity, invalid identity hash,
package/version/signature drift, layout mismatch, disconnection, or backgrounding stops the run.
Accessibility, root, global hooks, HID/UHID, uinput, minitouch, generic shell execution, process or
memory inspection, protocol interception, evasion, and online learning are outside the supported
surface.

See [BOUNDARIES.md](BOUNDARIES.md) before enabling any mobile input.

## T8 status

T8-v2.7 is permanently frozen failed: its reports may be verified but not used to justify more
collection, threshold changes, or another four-class fit.

T8-v3 predicts five RGB-observable states from a V5-initialized ResNet-18 and 16-frame causal TCN:

- `enemy_visible`
- `attack_opportunity`
- `basic_ready`
- `skill1_ready`
- `skill2_ready`
- derived `confidence` and `abstain`

The single seed-0 run did not pass its frozen dev gates. Mean head macro-F1 was `0.450962`, minimum
positive recall was `0.314075`, normal-minus-shuffled margin was `0.023275`, and confidence
coverage was `0.039481`. Offline replay rejected the model; video-test, live Shadow, and all T8-v3
device stages remained unopened. These results are failure evidence, not a released policy.

T8-v4 narrowed the task to four local visual cues and used conservative dual-teacher weak targets.
Its RGB and temporal controls passed, but the selected model failed the frozen spatial-selectivity
gate. T8-v4 is therefore frozen failed with no replay, Shadow, or input permission. See
[docs/T8_V4_PROTOCOL.md](docs/T8_V4_PROTOCOL.md).

T8-v5 is a smaller offline successor that asks only whether each fixed correct ROI predicts the
frozen weak target better than time, a wrong ROI, and shuffled labels. Enemy, basic attack, and
skill1 are formal heads; skill2 is diagnostic-only because its frozen dev negative support is too
small. Basic attack passed, but enemy and skill1 failed the frozen wrong-ROI margin, so T8-v5 is
frozen without a TCN stage and remains non-promoting. See
[docs/T8_V5_ROI_PROTOCOL.md](docs/T8_V5_ROI_PROTOCOL.md).

The independent Basic-only MVP extracts only the passed basic ROI component and combines it with
the frozen enemy visual rule. Its complete video-dev replay produced six conservative candidates
with no invalid-screen or rate-limit violations. The admitted five-minute zero-control Shadow
completed all 1,500 cycles with low latency but produced zero candidates: the live basic score
never reached the frozen 0.80 threshold. The route is frozen without a probe or device input. See
[docs/T8_BASIC_MVP_PROTOCOL.md](docs/T8_BASIC_MVP_PROTOCOL.md).

A final deterministic rule fallback verified that the calibrated basic ROI appeared visually
ready. The first 0.80 smoke failed on capture variation; one fixed 0.75 engineering calibration
passed 100/100 cycles. Owner observation established that basic attack has no cooldown dimming, so
visual and execution coordinates were separated instead of requiring a false release event. The
corrected private touch point passed 20 actions, one minute, and five minutes without unexpected
input.

All four combat buttons were then moved to an acknowledged synchronous ADB tap sender. Two
independent 60-second mixed probes each executed basic attack, skill1, skill2, and skill3 exactly
five times, with 20/20 synchronous acknowledgements and zero unexpected actions per run. This is a
deterministic owner-testbed result; it is not model-driven gameplay and does not include movement,
aiming, or target selection.

The subsequent visual combat arbiter replaced the fixed button schedule with cooldown-aware
round-robin selection. Its 60-second gate executed 20 actions and its five-minute gate executed 58;
every command was synchronously acknowledged and owner-observed as normal. Skills must visibly
enter cooldown and recover before rearming. See
[docs/VISUAL_COMBAT_ARBITER_PROTOCOL.md](docs/VISUAL_COMBAT_ARBITER_PROTOCOL.md).

Its first formal event package has two diagnostic sessions and 78 synchronously bound actions.
Training remains blocked: the initial events have no RGB/features and only fixed-rate derived
timestamps. The next dataset gate requires twelve new timestamped feature sessions frozen as
8 train, 2 dev, and 2 test.

Mobile Operation Base v1 closes the first engineering part with one guarded two-pointer scrcpy
session. It continuously moves through eight directions while concurrently executing combat and
the single recommended-equipment purchase, observes the minimap, and stores four derived ROI
views. The five-minute gate and a live death/respawn hard-stop test passed. See
[docs/MOBILE_OPERATION_BASE_PROTOCOL.md](docs/MOBILE_OPERATION_BASE_PROTOCOL.md).

Operation Policy v1 starts the offline second part without reopening the failed T8 lineages. It
uses the frozen operation/combat evidence to train 200/500 ms inverse-dynamics heads, admits only
their high-confidence agreement on video-train/video-dev, and compares simple 16-frame movement
and combat policies. Purchase and hard-stop remain deterministic; the entire first contract is
zero-control and cannot connect to the phone. See
[docs/OPERATION_POLICY_V1_PROTOCOL.md](docs/OPERATION_POLICY_V1_PROTOCOL.md).
Its seed-0 inverse-dynamics gate is now frozen failed: spatial encoder features improved the
negative-control margin but did not recover reliable movement directions or combat classes. The
run stopped before video pseudolabels, policy fitting, Shadow, or phone input.

The separate Operation Direct Policy check used existing executed actions without connecting the
phone. It also failed: direction changes and combat classes were not predictable from automatic
round-robin schedules. Operation Base therefore closes the actuator and data-binding layer, not
tactical policy supervision. See
[docs/OPERATION_DIRECT_POLICY_V1_PROTOCOL.md](docs/OPERATION_DIRECT_POLICY_V1_PROTOCOL.md).

The active engineering route is now Operation Movement Teacher v1. It keeps the existing selected
combat model and learns only movement from a state-conditioned high-resolution minimap teacher.
Its offline 1,485-frame audit passed with 0.7838 coverage, all eight directions, and 3.59-pixel
player-jump P95. Live collection remains staged and fail-closed. See
[docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md](docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md).

Adaptive Layout and Hero Profiles v1 separates device geometry from hero skill behavior. Button
groups are located by structure rather than skill-icon appearance; local hero profiles define how
the three fixed skill slots execute. Unknown heroes remain skill-disabled. See
[docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md](docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md).

## Project documents

- [AGENTS.md](AGENTS.md): implementation authority and module constraints.
- [BOUNDARIES.md](BOUNDARIES.md): permitted and forbidden execution surfaces.
- [DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md): concise current-state ledger.
- [docs/T8_V4_PROTOCOL.md](docs/T8_V4_PROTOCOL.md): frozen T8-v4 diagnostic and promotion protocol.
- [docs/T8_V5_ROI_PROTOCOL.md](docs/T8_V5_ROI_PROTOCOL.md): T8-v5 isolated-ROI evidence gate.
- [docs/T8_BASIC_MVP_PROTOCOL.md](docs/T8_BASIC_MVP_PROTOCOL.md): deterministic basic-only gates.
- [docs/VISUAL_COMBAT_ARBITER_PROTOCOL.md](docs/VISUAL_COMBAT_ARBITER_PROTOCOL.md): cooldown-aware four-button arbiter.
- [docs/MOBILE_OPERATION_BASE_PROTOCOL.md](docs/MOBILE_OPERATION_BASE_PROTOCOL.md): frozen movement, combat, purchase, minimap, and hard-stop base.
- [docs/OPERATION_POLICY_V1_PROTOCOL.md](docs/OPERATION_POLICY_V1_PROTOCOL.md): offline inverse-dynamics and causal movement/combat route.
- [docs/OPERATION_DIRECT_POLICY_V1_PROTOCOL.md](docs/OPERATION_DIRECT_POLICY_V1_PROTOCOL.md): frozen executed-action learnability check.
- [docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md](docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md): active state-conditioned movement route.
- [docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md](docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md): device geometry and skill-behavior contracts.
- [docs/DELIVERY_HISTORY.md](docs/DELIVERY_HISTORY.md): sanitized historical ledger.

## License

Licensed under the [Apache License 2.0](LICENSE).
