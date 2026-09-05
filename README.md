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

## Current development plan

The active roadmap is [Engineering convergence plan](docs/ENGINEERING_CONVERGENCE_PLAN.md).
The next cycle targets a reproducible **offline** MVP: rule Macro/Combat, one goal-conditioned
Movement learner, and the existing observation/execution/transition interfaces. Budget ceilings
are 80 engineering hours, 24 GPU-hours and 50 GiB of new artifacts. Navigation is evaluated through
action-driven episodes, not static direction F1. Rules-only delivery is explicitly not a learned
policy success. Stage A now passes: a fixed blue-side Houyi bottom-lane rule trajectory moves through
RichPixelArena, writes every causal transition, reaches its target, commits STOP, and exits with zero
reward and zero device input. Stage B then passed the simulator overfit32 gate with a 686,281-parameter
task-specific GroupNorm+GRU, while both old-P0 branch variants failed. Full training and real-video
validity remain unproven. The task-specific model is now the default and all four diagnostic
attempts are closed. Run `make movement-mvp-stage-c-smoke` for the current regression surface.

The first Stage C candidate is now frozen failed. Its 64/24 trajectory source passed and training
loss converged, but epoch 20 completed only 15/24 dev episodes with 27.1% collision steps. Random
completed 18/24, exposing an infeasible +8 comparison on a 24-episode set. No model is promoted and
holdout remains unopened. The v2 contract now isolates navigation, requires three consecutive STOPs
and uses failure-inclusive efficiency. Its 64/24 recovery data passed, but both the uniform-sampling
and class-balanced candidates reached 0/24. This round's corrections are exhausted. Use
`--config configs/movement_mvp_stage_c_v2.json` for this route and `--reference-only` when
reevaluating an older checkpoint; reference results cannot promote a model.

The rule data path runs independently of learning. With the project Python environment active and
`HOK_LARGE_ROOT` set, use a new local output directory:

```bash
MOVEMENT_RUN_DIR="${HOK_LARGE_ROOT:?set HOK_LARGE_ROOT}/runs/hierarchical-movement-mvp/rule-batch-demo"
python -m hok_agent movement-mvp --mode rule-batch --episodes 10 --step-budget 4 --output-dir "$MOVEMENT_RUN_DIR"
python -m hok_agent movement-mvp --mode rule-batch --episodes 10 --resume --output-dir "$MOVEMENT_RUN_DIR"
```

This repeats the fixed Stage A scene using a structured simulator rule. It writes one SQLite Store,
an immutable `run-contract.json`, derived frame bundles and `batch-summary.json`; every episode ends
after three STOPs. Resume deterministically replays committed transitions and continues at the next
step. Missing/corrupt committed evidence or changed bindings stop recovery. This is PixelArena
mid-episode recovery, not learned navigation or phone-state recovery, and it loads no model or phone
interface. Actual results and remaining work are recorded only in
[DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md).

Create and independently verify the final local R0 package with the same top-level command:

```bash
MOVEMENT_EVIDENCE_ROOT="${HOK_LARGE_ROOT:?set HOK_LARGE_ROOT}/runs/hierarchical-movement-mvp"
python -m hok_agent movement-mvp --mode package \
  --source-run "$MOVEMENT_EVIDENCE_ROOT/stage-d-rule-batch-v2-recovery" \
  --control-run "$MOVEMENT_EVIDENCE_ROOT/stage-d-rule-batch-v2-continuous" \
  --output-dir "$MOVEMENT_EVIDENCE_ROOT/r0-delivery-v1"
python -m hok_agent movement-mvp --mode package --verify-only \
  --output-dir "$MOVEMENT_EVIDENCE_ROOT/r0-delivery-v1"
```

The package is an immutable local evidence directory with a resolved config, final summary, compact
SQLite backup, 100 derived frame bundles and a final manifest. It contains no trained checkpoint.
Its delivery grade is rule-only `R0_RULE_OFFLINE`; it does not establish learned navigation or
real-video/mobile performance.

The next-cycle real-RGB observability preflight is also available through `movement-mvp`:

```bash
python -m hok_agent movement-mvp --mode real-rgb-preflight \
  --config configs/movement_real_rgb_preflight_v1.json \
  --target-root "$HOK_LARGE_ROOT/datasets/v5-target-file-atomic-v2" \
  --output-dir "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-observability-v1"
```

The frozen v1 run failed: overall self/target pair coverage was 0.4792, one train session reached
only 0.0938, and marker jumps reached 0.4128. It read no test frames, stored no RGB, trained no model
and sent no input. A new minimap detector/tracker contract is required before R2 or Movement
training; rerunning v1 with lower gates is not an accepted repair.

V2 removes unstable red-target selection from the canvas and accepts a semantic goal from the
deterministic Macro context:

```bash
python -m hok_agent movement-mvp --mode real-rgb-goal-canvas \
  --config configs/movement_real_rgb_goal_canvas_v2.json \
  --prior-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-observability-v1/report.json" \
  --target-root "$HOK_LARGE_ROOT/datasets/v5-target-file-atomic-v2" \
  --output-dir "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-goal-canvas-v2"
```

The frozen v2 run passed crop, repeatability and two-goal counterfactual checks on all 288 frames.
It does not verify player localization, the gameplay meaning of the fixed lane coordinate, or policy
performance, so training, promotion and R2 remain closed.

The next simulator-only learnability gate uses the same minimap-plus-goal-ring task:

```bash
python -m hok_agent movement-mvp --mode goal-canvas-overfit32-materialize \
  --config configs/movement_goal_canvas_overfit32_v1.json \
  --goal-canvas-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-goal-canvas-v2/report.json" \
  --output-dir "$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/goal-canvas-overfit32-v1"
python -m hok_agent movement-mvp --mode overfit32 \
  --config configs/movement_goal_canvas_overfit32_v1.json \
  --dataset "$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/goal-canvas-overfit32-v1/overfit32.npz" \
  --output-dir "$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/goal-canvas-overfit32-seed0-v1" \
  --device cuda --architecture task-specific
```

The frozen run passed accuracy 1.0 and loss 0.00614 with all nine recalls at 1.0. Its checkpoint is
diagnostic-only; formal trajectory training must initialize a fresh model.

The fresh 64/24 goal-canvas candidate has also completed. Teacher and exact-RGB geometry reached
24/24, but the learned epoch-20 checkpoint reached only 9/24 despite training loss 0.01414. It is
frozen failed; no holdout or R2 stage opened. The next learned correction must change only the
model's spatial-relation bias under a new contract, rather than add more epochs or data to this run.

The plan supersedes the future schedule and growth proposals in the historical sections below.
It does not change frozen results, reopen video-test, authorize phone control or start RL.
See [DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md) for executed state.

## Historical architecture and reusable components

```text
V1–V3  frozen deterministic and behavior-cloning regressions
V4     local video/V4L2 -> read-only RGB hypotheses
V5     PixelArena source teacher -> SimSiam -> pseudo labels -> Mean Teacher
V6     RGB-derived tracking and causal temporal diagnostics
V7     Rich PixelArena -> factorized ResNet-18 policy -> PixelArena-only loop
T8     mobile/video demonstrations -> causal policy diagnostics -> Shadow -> bounded gates
HP-v0  FrameBus -> RGB events + shared temporal policy -> deterministic Router -> unified Replay
```

V5/V6 base training uses no human action, frame, HUD, tracking, or temporal labels. T8 is a
separate lineage and may use only its own standardized observed or actually dispatched testbed
events. Legal actions, cooldown state, and structured truth never enter an RGB Actor.

The visual-policy organization was informed by
[ResnetGPT](https://github.com/FengQuanLi/ResnetGPT),
[WZCQ](https://github.com/FengQuanLi/WZCQ), and
[wzry_ai](https://github.com/myBoris/wzry_ai). This repository does not copy their device-control
code, data, weights, coordinates, assets, or recordings.

## Hierarchical Policy v0 historical development route

The next development route keeps one RGB PolicyBundle with a shared temporal representation and
three logical heads: Macro, Movement, and Combat. An independent, versioned VisualEventEngine
derives terminal, death/respawn, and self-health events from RGB for RewardHub. A deterministic
Router owns freshness, masks, pointer conflicts, and persistent-joystick execution. Demo, simulator,
controller, online, and failure rows share one episode-major TransitionStore with different source
tags and samplers.

The deliberately small E0 implementation is complete: an immutable latest-frame FrameBus,
VisualState/Event plus exact-once fusion, and a transactional SQLite UnifiedTransitionStore now
validate proposal freshness, action/capture ordering, terminal retention, and episode continuity.
E0 itself is offline-only and has no detector, trained Bundle, online RL, MoE, continuous joystick output,
PPO, or model-driven mobile input. Run its focused regression with `make hierarchical-e0-smoke`.
The full contracts, data roles, parameter tiers, training order, and 1/3/10-episode gates are in
[docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md](docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md).

E1a now adds an offline centered-health-bar diagnostic for HP-change candidates and temporal
death/respawn events. Its first cross-session width contract failed; the single preserved repair
passed the engineering gate with zero train/challenge false deaths and one dev death/respawn pair.
It remains explicitly non-promoting: numeric HP accuracy and independent semantic accuracy are not
verified, so Reward stays disabled. Run its synthetic regression with
`make hierarchical-e1-health-smoke`.

The frozen E1b OCR coverage audit failed: only 1/8 train and 2/4 dev sessions exposed a result
screen in the sampled tail, and neither split produced a reliable `WIN` or `LOSS` token. The report
contains only allowlisted token counts and anonymous IDs. This lineage is not repaired by changing
OCR confidence, tail duration, or sampling rate; the next terminal route must use short-video
crystal-destruction transitions. Run its local regression with
`make hierarchical-e1-terminal-smoke`.

The separate E1c anchor preflight then scanned the final decoded frame of every train/dev session
and found 34 train plus 7 dev result-page anchors with zero outcome conflicts. Video position was
not a label, anchor frames are forbidden from future model input, and video-test was not decoded.
This passes only the data-support gate for constructing preceding short clips; dynamic terminal
accuracy, WIN/LOSS, Reward, and promotion remain unverified. Run its regression with
`make hierarchical-e1c-anchor-smoke`.

E1c-clip materialization produced 34 train and 7 dev same-session triplets: one terminal-transition
candidate, one nearby late-game negative, and one earlier negative per anchored session. The model
shards contain only RGB sequences, labels, and split-only anonymous IDs; anchor frames, timestamps,
relative offsets, OCR, and paths are absent. This passes materialization only, not semantic
terminal detection. Run `make hierarchical-e1c-clip-smoke`.

The mandatory E1c-probe stopped before visual training: within-session materialization ordinal
predicts all three labels with train/dev macro-F1 `1.0/1.0`. Visual accuracy on this dataset could
not distinguish terminal semantics from match-time progression. The fixed-offset lineage is frozen;
no GPU model was trained. Run `make hierarchical-e1c-probe-smoke`.

E1d replaces fixed offsets with visual consensus: temporal change, white explosion, and persistent
structure change. One preserved repair produced 33 train and 7 dev weak candidates. WIN/LOSS and
Reward remain closed. Run `make hierarchical-e1d-crystal-smoke`.

E1d-clip built 33 train and 7 dev event-centered same-scene pairs. Hash-based pair reversal reduced
ordinal-only accuracy to 0.424/0.429. Run `make hierarchical-e1d-clip-smoke`; Reward remains closed.

The CPU seed-0 probe passed overfit32 and reached dev temporal macro-F1 1.0 versus 0.8571 for
last-frame and shuffled controls. This is learnability evidence only. Run
`make hierarchical-e1d-probe-smoke`.

The three pre-test checkpoints are frozen in safetensors with bundle hash
`55a679883119cdbf1a6a7703f945d61ce33408bad84013362e66355e83345c79`. Their dev metrics exactly
reproduce the probe. A one-shot 23-session test contract is frozen before test decoding; it permits
no retraining or threshold adjustment.

That one-shot test was consumed and failed at runtime on a test session without a usable
pre-result sequence. No metric report was produced, and the frozen failure explicitly sets
`rerun_allowed=false` and `integration_allowed=false`. Offline EventEngine integration was not
started.

The follow-up E1e diagnostic did not reopen test. It continued across all 85 unused unanchored
train/dev sessions, but only 6 train and 1 dev pairs were eligible versus 10/3 required; 63 sessions
lacked a usable pre-result sequence. E1e is frozen insufficient and cannot replace the formal test.

P0 then compared the existing epoch-3 SimSiam adapter with its exact source encoder and a random
ResNet-18 on frozen E1d features. All three reached dev macro-F1 1.0, so the adapter showed no
incremental value and is rejected as the new PolicyBundle initializer. Run
`make hierarchical-p0-adapter-smoke`.

The harder P0 temporal-order dataset contains 128 train and 32 dev pairs with identical frame sets
and endpoints. Adapter temporal macro-F1 was 0.3333 versus source 0.4687 and random 0.5142, so the
old adapter is rejected again and cannot initialize the new PolicyBundle.

The separate seed-0 temporal SSL pilot passed its 32-sample overfit and non-collapse checks, and
improved dev macro-F1 to 0.7031, 0.1889 above the best frozen baseline. It nevertheless missed the
pre-frozen 0.75 dev gate. The run is frozen failed without an encoder checkpoint; it cannot
initialize PolicyBundle or open policy training, Reward, test, capture, or input. Run the contract
regression with `make hierarchical-p0-ssl-smoke`.

The separately versioned P0 temporal SSL v2 then indexed 1,648 train-only pairs across all 103
video-train sessions without copying RGB or using video-dev for checkpoint selection. Its fixed
last epoch reached frozen dev macro-F1 0.7907 and passed every predeclared gate. The resulting
ResNet-18 plus GRU representation is allowed only as the P0 initializer; no policy head, Reward,
test, capture, or input is opened. Run `make hierarchical-p0-ssl-v2-data-smoke` and
`make hierarchical-p0-ssl-v2-smoke`.

P1 began with a raw-resolution Movement teacher audit over 103 train and 23 dev videos. A preserved
display-matrix repair recovered every rotated video, but the frozen teacher still covered only
18.44%/18.98% of sampled frames and dev produced only four stable west labels versus sixteen
required. The audit is frozen failed; no Movement Head was trained and its labels are explicitly
teacher recommendations, not observed human actions. Run `make hierarchical-p1-movement-audit-smoke`.

The separate P1 Macro data audit passed on project-owned PixelArena: 1,620 train and 412 dev causal
windows cover FARM_LANE, PUSH_STRUCTURE, and ENGAGE in every 40/10 episode. Class-prior and
time-only controls reached only 0.2019 and 0.3893 macro-F1. This permits one simulator-only Macro
Head learnability run from the frozen P0 representation; it is not real-video semantic evidence.
Run `make hierarchical-p1-macro-data-smoke`.

The frozen-P0 Macro Head run then failed: dev macro-F1 was 0.3868 versus 0.3893 for the time-only
control, and its gain over label shuffle was only 0.0845. Overfit32 also missed its loss gate.
No Macro Head checkpoint was saved. This does not invalidate P0 temporal-order learning, but it
does show that the frozen representation cannot directly supply the required simulator Macro
semantics through this fixed canvas/head. Run `make hierarchical-p1-macro-head-smoke`.

The P1 Combat data audit also stopped before training. Eight five-minute sessions contain adequate
button counts, but a 200 ms clock-only lookup reaches macro-F1 0.9331 and the sessions expose only
two positive action sequences. The artifacts also do not bind a verified Houyi identity. These
rows validate deterministic cooldown execution, not tactical action selection. Run
`make hierarchical-p1-combat-data-smoke`.

P1v2 therefore replaces the failed fully frozen transfer assumption with a bounded task-specific
adapter design. P0 through layer2 remains shared and frozen; Macro and Movement each own an
independently trained 8–14M layer3/layer4 plus GRU branch at 2 Hz and 10 Hz. Combat v0 remains the
10 Hz deterministic visual cooldown arbiter until non-clock, hero-bound data exists. Models emit
proposals only; the deterministic Router owns freshness, concurrent pointers, death, and hard stop.
Run `make hierarchical-p1v2-architecture-smoke`.

The first P1v2 Movement audit found only east/west moves in the Global Agent dataset: 376/226 train
and 115/45 dev, with all six other directions absent. Skill aim and wait were excluded rather than
misused as movement. A read-only check also found that historical V7 fit/acquisition move rows are
all ego-view east. No Movement branch was trained; a new balanced, visually conditioned 2D
PixelArena navigation source is required. Run `make hierarchical-p1v2-movement-data-smoke`.

The new source materializes 512 train and 128 dev RGB sequences with exact 64/16 support for each
direction, alternating blue/red ego views and zero group overlap. It permits one task-specific
Movement branch learnability run, but its claim is limited to approaching a locally visible target;
lane strategy and real-video transfer remain unverified. Run
`make hierarchical-p1v2-movement-2d-smoke`.

The first task-specific branch reached perfect eight-direction dev macro-F1 and recall, far above
the 0.1748 label-shuffle control, but failed its mandatory overfit32 gate after the sole batching
repair: accuracy 0.9375 and loss 0.1704 missed 0.95/0.05. No checkpoint was retained. The result is
learnability evidence only; a future normalization-stable branch must use a new versioned contract.
Run `make hierarchical-p1v2-movement-branch-smoke`.

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,events,events-ocr,bc,vision,shadow,preingest]'

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

Operation Movement Teacher keeps the persistent joystick but replaces the fixed movement schedule
with the frozen nearest-target minimap rule. Its spatial v1.1 repair audits a pool of automatic
sessions, freezes exactly three train and one dev session only after real eight-direction coverage,
then runs a balanced 32-window overfit before one offline seed-0 pilot. It does not use human
labels, PPO, Shadow, or model-driven phone input. See
[docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md](docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md).
The read-only `mobile-operation-team-side` preflight detects blue versus red from the unique
yellow self-card highlight on the loading panel and fails closed on gameplay or ambiguous frames.
The formal collector then uses that result for a summary-only 20-second marksman opener derived
from the existing blue-bottom human trace, mirrored for red, before any training row is recorded.

The separate `mobile-marksman-lane-controller` command skips movement learning entirely. It uses
the verified side, the same opener, a deterministic advance/hold lane cycle, existing visual
combat and purchase rules, and opener replay after death. See
[docs/MARKSMAN_LANE_CONTROLLER_PROTOCOL.md](docs/MARKSMAN_LANE_CONTROLLER_PROTOCOL.md).

The frozen simulator foundation is Global Agent v1: a structured simulator rule teacher completes full games and
labels macro intent plus semantic target zone; an RGB student learns those high-level decisions,
while the existing deterministic navigation, combat, purchase, layout, hero-profile, and safety
modules execute them. Its five offline stages are complete: the frozen GlobalArena rule regression
is 19/20 (with 20/20 tower progress), seed-0 BC reached 7/10 pure-student simulator terminals, the
only DAgger round reached 9/10, and the fresh 20-seed holdout selected DAgger at 18/20. The adapter
improved video consistency but reduced simulator terminals, so it is not promoted. The selected
student then passed only 2/6 fixed challenge states. Two separately authorized zero-control mobile
diagnostics passed runtime safety but produced constant `DISENGAGE/OWN_BASE`; the 10-minute Shadow
and every input stage remain closed.
See [docs/GLOBAL_AGENT_V1_PROTOCOL.md](docs/GLOBAL_AGENT_V1_PROTOCOL.md).

The completed non-promoted repair is Human IfO Bridge v1. It uses complete human-match videos as observation-only
behavior demonstrations, learns Human/Sim temporal representations, trains inverse macro dynamics
from GlobalArena truth, and then fits Human-BC with frozen-Dagger distillation. Scenario cards are
diagnostic-only; direct pixel-similarity reward and early reinforcement learning are not admitted.
See [docs/HUMAN_IFO_V1_PROTOCOL.md](docs/HUMAN_IFO_V1_PROTOCOL.md).

The reproducible entrypoints are `global-agent-evaluate`, `global-agent-materialize`,
`global-agent-train`, `global-agent-dagger`, `global-agent-domain-adapt`, and
`global-agent-replay`, `global-agent-holdout`, and `global-agent-challenge`. Their Make targets use
`HOK_LARGE_ROOT`; dataset, checkpoint, video, and report artifacts are never committed. The public
summary is [docs/GLOBAL_AGENT_V1_OFFLINE_EVIDENCE.json](docs/GLOBAL_AGENT_V1_OFFLINE_EVIDENCE.json).

Adaptive Layout and Hero Profiles v1 separates device geometry from hero skill behavior. Button
groups are located by structure rather than skill-icon appearance; local hero profiles define how
the three fixed skill slots execute. Unknown heroes remain skill-disabled. See
[docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md](docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md).

## Project documents

- [AGENTS.md](AGENTS.md): implementation authority and module constraints.
- [BOUNDARIES.md](BOUNDARIES.md): permitted and forbidden execution surfaces.
- [DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md): concise current-state ledger.
- [docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md](docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md): active
  development architecture, contracts, training order, and acceptance gates.
- [docs/GLOBAL_AGENT_V1_PROTOCOL.md](docs/GLOBAL_AGENT_V1_PROTOCOL.md): frozen full-episode macro-policy foundation.
- [docs/GLOBAL_AGENT_V1_CONVERGENCE_ROADMAP.md](docs/GLOBAL_AGENT_V1_CONVERGENCE_ROADMAP.md): frozen v1 route, gates, and stop conditions.
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
