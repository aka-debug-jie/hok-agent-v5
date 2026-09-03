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

## Hierarchical Policy v0 development route

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
