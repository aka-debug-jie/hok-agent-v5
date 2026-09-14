# Delivery progress

This is the only current-state ledger. The sanitized historical ledger is preserved in
[`docs/DELIVERY_HISTORY.md`](docs/DELIVERY_HISTORY.md). Large datasets, checkpoints, recordings,
and run evidence are local artifacts below `HOK_LARGE_ROOT`; they are not distributed in Git.

## Current execution state

Updated 2026-09-14: the active-probe route is closed after two budgeted batches failed the A gate.
Only this section schedules work; all experiment entries below are historical evidence.

```text
OBJECTIVE: prepare the bounded multi-direction active probe for no-source identity and control
STATUS: STOPPED
NEXT_ACTION: none on this route; any further probe work requires a new owner decision outside the stopped contract
INPUT_EVIDENCE: frozen action-response audit, the four user probe requirements and the two authorized batches
CHANGED_FILES: probe contract, audit, planner, runner/CLI, runner fixes (scene readiness, guard refresh, advancing observations) and focused tests
PRIMARY_METRIC: pre-registered coverage/fate accounting and pulse-versus-control separation
BASELINE: v1 audit reported paired-event responses only, with no release semantics or denominators
RESULT: two budgeted batches failed the A gate. Batch 2: coverage 1.00/0.972, paired 0.875/0.896, all eight directions, longest valid 96.6/58.4 s, fixed-UI responsive 0, identity switches 0/1, but direction-correct 0.214/0.140 and pulse median projection 0.029/-0.021 px with pulse-minus-control -1.73/-0.91 px; audit ACTIVE_PROBE_GATES_FAILED
ENGINEERING_HOURS_USED_AND_CAP: not instrumented yet, cap 4 h
GPU_SECONDS: 0, cap 0
NEW_BYTES: 118,127,296 total (batch 1: 88,820,785; final batch: 29,306,511); contract budget 52,428,800
STOP_REASON: two budgeted batches failed the A gate; the pre-declared global stop judges the no-source identity/control direction data-source limited; no further batch, threshold change or rerun
NEXT_DECISION: option (c) entered by rule: read-only or local scope only; a future restart needs a new owner decision
```

- The main checkout's older Global Agent `CURRENT GOAL` statement is historical; this worktree
  ledger is the only current scheduler until this branch is merged.
- Completed scope: N1 visual evidence, N2 same-run Store/recovery and one verification/handoff.
- The existing `R1_HIERARCHICAL_RULE_OFFLINE` package at commit `727d360` remains frozen.
  It contains a synthetic 10-episode/120-step geometry report beside the older R0/R1 runtime evidence.
  It does not establish that the new geometry policy already shares the old Store/resume path.
- The batch used no training, video-test, mobile connection or human recording/label. N1 exposed no
  new valid source, so this simulator batch is closed and cannot substitute another training cycle.
- Planning and documentation time for this reset was not instrumented and is UNKNOWN; it is not
  included in the unstarted N1/N2 0/12 h figure.

### 2026-09-09 development review and factual corrections

- Code inspection: `run_change_geometry_replay` in movement_goal_canvas.py appends plain step
  dictionaries and writes report.json; it has no UnifiedTransitionStore or recovery argument.
  `run_rule_batch` in movement_mvp.py separately supplies timestamped transitions and recovery.
  The package nests their evidence; component composition must still be demonstrated by N2.
- The current geometry replay gets positions from RichPixelArena and renders exact markers;
  352/352 and 10/10 are therefore simulator-only evidence, not real-player observability.
- The frozen package has 403 manifest-listed payload entries and 406 files in total, including
  three manifest.json files. The earlier wording “403 files” was incorrect.
- The shared environment currently reports Python 3.11.15, consistent with pyproject >=3.11.
  The earlier “Python3.10” description was incorrect.
- The 492-test full check occurred before the final nine-line verifier edit; the final code then
  passed 25 focused tests, Ruff, strict mypy and real-package verification. Do not describe that
  historical full run as a fresh full-suite result for the later code tree.
- The new plan defines one task at a time, same-question budgets, behavior-based acceptance,
  short development batches and risk-based checks. Previous AGENTS/README/plan text is retained
  verbatim in docs/DELIVERY_HISTORY.md as non-active historical snapshots.
- This batch edits documentation only. No algorithm, dataset, threshold, checkpoint, device
  interface or frozen external artifact was changed. Verification is diff/link/consistency only.
- Documentation checks passed: three verbatim historical snapshots, two unchanged execution-boundary
  sections, twelve added local links/anchors, one active task card and git diff --check.
  Source, tests, configs, game rules, dependencies and Makefile have no changes; no training or
  historical test suite was run for this plan reset.

### N1 real navigation demonstrator and N2 integrated runtime (2026-09-09)

- N1 reproduced localization v2 on all4,455 cached 128x128 minimap frames. Direct observations are
  208/1,485 for session002,0/1,485 for003 and4/1,485 for005; maximum unknown streak is1,485.
  Four10-second GIFs show the densest valid002 interval, longest unknown002 interval and fixed-UI
  intervals in003/005. A same-frame counterfactual changes the explicit goal and proposal SE→NW.
- N1 remains `DATA_SOURCE_LIMITED`; localization was not changed. It used no gap filling, recorded
  future as action effect, training, test or device input. Output:
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/n1-real-navigation-demo-v1`.
  Summary file SHA-256 `546147dfbc3f9105bed3194be9dc28b78101e6685afabb2b0dcfd047568ce304`.
- N2 extends the existing rule-batch path with RGB goal-marked minimaps and optional navigation
  context in the existing Store. Every step checks the current goal; matching direction yields
  KEEP, exact arrival yields Router STOP, intermediate STOP advances the goal, and the final goal
  requires three STOPs. The first smoke exposed the old one-cell stop radius; the new multi-goal
  path alone uses exact radius0 while legacy callers retain radius1.
- The paused/resumed and continuous runs both pass10/10 episodes with140 transitions,10 terminal
  rows,40 goal versions,30 goal changes,80 direction actions,40 KEEP and60 Router STOP. All eight
  directions occur. Transition SHA-256 is
  `a56fc55e205536b181b034b3db69bd1d0151989a76eadaeee78de1a57be01b91` and frame-view SHA-256 is
  `ee158ec107f1e972e8504e735a4a43335c846b18c543a0a0998f87bd3975bcb8` in both runs.
- Recovery restored four committed transitions; SQLite integrity is ok, all rewards are0, model
  runs0 and input0. Outputs:
  `$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/n2-multigoal-recovery-v1` and
  `$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/n2-multigoal-continuous-v1`.
- New artifacts total12,809,566 bytes. GPU use is0. Observed implementation wall-clock through the
  final delivery check is about0.54h; historical engineering time remains unknown.
- Focused N1/N2, Store and boundary tests passed. The first full check found one static-boundary
  failure because the new GIF test imported Pillow outside the existing allowlist; adding that
  exact focused test path was the only repair. The repeated `make check` passed Ruff, strict mypy
  on70 source files, all497 tests in118.04s and project safety on273 files/131 Python files/72,802
  nonblank Python lines. `git diff --check` also passed.

### P0 session002 partial navigation Shadow (2026-09-09)

- Added `movement-mvp --mode real-navigation-shadow`. It reads the frozen N1 summary and all six
  session002 minimap shards, reuses the same detector and fixed goal, and writes only one
  `shadow.jsonl` plus a self-hashed summary. No RGB, SQLite or model artifact is copied.
- All1,485 frames are processed at the recorded200ms period. Direct candidates are208 and unknown
  frames1,277 across27 valid runs; the longest valid run is33 frames/6.6s. Proposals are E149,
  SE39 and NE20. Commands are DOWN27,KEEP175,MOVE6,UP27 and NOOP1,250.
- Every unknown has `proposal=null` and Shadow STOP. The first unknown after a direction produces
  UP, continuous unknown produces NOOP, and reacquisition starts with DOWN. Executed actions,
  transitions, training, test, GPU and input are all0; recorded future is not used as action effect.
- Status is `PARTIAL_OFFLINE_SHADOW_COMPLETE_DATA_SOURCE_LIMITED`. Output:
  `$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/session002-partial-navigation-shadow-v1`.
  Summary file SHA-256 `1c9eba075182b12845eda852c0c8f9a8d5a5ee8e4c4e355730b0dd9634400327`;
  shadow JSONL SHA-256 `7751a71bf13f05ed7679ac4fe4065a734ccf41a267588a61d0fec15c0583b9ef`.
- The output occupies578,468 bytes and runtime is3.88s. Observed implementation wall-clock through
  final checks is about0.21h, GPU0. This completes the selected partial route; it does not meet the
  80% coverage or continuous10s observation gate, so navigation stops pending a new reference.
- Focused Shadow/boundary tests, Ruff and strict mypy passed. The single full `make check` passed
  all499 tests in113.97s and project safety on276 files/133 Python files/73,209 nonblank Python
  lines. `git diff --check` passed.

### No-source action-response identity audit (2026-09-11)

- Added `movement-mvp --mode action-response-identity-audit`. Candidate pairs are generated from
  the two RGB frames without an action argument; the recorded action is applied only afterward to
  score1,000ms displacement. Only unique start/end candidates within the interior or fixed-UI
  group are retained.
- The three sessions contain54 acknowledged non-wait movement events and62 unique candidate pairs.
  Session002 contributes17 interior pairs over north/north-east/south/south-east/south-west;
  14/17 have projection at least1 pixel, responsive fraction0.8235 and median projection2.009px.
- Fixed UI contributes45 pairs across all sessions. Responsive pairs are0/45 and median displacement
  is0.054px. Sessions003/005 contribute no interior pair. All six frozen gates pass.
- Status is `ACTION_RESPONSE_SEPARATES_FIXED_UI_SESSION002_ONLY`. This supports a future active
  probe design but keeps semantic identity and multi-session identity false. Training, test, GPU
  and input are0. Output:
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/action-response-identity-v1`.
  Report file SHA-256 `a657ff96423feeb2ec717a82b2cde4b0053b41d91341796cd19551c0b01bb7ff`;
  report self-hash `d231ffa189e247442974a1704ccb7f695ded50efea8270296f4db51f584b1a3b`.
- Output size is29,333 bytes and formal runtime2.40s. Existing data cannot validate a second
  responsive session, so the next evidence must come from a new controlled action sequence rather
  than another pass over these recordings.
- Focused audit/boundary tests, Ruff and strict mypy passed. The single full `make check` passed
  all501 tests in116.01s and project safety on277 files/133 Python files/73,647 nonblank Python
  lines. `git diff --check` passed.

## Public release state

- Version: `0.1.0`.
- License: Apache-2.0.
- Public repository: `aka-debug-jie/hok-agent-v5`.
- The public tree contains no authorized mobile build identity, calibrated layout, device serial,
  recording, model, dataset, or source locator.
- Mobile input is fail-closed until a local Git-ignored identity file and layout are supplied and
  every serial, package, version, signature, foreground, display, duration, and action guard passes.

## Route status

Future scheduling is governed by `docs/ENGINEERING_CONVERGENCE_PLAN.md`; the historical route
restrictions below are not a queue of new work. Frozen experiment outcomes remain unchanged.

| Historical route | Last recorded result | Promotion boundary |
|---|---|---|
| Engineering convergence | `R1_ENGINEERING_OFFLINE_ZERO_REWARD`: R0 rule runtime, zero-reward Event replay and four failure boundaries packaged and verified | Cycle closed; no policy/Reward/mobile promotion without a new semantic source |
| Houyi-bound real data | `HOUYI_BOUND_REAL_DATA_NOT_AVAILABLE`: 103 train/23 dev metadata have no hero identity; profile and real-session bindings are absent | Requires future episode declaration plus immutable identity reference; no retroactive labels |
| Hierarchical Policy v0 | Historical `P1V2_MOVEMENT_BRANCH_FAILED`: full dev F1 1.0, but repaired overfit32 was 0.938 with loss 0.170 | No checkpoint; static-direction result is not action-driven navigation evidence |
| Global Agent v1 | `SHADOW_ROI_REPAIR_COMPLETED_DIVERSITY_NOT_DEMONSTRATED`: v1 and local-ROI v1.1 both passed runtime | Challenge 2/6 blocks all input; constant candidate output blocks 10m Shadow |
| Global challenge curriculum | `FROZEN_NON_PROMOTED`: v1 reached canonical 4/6 but parameter holdout only 12/24, stuck rose 4.99%→6.41%, and tower damage fell 12.0→11.85; conservative v2 returned to 2/6 and still regressed episodes | Both candidates rejected; no further curriculum weighting, frozen Dagger remains selected |
| Observable-factor representation | `PERMANENTLY_FROZEN_FAILED`: frozen-latent probe found health/base/distance/ordinary-lane F1 0.22–0.49; the only layer4/project auxiliary update improved some probes but fell to 12/20 terminals, 2/6 canonical and 8/24 parameter holdout | Attempt exhausted; no more encoder unfreezing, auxiliary weighting, or model optimization; v1 Dagger is permanent |
| Human IfO Bridge v1 | `ENCODER_REBIND_NON_PROMOTED`: Human-adapted frozen encoder plus GlobalArena-only supervision matched Dagger at 18/20 terminals and 20/20 tower progress, but challenge stayed 2/6 and stuck rose 4.99%→5.69% | Candidate rejected; Human video is visual-adaptation evidence only, frozen Dagger remains selected, no Shadow or input |
| V1/V2/V3 | Frozen regression baselines | No schema or identity changes |
| V4 | Offline video and explicit V4L2 read-only inference | No control output |
| V5 | Zero-label real-video adaptation pipeline implemented | Non-promoting without frozen local evidence |
| V6 | RGB-derived tracking and temporal diagnostics implemented | Outputs remain `ABSTAIN` |
| V7 | Rich PixelArena closed loop implemented and regression-tested | PixelArena only |
| T8-v1–v2.6 | Historical demonstration, causal-policy, Shadow, and bounded-probe evidence | Local evidence only |
| T8-v2.7 | `FROZEN_FAILED` | No recollection, threshold changes, or four-class retraining |
| T8-v3 | Video-state seed-0 pilot failed admission | No replay, Shadow, or device input |
| T8-v4 | `FROZEN_FAILED`: weak targets learnable, spatial-selectivity gate failed | No more repair, training, replay, Shadow, or input |
| T8-v5 | `FROZEN_FAILED`: only basic passed the per-head ROI gate | No TCN, replay, Shadow, capture, or input |
| Basic MVP | `FROZEN_FAILED`: offline passed, five-minute Shadow produced zero candidates | No probe or control stage |
| Basic rule engineering | Corrected private touch point; 20-action, 1-minute, and 5-minute runs passed | Deterministic basic only |
| Synchronous combat probe | Two 60-second repeats passed; each button executed 5/5 per run | Four tap buttons only; no movement/aim/target |
| Visual combat arbiter | 60-second and five-minute cooldown-aware gates passed | Deterministic four-button loop; no model/movement/aim/target |
| Visual combat event data | 2 diagnostic sessions, 1,770 rows, 78 synchronized actions | Training blocked until 12 timestamped feature sessions |
| Mobile Operation Base | `PASSED`: 5-minute movement+combat+purchase+minimap and live death stop | First part frozen; no enemy/target/aim/tactics yet |
| Operation Policy v1 | `FROZEN_FAILED`: source-clock/spatial IDM still failed movement and combat gates | No pseudolabel, policy, test, Shadow, capture, or input |
| Operation Direct Policy v1 | `FROZEN_FAILED`: executed schedules failed transition and combat gates | No Shadow, capture, or input |
| Operation Movement Teacher v1 | Fresh blue session 005 passed 1,485 rows at 0.9785 teacher coverage with the trace-derived bottom-lane opener excluded from training; eligible pool is now 002/003/005, but 005 still contains only south/south-west | Direction audit remains incomplete; do not freeze split or train until real east/west/north-west support is collected |
| Deterministic Marksman Lane Controller v1 | Implemented automatic side binding, side-specific opener, lane advance/hold, visual combat/purchase, and death-triggered re-lane state machine | Non-learning bounded owner-testbed control only; no model or training claim |
| Adaptive Layout / Hero Profiles v1 | Geometry and behavior contracts implemented with synthetic tests | Read-only per-device calibration required before integration |
| Global combat feature cache v1 | 32x1024 float16 cache completed; frozen-feature TCN head failed dev evidence | Preserve cache; do not promote the new head |

## T8-v2.7 freeze

The three existing calibration failures are bound by one immutable failure manifest. The contract
sets rerun, threshold-change, four-class-training, Shadow, and device-input permissions to false.
The frozen reports remain local and are not rewritten for publication.

## T8-v3 video-state closure

T8-v3 replaced hidden action choice with five RGB-observable state outputs:

- `enemy_visible`
- `attack_opportunity`
- `basic_ready`
- `skill1_ready`
- `skill2_ready`
- derived `confidence` and `abstain`

Skill priority, cooldown, global rate limiting, and repetition caps remain deterministic execution
state rather than Actor input. The frozen data contract reused V5-initialized 16x512 causal
features from 103 video-train and 23 video-dev sessions, producing 12,544 train and 3,394 dev rows.
Video-test was not opened.

The single allowed seed-0, eight-epoch run completed but failed its immutable admission gates:

| Metric | Result | Required |
|---|---:|---:|
| Mean dev head macro-F1 | 0.450962 | 0.70 |
| Minimum positive recall | 0.314075 | 0.55 |
| Normal minus shuffled macro-F1 | 0.023275 | 0.15 |
| Confidence coverage | 0.039481 | 0.50 |
| Black/gray OOD abstention | 1.000000 | 0.95 |
| Logical violations after mask | 0 | 0 |

The two passing safety diagnostics do not override the four failed learning gates. Offline hybrid
replay correctly rejected the model and created no output. Five-minute read-only Shadow, the
20-action probe, one-minute run, and five-minute run were not started. T8-v3 device input remains
zero.

## T8-v4 zero-label diagnostic closure

The optimized T8-v4 protocol is frozen in
[`docs/T8_V4_PROTOCOL.md`](docs/T8_V4_PROTOCOL.md). It learns only
`main_view_enemy_cue_visible`, `basic_attack_button_visual_enabled`,
`skill1_button_visual_ready`, and `skill2_button_visual_ready` under one fixed layout and action
schema. `attack_opportunity`, `target_attackable`, `safe_to_attack`, and skill3 are outside the
first contract. Candidate basic attack, skill1, and skill2 outputs are deterministic offline logs
only; they do not mean an action is safe, valid, in range, or guaranteed.

The first cycle uses two independent automatic teachers over the frozen 103 video-train and 23
video-dev sessions. Only confident, mutually consistent, perturbation-stable outputs enter masked
diagnostic loss; all other rows remain `uncertain`. No human labels or annotation interface are
used. Seed 0 compares class prior, time-only,
last-frame linear, pooled MLP, and the existing causal TCN, then applies gameplay/HUD masks and
swaps plus temporal controls. Machine contracts and command implementations exist locally. The
PixelArena source teacher passed its frozen synthetic-dev gate on all four heads. The first 103/23
real-video consensus pass found an incorrect full-frame coordinate transform and failed accepted
class coverage. The one allowed repair normalized the detected content box without changing the
0.80 confidence threshold, model, split, or label rule. The repaired audit passed: every train/dev
head had both accepted classes, minimum accepted coverage was 0.2098, and accepted perturbation
stability was 1.0.

The single seed-0 diagnostic then found weak-target RGB and temporal signal. Causal-TCN mean dev
macro-F1 was 0.6442 versus 0.4440 for time-only and 0.4708 for label shuffle; it exceeded Pool-MLP
by 0.0569. Spatial selectivity nevertheless failed: the minimum relevant-region confidence drop
was below zero and the maximum irrelevant-region drop was 0.6550. The final decision is therefore
`spatial_selectivity_demonstrated=false`, `semantic_accuracy_verified=false`, and
`promotion_allowed=false`. This is weak-supervision evidence, not real-video semantic accuracy.

T8-v4 remains `control_output=false`. Offline replay, Shadow, and device input are blocked until
their earlier gates pass under separately frozen evidence.

## T8-v5 ROI-isolation closure

T8-v5 reused the frozen T8-v4 repair-1 weak targets and produced separate correct-ROI and
wrong-ROI ResNet-18 features without storing RGB. The single seed-0 comparison used class prior,
time-only, correct-ROI linear, wrong-ROI linear, and label-shuffle baselines. Skill2 was
diagnostic-only because the frozen dev split contains four accepted negative examples.

Basic attack passed all three formal margins: correct-ROI macro-F1 was 0.9554 and its gains over
time-only, wrong ROI, and shuffle were 0.5040, 0.1531, and 0.6618. Enemy cue reached 0.7384
correct-ROI macro-F1 but its wrong-ROI margin was only 0.1022. Skill1 reached 0.8224 but its
wrong-ROI margin was only 0.1213. Both are below the frozen 0.15 requirement, so the combined gate
failed. T8-v5 is frozen without a TCN value test, semantic-accuracy claim, replay, Shadow, capture,
or device input.

## Operation Policy v1 closure

Operation Policy v1 implemented the offline contract, inverse-dynamics, consensus-video, and
16-frame causal-policy command surfaces in
[`docs/OPERATION_POLICY_V1_PROTOCOL.md`](docs/OPERATION_POLICY_V1_PROTOCOL.md). The first pooled
512-feature IDM run failed. One implementation repair preserved that report, changed source pairs
from delayed capture time to the frozen 5 Hz scheduled clock, and exposed the same frozen
ResNet-18 encoder's 4x4 spatial map without lowering any admission threshold.

The repaired seed-0 run still failed. Movement dev macro-F1 was `0.2472` at 200 ms and `0.2059` at
500 ms versus the required `0.70`; several direction recalls remained zero. Normal-minus-shuffle
movement macro-F1 was `0.1731` and `0.1318`, so the 500 ms control also missed the required `0.15`.
Combat macro-F1 was `0.1714` and `0.2124` versus the required `0.55`, with skill recall near zero.

The gate stopped before video pseudolabel materialization. No video-test shard, policy training,
Shadow, capture, or device input was opened. Both reports remain under
`HOK_LARGE_ROOT/runs/operation-policy-v1/`; frozen T8 evidence was not changed.

Operation Direct Policy v1 then used the existing execution events directly, without video action
inference or a phone connection. Pool-MLP was selected over the causal TCN. Dev movement macro-F1
was `0.1618`, combat macro-F1 was `0.1913`, and only one of eleven movement transitions was
correct. These schedules verify the actuator but were not chosen from gameplay state, so they do
not supervise tactical action selection. This route is frozen before Shadow or input.

## Verification baseline

The release gate is:

```bash
make check
make accept
make accept-v2
make pixel-smoke
make shadow-live-smoke
make alignment-smoke
make temporal-smoke
make rich-smoke
git diff --check
```

The repository safety check also requires exactly four root Markdown authority files, no dependency
boundary findings, and no checked-in large-data or mobile-private artifacts.

## Current limitation

Hierarchical Policy v0 E0 is implemented and E1a has a non-promoting health/death diagnostic, but
terminal outcome coverage and numeric HP accuracy remain unverified. The project has the
FrameBus/Event/Transition data plane, but not a trained three-head PolicyBundle, RewardHub, or
EventEngine-backed replay. Earlier Global Agent, Human IfO, T8, and operation-policy
results remain evidence and reusable components, not reopened parallel routes.

## Hierarchical Policy v0 E0 closure

- `FramePacket` carries immutable in-memory RGB views and persists only anonymous references and hashes.
- `LatestFrameBus` keeps one newest frame and cannot accumulate a capture backlog.
- Visual events are version-bound and deduplicated once per episode; E0 contains no detector.
- Proposals preserve source/applied observation IDs, carried-forward state, freshness, and Bundle version.
- `UnifiedTransitionStore` uses SQLite transactions, retains invalid rows as non-training evidence,
  checks episode continuity, and stores terminal transitions before the caller exits.
- Focused E0 tests: 16 passed. Full repository: 329 passed, strict mypy 40 modules, safety check passed.
- No dataset, GPU, phone capture, input command, model inference, or gradient update was used.

## Hierarchical Policy v0 E1a health diagnostic

- The initial maximum health-bar width of 16 pixels failed with 7 challenge false deaths; its
  report is preserved under `HOK_LARGE_ROOT/audit/hierarchical-event-e1/health-engineering-v1/`.
- The one allowed width repair changed only 16→24 pixels. It kept all temporal and gate thresholds.
- Repaired metrics: train visibility 1.0, train false deaths 0, dev deaths 1, dev respawns 1,
  challenge false deaths 0.
- Repaired report SHA-256: `454f28198d6f388f3975eadd3770d256920a967d02e921e91bafb9dcfc2d4a90`.
- The report is path-free and self-verifying. It used no test split, phone capture, input, or GPU.
- `semantic_accuracy_verified=false`, `self_hp_numeric_accuracy_verified=false`,
  `reward_allowed=false`, and `promotion_allowed=false` remain mandatory.
- Focused E1a tests: 3 passed. Full repository: 332 passed, strict mypy 41 modules, safety passed.

## Hierarchical Policy v0 E1b terminal OCR closure

- The frozen selection used 8 train and 4 dev sessions; video-test frames remained unopened.
- Candidate mining used the final 24 seconds at 1 Hz, but time never became a label.
- Train GAME_END coverage was 0.125 and dev coverage was 0.5; WIN/LOSS coverage was 0 on both.
- Outcome conflicts were 0, so the failure is missing evidence rather than contradictory OCR.
- Report SHA-256: `dfbae056925e179f93cd475d081ac4731aba86d3c37dbf422e0ba95efa363b4f`.
- No arbitrary OCR text, raw frame, source path, phone capture, input, GPU, or test frame was stored.
- E1b is frozen failed. Confidence, tail duration, and sampling frequency are not retuned.
- Focused E1b tests: 3 passed. Full repository: 335 passed, strict mypy 42 modules, safety passed.

## Hierarchical Policy v0 E1c result-page anchor preflight

- All 103 train and 23 dev final decoded frames were checked; video-test frames were not decoded.
- Anchors require allowlisted result-page OCR. Final-frame position itself is not a label.
- The preflight found 34 train anchors and 7 dev anchors with zero outcome conflicts, passing the
  frozen minimum support of 10/4.
- Anchor frames are inventory evidence only and are forbidden from future dynamic-model input.
- Report SHA-256: `68bfee35e70e9de656f783d802daae708bea53a9dfb5025ff7fbf9ebe7aaf27e`.
- Dynamic terminal accuracy, WIN/LOSS truth, Reward, promotion, phone input, and online learning
  remain false.
- Focused E1c-anchor tests: 2 passed. Full repository: 337 passed, strict mypy 43 modules, safety passed.

## Hierarchical Policy v0 E1c dynamic clip materialization

- Each anchored session contributes a far negative, near negative, and terminal-transition candidate.
- Train/dev contain 34/7 complete triplets; no session was incomplete and no session crossed splits.
- Anchor-frame overlap is zero. Model input contains only 16-frame RGB sequences.
- Train/dev shards contain 102/21 clips and occupy about 37 MB in total.
- Report SHA-256: `46094be70f37eb1514f2bd9405e8ae14082b28f92b245caefbe340155582ac0f`.
- Semantic accuracy, WIN/LOSS, Reward, promotion, phone input, and online learning remain false.
- Focused E1c-clip tests: 3 passed. Full repository: 340 passed, strict mypy 44 modules, safety passed.

## Hierarchical Policy v0 E1c time-confound closure

- The mandatory preflight used only within-session materialization ordinal, not RGB or target labels.
- Train and dev accuracy/macro-F1 were all 1.0, above the frozen 0.7 maximum.
- No visual model, temporal model, last-frame control, shuffle control, or GPU run was started.
- Report SHA-256: `a7b2675400928eb6ca7a0854ccae2291347c46290d469e91b7bd4e28370aee0e`.
- Fixed-offset clips remain diagnostic-only and cannot train EventEngine or generate Reward.
- Focused E1c-probe tests: 3 passed. Full repository: 343 passed, strict mypy 45 modules, safety passed.

## Hierarchical Policy v0 E1d one-shot test closure

- Checkpoint bundle SHA-256: `55a679883119cdbf1a6a7703f945d61ce33408bad84013362e66355e83345c79`.
- The immutable test contract was committed before any test frame was decoded.
- The test opened at least one test session, then stopped on `TEST_SESSION_NO_PRE_RESULT_SEQUENCE`.
- Exact opened-session count is unavailable because the failure occurred before report finalization.
- No training, threshold tuning, repeat test, or EventEngine integration occurred.
- Failure report SHA-256: `36f3f69ff65158bda3788297975a8c394a992a8b747babee70d21d489fe9991c`.
- `rerun_allowed=false`, `integration_allowed=false`, and `reward_allowed=false` are frozen.

## Hierarchical Policy v0 E1e unused-session diagnostic

- 85 unused unanchored train/dev sessions were processed; test frames were not reopened.
- Eligible pairs were 6 train and 1 dev versus the frozen 10/3 minimum.
- Ineligible counts: no pre-result sequence 63, no visual consensus 14, no matched pair 1.
- Report SHA-256: `1328583fa99329887a6fa5722b9be06fb6575ab09f9add8ca29cd276629d1fea`.
- This diagnostic cannot replace formal test or reopen EventEngine integration.

## Hierarchical Policy v0 P0 existing-adapter value gate

- Exact source hash `9e0965…` and selected adapter hash `05c948…` were compared with random ResNet-18.
- Adapter, source, and random dev macro-F1 were all 1.0; adapter margins were zero.
- Adapter features did not collapse, but the task was too easy to demonstrate representation value.
- Report SHA-256: `d0f88a86fd8adb3131c389dab5210ea33c746503b97f87368a74e07573198c9e`.
- `p0_initialization_allowed=false`; no test, Reward, phone input, or online learning was used.

## Hierarchical Policy v0 P0 temporal-order gate

- Chronological and middle-shuffled pairs share identical frames and identical first/last frames.
- Materialization passed with 128 train and 32 dev pairs; ordinal accuracy was 0.469/0.563.
- Adapter/source/random temporal macro-F1 was 0.333/0.469/0.514; adapter last-frame was 0.333.
- Report SHA-256: `331e1fc763416dbc4c1e08520f0a597e20b45be68141d2beeadfccb617fdcf0e`.
- The old adapter is rejected; no test, Reward, phone input, or online learning was used.

## Hierarchical Policy v0 P0 temporal SSL pilot

- The frozen seed-0 ResNet-18 plus GRU pilot used only the 128/32 temporal-order train/dev pairs.
- Its only repair set deterministic CuBLAS workspace state before the first training update; model,
  data, and gates were unchanged.
- Overfit32 reached 1.0 accuracy; dev macro-F1 was 0.7031 versus the frozen 0.75 requirement.
- Baseline margin and all non-collapse checks passed, but the dev gate failed, so no encoder was saved.
- Embedded report SHA-256: `f66fe1c991dc74b3bc81792cd25ddf6a1dd93ae8a758bbf1245d418b8ba169b9`.
- The lineage is frozen without retuning or rerun; test, PolicyBundle initialization, Reward, capture,
  input, online learning, and promotion remain closed.

## Hierarchical Policy v0 P0 temporal SSL v2

- A train-only index covers all 103 video-train sessions with 1,648 pairs, zero invalid windows,
  and zero duplicates; video-dev and video-test were not opened during materialization.
- The first training launch stopped before any gradient update because NHWC input had not been
  converted to NCHW. The tensor-layout fix added a non-symmetric regression test and changed no
  data, model, epoch, or gate.
- Fixed-last-epoch seed-0 training passed overfit32 at 0.9688 and reached frozen dev macro-F1
  0.7907, 0.2765 above the best frozen baseline. All non-collapse checks passed.
- Index/report SHA-256: `a4f96fb39aa502ad987735ebf513d02e0fc2c420f9659e3e294aa367c21b814d` /
  `543c811d28cc427c3ff790d3493a710c5f1d5f2703d7c48e24240f1fdae00166`.
- Representation/report SHA-256: `d1ce0a9c44710586e6df3124371ffa0171dc1806510efe2d9d02d5e733d1c848` /
  `f39f7d3e84f34944016db935ff13771afe86787231cc60f7ffd6e8c6532dbcb0`.
- The representation may initialize P0 only. P1 Head training, Reward, test, capture, input, online
  learning, and whole-policy promotion remain closed.

## Hierarchical Policy v0 P1 Movement teacher audit

- The frozen 103/23/23 video split was mapped back to all 149 original MP4 files without persisting
  source paths. Only train/dev were decoded; video-test remained unopened.
- Each train/dev session was sampled at three fixed fractions with 64 frames each. Stable labels
  required three consecutive teacher decisions and mean teacher recommendations, not human actions.
- The initial run mishandled MP4 Display Matrix rotation. Its failure report SHA-256 is
  `e78f464156688e4deefcbafc50c8f5b220bd73243f893f5e961272723833eb49`.
- The sole repair applied the display matrix before ROI cropping and changed no sampling, teacher,
  split, or gate. All 103 train and 23 dev sessions then yielded their full 192 sampled frames.
- Detection coverage was only 0.1844/0.1898; detected-session fraction was 0.5922/0.6087, with train
  below 0.6. Dev stable west support was 4 versus 16 required. Direction-session support passed.
- Repair report SHA-256: `712b122096fec4972e10a579b2daed453bdb4383ab052df4e8f5780695c6ce5e`.
- The lineage is frozen failed. No Movement Head training, PolicyBundle assembly, Reward, capture,
  input, online learning, or promotion is authorized.

## Hierarchical Policy v0 P1 Macro data audit

- The source is the project-owned PixelArena structured rule teacher, not the failed real-video
  Auto Semantic Teacher and not human annotation.
- After Router-owned DISENGAGE/RECALL rows were excluded, FARM_LANE, PUSH_STRUCTURE, and ENGAGE
  retained 1,620 train and 412 dev causal windows.
- Each intent appears in every 40 train and 10 dev episode; minimum per-intent support is 486/105.
- Class-prior and time-only dev macro-F1 are 0.2019 and 0.3893, both below frozen ceilings.
- Report SHA-256: `de7f8c4b265e69694d01210cbec49948907c389041df4792c294a90994bcea7c`.
- This opens one frozen-P0 simulator-only Macro Head learnability run. Real-video semantics,
  Movement, Combat, PolicyBundle assembly, Reward, test, capture, input, and promotion remain closed.

## Hierarchical Policy v0 P1 Macro Head

- The P0 ResNet-18 plus GRU was frozen. Only a roughly 100K-parameter three-class MLP Head trained
  on a fixed main/minimap/HUD policy canvas for 50 fixed epochs without dev selection.
- Overfit32 accuracy was 0.9688, but loss 0.1717 missed the frozen 0.05 ceiling.
- Dev macro-F1 was 0.3868 versus time-only 0.3893 and label-shuffle 0.3023. ENGAGE/FARM recall was
  only 0.2571/0.2291.
- Report SHA-256: `bd781082c4811dd9985f5be694ead7d851dca560979d141cebe38e87a2e94d8f`.
- No Macro Head checkpoint was saved. The canvas, Head, epochs, loss, and gates are not retuned.
- P0 keeps its temporal-order evidence, but frozen P0 is not proven to expose simulator Macro
  semantics. PolicyBundle assembly, Reward, test, capture, input, online learning, and promotion
  remain closed.

## Hierarchical Policy v0 P1 Combat data audit

- Eight five-minute visual-combat sessions passed shard and split verification. Each session has
  1,485 rows and the same class counts: basic 30, skill1/2 10 each, and skill3 8.
- Positive numerical support passed, but a 200 ms scheduled-clock lookup reached dev macro-F1
  0.9331 versus the frozen 0.5 ceiling.
- The eight sessions exposed only two unique positive action sequences versus four required, and
  the artifacts do not mechanically bind the declared Houyi identity.
- Report SHA-256: `f3709da7d6d5df05e6a44fec34976f092e1eef01f37bc82cf817a9912e54be3e`.
- These rows validate cooldown round-robin execution only, not tactical choice. No Combat Head was
  trained; the visual cooldown arbiter remains deterministic.
- All three P1 Head routes are now blocked. No PolicyBundle assembly, Reward, test, capture, input,
  online learning, or promotion is authorized.

## Hierarchical Policy v0 P1v2 architecture decision

- The failed fully frozen transfer assumption is not reopened. P0 through ResNet layer2 becomes a
  shared frozen trunk only.
- Macro and Movement each own separate trainable layer3/layer4, GRU128, and Head branches with an
  8–14M parameter budget. They train independently and propose at 2 Hz and 10 Hz.
- Movement outputs eight directions to persistent-joystick execution. Combat v0 stays the 10 Hz
  deterministic visual cooldown arbiter until hero-bound, non-clock tactical data exists.
- Models never write input. The deterministic Router owns observation-version checks, concurrent
  movement/combat pointers, death, and hard stop.
- The decision is architecture-only. Branch training, PolicyBundle assembly, Reward, online RL,
  capture, input, and promotion remain closed.

## Hierarchical Policy v0 P1v2 Movement simulator audit

- Only executed `move` actions were accepted. Skill aim and wait were excluded from Movement labels.
- Global Agent train/dev contained 602/160 move rows but only east/west: 376/226 and 115/45.
  All other six directions had zero samples and zero episode support.
- Dominant-direction fractions were 0.6246/0.7188 versus a frozen 0.6 ceiling.
- Report SHA-256: `2ace0d94c5acb53b99f35b52d2d30f88945bae6d2cdecd2f5d97507063491ec1`.
- A read-only diagnostic of historical V7 fit/acquisition rows found only ego-view east moves, so
  that frozen dataset cannot repair the coverage gap.
- No Movement branch was trained. The next source must be a separately versioned, balanced,
  visible-target 2D PixelArena curriculum; the old action space and labels are not changed.

## Hierarchical Policy v0 P1v2 balanced Movement source

- The new source contains 512 train and 128 dev 16-frame RGB sequences with alternating blue/red
  ego views. Each of eight directions has exact 64/16 group support.
- Train/dev group overlap and direction imbalance are both zero. Structured state, side, group ID,
  and labels are excluded from model input.
- Report SHA-256: `f73e379dfe79ab24a7087e260e6d321f7c0e5f672f6ae2b6cbae4edea946880b`.
- This opens one task-specific Movement branch learnability run for local visible-target approach
  only. Lane strategy, real-video semantics, PolicyBundle assembly, Reward, online RL, capture,
  input, and promotion remain closed.

## Hierarchical Policy v0 P1v2 Movement branch

- P0 through layer2 was frozen; layer3/layer4, GRU128, and the eight-direction MLP Head exposed
  10,841,096 trainable parameters.
- The initial overfit harness formed consecutive single-class mini-batches and failed at 0.4062.
  Failure report SHA-256: `4a068f6ebd9031bd142b8a07ca1b523989a27f6d37e742eed148642d4370dbed`.
- The sole repair deterministically shuffled overfit32 batches without changing model, data,
  optimizer, epochs, or gates.
- Full-data dev macro-F1 and every direction recall were 1.0 versus label-shuffle 0.1748, but the
  repaired overfit32 reached only 0.9375 accuracy and 0.1704 loss versus 0.95/0.05 requirements.
- Repair report SHA-256: `ac6490178f02eec97e1dbe75b4a6875c4bd7bfc8085c36c36de2ba9d9c7bf55c`.
- No checkpoint was saved. A future normalization-stable candidate must be separately versioned;
  PolicyBundle, lane strategy, real-video semantics, Reward, online RL, capture, input, and
  promotion remain closed.

## Engineering convergence execution state

The active task is at [Current execution state](#current-execution-state).
The entries below record completed experiments and their original next-step decisions; they do not
schedule new work. The frozen package status remains `R1_HIERARCHICAL_RULE_OFFLINE_COMPLETE`.

### Hierarchical simulator rule delivery (2026-09-09)

- Added `movement-mvp --mode package-hierarchical-rule` with an independent read-only
  `--verify-only` path. It nests the verified R1 engineering package and binds exactly six frozen
  Movement reports: deterministic persistence, change-only ownership, the initial neural pilot,
  both rejected neural replays/pilots, and the passing geometry replay.
- Delivery grade is `R1_HIERARCHICAL_RULE_OFFLINE`. The selected simulator chain is Macro exact
  goal geometry change, deterministic previous-direction persistence and deterministic Router
  STOP. It contains no promoted checkpoint and does not grant real-RGB, Reward, mobile or RL use.
- Real package passes352/352 dataset decisions and10/10 replay episodes with40 geometry changes,
  40 KEEP,40 STOP and120 arena steps. Model runs and input commands are both zero.
- Package:
  `$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/r1-hierarchical-rule-v1`.
  It contains403 manifest-listed payload entries (406 total files) and32,730,021 bytes. Manifest file SHA-256
  `563774c64f09058f09c481ebb5794ac2080dfdb33775908daa566e764cad152d`;
  summary file SHA-256
  `6c2a7b70272b0313fcc6b5136c1e6399e95c44a6a9046e3fb19532359706def6`.
- A new process completed read-only verification with the same result. This closes the simulator
  Movement cycle; future work must begin from separately observable real player and Macro goal
  evidence rather than another model/data retry on these synthetic markers.
- Delivery validation passed Ruff, strict mypy on70 source files, all492 pytest tests in108.64s,
  project safety on271 files/131 Python files/71,842 lines, and `git diff --check`. The first plain
  `make check` used the ambient Conda `python3` without mypy and stopped before pytest; rerunning the
  identical target with the shared project environment passed completely (version correction above).

### Deterministic geometry change replay (2026-09-09)

- Added a zero-parameter geometry replay bound to the failed v2 model, position-v2 dataset and
  exact original replay routes. It loads no checkpoint and changes no target/route geometry.
- On the v2 dataset, geometry direction is train256/256 and dev96/96 with all recalls1.0.
  This follows exact synthetic player/goal marker geometry and is not a real-RGB claim.
- Original multi-waypoint replay passes10/10 episodes:40 geometry changes,40 executor KEEP,
  40 Router STOP and120 arena steps. All eight directions occur; every waypoint is reached.
- Status `SIMULATOR_GEOMETRY_CHANGE_REPLAY_PASSED`; report file/self
  `191ae4065d1dd78fed28c8329c311f4875daf39eb6abcae503bf79f6a3c622a5` /
  `3ba0df0a1a2ed2ec44dfc81a3cb865d60f5010a1c27ec98ba40deb65a64783cf`.
  Model runs0, GPU0, input0, video-dev/test0.
- Simulator hierarchy is now selected as Macro goal geometry change + deterministic direction
  persistence + Router STOP. All learned continuation/change checkpoints stay rejected.
- Next: package this offline simulator rule evidence. Real use remains blocked on reliable player
  localization and observable Macro goal rendering; no further Movement training without that source.

### Position-held-out change pilot v2 (2026-09-09)

- Ran the sole fresh last-frame attempt with the same seed0, AdamW1e-3, batch32,30 epochs and
  0.95 accuracy/F1 plus0.90 recall gates. The old checkpoint was not loaded; GRU was not repeated.
- Best epoch5 has train accuracy1.0 but position-held-out dev accuracy0.20833, macro-F10.19872,
  loss5.76653. N/S/NW/SE recalls are0; W/E are0.333 and NE/SW0.5. Every gate except train and
  first-update fails. Dev result is unchanged through epoch30 while loss increases to7.735.
- The explicit E/W row repair did not solve x-position generalization. Evidence supports an
  absolute-position shortcut in this CNN, not a learned player-to-goal relation. The attempt is consumed.
- Status `CHANGE_POSITION_V2_PILOT_FAILED_ABSOLUTE_POSITION_SHORTCUT`; report file/self
  `d889134b37cb240c577c71da3d211901f4c2b8ca65cfafa68676dff2b6e485a5` /
  `1132e9bc903dee9da10b48efc5f9c5fdb7896ca3b21af9bbcda51cef6f5c461e`;
  rejected checkpoint `22c771f08611e91753e121896a273e5326b8f8e582da973bed476ba51e2b8fc6`.
  Runtime12.285s, peak280,201,216 GPU bytes. No replay/input/video-dev/test.
- No same-data retry, model growth or relational architecture tuning. Next audit uses the existing
  zero-parameter goal-canvas geometry rule on v2 data and unchanged routes. If exact, simulator
  change control remains deterministic; real use still needs a valid visual player/goal source.

### Position-held-out change dataset v2 (2026-09-09)

- Added the sole data correction allowed by the failed replay. Dataset size,64/24 episode groups,
  256/96 samples, eight-class balance, marker,15-old/1-new goal sequence and no-STOP boundary stay fixed.
- E/W current row now cycles2/3/4 in both splits. Train current x is6/7/8; dev current x is5/9.
  Current-position sets have zero overlap. The replay routes exercise both train x7 and held-out x5/9.
- All352 clips remain unique and every final RGB frame resolves to its balanced direction label.
  Counterfactual targets, no previous-action input, no arrow and simulator-only boundaries remain.
- Output `$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/movement-change-events-position-v2`.
  Dataset SHA-256 `0fcfa4c20f9911b00277ca899a0f282a4d10b2de1e396662c1d7f401d8f46613`;
  manifest file/self `43e0fe195764b1fb73d1ea095585ce3b675d864a77aa4298037ec932e5f5c1bb` /
  `8bf52c58f2cf97591e37746b740f1c48f5bfc244a600d4c55bd829293dfc06e4`.
  Compressed dataset56.95MiB; independent verify-only passed.
- Status `CHANGE_EVENT_POSITION_V2_DATASET_VALIDATED`. Next is one fresh last-frame-only model
  using the same seed/optimizer/30 epochs/gates. GRU is not rerun because failure was spatial.
- No model/GPU/input/real RGB/video-dev/test. Focused tests and checks required before commit.

### Multi-waypoint change replay failure (2026-09-08)

- Added a hash-bound 10-episode replay with three four-waypoint routes. Each waypoint contract is
  model change, executor KEEP, then Router STOP. It loads only the selected last-frame checkpoint.
- The first implementation applied a wrong prediction before completing route validation and raised
  `illegal blue action` without a report. The sole execution-order repair moved all40 model calls
  into a fail-closed preflight. Checkpoint, route geometry and thresholds did not change.
- Formal preflight is20/40 and executes zero arena steps. N/S/NW/NE/SW/SE are20/20 combined;
  W is0/10 and E0/10. E/W at row2 or4 are predicted as N/S/NE. Thus no KEEP/STOP episode runs.
- Root cause is dataset construction: E/W training and simulator dev examples always place the
  player on row3, while the new routes request E/W on rows2/4. Prior dev1.0 therefore demonstrates
  within-generator fit, not player-position generalization.
- Status `CHANGE_REPLAY_FAILED_POSITION_SHORTCUT`; report file/self
  `43fa89e115830e79a87110c34f2a66f0a154b39ac714c0212d36a899ca1defbf` /
  `59db9569ecc8dd1e01bc77151b789292cc9e99b9e3dada2ca91f35b09412d924`.
  Same checkpoint/route retry and simulator integration are closed. No device input/video-dev/test.
- Next: one simulator-data correction only—cover E/W at rows2/3/4 and create a position-held-out
  dev split. Materialize and validate before any new model run; other direction geometry stays fixed.

### Simulator change-policy comparison (2026-09-08)

- Added one fixed fresh-init comparison on the balanced simulator dataset. Both use the same
  convolutional encoder/projector and8-output head. Temporal variant adds a GRU (686,152 params);
  control consumes only the current final frame (587,080 params). No previous checkpoint loaded.
- Both pass all gates with train/dev accuracy1.0, dev macro-F11.0 and every recall1.0.
  GRU best epoch10/loss0.04837/runtime13.714s; last-frame best epoch5/loss0.05655/runtime11.905s.
- The predeclared0.01 tolerance selects last-frame because accuracy/F1/recall tie and it removes
  99,072 parameters. Selected checkpoint SHA-256
  `e7cd269756aa691401ed54616161a65b9dab6fe071b23310480d69863a93cf0d`.
  GRU checkpoint is retained as unselected evidence only.
- Status `SIMULATOR_CHANGE_POLICY_PASSED_LAST_FRAME_SELECTED`. Report file/self
  `462080a6d094c176c36589c2e5a89161917cedf24a548d6a4989abf67f6ebaa0` /
  `f33d0f66ff9a1ee8b11b9f68724c07991e2d7853a99f8969a5e8d4df01516042`.
  Peak GPU allocation986,850,816 bytes; no phone/video-dev/test.
- This permits simulator integration only. It does not establish real goal rendering, real RGB
  behavior or gameplay quality. Router still owns STOP; executor owns persistence.
- Next: load only the selected checkpoint in an offline multi-waypoint replay, call it on Macro
  goal changes and verify deterministic KEEP/STOP between events.

### Simulator change-event dataset (2026-09-08)

- Added `change-materialize` under the change-only contract. Dataset has64 train and24 dev episode
  groups, four goal-version changes each: train256/dev96. Every direction has32/12 samples.
- Each 16-frame synthetic minimap clip uses fifteen old-goal frames and one current new-goal frame.
  The label is derived from player-to-new-goal geometry. Previous action and direction arrows are
  absent; STOP is absent. Current goal frame is included because Macro updates precede the decision.
- All352 clips are unique, train/dev episode overlap0. Every old frame resolves to one different
  direction and every final frame resolves to its stored label. Counterfactual new goals are checked
  with the same last-frame render seed so target geometry, not noise, changes the label.
- Output `$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/movement-change-events-v1`.
  Dataset SHA-256 `1fd0d5a10d64adc99166f22babc1e566c1febe02c758e0007011c4278dfcb490`;
  manifest file/self `956e25691389a0d4b3d5c256973a0ed1f643f58be126ff8da8369e4ee2e8ede6` /
  `caccd9bc122d4a3a594e1b205f139ef7859835328a9a96c4322c6e4c876c49de`.
  Compressed dataset56.95MiB; independent verify-only passed.
- Status `CHANGE_EVENT_DATASET_VALIDATED`. Next pilot may compare fresh 8-head GroupNorm+GRU with
  a last-frame control. If they match, select the simpler model. No historical weights or real-RGB claim.
- No GPU/model/phone/video-dev/test. Focused tests, Ruff, strict mypy, safety and diff checks are
  required before commit.

### Change-only Movement contract (2026-09-08)

- Added a hash-bound change-only contract on the existing synthetic hollow-goal-ring canvas.
  The model vocabulary is N/S/W/E/NW/NE/SW/SE; previous action is execution state, never Actor input.
- The model is invoked only for Macro goal-version change, stuck recovery or respawn reset.
  Eight opposite-direction change cases produce the expected target and executor `MOVE`.
  Eight same-direction goal-change cases invoke the policy but resolve to executor `KEEP`.
- Eight steady cases skip the model and use deterministic `KEEP`. Goal reached, unknown goal and
  terminal cases skip the model and produce Router `STOP`/executor `UP`. Model never outputs STOP.
- Status `CHANGE_ONLY_MOVEMENT_CONTRACT_PASSED`; report file SHA-256
  `d21028844ad9cd4ee2fbd5fbff5a17810d7e288e596ef0e066a67975851de528`.
  The report binds the exact persistence evidence and both Movement source files.
- No model, training, GPU, phone, video-dev or video-test. This defines responsibility only; it
  does not establish learned direction changes or gameplay quality.
- Next: materialize a separately versioned simulator change-event dataset using the same eight
  direction targets and observable goal ring. No failed checkpoint reuse.

### Deterministic direction persistence (2026-09-08)

- Added JSON-only `joystick-persistence-audit`. Every continuation manifest row is rebound to its
  source candidate sequence and must have identical direction at target and two preceding ticks.
  Candidate PTS, label PTS and manifest prior timestamps are checked; Actor-video PTS may differ
  by at most25ms (observed maximum23ms), with both clocks before the target.
- Baseline predicts the previous executed direction. It reaches train203/203 and dev80/80,
  accuracy1.0 and every supported class recall1.0. This is exact by continuation-target definition,
  not learned generalization or gameplay performance.
- Existing executor `_movement_command(previous,current)` returns `KEEP` for each of N/S/W/E/
  NW/NE/SW/SE when directions match. Runtime already maintains/rebuilds `previous_action`.
  Thus no additional neural model or transport is needed for smooth action persistence.
- Status `DETERMINISTIC_DIRECTION_PERSISTENCE_EXACT`; report file/self
  `8ff2a3e2c54be71ac8f3e8c19c5211e69ee62572ff135969bb4aefbcbcee7fc3` /
  `7afffddf73a6ff9439a9e506c55c2d6f6198960ea397cfb82b84ea1e831e1a63`.
  No RGB decode/model/GPU/input/video-dev/test.
- Learned continuation is closed and its checkpoint remains rejected. Persistence stays in the
  deterministic executor; STOP stays in Router. Future Movement learning must address a new
  direction conditioned on an observable Macro goal, which current real-video labels do not supply.

### Joystick continuation pilot result (2026-09-08)

- Added one fresh-init eight-output task-specific GroupNorm+GRU. Spatial encoder/GRU dimensions,
  seed0, AdamW1e-3, batch8, balanced sampling,30 epochs and selection rule are unchanged; only
  the nine-output head becomes eight outputs (686,152 parameters). No previous checkpoint loaded.
- Best epoch30 memorizes train at accuracy1.0. Internal-dev accuracy0.25, macro-F10.18971,
  loss3.11219; N/S/NE/SW recalls0.429/0.294/0.190/1.0, while W/E/NW/SE are zero.
  Majority-N accuracy0.2625/macro-F10.05198. Source accuracy ranges0–0.6.
- Only train-accuracy and first-update gates pass. Dev accuracy/F1, six nonzero recalls, majority
  gain and every-source accuracy fail. Across checkpoints dev accuracy remains0.175–0.25 and
  macro-F10.107–0.190 while train reaches1.0. Same-data retry and tuning are closed.
- Status `JOYSTICK_CONTINUATION_PILOT_FAILED_NO_GENERALIZATION`. Report file/self
  `ca0e95c0f56289971b72c92fc273efba2140bfc3d4ae1dfebcb7f532081302da` /
  `202ca356e029e90bc27d24e9655f838208952b720f3b1b8ac23100499e7f97d7`;
  diagnostic checkpoint `c87a7ab9220342eed79d94823e3580b13a8c0df22242b868e30fa3d7958650ee`.
  Runtime10.123s on RTX4090, peak312,570,880 bytes. No input or video-dev/test.
- By target definition, previous executed direction equals every continuation target. Next audit
  compares a deterministic previous-direction persistence baseline using manifest state only.
  If exact, learned continuation is redundant and learning should return to observable goal/change intent.

### Joystick continuation dataset (2026-09-08)

- Added `joystick-materialize-continuation`. It binds the continuation audit and all five source
  reports, then selects every third-or-later equal direction candidate. STOP is absent from both
  arrays and manifest. Source paths and original-resolution RGB are not persisted.
- Fixed split produces train203 from16 sources and dev80 from5. Counts train
  N87/S40/W9/E18/NW14/NE15/SW7/SE13; dev N21/S17/W1/E8/NW5/NE21/SW2/SE5.
  Both contain all eight directions; W/SW remain rare and must be reported individually.
- Each sample contains16 masked 128×128 RGB frames ending before the target. Input-label gaps are
  train81–100ms and dev82.756–99.933ms. Source overlap0; all283 clip hashes unique; joystick region zero.
- Output `$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/joystick-continuation-v1`.
  Dataset SHA-256 `a6b4d566f5ae873e4d74fb35fcccb99e114941d934cbabd37aa0d4422b26f85a`;
  manifest file/self `797f77c8b54cef49119a631e07e0ac59f487ea7a4b2a8c064147cd8c5b36a75e` /
  `1c43637833d8163199add19afdfee426cec9141fb37e3891b89e5023669b10d4`.
  Compressed dataset153.4MiB; independent verify-only passed.
- Status `JOYSTICK_CONTINUATION_DATASET_VALIDATED`. One fresh-init eight-class pilot may reuse
  the existing spatial encoder/GRU architecture with a new eight-output head and balanced sampling.
  No checkpoint reuse. Router retains STOP. Capability is continuation only.
- No GPU/model/input/video-dev/test. Focused tests, Ruff, strict mypy, project safety and diff checks
  are required before commit.

### Joystick continuation target audit (2026-09-08)

- Added JSON-only `joystick-continuation-audit`, binding all frozen candidate reports and the failed
  scale21 pilot. It separates run onset, one-prior-frame continuation and two-prior-frame
  continuation. No RGB decode, model, threshold or split change.
- Learned direction target is eligible only at the third or later equal candidate in one run.
  Thus two prior same-direction sampled frames precede the label and may expose established motion
  in RGB. Train has203 targets; dev80. Counts train N87/S40/W9/E18/NW14/NE15/SW7/SE13;
  dev N21/S17/W1/E8/NW5/NE21/SW2/SE5. All fixed gates pass.
- Source support train is10/9/4/7/7/5/4/3 and dev4/3/1/2/2/5/1/1 in the same action order.
  Rare W/SW/SE remain fragile but nonzero; future metrics must retain per-class recall.
- STOP is excluded from learning. Although two-prior-center frames number train104/dev10, centered
  UI may mean death, overlay, disabled control or intentional stop. Deterministic Router owns STOP.
- Status `JOYSTICK_DIRECTION_CONTINUATION_SUPPORT_PASSED`; report file/self
  `2c87d6e90c90302a4ad5c558654bf760ca9c7becf82461b4a5a9d809971794fc` /
  `f0b089f9dae2b294e41bd71ef26f2d9c52a58f8ad457ef5ad156182c6ca1beda`.
  Output about3KiB; no GPU/input/video-dev/test.
- Next: materialize the frozen203/80 indices into an eight-direction, masked causal RGB dataset.
  This learns movement continuation only, not direction-change onset or tactical intent.

### Scale21 grouped pilot result (2026-09-08)

- The single scale comparison reused the exact 686k GroupNorm+GRU, seed0, AdamW1e-3, balanced
  sampling, batch8,30 epochs, evaluation schedule and gates. Samples per epoch followed dataset
  size201. Both prior checkpoints were forbidden; initial state is fresh and hash-recorded.
- Best epoch10: train accuracy0.7214, internal-dev accuracy0.1667, macro-F10.14652, loss2.33794.
  N/S/NW/NE/SE recalls are0.231/0.125/0.25/0.222/0.333; STOP/W/E/SW are zero. Source
  accuracies are0.125/0.294/0/0/0.2. Majority-N accuracy0.2708/macro-F10.04736.
- Only nonzero-recall and first-update gates pass. Scaling from73/25 raises macro-F1 from0.0974
  and nonzero classes3→5, but does not materially improve accuracy0.16→0.1667 or generalization.
  Later epochs raise train accuracy to0.955 while dev F1 falls, confirming overfitting.
- Status `JOYSTICK_SCALE21_PILOT_FAILED_NO_GENERALIZATION`. Same-data retry, model/extractor
  tuning, formal training and checkpoint promotion are closed. The failed checkpoint is evidence only.
- Report `$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/joystick-scale21-pilot-seed0-v1`.
  Report file/self `056f299bd6e8e9816622294eeb9d3fdfcc532b28a4ec123b124a93be2105dd93` /
  `1cc211c4b31ccec6203e079f972707d2b17339c071cc740bed704b56a2d1fbeb`;
  checkpoint `3b562e2eb5c1e974d14d8f016507a1b34f031ff8eac749e2248f275c7b93e1e3`.
  CUDA runtime10.175s, peak312,573,440 bytes. No device input or video-dev/test.
- Current data selects each stable-run onset. Predicting that direction from RGB ending before
  the onset may include unobservable human intent. This is a hypothesis, not a proven cause.
  Next: JSON-only audit separating run onset from continuation frames before any new training.

### Scale21 grouped dataset (2026-09-08)

- Fixed internal dev without label selection: retain prior `0e34...`/`1221...`, then add the
  first/middle/last compatible new identities by sorted anonymous hash (`1720...`, `3927...`,
  `493c...`). The other16 sources are train. Neither video-dev nor video-test is used.
- One sample per stable direction-run onset plus each explicit release STOP produces train201 and
  dev48. Train counts STOP26/N45/S27/W14/E25/NW17/NE20/SW12/SE15; dev counts
  STOP1/N13/S8/W1/E7/NW4/NE9/SW2/SE3. Both splits contain all nine classes.
- All249 masked RGB clips are unique and causal. Input-label gaps are train81–100ms and
  dev81.3–100ms; variable gaps reflect decoded presentation timestamps. Source overlap is zero.
- Two implementation corrections stopped before decoding: the new conclusion filename differed
  from the old convention, then one copied full dev hash mismatched after the first eight chars.
  Correcting the exact filename/hash restored the predeclared split; labels and extractor unchanged.
- Output `$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/joystick-scale21-grouped-v1`.
  Dataset SHA-256 `b29d952880a36dbc9796758ed20507eb7836f25ac16514b7d3a3e14aca240399`;
  manifest file/self `39f04c197ba3bcd1a5a6d33d22f2da23f8dfdc628473f778278d42461a90f4f2` /
  `0bdeb3690f674e23eb065113528a36747bb6faa5b94de25b4e6ae3143fa627cb`.
  Compressed dataset135.1MiB; independent verify-only passed.
- Status `JOYSTICK_SCALE21_GROUPED_DATASET_VALIDATED`. One fresh-init pilot may reuse the exact
  prior model, balanced sampling, optimizer,30 epochs and gates. No failed checkpoint reuse,
  model/extractor tuning, input or test access.
- Focused tests, Ruff, strict mypy, project safety and diff checks are required before commit.

### Frozen-extractor scale24 audit (2026-09-08)

- Fixed 24 additional anonymous train identities and five four-second windows at 15/35/55/75/90%.
  The strict first run stopped because 11 are portrait 1080×2408. Its failure staging is retained.
  The sole repair used container-header geometry only: keep width>height and rotation0, exclude
  portrait sources, add zero replacements, bind all 24 geometry rows in the new run contract.
- The 13 compatible sources complete 2600 frames with 653 candidates (25.12%). Every source has
  candidates; minimum coverage3.5%. Direction support sources: N9, S10, W4, E10, NW8, NE8,
  SW7, SE5. Stable runs:29/17/9/25/13/15/10/14; release STOP19.
- Combined with the previous eight sources: 21 sources, 4160 frames, 1210 candidates (29.09%),
  27 release STOP. Direction support sources are N16, S16, W8, E14, NW13, NE15, SW11, SE8;
  stable runs58/35/15/32/21/29/14/18. This is enough for a larger grouped data build.
- Representative QA inspected four low/mid/high coverage sheets. Accepted examples generally
  align with the visible control; this remains automatic weak-label evidence, not semantic truth.
- Result `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/joystick-scale24-audit-v2-landscape-subset`.
  Report file/self `04f6462ae97de4f5c600649e55ae26ff4eb2366e3b7b99edc147fb19c3a44a84` /
  `4354ae8223ee05d487146bc12a0e2da1f8e74e10444eb73b917dc08b67a2edde`.
  Artifacts22.8MiB; no native cache/model/GPU/input/dev/test.
- Source expansion is now closed. Next: fix a 21-source group split and materialize one causal,
  masked dataset from each stable-run onset plus release STOP. No model run or extractor tuning.

### Joystick grouped pilot result (2026-09-07)

- Added a single hash-bound grouped pilot: fresh seed0 task-specific GroupNorm+GRU, class-balanced
  replacement sampling, AdamW lr1e-3, batch8, FP32, 30 epochs. Internal dev is checked at six
  fixed epochs and selected by macro-F1, loss, then earlier epoch. No overfit checkpoint loaded.
- Best epoch30 memorizes train at accuracy1.0. Internal-dev accuracy0.16, macro-F1 0.09735 and
  loss3.07256. Only N/S/NW recall are nonzero (0.286/0.25/0.333); the other six are zero.
  Source accuracy is 0.0 and0.2353. Majority-N baseline is accuracy0.28/macro-F1 0.04861.
- Only train-accuracy and first-update gates pass. Dev accuracy/F1, nonzero recalls, majority gain
  and each-source accuracy fail. Earlier checkpoints do not change the conclusion: dev accuracy
  stays0.12–0.20 and macro-F10.051–0.089 through epochs5–25.
- Status `JOYSTICK_GROUPED_PILOT_FAILED_NO_GENERALIZATION`. It does not isolate data volume,
  automatic-label noise, source appearance shift or stochastic human action as the cause.
  Same-data retry, architecture tuning, formal training and checkpoint promotion are closed.
- Report: `$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/joystick-grouped-pilot-seed0-v1`.
  Report file/self `2d9d917df08255cad30d90a8d3cb9e141610289724b4c5a8615e3c333669282b` /
  `f39403b7efd4cdbcf4b7712437d9c2aec9a61d6ba344daf16725ca4a8a982102`;
  diagnostic checkpoint `4a23c54732c89ecd9cbb96df43094dc674c62ed7b8933c50e92b66d82106d1b5`.
- Runtime4.662s on RTX4090, peak312,573,440 bytes. First update loss2.22262, gradient4.51475,
  parameter change and finite checks pass. No device input or video-dev/test.
- Next work can only enlarge frozen-extractor evidence on a predeclared subset of existing train
  videos, checking coverage/support before another model contract. Do not tune the current pilot.

### Joystick grouped pilot dataset (2026-09-07)

- Added `joystick-materialize-pilot` with fixed source groups. Internal dev uses train-cohort
  identities `0e34a785...` and `12214351...`; the six others are pilot train. Neither internal-dev
  source supplied the original joystick template. Original video-dev/test remain unopened.
- One sample is taken at each stable two-frame direction-run onset plus each explicit release STOP.
  Result: train73 and internal-dev25. Both contain all nine classes. Train counts are STOP7, N22,
  S14, W5, E5, NW5, NE10, SW3, SE2; dev counts STOP1, N7, S4, W1, E2, NW3, NE4, SW1, SE2.
- Every clip is 16×128×128 RGB ending before its target. Gaps are train95–100ms/dev96–99ms;
  joystick pixels are zero. Train/dev source overlap0; all98 clip hashes unique. No source paths.
- Output `$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/joystick-grouped-pilot-v1`.
  Dataset SHA-256 `a28fe2fc49849b056b25902574523808e979d591ce826dc361c3113dee40ee0a`;
  manifest file/self `cd8d3bf74473c8e6729b08b6048ff3e3f707210968ff4602ae09b3f9f3d6ae90` /
  `80c3392bb3681f9eb0e28fa3c6b44234989ac4bea617a7b7d2c9ddb1d24591f9`.
  Compressed dataset53.4MiB; independent verify-only passed.
- Status `JOYSTICK_GROUPED_PILOT_DATASET_VALIDATED`. One fresh-init class-balanced grouped pilot
  may run next. It cannot load the overfit32 checkpoint. Because dev STOP/SW/W support is one,
  report macro-F1 and every recall; no promotion or semantic-accuracy claim.
- No GPU/model/input/video-dev/test in this step. Focused tests, Ruff, strict mypy, project safety
  and diff checks are required before the local commit.

### Joystick real-RGB overfit32 diagnostic (2026-09-07)

- Added a separate hash-bound contract and `joystick-overfit32` runner. It accepts only the
  validated joystick dataset, task-specific 686,281-parameter GroupNorm+GRU, seed0, AdamW,
  lr1e-3, batch8, FP32 and 200 updates. Attempt limit is one; no architecture fallback.
- RTX4090 run passed: eval accuracy1.0, cross-entropy0.00544488 and every class recall1.0 against
  gates0.95/0.05. First update loss2.22312, gradient norm6.75958 and parameter change all valid.
  Runtime2.273s; peak allocated GPU memory312,573,440 bytes.
- Diagnostic checkpoint metadata binds dataset and contract. Checkpoint SHA-256
  `b759921e9faf8176965d68ac9786f4fd1b8e2d04e047ac1bf3d9d1272221019f`;
  report file/self `f5bcdb36ed99c0de69cbc2f425ca9dc8f069aa4b38d2aefbc8e49a55da441542` /
  `98109db213f084127c36585e919efeaffc6a715537518d864b8a3ccffc986ed1`.
- Status `JOYSTICK_REAL_RGB_OVERFIT32_PASSED`. It proves memorization and the real-RGB training
  path only. Generalization, semantic action accuracy and gameplay performance remain unverified.
  The checkpoint cannot initialize or enter formal training; the sole attempt is consumed.
- Next: materialize one fixed six-train/two-internal-dev grouped pilot from existing frozen
  candidate windows. Any pilot model must initialize fresh at seed0. No video-dev/test access.
- Validation includes contract self-hash, fail-closed unbound-data test, checkpoint metadata/
  parameter count and the actual CUDA gate. Full focused checks are recorded below.

### Joystick causal overfit32 dataset (2026-09-07)

- Added `joystick-materialize32`. It binds the final cohort conclusion and all four source reports,
  selects eight release STOP events plus three stable two-frame runs per direction from different
  source groups, then resolves the eight anonymous train identities without persisting paths.
- Each sample decodes exactly sixteen 10Hz RGB targets before the label PTS. The lower-left 35%
  width × bottom55% height is zeroed before resize and again at 128×128. The current joystick frame
  and next-frame label confirmation are absent from Actor input. Actual last-input-to-label gaps
  are 96–99ms; all 32 clip hashes are unique.
- Dataset shape is `[32,16,128,128,3]` uint8. Class counts: STOP8; each direction3. All direction
  classes use three different source groups; the overall dataset covers eight sources. This source
  balance is diagnostic, not a formal split or generalization estimate.
- Output: `$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/joystick-overfit32-v1`.
  Dataset SHA-256 `a05279e3353cf306e3dfade5d391d2ce8792381cd0eb5a4fca9b4fe487c90155`;
  manifest file/self `73d90323922a0d87421e57ce1f3d3c50ea1b8824bf9f1c276af93410b1f368bb` /
  `540704c0bf68704e3e84e10883c0724b493eae57210cf6058de968a238493114`.
  Dataset is17.4MiB. Independent verify-only passes class, causality, mask, shape and hash checks.
- Status `JOYSTICK_OVERFIT32_DATASET_VALIDATED`. One existing task-specific GroupNorm+GRU
  fresh-init diagnostic overfit is permitted next. It may test memorization only; no formal
  training, policy promotion or semantic-accuracy claim.
- No model/GPU/input/dev/test. Source RGB is stored only in masked 128×128 diagnostic clips.
  48 focused tests, Ruff, strict mypy, project safety and diff checks pass.

### Final train-source joystick expansion (2026-09-07)

- The last bounded expansion used the three remaining predeclared train identities and fixed
  10/30/50/70/90% windows: 600 frames, no calibration/dev/test. Per-source candidate coverage is
  28.0%, 36.5%, 43.5%. Three QA sheets (60 displayed frames) were reviewed without an obvious
  gross control mismatch; this remains automatic weak-label evidence, not semantic accuracy.
- Combined immutable evidence now covers all eight selected train sources: 1560 frames, 557
  candidates (35.71%). Stable direction runs/frames are N29/108, S18/65, W6/18, E7/19,
  NW8/24, NE14/48, SW4/12, SE4/9. Supporting sources are N7, S6, W4, E4, NW5, NE7, SW4, SE3.
  There are exactly eight explicit direction-to-center release STOP events.
- `cohort-conclusion.json` binds the four input reports and freezes status
  `JOYSTICK_8_TRAIN_SOURCE_WEAK_LABEL_SUPPORT_PASSED`. Train-source expansion is closed.
  This permits only diagnostic dataset materialization from stable directions and release STOP.
- Materialization must use a 16-frame RGB Actor window ending at the preceding PTS, exclude the
  lower-left joystick region, preserve source groups/hashes, and retain automatic-label limits.
  Policy training, checkpoint promotion and semantic-accuracy claims remain closed until that
  dataset passes mechanical causal/split validation.
- Final expansion report file SHA-256:
  `496efcd3ef2394156ea81d9b8e294afbaf6880e22c08cb9ac1908b340aa26943`;
  artifacts about5.2MiB. No native cache/model/input/dev/test; GPU0.
- Validation: 46 focused tests, Ruff and strict mypy passed before execution. Final project safety,
  diff checks and evidence hashes passed. No further train-source scan is pending.

### Cross-train joystick transfer (2026-09-07)

- Added `joystick-transfer` with four predeclared additional train identities and fixed 20/50/80%
  four-second windows. The exact v3 contract, template and AST/settings fingerprint are required.
  It cannot calibrate, tune thresholds, open dev/test or write native RGB/policy samples.
- All four sources yield candidates. Per-source coverage: 14.17%, 44.17%, 53.33%, 25.00%;
  mean34.17%. Stable direction support across sources: N3, S2, W2, E1, NW2, NE3, SW1, SE1.
  Only two explicit direction-to-center release STOP events occur. Cross-source gate fails.
- Four QA sheets (48 uniformly sampled frames) were reviewed. Displayed accepted examples generally
  align with the visible floating control; unknown examples may show remote raw matches but emit no
  candidate. This is bounded visual consistency, not independent frame-level semantic accuracy.
- Result: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/joystick-cross-train-transfer-v1`,
  status `JOYSTICK_CROSS_TRAIN_TRANSFER_PARTIAL`, about4.3MiB. Report file SHA-256:
  `fdbf619fb4b67bd9fd328baa88c01ab66e2140e82d5361b2fd92cd652ddd0398`.
- No policy dataset or model. A larger existing-train scan is useful only with unchanged extractor
  and predeclared sources/times; it would improve support counts, not establish semantic accuracy.
  The deterministic Movement baseline remains deployable engineering evidence.
- Validation: 46 focused tests, Ruff and strict mypy (70 sources) passed. Source splits were checked
  before decoding; GPU/input/dev/test all zero. Native RGB remained memory-only.

### Joystick causal eligibility (2026-09-07)

- Added JSON-only `joystick-eligibility`; it reads the completed coverage report and writes no
  RGB, training sample or model. Direction evidence requires runs of at least two consecutive
  equal candidates. All eight directions pass, but support is thin: stable runs N8, S1, W1, E2,
  NW3, NE3, SW1, SE1; stable frames total54.
- STOP is accepted only as a run onset with a visible direction within the preceding 500ms.
  This rejects prolonged center/dim periods. Four release events remain versus the fixed minimum8:
  N at409ms, NW at194ms, and S at211/307ms. Thus STOP support fails.
- Future causal alignment is now explicit: the Actor RGB window must end at the preceding PTS;
  the joystick target belongs to the current PTS. Observed gaps are 96–114ms. Direction targets
  use the current and following candidate for retrospective stability confirmation; the following
  frame is label construction evidence and cannot enter Actor input.
- Final status `JOYSTICK_CAUSAL_CANDIDATES_INSUFFICIENT_STOP_AND_SESSION_SUPPORT` at
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/joystick-training-eligibility-v1`.
  No policy samples materialized. One recording cannot establish cross-session support.
- Next bounded work is transfer of the unchanged v3 extractor to additional existing train
  sources. Long centered/dim runs cannot be added as STOP. No detector tuning or training.
- Validation: 45 focused tests, Ruff, strict mypy (70 sources), safety and diff checks passed.
  The eligibility report is about2KiB; GPU/input/raw decode/dev/test all zero.

### Frozen-extractor train coverage (2026-09-07)

- Added `joystick-coverage`, using the exact v3 contract/template and an AST-plus-settings
  fingerprint. Twelve new fixed four-second windows from the same train source contain 480
  native-crop observations at nominal 10Hz. No calibration, dev/test decode or policy training.
- 177/480 candidates (36.875%); unknown303. Counts: STOP62, N35, S12, W8, E14, NW14,
  NE17, SW6, SE9. All nine classes occur. STOP spans 13 runs in 8 windows; other classes
  span 4–19 runs and 3–6 windows. Runs/windows are correlated within one original recording.
- All twelve QA sheets reviewed. STOP40 comes from dimmed f50/f70/f80/f90 windows where
  scene eligibility is unresolved; these are not verified intentional stopping decisions. The
  remaining STOP22 includes visible recenter/release sequences, still candidate evidence only.
- First scan completed but JSON serialization failed on NumPy QA indices. One Python-int
  conversion fixed it; the same scan was repeated with all twelve QA images byte-identical.
  Failed staging `.joystick-coverage-h0euv5d2` is preserved with a failure note. No parameter change.
- Result: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/joystick-train-coverage-v1`;
  final status `JOYSTICK_TRAIN_NINE_CLASS_CANDIDATE_COVERAGE`. Report file SHA-256
  `b21b41276009181d5e8a8d065aee655c60a430db25c4f238cc66acf3ce1cc891`.
  Completed output about14.3MiB; native RGB not duplicated; failed output retained separately.
- Next: separate UI-state candidates from scene-eligible action targets and resolve timing on
  existing evidence. Do not tune templates or treat single-session coverage as training readiness.
- Validation: 42 focused tests passed before the serialization fix; three coverage tests including
  the new end-to-end JSON/no-calibration regression passed afterward. Ruff, strict mypy, safety,
  report/contract/template fingerprints and QA hashes passed.

### Joystick geometric-base extraction (2026-09-07)

- `--geometric-base` now enables shared-scale search plus four independently normalized directional
  marker responses. A candidate center needs three markers at the expected cross geometry;
  the third-best response is its score. One marker may be occluded, without reducing thresholds.
- Train candidates improve 56→69/120. Reused dev improves 3→13/120: E2, SE4, NW2, W4, NE1;
  no STOP, N, S or SW. Dev threshold support is knob76/base13, not localization accuracy.
- All thirteen accepted dev frames were rendered and inspected in `accepted-dev-qa.png`:
  no obvious center or gross direction mismatch. The QA is informal inspection, not manual labels
  or independent accuracy. Unknown outputs remain non-training.
- Artifacts: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/joystick-extraction-v3-geometry`;
  final `qa-conclusion.json` status `JOYSTICK_GEOMETRY_CANDIDATES_PARTIAL`. Approximately 7.5 MiB.
  Source/template/report hashes verified. No raw decoding, model, input or test access.
- 9/9 synthetic scale cases and 40 focused tests passed, including single-marker occlusion,
  two-marker rejection, incorrect marker geometry, joint scale/coordinate restoration and STOP.
  Ruff and strict mypy (70 sources) passed.
- Next action changes from detector tuning to bounded data coverage checks with this frozen
  extractor on additional times from the existing train source. No new recordings, templates,
  threshold search or policy training are requested; existing dev remains disclosed regression.

### Joystick scale normalization (2026-09-07)

- Added `--normalize-scale`: base and knob share one of nine scales between 0.6 and 1.25.
  Coordinates return to native pixels; STOP/range checks use normalized displacement.
  Existing match/contrast thresholds are unchanged.
- Nine train-derived synthetic cases pass coordinate/action consistency (W/NW/STOP at
  0.75/1/1.25 scale and translated centers). This is not nine-class semantic accuracy. Initial
  harness clipping was corrected by retaining the full crop and using grid-aligned translations.
- Train candidates remain 56/120 with identical action sequence to v1. Previously inspected dev
  yields 3/120 candidates (all W at scale0.85, f30 indices37–39), up from zero. Dev knob-score
  support is 72/120 at >=0.65, but base-score support only 3/120 at >=0.35. Three dev QA sheets
  show useful knob alignment, persistent base uncertainty and abstention under shop overlays.
- Artifacts: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/joystick-extraction-v2-scale`.
  `qa-conclusion.json` closes as `JOYSTICK_SCALE_REGRESSION_PASSED_BASE_CUE_LIMITED`.
  Approximately 6.3 MiB; no decoding/model/input/test. Templates frozen before dev reuse.
- Next: directional-marker geometry for the base, not another scale/threshold sweep. Candidates
  remain non-training. Existing dev is a regression set, not an untouched benchmark.
- Validation: 37 focused tests, then 7 joystick tests including the added failure-path test proving
  synthetic failure writes a report before any dev read. Ruff and strict mypy passed. QA records
  the sole post-run source change (line wrapping only) and both reproducible source hashes.

### Joystick extraction v1 (2026-09-07)

- Implemented `movement-mvp --mode joystick-extraction --source-run <visibility-cache>
  --output-dir <new-external-directory>`. Reads cached RGB/PTS only. Base and knob are located
  separately every frame; low-match, ambiguous, low-contrast and excessive-displacement rows
  return unknown. STOP requires two consecutive visible centered observations within 12 pixels.
- Train-only development: fixed f30/frame39 centered reference, automatically located by Hough;
  97 aligned train base patches produce a median template. A grayscale single-reference trial
  yielded 38 train candidates; a blue-excess alternative yielded 1 and was rejected. Median
  aggregation yielded 56 and was selected before dev; no dev threshold adjustment occurred.
- Frozen contract/template precede dev reads. Results: train f10/f30/f60 produce 1/26/29
  candidates, total 56/120 (46.67%), including only one STOP. Dev is 0/120, all low-match unknown.
  Six QA sheets show useful train knob detection but base drift/background sensitivity and
  failed dev raw coordinates. Candidate coverage is not accuracy or tactical action supervision.
- Read-only post-evaluation circle checks support a scale mismatch hypothesis: three train
  radii 58.0/73.4/74.1 vs prominent dev 48.3/48.6/61.3 pixels. Circle types are not independently
  paired, so scale is not established as the sole cause. No extractor changes followed dev.
- Artifacts: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/joystick-extraction-v1`,
  approximately 6.3 MiB. `qa-conclusion.json` closes the batch as
  `JOYSTICK_FIXED_SCALE_EXTRACTOR_NOT_TRANSFERABLE`; no candidates are promoted to training.
- Next bounded implementation should normalize control scale, first using synthetic transforms
  of train crops. The inspected dev is now a regression set, not an untouched benchmark.
  Historical movement failures and visibility evidence remain unchanged.
- Validation: 35 focused tests, Ruff and strict mypy (70 sources) passed. Translation, ambiguity,
  dynamic base offsets, STOP confirmation and reset after unknown have regression coverage.

### Joystick visibility pilot (2026-09-07)

- User confirmed that existing recordings retain translucent live joystick feedback. The new
  pilot sampled the existing native-player train/dev sources at duration fractions 0.1/0.3/0.6,
  40 frames each at nominal 100ms spacing, using actual PTS. No test container was opened.
- Six native lower-left RGB bundles and contact sheets are saved under
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/joystick-visibility-v1`.
  Crops are 594×840 (train) and 594×819 (dev); actual maximum sampled PTS gap is 116ms.
  Data plus machine report/QA images occupy 156,030,102 bytes before the QA conclusion.
- All six sheets (72 displayed frames of 240 sampled) were inspected. Both sources show
  displaced knob/directional decoration. Train includes maintained displacement, changes and
  dim centered appearance; dev includes shop occlusion. These are feasibility observations,
  not frame labels, full nine-class support or measured automatic extraction accuracy.
- `qa-conclusion.json` closes the pilot as `JOYSTICK_FEEDBACK_VISUALLY_OBSERVABLE`.
  Next: one automatic extractor on cached windows; unknown for occlusion/ambiguity, STOP only
  with positive center/release evidence. Joystick must not enter future Actor tensors.
- No model, action labels, new recording, phone or test access. The source-bound report records
  the executed source; QA records the subsequent equivalent explicit-keyword NumPy type fix.
  Validation: 32 focused tests and strict mypy passed; no full historical suite required.

### Short-gap optical flow closure (2026-09-07)

- One RGB-only Lucas–Kanade run completed under
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/player-flow-gap-v1`.
  `report.json` preserves machine results; `qa-conclusion.json` closes QA and the final decision.
- Interior initialization `[16,112)` on both axes yields 180 direct detections in 002 versus
  208 original v2 detections. Nine accepted gaps add 15 retrospective frames; coverage rises
  12.12%→13.13%, continuous frames 177→194 (+9.60%), longest run 33→37 and independent
  16-frame windows 5→8. Sessions003/005 have no interior seeds and zero windows.
- 002 has 12 valid response events out of 30 sends (40%), 11/12 positive projections and
  median 1.88455 pixels. All six positions must exist and no conflicting intervening send is
  allowed. Timing is scheduled sampling, not measured action latency.
- All nine available gap clips were visually inspected: lower-map interior portrait, no fixed
  top-right UI capture or obvious identity switch. This is not independent identity accuracy.
  Fewer than the planned 12 clips exist; no clips were duplicated to fill the quota.
- Final `PLAYER_FLOW_GAP_INSUFFICIENT`: gain <50%, gaps <10 and zero sessions reach 20 windows.
  No training/retuning remains pending. Houyi identity is not the blocker for generic tracking.
- Report file SHA-256: `f3da05c01d10fd1f42b430864495370d090667048a52d00e19cbf7a7cf0e8a5d`.
  Validation: 31 focused tests, Ruff, strict mypy (70 sources), safety and diff checks passed.
  GPU 0; artifacts about 1 MiB; raw decoding, test frames and phone input 0.

The 2026-09-05 planning update makes Movement the only first-cycle learned component and retains
Macro/Combat as explicit rule baselines. It replaces repeated architecture/failure freezes with
bounded new train/dev runs; old reports/configs/test consumption stay immutable. The old branch
still failed its overfit gate. Its static-position data cannot establish navigation; BatchNorm is
only an unisolated hypothesis, not a proven cause. No new model or application result is claimed.
The new plan defines stages A-E, budgets, fallback delivery grades, simulator episode metrics and
a later separately authorized self-built-App/RL route. Runtime and model configs are unchanged.
The follow-up planning update adds concrete A-E technical notes: action-driven arena adapter,
goal-marked RGB, nine-action STOP mapping, shared diagnostic/training path, indexed trajectory BC,
window inference/cache semantics, transactional persistence/recovery and a single lazy CLI.
Stage A implementation is complete; training has not started. Review follows risk tiers: documentation-only
diff/link checks, focused development tests, and one full suite at deliverable code freeze.
Repeated agent review and per-commit historical full suites are no longer default requirements.
Budgets and device/test boundaries are unchanged; review/documentation share is capped at 8 hours
within the existing 80-hour engineering budget, not extra budget.
The MOBA-source follow-up updates the same plan using official hok_env/Hokoff documentation,
fixed-upstream wzry_ai train.py, and the OpenAI Five report (references in section 10).
It adds a no-training RGB geometry baseline, observable teacher-task constraints, fixed temporal
sampling, bounded standalone checkpoint evaluation, failure-inclusive episode totals and timing
breakdowns. Source designs are not local performance evidence. No external framework or budget
expansion was introduced.

## Engineering convergence stage A

- Added the single `movement-mvp --mode stage-a` offline entrypoint and fixed Houyi/marksman/
  blue/bottom configuration. No mobile module is imported or executed.
- Reused `RichPixelArena.reset/step/observe`: actions `E×6` changed position from `(2,4)` to `(8,4)`,
  then `STOP` preserved the goal position. The vocabulary prepends STOP while retaining the existing
  eight RichPixelArena direction order.
- Added backward-compatible `episode_end_kind`; natural navigation completion, timeout/video EOF,
  and runtime failure map to TERMINATED, TRUNCATED, and ERROR. Historical rows remain readable.
- Seven valid transitions and eight derived RGB frame bundles were stored. The terminal transition
  was committed before episode exit; reward total and device input were both zero.
- External artifact basename: `stage-a-seed0-v1` (76 KiB). Summary SHA-256:
  `52f1227c3ddfa4c91d5d1b76ea1895eb2d94262eb9a86e879c92172627de8555`.
- Focused tests: 13 passed. Ruff, strict mypy (66 source files), project safety (236 files,
  123 Python files, 55,890 nonblank Python lines, four root Markdown files), and diff check passed.
  No full suite was rerun under the risk-tiered policy. Exact engineering time was not instrumented;
  GPU time is zero.

## Engineering convergence stage B

- Materialized 32 independent causal RGB windows: STOP 8 and each of eight directions 3;
  all windows contain actual prior movement and execute their label only after frame 16.
  Dataset SHA-256: `ea75e7ec0b4e85bdff217b326b11bfd0bd8bf3fb7fc0c6b5103041a6dd936328`.
- The old P0 branch failed at 0.5938 accuracy / 0.7872 loss. Freezing all BatchNorm was worse at
  0.25 / 2.1224, so that hypothesis is closed. Neither run called full training.
- A 686,281-parameter task-specific GroupNorm+GRU passed with eval accuracy 1.0, loss 0.00745 and
  recall 1.0 for all nine actions. Report/checkpoint SHA-256:
  `c16cca8157e9cc99b3e4c363df35ea914d928d2b37a6cece59a12a865f085981` /
  `f7e0df57c947426e69f6a894fba5799d46a23710e247412201cfe48a7c27f918`.
- One task-specific attempt completed training but hit an evaluation interface error; one minimal
  retry produced the result above. The report normalization label was then corrected to GroupNorm;
  metrics and checkpoint were unchanged.
- Completed reported GPU kernels total about 10.3 seconds; the failed reporting attempt was not
  separately instrumented, so exact total GPU time is unavailable but remained below one minute.
  Dataset plus three diagnostics use about 93 MiB, dominated by two preserved 45 MiB P0 checkpoints.
- Focused tests: 20 passed. Ruff, strict mypy (67 source files), project safety (238 files,
  125 Python files, 56,447 nonblank Python lines) and diff check passed. Full training was not run.
- The target marker is PixelArena-color-specific. Real-video validity is not tested or claimed;
  stage C is simulator-only and R2 remains closed. Next work is 64/24 trajectory BC and dev rollout.
- The task-specific architecture is now the default for future formal training. P0 is explicit-only
  and remains a failed control; there is no automatic fallback. All four diagnostic attempts are
  consumed, so the checked-in contract rejects a fifth overfit run.
- The shared training path now records loss, gradient norm, finiteness and immediate parameter
  change after the first optimizer update. A focused CPU failure-path test proves a failed gate
  still writes a diagnostic checkpoint, reports `full_training_called=false`, and forbids reusing
  that checkpoint for formal training. Stage C must create a fresh seed-0 model.

## Engineering convergence stage C initial candidate

- Materialized 64 train and 24 dev action-driven trajectories with zero scenario overlap and exact
  initial-direction support of 8/3 per direction. Teacher completed 64/64 and 24/24. Frames are
  stored once per episode and windows are indices with 100 ms timestamps. Manifest self-hash:
  `a4d9499e367e165f7f9a1f33507a507b21754e690069e9a452e20407ad9dc913`.
- The fresh 686,281-parameter model trained for 20 epochs on 232 windows. Loss decreased
  `1.8618→0.0272`; first update was finite and changed parameters. Training took 8.07 seconds and
  peaked at 312,573,440 CUDA bytes. No diagnostic checkpoint or dev rollout was used in training.
- Independent dev evaluated only epoch 10 and 20 on the same 24 scenarios. Epoch 10 reached 9/24;
  epoch 20 reached 15/24 and was selected only within this failed run. Its collision fraction was
  0.2708 versus the 0.05 gate; it is not promoted. Initial-direction successes for epoch 20 were
  E 2/3, N 2/3, NE 1/3, NW 2/3, S 3/3, SE 2/3, SW 2/3, W 1/3.
- Baselines were teacher 24/24, RGB geometry 24/24, random 18/24, and fixed-east 2/24. With 24
  episodes, the frozen requirement to exceed the better random/fixed baseline by 8 is impossible
  once random reaches 18 (maximum possible gain is 6). The report remains FAILED; the gate is not
  relaxed after results. Collision failure independently prevents promotion.
- Dataset/train/dev report file SHA-256 values are
  `8165340bd12c493234e3f64744dcd84be37c35f8f68f309dc9ac0bc27e449ba3`,
  `b6772e72d89cfbe0f18833d047859f13b27c7e2f07bfa45579290f48256fdd41`, and
  `3c12eba93e85757353d704f9a99458841b21b13c7ac657f569c26bf735c79b7e`.
  Epoch-20 checkpoint SHA-256 is
  `c62bfa4af9f507ea594f3a3d2e4dcc23092b4c75b06b765ee7e23d864828b902`.
- Holdout, real video, phone input and RL remain unopened. No second training run is authorized
  until a separately versioned evaluation contract fixes the impossible comparison before seeing
  any new candidate. Class imbalance and absent learner-deviation recovery data are hypotheses,
  not established causes.
- New Stage C artifacts use about 5.7 MiB. Focused regression: 24 passed; Ruff, strict mypy
  (67 source files), project safety (238 files, 125 Python files, 57,130 nonblank Python lines),
  and diff check passed. The full historical suite was not rerun under the risk-tiered policy.

## Engineering convergence stage C v2 recovery correction

- Pre-run config SHA-256: `1242547ec87045c2d4ab29eb966923be58eb9f9db18d117324350d2abeaffac1`.
  The same 24 dev scenarios now require three consecutive STOPs; navigation-only damage settings
  remove combat/respawn confounding. The feasible comparison uses failure-inclusive mean steps.
- New data has 64/24 successful teacher trajectories, 16 train recovery episodes, 535 train windows
  and no scenario overlap. Actual perturbations remain in RGB history but are excluded from labels.
  Manifest hash: `234d91ba626599a1208defd11e11d6c3cacf4efa564bb9ddc36e8ca56e35da00`.
- Old epoch-20, reevaluated as reference-only under v2, remains 15/24 with collision fraction 0.66.
  Same-protocol baselines: teacher 24/24, geometry 21/24, random 1/24, fixed-east 0/24. Geometry
  also fails the collision gate; neither rule nor learned success is inferred from an old protocol.
- The recovery-only fresh run took 24.67 seconds, peak CUDA memory 312,573,440 bytes. Epoch-20 loss
  is 1.33646; both saved checkpoints reach 0/24. This candidate is frozen failed. STOP is 336/535
  supervised windows, so the remaining correction changes only the training sampler to class balance.
  Model, data, epochs, learning rate, optimizer and evaluation criteria remain fixed.
- External runs: `stage-c-v2-reference-v1`, `stage-c-bc-seed0-v2-recovery`,
  `stage-c-dev-seed0-v2-recovery`. Old v1 report/checkpoint/config hashes remain unchanged.
- Focused tests passed (26 plus one new checkpoint/reference binding test); Ruff, strict mypy and
  project check passed. No full historical suite or overfit diagnostic was rerun.

## Engineering convergence bounded correction closure and rule data path

- The class-balanced run reused the exact v2 dataset, config, seed-0 fresh architecture and 20-epoch
  schedule. Only the sampler changed. Actual sampled class counts range 1,134–1,269 over 10,700
  samples, so imbalance was mechanically reduced. Epoch-20 loss remained 2.21404 and both epoch-10
  and epoch-20 reached 0/24. This does not establish the cause of the failed learning; it does rule
  out describing class balance alone as a successful repair. Both bounded correction attempts are
  exhausted. No third correction, fifth overfit diagnostic, holdout or real-video run was started.
- Balanced train/dev report hashes:
  `8108416518414ed5a31a48b81bb8cb32d404722139f2550c8eaca63d7856b6a3` /
  `b34f9818c25790dbd19fb2914c78cde4172cb6f76298efb07880d0a0894d9ab8`.
  Balanced epoch-20 checkpoint: `65c441a528c6470bafc4aa94f6d2925b5e8a93d8cf5665b6b1fec0ce97c33a8b`.
  Recovery-only dev report: `8afbbee38f2b91b845a7bb9157c991112926599f64562af42e001c523c52e773`.
  Reference-only report: `b7f995fd544c2dbfe146b8714fa669f6f0be9f3f6795bb499d7b6e257fb085bc`.
- Added `movement-mvp --mode rule-batch`: fixed blue Houyi navigation, unique episode IDs, one
  UnifiedTransitionStore, real simulated displacement and three STOPs before terminal append/exit.
  Three separate CLI invocations completed cumulative 1 -> 3 -> 10 episodes, resuming only after
  completed episodes. Previously committed episodes were not replayed. This repeats one fixed
  scene with a structured simulator rule; it is not a learned or RGB-policy generalization result.
- Reopened the real SQLite artifact: integrity check `ok`, 10 episodes, 90 valid transitions,
  exactly 10 terminal rows, all observation/next-observation frame references present, every last
  three actions STOP, reward sum 0 and input 0. Artifact: `stage-d-rule-batch-v1/batch-summary.json`,
  SHA-256 `18d3f1f7b6007ddecc07dea1781403b0d0c792027b1a52161d93a9ff714a75f7`.
- This historical v1 run resumed only at completed-episode boundaries. D0 v2 below adds the
  separately versioned mid-episode recovery evidence. No failed learner is installed as a fallback.
- The two new training runs report 48.39 seconds total and ~298 MiB peak CUDA allocation each;
  three named dev/reference evaluations report 93.41 seconds. Named artifacts add 17,040,031 bytes
  (~16.25 MiB). Earlier ad-hoc forward timing and cumulative engineering time were not instrumented,
  so exact total budget remaining is not asserted. These runs stay within the original budget.
- Final verification: 28 focused tests passed; the checkpoint/reference test passed again after
  replacing its new direct vision-library import with the existing trainer I/O boundary. Ruff,
  strict mypy (67 source files), project check (239 files, zero findings, four root Markdown files)
  and diff check passed. The intermediate recovery commit preceded this final test-import repair;
  it is not a release freeze. The full historical suite was not rerun for this partial D delivery.
- All dev policies retain the same RichPixelArena legal-action execution boundary. Actor tensors
  contain only RGB; the random baseline samples uniformly over currently allowed movement/STOP
  actions, not an unconstrained nine-action space. Rule batch recovery is a storage capability,
  not evidence of a learned model or of in-episode checkpoint restoration.

## Engineering convergence D0 R0 offline delivery

- `movement-mvp --mode rule-batch` now accepts `--step-budget N`. A bounded invocation returns
  `PAUSED` after its Nth newly committed transition; `--resume` reconstructs the current arena by
  replaying only committed SQLite actions from the fixed seed. It verifies every step, observation
  chain, deterministic RGB view hash, policy action, terminal flag and frame bundle before appending.
- `run-contract.json` binds the resolved config, action order, 100 ms step, fixed rule and source
  hashes. Contract, summary and frame bundles use same-directory temporary files, `fsync` and atomic
  replacement. A committed missing/corrupt frame, changed config/source binding or changed stored
  action is rejected without continuing. A valid orphan frame is verified and reused.
- The real interrupted run paused at episode 0 step 4, then a new process recovered 4 committed
  transitions and completed 10/10 episodes. It contains 90 valid transitions, exactly 10 terminal
  rows, 100 frame bundles, three final STOP actions per episode, zero reward and zero device input.
  SQLite integrity is `ok`; delivery grade is `R0_RULE_OFFLINE`.
- The uninterrupted control independently completed the same 10/10 episodes. Both runs have
  transition-content SHA-256
  `b58bb1cee16fd906e3d6321095ff230e8423cebf2567c0c652d24be93d80b52c`
  and frame-view-manifest SHA-256
  `9977bd0f25ab90531444bb28cd39e72794514f3569e8cd09193a899e34f2072d`.
- Interrupted summary file SHA-256:
  `d0526cde0cf003891883c176357a6807310759fb31b2a97512b56fe887f94a37`;
  continuous summary file SHA-256:
  `5e81bf18d3e985dba32f151fc396a782cc3db9c888a1fd448ba6962226538d9a`;
  run-contract file SHA-256:
  `01fea6d9c17ddd57645a2eca31c8baad3f2b834948baf01a175508b06312aa4b`.
  The two directories total 1,089,030 bytes, below the 5 MiB D0 allowance.
- Artifacts: `stage-d-rule-batch-v2-recovery` and `stage-d-rule-batch-v2-continuous` below
  `HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/`. This proves deterministic PixelArena recovery
  and the offline transition data path only. Learned navigation, real RGB transfer, phone state
  recovery, Reward and RL remain false/unopened.
- Delivery-freeze verification passed: 32 focused tests, Ruff, strict mypy, `git diff --check`,
  and the complete `make check` with 390 tests in 63.31 seconds. Project safety found zero issues
  across 239 files and retained exactly four root Markdown authority files.

## Engineering convergence E R0 package closure

- Added lazy `movement-mvp --mode package`. Creation requires the accepted interrupted and
  continuous D0 directories plus a new output directory; `--verify-only` reads an existing package
  without modifying it. The packaging module imports no phone, video, training or model module.
- Creation independently audits each D0 summary self-hash, run contract, 90 SQLite transitions,
  10 terminal rows and 100 RGB bundles. It requires the interrupted run to recover four committed
  transitions and the control to recover zero, then requires identical transition and frame-view
  hashes. Failed checkpoints and SQLite WAL/SHM files are excluded.
- The immutable local directory is
  `$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/r0-delivery-v1`. It contains one resolved config,
  one final summary, one SQLite backup, 100 frame bundles and a final manifest: 104 files total,
  521,244 bytes. Manifest payload covers 103 files and 502,353 bytes.
- A separate verify-only process passed with 10 episodes, 90 transitions, 10 terminal transitions,
  100 frames, zero reward, zero input and SQLite integrity `ok`. Transition SHA-256 remains
  `b58bb1cee16fd906e3d6321095ff230e8423cebf2567c0c652d24be93d80b52c`; frame-view SHA-256 remains
  `9977bd0f25ab90531444bb28cd39e72794514f3569e8cd09193a899e34f2072d`.
- Manifest file SHA-256:
  `63fbe6e92f382e6f4a239d80166ffd0a4cb152d4c36fbb6c9166e5cff4e7d9ab`;
  summary file SHA-256:
  `83c08d9626040b15a44b20a1361c0a6c9b085e6d03797293c435cd13d049962e`.
  The package contains no absolute source paths, model checkpoint, raw video or device identity.
- Final capability is action-driven deterministic PixelArena navigation with causal transition
  persistence and mid-episode recovery. Learned navigation is false: the initial candidate reached
  15/24 with collision failure, and both bounded corrections reached 0/24. Real RGB, phone control,
  Reward, RL and holdout remain unopened. The cycle therefore closes at `R0_RULE_OFFLINE`.
- Final verification passed: 41 focused tests, Ruff, strict mypy, `git diff --check`, and the full
  `make check` with 399 tests in 66.22 seconds. Project safety reported zero findings across 241
  files and exactly four root Markdown authority files.

## Real RGB Movement observability preflight v1

- Opened a new diagnostic cycle with contract SHA-256
  `d3755cb682c425dff4e55fc6f7c571b13a7f1ffc63e834899ed8630d14c946ce`.
  It fixes two train sessions and one dev session, three 32-frame segments per session at 100 ms,
  the content-box/minimap color rules and pre-run gates. It permits no training, test, raw-RGB
  persistence, device input or R2 promotion.
- The existing target manifest has 103 train, 23 dev and 23 test sessions. This run opened only
  nine bound train/dev shards and sampled 288 frames. Test frames read, human labels, saved RGB,
  training calls and device inputs are all zero.
- Content boxes were found for all three sessions, including one stored portrait-letterbox session
  canonicalized counter-clockwise. All selected rows report stored rotation 0, so rotation diversity
  is not verified. Successful decode and content-box detection are not direction or semantic proof.
- The formal result is `TARGET_CONDITION_NOT_OBSERVABLE`. Overall self/target pair coverage is
  `0.4792 < 0.50`; per-session coverage is `0.6979`, `0.0938`, and `0.6458`, so the 0.20 minimum
  fails in one train session. Unknown fraction is 0.5208. Marker jump fraction is
  `0.4128 > 0.20`. Test isolation and content-box checks pass; all three observability checks fail.
- Manual temporary visualization confirmed that the fixed narrow minimap ROI misses the player cue
  in the low-coverage session and the nearest-red-pixel target switches among unrelated red cues.
  A non-formal in-memory check using a wider crop plus the older high-resolution component rule had
  lower coverage, so no repair result was saved or promoted. This is a diagnosis, not a second test.
- Formal report:
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-observability-v1/report.json`,
  114,901 bytes. File SHA-256:
  `7d81fec8aa3e5497d46d329dc23a24022ca4cd8fc13798e098e3b874f24256b9`;
  report self-hash:
  `d08cf20e17967343a43aed8bbe7476f59b2e829155a210e9930de349aa644591`.
- Semantic accuracy, learned navigation, promotion and R2 remain false. The next admissible step is
  a separately frozen detector/tracker contract that defines a full minimap crop, component or
  template evidence and temporal identity tracking before reading additional sessions.
- Verification passed: 17 focused tests, Ruff, strict mypy (69 source files), project safety
  (244 files, 129 Python files, zero findings, four root Markdown files), and `git diff --check`.
  The previous E delivery full-suite result remains bound to its earlier code; this diagnostic used
  the plan's focused-check policy and did not rerun all historical tests.

## Real RGB minimap goal canvas v2

- v1 showed that low-resolution player/red-target detection is not stable. V2 changes the task
  boundary: Macro supplies the semantic goal for the fixed blue-marksman-bottom context, while the
  canvas normalizes the complete minimap crop and draws one hollow goal ring. It no longer uses a
  nearest-red-pixel target or requires player localization to generate the canvas.
- Contract SHA-256:
  `243aa47aa164af2e9783a9ad08aa7cd372c602537704a75685374c72c9704e28`.
  It binds the failed v1 report, the same two train/one dev sessions, the same nine 32-frame clips,
  crop `[0,0,40,48]`, fixed goal/counterfactual coordinates and marker geometry. Test, training,
  RGB persistence, R2, promotion and device input remain disabled.
- Formal result: `GOAL_CANVAS_GENERATION_PASSED_SELF_LOCALIZATION_UNRESOLVED`. Content-box,
  nonblack crop, counterfactual goal change, deterministic repeat and test-isolation checks all
  pass. All 288/288 frames changed input when the goal changed and reproduced the same input for
  the same goal. Per-session minimum nonblack crop fractions were 0.9980, 0.9942 and 1.0.
- This result verifies only reliable target-conditioning construction on existing normalized RGB.
  It does not verify that the blue-bottom coordinate is semantically correct, that the policy can
  localize itself, or that the canvas improves navigation. `promotion_allowed=false` and
  `r2_allowed=false` remain fixed.
- Formal report:
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-goal-canvas-v2/report.json`,
  173,427 bytes. File SHA-256:
  `36e9ebbc30129d97df5ff66158b4e418c6e9f7052ef97b204f411858dbd1b659`;
  report self-hash:
  `8321314577d3405b9f78628aa44700ae864f063a768baf151db7a3905ea8ad0c`.
- The next admissible step is a separately frozen 32-sample simulator overfit that uses the same
  minimap-crop-plus-goal-ring input. Passing that gate would show task learnability only; it would
  still require semantic coordinate and real-domain checks before R2.
- Verification passed: 18 focused tests, Ruff, strict mypy (69 source files), project safety
  (245 files, 129 Python files, zero findings, four root Markdown files), and `git diff --check`.
  No model, GPU run or full historical test suite was needed for this bounded canvas change.

## Minimap goal-canvas overfit32 v1

- A separate contract binds the passed v2 goal-canvas report and a new simulator-only nine-action
  learnability check. It uses 16 frames, STOP 8, each direction 3, seed 0, AdamW, batch 8,
  learning rate 1e-3 and at most 200 updates. Contract SHA-256:
  `a5795ceb46606787aa2ce39bf0d6b549376d9547c2da76d0b4956e0acd8a11af`.
- Materialization created 32 independent action-driven episodes. Every input window contains
  state change, the label action executes only after frame 16, and eight same-player/background
  target counterfactuals change both image and expected action. Inputs contain only synthetic
  minimap RGB plus the hollow goal ring; no direction arrow or structured coordinate enters the
  model. Dataset/report SHA-256:
  `8b7104f55b71fc6a54b9be3d7c6204c50d1d357288c2e2bc30e9abc14ac33cde` /
  `8e8381dffa3e2dda46fada164947c2705648feab99fdad8dea5706afdb7e45e4`.
- The sole seed-0 CUDA diagnostic reused the selected 686,281-parameter GroupNorm+GRU and shared
  `train_step`. First update loss was 2.22269, gradient norm 7.46891 and parameters changed. After
  200 updates, eval accuracy was 1.0, cross-entropy 0.006135 and every nine-action recall was 1.0.
  Report file SHA-256:
  `a80d84fd7f83ecdbef471ab88ca8ba6cdcd89895a4c27f63099e4ad435de931c`.
- Diagnostic checkpoint SHA-256:
  `4ced74317fa228d2f0f2b241cfc038adbca7173d826e545f98070e285ebb121d`.
  It is explicitly diagnostic-only and cannot initialize formal training. The run took 3.75 seconds;
  dataset/run artifacts use about 5.2/2.7 MiB. Real RGB training frames, test reads and device input
  are zero; formal training, semantic lane coordinates, R2 and promotion remain closed.
- This proves only that the new synthetic input/label/training chain can memorize 32 causal samples.
  The next step is fresh 64/24 simulator trajectory training initialized from seed 0, followed by
  independent rollout. It does not reopen the previous failed candidate or its exhausted repairs.
- Verification passed: 15 focused goal-canvas/real-RGB/training tests, Ruff, strict mypy (70 source
  files), project safety (248 files, 131 Python files, zero findings, four root Markdown files),
  and `git diff --check`. No full historical suite was rerun for this bounded diagnostic.

## Minimap goal-canvas Stage C candidate v1

- The formal contract uses the passed overfit dataset/report, fresh seed-0 initialization, 64 train
  and 24 dev episodes, 16 frames, 100 ms steps, three STOP confirmations, class-balanced sampling,
  20 epochs and checkpoints at epochs 10/20. The initial jq-computed self-hash was rejected before
  data creation because jq normalized `0.0` differently; the runtime Python canonical hash was
  corrected to `4bcf99ca5641a80f81045c77de2a2a35a6fbf695d32ef464992008acc1a4bdc6`.
  No data or model result existed before that mechanical correction.
- Trajectory materialization passed with teacher 64/64 train and 24/24 dev, zero scenario overlap,
  360 train windows and no diagnostic checkpoint load. STOP contributes 192 windows; class-balanced
  training sampled all nine actions between 755 and 826 times. Data report/manifest file SHA-256:
  `5e618f1746ad9c690b0e63bde390be4206b46fbb5d68b3f9fa87513768642cf2` /
  `0bdf86482f1d0d591e0da080f50bbb9d0e7ba29e6eb5f26a9dd636c8d9da3fd9`.
- Fresh 686,281-parameter training was stable: first update loss 2.20692, gradient norm 4.26885,
  parameters changed, and epoch loss fell from 1.55928 to 0.01414. Training took 15.84 seconds and
  peak CUDA allocation was 312,573,440 bytes. Epoch-10/20 checkpoint SHA-256 values are
  `800d2209528ceb7b7f5584821b2885457defec57a617d3a37eca39eb40a543b9` and
  `a3e4647ac06f5bfe0a22d38c82669d37ef78005eda8d1c1f6e487226d3c3fc57`.
- Independent dev failed. Epoch 10 reached 6/24 with collision fraction 0.7292; epoch 20 reached
  9/24 with collision and oscillation both zero, mean 82.875 steps, but missed the 21/24 success
  gate. Teacher and exact-RGB geometry each reached 24/24 at 5.625 mean steps; random reached 1/24
  and fixed-east 0/24. All gates except learned success passed.
- The selected epoch-20 failed episodes contain 1,525 premature STOP requests. Initial mismatches
  span cardinal and diagonal relations, so more STOP reweighting alone is not supported. Low train
  loss plus weak unseen-position rollout is consistent with position memorization, but that causal
  explanation remains a hypothesis until a relational architecture control is run.
- Training/dev report file SHA-256:
  `663a779b153b15115477bb5a1c86a551cd326d099c539b40a5574fae76d3a64d` /
  `b9c4119394e067f8761ea8e20d3b5ffbd82c4cde6471ab8a59a59e6b6bf20488`.
  New data/train/dev artifacts use about 5.4/5.3/1.6 MiB. Evaluation took 24.08 seconds.
- This candidate is frozen failed without holdout, real-RGB training, R2, phone input or promotion.
  The next admissible learning attempt must separately freeze a spatial-relation model and cannot
  change data volume, epochs and sampler at the same time. Real player-cue evidence remains an
  independent prerequisite for any R2 claim.
- Verification passed: 30 focused movement/goal-canvas/real-RGB tests, Ruff, strict mypy (70 source
  files), project safety (249 files, 131 Python files, zero findings, four root Markdown files),
  and `git diff --check`. No full historical suite was rerun for this failed experimental candidate.

## Goal-canvas relational model diagnostic v1

- The controlled correction kept the overfit32 dataset, 64/24 trajectory manifest, 200-update
  diagnostic limit, 20-epoch formal limit, class-balanced sampler and all dev gates fixed. Only the
  model changed from flattened spatial features to a 93,611-parameter GroupNorm CNN with two
  learned spatial attention slots, their coordinate difference and a GRU. No coordinate labels or
  attention supervision were used. Contract SHA-256:
  `131535e7dd950d217a7d2fc5a772de1189a203662b290c287c18c69d205cb1b5`.
- The sole overfit32 run failed: eval accuracy 0.625 and loss 1.25823. STOP, NE, NW, SE and SW
  recalls were 1.0, while N, S, E and W recalls were zero. First update loss 2.16892 and gradient
  norm 1.07203 were finite and parameters changed, so the stop is learnability rather than an
  optimizer execution failure. Formal 64/24 training was not called.
- Report/checkpoint SHA-256:
  `0ce9184f6c417021ca25d3b9e62e32dcc5d2b4ce51bccaf25685686a48739455` /
  `7974b841272edc46710420cbb7606fdda01343d78bb7ff89a1b4b48d42a7b874`.
  The checkpoint is diagnostic-only and cannot be reused. Runtime was 4.13 seconds and the run uses
  about 372 KiB. Real RGB training, test, holdout, R2, promotion and device input remain zero/closed.
- The preserved report says `normalization_mode=train_batch_norm`, which is a reporting bug: the
  relational implementation contains GroupNorm and no BatchNorm. The code now reports GroupNorm
  for future relational runs; metrics and checkpoint were not changed and the consumed run was not
  repeated. This correction does not improve or reclassify the failed result.
- Action loss alone did not assign stable player/goal meaning to the two slots. The next admissible
  experiment must separately freeze automatic synthetic localization supervision and pass both
  slot-localization and action-overfit gates before any formal trajectory training. Real player-cue
  viability remains a separate requirement.
- Verification passed: 27 focused movement/goal-canvas tests, Ruff, strict mypy (70 source files),
  project safety (250 files, 131 Python files, zero findings, four root Markdown files), and
  `git diff --check`. No full historical suite was rerun for this stopped diagnostic.

## Goal-canvas automatic-localization diagnostic v1

- A new contract adds simulator-derived `player_xy` and `goal_xy` targets for the two attention
  slots while keeping Actor input exactly `rgb_sequence`. It uses the same 32 clips, action labels,
  200 updates, batch 8 and learning rate 1e-3. Action and localization losses have weight 1.0.
  Contract SHA-256:
  `0ea3077aee7758a3f3dc89911cac594dc1b428e5fb4212511cd28476b46b285c`.
- The derived dataset passed with STOP 8, eight directions each 3, 16 frame player/goal coordinate
  sequences and 32 independent episodes. Coordinates are automatic training targets and are not
  present in the Actor input. Dataset/report SHA-256:
  `842832fc9fc958835c31ebc205d155499a50b6867e333681634291a076704360` /
  `2b2b9b55423ba84f320aa3a4672950fa45954375917265879193e5059e24f48d`.
- The sole joint diagnostic failed. Mean slot error was 3.9356 pixels, passing the 5-pixel gate;
  player/goal errors were 4.0895/3.7818 pixels. Slot-cell accuracy was only 0.6875 versus 0.95.
  Action accuracy was 0.4375 and loss 1.05534 versus 0.95/0.05; only STOP, NW and SE recalls were
  1.0. First update was finite and changed parameters. Formal training was not called.
- Report/checkpoint SHA-256:
  `178681bd0df868ba8ec7e3db3f4b137bafe1e279b507e6782d94fa2780d80c33` /
  `9b6d02d046fd6d6e4cd39e8de1193d3cb502aef763c840c8e6315738f8d9c3f6`.
  Runtime was 4.25 seconds, peak CUDA allocation 572,406,784 bytes; data/run artifacts use about
  5.2 MiB/372 KiB. The checkpoint is diagnostic-only and cannot initialize another run.
- Automatic supervision improved coordinate proximity but did not jointly solve discrete slot
  identity and action learning. The next admissible hypothesis is a separately frozen two-stage
  diagnostic: train slots first, freeze them, then fit the action head/GRU. No formal training,
  test, holdout, real-RGB training, R2, promotion or device input is opened by this result.
- Verification passed: 15 focused goal-canvas/localization tests, Ruff, strict mypy (70 source
  files), project safety (251 files, 131 Python files, zero findings, four root Markdown files),
  and `git diff --check`. No full historical suite was rerun for this stopped diagnostic.

## Goal-canvas two-stage diagnostics v1-v2

- V1 fresh-initialized the same 93,611-parameter model, trained only spatial/attention parameters
  for 200 updates and required the previously frozen exact 16x16 cell gate. Mean error was 3.9704
  pixels, but exact-cell accuracy was 0.6699, so action training was correctly skipped. Report and
  checkpoint SHA-256:
  `35bcb0460b17680751800b8bf907ea08339b8366642212871c6a3bbb366de190` /
  `a690744cfefb8917d08863928facc723db6271f7a13d96d601711db2ce4bd7fb`.
- A read-only diagnosis showed every predicted slot was at most one 16x16 cell from truth;
  within-one-cell accuracy was 1.0. V2 therefore froze a different metric before rerunning:
  Chebyshev cell distance <=1 must cover at least 0.95, while the 5-pixel mean-error gate remained.
  It did not change data, model, updates, learning rate or action gates. Contract SHA-256:
  `1d2f872c23bd7c3bbb9cbd7b0099b1f6c42fd943807b0d2e032cde436d15ae0d`.
- V2 localization passed at 1.0 within-one-cell accuracy, maximum one-cell error and 3.9706-pixel
  mean error. Spatial and attention parameters were then frozen and remained byte-for-byte
  unchanged through 200 action updates. The action stage improved over joint training but failed:
  accuracy 0.8125 and loss 0.87116 versus 0.95/0.05; E and N recalls remained zero.
- V2 report/checkpoint SHA-256:
  `90f74fa2e2bd663b2eae95ead7893bbcac487e0f78147e7dd96de7fad6de6a84` /
  `b9ac5ba704526e54f167d714bb5963762c49379962885b0a34f334f5986d9b43`.
  Runtime was 5.12 seconds, peak CUDA allocation 572,092,928 bytes; each v1/v2 run uses about
  372 KiB. Both checkpoints are diagnostic-only and cannot initialize another run.
- The metric repair does not reclassify V1. V2 demonstrates stable coarse localization and clean
  stage freezing, but action learnability still fails. Further synthetic head, update or sampler
  tuning is stopped to avoid optimizing the toy renderer. The next evidence must concern the
  high-resolution real player cue; no formal training, test, holdout, R2 or device input is open.
- Verification passed: 16 focused localization/two-stage tests, Ruff, strict mypy (70 source files),
  project safety (253 files, 131 Python files, zero findings, four root Markdown files), and
  `git diff --check`. No full historical suite was rerun for these stopped diagnostics.

## Existing real minimap player-cue preflight v1

- Rehashing the 270 GiB raw-video directory merely to recover three privacy-discarded source paths
  was rejected as inefficient. The audit instead reuses three already identity-bound derived
  minimap sessions (002/003/005), each captured by cropping the high-resolution observation before
  resizing to 128x128. No new recording, raw frame or source locator is created.
- Contract SHA-256:
  `cec63ae0762fe897c32a72b71a743bc81a85d1a4c40a518d32f0acfd503d2b8e`.
  It binds all three summary files and their 18 observation shards, the existing green/red paired
  component rule, minimum per-session coverage 0.95, minimum single-candidate fraction 0.50 and
  maximum player-jump p95 5 pixels.
- All gates pass over 4,455 frames. Session coverage is 0.9771, 0.9960 and 0.9785; single-candidate
  fractions are 0.8794, 1.0 and 0.9972; player-jump p95 is 1.3174, 0.0 and 0.1022 pixels. Maximum
  missing streaks are 6, 4 and 6 frames. These results are consistent with a stable engineering cue.
- The cue comes from an automatic paired-color rule without independent human truth. Therefore
  `semantic_identity_verified=false` and `direction_accuracy_verified=false`; coverage does not
  prove that every selected component is the controlled hero. R2 remains false.
- Formal report:
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-cue-v1/report.json`, 4,834 bytes.
  File SHA-256:
  `b0f01f205845d44b14be6bfe1c691093bbdd8c70c55bc2c7d0ef89a314ad1695`;
  report self-hash:
  `46596f93180baef63580180a2b3c2d44f58ba1c41fc606566278570336e14295`.
- Human labels, test frames, training calls, new recordings and device input are all zero. The next
  admissible step is read-only composition of this cue with the fixed Macro goal canvas, reporting
  availability and temporal direction stability without claiming semantic action accuracy.
- Verification passed: 5 focused real-RGB/player-cue tests, Ruff, strict mypy (70 source files),
  project safety (254 files, 131 Python files, zero findings, four root Markdown files), and
  `git diff --check`. No full historical suite was rerun for this read-only preflight.

## Real player cue plus Macro goal continuity v1

- The frozen player-cue and goal-canvas reports are both file-hash and self-hash bound. The new
  read-only command composes the same 4,455 derived 128x128 minimap frames with the fixed Macro goal
  at `(0.78, 0.78)`, derives one of `STOP/N/NE/E/SE/S/SW/W/NW`, and applies a three-frame
  confirmation filter. It persists only one aggregate JSON report. Contract SHA-256:
  `fbf343b03beb1fe8e053c54798fcc597445cbe741df5653452b47ea3be3a86d3`.
- All engineering gates pass. Raw direction coverage is 0.9771/0.9960/0.9785 and confirmed
  coverage is 0.9758/0.9946/0.9771. Raw adjacent-frame direction-change fractions are
  0.0091/0/0; confirmed switches per minute are 2.2237/0/0. Every goal canvas changes the input
  and deterministic regeneration matches.
- This evidence is insufficient for policy training: the direction union is only `E/S/SE`, with
  sessions 003 and 005 entirely `S`. The contract therefore records
  `all_nine_directions_observed=false`, `direction_accuracy_verified=false` and
  `policy_training_allowed=false`. Continuity is not semantic correctness or navigation quality.
- Formal report:
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-goal-continuity-v1/report.json`,
  6,434 bytes. File SHA-256:
  `94a85bab8b54a0992e92b52ca57ab51c3cb9b601b6549dea34f97f612a9a9800`; report self-hash:
  `83d6c6fff00a607db6ab8155d54adeaa2e12abd7c1105b236703b7fa18a513c5`.
- Human labels, raw RGB persistence, test frames, training calls and device input remain zero. The
  current cycle stops here rather than tuning thresholds or training on a nearly single-direction
  target.
- Verification passed: 5 focused real-RGB composition tests, Ruff, strict mypy (70 source files),
  project safety (255 files, 131 Python files, zero findings, four root Markdown files), and
  `git diff --check`. No full historical suite was rerun for this bounded read-only audit.

## Real counterfactual goal overfit32 v1

- The data contract reuses the three bound minimap sessions but finds full nine-way geometric room
  only in session 002. Five mutually non-overlapping 16-frame source windows are each reused with
  different hollow Macro goal rings to form 32 samples: STOP 8 and every direction 3. No arrow or
  structured coordinate enters the model tensor; labels are geometric counterfactuals rather than
  observed or executed actions.
- Data materialization passed with 512 derived RGB frames, nine verified goal classes, zero
  cross-session windows and no raw fullscreen persistence. Dataset SHA-256:
  `0d17137539b12654d007e995eff8a697c78920b9de4a8d0eb91b5530690dd5bc`;
  data-report file/self SHA-256:
  `c77e49bddedf043c598fadd5dc94723dfd2d97508e1b3b5a4a1c6749ecaaa721` /
  `81b4aafdf01e24f4d600c20149b3f72be73c3712b609ab52f26dd8d4b43e2588`.
- The sole seed-0 CUDA diagnostic used the existing 686,281-parameter GroupNorm+GRU for 200 updates.
  First-update loss/gradient norm were 2.19159/4.81586. Final eval accuracy was 1.0, loss 0.007323,
  and all nine recalls were 1.0. The RTX 4090 run took 2.31 seconds; checkpoint SHA-256:
  `7833bc64a599c9914f06d70a91bc576263a6de731cc1453a38d4cdda10cdd5f4`.
- This proves only that the current tensor/label/model path can memorize goal-conditioned relations
  on five source windows. It does not prove held-out-window, session or action accuracy. The report
  therefore fixes `next_stage_allowed=false`, `formal_training_allowed=false`,
  `source_window_generalization_verified=false`; the diagnostic checkpoint cannot initialize a
  formal model.
- Dataset:
  `$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/real-counterfactual-overfit32-v1/`;
  run: `$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/real-counterfactual-overfit32-seed0-v1/`.
  Human labels, test frames, formal training and device input remain zero.
- Verification passed: 18 focused real-RGB/training tests, Ruff, strict mypy (70 source files),
  project safety (257 files, 131 Python files, zero findings, four root Markdown files), and
  `git diff --check`. No full historical suite was rerun for this bounded diagnostic.

## Player localization audit v2 and grouped model value gate

- Visual QA identified the old paired-color cue's near-continuous top-right candidate as a fixed UI
  marker. Audit v2 freezes exclusion `[112,0,128,16]` without changing color/component thresholds.
  Its first zero-noise interpretation was preserved under
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-localization-v2-zero-tolerance-failed`;
  four isolated session-005 detections motivated the finalized unsupported-session ceiling of 1%.
- The final audit opens all 4,455 bound minimap frames. Session 002 retains 208 detections
  (coverage 0.1401), single-candidate fraction 1.0 and jump p95 4.72 pixels. At the fixed 1,000 ms
  response lag, 15/17 sent-action events have positive displacement projection (0.8824), with
  median projection 2.009 pixels. Session 003 retains 0 detections; session 005 retains 4
  non-continuous detections (0.0027) and zero response events. The supported scope is therefore
  session 002 partial only, not three-session localization.
- Contract SHA-256:
  `e2b281d32b5376e03a61e8c7783fa7e19eccf2b49cf351f98613ca4dae0623cb`.
  Final report file/self SHA-256:
  `11684e3d4050b9d45af3b18fb8fd2567f48e2597f0cfc9d8babdb420e42fc9e0` /
  `4ab5200b045620f8587a3aec604d20e0eddbdc58aff2dc70c31db36150ffb353`.
  Three hash-bound QA contact sheets remain beside the report.
- The grouped value gate uses five non-overlapping session-002 source windows. A 20-pixel goal
  distance is the smallest implementation repair that supplies all five groups while remaining
  above the fixed 8-pixel STOP radius; the 24-pixel preflight stopped before training. Each group
  contains all nine actions, producing 45 unique clips per variant with no group crossing.
- One command executes five folds for each of `full`, `player_masked` and `goal_only`: 15 fresh
  seed-0 models, 200 updates each, no checkpoint persistence. Full mean accuracy/macro-F1 are
  0.7778/0.7370 and worst-fold accuracy is 0.3333. Player-masked accuracy is 0.6889 and goal-only
  accuracy is 0.6667, so full gains are only 0.0889 and 0.1111 versus the required 0.15. Only
  group isolation and aggregate per-class recall pass; the remaining five gates fail.
- Final status is `REAL_COUNTERFACTUAL_MODEL_SHORTCUT_OR_NO_GENERALIZATION`. The run took 33.17
  seconds on CUDA with peak allocation 309,823,488 bytes. Contract SHA-256:
  `afb7cc2295d328371146e0f04dda2b59fe129d367ea2561850d966ffef39fa4f`.
  Report file/self SHA-256:
  `1ee4ad8ddbd73025deef0463b7d341b0d288f2f6546df8b8ed8e5ff10f6e0691` /
  `8d549f28471ae424909eb3d5de3e91679613b132abb7f4345e6de5afeb9fa356`.
- No fold checkpoint, new training dataset, test frame, human training label or device input was
  produced. The previous 32-sample checkpoint remains diagnostic history and is not promoted.
- Verification passed: 23 focused real-RGB/training/boundary tests, Ruff, strict mypy (70 source
  files), project safety (259 files, 131 Python files, 62,893 Python lines, zero findings,
  four root Markdown files), contract/report self-hash checks and `git diff --check`. No full
  historical suite was rerun.

### Bounded appearance tracking and paired QA (2026-09-06)

- Added `movement-mvp --mode real-player-tracking-audit` in the existing offline module. It reuses
  v2 source/hash bindings; no dependency, model, phone interface or separate protocol was added.
- A 15x15 median template comes automatically from the first 16 interior v2 candidates in 002.
  Candidate matching removes the red-pair requirement, retains the green component proposal and
  fixed-UI exclusion, searches +/-3 pixels, and requires correlation >=0.70, distinct-peak margin
  >=0.05, two consecutive matches and <=8 pixel inter-frame displacement. Missing/ambiguous
  frames return unknown immediately; discontinuity requires confirmation again. Settings were
  fixed before the single full scan, not adjusted after its result.
- Results: 002 has 155/1485 tracked frames (0.1044), 21 acquisitions and a longest unknown stretch
  of 1224 frames; 003/005 have zero tracked frames. No coverage improvement was demonstrated.
  The 9 eligible 002 action-response events include 8 positive projections, median 2 pixels.
  This check requires all six frames observed and no intervening different dispatched action,
  unlike the earlier endpoint-only audit; event-count differences cannot isolate tracker effects.
- Developer inspected all three paired QA sheets. 002's early matches align visually with the
  moving portrait; later samples show the main view near the fountain and a missing/clipped
  minimap portrait. 003 sampled main views remain at the fountain, with an edge-clipped portrait
  in the initial minimap. 005 mostly shows the fountain too, despite other minimap markers moving.
  Do not label those markers as the controlled player simply because they are green. These
  observations do not establish framewise identity accuracy, death/respawn, or camera-follow state.
- Machine report remains `PLAYER_APPEARANCE_TRACKING_DIAGNOSTIC_ONLY`; the engineering decision is
  `PLAYER_LOCALIZATION_REPAIR_NOT_PROMOTED`. False-lock rate and reacquisition latency are unknown,
  not zero. Do not integrate this candidate into navigation or create training labels from it.
- Artifacts: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-appearance-tracking-v1`;
  three paired sheets, one template PNG, one report; 2,170,046 bytes total. Report file SHA-256:
  `17c0a7deb4dd1539a34d59af255121108b6a4f40c5be4113fc9a9e19d9b80f93`;
  self hash `80281a2bb1b096418d1afb20b48774edea887ee32f84a3407d49488105f612ea`.
  The executed implementation hash matches the source. All historical artifacts remain unchanged.
- Verification: 13 focused real-RGB/boundary tests passed, including missing-frame abstention,
  red-free tracking, distractor/UI rejection, ambiguous identities, jump/reacquisition, source
  tampering, overwrite rejection, no checkpoint and lazy CLI dispatch. Ruff, strict mypy on 70
  source files, project safety and `git diff --check` passed. No full historical pytest or GPU
  diagnostic was run. Model runs, GPU time and device commands are all zero; engineering hours
  were not precisely metered.
- Continue only with a half-day-capped inspection of existing source ROI recoverability. If the
  complete spawn corner exists, re-extract once and verify visually; otherwise retain the data
  limitation. Do not demand new recording, fabricate off-ROI coordinates or repeat template tuning.

### Existing-source minimap recoverability inspection (2026-09-06)

- Checked the three teacher session directories, source summaries, hash-matched private ROI
  configuration and `_observation_roi_frame` implementation. The ROI hash matches all three:
  `a9a17abc8927050327390a43525b24c43b91987948c44e1f916d9b760379e809`.
  Only four 128x128 derived views are saved; configured per-ROI `output_size` does not override
  the sampling function's default 128. Other saved views do not cover the missing lower-left
  minimap region. No same-session full frame is present in these bound session directories.
  This is not a whole-disk proof that no independently recorded copy exists.
- Checked one hash-verified target shard for each of the three previously selected train/dev
  sessions (768 cached rows). Their whole-frame cache is also only 128x128; enlarging that cache
  cannot restore native minimap detail. No test shard was opened.
- Loaded the existing cohort/owner/privacy/pre-ingest bindings and matched all three selected
  raw source identities in memory. Only selected train/dev video files were opened; other file
  entries were stat-enumerated to resolve identities. Raw source paths were not persisted.
- Raw stream sizes: `0667d97c` train is 2400x1080; `c1121610` dev is 2340x1080;
  `03f37224` train is 1080x2408 with frame display-matrix rotation. This is source availability,
  not proof that these recordings have the same hero, layout or gameplay as the teacher sessions.
- Nine time-point selections (20/50/80 percent in each source) generated three derived corner
  QA sheets. Both landscape sources visibly retain the map and lower-edge margin; the dev middle
  sample is obscured by the shop and must remain unknown. The portrait sheet is rejected: the
  applied rotation sign produces an upside-down HUD. Keep that diagnostic evidence, defer this
  source, and do not change historical rotation/teacher pipelines or spend another batch on it.
  Three preliminary previews were also decoded. Keyframe preroll is recorded for the nine final
  selections, not for preliminary previews; do not describe this as only nine decoded frames.
- Conclusion: `EXISTING_LANDSCAPE_SOURCE_RECOVERABLE_QA_ONLY`. Original teacher missing pixels are
  unrecoverable from their crops; the two landscape sources are alternative data. No player
  accuracy, tactical label, hero identity, navigation result or promoted checkpoint is claimed.
- Evidence: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/minimap-source-recoverability-v1`.
  Report SHA-256 `04046ae8bb928d63cb37c2b2e38b9a575a05fbc89b237c1178daa0d0dd139132`.
  Source identities, decoded PTS, display geometry, derived RGB hashes, QA image hashes and
  rejected portrait preview are retained without raw source locators or full-frame copies.
- Only documentation changed in the repository. Verification uses JSON/hash/QA checks and
  `git diff --check`; no redundant test, model training or GPU run. Phone input and test decoding
  remain zero. Next: one short window per landscape source, native-resolution minimap with edge
  margin, controlled-player identity first, then continuity. Existing failed tracker stays frozen.

### Native-resolution player-cue pilot (2026-09-06)

- Implemented `movement-mvp --mode native-player-pilot`, lazy offline video loading, and a
  reusable RGB-only green-ring candidate/continuity helper in the existing real-RGB module.
  No model or new package dependency was added. PyAV is allowlisted here only for the two
  selected cohort-bound landscape sources. Nonselected video entries are stat-enumerated, not
  opened; source paths never enter the saved report. Existing source/owner/privacy bindings
  are loaded before decoding.
- Each source contributes 16 frames at nominal 200 ms from its 20-percent time point. Crop first,
  then resize to 256x256; save map/main-view arrays, actual timestamps and native crop hashes.
  The train window spans 175.304–178.304 s and dev spans 148.2785–151.2835 s. Maximum actual
  sampling gaps are 212/214 ms, not a claim of exact 200 ms timestamps or absence of source loss.
- Synthetic boundary testing caught floating-point target-time comparison skipping exactly
  aligned frames. Fixed this with integer microseconds and verified EOF/portrait rejection.
  One final materialization after that fix produced arrays identical to the initial real windows
  in every saved field; both artifact versions are preserved. This was a sampling-code validation,
  not another model run or threshold search.
- The fixed green-ring heuristic uses RGB >= green 150, green-red >=30 and green-blue >=20,
  radii 8–14 px and >=7/8 circumference hits after 3x3 dilation, with a non-green core. It is
  uncalibrated. Initial global-unique confirmation retained dev 4/16 and train 0/16. Completing
  the local association logic (unique candidate within 12 px when a prior point exists, without
  color/shape threshold changes) retains a consecutive 9/16 dev frames and 0/16 train frames.
  Missing or discontinuous candidates emit unknown; a distant candidate cannot immediately
  replace a confirmed point. Green terrain still generates unconfirmed distractors.
- Developer inspected both 16-frame minimap sheets and matching main-view sheets. Dev's nine
  confirmed circles visually follow the green portrait. Train portraits overlap and obscure its
  border; only one isolated raw ring candidate exists. This is a visible-cue diagnostic, not
  independently measured identity accuracy, false-lock rate, action accuracy or generalization.
  Hero identity is not bound to Houyi; no data is eligible for policy training.
- Final artifacts: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/native-player-pilot-v2-integer-time`.
  Report file/self SHA-256:
  `c93974432616725923e9c2261993850c05c3ae878c3303b73de9305d6b205b9d` /
  `81e35cb4e55f1bcd7101a61fcba291dcbb8fc05de92b3783f8396c55301a1de3`.
  The executed final source hash matches the implementation. Initial `native-player-pilot-v1`
  remains unchanged; both directories total 30,115,414 bytes. All model runs, GPU time and input
  commands are zero; no test video, full-frame copy or action label was created.
- Verification: 17 focused real-RGB/boundary tests; the five native/ring/CLI-related focused
  selections reran after final integration; Ruff, strict mypy on 70 source files, project safety,
  source/artifact hashes, v1/v2 array equality and `git diff --check`. No full historical suite.
- This bounded pilot is closed as `NATIVE_PLAYER_CUE_PARTIAL_NOT_PROMOTED`. Preserve unknown,
  do not lower detection thresholds to force the occluded train window through, and do not feed
  the dev example back into training. Next inspect only the cached map/background separation and
  failure reasons before deciding whether additional visible train-only footage is useful.

### Cached map/background diagnostic (2026-09-06)

- Reused the existing native pilot via `--mode native-player-pilot --source-run <cached-run>`;
  no new command or detector. Factored unchanged temporal confirmation into a reusable helper
  that also explains unknown: no ring, ambiguity, first confirmation or discontinuity. The
  unfiltered output is mechanically checked against the source report before comparison.
- Compared provisional QA rectangles for interior, edge margin and external context. They are
  not automatic semantic map masks or training labels. Filtering happens to candidate lists,
  not pixels; the unsafe interior-only variant is diagnostic only and is not promoted.
- Dev: 55 raw candidates comprise 14 interior, 22 edge-margin and 19 context candidates.
  Keeping interior+edge leaves 36; interior-only leaves 14. All three variants confirm exactly
  9/16 frames. Train: only one interior candidate, 15 frames without ring evidence, and zero
  confirmed frames in every variant. Thus the tested background pruning cannot by itself repair
  these windows; do not add more filtering or lower ring thresholds to force coverage.
- Evidence: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/native-player-background-audit-v1/report.json`;
  self hash `85898f623ba4bef21fd96b6cfff0c1ffca4a9df305b154e04ea971cc0374635a`.
  It binds cached NPZ hashes and reproduces the frozen track. No RGB modification, video decode,
  model run, GPU work, action label or runtime filter promotion occurred.
- Next bounded action is visibility screening of at most three earlier windows in the same
  approved train source. Dev/test will not be decoded or reassigned; a visible candidate is not
  sufficient to verify controlled-player identity or unlock training.

### Train-only visibility screening (2026-09-06)

- Added `--train-visibility-scan` to the existing native pilot rather than a new command or
  protocol. It opens only the existing `0667d97c` train source and inspects exactly three fixed
  windows at 5/10/15 percent, each 16 frames with unchanged 200 ms sampling, crop and ring rules.
  The cached-audit path and source-scan flag cannot be combined. Dev/test video files are not
  opened by this scan, and source identities/splits remain bound to the existing cohort.
- Outcomes: 43.826–46.839 s gives 0/16 confirmed frames; 87.652–90.654 s gives 13/16;
  131.478–134.470 s gives 15/16. The last window has a unique raw ring candidate in all 16
  frames and loses only the first frame to confirmation warm-up. The middle window has a
  briefly obscuring panel; unknown and reacquisition are retained. The earliest window and
  original 20-percent window retain overlapping-portrait failures without threshold changes.
- Developer inspected all three map sheets and both clearer main-view sheets. The candidates
  follow the green portrait with main-view activity consistent with gameplay. This supports
  selected cue visibility, not independently verified controlled-player identity or exact
  coordinates. The main-view/hero appearance is not established as Houyi. These two useful
  windows are from the same video, not two episodes; they are selected QA examples, not a
  random benchmark or demonstrated model improvement. No policy training labels were created.
- Preferred next QA fixture is `0667d97c-f15-native-window.npz`; the 10-percent fixture is a
  retained short occlusion/reacquisition example. They are already saved; do not decode or
  re-run the scan just to inspect them. All three windows, including the failure, remain in
  `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/native-train-visibility-v1`.
- Report file/self SHA-256:
  `3d36508a0b443fe7adf5a029f7c2c9a595f1bf68eec5d127662f628266cbb45d` /
  `5ffc25862858e67e06b33a0931c266dbf9d28c2396cce0680544b6a0863bac54`.
  All nine artifact hashes and the executed implementation hash were checked. New scan
  artifacts total 22,040,523 bytes; 48 new sampled frames plus decoder preroll, zero GPU/model
  runs and zero input. Cached audit report file hash is
  `c2873757ffaeba51f7072f67db988923d99a82339f5b074de3cce491aba12bcc`.
- Verification: 19 focused real-RGB/boundary tests passed for the cached audit; seven affected
  native/ring/background selections passed after the scan extension, including a mechanical
  check that only train is decoded at the three fixed fractions. Ruff, strict mypy on 70 source
  files, project safety, artifact hashes and `git diff --check` passed. No historical full suite.
- State is `NATIVE_TRAIN_VISIBLE_WINDOWS_FOUND_QA_ONLY`. Continue from saved clear clips, not
  more background filtering or repeated attempts on occluded frames. Identity verification and
  coordinate stability remain the next questions; formal Movement training and navigation
  integration remain closed.

### Cached identity and coordinate closure (2026-09-06)

- Used only the saved 10/15-percent train windows: 32 original cached frames from one source.
  Added a sampled-map-cell to source-screen-pixel conversion and fixed translation diagnostics,
  with tests for axis order, exact sampling-grid round trips, unchanged input arrays and zero
  evidence not being misreported as a successful comparison. No decoder or new CLI was added.
- Four shifts, up/down/left/right by 4 pixels, preserve every baseline confirmation/unknown
  decision. All 112 supported comparisons have zero coordinate equivariance error. These are
  transformed observations of 28 confirmed frames, not 112 independent examples or a measured
  localization accuracy. Adjacent point displacement maxima are 3.00/2.24 pixels, medians 1/1;
  these include real movement and quantization and must not be called localization error.
- The conversion uses the exact integer linspace sampling grid and the hash-bound 2400x1080
  source geometry from the recoverability report. Round trips are exact for every confirmed
  point. Output is source-screen XY, not game-world coordinates or a proven lane mapping.
- Enlarged portrait/main/map QA is visually consistent with the same moving green portrait in
  both windows, unlike the historical fixed corner UI. Camera panning/viewport overlays are
  visible in the 10-percent clip; main-view screen center is not used as a position label.
  Independent self identity, Houyi identity, action supervision and global semantic accuracy
  remain unverified. Stop repeated inspections of the same selected examples here.
- Evidence: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/native-player-coordinate-identity-v1`;
  report file SHA-256 `57837b07ba8d8c28713478c03a6067d7451c514ec20ae48ef20d36a0a171505a`;
  `identity-contact.png` SHA-256
  `77b65f181d49942f4801825d07d40fd6dca31605335f220eb590c21f662b1c7c`.
  The report binds source NPZ/geometry/implementation hashes. No raw video, dev/test, model,
  GPU, new action labels or device input was involved.
- Verification: 21 focused real-RGB/boundary tests, Ruff, strict mypy on 70 source files,
  project safety and `git diff --check` passed. State is
  `NATIVE_COORDINATE_GEOMETRY_PASSED_IDENTITY_QA_ONLY`. These are fixed non-promoting perception
  fixtures; any future weak visual-anchor learning task must state its own limited purpose
  and cannot reopen failed Movement training or relabel these coordinates as ground truth.

### Weak visual-anchor cohort audit v1 (2026-09-06)

- Added `movement-mvp --mode native-anchor-cohort-audit`; it binds 8 train and 4 dev session
  identities to the existing cohort. A preceding exploratory pass read one 15-percent window per
  selected session; the formal audit explicitly excludes 15 percent and uses fixed 10/30/60-percent
  windows. This is a transparent engineering audit, not a pristine random benchmark.
- Materialized 36 session-window groups, 576 minimap frames and 12 compact NPZ files. Per-session
  map sheets show all 48 frames and confirmed cues; main-view sheets show first/middle/last frames
  per window for developer QA. Only derived map arrays persist. No raw source locator/full frame,
  action label, test frame, model or checkpoint is stored.
- Frozen data-support gates: at least 6/8 train and 3/4 dev sessions must contain one window with
  at least 8 confirmed frames; totals must reach 128 train and 48 dev confirmed frames; split
  identities must be disjoint. Results: train 7/8 and 130; dev 2/4 and 64; split isolation passes.
  Only the dev session-count gate fails, so overall status is
  `WEAK_VISUAL_ANCHOR_COHORT_INSUFFICIENT`. Do not lower it or start training from the passing totals.
- Developer inspected all four dev map/main sheets. Confirmed examples often follow a plausible
  moving green portrait, but shop panels obscure entire windows and ring presentation is not
  invariant across recordings. A quick cached cyan-line probe finds many tower/path/UI components;
  it is exploratory output only and was not implemented as a viewport detector.
- Evidence: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/native-anchor-cohort-v1`;
  report file/self SHA-256:
  `e7d003cf7c694ae1a073ba68365c86fa8a2d04ffaa5034a16f02b854aa6b33b9` /
  `b9a09feef7cb1394857e79fdc15788097f90ffc8d3fe1f54a37debdd0f432b97`.
  All 36 artifact hashes pass; 37 files occupy 94,733,407 bytes. Model/GPU/device-input counts are zero.
- Verification: focused native-cohort tests cover all 36 fixed group calls, split isolation,
  nonselected test exclusion, output immutability and non-promoting flags. The affected real-RGB
  and boundary suite, Ruff, strict mypy, project safety and `git diff --check` passed.
- One v2 data-coverage repair may add four new dev landscape sessions in anonymous order while
  freezing the detector, fractions, support definition and all v1 evidence. It cannot tune from
  dev, change the 3-session gate or train a policy. If combined support remains below three dev
  sessions, stop the green-ring weak-supervision route.

### Weak visual-anchor dev coverage repair v2 (2026-09-06)

- Added one separately versioned `native-anchor-cohort-repair` mode. It requires the immutable v1
  failed report and validates its source identities, status and five gate outcomes. Four new dev
  landscape sessions were selected in anonymous hash order: `cdcad062`, `d13bfda2`, `d41ffac3`,
  `d81cc35a`. Detection thresholds, 10/30/60-percent windows, support definitions and all v1
  artifacts remain unchanged.
- The repair adds 12 windows/192 frames and combines counts with v1 without copying or rewriting
  prior artifacts. Combined train remains 7/8 supported sessions and 130 confirmed frames; dev
  rises from 2/4/64 to 6/8/171. All five frozen gates pass, including session split isolation.
- Developer inspected all eight new map/main sheets. Confirmed cues follow plausible green
  portraits rather than the old fixed corner UI; shop overlays, clustered portraits and absent
  rings still produce unknown. This QA does not independently verify controlled-player identity,
  Houyi identity, semantic accuracy or action correctness.
- Result: `WEAK_VISUAL_ANCHOR_COHORT_SUPPORTED_QA_ONLY`. It authorizes only a new data audit for
  session-isolated counterfactual visual-anchor relations. It does not authorize Movement policy
  training, deployment, phone input, reward or RL.
- Evidence: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/native-anchor-cohort-repair-v2`;
  report file/self SHA-256:
  `6a2c4baa305485d8959157a2675d7300b00d2daea58658b8811fdc02b8679896` /
  `9b051154b9275e9914aff338509e3a6f88a95ef0eff4b413a5b2257ad528e79a`.
  The 13 new files occupy 32,131,242 bytes and bind the v1 report file/self hashes. GPU/model/input
  counts remain zero; no test frame or raw source locator was persisted.
- Focused tests cover prior-report binding, exact four-source decode, 12 new group calls, combined
  support and non-promoting flags. Affected real-RGB/boundary tests, Ruff, strict mypy, project
  safety and `git diff --check` passed.

### Weak-anchor counterfactual data gate (2026-09-06)

- Added `native-anchor-counterfactual-audit`, reading only the 16 cached cohort NPZs bound by the
  immutable v1/v2 reports. It opens no video. Every current green-ring track must exactly reproduce
  its frozen per-window positions before a group can be considered.
- Eligibility requires at least 8 confirmed frames in the 16-frame window, a confirmed final-frame
  anchor, and all nine targets fitting inside a 256x256 canvas at 24-pixel offset and 7-pixel ring.
  Existing 128x128 counterfactual behavior remains the default; canvas size is now explicit for
  this 256px audit and regression-tested.
- Result: train has 8 groups from 7 sessions and dev has 9 groups from 6 sessions. Each group
  supplies exactly `STOP,N,S,W,E,NW,NE,SW,SE`, producing 72/81 balanced synthetic-relation samples.
  Group IDs are unique and train/dev session hashes are disjoint. All seven gates pass.
- Status is `WEAK_ANCHOR_COUNTERFACTUAL_DATA_SUPPORTED`. It permits one seed-0 weak visual-relation
  diagnostic after materialization. It is not executed-action BC: controlled-player/Houyi identity,
  game-world coordinates, navigation performance, Movement policy training and deployment stay false.
- Evidence: `$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/native-anchor-counterfactual-data-v1`.
  Report file/self SHA-256:
  `0ff57047baaa350ef7aa54aa61fa664082bd9a03cd75d8c83a7c7ff04d3baa91` /
  `a00ecaa7fb04ab466393ece8bc2d7f80d08f5da56f50294e59dd7772967369c8`.
  The 14,827-byte report binds v1/v2 file/self hashes and every cached NPZ hash; no new RGB is stored.
  Model/GPU/video/test/device-input counts remain zero.
- Focused tests cover balanced group construction, split isolation, cached-hash binding, output
  immutability, explicit canvas geometry and non-policy flags. Affected real-RGB/boundary tests,
  Ruff, strict mypy, project safety and `git diff --check` passed.

### Weak-anchor counterfactual materialization (2026-09-06)

- Added `native-anchor-counterfactual-materialize`. It verifies the passing audit and every source
  cache hash, then stores each of the 17 source clips once. A separate 153-row index binds group,
  nine-class label and target coordinate. No RGB is copied nine times.
- Dataset arrays are `source_clips[17,16,256,256,3]`, group/session/split/anchor/source-hash rows,
  and sample group/label/target arrays of lengths 153/153/153. All 17 source clip and group hashes
  are unique; each of nine labels has exactly 17 logical rows. Train/dev remain 8/9 groups and
  72/81 samples.
- Target RGB is not persisted. Training must copy a source clip, draw the fixed hollow yellow
  ring (`radius=7`, `thickness=2`, RGB 245/225/45) at the indexed 256px target, then use fixed
  even-index nearest sampling to 128px. The source cache stays unchanged.
- Evidence: `$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/native-anchor-counterfactual-v1`.
  Dataset SHA-256 `4464e2fe14290d9c1198276945f920cf9accbff01290b83cb098db616dc81e95`;
  report file/self SHA-256:
  `fe3063e66ce47f45dd3105639a08b109b53671a9d7daeaf3f12cc67673ce8148` /
  `b0f2bef291fc121c8c0fb552be29b9b54267a968b9352731922b107a56290ded`.
  Two files occupy 33,839,002 bytes.
- This opens one relation diagnostic only. Executed-action labels, identity verification,
  Movement policy training, test/video decode, model runs, GPU and device input remain zero.

### Weak-anchor relation diagnostic closure (2026-09-06)

- Added `native-anchor-relation-train` using the existing 686,281-parameter GroupNorm+GRU,
  fresh seed 0, AdamW `lr=1e-3`, batch 8 and 400 updates. It first trains/evaluates the same
  36 samples from four train groups. Formal full/anchor-masked/goal-only models run only after
  overfit reaches accuracy 0.95 and cross-entropy 0.05. Every model records first-update loss,
  gradient and parameter-change evidence. No checkpoint is ever written by this diagnostic.
- Initial rendering drew radius 7 on the 256px source and then nearest-sampled to 128. Labels are
  balanced, all 153 clips per variant are unique, and each non-STOP target changes 512 spatial
  pixels across 16 frames versus the STOP clip; nevertheless overfit36 stayed at accuracy 0.1111,
  cross-entropy 2.2029 and predicted only W. Formal models were correctly skipped.
- The sole repair changes only rendering order: resize source/masked/neutral RGB to 128 first,
  then draw the same radius-7 target at the halved indexed coordinate. Dataset, split, detector,
  model, optimizer, updates and gates are unchanged. It is explicitly bound to the initial failed
  report and cannot run as an unbound fallback.
- Repaired overfit36 passes at accuracy/macro-F1 1.0 and cross-entropy 0.00647. Formal session-
  isolated evaluation then fails: full, anchor-masked and goal-only each have accuracy 0.1111,
  macro-F1 0.0222 and predict only SW; full control gains are both zero. First-update evidence
  passes for all four v2 models. This distinguishes local memorization from cross-session signal.
- Frozen status: `WEAK_ANCHOR_RELATION_DIAGNOSTIC_FAILED`. No checkpoint exists. Do not add
  updates, architecture variants, windows, more weak labels, alternate ring thresholds or another
  localization feature to this lineage. It establishes neither action BC nor navigation.
- Initial run report file/self SHA-256:
  `909c31f8ddad8dcde7cfcd44ab32cd9dbc3f6558dbdc69debf5180370ac898a8` /
  `1e150ab4872d45ac49a7d82a17785a0ea1cbf87c10d2410dca448513b226af1e`.
  Repair report file/self SHA-256:
  `cea872c9320d4d50d99419d3cade48da4f683828b5322cb800f8d0a4bab994ca` /
  `323e487414ca540463b5a936439ca409d8905d4fad727fad3dcecac4c86ef58a`.
  Reported training-loop wall time is about 5.0 seconds initial plus 19.9 seconds repaired; variant
  generation and process overhead are not GPU-metered. Five total model runs, zero input/test/video.
- Focused tests cover source-preserving variant generation, render-order difference, overfit-stop,
  repair binding, control gates and no-checkpoint output. Affected real-RGB/training/boundary tests,
  Ruff, strict mypy, project safety and `git diff --check` passed.

### Observable death/respawn Event-to-Store replay (2026-09-06)

- Reused the immutable E1a width-repair contract/report and its single frozen dev-death session.
  Detector thresholds, temporal confirmation, source frames and event types are unchanged. The
  replay opens only derived `main/minimap/hud` shards whose file and summary hashes match the
  frozen report; it does not decode video or open test.
- `HealthTemporalEventEngine` initializes on frame 0, then attaches events produced by each next
  observation to the corresponding zero-action transition. The 285 frames produce 284 contiguous
  transitions and reproduce exactly 1 DEATH, 1 RESPAWN and 17 SELF_HP_DELTA events. Death appears
  at step 69 (17.0 s) and respawn at step 85 (20.2 s). Episode-local exact-once remains active.
- Every proposal is deterministic `HOLD/NONE/WAIT`; every executed action is acknowledged as
  NOOP without an input attempt. Reward version is `reward-disabled-v0`; all four components,
  total and reward event IDs are zero/empty. All rows are intentionally `training_eligible=false`
  because semantic/HP accuracy is unverified and Reward is disabled.
- Fixed TransitionStore propagation semantics: an intentionally non-training but causally valid
  transition no longer makes the next row causally invalid. Propagation remains when the previous
  payload itself has `causal_order_valid=false`. Dedicated tests cover both paths.
- All 284 stored rows pass standalone validation and retain causal order; training count is zero.
  The final VIDEO_EOF/TRUNCATED row is the only terminal and is committed before summary creation.
  SQLite `integrity_check=ok`; all 285 derived frame bundles and the frame manifest are hash-bound.
- Evidence: `$HOK_LARGE_ROOT/runs/hierarchical-event-e1/death-respawn-transition-replay-v1`.
  Report file/self SHA-256:
  `268f0c8ab65ea5895d06b27d928c9c29f309de75b69fbe603a666b223dccb133` /
  `2e7dcbd634aafdbafe581b1b3cd1cf26a0cba450f4805fb8af9c37ece5682ecb`.
  SQLite SHA-256 `dba9f55c19f48c6d7d7909dc499d678aff09d1c70f01c61cfe7b431fa5eff048`.
  The 288 files occupy 31,831,775 bytes. Input, model, GPU, mobile capture and test counts are zero.
- This is `DEATH_RESPAWN_EVENT_TRANSITION_REPLAY_PASSED`, an engineering result only. It does not
  validate event semantics or authorize death reward. Next use existing non-test artifacts to
  count cross-session candidate windows and false positives before any RewardHub change.

### Existing operational death-candidate inventory (2026-09-06)

- Added an exclusive `--candidate-session ID=/absolute/directory` mode to the same E1 health CLI.
  It reuses the frozen detector and reads only existing derived RGB shards; output stores anonymous
  session IDs and summary hashes, never source paths. No raw video or test is decoded.
- Inventoried every current PASSED operational session with persisted ROI RGB: 8 sessions and
  7,080 frames. Only `death60s` emits a paired DEATH/RESPAWN; the other seven are negatives with
  neither event. There are zero unpaired sessions and zero death events on sessions whose legacy
  hard-stop count is zero.
- `teacher002` contains six legacy hard-stop frames but health visibility is 1.0 and the frozen
  engine emits no death. This is useful disagreement evidence: hard-stop cannot be repurposed as
  automatic death truth. SELF_HP_DELTA counts likewise remain diagnostic and do not authorize HP
  reward.
- Frozen gate requires at least three paired positive and three negative sessions. Negatives pass
  7/3; positives fail 1/3. Status is `DEATH_RESPAWN_CANDIDATES_INSUFFICIENT`; semantic accuracy,
  Reward, training, promotion and device input remain false.
- Evidence: `$HOK_LARGE_ROOT/audit/hierarchical-event-e1/death-respawn-candidate-inventory-v1/report.json`;
  file SHA-256 `5f2fddf77d3a9f4b0cba56af19635f7523c903ac65fadb71e005f0b84f02186e`;
  self hash `8adb406226af73a2ae456ff626c27d1998c3c8825001e8cb60a45d389767abce`;
  3,861 bytes. Model/video/GPU/test/input counts are zero.
- Focused tests cover one-positive/three-negative failure, paired-event accounting, zero-hard-stop
  false-positive gate, path-free report and immutable output. Next work may only search existing
  train/dev video for an independent death countdown/respawn cue; E1a thresholds remain frozen.

### Health plus death-banner consensus (2026-09-06)

- Added `HealthBannerConsensusEngine` without changing the frozen health detector. It retains
  SELF_HP_DELTA events, buffers a health-derived death, and confirms DEATH only while the existing
  full-frame death-banner hard-stop is active. It confirms RESPAWN only after the banner clears
  and health returns to stable ALIVE. Episode-local fusion keeps each event exact-once.
- Unit evidence proves neither cue is sufficient alone: health disappearance without hard-stop
  and hard-stop while health stays visible both emit no death. A reporting test also covers the
  banner appearing before the three-frame health confirmation; rejected rising edges are settled
  after the full session, not prematurely counted.
- The same eight-session inventory was rerun with `--banner-consensus`. The known death-stop
  session retains exactly one DEATH and one RESPAWN. Session 002 has two hard-stop rising edges;
  both are rejected because health remains alive. The other six sessions remain event-free.
  There are seven negatives, zero unpaired sessions and two rejected misleading rises.
- The known-positive, negative-support, no-unpaired and rejection checks pass; the frozen minimum
  of three positive sessions still fails. Status is `DEATH_BANNER_CONSENSUS_DATA_INSUFFICIENT`.
  This improves candidate specificity but does not verify semantic accuracy or authorize Reward.
- Report: `$HOK_LARGE_ROOT/audit/hierarchical-event-e1/death-banner-consensus-v1/report.json`;
  file/self SHA-256:
  `921b45d55a0f086ecb8de95178db362bf54c0f771daf146119c628864cd97e95` /
  `e4bc50722c3c4b23561b7ae195b1188ecdd23d7f12bd53b0a6b4dc208094f9f7`.
  The 3,919-byte report persists no source locators. An 18-frame developer-only main/HUD contact
  sheet supports the visible dim/death/return sequence; it is not a training label. Its SHA-256 is
  `2b58fbfc4b8bb5b9284675a31b49dfd7a5b1181bf92c6bd4f2caeb278d3097bf`.
- No video/test/model/GPU/input work occurred. Next is a capped 12-session train/dev raw-video
  visibility preflight for cross-layout banner/countdown cues; do not reuse local hard-stop bits
  as external-video labels.

### Native raw-video death-cue preflight closure (2026-09-06)

- Added `movement-mvp --mode native-death-cue-preflight` in the already video-allowlisted
  `movement_real_rgb.py`. It binds the same fixed 8 train/4 dev anonymous landscape sessions and
  cohort/privacy evidence used by the prior native audit. No test source is selected.
- Sequentially decoded all 652,190 video frames once and sampled 52,868 observations at 200 ms.
  The center-health input crops before resizing to 128. The independent banner ROI is normalized
  from the frozen mobile `[720,0,880,22]` geometry; red/white minimum pixel counts become fixed
  area fractions 0.28409/0.01136. No threshold was selected from these videos.
- Automatic output superficially reports 11 paired positive sessions and one unpaired session.
  This fails the no-unpaired gate. More importantly, banner detections number 118–850 frames per
  session and 18–398 frames also lack the center health cue, inconsistent with a short death
  overlay and indicating broad domain mismatch.
- Developer inspected five representative train/dev contact sheets. The top-center ROI repeatedly
  selects scoreboard values, kill notifications and persistent red UI; candidate frames visibly
  show ongoing movement or combat. The center-health detector is also not identity/layout invariant.
  Consequently all 11 automatic paired outputs are rejected as semantic death evidence; the QA
  does not replace them with manual labels or estimate a true positive count.
- Frozen conclusion is `NATIVE_DEATH_CUE_PREFLIGHT_DOMAIN_MISMATCH`, not a near-pass. Do not adjust
  ROI, color fractions, health crop, confirmation timing or model capacity. Do not materialize
  candidate clips, train Reward, open test or branch to tower/economy weak labels without a new
  independent semantic source.
- Evidence: `$HOK_LARGE_ROOT/audit/hierarchical-event-e1/native-death-cue-preflight-v1`.
  Machine report file/self SHA-256:
  `082d69886d505e034c8f9b1e68ce4f19566d485f323479d04e847f3fc406cca7` /
  `5ae851ee573bf2f439af4ef729b73b1877ea3719fa7344738cc7201fb9a68885`.
  QA conclusion file SHA-256:
  `e32c9219892136d3fc4b366001a71fe77a37fc72c96b30dde5066da7f3434926`. Twelve contact sheets
  plus reports occupy about 13.7 MiB; no full-resolution frame or source locator is persisted.
- Tests cover normalized frozen pixel counts, exact center crop geometry, fixed train/dev source
  selection, test exclusion and output immutability. GPU/model/input counts remain zero.

### R1 offline engineering cycle package (2026-09-07)

- Added `movement-mvp --mode package-cycle` and a read-only `--verify-only` path in the existing
  delivery module. Creation requires an already verified R0 package, an intact Event replay and
  exactly four distinct failure statuses. Existing R0 packaging remains unchanged.
- The Event audit reopens SQLite read-only, validates every transition, observation chain, frame
  bundle/view hash, event count, terminal placement, reward and training eligibility. SQLite is
  copied through backup into DELETE journal mode; no WAL/SHM sidecar is required.
- Four failure records are included without models or bulk QA media:
  `WEAK_ANCHOR_RELATION_DIAGNOSTIC_FAILED`,
  `DEATH_BANNER_CONSENSUS_DATA_INSUFFICIENT`,
  `NATIVE_DEATH_CUE_PREFLIGHT_INSUFFICIENT`, and
  `NATIVE_DEATH_CUE_PREFLIGHT_DOMAIN_MISMATCH`. Their source file hashes and statuses are bound
  by the package summary and outer manifest.
- The real package contains R0's 90 transitions/100 frames and Event's 284 transitions/285 frames.
  Event counts are DEATH 1, RESPAWN 1 and SELF_HP_DELTA 17. Total reward and input commands are
  zero; Event training-eligible count is zero; promoted checkpoint is null. It contains 398 files
  and 32,440,974 bytes.
- A new process completed `package-cycle --verify-only` with identical results. Delivery grade is
  `R1_ENGINEERING_OFFLINE_ZERO_REWARD`; deterministic Movement, mid-episode simulator recovery and
  Event-to-Store are true, while learned Movement, semantic Reward, real-video policy, mobile
  control and RL are false.
- Package: `$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/r1-offline-engineering-v1`.
  Manifest SHA-256 `3f92db327ef964922e0bb1d5f8d56785d82a673fd242ac63095a262c860b5ad5`;
  summary SHA-256 `e70c12a9ae41b3691da88cbb7486077fa820b8850552a2dadc154fff1b9aea46`.
- Focused tests cover creation, nested verification, seven tamper locations, exact failure set,
  no checkpoint/SQLite sidecars, output non-overwrite and read-only CLI behavior. This ends the
  current engineering cycle; packaging is not a policy promotion.
- Delivery verification: 18 focused package tests passed. The first full `make check` invocation
  used a shared environment without this checkout's `src` on `PYTHONPATH` and failed collection
  with 33 wrong-package import errors; no test body ran. `RUN_PYTHON` now binds
  `PYTHONPATH=$(CURDIR)/src`. The exact Makefile command then passed Ruff, strict mypy, all 448
  tests in 106.79 seconds and project safety (259 files, 131 Python, 66,837 lines, zero findings).
  This is the final full check for this cycle; it is not repeated per evidence file.

### Existing Houyi-bound data audit (2026-09-07)

- Added `movement-mvp --mode houyi-data-audit`. It reconstructs anonymous candidate identities
  from the frozen pre-ingest/cohort evidence, checks split before opening a container, reads only
  container/stream metadata and decodes no frame. It also audits the current hero-profile template,
  cohort owner attestation and existing summary hero fields without persisting source locators.
- Formal results: all 103 train and 23 dev metadata containers open successfully; test container
  opens and test frame decodes are both zero. The nine metadata keys are generic creation/Android/
  MP4 handler fields, each present in all 126 records. Filename/directory/container hero keyword
  hits are zero.
- `configs/hero_profile.example.json` is `TEMPLATE_NOT_CONFIGURED` with empty hero ID. The cohort
  owner attestation contains no hero field. Among 152 summaries, one declares `hero=houyi`, but it
  is the PixelArena Stage A simulator result and lacks a real-session identity binding. Complete
  real Houyi bindings are zero.
- Status is `HOUYI_BOUND_REAL_DATA_NOT_AVAILABLE`; training and hero-data contract creation remain
  false. Existing recordings cannot be relabeled from visual resemblance, role assumptions,
  skill availability or configuration declarations.
- The initial v1 report exposed a mechanical observability bug: `Counter.update(metadata)` treated
  metadata values as counts and concatenated strings. It is preserved at
  `$HOK_LARGE_ROOT/audit/hierarchical-policy-v0/houyi-existing-data-binding-v1` and is not evidence.
  The sole fix counts `metadata.keys()` and adds an integer-count regression test; identity results
  and split behavior are unchanged.
- Final report:
  `$HOK_LARGE_ROOT/audit/hierarchical-policy-v0/houyi-existing-data-binding-v2-metadata-count-fix/report.json`.
  File/self SHA-256:
  `8d3bafc33dcf855b31e6323c377f09e9adbbb157b1935e4919b3ae73fdacaf53` /
  `67fd1e2e429618ec160ee26f1f12c9836443e0d6cd03c14558163abd4201d67c`;
  1,877 bytes. The rejected v1 file SHA-256 is
  `5cbbbe7eedd3db5daaf14ee1038cc41fe267d34dd50f9517944a4240497a517f`.
- Boundary deviation: before this formal implementation, an exploratory script opened container
  and stream metadata for all 149 MP4s, including 23 test files, before split filtering. It decoded
  zero test frames, found zero hero hits and was not used for selection, thresholds or models.
  Formal code and tests now prove test filtering occurs before `av.open`; no further test access.
- Next work can only prepare a future-only episode declaration/reference contract. It must not
  bind historical sessions and does not itself authorize capture, input, Reward or training.

## Frozen Global Agent execution state

```text
CURRENT GOAL: Freeze Global Agent v1 model evidence
BLOCKING FAILURE: observable health/distance factors are weak, but the only auxiliary encoder update destroys complete-episode performance
NEXT ACCEPTANCE COMMAND: none; frozen Dagger is the permanent Global Agent v1 policy
DO NOT WORK ON: scenario-card training, second DAgger, early PPO, 10m Shadow or phone input
```

The episode score remains lexicographic: safety violations, non-timeout terminal, tower progress,
stuck time, teacher fallback, win rate, then local metrics. Stage 1A passed; Stage 1B reached 19/20
normal crystal terminals and 20/20 tower-progress episodes. The 40/10 dataset contains 3,276 rows. Seed-0 TCN achieved intent/zone
macro-F1 `0.8465/0.7257` and 7/10 pure-student terminals. The only DAgger round reached 9/10,
raised mean tower damage from `11.1` to `12.0`, and reduced fallback from `0.0828` to `0.0525`.
The video adapter improved unlabeled video-dev consistency from `0.01581` to `0.00780`, but reduced
simulator terminals `9/10→8/10`; strict promotion rejects it and keeps DAgger. Fresh holdout seeds
selected DAgger (`18/20`) over adapted (`17/20`). Its six-state student challenge passes only 2/6,
so input remains closed. The separately authorized 60-second zero-control Shadow completed 114/114
cycles at p95 end-to-end latency 37.2 ms, with zero hard stops and zero input, but ran on a paused
screen and emitted only `DISENGAGE/OWN_BASE`; it validates transport stability only and is
semantically unevaluable. The failed adapter
candidates, challenge reports, and the DAgger report missing tower-damage comparison remain
preserved evidence.

The following active-scene diagnostic completed 114/114 cycles at p95 end-to-end latency 33.1 ms,
with zero hard stops and zero input. Its 114 frame hashes were all distinct, proving a visually
dynamic source, but all candidates still remained `DISENGAGE/OWN_BASE`. It therefore passes runtime
safety only; candidate diversity was not demonstrated. The report does not claim a verified
real-video semantic error, and it does not admit threshold changes, retraining, a 10-minute Shadow,
or input.

The single permitted geometry repair cropped the locally configured main view, minimap, and HUD
before model resizing. It again completed 114/114 zero-input cycles at p95 32.9 ms and again emitted
only `DISENGAGE/OWN_BASE`. This excludes the previous whole-screen resize path as a sufficient
explanation and closes further preprocessing variants in this lineage.

The frozen training labels are not retreat-dominated, while both video-dev replay and mobile Shadow
are. At v1 freeze time, the remaining blocker was cross-domain imitation from action-free human
observations, not device transport, layout geometry, or a retriable threshold. Human IfO Bridge v1
replaced the scenario-card proposal for that bounded repair; its result is now frozen non-promoted.

The project demonstrates reproducible RGB policy research in project-owned PixelArena and
read-only/strictly bounded mobile-testbed infrastructure. It does not establish commercial-game
skill, tactical optimality, general transfer, or authorization to control any third-party client.
T8-v4 supplies the diagnostic protocol, offline implementation, preserved initial failure, single
coordinate repair, repaired audit, and seed-0 decision. The permitted repair has been consumed and
the spatial-selectivity gate failed, so the lineage is frozen as insufficient weak-supervision
evidence. No larger model, additional training, replay, Shadow, or device input is allowed.
T8-v5 demonstrates strong partial ROI signal, especially for basic attack, but does not separate
enemy and skill1 evidence sufficiently from correlated wrong regions. Its per-head gate failed,
so this lineage is also frozen rather than expanded into a temporal model.
It is a separate lineage and must not be presented as a continuation of the failed v2.7 or v3
pilots by threshold relaxation.
