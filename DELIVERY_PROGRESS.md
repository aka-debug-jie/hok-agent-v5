# Delivery progress

This is the only current-state ledger. The sanitized historical ledger is preserved in
[`docs/DELIVERY_HISTORY.md`](docs/DELIVERY_HISTORY.md). Large datasets, checkpoints, recordings,
and run evidence are local artifacts below `HOK_LARGE_ROOT`; they are not distributed in Git.

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

| Route | Current result | Promotion boundary |
|---|---|---|
| Engineering convergence | `D0_R0_RULE_OFFLINE_PASSED`: interrupted and continuous 10-episode runs matched at 90 transitions and 100 frame bundles | E packaging/conclusion only; no learned promotion/holdout |
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

```text
CURRENT GOAL: complete E packaging and close this engineering cycle at R0
CURRENT STATUS: D0_R0_RULE_OFFLINE_PASSED; learned Movement remains not promoted
BLOCKING FAILURE: both new Movement candidates are 0/24; no training attempts remain this cycle
NEXT ACCEPTANCE: one final package/claim audit and deliverable code freeze; no further training
COMMAND STATUS: step-4 interruption -> resume -> 10 and uninterrupted 10 matched; holdout unopened
BUDGET: cycle remains capped at 80 engineering hours / 24 GPU-hours / 50 GiB new artifacts
DO NOT WORK ON: old test, E1 terminal research, phone input, online RL, model growth, MoE, PPO, new human labels
```

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
