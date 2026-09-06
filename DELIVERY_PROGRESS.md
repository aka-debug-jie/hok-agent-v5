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
| Engineering convergence | `NATIVE_TRAIN_VISIBLE_WINDOWS_FOUND_QA_ONLY`: fixed cue confirms 13/16 and 15/16 in two earlier windows of the same train video; original train window remains 0/16 | Selected perception QA only; not independent episodes, identity accuracy or policy training data |
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
CURRENT GOAL: use the existing clear train-side clips to check identity and coordinate stability
CURRENT STATUS: NATIVE_TRAIN_VISIBLE_WINDOWS_FOUND_QA_ONLY; failed learned model and R0 remain frozen
BLOCKING FAILURE: visual cue is available in selected clips but player identity and policy-label validity remain unverified
NEXT ACCEPTANCE: perception QA on the saved 10/15-percent clips; no detector retuning, source expansion or training
COMMAND STATUS: cached background audit plus one capped three-window train scan complete; dev/test untouched by the scan; zero model runs
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
