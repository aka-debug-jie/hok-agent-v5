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

| Route | Current result | Promotion boundary |
|---|---|---|
| Hierarchical Policy v0 | `P0_TEMPORAL_SSL_FAILED`: new seed-0 SSL passed overfit/non-collapse and reached dev F1 0.703, but missed the frozen 0.75 gate | Freeze this failed lineage; no encoder initialization, Reward, phone input, test, or online learning |
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

## Hierarchical Policy v0 execution state

```text
CURRENT GOAL: freeze the failed P0 temporal SSL pilot and choose a separately versioned next objective
BLOCKING FAILURE: seed-0 temporal SSL dev macro-F1 0.7031 is below its frozen 0.75 gate
NEXT ACCEPTANCE COMMAND: make hierarchical-p0-ssl-smoke; no retraining command is authorized
DO NOT WORK ON: phone model control, online RL, 200M model, MoE, continuous action, PPO, multi-critic
```

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
