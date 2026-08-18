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
| Operation Movement Teacher v1 | Offline audit, zero-input and input smoke passed; two 5-minute sessions eligible | Collection paused for direction-diversity review before split freezing |
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
