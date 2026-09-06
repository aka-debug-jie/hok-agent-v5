# AGENTS.md

`DELIVERY_PROGRESS.md` is the only current-state ledger. Read it, `README.md`, and
`BOUNDARIES.md` before changing this repository.

## Current planning authority (2026-09-05)

`docs/ENGINEERING_CONVERGENCE_PLAN.md` governs future work ordering and resource budgets.
Stages A/B passed, but the first Stage C simulator candidate failed: epoch 20 reached 15/24 with
collision fraction 0.271. Random reached 18/24, so the frozen +8 comparison is infeasible on 24
episodes. The separately versioned `configs/movement_mvp_stage_c_v2.json` now defines the next
bounded correction: the same 24 dev scenarios, navigation-only damage settings, three consecutive
STOPs and failure-inclusive step efficiency. Recovery-only and class-balanced corrections both
failed at 0/24; this round's two correction attempts are exhausted. Do not start another training
or overfit diagnostic. The cycle is closed at `E_R0_DELIVERY_COMPLETE`: D0 passed 10 episodes after
a step-4 pause/resume and exactly matched an uninterrupted control; E packaged and independently
verified that evidence without weights. Any real-RGB or learned-Movement continuation requires a
new cycle. The first new-cycle step is now frozen failed as
`REAL_RGB_OBSERVABILITY_V1_FAILED`: 3 sessions / 9 clips / 288 train-dev frames reached 0.4792
pair coverage and 0.4128 marker-jump fraction; one train session reached only 0.0938 coverage.
Do not run R2, train from these candidates, lower the gates, or open video-test. A separately
versioned v2 goal canvas now passes deterministic crop/marker/counterfactual checks on the same 288
frames by taking its semantic goal from Macro instead of detecting red targets. Player localization,
lane-coordinate semantics and policy value remain unresolved. The new simulator goal-canvas
overfit32 contract passed at accuracy 1.0 / loss 0.00614 with all nine recalls at 1.0. Its sole
diagnostic attempt is consumed and its checkpoint is not reusable for formal training. The next
work created fresh 64/24 simulator trajectories and one formal candidate. Training loss reached
0.01414, but dev rollout reached only 9/24 versus 21/24; it is frozen failed. Geometry reached
24/24, while failed learned episodes repeatedly requested premature STOP. Do not add epochs,
trajectories or sampling repairs to this architecture. A future learned correction requires a new
spatial-relation architecture contract and must still resolve the real player cue before R2. The
first 93,611-parameter two-slot relational model is now frozen failed at overfit32 accuracy 0.625 /
loss 1.258; N/S/E/W recalls were zero and no formal training ran. Do not rerun or proceed to Stage C.
The first automatic-localization correction is also frozen failed: mean slot error passed at 3.94
pixels, but slot-cell accuracy was 0.6875 and action accuracy/loss were 0.4375/1.055. Formal
training did not run. Two-stage v1 stopped at its exact-cell gate; v2 used a predeclared one-cell
tolerance, passed localization at 1.0 within-one-cell / 3.97 pixels, froze the localizer exactly,
but action accuracy/loss were only 0.8125/0.871. Formal training did not run. Synthetic model tuning
is stopped. Player-localization audit v2 proves that the old near-100% cue coverage was dominated
by a fixed top-right UI marker. After excluding it, session 002 has 0.1401 partial coverage and
action-response support; sessions 003/005 have no usable track. The old cue, continuity and
overfit32 results are non-promoting. A five-group, three-variant grouped evaluation then failed:
full mean accuracy was 0.7778 and exceeded player-masked/goal-only controls by only 0.0889/0.1111.
Freeze the result as `REAL_COUNTERFACTUAL_MODEL_SHORTCUT_OR_NO_GENERALIZATION`; do not tune this
model path. The next work may only repair automatic player localization coverage. Formal training,
R2 and device input remain closed.
That bounded continuation is now closed: native-resolution weak-anchor coverage passed after one
four-dev-session expansion, but the session-isolated relation diagnostic failed. Its sole render
repair changed only target rendering order, passed overfit36 at 1.0/0.00647, then full,
anchor-masked and goal-only each collapsed to dev accuracy 0.1111 / macro-F1 0.0222. Freeze as
`WEAK_ANCHOR_RELATION_DIAGNOSTIC_FAILED`; no further anchor detector, epochs, model, samples,
checkpoint or Movement training. A new cycle must choose a separately observable target; R0 rule
Movement remains the engineering baseline and phone/test/RL remain closed.
The next observable-data cycle now passes an offline engineering replay only: the frozen E1a
dev-death session produces 284 causal, explicitly non-training transitions with one DEATH, one
RESPAWN, 17 SELF_HP_DELTA events, zero reward and a terminal VIDEO_EOF row committed before exit.
Freeze detector thresholds and E1a evidence. This permits only a read-only cross-session death
candidate audit on existing train/dev artifacts; semantic accuracy, Reward, promotion, phone,
test and RL remain closed.
That operational inventory is now complete and insufficient: one paired positive session versus
three required, with seven negative sessions and no false death events. Session 002 has six legacy
hard-stop frames but zero frozen-engine death events, so hard-stop is not a death label. The next
cycle may only seek additional dynamic candidates in existing train/dev video using a separately
observable second cue; do not change E1a thresholds or open test/Reward/training/device input.
The frozen two-cue consensus retains the sole known death/respawn and rejects both session-002
hard-stop rising edges. It still has only one positive session, so Reward remains closed. The next
bounded work may inspect at most 12 existing train/dev raw videos for a visible death-banner or
countdown layout before materializing clips; do not infer labels from hard-stop outside its source.
Movement is the only first-cycle learned component;
Macro/Combat remain deterministic interim components. The cycle is capped at 80 engineering
hours, 24 GPU-hours, and 50 GiB incremental artifacts, including failed runs and controls.
Do not connect a phone, reopen consumed video-test, start RL, or add input gates/transports.

For new development under this plan, bounded train/dev debugging may use new run IDs without
creating an architecture per failure. Old reports, configurations, checkpoints, test consumption,
and failure conclusions remain immutable; old frozen commands are not reopened. This supersedes
the future-work prescriptions below to create a normalization-stable candidate or permanently
block all new integration. It does not promote a failed model. Stage A supplies STOP/end-kind
contracts and a state-changing rule trajectory. Stage B supplies goal-marked RGB and selects the
686,281-parameter task-specific GroupNorm+GRU only for the next simulator pilot. Its target marker
is not real-video compatible; R2 remains closed. All four overfit diagnostic attempts are consumed;
do not run a fifth. Stage C must initialize this architecture afresh rather than load the diagnostic
checkpoint. No hidden-state inputs.
Keep the existing execution/data boundaries and dependency allowlist. Do not expand scope merely
because unused budget remains. `DELIVERY_PROGRESS.md` alone records executed progress.
Use the plan's stage A-E implementation notes and risk-tiered verification policy. Documentation
updates need diff/link/consistency checks, not model runs or full pytest. Local code changes use
focused tests; run the full suite once for a deliverable code freeze. Do not default to duplicate
agent review, per-commit historical smoke suites, or repeated approval for budgeted debug choices.

## Historical route decisions and retained boundaries

The route-specific decisions below describe preserved experiments. For future schedule, parameter
budgets and debugging policy, the current planning authority above takes precedence; historical
result and device-authorization restrictions continue to apply.

- Hierarchical Policy v0 is the active development-preparation successor governed by
  `docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md`. It keeps one RGB PolicyBundle with a shared temporal
  representation and separate Macro, Movement, and Combat heads; VisualEventEngine and the
  deterministic Router remain outside the Actor. E0 FrameBus, VisualEvent schema, and
  UnifiedTransition validation are complete offline. E1a health/death/respawn passed only its
  non-promoting engineering diagnostic after the single allowed cross-session bar-width repair;
  numeric HP accuracy, semantic accuracy, Reward, and promotion remain false. E1b terminal OCR is
  frozen failed at train/dev GAME_END coverage `0.125/0.5` and outcome coverage `0/0`; do not change
  OCR confidence, tail duration, or sampling rate. The next terminal candidate must be a separately
  versioned dynamic short-clip transition contract. E1c result-page anchor preflight passed with
  34 train and 7 dev anchors and zero conflicts, but anchor frames cannot enter a future model and
  WIN/LOSS remains unresolved. E1c-clip materialization produced 34/7 complete same-session
  triplets with zero anchor overlap and no timestamps in model inputs. These are weak transition
  candidates only. E1c-probe is frozen blocked because within-session materialization ordinal gives
  train/dev time-only macro-F1 `1.0/1.0`; no temporal, last-frame, or shuffle model may be trained on
  these labels. A new label source must identify the visual event rather than a fixed anchor offset.
  E1d crystal consensus is that separate source: after one preserved threshold repair it found
  33 train and 7 dev candidates. E1d-clip then built 33/7 event-centered pairs and reduced
  ordinal-only accuracy to about `0.424/0.429`. The CPU seed-0 probe passed overfit32 and reached
  dev temporal macro-F1 1.0 versus 0.8571 for last-frame and shuffle. It remains weak-label-only;
  WIN/LOSS and Reward stay false. The pre-test bundle is now frozen at bundle hash
  `55a679883119cdbf1a6a7703f945d61ce33408bad84013362e66355e83345c79`; the one-shot 23-session
  test contract was immutable before test decoding. The one-shot test is now consumed and frozen
  failed with `TEST_SESSION_NO_PRE_RESULT_SEQUENCE`; it did not complete metrics. Rerun, repair,
  threshold tuning, offline EventEngine integration, Reward, and promotion are prohibited.
  E1e subsequently evaluated only unused, unanchored train/dev sessions with robust per-session
  rejection. It found only 6 train and 1 dev eligible pairs versus 10/3 required and is frozen
  insufficient; it cannot replace the consumed test or reopen integration.
  P0 adapter-value testing rejected the old epoch-3 SimSiam adapter as a Hierarchical Policy
  initializer: adapter, source, and random frozen encoders all reached dev macro-F1 1.0, so the
  required source/random margins were zero. Do not promote that adapter or retune this easy probe.
  The harder P0 middle-shuffle task also rejected it: adapter temporal dev macro-F1 `0.3333`,
  source `0.4687`, and random `0.5142`. The old adapter is permanently unavailable as the new
  PolicyBundle initializer. The separately trained seed-0 temporal SSL pilot passed overfit32 and
  non-collapse checks and improved dev macro-F1 to `0.7031`, but missed its frozen `0.75` gate.
  Its sole pre-update repair only set deterministic CuBLAS workspace state; model, data, and gates
  did not change. The run is frozen failed with no encoder checkpoint. Do not retune or rerun this
  lineage, and do not initialize PolicyBundle from it. A separately versioned v2 indexed 1,648
  train-only pairs across all 103 video-train sessions and used a fixed last epoch without dev
  selection. It passed at dev macro-F1 `0.7907`; representation SHA-256 is
  `d1ce0a9c44710586e6df3124371ffa0171dc1806510efe2d9d02d5e733d1c848`. This representation may
  initialize P0 only. P1 head data and learnability gates must be frozen independently before any
  policy training. The first P1 Movement raw-video teacher audit is now frozen failed after its
  sole display-matrix repair. All 103/23 train/dev sessions were sampled, but detection coverage
  was only `0.1844/0.1898`, train detected-session fraction was `0.5922 < 0.6`, and dev stable west
  support was `4 < 16`. Do not retune this teacher, ROI, sampling, confirmation, or gates; no
  Movement Head training is authorized from this lineage. Continue with an independently frozen
  Macro Head data contract while Movement remains blocked.
  The independent P1 Macro simulator-data audit passed for the three Actor-owned intents:
  train/dev windows are `1620/412`, every intent appears in all `40/10` episodes, class-prior F1
  is `0.2019`, and time-only F1 is `0.3893`. Its report SHA-256 is
  `de7f8c4b265e69694d01210cbec49948907c389041df4792c294a90994bcea7c`. This opens one
  simulator-only Macro Head learnability run from the frozen P0 representation; it does not verify
  real-video semantics or reopen Movement, Combat, PolicyBundle assembly, Reward, or input.
  That run is now frozen failed: overfit32 reached `0.9688` accuracy but loss `0.1717 > 0.05`, and
  dev macro-F1 `0.3868` did not beat time-only `0.3893`; its gain over label-shuffle was only
  `0.0845`. No Macro Head checkpoint was written. Do not retune the canvas, head, epochs, loss, or
  gates in this lineage. P0 remains valid only for its temporal-order evidence; it is not proven to
  expose frozen simulator Macro semantics.
  The independent P1 Combat data audit is also frozen failed. Although all positive classes have
  numerical support, a 200 ms time-only lookup reaches macro-F1 `0.9331`, the eight sessions contain
  only two positive action sequences, and no artifact binds the declared Houyi identity. These are
  cooldown round-robin execution labels, not tactical choices. Do not train a Combat Head from them;
  the existing visual cooldown arbiter remains deterministic only.
  P1v2 now freezes the corrective architecture decision without reopening those failures: P0 through
  layer2 is a shared frozen trunk, while Macro and Movement receive separate trainable layer3/layer4,
  GRU, and Head branches with 8–14M parameters each. They train independently and propose at 2/10 Hz.
  Combat v0 remains the deterministic visual cooldown arbiter at 10 Hz until hero-bound, non-clock
  tactical data exists. The deterministic Router alone owns version checks, pointer concurrency,
  death, and hard stop. No branch training is opened until its new data contract passes.
  The first P1v2 Movement simulator audit is frozen failed: Global Agent train/dev contain only
  east/west move labels (`376/226` and `115/45`), with all other six directions absent and dominant
  fractions `0.6246/0.7188`. Skill aim and wait were correctly excluded. A read-only check of the
  historical V7 fit/acquisition rows found only ego-view east moves, so it cannot repair coverage.
  Do not shrink the eight-direction contract or relabel skill aim. The next source must be a new,
  balanced, visually conditioned 2D PixelArena navigation curriculum.
  That separate 2D source now passes materialization with 512/128 train/dev sequences, exactly
  64/16 groups per direction, both ego sides, zero imbalance, and zero group overlap. Its report
  SHA-256 is `f73e379dfe79ab24a7087e260e6d321f7c0e5f672f6ae2b6cbae4edea946880b`.
  It opens one task-specific Movement branch learnability run for local visible-target approach
  only; lane strategy, real-video semantics, Bundle assembly, Reward, online RL, and input stay false.
  That branch run is frozen failed after its sole overfit-order repair. Full-data dev macro-F1 and
  every direction recall were `1.0` versus label-shuffle `0.1748`, but overfit32 reached only
  `0.9375` accuracy with loss `0.1704`, missing `0.95/0.05`. No checkpoint was saved. Do not change
  batch size, steps, gates, or BatchNorm behavior in this lineage. A future norm-stable candidate
  must be separately versioned; the successful full-data metric is learnability evidence only.
  It does not authorize model-driven
  mobile input, online learning, a learned Router, continuous joystick parameters, MoE, PPO, or
  model growth. Every frozen Global Agent, Human IfO, T8, and Operation result remains immutable.
- Global Agent v1 is the frozen simulator foundation and is governed by
  `docs/GLOBAL_AGENT_V1_PROTOCOL.md`. A structured simulator rule teacher must first complete full
  games and label only `intent_id`, `target_zone_id`, and an auxiliary `scene_id`. The RGB student
  receives main/minimap/HUD sequences only. Target-zone navigation, combat modes, purchase, hero
  profiles, layout adaptation, and safety remain deterministic execution modules. Behavior
  cloning precedes simulator DAgger; PPO and mobile control remain blocked by complete-episode
  gates.
- Human IfO Bridge v1 is a completed, non-promoted representation repair governed by
  `docs/HUMAN_IFO_V1_PROTOCOL.md`. It may learn a shared Human/Sim temporal representation, train
  inverse macro dynamics only from GlobalArena truth, pseudolabel only video-train/video-dev, and
  fit one seed-0 Human-BC model with frozen-Dagger distillation. Its frozen encoder-rebind result
  is reusable evidence, not an active retraining route; scenario cards remain diagnostic-only.
  Gate A-D cannot capture from or send input to a phone. Human-dev is diagnostics-only and must be
  mechanically excluded from Human-BC; H5 is not planned and remains simulator-only.
  Gate A reads exactly one Git-ignored local cohort manifest that binds 20 train and 5 dev anonymous
  session hashes to one declared hero, role, and mode; missing, mixed, or unqualified rows stop
  before any selected training frame is opened.
- Global Agent work has a hard WIP limit: one global feature task plus one highest-frequency
  blocking failure. Every active experiment must name the targeted lexicographic episode metric:
  safety, non-timeout terminal, tower progress, stuck time, fallback rate, or win rate. Local F1
  alone is diagnostic and never opens a new route.
- The parameterized challenge curriculum is frozen after two bounded simulator-only candidates.
  One improved canonical cards while regressing complete episodes; the conservative repair removed
  the card gain and still failed disjoint parameter holdout. Do not retune curriculum weights or
  train on the fixed six cards; the original Dagger checkpoint remains selected.
- The observable-factor probe and its single authorized auxiliary representation update are frozen.
  The probe exposed missing health/base/distance factors; layer4/project fine-tuning then regressed
  full episodes to 12/20 and did not improve challenge coverage. Do not add another auxiliary run,
  unfreeze more layers, or tune factor weights. The original Dagger checkpoint is permanent for v1.
- Stage 6A is an explicitly authorized, zero-control Global Agent Shadow surface governed by
  `docs/GLOBAL_AGENT_V1_SHADOW_PROTOCOL.md`. It may bind the promoted DAgger model to an attested
  foreground self-built App and explicit V4L2 RGB node, but may only write candidate logs with
  `input_commands_sent=0`. Challenge-pack failure continues to block every input stage.

- V4: read a privacy-reviewed local recording or an explicitly selected Linux V4L2 UVC
  capture node and emit host-side JSON/terminal hypotheses. The separately bounded
  `mobile-testbed` route may capture and send bounded ADB touchscreen tap/swipe events only to
  the owner-authorized test app whose package is declared by a local private build-identity file.
- V5: train a separate causal PixelArena source teacher, then run unlabeled real-video
  SimSiam adaptation, conservative pseudo-label filtering, and one Mean Teacher round. Base
  training, validation, and model selection use no human labels.
- V6: add RGB-derived dual-hero/HUD tracking and a causal eight-frame temporal diagnostic from
  RGB only. It also uses no human labels.
- V7: add an independent Rich PixelArena identity with 2D movement and factorized skills.
- T8: turn foreground-guarded automatically executed sessions in the self-built mobile test app
  into a separate RGB-plus-executed-action behavior-cloning corpus, then train a factorized visual policy,
  validate it in Shadow, and admit only bounded test-app actions. The execution event is an
  automatically recorded supervision signal, not a manually annotated frame/action label and
  never enters V5/V6 base training.
- T8-v4 is a separate read-only visual-causality diagnostic lineage. Its first contract learns only
  `main_view_enemy_cue_visible`, `basic_attack_button_visual_enabled`,
  `skill1_button_visual_ready`, and `skill2_button_visual_ready` under one fixed layout and action
  schema. Two independent automatic teachers produce conservative consensus pseudolabels from
  the frozen 103/23 video train/dev sessions; accepted labels may enter only the masked T8-v4
  diagnostic loss. No human labels or annotation UI are permitted. Candidate actions are offline
  logs with `control_output=false`.
- T8-v5 is a separate offline ROI-isolation diagnostic after the frozen T8-v4 spatial failure. It
  reuses only T8-v4 repair-1 weak labels and the frozen adapter. Enemy, basic attack, and skill1
  must each beat time-only, wrong-ROI, and shuffle controls; skill2 is diagnostic-only. T8-v5 may
  store derived ROI features but no RGB, video, source path, human label, or device data.
- Basic-only MVP is an independent deterministic delivery route governed by
  `docs/T8_BASIC_MVP_PROTOCOL.md`. It may extract only the independently passed T8-v5 basic ROI
  component, but T8-v5's failed combined decision cannot authorize input. A passing all-dev
  offline replay may admit one five-minute zero-control Shadow. No probe contract exists before
  that Shadow passes.
- Operation Policy v1 is the separate offline second-part route governed by
  `docs/OPERATION_POLICY_V1_PROTOCOL.md`. It uses the frozen Mobile Operation Base and visual-combat
  evidence to train seed-0 inverse dynamics, then may label only video-train/video-dev and fit one
  16-frame causal movement/combat policy. Purchase and hard-stop remain deterministic. It cannot
  open video-test, Shadow, capture, or device input in its first contract.
  Its pooled and one allowed source-clock/spatial IDM runs failed; the lineage is frozen before
  pseudolabel and policy stages and may not be reopened by threshold reduction or model growth.
- Operation Direct Policy v1 is the separate executed-action check governed by
  `docs/OPERATION_DIRECT_POLICY_V1_PROTOCOL.md`. Its seed-0 movement-transition and combat gates
  failed. Automated actuator schedules must not be described as gameplay-state demonstrations;
  this route is frozen without Shadow or input.
- Operation Movement Teacher v1 is a supporting deterministic module governed by
  `docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md`. It replaces fixed patrol with the frozen
  high-resolution minimap detector, uses the existing persistent joystick executor, and trains
  only a movement head. The selected T8-v2.6 seed-1 combat model is immutable and bound by hash.
  A zero-input smoke, bounded input smoke, four-session pilot, and movement gate are mandatory
  before more collection, fusion Shadow, or model input.
  Its direction-diversity repair may select exactly four sessions from a larger automatic
  candidate pool and train one separately versioned 8-frame 2x4-spatial pilot. The nearest-target
  teacher, frozen gates, human-label prohibition, and zero model-input boundary do not change.
- Deterministic Marksman Lane Controller v1 is a separate non-learning owner-testbed route governed
  by `docs/MARKSMAN_LANE_CONTROLLER_PROTOCOL.md`. It may use only the verified side, frozen
  marksman opener, eight-second lane advance/four-second hold cycle, existing visual combat and
  purchase rules, and death-triggered opener replay. It produces no movement training target.
- Adaptive Layout and Hero Profiles v1 is governed by
  `docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md`. Device geometry is icon-independent and local;
  hero profiles define only fixed-slot behavior. Until a local profile and adaptive-layout hash
  pass read-only calibration, skills are disabled and no execution route may infer a hero from a
  variable skill icon.
- A future, separately authorized post-training phase may use only owner gameplay-quality
  preferences over completed PixelArena games. It is not part of V5/V6 base training.
- Preserve V1/V2/V3 and the offline V4 route as frozen regressions. Never overwrite their
  configs, renderers, models, hashes, schemas, or run evidence.

## Non-negotiable boundaries

- Automatic closed-loop actions run only inside project-owned PixelArena or the owner-authorized
  self-built mobile test app declared by the local private identity file, through
  `mobile_testbed.py`.
- A matching Android package name is not proof that the foreground app is the self-built test app.
  Before any further mobile input, the installed artifact's project ownership and build identity
  must be independently closed by a frozen identity record. Until that record exists, the mobile
  surface is read-only and may be used only for capture/diagnosis; package-name, serial, foreground,
  and display checks alone do not authorize input.
- The mobile-testbed route requires an explicit authorized USB serial, a bounded duration and
  action count, a local private build identity, and a versioned external layout. The public tree
  contains templates only; absent or invalid local identity/layout data keeps all input disabled.
  Its ADB touchscreen input chain is gated on the configured foreground package; no bypass
  condition, override, fallback, or step is documented or permitted. Runtime probes record no raw frames. Its
  demonstrator-capture mode may write only normalized derived RGB tensors paired with
  timestamped execution events below `HOK_LARGE_ROOT`; it never writes source locators or full
  phone-video copies. It may use an explicit local V4L2 loopback fed by scrcpy in `--no-control`
  mode for continuous USB video.
- A separately versioned T8-v2 demonstrator may let the owner operate the self-built app directly
  on its virtual controls while a narrowly allowlisted, read-only `adb shell getevent` process
  observes touchscreen down/move/up events. It requires the same explicit serial, foreground
  package, display, duration, and immediate-stop gates; it may not use root, auto-select an input
  device, retain a raw event dump/device path, or claim that an observed touch was internally
  accepted by the app. Persisted labels are standardized `observed_touch_action` records only.
- If that read-only probe is unavailable, the permitted fallback is a focused host keyboard
  keydown/keyup adapter that sends actions through the existing guarded ADB tap/swipe executor and
  records only actions actually dispatched. Global keyboard hooks, scrcpy control, HID/UHID,
  uinput, minitouch, and visually inferred pseudo-actions are not T8-v2 demonstration sources.
- The separately versioned T8-v2.1 live demonstrator may replace the V4L2-plus-ADB fallback with
  one pinned scrcpy 1.25 video/control session. It is confined to `mobile-demonstrate-keyboard-v2-live`,
  the explicit serial, the exact foreground package and display identity, the external layout,
  and a focused Tk keydown/keyup window. Its two pointer IDs are fixed to joystick and combat;
  it exposes no generic shell, arbitrary server, global hook, model-driven action, or other package.
  Every run has an explicit positive duration; an optional action cap is an emergency stop, not a
  dataset-size ceiling. Historical T8-v2 data and evidence may not mix with this lineage.
  Fixed control-smoke and inverse-probe schedules are non-formal diagnostics with
  `training_eligible=false`; inverse-probe output may calibrate retrospective video candidates but
  cannot enter a policy split or authorize model-driven input.
- Formal T8-v2-auto data uses only actions dispatched through the existing serial,
  foreground-package, display, duration, rate, and action-count guards. It stores 16-frame causal
  windows ending at least 100 ms before each execution event and exactly twelve accepted five-minute
  sessions. Direct-touch artifacts remain diagnostic-only and cannot mix with this lineage.
- T8-v2.5 is a separate RGB-conditioned collection lineage. It may load only the frozen numeric
  calibration from the accepted offline v2.3 teacher report, never a checkpoint, and may choose
  only `wait`, `basic_attack`, `skill1`, or `skill2` from current and two-second-history RGB.
  Non-wait decisions execute only after a 100 ms causal delay through the existing guarded ADB
  executor; waits are explicit no-input decisions. Admission is staged as read-only dry-run,
  20 actions, one minute, then five minutes. Its data and hashes never mix with v1-v2.4.
- T8-v2.6 may run at most one separately versioned 60-second/20-action trained-model probe only
  after its selected-model, split, offline-test, replay-Shadow, and layout identities all pass the
  dedicated admission function. The executable vocabulary is exactly basic attack, skill1, and
  skill2 taps. Movement, skill3, aiming, holding, target intent, fallback actions, threshold changes,
  and evidence substitution are forbidden. A 500 ms action interval, three-identical-action cap,
  background guard watchdog, per-send synchronous guard, and five-second zero-input scene-readiness
  gate are mandatory.
- T8-v2.7 is frozen failed. Its three immutable reports and failure-freeze manifest must be kept;
  repeated collection, threshold changes, and further four-class head training are forbidden.
- T8-v3 is a separate video-state lineage. Its RGB-only output vocabulary is exactly
  `enemy_visible`, `attack_opportunity`, `basic_ready`, `skill1_ready`, `skill2_ready`, plus
  confidence/abstain. A deterministic execution boundary, not the model, owns skill priority,
  cooldowns, and repetition limits. Only video-train may fit seed 0 and video-dev may select it;
  video-test stays unopened. Offline replay, five-minute zero-control Shadow, and the 20-action,
  one-minute, and five-minute input stages are strictly ordered. A failed gate blocks every later
  stage without threshold changes, retries, or evidence substitution.
- T8-v4 does not modify T8-v3. It omits learned `attack_opportunity`, derives only
  offline `candidate_basic_attack`, `candidate_skill1`, and `candidate_skill2` logs, and excludes
  skill3, target range, attackability, safety, and
  tactical intent. It must pass label-validity, visual-learnability, temporal-necessity, and
  evidence-selectivity gates in order. Its first cycle cannot open offline replay, Shadow, or any
  device-input stage. The authoritative protocol is `docs/T8_V4_PROTOCOL.md`.
- T8-v5 does not reopen T8-v4. Its first gate is single-frame ROI evidence only and excludes TCN,
  replay, Shadow, mobile capture, and input. A TCN value test is permitted only after all three
  formal ROI heads pass. The authority is `docs/T8_V5_ROI_PROTOCOL.md`.
- Basic-only MVP permits only `wait` and an offline/read-only `candidate_basic_attack`. It uses the
  frozen enemy rule plus the passed basic ROI component with fixed persistence and rate limits.
  Offline replay and read-only Shadow both retain `device_input_allowed=false`. Its frozen Shadow
  produced zero candidates, so repeat Shadow, threshold changes, probe, and control are closed.
- The separate Basic rule engineering fallback used no learned head. Its one capture-domain
  calibration initially assumed a visual release between taps; owner observation established that
  basic has no cooldown dimming. The corrected private execution point passed 20-action,
  one-minute, and five-minute bounded runs while retaining the original visual ROI.
- The synchronous combat probe is confined to acknowledged numeric taps for basic attack and the
  three skills in the owner testbed. Two 60-second repeats passed with five actions per button.
  It cannot send movement, swipe/aim, target selection, arbitrary shell input, or model decisions.
- The visual combat arbiter is governed by `docs/VISUAL_COMBAT_ARBITER_PROTOCOL.md`. It may select
  only the four fixed combat taps by cooldown-aware round-robin and must synchronously acknowledge
  every send. Its 60-second and five-minute gates passed. It cannot add movement, aim, target
  selection, enemy semantics, arbitrary commands, or model decisions without a new contract.
- Visual combat event data requires actual elapsed timestamps, 16-frame derived RGB or frozen
  encoder features, anonymous sessions, synchronous action binding, and at least twelve sessions
  before an 8/2/2 split. The two initial event-only diagnostic sessions cannot train a model.
- Mobile Operation Base v1 is governed by `docs/MOBILE_OPERATION_BASE_PROTOCOL.md`. It is a new,
  explicitly authorized owner-testbed route through `mobile-operation-base`, using the pinned
  scrcpy 1.25 server, pointer 0 for persistent joystick movement, and pointer 1 for combat or the
  single recommended purchase. Its 60-second, five-minute, and live death-stop gates passed. It
  cannot add enemy semantics, target selection, aiming, tactics, another package, or another input
  transport without a new contract.
- Operation Policy v1 does not modify any frozen T8 lineage. Its only learned outputs are
  nine-class movement and five-class combat against automatic inverse-dynamics targets. Every
  first-contract artifact keeps `semantic_accuracy_verified=false`, `promotion_allowed=false`,
  `control_output=false`, and `device_input_allowed=false`; a pass can request only a separately
  reviewed read-only Shadow contract.
- Never target an unapproved client or account; outside the pinned T8-v2.1 demonstrator and
  Mobile Operation Base v1,
  never add scrcpy control, Accessibility, root,
  hooks, injection, memory/process inspection, protocol
  interception, evasion, a generic shell runner, or network capture input. The input-control
  chain is confined to the mobile-testbed route and the exact locally attested package; it is never
  added to any other client or account.
- Live capture accepts only an explicit, non-symlink `/dev/videoN` Linux V4L2 character
  device. Numeric camera indexes, other device nodes, URIs, and network streams fail closed.
- V5/V6 base training, validation, model selection, and diagnostic evaluation use no human
  action, frame, HUD, tracking, or temporal labels. Do not collect them for these stages.
- The only permitted human label classes are an owner judgment of completed gameplay quality:
  `A_BETTER`, `B_BETTER`, `TIE`, or `UNJUDGEABLE` for a pair of complete, read-only
  PixelArena games. It may be used only in an explicitly authorized, versioned post-training
  phase after the base model is frozen, and may train only a separately versioned post-training
  descendant. It is never a per-action target and never unlocks client control.
- RGB Actors receive RGB only. T8 touch, keyboard, or v2.5 rule decisions are supervision targets during offline
  training and never actor inputs at training or inference. Legal actions and structured state may be used by a
  PixelArena teacher or execution boundary, never by an encoder, temporal hidden state, or
  commercial-domain input.
- Hierarchical Policy episode metadata selects a frozen configuration or hero adapter before an
  episode; it does not enter the RGB Actor tensor or hidden state. Visual events, reward components,
  action masks, executed-action acknowledgements, and action timestamps likewise remain outside
  Actor inputs. The EventEngine is frozen and hash-bound during each Policy training batch, and the
  Router remains deterministic.
- Do not copy code, weights, action maps, coordinates, assets, recordings, or device setup
  from the three reference repositories.

## Dependency and module allowlist

- Base PixelArena imports remain free of Torch, torchvision, safetensors, PyAV, OpenCV,
  Tk, and device APIs; CLI imports optional stages lazily.
- Torch/torchvision/safetensors are allowed only in `bc.py`, `pixel.py`, `alignment.py`,
  `temporal.py`, `v6_zero.py`, `rich_pixel.py`, `t8.py`, `t8_v3.py`, `t8_v4.py`, `t8_v5.py`,
  `t8_basic_mvp.py`, `t8_shadow.py`, `operation_policy.py`, and their
  focused tests, plus `global_policy.py`, `human_ifo.py`, `human_inverse.py`, and their focused tests
  for offline Global Agent BC, Human IfO, inverse macro dynamics, single DAgger round, video
  adaptation, and zero-control replay.
  `hierarchical_e1d_probe.py`, `hierarchical_e1d_checkpoint.py`, and
  `hierarchical_e1d_test.py`, and `hierarchical_e1e_unused.py` may use Torch only for offline E1d
  diagnostics, checkpoint freeze, and evaluation; evaluation modules contain no optimizer or backward path.
  `hierarchical_p0.py` may use frozen Torch/torchvision encoders and train only its small probe head.
  `hierarchical_p0_temporal_probe.py` has the same frozen-encoder boundary.
  `global_shadow.py` and `test_global_shadow.py` may use the same frozen Global Agent model only for
  authorized zero-control Shadow; they may never add an input sender.
- PyAV is allowed only in `shadow.py`, `capture.py`, `alignment.py`, `pre_ingest.py`,
  `v5_data.py`, `mobile_testbed.py`, `hierarchical_e1_terminal.py`,
  `hierarchical_e1c_anchor.py`, and focused tests. The E1 terminal modules may decode only their
  frozen train/dev offline samples; they cannot open video-test, a live capture source, or any
  input surface.
  `movement_real_rgb.py` may decode only the two cohort-bound landscape train/dev sources
  selected by `native-player-pilot`, crop before resizing, and persist derived offline QA windows.
  It cannot decode test, connect a device, create action labels or train from these windows.
- No annotation UI is an active V5/V6 surface. The T8 calibration picker may use Tk only to pick
  in-memory layout coordinates for the owner-authorized self-built test app; it never writes a
  screenshot or creates training labels. `movement_real_rgb.py` may use Pillow only to write
  non-interactive, offline QA contact sheets under `HOK_LARGE_ROOT`; it exposes no annotation UI
  and creates no training label. A future Tk/Pillow preference UI, if authorized,
  may be offline only and may show complete read-only PixelArena game pairs for quality ranking.
  It
  must never open a capture node, collect action/frame labels, or feed commercial-client data
  into a control path.
- CUDA is a formal V3/V5/V6/V7 acceptance surface; CPU CI smoke is non-promoting. A sandbox
  GPU probe failure is not evidence that the host RTX 4090 is unavailable.

## Mechanical gates

- Project/Python file counts and nonblank Python lines are reported for observability only. They
  have no pass/fail ceiling. Dataset bytes and session counts likewise have no global ceiling.
- Exactly four root Markdown authority files.
- Verification follows section 9 of `docs/ENGINEERING_CONVERGENCE_PLAN.md`: docs-only changes use
  diff/link/consistency checks; local code uses affected tests and Ruff; cross-module/schema or
  dependency changes add relevant integration tests, strict mypy and safety checks. At deliverable
  code freeze run `make check` plus `git diff --check` once. Process commits do not independently
  require full pytest or all historical smoke targets. Reuse unchanged-code results, never invent them.
- Never manufacture GPU, hardware, recording, preference, or accuracy evidence. Missing UVC
  hardware is `READY_FOR_HARDWARE`; missing recordings block only real-domain work, while
  missing future preference data blocks only its separately authorized post-training phase.
- Put every new large dataset, derived frame shard, training cache, checkpoint, and formal
  training run under `HOK_LARGE_ROOT` (default: `$HOK_LARGE_ROOT`).
  Keep raw recordings in their existing external location, never duplicate them into the
  repository, and never persist their source locators in manifests. Do not replace strict
  regular-file inputs with repository symlinks. Existing hash-bound frozen `runs/` evidence
  stays in place unless an explicit verified migration is separately authorized.
- Before a real-domain shard or manifest is accepted, require four separately descriptor-bound
  local artifacts: descriptor-identified pre-ingest component evidence, the explicit owner declaration,
  a fixed zero-redaction/rotation privacy context, and the mechanically derived clean-component
  split record. Owner declarations are operator-provided attestations, not independent proof
  artifacts. `zero-redaction` means no pixel masking, and is not a visual anonymity claim.
  A pre-ingest result is diagnostic only; it never authorizes training, release,
  or advice by itself.

- Current active V5 pre-ingest contract is file-atomic. Each complete MP4 is one component, and pre-ingest runs only parallel regular-file and integrity-identity checks (descriptor/stat checks, regular-file validity, and metadata/container parse checks). It does not perform automatic duplicate, re-encode, overlap, or near-similarity scanning and does not claim independent duplicate-game proof.
