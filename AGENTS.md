# AGENTS.md

`DELIVERY_PROGRESS.md` is the only current-state ledger. Read it, `README.md`, and
`BOUNDARIES.md` before changing this repository.

## Active route

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
- Operation Movement Teacher v1 is the active modular route governed by
  `docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md`. It replaces fixed patrol with the frozen
  high-resolution minimap detector, uses the existing persistent joystick executor, and trains
  only a movement head. The selected T8-v2.6 seed-1 combat model is immutable and bound by hash.
  A zero-input smoke, bounded input smoke, four-session pilot, and movement gate are mandatory
  before more collection, fusion Shadow, or model input.
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
- Do not copy code, weights, action maps, coordinates, assets, recordings, or device setup
  from the three reference repositories.

## Dependency and module allowlist

- Base PixelArena imports remain free of Torch, torchvision, safetensors, PyAV, OpenCV,
  Tk, and device APIs; CLI imports optional stages lazily.
- Torch/torchvision/safetensors are allowed only in `bc.py`, `pixel.py`, `alignment.py`,
  `temporal.py`, `v6_zero.py`, `rich_pixel.py`, `t8.py`, `t8_v3.py`, `t8_v4.py`, `t8_v5.py`,
  `t8_basic_mvp.py`, `t8_shadow.py`, `operation_policy.py`, and their
  focused tests.
- PyAV is allowed only in `shadow.py`, `capture.py`, `alignment.py`, `pre_ingest.py`,
  `v5_data.py`, `mobile_testbed.py`, and focused tests.
- No annotation UI is an active V5/V6 surface. The T8 calibration picker may use Tk only to pick
  in-memory layout coordinates for the owner-authorized self-built test app; it never writes a
  screenshot or creates training labels. A future Tk/Pillow preference UI, if authorized,
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
- Before each commit run Ruff, strict mypy, full pytest, the project size/safety gate,
  relevant frozen regression gates, the stage CPU smoke, and `git diff --check`.
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
