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
- Never target an unapproved client or account; outside the single pinned T8-v2.1 demonstrator,
  never add scrcpy control, Accessibility, root,
  hooks, injection, memory/process inspection, protocol
  interception, evasion, a generic shell runner, or network capture input. The input-control
  chain is confined to the mobile-testbed route and the exact locally attested package; it is never
  added to any other client or account.
- Live capture accepts only an explicit, non-symlink `/dev/videoN` Linux V4L2 character
  device. Numeric camera indexes, other device nodes, URIs, and network streams fail closed.
- V5/V6 base training, validation, model selection, and diagnostic evaluation use no human
  action, frame, HUD, tracking, or temporal labels. Do not collect them for these stages.
- The only permitted future human label is an owner judgment of completed gameplay quality:
  `A_BETTER`, `B_BETTER`, `TIE`, or `UNJUDGEABLE` for a pair of complete, read-only
  PixelArena games. It may be used only in an explicitly authorized, versioned post-training
  phase after
  the base model is frozen, and may train only a separately versioned post-training descendant.
  It is never a per-action target and never unlocks client control.
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
  `temporal.py`, `v6_zero.py`, `rich_pixel.py`, `t8.py`, `t8_v3.py`, `t8_shadow.py`, and their
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
