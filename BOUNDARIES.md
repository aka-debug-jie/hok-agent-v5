# Boundaries

## Allowed execution surfaces

Mobile input additionally requires a local, Git-ignored project-owned build identity, not merely a
matching package name. `HOK_MOBILE_IDENTITY_PATH` defaults to
`configs/mobile_testbed_identity.local.json`; every new device session must match its package,
version, and signature before bounded input is available. The public example is deliberately
invalid and cannot admit input. Read-only V4L2 capture and offline analysis do not depend on that
input admission.

- Deterministic structured and RGB closed loops inside project-owned PixelArena V1 and the
  independently versioned Rich PixelArena V2.
- Offline analysis of a non-symlink regular local recording.
- Read-only capture from one explicitly named, non-symlink `/dev/videoN` Linux V4L2
  character device through PyAV. Capture is latest-frame-only, bounded, local, and emits
  terminal/JSON diagnostics on the host.
- Bounded screen capture and bounded ADB touchscreen tap/swipe input for the owner-authorized self-built mobile test app
  declared by the local private build-identity file, through one explicit USB serial and
  `mobile_testbed.py`. Every run has a finite duration and action cap; screen frames stay in
  memory and events contain hashes only. The chain is gated on the foreground package being
  the configured one and on the explicit authorized serial; it may use only one explicit local
  V4L2 loopback from no-control scrcpy for continuous USB video. No bypass condition or step
  for the locally attested package gate is documented or permitted.
- A separately versioned T8 demonstrator-capture route for that same self-built test app. It may
  automatically pair normalized RGB tensors with the timestamped input events actually issued by
  the owner during a bounded session. These are execution logs, not human frame/action
  annotations; they train only the T8 behavior-cloning descendant, never V5/V6.
- A T8-v2 direct-touch demonstrator may instead observe the owner's physical interaction with the
  app's virtual controls through one narrowly allowlisted, read-only `adb shell getevent` process.
  It runs only for the explicit serial while the locally attested package and frozen display identity are
  active. It may standardize down/move/up coordinates into `observed_touch_action` factors, but it
  cannot call a game-internal API or claim that the app internally accepted the touch.
- Formal T8-v2-auto training accepts `executed_action` sessions produced only by the bounded,
  foreground-guarded ADB executor. Direct-touch and interactive-keyboard artifacts remain
  diagnostic-only; source mixing, raw video, raw touch events, and device paths remain excluded.
- T8-v2 may persist one derived RGB frame and its current standardized action state per bounded
  sample, plus sparse semantic transition events. The offline loader alone reconstructs the
  16-frame causal RGB window; raw touch reports and source frame paths remain excluded.
- T8-v2.1 may use one exact-hash scrcpy 1.25 server session for both H.264 RGB and two-pointer
  control, only through `mobile-demonstrate-keyboard-v2-live`. A focused Tk window supplies
  keydown/keyup state; a 100 ms device watchdog and 500 ms maximum snapshot age keep serial,
  foreground package, and display identity active. The command has an explicit finite duration,
  writes streaming derived-frame shards, and cannot load a model or select another package.
  Its optional fixed 20-second diagnostic sequence writes source
  `bounded_scrcpy_control_smoke_v1`, is always non-formal and `training_eligible=false`, and may
  validate only transport/lifecycle behavior. Formal data accepts focused human key events only.
  The separate `bounded_scrcpy_inverse_probe_v1` source is also non-formal and
  `training_eligible=false`; it may persist only derived before/after RGB and dispatched combat
  factors for offline inverse-dynamics calibration. It cannot enter a policy split or authorize
  model-driven input.
- T8-v2.5 may run one separately versioned deterministic RGB rule teacher through no-control
  V4L2 capture and the guarded ADB executor. It loads frozen numeric activity/ROI calibration
  only, never a model checkpoint, and selects only `wait`, `basic_attack`, `skill1`, or `skill2`.
  A non-wait decision is training-eligible only when its serial/package/display guard is rechecked
  and dispatch occurs at least 100 ms after the bound RGB observation. It progresses only through
  a zero-input dry-run, 20-action probe, one-minute run, and five-minute session.
- T8-v2.6 trained-model input is confined to one separately reviewed 60-second, 20-action probe
  after the immutable three-seed selection, sealed offline evaluation, and sealed five-minute
  replay Shadow all pass and bind the same model, split, and layout hashes. The probe may tap only
  `basic_attack`, `skill1`, or `skill2`; movement, skill3, aiming, holding, and target selection are
  unavailable. It requires four consecutive identical predictions, confidence `>=0.45`, normalized
  predictive entropy `<=0.80`, at least 500 ms between taps, and no more than three consecutive
  identical taps. The foreground/display watchdog and guarded sender remain mandatory, and any
  changed identity, stale evidence/frame, missing coordinate, or disconnection stops the run.
  Before any tap, a five-second zero-input warmup must observe one stable admitted non-wait combat
  candidate; an empty/static scene stops without opening the action phase.
- T8-v2.7 is a preserved failed lineage. Its reports may be verified but not overwritten, and its
  four-class action head may not be retrained or admitted by changing thresholds.
- T8-v3 may train only the five RGB-observable state heads named in its frozen contract from the
  103 video-train sessions, with the 23 video-dev sessions used only for selection and video-test
  unopened. Confidence/abstain is model output; skill priority, cooldown, action frequency, and
  consecutive-action limits live only in the deterministic execution boundary. A passing seed-0
  report is required before offline replay; a passing replay is required before five-minute
  zero-control Shadow; that Shadow is required before the 20-action, one-minute, and five-minute
  input stages. Failure at any gate prohibits all later stages and sends no fallback action.
- T8-v4 is a new read-only lineage governed by `docs/T8_V4_PROTOCOL.md`. It learns four local
  visual-cue heads under one fixed layout and action schema. Two independent automatic teachers
  may produce confidence- and perturbation-gated pseudolabels from the frozen 103/23 train/dev
  sessions. Accepted labels enter only the masked T8-v4 diagnostic loss; no human labels,
  annotation UI, or video-test access are allowed. Derived candidate actions are offline logs with
  `control_output=false`; replay, Shadow, and device input remain closed.
- T8-v5 is an offline ROI-isolation lineage governed by `docs/T8_V5_ROI_PROTOCOL.md`. It may reuse
  only the frozen T8-v4 repair-1 weak targets and frozen adapter. Its first gate uses single-frame
  correct-ROI, wrong-ROI, time-only, prior, and shuffle controls. Skill2 is diagnostic-only. It
  stores derived ROI features only and cannot open TCN, replay, Shadow, capture, or input unless a
  later contract is explicitly admitted after all three formal ROI heads pass.
- Basic-only MVP is a separate deterministic component route governed by
  `docs/T8_BASIC_MVP_PROTOCOL.md`. It may extract the passed basic ROI head without promoting the
  failed T8-v5 model. A hash-bound passing offline replay may admit one five-minute read-only
  Shadow through explicit serial and V4L2 capture. Both stages send zero input. The frozen Shadow
  produced zero candidates and failed, so no repeat, threshold change, 20-action probe, one-minute
  run, or five-minute control contract is permitted.
- The Basic rule engineering fallback is separately frozen. Its v2 probe required the basic ROI to
  fall below threshold after every tap before rearming. Owner observation established that basic
  attack has no cooldown dimming, so a private execution point was separated from the visual ROI.
  The bounded 20-action, one-minute, and five-minute basic runs passed without unexpected actions.
- The synchronous combat probe accepts only acknowledged `input touchscreen tap` commands for the
  four fixed combat buttons in the owner testbed. It uses a Git-ignored execution layout and may
  not expose coordinates. Movement, swipe/aim, target selection, arbitrary commands, and model
  decisions are forbidden. Two 60-second, five-per-button repeats are the frozen maximum evidence.
- The visual combat arbiter may use only cooldown-aware round-robin selection among the same four
  fixed taps. Skills require a confirmed cooldown and recovery before rearming; basic attack uses
  only the global interval. Its frozen evidence is one 60-second and one five-minute run. Movement,
  aim/swipe, target selection, enemy interpretation, arbitrary commands, and model control remain
  forbidden.
- Arbiter event logs alone are not a training dataset. Training requires twelve separately frozen
  anonymous sessions with actual elapsed timestamps and derived RGB or frozen-encoder features;
  raw video, source paths, serials, coordinates, and pre-threshold training remain forbidden.
- Mobile Operation Base v1 is a separately authorized exception to the older scrcpy-control
  restriction, confined to the `mobile-operation-base` command, exact self-built App identity,
  pinned scrcpy 1.25 server, private observation/execution layouts, and one finite session. Pointer
  0 owns only persistent joystick movement; pointer 1 owns only the four combat taps or the single
  recommended-purchase tap. Death/respawn/ended or unknown screens force both actions to stop.
  Enemy semantics, target selection, aiming, arbitrary coordinates/commands, other packages, and
  online model control remain forbidden.
- Operation Policy v1 is offline-only and governed by
  `docs/OPERATION_POLICY_V1_PROTOCOL.md`. It may train inverse-dynamics movement/combat heads from
  the frozen Operation Base and visual-combat evidence, apply admitted heads to the 103/23
  video-train/dev splits, and train one seed-0 causal policy against accepted automatic targets.
  It may not open video-test or connect to capture/device input. Purchase and hard-stop remain in
  the deterministic Mobile Operation Base rather than learned outputs.
  Its repaired spatial IDM failed the frozen movement/combat gates, so pseudolabel, policy,
  Shadow, capture, and input stages remain closed.
- Operation Direct Policy v1 may consume only the already frozen executed-action sessions for one
  offline seed-0 learnability check. Its transition and combat gates failed, so it is frozen and
  cannot open Shadow, capture, or device input.
- Operation Movement Teacher v1 may use the Mobile Operation Base two-pointer transport only in
  the owner-attested self-built App. Its minimap decision has no fixed/random fallback; missing
  evidence becomes a bounded hold then `wait`. Zero-input smoke precedes input smoke, and input
  smoke precedes four automatic five-minute pilot sessions. The route cannot load a movement model
  or open fusion Shadow before the frozen pilot gate passes.
- Adaptive device layouts and hero profiles are local, Git-ignored control prerequisites. Layout
  calibration may run read-only; skill execution requires a configured hero profile, stable
  content box, per-group confidence, and identity-bound hash. Unknown heroes, unknown skill modes,
  icon-only hero inference, and layout drift must disable skills rather than guess coordinates.
- Offline synthetic training, unlabeled real-video representation learning, conservative
  pseudo-label research, one Mean Teacher round, and RGB-derived tracking/temporal diagnostics.
- A future, separately authorized post-training phase may use owner gameplay-quality
  preferences over complete PixelArena game pairs. It cannot use action, frame, HUD,
  tracking, or temporal labels, and it cannot operate on a live commercial-client
  feedback loop.

## Never part of the executable surface

- Treating package name, USB serial, foreground state, or display geometry as proof that an app is
  the project-owned self-built test app. Those checks remain necessary but are not sufficient.
- Any input to an unapproved client or account; scrcpy control outside the pinned T8-v2.1
  demonstrator and Mobile Operation Base v1, Accessibility, macros, mechanical input, or account
  automation.
- The ADB touchscreen input chain is confined to the owner-authorized self-built test app declared
  by the local private identity file, through `mobile_testbed.py`. The configured-package gate
  admits no bypass: no condition, override, fallback, or step that routes the chain to any other
  package, client, or account is documented or permitted.
- Root, hooks, injection, process/memory access, protocol interception/change, anti-cheat
  detection/evasion, automated real matches, or any attempt to conceal automation.
- Persistent raw touchscreen event dumps, touchscreen device paths, automatic input-device
  selection, unrestricted `getevent`, global keyboard hooks, or touch labels inferred from RGB.
- Numeric camera indexes, arbitrary device nodes, symlinked devices, URLs, RTSP/TCP/UDP or
  other network streams, generic shell commands, device enumeration, or automatic source
  selection.
- Online learning from a commercial client, reward adaptation, real-client policy
  promotion, or mapping Rich PixelArena skills to a real-client control surface.
- A trained checkpoint, learned policy, generic vision model, or mutable online rule choosing
  T8-v2.5 collection actions. The v2.5 collector is one frozen deterministic rule contract;
  trained-model input remains blocked until offline, Shadow, and separately reviewed probe gates.
- GameCore assumptions, license probing, unknown binaries/weights, or credential storage.

## Data, Actor, and output boundary

V4 video is read-only. The owner-authorized mobile-testbed runtime may additionally send bounded
ADB touchscreen tap/swipe events only to its
explicit self-built test-app serial with the package declared by the local private build-identity file. Runtime probes
write only bounded JSONL
events and a summary; they never store raw frames, thumbnails, audio, or video. The T8
demonstrator-capture route may instead persist normalized training tensors plus standardized
`executed_action` or `observed_touch_action`
records under `HOK_LARGE_ROOT`; it cannot retain full source frames, audio, video, source paths,
raw touchscreen events/device paths, account identifiers, rewards, legal masks, structured state,
truth, or privileged fields. An observed physical touch proves only the standardized touch
gesture, not internal app acceptance. For the
authorized zero-redaction route,
rotation-normalized 128x128 derived frames may be stored only in Git-ignored,
session-partitioned NPZ shards after an owner-attested privacy context and mechanically derived
component record.
“Zero redaction” means no pixels are blacked out; it is not a visual-anonymity claim. Owner
attestations are explicit declarations by the operator, not independent verification proofs.
For active V5 pre-ingest, this record is file-atomic at the MP4 level: each complete MP4 is one
pre-ingest component. The pre-check does parallel regular-file and integrity identity only; it does not
perform duplicate/re-encode/overlap/near-similarity reconstruction scans and does not claim to
prove independent repeated gameplay or duplicate-game evidence.
Original recordings stay outside the repository. Persisted real-domain records
must not contain paths, account identifiers, legal masks, rewards, structured state, truth,
or privileged fields.

New large derived datasets, caches, checkpoints, future preference-review media, and formal
training runs live under `HOK_LARGE_ROOT`; the public default is
`.local-data/hok-agent-v5`. Manifests may contain only artifact basenames and
anonymous hashes, never raw-video locators. Repository symlinks are not a substitute for
strict regular-file loading. Existing frozen local run evidence is preserved in place.
There is no administrative maximum for repository code size, dataset bytes, or session count.
Per-run duration, optional emergency action caps, bounded shard size, memory queues, concurrency,
storage availability, and immutable frozen-manifest rules remain safety and reproducibility gates.

V3/V5/V6 and T8 Actors accept RGB tensors or RGB sequences only. T8 demonstration events are
offline supervision targets, never actor inputs. Tracking values used by V6 must
be produced internally from RGB. A caller cannot supply legal actions, structured state,
teacher identity, reward, truth, account, or device state. PixelArena legal domains are
transient teacher/execution-boundary data and never enter an encoder or hidden state.

V5/V6 base data, training, validation, model selection, and diagnostics contain no manually
provided action, frame, HUD, tracking, or temporal labels. T8 may consume its own automatically
recorded self-built-test-app dispatched actions or standardized observed touches as
behavior-cloning targets, but it remains a
  separate dataset, model lineage, evaluation suite, and action boundary. T8-v4 may additionally
hold a separately versioned, path-free dual-teacher weak route producing three-value visual-cue
outputs and frame-interpretability diagnostics. It is evaluation-only: it cannot train or tune a
model, seed pseudo-labels, supervise an action, or release control. The other permitted human
annotation is an owner judgment of which complete PixelArena game is better (or `TIE`/`UNJUDGEABLE`);
it must be stored as a separate, versioned post-training preference artifact and is never an action
target.
It may train only a separately versioned offline descendant for PixelArena quality or read-only
advice calibration.

Until a separately authorized post-training contract is implemented and independently accepted,
every real-domain row has `advisory_action=ABSTAIN` and `control_output=false`. A preference
record alone cannot release an advice class, cannot tune a base V5/V6 model, and can never become
an executable client action.

## Claim boundary

V7 may prove only that an RGB factorized policy completes the fixed project-owned Rich
PixelArena task. V4–V6 may prove only read-only decoding and zero-label domain-alignment/
temporal diagnostics. T8 may eventually prove only bounded execution in the owner-authorized
self-built test app under its frozen layout and evaluation contract; it has no trained-policy or
gameplay-quality claim yet. Future preference post-training remains a separate offline claim scope.
Nothing in this repository establishes Honor of Kings skill, GameCore equivalence, suitability
for any unapproved client, tactical optimality, or transfer outside the fixed abstract vocabulary.
