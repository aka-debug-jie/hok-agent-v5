# Delivery Progress

- Last update: 2026-08-15
- Current task: `T8-V2-VIDEO-ADAPTED-LOWERED-PERFORMANCE-BC`
- Status: `T8_V2_PILOT_AND_STRICT_CAUSAL_VIDEO_COMBAT_DIAGNOSTIC_FAILED`
- Product Actor input: `RGB_ONLY`
- Closed-loop environment: `PIXELARENA_ONLY`
- Owner-authorized mobile-testbed action output: `IMPLEMENTED_BOUNDED_ADB_TOUCHSCREEN_ONLY`
  (foreground-package/display-guarded tap/swipe for the locally attested owner testbed package)
- HoK capability claim: `false`
- GameCore equivalence claim: `false`

## Frozen delivered baselines

- Minimal V1 deterministic PixelArena lifecycle/replay: `COMPLETE`
- Minimal V2 550-parameter structured BC: `COMPLETE`, secondary baseline only
- Minimal V3 six-class RGB PixelArena BC: `COMPLETE`, frozen
- Minimal V4 offline local-recording Shadow: `COMPLETE`, always abstains

Frozen V3 evidence remains in Git-ignored `runs/pixel-v3-v1`: three training seeds passed;
promoted seed 0 model SHA-256 is
`df511e9b19327886da359400055dcc99aad6520a495c6d5e0495031c86b44eed`; promoted RTX
4090 batch-1 FP32 forward p95 was 1.932 ms. These numbers describe only fixed PixelArena.

Frozen V4 offline evidence: 45 tests passed at delivery; generated local MP4 analysis
produced 3/3 `ABSTAIN` rows and zero advice/control outputs. This did not use a phone,
commercial recording, capture card, or live device.

## Active V4–V7 state

| stage | implementation | formal external gate | current disposition |
|---|---|---|---|
| V4 live Shadow | `FRAMEWORK_IMPLEMENTED` | 10-minute 1080p60 UVC / 10 Hz run | `READY_FOR_HARDWARE` |
| T8 mobile testbed | `V1_FAILED_PRESERVED_V2_CAUSAL_POLICY_FAILED_V24_ONSET_AUDIT_FAILED` | visible-onset correction is not the missing causal signal; keep RGB teachers offline and obtain a genuinely event-bound action source | `ONSET_GATE_FAILED_NO_GPU_PILOT` |
| V5 Visual Alignment | `MANIFEST_V2_PRE_INGEST_SOURCE_PRODUCER_NON_PROMOTING_FRAMEWORK_IMPLEMENTED` | file-atomic pre-ingest over complete MP4s (regular-file + integrity checks), 12+ clean components with 8/2/2 | `FILE_ATOMIC_PRE_INGEST_RUNNING` |
| V6 Temporal Coach | `RGB_ONLY_FRAMEWORK_ABSTAIN_ONLY` | frozen V5 zero-label base + RGB temporal diagnostics | `WAITING_V5_AND_ZERO_LABEL_DATA` |
| V7 Rich PixelArena V2 | `COMPLETE` | three-seed CUDA classification/closed-loop/replay/latency | `FORMAL_PASSED` |

The missing capture node does not block contracts, simulated-source tests, CPU smokes, or Rich
PixelArena. V5/V6 no longer require manual action, frame, tracking, or temporal labels. Read-only
raw recording candidates now exist, but file-atomic pre-ingest and the zero-label evidence
chain are in progress. Current pre-ingest scope is regular-file/integrity checks only; duplicate/re-encode/overlap/near-dup checks are
not claimed. These gaps still block any live throughput, real-domain quality claim, or
commercial-video advice claim.

## 2026-08-13 target supersession: zero-label base route

This entry supersedes active-route references below to legacy two-reviewer action audits,
manually labeled tracking keyframes, or manually labeled temporal segments. Those references
describe retained, fail-closed legacy scaffolding only; they do not authorize collection,
training, release, or advice. The base route is now entirely zero-label.

This entry also supersedes earlier pre-ingest notes describing relation-graph reconstruction and similarity-based component construction.

The route reset is `COMPLETE`: the active implementation target is now V5 source/data work,
not any manual-label or independent reviewer collection.

T8 is a separate self-built-test-app behavior-cloning route. Its training targets will be
automatically captured execution events from owner-operated bounded sessions, aligned to
normalized RGB tensors; this is not a request for manual per-frame/per-action annotation and does
not modify the V5/V6 zero-label contract. The present T8 runtime has a layout, continuous stream,
and capped executor plus keyboard demonstrator writer, guided candidate-layout calibrator,
offline factorized BC/TCN training entrypoint, and read-only T8 Shadow. No recorded demonstrator
dataset, trained T8 checkpoint, held-out metric, Shadow run, or autonomous gameplay result exists yet.
The static-target calibration assumption has been removed: hero/minion/tower/crystal are dynamic
visual entities, so current T8 freezes target intent to `none` and neither asks for their coordinates
nor creates `h/g/t/r` pseudo-labels. A non-trivial target-intent head remains blocked on a separate
RGB localization/tracking and automatically bound execution-event contract.
The separately bounded 60-second/20-action probe is implemented but rejects any missing or failed
training/Shadow evidence; it has not been run.

The legacy per-frame action-audit UI is disabled in code. It cannot be repurposed as the future
preference interface.

The only future human input is a separately authorized post-training gameplay-quality preference
over two complete PixelArena games: `A_BETTER`, `B_BETTER`, `TIE`, or `UNJUDGEABLE`.
It is not a per-frame or per-action label; it cannot alter V5/V6 base training, validation,
model selection, thresholds, or release; it may train only a separately versioned offline
post-training descendant; and it cannot enable real-client control. No preference artifact,
preference model, or post-training checkpoint exists yet.

## Frozen acceptance gates

V4 live: explicit non-symlink `/dev/videoN`, latest-frame-only bounded capture, 1080p60 input,
10 Hz inference, at least 99% scheduled cycles, end-to-end p95 at most 100 ms, zero raw frame
persistence and zero control output.

V5: source teacher has no tick/modulo shortcut; real training has zero human labels; session
splits have no overlap; SimSiam avoids collapse; pseudo labels meet all frozen agreement/OOD
filters; and Mean Teacher runs once. Its real-domain output stays non-promoting and `ABSTAIN`.

V6: tracking and temporal features are derived internally from RGB only. It has no manually
labeled tracking or temporal gate in the base route, stays non-promoting, and emits `ABSTAIN`.
Any later gameplay-quality preference post-training requires a new, separately frozen contract.

V7: independent Rich V2 identity/config/renderer/action hashes; 512 grouped episodes, two
variants and frozen 56/14/15/15 splits; all syntax templates represented; three-seed
factorized ResNet-18 BC; joint exact at least 90%, balanced accuracy at least 85%, each
template recall at least 70%, every factor head at least 95%, raw illegal/correction at most
2%, executed illegal zero, closed-loop completion and side-symmetry gates, negative controls,
fresh-process exact replay/tamper rejection, and RTX 4090 forward p95 at most 10 ms.

## Observed implementation and checks

- Authority reset commit: `f3ff6735fd977cf5993ec0cb0d8398d3b8e6a0fe`.
- V4 live implementation commit: `b8af9607d6dd38a2c66b0552b3411aa0f452faf8`.
- V5 alignment framework commit: `2f1b29d85711f828dc092cb0b6b19a3291c87412`.
- V6 fail-closed temporal framework commit: `f9e9a7f642890720a2cd2c635cb366f5f633b028`.
- V7 failure-evidence commit: `cc10c8e`; side-symmetric RichArena commit: `669824d`;
  observable-teacher/data-contract commit: `c6794738caa14c469a7f12033814ec1b7206c9a0`.
- `make check`: Ruff passed; strict mypy passed for 18 source files; `89 passed`; project
  gate passed with 40 files, 31 Python files, 8,485 nonblank Python source lines, and four
  root Markdown files.
- Frozen regressions: V1 acceptance passed; V2 three-seed structured BC passed; V3 CPU
  smoke passed. V4 capture tests passed `7/7`; V5 contract smoke passed with no release;
  V6 smoke emitted only `ABSTAIN`; V7 CPU smoke passed with five factor heads.
- Final V7 trajectory/data result: 512/512 unique public trajectory groups, two render
  variants, 57,120 RGB samples, and 504/512 teacher crystal completions (`98.4375%`).
- First authorized host RTX 4090 V7 formal attempt reached seed-0 validation but failed with
  `CUDNN_STATUS_EXECUTION_FAILED` because validation was sent as one unbounded batch. The
  code now validates in fixed 256-frame batches.
- The single post-fix retry reached seed-0 sealed classification and failed its frozen
  threshold. No run directory or PASSED report was retained. Review then found and fixed
  red-side 180-degree self-view labels still using world directions; current ego-direction
  code passed static/unit/CPU smoke gates.
- The diagnostic failure path was then changed to atomically retain only a non-promoting
  `status=FAILED` report. The first retained failure
  `runs/rich-v7-v1.failed-1786545699790441608/report.json`
  (SHA-256 `475779f9a930d614053edbfb4da2957ddd54562bcaededbf28eb6edce5215ebd`)
  exposed the red-side NULL failure and side gaps. The root cause was sequential blue-first
  minion movement and non-ego random/tie ordering; commit `669824d` changed these to
  simultaneous, side-symmetric rules and added mirrored episode tests.
- The second retained failure
  `runs/rich-v7-v1.failed-1786547313268648597/report.json`
  (SHA-256 `5a9bdeb57062f220e021cfb1ed63f0a13e119007cd17d31cc24ccde03c9a0550`)
  passed side symmetry but seed 2 achieved random completion `0.90` against a matched
  teacher `0.95`, narrowly missing the frozen relative gate. Diagnosis found a hidden
  episode-seed teacher mode that gave conflicting labels to identical semantic RGB frames.
  Commit `c679473` removed that unobservable input, made the tactic depend only on visible
  public state, and added a zero-conflicting-label collection gate. Data definition, model
  architecture, training seeds, and thresholds were not relaxed.
- Final `make accept-v7` started from clean code commit
  `c6794738caa14c469a7f12033814ec1b7206c9a0` and passed on
  `NVIDIA GeForce RTX 4090`, Python 3.11.15, Torch 2.5.1+cu121. The retained run is
  `runs/rich-v7-v1`; its `report.json` SHA-256 is
  `873df770673366fcc9e965d33239b157c2309c059976d82ff258b48cf89416bf`.
- Sealed test joint/balanced accuracy for seeds 0/1/2 was respectively
  `0.997281/0.998851`, `0.994449/0.993821`, and `0.998641/0.999601`; every seed and every
  frozen classification gate passed. The selected validation-loss seed was 2.
- Closed-loop NULL completion for seeds 0/1/2 was `1.00/1.00/1.00`; random completion was
  `1.00/0.95/1.00`; matched teacher completion was `1.00` for all three. Blue/red completion
  was `1.00/1.00`, `0.95/0.95`, and `1.00/1.00`, so every side gap was zero. Raw illegal and
  mask-correction rates were `0/0`, `0.003846/0.003846`, and `0/0`; executed illegal actions
  were zero for every seed.
- Negative controls passed: selected seed 2 actual joint accuracy `0.998641`, black-frame
  joint accuracy `0.240825`, mismatched-frame joint accuracy `0.153829`; drops were
  `0.757816` and `0.844812`. Fresh spawned-process replay verified 71 ticks and terminal
  outcome; config, action, and event tampering were all rejected. RTX 4090 batch-1 FP32
  forward p95 was `3.548 ms` against the frozen `10 ms` limit.
- The final run directory contains exactly five files. Recorded and independently recomputed
  SHA-256 values are: dataset
  `4f97f3cfc74b534ef4a159004b21d4b31c9167740bd59914df37aecc5c0390cd`, seed-0 model
  `112081cac1c0c9a3629e78647f4f568ecf40bab95e34c08e22aee58a18653324`, seed-1 model
  `592770cb38350037c37194b86c23f89afb9bcbc3a69918bd9797d56099825170`, and seed-2 model
  `ca3ac0bd1f6265911b08de2cd4ee496ae01f89a46a1f1ebf8ea1ff02801b3f3c`.
- The original project shell exposed a conflicting CUDA library path. Direct pytest
  collection failed on `libcusparse`/`nvJitLink`; all Make targets now run with
  `LD_LIBRARY_PATH` unset and the pinned project Torch 2.5.1 environment passed.
- No `/dev/video*` was visible at planning time; this is not evidence about future host
  capture-card availability. No live V4 run was attempted.
- No formal real-recording manifest, sealed action audit, or tracking-label set has been
  supplied. No V5/V6 real-domain accuracy or advice class is released.
- 2026-08-13 read-only E-drive pre-ingest audited 149/149 MP4 containers successfully:
  134,678.097 seconds (37 h 24 min 38 s), all H.264, with 78 AAC-audio and 71 silent files.
  Anonymous full-content review confirmed two exact duplicate pairs. Logical organization
  retains 145 canonical videos of at least five minutes (37.03 h), reserves two short videos,
  excludes two redundant copies and one promotional JPG, and leaves every split unassigned.
  Five samples per video decoded 745/745 with no black sample; 27 visually reviewed samples
  confirmed gameplay and visible UI/name privacy risk. The Git-ignored evidence is
  `runs/wzry-data-audit-20260813-v1`; `SHA256SUMS` hashes to
  `f43f6628644007c09a64a17c3d8147fe45a9ef382e908f8735ceed9acded7a29`.
  Raw files were not changed, copied into Git, or persisted by path. Superseded historical note: this is not a formal V5
  manifest for the current file-atomic contract; partial overlap/re-encode grouping, privacy masking,
  independent-session proof, and source-rights documentation remain unresolved.
- New large training storage is routed through `HOK_LARGE_ROOT`, defaulting to
  `$HOK_LARGE_ROOT`, with dedicated datasets/checkpoints/runs/cache/
  audit/staging directories and a fail-closed read-write mount preflight. A later host-namespace
  check corrected the initial sandbox-only observation: `/dev/sda2` is mounted `fuseblk rw` on
  the host, `/etc/fstab` already binds UUID `<redacted-storage-uuid>` to `<local-data-volume>`, and the
  generated local large-data mount is active under `local-fs.target`. The sandbox exposes the
  mount read-only, so its failed write probe and `storage-preflight` result are not host-state
  evidence. No directory was created and no frozen run was moved. Before starting heavy writes,
  review the repeated host `ntfs-3g` `Failed to read index block: Input/output error` events
  recorded on 2026-08-09, 2026-08-11, and 2026-08-12; no force-remount or repair was attempted.
  `make storage-show` resolved every large-output root under E, and the storage/formal-target
  dry-run preserved that routing. The post-change `make check` passed Ruff, strict mypy, 83
  tests, and the 40-file / 31-Python-file / 8,999-line size gate. `make accept pixel-smoke shadow-live-smoke
  alignment-smoke temporal-smoke rich-smoke` also passed; these were CPU/fail-closed
  regressions and did not create GPU or hardware evidence. Run `make storage-preflight` and
  `make storage-init` from the host shell after the NTFS I/O issue has been reviewed.
- 2026-08-13 pre-data closure replaced bare in-memory V5 promotion inputs with strict
  regular-file manifest/source/target/pseudo/model/ledger/audit loading, persisted the single
  Mean Teacher EMA model, and retained legacy release/audit scaffolding as fail-closed only.
  The source-only producer defines its source-validation collapse thresholds; no accepted
  real-domain adapted/EMA collapse evidence exists. The active V6 route is RGB-only and
  zero-label; its public wrapper remains `ABSTAIN` without a release path.
- Pre-data closure checks observed: Ruff passed; strict mypy passed for 18 source files;
  full pytest `83 passed`; project gate passed with 40 files, 31 Python files, 8,999 nonblank
  Python lines, and four root Markdown files. V1 and V2 acceptance passed; V3, V5, V6, and V7
  CPU smokes passed; V4 capture tests passed `7/7`. No hardware, real recording, label, CUDA
  formal training, or real-domain capability evidence was created in this closure.
- 2026-08-13 V5/V6 ingress closure (implementation only): added a strict V5 manifest v2 that
  requires descriptor-identified pre-ingest component evidence, an explicit operator declaration, a
  mechanically derived component cohort, and an owner-authorized zero-redaction/rotation privacy context. Components,
  rather than manifest rows, satisfy the 8/2/2 split gate. Derived-shard alignment binds descriptor,
  PTS, split, rotation, privacy transform, and declaration. Added `v5-pre-ingest`, which
  reads only regular local MP4s via one descriptor per candidate, emits metadata-identity
  component evidence, rejects in-place mutation, and has bounded full-duration
  sampling. This historical relation scan behavior is superseded by the current file-atomic
  contract. No raw MP4 was read by that implementation delivery, no E-drive output was written,
  and no real-domain release or advice artifact was manufactured.
- Legacy V6 artifact-binding code remains unavailable to the active route. The active RGB-only
  V6 diagnostic neither accepts human tracking/action/audit inputs nor exposes an advice class;
  its public output is always `ABSTAIN` with `control_output=false`.
- 2026-08-13 source-producer framework added a fixed 128-episode causal PixelArena corpus,
  three fixed source seeds, source-validation-CE-only selection, and a source-baseline collapse
  diagnostic. Its outputs are source-only, are constrained below `HOK_LARGE_ROOT`, and do not
  read raw recordings, write a V5 release, or enable advice. A prior host RTX 4090 source-only
  run at `HOK_LARGE_ROOT/datasets/v5-source-v1` produced 5,040 synthetic rows and seed CE values
  `0.0014457920/0.0050881389/0.0029133596`, selecting seed `0`; however, its renderer hash
  (`84d272...`) differs from the current renderer contract (`45e757...`). It is retained only as
  stale non-promoting evidence and cannot be consumed by this route. The current-contract rerun at
  `HOK_LARGE_ROOT/datasets/v5-source-v2-current-contract` completed all three CUDA source seeds:
  seed `0/1/2` validation CE was `0.0014457920/0.0050881389/0.0029133596`, selecting seed `0`.
  Its renderer/producer/teacher/action hashes are `45e757...`/`1bb30d...`/`932450...`/`24b668...`
  and match the current source contract. This remains source-only, non-promoting evidence. No real-video adaptation, real-domain
  validation, advice release, or client capability claim exists.
- 2026-08-13 host storage preflight observed `/dev/sda2` mounted at `<local-data-volume>` as
  `fuseblk rw`; current-boot kernel evidence contained no `sda` or NTFS I/O error. The formal
  `v5-pre-ingest` scan of the authorized 149 MP4 candidates is now running. It writes only the
  single file-atomic component artifact `HOK_AUDIT_ROOT/v5-file-atomic-pre-ingest-v2.json` after all candidate
  regular-file/integrity checks; no raw frame, audio, or source locator is written.
- Historical ingress-closure verification: `make check` passed Ruff, strict mypy for 19 source
  files, and `93 passed`; that earlier snapshot reported 41 files, 32 Python files, and 8,397
  nonblank Python lines. `make accept`, `accept-v2`,
  `pixel-smoke`, `shadow-live-smoke`, `alignment-smoke`, `temporal-smoke`, and `rich-smoke`
  passed as CPU or fail-closed regressions. These commands did not create hardware, raw-video,
  E-drive, CUDA-formal-training, release, or real-domain-advice evidence.
- 2026-08-13 V5 base-chain closure: added path-only commands and Make targets for freezing a
  training config, SimSiam adaptation, model-generated prediction evidence, pseudo materialization,
  and one Mean Teacher round. Pseudo materialization now accepts only a generated evidence directory,
  recomputes every `2 models x 3 timestamps x 3 deterministic safe views` record, rejects a
  hand-written prediction JSON, and blocks Mean Teacher before any output write unless at least
  200 rows survive. Both adaptation and Mean Teacher train only source-train rows; a ledger-write
  failure removes the newly written EMA. Current focused verification passed Ruff, strict mypy for
  the touched source modules, `14` V5 alignment/data tests, the V5 and V6 non-promoting smokes,
  Make dry-runs, `git diff --check`, and the current project gate (`45` files, `36` Python files,
  `13,320` nonblank Python lines, four root Markdown files). No cohort, target shard, real-domain
  adaptation, pseudo artifact, EMA, release, advice, hardware, or client-control evidence was
  created. The host pre-ingest job remains in progress and has not produced its final artifact.
- 2026-08-15 scale-policy update: repository/Python file counts and nonblank Python lines remain
  reported but no longer have pass/fail ceilings. Dataset bytes and session count likewise have no
  global ceiling. Per-run termination, shard/memory/concurrency bounds, storage checks, device
  gates, and frozen-manifest immutability remain enforced. This supersedes the earlier 100/100/20,000
  and 1,000/1,000/200,000 administrative limits.
- 2026-08-13 historical scope note: project limits were set to 100 project files, 100 Python files,
  and 20,000 nonblank Python lines; the matching outer reference-size guard was 1,000/1,000/200,000.
  Current V5-base-chain verification reports 45 project files, 36 Python files, 13,563 nonblank
  Python lines, and four root Markdown authority files. `make storage-preflight` passed on the
  host; Ruff, strict mypy for 21 source files, the project gate, `git diff --check`, and 42
  focused V5/V6 tests passed. The pre-ingest process remains active and has not produced an
  artifact, so this is code/contract evidence only—not a real-domain training result.
- 2026-08-13 cohort-loader hardening: the downstream V5 manifest loader now requires the component
  cohort to equal every clean pre-ingest component and recomputes the fixed all-clean lexicographic
  8/2/2 allocation. Omitted components, reassigned but count-valid splits, and privacy-recipe
  tampering fail closed. The component artifact is mechanical evidence, distinct from the separate
  operator attestation. Focused alignment/data tests passed `20`; no real-domain artifact was made.
- 2026-08-14 T8 steps 1–3 implementation closure: all mobile input paths now preflight a new
  output, require the explicit online serial, and recheck foreground package
  `<owner-testbed-package>` plus display identity before every sent tap/swipe. Continuous capture
  accepts only a non-symlink `/dev/videoN` character device. Formal demonstrator collection now
  requires the complete calibrated layout, `scrcpy --no-control` V4L2 capture, and
  event-anchored eight-frame derived RGB windows. Formal T8 v1 data is eight named complete
  five-minute sessions with a frozen 4/2/2 train/dev/test manifest, split-level factor coverage,
  session hashes, and event/shard binding below `HOK_LARGE_ROOT`. HID/UHID, uinput, minitouch,
  controller input, layout calibration, smoke collection, formal sessions, training, Shadow, and
  autonomous execution were not run or claimed. Focused mobile/T8/boundary/environment tests and
  full project verification passed: Ruff, strict mypy, `143 passed`, and the 52-file / 42-Python-file
  / 17,075-line safety gate.
- 2026-08-14 T8 steps 5–7 code closure: T8 now requires the current-contract, source-only V5
  ResNet-18 source checkpoint as a hash-bound `/255` initialization, runs three seeds with
  dev-total-CE-only selection, and keeps test data outside selection. A sealed evaluator records
  joint/factor metrics, confusion, session-local switching, and fixed negative controls. Five-minute
  Shadow requires that evaluator, emits only `ABSTAIN`/`control_output=false`, and gates coverage,
  scheduled-cycle latency, confidence, OOD, and stability. No formal T8 data, training, evaluation,
  Shadow, or test-app execution evidence was created by this code change.
  Post-change verification passed: actual V5 source-contract loading resolved the selected model
  SHA `9e0965…e4fae4` with 120 encoder tensors; `make t8-contract-smoke` passed 14 tests; and
  `make check` passed Ruff, strict mypy, `144 passed`, and the 52-file / 42-Python-file / 17,484-line gate.
- 2026-08-14 T8 dynamic-target correction: review of ResnetGPT, WZCQ, and wzry_ai confirmed
  reusable fixed-coordinate patterns for control buttons, but no basis for treating moving
  hero/minion/tower/crystal entities as layout coordinates. T8 layout v3 therefore removes those
  four points, calibration now requests only skill1/2/3 and confirms 12 fixed-control actions,
  `h/g/t/r` no longer create manual pseudo-labels, and target intent is fail-closed at `none` until
  a separate RGB localization/tracking contract exists. Data/session/split/model/training/evaluation/
  Shadow schemas were advanced to reject the superseded contract. No phone input, collection,
  training, evaluation, or Shadow run occurred. `make t8-contract-smoke` passed 15 tests; full
  `make check` passed Ruff, strict mypy, `145 passed`, and the 52-file / 42-Python-file /
  17,461-line gate.
- 2026-08-14 T8 static-layout host calibration: the first movement check exposed that the old
  joystick center/radius/vector had never been owner-confirmed. Calibration was extended to pick
  joystick center plus a north drag endpoint, derive radius/vector mechanically, use an 800 ms
  movement only for visible confirmation, and support `r` to resend the current action without
  advancing. The owner then confirmed all eight directions, basic attack, and skill1/2/3. The
  resulting `configs/mobile_testbed_layout.local.json` is complete, contains no dynamic
  target coordinates, and has SHA-256
  `<redacted-layout-sha256>`.
  No demonstration corpus, training, evaluation, Shadow, or autonomous execution evidence was
  created by calibration.
- 2026-08-14 T8 host demonstration smoke: E-drive write/fsync/delete passed. Interactive terminal
  capture was changed to wait at an explicit Enter gate, read keys with unbuffered `os.read`, accept
  ASCII case-insensitively, disable XIM for the launched xterm, and print accepted/ignored keys.
  `t8-keyboard-reception-smoke-001` bound north plus basic attack to two eight-frame samples.
  The first 60-second artifact retained a valid but coverage-incomplete diagnostic. The succeeding
  61.02-second `t8-demonstration-smoke-002` covered all eight movement directions, basic attack,
  skill1/2/3, and three aimed skill events with 12/12 executable inputs sent. The separate
  `t8-wait-reception-smoke-001` confirmed the pure `(wait, none, none, none, 0)` event with no input
  sent. All inspected event/NPZ/frame/manifest hashes bound, timestamps increased, layout SHA was
  `<redacted-layout-sha256>`, and no raw frame was
  persisted. These are smoke artifacts, not any of the eight formal sessions.
- 2026-08-14 T8 formal session 001: the atomically published 301.017-second session passed the
  production `_load_session` validator with 13 eight-frame samples, 12 executed actions, layout SHA
  `<redacted-layout-sha256>`, and session SHA
  `ab577fc24014ce729d985536ab9f686d2ccfb805c3d719aa69e9ad830e6bfbf5`. Movement, ability,
  hold-duration, and pure-wait vocabularies were covered. Aim contained `none` and `north` only;
  sessions 002 onward must include all eight non-none aim directions so every frozen split can
  satisfy coverage. This is collected data only; no training, evaluation, or Shadow was run.
- 2026-08-14 T8 bounded scripted collection entry: an explicitly seeded controller now uses the
  existing guarded ADB executor to cover every implemented movement, ability, aim, and hold factor
  without terminal input. Summary and event rows identify this source as
  `bounded_scripted_controller_v1`; the T8 loader binds and preserves source identity while keeping
  legacy `session-001` compatible as `terminal_keyboard`. Focused Ruff and strict mypy passed, and
  the 17 mobile/T8/Shadow tests passed with the project environment's conflicting external CUDA
  library path removed. No new formal session, training, evaluation, or Shadow run is claimed yet.
- 2026-08-14 T8 scripted data/training result: sessions 002–008 completed for 301.02–301.07 seconds
  each with 200 event-bound samples and full `[9,5,9,1,3]` factor coverage. Together with the
  owner-keyboard session 001, the production loader froze train/dev/test as 4/2/2 with split SHA
  `dc9679e5320f98ae403a45977c9ff8a65e05f4c45d8f39f8375e14daeaf14fff`. The V5-bound three-seed
  CUDA run selected seed 1 exclusively by dev total CE; its model SHA is
  `3a5008ace55d4eeb792a8548f06be0202659d1a11b5efefd6ba4037eb014c6a3`. Sealed test evaluation
  failed: joint exact `0.0`, movement/ability/aim accuracy `0.7675/0.26/0.29`, switch-rate error
  `0.9975`, and both negative controls failed. The model collapsed to per-head plurality classes
  because the seeded open-loop actions are not inferable from preceding RGB. This corpus is valid
  execution-event diagnostic data, not a successful expert-demonstration corpus. Shadow was not
  run because its offline admission gate failed.

## T8-v2 target: video-adapted practical baseline

This target is pre-registered for a new `t8-demonstrations-v2` / `t8-policy-v2` lineage. It does
not alter, reinterpret, or rerun the sealed v1 test result. V1 remains `FAILED`; its data, split,
models, thresholds, and evaluation report stay frozen as negative-control evidence.

### Fast implementation route

1. Use only the existing `103` V5 train-video sessions for RGB representation adaptation. Start
   with a deterministic 2 Hz subset of the existing 10 Hz target shards, approximately `181k`
   frames, and run a short V5-initialized ResNet-18 SimSiam/temporal-consistency adaptation. The
   `23` video-dev sessions may select the encoder; the `23` video-test sessions remain unopened for
   selection, threshold tuning, normalization statistics, pseudo labels, or error analysis.
2. Collect at least `12` new five-minute owner-operated sessions (`8/2/2` train/dev/test). Prefer
   direct use of the phone's virtual controls with a narrowly allowlisted, read-only
   `adb shell getevent` source; if that source is unavailable, use a focused host keyboard
   keydown/keyup adapter through the already accepted foreground-package/display-guarded ADB
   executor. RGB and standardized observed actions are paired automatically; no per-frame or
   per-action human annotation is requested.
   Every split must contain all represented movement, combat, aim, and hold factors.
3. Replace the v1 independent-head actor with a small video-adapted ResNet-18 plus a `16`-frame,
   approximately `1.6`-second causal residual TCN. Use channel-mixing `1x1` layers, masked
   class-balanced losses, and action legality masks. Movement and combat are separate factors;
   aim and hold losses apply only when semantically active. The constant `target=none` factor is
   recorded in the contract but is not counted as a learned-performance head.
4. Run a one-seed pilot and the time-alignment/label-shuffle controls before any three-seed run.
   Only after the pilot gate passes may three fixed seeds train. Selection uses validation data
   only; sealed test data is opened once after code, weights, thresholds, and metrics are frozen.
5. Run a five-minute read-only Shadow only after the lowered offline gate passes. Shadow writes
   predictions and diagnostics with `control_output=false`; model-driven input remains disabled.
6. Treat inverse-dynamics recovery from the 37-hour video corpus as an optional post-baseline
   enhancement, not the critical path. It first requires a separate held-out before/action/after
   probe to prove high-precision recovery. Pseudo actions may be generated only for video-train
   sessions and may never replace the sealed execution-event test set.

### T8-v2 visual-only input and demonstration-source contract

- The policy receives RGB only. Touchscreen or keyboard records are offline supervision targets
  and never encoder, temporal-state, training-input, Shadow-input, or runtime-policy features.
- The optional direct-touch source is read-only and bounded to the explicit serial, official
  foreground package, frozen display identity, and session duration. It neither calls an app API
  nor emits device input. Raw touchscreen events and input-device paths are not persisted.
- A direct touch is stored as `observed_touch_action`; it proves the normalized gesture but not
  internal app acceptance. A guarded keyboard-to-ADB fallback stores `executed_action` only after
  successful dispatch. Both sources bind their provenance into the session manifest and may not
  be silently mixed within one session.
- Failure to read the passive touch source falls back cleanly to the focused keyboard adapter.
  Root, automatic input-device selection, unrestricted shell access, global keyboard hooks,
  scrcpy control, HID/UHID, uinput, minitouch, and RGB-inferred action labels remain excluded.
- The passive probe, calibration, v2 collector, frozen-split writer, train/dev-only video adapter,
  and one-seed pilot code are implemented and unit-tested. The video adapter is complete; no formal
  v2 corpus, frozen split, Pilot, evaluation, or Shadow pass is claimed.
- 2026-08-14 route correction: host keyboard testing showed high ADB swipe latency, discontinuous
  movement, and no natural multi-touch, so keyboard and scripted data are diagnostic-only. The
  incomplete session-001 attempt was stopped and never published formally. Passive Type-A touch
  collection is again the active route: fixed combat-button regions now take precedence over the
  joystick region, and the formal loader accepts only `observed_touch_action` sessions bound to one
  touch-calibration hash. This semantic fix passed focused tests; a short direct-phone revalidation
  is required before the first five-minute session.
- 2026-08-14 Type-A continuous-state contract implementation: the passive observer now emits
  stable logical contact slots with explicit down/move/up phases, locks each contact to its
  joystick or fixed-button role, and supports one joystick plus one combat contact. The touch
  writer now stores one derived RGB frame with the current state at 10 Hz and emits semantic
  transitions only when factors or hold buckets change; the loader rebuilds 16-frame causal
  windows offline. `mobile-demonstrate-touch --semantic-smoke` and `make t8-v2-touch-smoke`
  implement the fixed 20-second zero-control gate. Formal sessions now require 300 seconds,
  2,850 samples, effective movement, and no concurrent-combat conflict; full vocabulary remains
  a split-level gate. This is implementation and focused-test evidence only: no new smoke or
  formal phone session is claimed.
- 2026-08-14 Type-A dual-contact correction: reports with duplicate tracking identifiers now
  preserve existing slots by normalized spatial continuity and allocate a new slot for a distant
  second contact; the focused regression covers reversed report order. The immediately following
  20-second read-only smoke observed only `wait`, so it provides no live confirmation or rejection
  of the correction and remains diagnostic-only. Subsequent diagnostic summaries expose only
  aggregate first-contact roles (`joystick`, fixed button, or `unknown`), never raw coordinates.
- 2026-08-14 direct-phone revalidation: a 20-second read-only smoke observed one `joystick` and
  one `basic_attack` contact, yielding 200 simultaneous movement-plus-basic-attack samples with
  zero conflicts and zero control output. This confirms the corrected dual-contact path; the
  smoke remains incomplete because skill1/2/3 and skill-drag aim were not exercised.
- 2026-08-14 right-button recalibration: owner completed the four-point `touch-calibration-v2`
  procedure. Its calibration hash is `<redacted-calibration-sha256>`,
  bound to the current layout and Type-A descriptor; Makefile and examples now select v2.
- 2026-08-14 v2-calibration smoke: the 20-second attempt observed only a joystick contact, so it
  cannot assess fixed-button mapping and is diagnostic-only; no formal data claim follows.
- 2026-08-14 collection simplification: a formal v2 session now requires duration, sample count,
  effective movement, and zero touch conflicts only. Skill and aim diversity are pooled and
  enforced when the frozen 8/2/2 split is created, so the owner can simply play normal sessions.
- 2026-08-14 auto-execution route authorization: touch calibration stops as a critical path. The
  independent `formal-auto-v1` lineage uses the frozen layout, guarded ADB executor, 16-frame RGB
  windows, and exact `executed_action` events. A 21.04-second smoke dispatched 27 actions and
  covered every combat class, every aim class, and all hold buckets; it was diagnostic-only.
- 2026-08-14 formal-auto session-001: 301.09 seconds produced 300 causal windows and 292 dispatched
  actions with complete movement/combat/aim/hold coverage. The formal loader verified manifest,
  event/frame binding, action contract, and tensor shapes; session identity is
  `edd59d4404c55a4bc2cd997fa366b0c8eab8ab32f9727334937ae1e2921cf87c`.
- 2026-08-14 keyboard-v2 implementation: `mobile-demonstrate-keyboard-v2` now records guarded
  `executed_action` labels with 16-frame/100 ms causal windows, one-shot 200/500/900 ms holds,
  live coverage reporting, atomic action-contract/session hashes, and a 300-second/180-sample/full-
  coverage formal gate. Formal loading accepts only this schema and fixed twelve-session 8/2/2
  identity. The video adapter now consumes 4D RGB correctly, streams a deterministic per-session
  2 Hz index, freezes encoder BatchNorm consistently, performs one encoder pass per view, and
  never opens video-test shards. The Pilot command now runs normal and shuffled seed-0 controls as
  a pair, enforces per-class recall and a 0.10 joint gap, then stops at manual review. This is code
  implementation and unit-test evidence only: no new phone session, adapter checkpoint, Pilot,
  test evaluation, or Shadow run is claimed.
- 2026-08-15 T8-v2.1 live-control implementation: the new
  `mobile-demonstrate-keyboard-v2-live` route uses the pinned system scrcpy 1.25 server hash for a
  single H.264/control session, a focused Tk keydown/keyup window, continuous joystick
  DOWN/MOVE/UP, and a simultaneous combat pointer. It writes current action state at 10 Hz into
  streaming 16-frame/100 ms causal shards and sparse semantic transitions, with no V4L2 node,
  `adb shell input`, raw video, raw keys, serial text, or device path in the dataset. A background
  serial/package/display watchdog gates each socket dispatch. Focused protocol, lifecycle,
  two-pointer, direction-change, writer, boundary, Ruff, and mypy checks are implementation
  evidence only; no v2.1 phone smoke, formal session, training, evaluation, or Shadow pass is yet
  claimed. Existing v2 auto sessions remain sealed negative-control evidence.
  Independent high-risk review then blocked live use until non-finite durations, stale/ended video,
  server-process identity, handshake cleanup, failed-formal publication, strict server-contract
  loading, and split atomicity were corrected. The implementation now rejects NaN/Infinity before
  device access, requires fresh advancing frames at stream FPS >=10, checks the launched pinned
  server process after the two-socket handshake, publishes incomplete formal attempts only below
  `diagnostics/`, and atomically freezes manifests. A separate exact-three-session 2/1 pilot split
  precedes the unrestricted-N (minimum twelve) fixed-seed split; Pilot opens no sealed test-session
  files. These corrections have focused test evidence only and do not change the no-live-evidence
  statement.
  Final read-only host preflight found `<local-data-volume>` mounted `fuseblk ro` and the current
  Codex process has no `DISPLAY`. Therefore the code is ready for a 20-second smoke, but no smoke
  was launched: the required external dataset root is not writable and the focused Tk window
  cannot open in this process. This is a host-state blocker, not a code or model result.
  Host-side execution subsequently confirmed the E mount is writable and DISPLAY `:1` exists.
  The first live smoke failed closed before collection because the original 250 ms guard age sat
  on the measured 121--156 ms two-command ADB check plus its 100 ms polling interval. The maximum
  snapshot age is now 500 ms while the watchdog still polls every 100 ms; this retains sub-second
  package/display stop behavior without treating normal host ADB jitter as stale.
  The retry completed 20.125 seconds through the pinned scrcpy socket route and wrote 198 causal
  samples to `smoke-live-1786782137`; server SHA, layout SHA, no-raw-video rules, and device guards
  passed. The focused window received no key transitions, so `semantic_events=0`,
  `touch_messages_sent=0`, and all labels are `wait`. This artifact validates video/timing/storage
  only, is explicitly non-formal, and is forbidden from training; keyboard focus/action coverage
  remains the next smoke item.
  A second focused-window retry also produced zero semantic events. A host-only Tk self-test then
  confirmed that its KeyPress/KeyRelease bindings and 30 ms release debounce work, so the zero-input
  artifacts do not justify further owner repetition. The default Make smoke is now a fixed
  20-second transport/lifecycle diagnostic with a distinct source and
  `training_eligible=false`; formal collection remains human-key-only.
  The first backend-only control diagnostic completed at
  `diagnostics/control-smoke-1786782844`: 20.058 seconds, 198 causal samples, 18 scrcpy touch
  messages, 16 semantic transitions, zero conflicts, and 0.417 ms keyboard-semantic-to-socket p95.
  It covered sustained movement, all four combat buttons, two aim directions, all hold buckets,
  and 43 simultaneous movement-plus-combat samples; `core_complete=true`. Manifest verification
  passed, the formal T8-v2.1 loader rejected it as required, and the target package/display remained
  foreground at 1600x720 rotation 1. This closes transport/lifecycle smoke only, not human
  demonstration data, training, evaluation, or Shadow.
- 2026-08-14 keyboard-v2 smoke: the 61.07-second owner-terminal smoke recorded 11 samples, seven
  dispatched actions, four explicit waits, movement, and basic attack. A separate 21.10-second
  bounded backend sequence recorded 13/13 dispatched actions and covered skill1/2/3, every
  non-none aim direction, and 200/500/900 ms hold buckets. Together they cover the complete v2
  movement/combat/aim/hold vocabulary under action-contract SHA
  `e8f0963a9f4621b12a62a8d7551f7fbcf355b1f53578bdac63f3a22ec9221a2a`. Both artifacts have
  `formal_session=false` and are diagnostic-only. Formal collection remains 0/12.
- 2026-08-14 T8-v2 video adapter completion: the actual five-epoch RTX 4090 run consumed 181,366
  deterministic 2 Hz video-train frames and 46,688 video-dev frames, with
  `video_test_accessed=false`. Validation loss selected epoch 3 at `-0.996697201794141`; its model
  SHA is `05c948c1adbec93293450cb8f42b265790e3e323aa800940e3ef30147bcb684a`. All five checkpoint
  hashes match the atomically published report, whose SHA is
  `bc6dc3419a7a7025390ead221f85ae128c73b150ccca1ee71914f795a962e458`. This completes visual
  adaptation only; it is not a policy, Pilot pass, test evaluation, or Shadow result.
- 2026-08-15 T8-v2 formal-auto corpus and Pilot: all twelve 301-second sessions completed with
  300 causal windows each, 3,498 dispatched actions in total, and complete per-session factor
  coverage. The frozen 8/2/2 manifest assigns sessions 001–008/009–010/011–012 and has split SHA
  `929c99ed2fe97dcd01ebfdb31741e0a4387e2361f5396b35f8e466a5b6d54778`. Before training, the
  conditional-head gate was corrected to exclude the semantically invalid `none` class from
  active aim/hold coverage, and dev validation was changed from one unbounded tensor to exact
  batched loss accumulation; focused T8/Shadow tests, Ruff, source mypy, and diff checks passed.
  The completed paired seed-0 Pilot failed as pre-registered: normal joint exact was `0.028333`,
  shuffled joint exact was `0.030000`, the gap was `-0.001667`, and the normal primary gate and
  shuffled-label failure gates were both false. Test tensors were not accessed. The wrapper report
  SHA is `2c900d6a06abc4fb261603462a7c21b4e5d58034b09ff666294a820770edafa3`; status is
  `PILOT_DIAGNOSIS_REQUIRED`. This evidence does not authorize three-seed training, sealed test
  evaluation, Shadow, threshold relaxation, a larger model, or any model-driven device input.
- 2026-08-15 offline video-action probe: the first fixed-orientation diagnostic was rejected after
  stratified review showed mixed stored orientations. The superseding v2 probe detects each
  session's letterbox axis, rotates only portrait-content sessions, maps the frozen normalized
  button layout into the detected content box, and applies a fixed local-flash/onset/decay rule.
  Across eight video-train and two video-dev sessions, all 354 selected shards passed SHA binding;
  two sessions required counter-clockwise rotation and eight were already canonical. The probe
  found 131 basic-attack, 409 skill1, 324 skill2, and 284 skill3 candidates, with every button
  represented in every selected session. Visual review of the 16 highest-scoring derived-RGB
  candidates found 16/16 button-centered flashes and no obvious menu or whole-screen transition.
  The report SHA is `0b8c27c861a1459d0c9c70b30e51895c501728cc16950ce59adc662be5e4adb3`;
  video-test was not accessed. This is promising combat-action signal only, not measured precision,
  admitted pseudo labels, movement/aim/hold recovery, policy training, or Pilot promotion.
- 2026-08-15 offline video-action probe v3-v5: the frozen temporal rule now requires onset and
  five-frame decay, groups cross-button collisions within two frames, abstains below a `1.25`
  top-to-second score margin, and retains only candidates with a complete 16-frame causal history.
  A central dark-gray overlay filter removed 106 additional candidates. Expanding dev from two to
  four sessions without relaxing thresholds produced train counts `52/281/196/136` and dev counts
  `13/162/100/81` for basic attack/skill1/skill2/skill3; all aggregate class minima and the
  `0.056114` ambiguous-group ceiling passed. Admission still failed because one train and one dev
  session had zero basic-attack candidates. Visual review of 32 split-by-class samples also found
  one clear post-game/menu false positive in dev skill3, so no candidate was admitted as a training
  label. The v5 report and contact-sheet SHA-256 values are
  `8cdff1039843fe8ff5dfdb410cf2367840f619cd12b663ecfd3b7e00c5885bec` and
  `65a44f7d72cd5db7e5ba3613785d5653d4c96103c17d1095fcadaf26aa21ab90`;
  video-test remained untouched. This remains combat-signal diagnosis only and does not recover
  movement, aim, or hold, authorize policy training, or override the failed causal Pilot.
- 2026-08-15 offline video-action probe v6: review of the lowest small-map edge-strength candidates
  showed that the three lowest were exactly the remaining non-gameplay screens and the fourth was
  normal gameplay. A frozen small-map HUD edge threshold of `7.0` therefore rejected those three
  candidates without changing any temporal, collision, class-count, or ambiguity threshold. The
  retained train counts remain `52/281/196/136`; dev counts are `11/162/100/80`, so aggregate class
  minima still pass. A fresh score-stratified 32-sample review found no obvious non-gameplay frame.
  Admission remains blocked because one train and two dev sessions have zero basic-attack
  candidates. The report/contact-sheet SHA-256 values are
  `44db77c1ecedd23fedf2bc44929c4adc31466c6a8d22c833da42579d46e78202` and
  `d84286409e5e0e8e936e09e62791d3568f709bfb998753ee5bcb4c2a9e3ea17f`;
  video-test remained untouched and no labels were admitted.
- 2026-08-15 video combat candidate materialization: the extra per-session-all-classes gate was
  corrected to the requested split-level coverage contract, with the stronger requirement that
  every class appear in at least half of each selected split's sessions. Basic attack appears in
  `7/8` train and `2/4` dev sessions; every skill appears in all selected sessions. The resulting
  v7 probe passed all frozen aggregate, ambiguity, split-redundancy, visual-review, and test-seal
  gates. It materialized 1,018 combat-only candidate samples as 17 compressed shards containing
  canonical derived RGB causal windows of exactly 16 frames. Full independent re-read verified
  every shard hash, every window content hash, array schema, counts, and session isolation. The
  final manifest SHA is `990a433fa48498415beb6adee2df82ad5dde71b9ca55f27ec7d157cb0948be5a`;
  the verification report SHA is `286e4d3a45ad733890ba495494e7471803975695e94a4543a2e6001b5d9b36d1`.
  The artifact explicitly remains `training_allowed=false`: it supports a combat-only paired
  learnability diagnostic next, not full T8 training, movement/aim/hold recovery, test access,
  Shadow, promotion, or device input.
- 2026-08-15 strict-causal correction and combat learnability result: pre-training review found
  that the first materialized v1 windows included the detected button-flash frame, which would
  leak an already executed action. That 1,018-window artifact remains frozen as failed evidence
  and was not trained. The replacement v2 uses exactly the 16 frames before each detected event;
  every observation ends exactly 100 ms before the event. Full re-read again verified all 17
  shards and 1,018 window hashes; its manifest and review SHA-256 values are
  `d2aa82297a169c390e4915fa5b486fe0bb3b12c8394193d7f4afe24b0436bdc7` and
  `9c82ca21e06cfa79672ba277213a887066d6f2ca350cfedfbc5c72bcecd13f03`.
  The frozen seed-0 combat-only diagnostic used the selected epoch-3 adapter as a frozen encoder,
  the existing causal TCN, class-balanced loss, 12 epochs, and dev weighted CE selection. Normal
  labels achieved accuracy `0.464589`, balanced accuracy `0.256914`, and macro-F1 `0.176309`;
  basic-attack/skill2/skill3 recall was `0/0.04/0`. The shuffled control macro-F1 was `0.092379`,
  leaving only a `0.083931` gap against the frozen `0.15` requirement. Both the primary and
  negative-control gates failed. The report SHA is
  `cc8985683f5fd3ad8f9a52a85ea4c92dfdddf5d488e27bdd34245bda99e27928`;
  video-test was untouched. Button flashes therefore support retrospective action recognition,
  not a learnable pre-action policy under this evidence. Three-seed training, full-policy claims,
  Shadow, promotion, threshold relaxation, and model-driven input remain blocked.
- 2026-08-15 T8-v2.1 controlled inverse-dynamics probe: a new non-formal
  `bounded_scrcpy_inverse_probe_v1` route completed three fully backend-operated runs without human
  keyboard input. The final independent 120-second run produced 40 balanced combat presses, 896
  causal samples, zero conflicts, and `0.250 ms` keyboard-to-socket p95. Four-class local-ROI
  recognition remained imperfect (`0.825` accuracy), with every residual error involving the
  skill3 region. The frozen engineering fallback therefore abstains from skill3 and admits only
  basic attack, skill1, and skill2 when the top-to-second ROI score ratio is at least `1.25`.
  Across the three runs this rule retained `23/39`, `23/39`, and `23/40` rows at `1.000` precision;
  the final shuffled precision was `0.000`. The final report status is
  `THREE_CLASS_PROBE_PASSED`, artifact SHA is
  `f0eff1df16eba42e1f3a1ff07794621b32074f0cfd0bcb77434666d9e8bdf390`, and policy training remains
  forbidden.
- 2026-08-15 three-class video candidate materialization: the already frozen 1,018-row strict-causal
  video artifact was re-read by hash and filtered without rescanning raw videos. Skill3 was removed,
  yielding 802 derived-RGB causal windows: train `52/281/196` and dev `11/162/100` for basic
  attack/skill1/skill2. No test data, event frame, future frame, raw video, or source path was read
  or persisted. The new manifest SHA is
  `de9d1f60623bc9c1617394b5a3ca3cffcdc118eefd1e0c253dc58607da5dbe28`.
  This authorizes only a three-class diagnostic learnability run; it does not override the failed
  full-policy Pilot or authorize formal training, Shadow, or device control.
- 2026-08-15 three-class strict-causal Pilot: the selected epoch-3 video adapter was frozen and its
  features were shared by a paired normal/shuffled seed-0 experiment. Dev selection used weighted
  cross-entropy only for 12 epochs. Both runs collapsed to the skill1 plurality: normal and shuffled
  accuracy were `0.593407`, macro-F1 was `0.248276`, per-class recall was `0/1/0`, and the
  normal-minus-shuffled macro-F1 gap was `0.000`. A follow-up equal-class resampling diagnosis over
  30 epochs still reached only `0.293174` macro-F1 at its best observed intermediate point and kept
  one class at zero recall. This rules out ordinary imbalance or too few optimizer steps as the
  main bottleneck. The paired report SHA is
  `64c31f66949b990c6ff748eee079024df6e82fa4e8a5ec8db9142d85ec8f9c0d` and status is
  `PILOT_DIAGNOSIS_REQUIRED`; video-test was not accessed. The practical maximum claim remains
  retrospective three-class action recognition. These video-derived labels do not provide a
  learnable next-action policy target, so additional seeds, larger models, formal policy training,
  Shadow, and model-driven device input remain blocked.
- 2026-08-15 retrospective three-class fallback: a separate 802-row artifact was materialized from
  the already frozen RGB windows with the dispatched action frame included; its manifest SHA is
  `f1512e011b0e9e338ef8fdb817491cf36d43a7f4112e9ff35f7066252b04757d` and video-test was not
  accessed. A frozen-center local-onset evaluator then reached dev accuracy `0.970696`, macro-F1
  `0.897317`, and per-class recall `0.636364/0.981481/0.990000`; the one-row shifted-label control
  fell to macro-F1 `0.282078`. The gate passed and authorizes automated retrospective annotation/QC
  for basic attack, skill1, and skill2. These metrics measure consistency with the frozen candidate
  labels, while independent precision support comes from the earlier three controlled inverse
  runs; therefore this result is not a causal policy score and does not authorize skill3, formal
  training, test access, Shadow, or device input. Report:
  `$HOK_LARGE_ROOT/runs/t8-policy-v2.1/video-retrospective-roi-v1/report.json`.
- 2026-08-15 retrospective v1 freeze and full batch: before further code changes, the exact
  recognizer source, CLI dependency surface, layout, v1-v7 probe configs, dataset manifest, probe,
  inverse evidence, and evaluation report were atomically frozen as 19 SHA-bound files under
  `$HOK_LARGE_ROOT/audit/t8-retrospective-v1`. Baseline verification
  passed with baseline SHA `d201fbd1ee5d8640d90f8b940fd8b728a597e5a0f89ba1072bd7c4cc2260ea59`.
  The new full-time-axis scanner then processed all 103 video-train and 23 video-dev derived-RGB
  sessions. It wrote 6,272 train and 1,697 evaluation-only dev events; all 126 session QCs passed.
  Accepted class counts across both splits are basic attack `339`, skill1 `4,064`, and skill2
  `3,566`. Independent output re-read verified every JSONL/QC hash, sequence, schema, count, and
  forbidden-field gate. The train/dev manifest identities are
  `08583416d352fcfebfc157a4bf781df42ceae7ba2dfdd9fa841e3699d8ff3669` and
  `fcdcae6db39bf2583c675d4f4d74cf869224e579bd9171f5aa941302d7330f08`;
  video-test was not opened, no RGB/source locator was persisted, and no device input was sent.
- 2026-08-15 retrospective v2 bounded calibration: 90 pre-registered ROI, temporal-reference,
  score, and confidence-threshold combinations were selected using video-train plus the first two
  inverse probes, with video-dev and the third inverse probe reserved for frozen validation. No
  three-class candidate met the precision-preserving basic-attack preselection gate, so the current
  v1 basic recognizer was retained rather than inflating recall by lowering precision. The best
  four-class candidate reached skill3 dev precision `1.000` and recall `0.850`, but independent
  inverse precision was `0.692308`; skill3 therefore remains abstained. The 2,418 skill3 candidates
  observed in the full train/dev scan were retained only as QC counts. Final calibration report SHA
  is `aabea65ad57f17b9c550a1b326e7fc2ec776308e549d01cc08dc95f7d5b2ed98`, status
  `BASELINE_RETAINED`, with `test_accessed=false` and no change to policy/Shadow/device-input gates.
- 2026-08-15 strict-causal four-class dataset: the frozen adapter encoded each target RGB frame once,
  then materialized three 16-frame feature-window variants ending 100, 200, or 300 ms before the
  detected event. Each lag contains 12,544 train rows (`none=6,272`, `basic_attack=275`,
  `skill1=3,218`, `skill2=2,779`) and 3,394 dev rows (`none=1,697`, `basic_attack=64`,
  `skill1=846`, `skill2=787`). Wait rows are deterministically sampled at least one second from any
  accepted event. Manifest identity is
  `0b041ad1e597aad8e96b1c1834f5ac861e1be97e4ac78b76b72a165785ac3e3b`; file SHA is
  `8ed8b7e1c5c7714008d2fb97a4548356def0c71ec708377cf899543d038bb247`. No event/future frame,
  test row, raw RGB, source locator, Shadow, or device input entered this artifact.
- 2026-08-15 strict-causal seed-0 pilot: all three frozen lags failed the normal-versus-negative-
  control gate. The best dev candidate was 300 ms with accuracy `0.497348`, macro-F1 `0.172163`,
  macro-recall `0.250304`, and per-class recall `[0.988214, 0.000000, 0.013002, 0.000000]` for
  `none/basic_attack/skill1/skill2`. Its macro-F1 advantages over shuffled-label and static-frame
  controls were only `0.005496` and `0.004287`; plurality accuracy was `0.500000`. Report SHA is
  `6659cf0a2f0dbf7292ebe261a95495e525d7b7b35b8db9a85d33415eeba7e0d3`, status
  `PILOT_DIAGNOSIS_REQUIRED`. Per the frozen rule, thresholds were not relaxed and three-seed
  training, video-test, Shadow, and model-driven input were not started.
- 2026-08-15 causal learnability decomposition: the selected 300 ms feature set was tested with
  four-class train fit, binary action-versus-wait, action-only three-class, and mixed-session row
  holdout probes using the same small TCN. Four-class train macro-F1 was `0.172596`; binary
  train/dev macro-F1 was `0.522250/0.508105`, only `0.013303` above shuffled labels and `0.006289`
  below the static-frame control. Action-only train/dev macro-F1 was `0.247347/0.240639`, with
  basic-attack recall `0.000000` and only `0.018854/0.017921` advantage over shuffled/static
  controls. The mixed-session holdout also predicted only `none`, so domain shift was not the
  explanation. All five branch gates are false, including `conditional_32_frame_allowed=false`;
  diagnosis is `NO_CAUSAL_LABEL_SIGNAL_IN_FROZEN_FEATURES`. Report SHA is
  `d69560284d4c4bfb849a6da206c312c622e89d5d22e248c4f47822bf558e5f59`. The planned 32-frame
  conditional model was correctly skipped; test, Shadow, and device input remained untouched.
- 2026-08-15 T8-v2.2 strict-causal raw-pixel probe: three RGB views were materialized for 6,272
  matched train events/waits and 1,697 matched dev events/waits, with no event/future frame or test
  access. Fine-tuning only ResNet-18 layer4 plus the probe head produced binary/action-only dev
  macro-F1 `0.507434/0.275973`; neither branch passed its shuffled, flat-frame, and two-second-shift
  controls. Dataset identity is
  `4fff110eba014972da88f53ca5e55ea7e290055c1b16fefcebbd226f6471ea14`; the probe report retains
  status `VISUAL_TEACHER_REQUIRED`. This rules out the frozen feature cache as the sole failure
  explanation but does not authorize more policy training.
- 2026-08-15 T8-v2.3 offline visual-teacher replay: train-only robust activity and three-button
  appearance calibration was validated once on frozen video-dev. Decisions were `none=1,666`,
  `basic_attack=567`, `skill1=652`, and `skill2=509`; deterministic repeat passed, all three actions
  exceeded 5% coverage, and current-versus-two-second-history agreement was `0.729523`, below the
  pre-registered `0.90` temporal-sensitivity ceiling. Status is `OFFLINE_TEACHER_READY`; report SHA
  is `ee78f640be2ba155fa4cb943a1411b8d99bec50e085287eabc66cf8d8446eec2`. It remains an offline
  video indexing/QC artifact with `training_eligible=false`, `live_execution_allowed=false`,
  `video_test_accessed=false`, and `device_input_allowed=false`.
- 2026-08-15 T8-v2.4 visible-onset audit: the implementation searches only the three 100 ms frames
  preceding each frozen retrospective peak and preserves explicit ROI/content, cross-button,
  timestamp, frame-hash, layout, calibration, and split bindings. The real audit scanned all 103
  train and 23 dev sessions. Independent inverse-holdout precision for basic/skill1/skill2 was
  `1.0/1.0/1.0`; train retained 5,259/6,272 events (`0.838489`) and dev retained 1,442/1,697
  (`0.849735`). The pre-registered gate nevertheless failed because dev ambiguity was `0.150265`,
  above `0.10`. Of the retained dev events, 1,409 remained at offset 0 and only 33 moved earlier,
  showing that peak-to-onset correction is not the main causal bottleneck. Status is
  `VISIBLE_ONSET_AUDIT_FAILED`; the downstream v2.4 materializer was verified to reject the report
  with no output directory, so no dataset, GPU pilot, video-test access, Shadow, or device input
  occurred.

### Lowered pilot gate

The one-seed pilot is admitted to full training only when all of the following hold on v2 dev:

- joint exact accuracy at least `0.30`;
- movement and combat macro-F1 each at least `0.55`;
- aim and hold macro-F1 each at least `0.45` on their active subsets;
- every represented active class recall at least `0.30`;
- joint exact accuracy at least `0.10` above the frozen plurality baseline;
- shuffled-label training fails materially, and the best strictly causal alignment does not
  require frames captured after the action.

Failure stops the run at data/alignment diagnosis. It does not authorize a larger model, test-set
tuning, threshold relaxation, Shadow, or device input.

### Lowered formal offline gate

The three-seed selected v2 model passes the one-time sealed evaluation only when all of the
following hold:

- joint exact accuracy at least `0.45`;
- movement accuracy at least `0.70` and combat accuracy at least `0.65`;
- aim accuracy at least `0.55` and hold accuracy at least `0.60`, each measured only where active;
- macro recall at least `0.50` for every learned head;
- predicted-versus-true action switch-rate error at most `0.25`;
- zero illegal action combinations after the frozen legality mask;
- shifted-label, plurality, black-frame, gray-frame, and OOD negative controls all pass;
- model/encoder/layout/dataset/split/config hashes bind exactly, with no raw video, source path,
  audio, reward, account identifier, or privileged field persisted.

These are lower capability thresholds, not lower safety thresholds. Missing classes, empty active
subsets, non-finite values, test access during selection, or a failed negative control are hard
failures and cannot be averaged away by high accuracy on `wait`, `none`, or `target=none`.

### Lowered read-only Shadow gate

After the offline gate passes, one uninterrupted five-minute Shadow must still satisfy:

- exactly zero tap, swipe, HID, minitouch, or other control output;
- at least `95%` scheduled-cycle coverage and inference p95 at most `150 ms` at 10 Hz;
- no illegal predicted combination after masking;
- confidence acceptance threshold fixed at `0.65` before the run;
- black/gray/OOD controls abstain at least `95%` of the time;
- no unbounded repeated action, and no disconnect, foreground-package change, display change, or
  capture failure.

Passing this Shadow establishes only a usable read-only v2 baseline. It does not admit autonomous
gameplay. Any later 20-action bounded probe remains a separately reviewed stage with the existing
serial/package/orientation checks, action cap, coordinate checks, and immediate stop behavior.

## Remaining blockers and next task

1. `V4-HARDWARE-ACCEPTANCE`: connect a read-only UVC capture card and run the frozen
   10-minute gate; no phone control connection is needed or allowed.
2. `V5-SOURCE-AND-ZERO-LABEL-DATA`: the current-contract three-seed source bundle is complete;
   the full-candidate, metadata-identified file-atomic component pass is in progress. If (and only if)
   the pre-ingest artifact is `READY_FOR_COMPONENT_SPLIT` with at least 12 clean components,
   freeze the all-clean 8/2/2 cohort, derive zero-redaction 128x128 RGB shards, and run the
   real-domain base chain. No manual action/frame/HUD/tracking/temporal label is requested or
   consumed.
3. `V6-ZERO-LABEL-TEMPORAL`: the RGB-only temporal framework is implemented but public output is
   structurally `ABSTAIN` until a frozen V5 zero-label base exists. After that base exists,
   train/save only RGB-derived tracking and temporal diagnostics; no keyframe or temporal audit
   label is requested and no advice-release path is active.
4. `POSTTRAIN-GAMEPLAY-PREFERENCE` (future, not implemented): after a base model is frozen, the
owner may rank complete PixelArena game pairs by gameplay quality. This is the sole allowed
   annotation phase, is offline and separately versioned, and cannot enable client control.
5. `T8-V2-VIDEO-ADAPTED-BC`: the automatically executed-action Pilot, earlier strict pre-action
   diagnostics, and the new 100/200/300 ms four-class diagnostic all failed their frozen negative-
   control gates. The verified retrospective detector remains useful for action indexing/QC, but
   the explicit causal experiment now shows that its visually detected event timestamp is not a
   learnable pre-action policy target. Further policy work requires a genuinely event-bound action
   source aligned to preceding RGB. T8-v2.5 now implements that replacement as a deterministic,
   checkpoint-free RGB-conditioned collector with a 100 ms decision-to-dispatch delay, single-copy
   10 Hz frame storage, 32-frame offline reconstruction, a 3-train/1-dev pilot split, and the later
   frozen 8/2/2 split. Its dry-run and staged live evidence have not yet been executed. Full-policy
   three-seed training, video-test, Shadow, and all trained-model input remain blocked.
6. V7 has no remaining frozen acceptance blocker. Further RichArena changes require a new
   versioned ruleset, new data, and a fresh acceptance run rather than changing this result.

Current maximum claim: the project-owned Rich PixelArena V2 RGB-only factorized ResNet-18
behavior-cloning agents passed the frozen three-seed RTX 4090 classification, negative-control,
closed-loop, replay/tamper, side-symmetry, illegal-action, and latency gates. V4 live hardware,
V5 zero-label real-domain alignment, V6 zero-label temporal diagnostics, and all future
post-training preference work remain externally blocked or fail-closed. This establishes no
Honor of Kings, GameCore, commercial-client automation, or out-of-environment capability.

## 2026-08-15 T8-v2.5 global unblock implementation

- Added explicit diagnostic admission for the failed v2.4 visible-onset artifact. This path is
  always marked `diagnostic_only`; even a positive metric cannot authorize three-seed training,
  Shadow, test access, or device input.
- Added `mobile-demonstrate-rgb-teacher-v25`. It freezes the accepted v2.3 numeric activity/ROI
  calibration, loads no model checkpoint, chooses only `wait/basic_attack/skill1/skill2`, and binds
  observation, decision, and guarded dispatch timestamps. Random scripted actions are not used.
- Added single-copy derived-frame shards, decision shards, session hashes, pilot/full split
  freezing, 32-frame/two-view reconstruction, class-balanced seed pilots, and shuffled/static/
  two-second-shift controls.
- Added Make targets for read-only dry-run, 20-action probe, one-minute smoke, five-minute formal
  collection, pilot/full split freeze, and seed-0 pilot. The live evidence state is
  `IMPLEMENTED_NOT_RUN`; no new formal v2.5 session or trained checkpoint is claimed.
- Live preflight found serial `<redacted-usb-serial>`, a valid 1600x720 `/dev/videoN` loopback, and the
  frozen `OFFLINE_TEACHER_READY` report with the calibrated layout hash. The phone display was
  `OFF` and the authorized package was not foreground, so the guarded dry-run was not started and
  zero wake/system/control inputs were sent.

### 2026-08-15 live dry-run and bounded probe evidence

- The host loopback was reloaded as `/dev/videoN` with `exclusive_caps=0`. Two consecutive
  five-second zero-input sessions completed, establishing that the V4L2 source can close and reopen
  without the earlier post-session capability loss.
- The 60-second zero-input dry-run completed in `61.020981` seconds with `300` samples, decision
  cycle coverage `1.0`, and exactly zero executed actions. It persisted derived RGB only and did
  not persist raw video, raw frames, device paths, or video-test access.
- The separately bounded 20-second probe completed in `21.02223` seconds with `99` samples and
  decision cycle coverage `0.99`. It executed four guarded actions: one `basic_attack`, two
  `skill1`, and one `skill2`, below the hard maximum of 20.
- All four dispatched rows satisfy `observation_end_timestamp_ns < decision_timestamp_ns <
  execution_timestamp_ns`. The run completed without a foreground/display gate failure, coordinate
  error, capture failure, or uncontrolled repeated input.
- This is diagnostic staged-loop evidence only. The session remains `formal_session=false`,
  `published_as_formal=false`, and `training_eligible=false`; it does not admit formal collection,
  policy training, test access, or Shadow by itself.

## 2026-08-16 T8-v2.5.1 live collection and v2.6 navigation diagnosis

- Reloading `/dev/videoN` with `exclusive_caps=0` fixed repeat-open failures. The scrcpy decoder
  now reconnects after a transient EOF and accepts at most two seconds of frame staleness before
  failing closed. Two consecutive five-second sessions, the 60-second dry-run, and later five-minute
  sessions completed without the earlier V4L2 capability loss.
- The live activity floor was frozen at `0.02` and a one-second global dispatch interval was added.
  The resulting 60-second v2.5.1 smoke completed with 297 decisions and 13 guarded combat actions
  (`skill1=4`, `skill2=9`) without an action-cap or continuous-input failure.
- One v2.5.1 formal session was atomically published at
  `datasets/t8-demonstrations-v2.5/rgb-conditioned-v1/session-001`: 301.121859 seconds, 1,488
  decisions, 43 combat actions, and complete counts `none=1445/basic_attack=13/skill1=19/skill2=11`.
  Its session identity is `b0d510f1779656f6de9a2688e6564b62116f120e4dc33720efca7808f5c3a797`.
- Repeated `session-002` attempts completed their full five minutes but contained 1,500-1,501
  `none` decisions and no combat action. Each was retained under `diagnostics` with
  `published_as_formal=false` and `training_eligible=false`; the formal `session-002` path was not
  occupied.
- A guarded warmup attack and bounded patrol were implemented as acquisition metadata outside the
  combat sample/event labels. Single-direction and four-direction patrols produced real scene
  motion (first-versus-last RGB MAE up to `0.121961`) but still encountered no main-view enemy cue.
- The experimental v2.6 path adds a conservative red-target cue (`red_pixels>=400` or horizontal
  red run `>=20`), real per-class cooldown arbitration, and RGB minimap navigation. The five-minute
  square-search diagnostic executed 61 bounded movement actions but retained 1,501 `none` combat
  decisions. A later 60-second minimap smoke derived all 12 movement directions from hashed RGB
  minimap player/target coordinates, but again observed zero enemy cues and zero combat actions.
- Current diagnosis: the live capture/control/navigation chain works, while the present scene does
  not expose a repeatable encounter target. Do not collect the remaining sessions, freeze a split,
  or train a policy from these empty attempts. The next admission condition is a repeatable
  target-bearing test scenario demonstrated by one 60-second run with all three combat classes;
  otherwise v2.6 remains diagnostic-only.

## 2026-08-16 mobile build-identity hold

- A read-only device audit found foreground component
  `<owner-testbed-package>/.SGameActivity`, version `<redacted-version>` (`versionCode=<redacted-version-code>`), first
  install time `2025-07-02`, last update time `2026-07-11`, and a roughly 571 MB base APK. The package
  manifest exposes Tencent push, Midas payment, QQ/WeChat authentication, vendor push, and other
  production-client components; no dedicated exported test-scenario Activity was found in the
  resolved activity table.
- This evidence conflicts with the repository's previous assumption that matching the official
  package name proves the installed artifact is the project-owned self-built test app. It does not
  independently prove ownership or provenance, so no further mobile input is authorized by the
  current evidence.
- All mobile input routes are suspended pending a frozen, independently verifiable project-owned
  build identity. Package, serial, foreground, and display gates remain necessary but are no
  longer treated as sufficient. Read-only V4L2 capture, existing-video batch processing, and
  offline evaluation remain permitted. Existing v1-v2.6 evidence is preserved and is not promoted.
- The video adapter is already `VIDEO_ADAPTER_SELECTED`; retrospective, causal four-class, and
  causal-pixel seed-0 experiments were already completed and correctly retained as diagnostic
  failures. Repeating those training jobs or collecting more empty patrol sessions would add no
  new evidence.

### Owner attestation and frozen identity closure

- The owner explicitly confirmed on 2026-08-16 that the observed artifact is the self-built test
  package. `configs/mobile_testbed_identity.local.json` now freezes package, version, observed
  signature identifier, owner attestation, and the pulled base-APK SHA-256. Its canonical identity
  hash is `<redacted-identity-sha256>`.
- `_open_device_guard` now checks installed version and signature against that frozen identity
  before reading the foreground/display identity. Guarded ADB and pinned scrcpy touch still fail
  closed if the build-identity gate is disabled, invalid, or drifting.
- A read-only live verification against serial hash source `<redacted-usb-serial>` matched the frozen
  identity. This restores only the already bounded mobile-testbed routes; duration, action count,
  foreground package, display, causal timing, and output restrictions remain unchanged.
- The first post-closure 60-second zero-input dry-run completed in `61.020838` seconds with 300
  samples, decision-cycle coverage `1.0`, zero input commands, zero errors, and combat counts
  `none=300/basic_attack=0/skill1=0/skill2=0`. Identity and capture are healthy, but the current
  scene still contains no admitted target cue. A new input patrol is therefore not justified by
  this evidence; the next live run remains gated on a target-bearing read-only admission.

## 2026-08-16 T8-v2.6 target-bearing admission and first formal session

- After enemy heroes and minions were added to the self-built scene, the previous red-cue rule
  still abstained on all 300 read-only decisions. Direct comparison of the derived RGB shards
  found a clean horizontal-red-run separation: the empty scene stayed in `7-9` pixels while the
  target-bearing scene stayed in `11-16`. The frozen row threshold was therefore corrected from
  `20` to `11`; a 10-pixel synthetic near-cue remains an explicit negative test.
- The corrected read-only admission produced `skill1=300` and `none=0`, proving that the target cue
  was continuously visible. The bounded 20-action probe then stopped at its cap after 25.301225
  seconds with complete combat coverage (`basic_attack=14`, `skill1=3`, `skill2=3`).
- The subsequent 60-second non-formal smoke completed with 283 decisions and 48 actions
  (`basic_attack=34`, `skill1=7`, `skill2=7`), no patrol/environment input, no cap failure, and no
  device/capture error.
- The first five-minute attempt was preserved as diagnostic evidence: 301.033446 seconds, 1,410
  decisions, 235 actions, and complete three-class coverage, but synchronous 100 ms causal waits
  reduced decision coverage to `0.94`, narrowly below the old `0.95` gate. Because 235 guarded
  actions alone consume at least 23.5 seconds, the contract now freezes a practical minimum formal
  coverage of `0.90`; duration, minimum 1,400 samples, causal timing, three-class coverage, event
  consistency, and all device gates remain unchanged.
- The replacement `session-001` was atomically published as formal and training-eligible: 301.021356
  seconds, 1,408 decisions, 235 actions (`basic_attack=173`, `skill1=31`, `skill2=31`), coverage
  `0.9386666667`, action-contract SHA-256
  `23caf7671df2f35fce5dd7fde5af54b779ca3cf81e36aa33d10cf23afd48b93e`, and session SHA-256
  `51c6f22172aac865fbd7f890872b0a68abce1ac048f7e5a3ae467051c883f0b9`. The existing offline
  session loader revalidated its manifest, hashes, frame shards, decision shards, and event contract.

## 2026-08-16 T8-v2.6 pilot cohort and seed-0 diagnosis

- Four additional five-minute sessions were atomically published and independently reloaded:
  `session-002` (1,407 decisions/235 actions), `session-003` (1,412/236), `session-004`
  (1,401/234), and `session-009` (1,412/233). Every session contains wait, basic attack, skill1,
  and skill2, shares action contract `23caf767...b93e`, and has no environment/patrol input.
- The pilot split was frozen without test access: train=`001/002/003`, dev=`009`, while `004`
  remains a future full-train candidate. Split SHA-256 is
  `8b6359603fc7419e64d3e43c149ebd44a18c5cd380e379c7cd46c4720acd7875`.
- The first seed-0 pilot initially exposed one loader bug after fitting: the final dev prediction
  omitted the supplied session-aware loader and attempted a legacy root-level `shards/` path. The
  one-line loader propagation fix passed all focused T8 tests; the failed run left no formal output.
- The corrected v1 report is `PILOT_DIAGNOSIS_REQUIRED`, not a global failure: normal four-class
  macro-F1 `0.439425`, action-vs-wait macro-F1 `0.727476`, and action-only macro-F1 `0.508988`.
  Normal four-class F1 exceeded shuffled labels by `0.209417`, static frames by `0.153767`, and
  2-second shifted frames by `0.176483`, establishing a real causal signal. The frozen gate failed
  because skill1/skill2 action recalls were `0.1875/0.34375`, below the `0.35` minimum. Report
  SHA-256 is `934fd4e36e9a7a9820e2d0415cdac47dc0422f54a6b0dcd5d8c79a59bf778098`.
- A separately preserved deterministic class-balanced sampling diagnostic did not fix the issue:
  four-class macro-F1 fell to `0.218768` and all three negative-control margins collapsed to about
  zero. Its report SHA-256 is
  `619dfdc217faa19f29733308441bf9805d8ed505b3b44b004aa992f6217124bb`; this route is rejected.
- Do not collect sessions 005-012, access test, lower the gate, or start three-seed training yet.
  The next minimal implementation is the already planned conditional combat contract: a binary
  action/wait head plus a three-class action head trained only on active labels, using the same
  frozen pilot split and all three negative controls.

## 2026-08-16 T8-v2.6 conditional-head pilot

- The v3 pilot replaces only the final classifier with a binary wait/action head and a conditional
  `basic_attack/skill1/skill2` head. Both heads share the same V5-initialized ResNet-18 layer4 and
  32-frame causal TCN. The action loss is evaluated only for active labels; the normal and shuffled
  runs use the same class weights, epochs, optimizer, frozen train/dev split, and controls. The
  rejected balanced sampler is disabled.
- The result materially improves action discrimination. Normal four-class accuracy is `0.716714`,
  action-only macro-F1 is `0.659598`, and action recalls are `0.899408/0.687500/0.375000` for
  basic attack, skill1, and skill2. All three action recalls now exceed the frozen `0.35` floor.
- The causal evidence is also stronger: normal four-class macro-F1 exceeds shuffled labels by
  `0.237606`, static frames by `0.176826`, and the 2-second time shift by `0.223922`. These exceed
  their frozen `0.15/0.10/0.10` margins, so the result is not explained by class priors or a single
  static frame.
- The strict pilot still remains `PILOT_DIAGNOSIS_REQUIRED`: four-class macro-F1 is `0.467909`,
  which misses the frozen `0.50` threshold by `0.032091`. Accuracy, action recall, and every
  negative-control margin pass; only this one criterion fails. The dominant residual error is the
  wait/action gate's 329 wait-to-action false positives, not loss of skill discrimination.
- The report is preserved at
  `runs/t8-policy-v2.6/pilot-seed0-v3-conditional/report.json` under the large-data root, with
  SHA-256 `8cc390ee1111f16fd198a66271b8fbc97e5deb181a108398d8b7bb2a864be6e9`.
  The model SHA-256 is `98bc19f5dcc785cbe81f7a355b4d7a94fbc94f0f22a1b4e62789f788d30fdda4`.
  Test remains unopened, Shadow remains disabled, and three-seed training is not yet admitted.
- Do not lower the gate or return to balanced sampling. The next bounded experiment should address
  binary gate precision while leaving the now-successful conditional action head and frozen data
  untouched; its threshold or loss rule must be frozen before another dev result is used for an
  admission claim.

### Rejected square-root gate-weight diagnostic

- A single predeclared v4 diagnostic reduced the binary gate's rare-class amplification from full
  inverse-frequency weighting to square-root inverse-frequency weighting while retaining the `0.5`
  decision threshold and every other v3 setting.
- Wait-to-action false positives fell from `329` to `96`, and four-class accuracy rose to
  `0.837819`, but skill2 recall collapsed from `0.375` to `0.0`. Four-class macro-F1 fell to
  `0.453974`, so the frozen action-recall and macro-F1 gates both failed.
- This loss-weight route is rejected and the active contract remains v3. The v4 report SHA-256 is
  `c5a5d1b12021d5766ca380ab724ecb19ea200cee29eae9aaa0bbf2b7147f4f72`; its model SHA-256 is
  `8742c8a2b5702a17e3c43e817498ffbe25b1f6dfc772fec5519c40de2b8fe796`.
- The next diagnostic must not retrain or perturb the successful shared/action representation. It
  should inspect the frozen v3 gate probabilities and determine whether one fixed decision
  threshold can reduce false positives without losing the `0.35` per-action recall floor.

### Gate-threshold selection and passing seed-0 pilot

- A read-only validation diagnostic evaluated the frozen v3 model on a predeclared `0.025` grid
  from `0.50` through `0.70`. Threshold `0.65` was the lowest value satisfying four-class
  macro-F1 `>=0.50` while retaining every action recall `>=0.35`; no test rows were opened.
- Threshold `0.65` was then frozen in code, model metadata, and future split manifests before one
  complete v5 normal/shuffled rerun. The run is `PILOT_REVIEW_READY` with four-class accuracy
  `0.795326`, four-class macro-F1 `0.503130`, and action-only macro-F1 `0.659598`.
- Basic attack, skill1, and skill2 action recalls are `0.899408/0.687500/0.375000`. Normal
  four-class macro-F1 exceeds shuffled labels by `0.275611`, static frames by `0.263391`, and the
  2-second shift by `0.235982`; all frozen accuracy, macro-F1, recall, and negative-control gates
  pass without lowering a threshold.
- The report SHA-256 is
  `937ffb122c3e8bbdb6c216d7c638a8a62417e17cbb4f0f9f88e43148d6d43d1c`; model SHA-256 is
  `13b128c379c1cc49527c611486ea5ce0ad8916a1c3f38ec49b9782e87d82cc2a`. The saved model metadata
  binds gate threshold `0.65`. Test and Shadow remain unopened/disabled.
- `three_seed_training_allowed` remains false solely because the current split is an explicit
  four-session pilot split. Formal 8/2/2 freezing requires sessions `001` through `012`; only
  `001/002/003/004/009` currently exist, so seven autonomous five-minute sessions
  (`005-008` and `010-012`) remain before the formal split and three-seed run.

## 2026-08-16 T8-v2.6 formal 12-session freeze

- The first `session-005` attempt completed safely with all classes but produced 1,394 samples,
  six below the frozen 1,400 minimum, and was retained as a non-training diagnostic. Formal
  collection duration was given a fixed five-second scheduling margin; all other gates and the
  300-action cap were unchanged.
- The replacement `session-005` and sessions `006/007/008/010/011/012` were autonomously collected
  and atomically published. Each ran about 306 seconds, contains 1,415-1,431 decisions, includes
  wait/basic attack/skill1/skill2, has decision coverage `0.9279-0.9384`, and records no environment
  or patrol input. No raw video, raw frames, device path, or manual annotation was persisted.
- All twelve required session identities are now frozen. The formal split is train=`001-008`,
  dev=`009-010`, test=`011-012`; split SHA-256 is
  `123b19636cf709faa20c171e50920629f9e5bb9eec57d7282f1cd50b4ebfda27`.
  The manifest binds gate threshold `0.65`, keeps `test_accessed=false`, and admits only train/dev
  to the existing loader.
- Mobile collection is complete and stopped. The next step is three independent seeds on the
  formal split, selected only by dev; test and Shadow remain closed.

## 2026-08-16 T8-v2.6 formal three-seed training

- Seeds 0, 1, and 2 completed independently on the frozen 8-session train and 2-session dev split.
  Every seed passed four-class accuracy/macro-F1, all three action-recall floors, and all shuffled,
  static-frame, and 2-second-shift margins. Every report records `test_accessed=false` and
  `three_seed_training_allowed=true`.
- Dev four-class macro-F1 is `0.549468` for seed 0, `0.700129` for seed 1, and `0.642840` for seed
  2. Dev four-class accuracy is `0.826102/0.876896/0.850441`. No test result was used to compare
  or select seeds.
- The frozen selection rule is maximum dev four-class macro-F1, so seed 1 is selected. Its model
  SHA-256 is `bce47dc1dc6332b7e348cfc6d6a9874efbbffadca14301dbfbe3bffa6063bd74`.
  The selection manifest internal SHA-256 is
  `54a51c85914948aad851de80c0d2d53f891aca663b4f893b446020ec5789d6e0`; the `selection.json`
  file SHA-256 is `3d64690c35d649aa7997478d187c55c89c101f65214b0a495b2c6720717a0824`.
- Formal training is complete. The next allowed operation is one sealed offline evaluation of the
  selected seed-1 model on sessions `011/012`. Shadow remains disabled until that report is frozen
  and passes its separate admission checks.

## 2026-08-16 T8-v2.6 sealed offline test evaluation

- A dedicated v2.6 evaluator validated the frozen split, selection manifest, selected model hash,
  gate threshold, and exact test identities before opening any test shard. Contract-failure tests
  prove that invalid metadata is rejected before session access. No training or threshold tuning
  occurred after the test was opened.
- The selected seed-1 model passed the one-time evaluation on sessions `011/012`: four-class
  accuracy is `0.886428`, four-class macro-F1 is `0.738686`, and action-only macro-F1 is
  `0.883416`. Basic attack, skill1, and skill2 recalls are
  `0.915452/0.863636/0.984848`, all above the frozen `0.35` floor.
- The causal controls pass by large margins. Static-frame four-class macro-F1 is `0.221535` and
  the 2-second-shift macro-F1 is `0.251919`, giving normal-minus-control margins of
  `0.517151/0.486768`. Predicted versus true switch rates are `0.320549/0.333568`, an absolute
  error of `0.013019` against the frozen `0.10` ceiling.
- The immutable report is
  `runs/t8-policy-v2.6/formal-v1/offline-test-v1.json` under the large-data root, with SHA-256
  `b9f8d42cbd8fcd859d369de832e3895272b72d47d69a951b5da361bcc0486fcf`. It records
  `test_accessed=true`, `strict_passed=true`, no persisted raw video/path data, no device input,
  and `shadow_allowed=true`.
- Offline evaluation is complete and must not be rerun or used for tuning. The next admitted stage
  is one five-minute read-only Shadow run with the frozen seed-1 model and gate `0.65`; any Shadow
  failure is diagnostic and must not alter the sealed offline result.

## 2026-08-16 T8-v2.6 read-only Shadow

- The first live attempt (`shadow-v1`) failed safely during V4L2 startup and preserved a diagnostic
  report with zero input commands. A second five-minute live attempt (`shadow-v2`) completed all
  `3000/3000` scheduled cycles with zero control output, but the foreground scene remained static:
  the model predicted wait throughout. Its report is diagnostic-only and failed the frozen switch
  and latency gates; it is not used to change the model or thresholds.
- Live profiling exposed two runtime defects independent of model quality: the schedule clock began
  before the video stream was ready, and each prediction redundantly encoded all 64 view frames.
  The runtime now starts timing after stream readiness, uses the existing background device-guard
  watchdog, computes entropy in float32, and caches the latest 32 encoded two-view features. The
  cached predictor is numerically equivalent to full-window inference and benchmarks at about
  `2.90 ms` mean / `2.93 ms` P95 on the host GPU.
- Because a zero-input static phone page cannot be compared meaningfully with active demonstration
  events, the formal read-only Shadow uses the original contract: replay one sealed five-minute
  demonstration session, predict only, and compare against its recorded executed events. No phone,
  raw frame, or model-driven device input is involved.
- The sealed `session-011` replay processed `1425` events and passed every frozen gate. Four-class
  accuracy is `0.896140`, macro-F1 is `0.787055`, and basic attack/skill1/skill2 recalls are
  `0.755814/0.878788/0.909091`. P95 prediction time is `0.003639` seconds, confidence coverage is
  `0.973333`, the explicitly limited predictive-entropy proxy inlier ratio is `0.984561`, and
  switch-rate error is `0.014747`.
- The formal Shadow summary is
  `runs/t8-policy-v2.6/formal-v1/shadow-replay-v1/summary.json` under the large-data root, with
  SHA-256 `4f51a0a5279c97400f8e08a213745d05e9ac6f0667861ac5521c0259d1916257`.
  It records `strict_passed=true`, `input_commands_sent=0`, and `control_output=false`. The OOD field
  remains honestly scoped as a predictive-entropy proxy, not a trained OOD detector.
- Offline evaluation and read-only Shadow are now complete. A trained-model device-control probe is
  a separate high-risk stage and remains unexecuted pending its own v2.6 safety contract and review.

## 2026-08-16 T8-v2.6 bounded-probe contract

- Added a dedicated v2.6 admission chain that revalidates the canonical selection manifest, sealed
  offline report, sealed replay-Shadow summary, split, model, and layout before opening a device
  input pipe. A changed hash, non-passing report, prior Shadow input, or mismatched layout fails
  before capture or action execution.
- The probe is fixed at 60 seconds and at most 20 actions. Its entire executable vocabulary is
  `basic_attack`, `skill1`, and `skill2` taps at the frozen 1600x720 layout coordinates. Movement,
  skill3, aim, hold, target intent, and fallback/corrected candidates are absent from the path.
- Dispatch additionally requires four stable predictions, confidence `>=0.45`, normalized
  predictive entropy `<=0.80`, a 500 ms minimum interval, and no more than three consecutive
  identical taps. A background watchdog checks serial/package/display continuously, and the
  synchronous guard runs again immediately before every tap.
- Focused tests cover evidence binding and the three-action-only surface. Execution remained blocked
  until the full repository check and live preflight passed.

### First bounded-probe result and scene-readiness gate

- The repository check passed before execution (`218 passed`, zero lint/type/boundary findings), and
  the live preflight matched the frozen selection, offline, Shadow, model, split, layout, serial,
  foreground, and 1600x720 rotation identities.
- The unique `probe-20-v1` run completed 61.030810 seconds and 599 inference cycles without a guard,
  capture, coordinate, or vocabulary error. All 599 candidates were `none`, so the fail-closed
  executor sent zero actions and recorded `control_output=false`. P95 scheduled-to-decision latency
  was `0.017545` seconds. The failed summary SHA-256 is
  `53e7302acef929cbdd2a6031c5f54bf54825fccfbeac49e2a2369789df0f6c77`.
- An independent frozen-rule zero-input dry-run then also produced `none=300` and zero combat
  candidates/actions. Its summary is
  `datasets/t8-demonstrations-v2.6/rgb-conditioned-v2/diagnostics/dry-run-1786880717/summary.json`
  with SHA-256 `2e04fbea2f41d33a5198bc9251423f36f9c146802a1615581b76c685bd947279`.
  This proves the immediate blocker is the current empty/static live scene, not confidence,
  entropy, latency, device transport, or a relaxed action gate.
- Future probes now have an additional five-second zero-input readiness phase. At least one stable,
  confidence/entropy-admitted non-wait combat candidate must appear before any tap is possible;
  otherwise the run stops immediately. No retry is justified until the self-built app visibly
  restores the target-bearing hero/minion scenario used for the successful formal sessions.
- After the phone was unlocked, `probe-20-v2` exercised that new gate: foreground/display/video
  checks passed, but all 48 warmup candidates were still `none`. It stopped after 6.051925 seconds
  with `scene_ready=false`, zero inputs, and no coordinate or vocabulary error. Its summary SHA-256
  is `c6c8247672452474cb70278d93fb9f09a0f5e25e582d1400175b9491bf13aba9`. Unlocking the phone is
  therefore insufficient; the target-bearing hero/minion test scene must be visibly active.

## 2026-08-16 T8-v2.7 current-scene head calibration

- A current enemy-bearing scene was independently confirmed by the frozen RGB teacher, while the
  selected v2.6 model predicted `wait` on every one of the same 50 causal windows. This isolates a
  live-scene domain shift rather than a stream or device-control failure.
- Two diagnostic sessions were kept separate: the 60-second run is calibration-train and the
  20-action run is calibration-dev. ResNet-18 and the causal TCN remain frozen; only the conditional
  gate and action heads are fitted. Test, Shadow, and device input are disabled for calibration.
- The first pilot reached dev four-class accuracy `0.834783` and macro-F1 `0.539568`, but failed the
  frozen negative-control gate because normal training did not beat shuffled labels. Its immutable
  report SHA-256 is `7106bc9a76d68d66a538d6cbb20bfc4baabb1c4604d714de2729fe45250b3144`.
- Enabling the existing four-class balanced sampler did not resolve the evidence problem: accuracy
  remained `0.834783`, macro-F1 was `0.455366`, skill2 recall was `0`, and the shuffled-label margin
  remained `0`. Its report SHA-256 is
  `ecefe694454731273ecca0069759aedfeede6ea43dda165ba5bd20a7a9946c8d`.
- The next admissible step is therefore new automatically generated, class-balanced current-scene
  train/dev evidence, not threshold relaxation or another fit on the same tiny sessions. The RGB
  teacher collector now has an opt-in diagnostic-only `--balanced-actions` policy: among classes
  whose cooldown has expired it dispatches only a least-executed class, while preserving the same
  serial, foreground, display, causal-delay, global-rate, and maximum-action guards. It is rejected
  for formal sessions and records its dispatch policy in the hashed action contract.
- The first balanced collection preflight stopped before capture and sent zero actions because the
  owner-authorized package was no longer foreground. No calibration result currently authorizes
  test access, Shadow, or trained-model device input.

### Balanced current-scene evidence and causal diagnosis

- After the owner-authorized scene returned to the foreground, the balanced collector completed an
  independent train session with `573` decisions and `44` executed actions. Its action counts are
  basic/skill1/skill2=`15/14/15`; it sent zero environment actions. The summary SHA-256 is
  `d9d79d986f19e198ec515f92e38d92b87e2c212bc91ab39f59ecd721cc0c726f`.
- The separate dev session stopped safely at its fixed cap with `250` decisions and exactly `7/7/7`
  executed actions. Its summary SHA-256 is
  `0ae5db7930d868945c40579f39da799f527c31d49f53de6d4c189d23c7a0793d`.
- Head-only calibration on those sessions did not pass: four-class accuracy/macro-F1 are
  `0.592000/0.302919`, action-only macro-F1 is `0.462745`, and the normal-minus-shuffled margin is
  only `0.021077`. The immutable report SHA-256 is
  `818eaa33e35a9e3c9600eb8473e1c9f1048aeb4cf1180f6d6e4f7d24e065fcbb`.
- This is a causal-contract failure, not a request for more of the same data. The balanced teacher
  selects the next skill from hidden per-class execution counts and cooldown state, so visually
  equivalent pre-action windows legitimately receive different skill labels. RGB-only heads cannot
  recover that hidden scheduling state, and label shuffling therefore changes little.
- The next model contract must expose bounded action-history/cooldown state to the conditional head,
  or keep skill selection as a deterministic legality/priority module while learning only the
  visually observable enemy/action gate. More RGB-only head fits, threshold relaxation, test access,
  Shadow, and model-driven device input remain disallowed.

## 2026-08-16 T8-v2.7 freeze and T8-v3 video-state closure

- T8-v2.7 is now mechanically frozen as `FROZEN_FAILED`. The freeze binds the three immutable
  calibration reports, sets rerun/threshold-change/four-class-training/Shadow/device-input flags to
  false, and names T8-v3 as the successor. The artifact is
  `audit/t8-v2.7-frozen-failure-v1/freeze.json` under `HOK_LARGE_ROOT`; its file SHA-256 is
  `95a8fb698df4da53ede8e7ddeb63b74fa23f4a4f2d710747b587913fed87f9d7` and its internal canonical
  freeze identity is `ccdcc083c985f322edf78d0ea09b635b9788de57840691d75af9920b3c18e5ee`.
- The T8-v3 dataset contract predicts only the RGB-observable states `enemy_visible`,
  `attack_opportunity`, `basic_ready`, `skill1_ready`, and `skill2_ready`; confidence and abstain
  are derived outputs. Skill choice remains in a deterministic boundary with priority
  `skill2 -> skill1 -> basic_attack`, fixed per-action cooldowns, a one-second global interval, and
  a three-identical-action cap. Structured state is not an Actor input.
- Materialization reused the frozen V5-initialized 16x512 causal features and the frozen RGB teacher.
  It read exactly 103 video-train and 23 video-dev sessions, never video-test, and wrote 12,544 train
  plus 3,394 dev rows. Every state has positive examples in both splits. The dataset is
  `datasets/t8-video-state-v3`; canonical manifest identity is
  `9efb46e5ae20e30540c5e4f9e57ccbbdcb8eacb7acf3a5ec58138976baa5a41c` and the manifest file SHA-256
  is `1aae53d1a6df514bb9b46b3bdd3690145eb4fbda47336ce1df7eb5b0bc22e503`.
- The one allowed seed-0/8-epoch GPU run completed. Its mean dev head macro-F1 is `0.450962`, minimum
  positive recall is `0.314075`, normal-minus-shuffled mean macro-F1 is `0.023275`, and confidence
  coverage is `0.039481`; the frozen requirements were `0.70`, `0.55`, `0.15`, and `0.50`.
  Black/gray OOD abstention was `1.0` and post-mask logical violations were zero, but those two
  successes cannot override the four failed gates. Status is `V3_STATE_PILOT_FAILED`.
- The immutable run report is `runs/t8-policy-v3/state-seed0-v1/report.json` with file SHA-256
  `b63613a1c0d75257da3f476e29d18a3a6841dfe2f373d79cefc56a26603d8dd2`; the rejected model file
  hash is `fcf0517e964ea1af31d441689942c096201b95455d4387956aea10d19616c189`.
  The explicit hybrid-replay invocation was rejected with `T8-v3 training did not admit offline
  replay` and created no replay directory.
- Per the frozen stage order, the five-minute read-only Shadow, 20-action probe, one-minute run, and
  five-minute run were not started. Device input commands sent in this T8-v3 closure are zero;
  video-test remains unopened. No repeated collection, threshold adjustment, additional seed,
  replay, Shadow, or live expansion is admitted from this failed lineage.
