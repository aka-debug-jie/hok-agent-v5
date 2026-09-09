> Historical archive only. Current work is defined at
> [DELIVERY_PROGRESS.md](../DELIVERY_PROGRESS.md#current-execution-state).
> The 2026-09-09 planning snapshots are at [planning snapshot](#planning-snapshot-20260909).

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

<a id="planning-snapshot-20260909"></a>
## Planning snapshot, 2026-09-09

The following three files are copied verbatim from commit
`727d36098065d7f9532f2ead2476d45654fd126d` before the development-plan reset.
They preserve historical commands, counts, hypotheses, restrictions and next-step wording.
They are not current instructions; consult the short AGENTS and convergence plan instead.
Literal fenced text preserves the original relative links without making them live links from here.

### Snapshot: AGENTS.md

````text
# AGENTS.md

`DELIVERY_PROGRESS.md` is the only current-state ledger. Read it, `README.md`, and
`BOUNDARIES.md` before changing this repository.

## Current planning authority (2026-09-05)

User-confirmed translucent joystick feedback authorizes a separate visibility pilot: the existing
two native-player train/dev sources, three fixed four-second windows each, lower-left native RGB.
`movement_real_rgb.py` may decode these windows with existing PyAV and write offline Pillow QA.
No action labels or training follow automatically; no test, mobile input or old detector retuning.
The subsequent authorized extractor may use only the six cached joystick windows and OpenCV.
Train-only templates are frozen before dev extraction; results remain candidate UI-state labels,
not validated policy targets. No new raw-video decode, model, phone or test access is required.
The authorized coverage continuation freezes geometric extractor v3 and samples twelve additional
four-second windows from the same existing train video only. No calibration or model training;
no dev/test decoding. Native RGB stays in memory; only candidate records/hashes and QA persist.
The eligibility continuation may read only the completed coverage JSON. It may count stable
direction runs, explicit direction-to-center release events and PTS gaps, but writes no policy
RGB/sample/model. The current frame is the target; Actor input must end at the preceding PTS.
The cross-train transfer audit may open only four predeclared train identities, at 20/50/80%,
using the frozen v3 template/fingerprint. It persists QA, hashes and candidates, not native RGB.
No dev/test source, threshold change, calibration, policy sample or training is authorized.
The final bounded expansion may open only the three remaining predeclared train identities at
10/30/50/70/90%. It uses the same frozen extractor and then closes train-source expansion.
That expansion passed minimum automatic weak-label support across eight train sources. The next
step may materialize only a small diagnostic dataset: stable two-frame directions plus explicit
release STOP, with 16 Actor frames ending before the label and joystick pixels excluded. It may
not train or promote a model until the materialized dataset passes causal/hash/split validation.
The resulting 32-sample dataset has passed those mechanical validations. One fresh-init diagnostic
overfit is allowed with the existing task-specific GroupNorm+GRU only. It cannot count as formal
training, generalization, semantic accuracy or checkpoint promotion; no architecture/threshold loop.
That single diagnostic passed and is consumed. Its checkpoint remains diagnostic-only. The next
step may materialize one fixed six-train/two-internal-dev source-grouped pilot from the existing
frozen candidate reports. It cannot load the diagnostic checkpoint or open video-dev/test.
The grouped pilot split is fixed: dev identities `0e34a785...` and `12214351...`, with the other
six train-cohort identities retained for pilot training. Materialization takes one stable-run
onset per direction plus release STOP; it stores masked causal Actor RGB and no source paths.
That 73-train/25-dev dataset is mechanically validated with all nine classes and zero source
overlap. A separately contracted fresh-init grouped pilot may run next with class-balanced train
sampling. It must report macro-F1 and every recall; no overfit checkpoint reuse or promotion.
The single grouped pilot is now frozen failed: train accuracy 1.0 but internal-dev accuracy 0.16
and macro-F1 0.0974. Do not rerun/tune this dataset or architecture. A continuation may only
scale frozen-extractor evidence on a predeclared set of additional existing train videos before
another model contract; original video-dev/test, phone and diagnostic checkpoints remain closed.
The next scale audit is fixed to the 24 anonymous train identities in
`configs/joystick_scale24_audit.json`, five four-second windows each. It uses frozen extractor v3,
writes no native RGB/model, and closes before any dataset materialization or training.
Its strict first run exposed 11 portrait sources. The sole geometry repair retained the same 24,
excluded portrait metadata mechanically, added no replacements, and completed 13 landscape
sources. Together with prior evidence, 21 train sources now permit one grouped dataset build;
extractor/model tuning and another source scan remain closed.
The 21-source dataset split keeps prior internal-dev `0e34...`/`1221...` and adds the first,
middle and last compatible new identities by sorted hash (`1720...`, `3927...`, `493c...`).
This 16/5 split is chosen without label-based source selection. Materialization remains model-free.
The 201/48 dataset passed validation. One scale21 pilot may use the same 686k model, seed,
optimizer, balanced sampling, 30 epochs and gates as the failed 73/25 pilot. Samples per epoch
tracks dataset size (201). Both prior checkpoints are forbidden and the attempt limit is one.
That scale21 pilot also failed: dev accuracy 0.1667/macro-F1 0.1465. No retry, model or extractor
tuning. The next work may only audit frozen label timing semantics by separating stable-run onset
from continuation frames. It writes no RGB/model and cannot reinterpret UI labels as intent.
The continuation audit defines an observable direction target only at the third or later equal
joystick candidate, so two prior same-direction frames precede the label. STOP remains a
deterministic Router action because centered-screen semantics are unresolved. Audit is JSON-only.
That audit passed with 203 train/80 dev direction-continuation targets. The next step may
materialize these frozen indices as an eight-direction dataset with the same 16/5 source split.
It may not include STOP, change the extractor, train a model or open original video-dev/test.
Materialization must bind the audit and five candidate reports, preserve the fixed16/5 sources,
write 203/80 masked causal clips with action order `N,S,W,E,NW,NE,SW,SE`, and verify all hashes.
That dataset is now validated with283 unique clips. One fresh-init continuation pilot may reuse
the existing spatial encoder/GRU architecture with a newly initialized eight-output head and
class-balanced sampling. No prior weights; deterministic Router retains STOP.
The continuation pilot contract fixes seed0, AdamW1e-3, batch8,30 epochs,203 balanced samples
per epoch and the prior selection rule. Gates are dev accuracy/F1>=0.35, six nonzero recalls,
majority-F1 gain>=0.20 and every dev source accuracy>=0.20. One attempt; no fallback.
That pilot failed at dev accuracy0.25/macro-F10.1897. No retry or model tuning. The next work may
only audit the deterministic previous-direction persistence baseline on the frozen continuation
manifest. This uses execution state outside Actor RGB and sends no input.
The baseline must reproduce every manifest sample from the two prior equal candidates and verify
the existing executor maps same previous/current direction to `KEEP`. It may not decode RGB,
load a checkpoint, choose a new direction or claim gameplay performance.
That audit passed exactly on203/80 samples and all eight executor mappings return `KEEP`.
Learned continuation and its checkpoints are now closed. Future Movement learning must target
direction change conditioned on a separately observable Macro goal; persistence remains deterministic.
The next contract may reuse synthetic hollow-goal-ring RGB only. The model vocabulary is eight
directions and invocation occurs only on Macro goal-version change, stuck recovery or respawn
reset. Executor owns persistence; Router owns STOP/unknown/terminal. No training or phone input.
The change-event dataset is fixed to64/24 episodes, four Macro goal changes each, 15 old-goal
frames plus one current new-goal frame, and balanced eight-direction labels. It is simulator-only,
contains no STOP/previous-action input, and must be validated before any training.
That dataset passed with256/96 balanced samples and352 unique clips. One fixed simulator pilot
may compare the fresh 8-head GroupNorm+GRU with a last-frame control. Model selection favors the
simpler control when performance matches. No prior weights or real-domain claim.
The comparison fixes seed0, AdamW1e-3, batch32,30 epochs and six evaluation points. Both models
must reach dev accuracy/macro-F1>=0.95 and each recall>=0.90. Within0.01, select the 587080
parameter last-frame control. One attempt per model; simulator integration only.
Both models passed; the 587080-parameter last-frame checkpoint is selected for simulator-only
integration. The next step may load only that exact checkpoint in an offline multi-waypoint replay.
Policy invocation/KEEP/STOP ownership must match the change-only contract; no real RGB or input.
The replay is fixed to10 episodes, four two-cell waypoints and three steps per waypoint:
model change, executor KEEP, Router STOP. It must load only the selected checkpoint and reach
all40 waypoints with40/40 predictions,40 KEEP and40 STOP. No transition to real RGB.
The formal route preflight failed20/40 with E/W at0/10 each and executed zero arena steps.
Do not retry this checkpoint or change routes. One simulator-data correction may vary E/W across
rows2/3/4 and create a position-held-out dev split; materialize/validate before training.
The v2 data correction keeps256/96 balanced samples. E/W cover rows2/3/4 in both splits;
train player x is6/7/8 and dev x is5/9 with zero position overlap. All other contracts remain
unchanged. No model run until v2 data validation passes.
V2 data now passes with zero position overlap and352 unique clips. One fresh last-frame-only
candidate may reuse the prior seed, optimizer,30 epochs and gates. The GRU comparison is not
repeated because replay failure isolated spatial position coverage. No old checkpoint.
If the single v2 last-frame attempt passes, bind a new replay contract to its report/checkpoint;
the original multi-waypoint routes remain unchanged. No threshold or route repair.
That attempt failed at train1.0/dev0.2083 and is consumed; no replay or model/architecture retry.
The next work may only audit the deterministic `goal_canvas_geometry_movement` baseline on v2
data and unchanged routes. It cannot claim real-RGB observability or authorize input.
The geometry replay must score352/352 on v2, then run the unchanged10 episodes with40 geometry
changes,40 executor KEEP,40 Router STOP and120 arena steps. It loads no model and preserves
simulator-only/zero-input boundaries.
It passed exactly. Simulator Movement now selects deterministic goal geometry + executor KEEP +
Router STOP; all neural change/continuation checkpoints remain rejected. The next step may package
this offline simulator evidence only. Real continuation requires a new player/goal visual source.
The resulting 201-train/48-dev dataset is validated with all classes, zero source overlap and
249 unique masked causal clips. One fresh-init scale21 pilot may reuse the prior optimizer,
balanced sampling, epochs and gates exactly; no failed checkpoint, extractor or split changes.

The user-authorized short-gap optical-flow experiment supersedes future-work restrictions only
for one offline run on existing teacher sessions 002/003/005. `movement_real_rgb.py` and its
focused tests may lazily import OpenCV for this run. No historical detector thresholds change;
no Movement training, raw-video decoding, phone or test access is authorized. Houyi identity
is not required for this generic localization experiment. Old outcomes remain immutable.

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
That 12-session raw-video preflight is complete and frozen domain-mismatched: the normalized mobile
top-center cue repeatedly selects scoreboard/kill/persistent red UI, while center-health geometry
is not cross-layout stable. Eleven paired automatic outputs are rejected by QA and one is unpaired.
Do not repair thresholds, materialize death clips, train Reward, or branch to tower/economy weak
labels without a new semantic source. Keep the zero-reward Event-to-Store baseline.
The cycle is packaged and independently verified as `R1_ENGINEERING_OFFLINE_ZERO_REWARD`: nested
R0 rule evidence, 284 non-training Event transitions, four frozen failure reports, zero reward,
zero input and no checkpoint. Do not reopen this cycle. Future work requires a new semantic source
or separately authorized hero-bound data contract; packaging is not policy promotion.
The first Houyi-bound data audit is frozen unavailable: train/dev container metadata has zero hero
keywords, the hero profile is unconfigured, and no real session carries an episode-scoped immutable
Houyi binding. One simulator declaration is not real evidence. A prior exploratory pass opened 23
test container headers but decoded zero frames and was unused; formal audit opens zero test. Do not
retroactively label existing video. Future hero-bound work requires declaration plus reference
evidence before collection and cannot authorize input by itself.
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
  `movement_real_rgb.py` may decode the cohort-bound landscape train/dev sources selected by
  `native-player-pilot` and the fixed 8-train/4-dev death-cue preflight, crop before resizing, and
  persist derived offline QA windows. It cannot decode test, connect a device, create action or
  semantic death labels, or train from these windows.
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
````

### Snapshot: README.md

````text
# HoK-Agent V5

[![CI](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml/badge.svg)](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

HoK-Agent V5 is an RGB-only MOBA research testbed built around project-owned PixelArena
environments, causal video models, offline Shadow evaluation, and fail-closed mobile-testbed
interfaces. It does not call a game-internal API and is not affiliated with or endorsed by a game
publisher.

The public repository intentionally contains no authorized mobile build identity, calibrated
coordinates, device serial, recording, dataset, checkpoint, or run artifact. Mobile input is
disabled unless the operator supplies local private evidence for a self-built test app and every
runtime guard passes.

## Current development plan

The zero-parameter geometry baseline and its immutable delivery package pass. The corrected
dataset is352/352 and the unchanged multi-waypoint replay is10/10 with40 changes,40 KEEP,40 STOP
and120 arena steps. Simulator Movement therefore uses deterministic goal geometry; learned
continuation/change checkpoints remain rejected. Real deployment still needs real player/goal
observability.

The position-held-out v2 model failed: train accuracy1.0 versus dev0.2083/macro-F10.1987, with
four zero-recall directions. Complete E/W row coverage did not overcome absolute-position
memorization. The checkpoint and neural simulator change branch are rejected for this cycle.
Next is a zero-parameter geometry baseline on v2 data and unchanged routes.

The position-corrected simulator dataset v2 is valid. E/W now cover rows2/3/4; train current x
is6/7/8 and dev x is5/9 with zero position overlap. Size and class balance remain256/96 and
32/12 per direction. Next is one fresh last-frame-only candidate; the GRU control is not repeated
because replay isolated a spatial coverage shortcut.

The selected simulator checkpoint failed multi-waypoint preflight20/40 and executed zero formal
arena steps. E/W are0/10 because the dataset placed every E/W example on row3; routes at rows2/4
expose a position shortcut. The checkpoint is rejected. The next step changes simulator data only:
cover E/W on rows2/3/4 and use position-held-out dev before any retraining.

The simulator change-policy comparison passes. GRU and last-frame models both reach train/dev
accuracy and macro-F11.0 with all recalls1.0. The predeclared simplicity rule selects the587,080
parameter last-frame checkpoint; GRU adds99,072 parameters without performance gain. This model
is simulator-only. Next is an offline multi-waypoint integration with Router STOP and executor KEEP.

The simulator change-event dataset is valid: train256/64 episodes and dev96/24, with eight
directions exactly balanced and no episode overlap. Each sequence shows the old goal for15 frames
and the new observable Macro goal in the current final frame. STOP and previous action are absent.
Next is one fresh simulator comparison between the 8-head GRU and a last-frame control.

The new change-only Movement contract passes. A model may emit only eight directions and is called
only when a Macro goal version changes or recovery requests reconsideration. The executor owns
`KEEP`; Router owns goal-reached/unknown/terminal `STOP`. The next step is a fresh simulator-only
change-event dataset with an observable hollow Macro goal ring, before any new training.

Direction persistence is now solved deterministically: previous-direction prediction is exact on
all203 train and80 dev continuation targets, and the existing executor maps every same-direction
request to `KEEP`. Learned continuation is therefore closed. Future Movement learning should only
propose direction changes from an observable Macro goal; Router retains STOP.

The eight-class continuation pilot also failed: train accuracy1.0, internal-dev accuracy0.25 and
macro-F10.1897, with four zero-recall directions. No retry or model tuning is allowed. Because
every continuation target is defined after two prior equal directions, the next step is a JSON-only
deterministic previous-direction baseline. This may show that learned continuation is unnecessary.

The eight-direction continuation dataset is now valid: train203/16 sources, internal-dev80/5,
source overlap0 and283 unique masked causal clips. It contains no STOP; deterministic Router owns
stopping. One fresh-init continuation pilot may use the existing spatial encoder/GRU with a new
eight-output head. Its scope is maintaining established motion, not choosing a new tactical turn.

The onset-versus-continuation audit now passes an eight-direction continuation target: 203 train
and80 internal-dev samples have two preceding same-direction UI frames. STOP is removed from the
learned action space and remains deterministic Router behavior because centered-screen semantics
are unresolved. The next step is dataset materialization only; this target teaches continuation,
not human turn onset or tactical intent.

The 201/48 scale21 pilot also failed source generalization: dev accuracy0.1667 and macro-F10.1465,
with four zero-recall classes and accuracy below the majority baseline. More samples helped only
slightly over the 73/25 pilot. Current labels select action-run onsets; the next step is a no-training
audit of whether continuation frames form a more observable target. Failed checkpoints stay closed.

The larger grouped dataset is now valid: 201 samples from16 train sources and48 from5 internal-dev
sources, all nine classes, zero source overlap and249 unique masked causal clips. The dev groups
were chosen by prior membership plus sorted anonymous hash positions, not action labels. One
fresh-init scale21 pilot may now reuse the previous model/training contract exactly; failed
checkpoints and original video-dev/test remain closed.

The frozen extractor now covers 21 train sources after a fixed scale audit. A strict 24-source
run exposed 11 portrait inputs; the sole repair mechanically retained the 13 landscape sources
without replacements. Aggregate evidence is 4,160 frames, 1,210 candidates and 27 release STOP
events. Source expansion is closed; next is one larger grouped causal dataset, not model tuning.

The frozen extractor now has evidence across 21 train sources: 4,160 sampled frames, 1,210
candidates, 27 release STOP events and every direction supported by 8–16 sources. The fixed
scale24 audit excluded 11 portrait sources by metadata and retained 13 landscape sources without
replacement. Source expansion is closed; next is one larger source-grouped dataset build.

The first grouped joystick pilot failed cross-source generalization: train accuracy1.0, but
internal-dev accuracy0.16 and macro-F10.0974, below even the majority baseline in accuracy.
Six classes have zero recall. The attempt and checkpoint are frozen; the same dataset/model is
not retried. The next useful step is more frozen-extractor evidence from additional existing
train videos, with coverage checked before any new model run.

The grouped weak-label pilot dataset is ready: 73 samples from six train sources and 25 samples
from two internal-dev sources, all nine classes in both, source overlap0, joystick masked and all
98 clips unique. The internal dev comes from the train cohort; original video-dev/test stay closed.
The next model must fresh-init and use class-balanced training. Dev macro-F1 and every recall are
required because STOP/SW/W each have only one dev example.

The single real-RGB joystick `overfit32` diagnostic passed at accuracy1.0/loss0.005445 with all
nine recalls1.0 on RTX4090. It uses the existing 686,281-parameter GroupNorm+GRU and proves only
that the 32 masked causal weak-label clips can be memorized. Its checkpoint is diagnostic-only.
Next is a fixed six-train/two-internal-dev grouped dataset; no video-dev/test or checkpoint reuse.

The first real-RGB joystick-supervised diagnostic dataset is now mechanically valid: 32 causal
samples, eight source groups, STOP8 plus three per direction, with the joystick region zeroed.
`joystick-materialize32` creates it and `--verify-only` checks hashes, shape, class balance,
timestamps and masking. This allows one fresh-init overfit diagnostic only; it is not formal
training or evidence of policy generalization.

The final bounded train expansion now supplies automatic weak-label support across eight sources:
all directions appear in at least three sources and eight explicit release STOP events exist.
Status is `JOYSTICK_8_TRAIN_SOURCE_WEAK_LABEL_SUPPORT_PASSED`. This unlocks only a small causal
dataset build; policy training and semantic-accuracy claims remain closed. Actor windows must end
before the joystick target and exclude the lower-left joystick pixels.

Frozen-extractor transfer across four additional train recordings is partial: all four contain
candidates (mean34.17%, minimum14.17%), but E/SW/SE have stable support in only one source and
only two explicit release STOP events exist. `joystick-transfer` writes QA/hashes/candidates only;
no policy samples. More existing train data can improve support counts but cannot by itself prove
semantic accuracy.

`joystick-eligibility` now separates stable direction runs from explicit direction-to-center
release events. All eight directions have a two-frame run, but only four release STOP events exist
and all data comes from one recording. Future Actor input must end one sampled frame before the
joystick target (96–114ms in current evidence); the following frame may confirm a label but cannot
enter Actor input. No policy samples or model are produced.

The fixed-extractor coverage check now observes all nine candidate classes: 177/480 frames
from twelve new train time windows. This is single-recording UI-state coverage, not training
readiness. Of 62 STOP candidates, 40 occur in dimmed late windows with unresolved scene context.
The next work is scene eligibility and frame/action timing, while keeping the extractor fixed.

`movement-mvp --mode joystick-coverage` checks twelve additional four-second windows from the
same train recording. Supply `--source-root`, `--cohort-dir`, `--pre-ingest`,
`--source-run <joystick-extraction-v3-geometry>` and a new `--output-dir`. It binds the frozen
v3 templates and extractor fingerprint; it cannot recalibrate. Reports distinguish class frames,
class runs and supporting windows. Only candidate records and QA are persisted; no new native
frame cache, model, dev/test decoding or policy-training dataset is produced.

Latest: `JOYSTICK_GEOMETRY_CANDIDATES_PARTIAL`. `--geometric-base` on joystick extraction uses
three-of-four directional markers and shared scale. Candidate coverage is train69/120, dev13/120;
dev has five directions and no STOP, so policy training is still unready. The next step is bounded
coverage checking on more existing train time windows with this extractor fixed.

The joystick extractor now supports `--normalize-scale`. Nine synthetic train-transform checks
pass; dev knob matching improves, but only 3/120 frames produce full direction candidates because
base matching remains weak. Current state: `JOYSTICK_SCALE_REGRESSION_PASSED_BASE_CUE_LIMITED`.
Candidate labels are not ready for policy training. Existing dev is used only as a regression set.

The first joystick extractor is implemented as `movement-mvp --mode joystick-extraction
--source-run <visibility-cache> --output-dir <new-directory>`. Train-only templates are frozen
before dev extraction. It yields 56/120 train candidates but 0/120 dev candidates, so it is not
transferable or training-ready. The next concrete issue is control-scale normalization and base
background sensitivity. Existing dev results can guide disclosed regression work, not serve as
an untouched benchmark again. Low-confidence detections remain unknown, never automatic STOP.

The current joystick visibility pilot found readable translucent knob displacement in both sampled
train/dev recordings. Six four-second native-crop windows are available for a bounded automatic
extractor; this is visual feasibility only, with no action labels or policy training yet.
`movement-mvp --mode joystick-visibility` takes the existing `--source-root`, `--cohort-dir`,
`--pre-ingest`, and a fresh external `--output-dir`. It opens only the two frozen train/dev sources.
Shop occlusion is unknown; future Actor inputs must exclude the joystick used to generate targets.

Previous continuation: `PLAYER_FLOW_GAP_INSUFFICIENT`. One offline optical-flow pass recovered
15 frames across 9 short gaps in session002, but continuous frames increased only 9.6%;
003/005 remain unsupported. No training was started. See the current progress ledger for evidence.
The optional `flow-audit` dependency group records the installed OpenCV version. Reproduction:

```bash
python -m hok_agent movement-mvp --mode real-player-flow-audit \
  --config configs/movement_real_player_localization_audit_v2.json \
  --prior-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-localization-v2/report.json" \
  --session-root "$HOK_LARGE_ROOT/datasets/operation-movement-teacher-v1" \
  --output-dir "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/player-flow-gap-v1"
```

Existing outputs are not overwritten. This is retrospective trajectory evidence, not real-time
localization; actions only enter evaluation. The completed batch has no parameter-search follow-up.

The active roadmap is [Engineering convergence plan](docs/ENGINEERING_CONVERGENCE_PLAN.md).
The next cycle targets a reproducible **offline** MVP: rule Macro/Combat, one goal-conditioned
Movement learner, and the existing observation/execution/transition interfaces. Budget ceilings
are 80 engineering hours, 24 GPU-hours and 50 GiB of new artifacts. Navigation is evaluated through
action-driven episodes, not static direction F1. Rules-only delivery is explicitly not a learned
policy success. Stage A now passes: a fixed blue-side Houyi bottom-lane rule trajectory moves through
RichPixelArena, writes every causal transition, reaches its target, commits STOP, and exits with zero
reward and zero device input. Stage B then passed the simulator overfit32 gate with a 686,281-parameter
task-specific GroupNorm+GRU, while both old-P0 branch variants failed. Full training and real-video
validity remain unproven. The task-specific model is now the default and all four diagnostic
attempts are closed. Run `make movement-mvp-stage-c-smoke` for the current regression surface.

The first Stage C candidate is now frozen failed. Its 64/24 trajectory source passed and training
loss converged, but epoch 20 completed only 15/24 dev episodes with 27.1% collision steps. Random
completed 18/24, exposing an infeasible +8 comparison on a 24-episode set. No model is promoted and
holdout remains unopened. The v2 contract now isolates navigation, requires three consecutive STOPs
and uses failure-inclusive efficiency. Its 64/24 recovery data passed, but both the uniform-sampling
and class-balanced candidates reached 0/24. This round's corrections are exhausted. Use
`--config configs/movement_mvp_stage_c_v2.json` for this route and `--reference-only` when
reevaluating an older checkpoint; reference results cannot promote a model.

The rule data path runs independently of learning. With the project Python environment active and
`HOK_LARGE_ROOT` set, use a new local output directory:

```bash
MOVEMENT_RUN_DIR="${HOK_LARGE_ROOT:?set HOK_LARGE_ROOT}/runs/hierarchical-movement-mvp/rule-batch-demo"
python -m hok_agent movement-mvp --mode rule-batch --episodes 10 --step-budget 4 --output-dir "$MOVEMENT_RUN_DIR"
python -m hok_agent movement-mvp --mode rule-batch --episodes 10 --resume --output-dir "$MOVEMENT_RUN_DIR"
```

This repeats the fixed Stage A scene using a structured simulator rule. It writes one SQLite Store,
an immutable `run-contract.json`, derived frame bundles and `batch-summary.json`; every episode ends
after three STOPs. Resume deterministically replays committed transitions and continues at the next
step. Missing/corrupt committed evidence or changed bindings stop recovery. This is PixelArena
mid-episode recovery, not learned navigation or phone-state recovery, and it loads no model or phone
interface. Actual results and remaining work are recorded only in
[DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md).

Create and independently verify the final local R0 package with the same top-level command:

```bash
MOVEMENT_EVIDENCE_ROOT="${HOK_LARGE_ROOT:?set HOK_LARGE_ROOT}/runs/hierarchical-movement-mvp"
python -m hok_agent movement-mvp --mode package \
  --source-run "$MOVEMENT_EVIDENCE_ROOT/stage-d-rule-batch-v2-recovery" \
  --control-run "$MOVEMENT_EVIDENCE_ROOT/stage-d-rule-batch-v2-continuous" \
  --output-dir "$MOVEMENT_EVIDENCE_ROOT/r0-delivery-v1"
python -m hok_agent movement-mvp --mode package --verify-only \
  --output-dir "$MOVEMENT_EVIDENCE_ROOT/r0-delivery-v1"
```

The package is an immutable local evidence directory with a resolved config, final summary, compact
SQLite backup, 100 derived frame bundles and a final manifest. It contains no trained checkpoint.
Its delivery grade is rule-only `R0_RULE_OFFLINE`; it does not establish learned navigation or
real-video/mobile performance.

The closed cycle is also available as one immutable offline engineering package. It nests the
verified R0 package, the zero-reward death/respawn Event-to-Store replay, and four machine-readable
failure boundaries. It contains no checkpoint and grants no new runtime capability.

```bash
python -m hok_agent movement-mvp --mode package-cycle \
  --source-run "$MOVEMENT_EVIDENCE_ROOT/r0-delivery-v1" \
  --event-run "$HOK_LARGE_ROOT/runs/hierarchical-event-e1/death-respawn-transition-replay-v1" \
  --failure-report "$MOVEMENT_EVIDENCE_ROOT/native-anchor-relation-v2-render-repair/report.json" \
  --failure-report "$HOK_LARGE_ROOT/audit/hierarchical-event-e1/death-banner-consensus-v1/report.json" \
  --failure-report "$HOK_LARGE_ROOT/audit/hierarchical-event-e1/native-death-cue-preflight-v1/report.json" \
  --failure-report "$HOK_LARGE_ROOT/audit/hierarchical-event-e1/native-death-cue-preflight-v1/qa-conclusion.json" \
  --output-dir "$MOVEMENT_EVIDENCE_ROOT/r1-offline-engineering-v1"
python -m hok_agent movement-mvp --mode package-cycle --verify-only \
  --output-dir "$MOVEMENT_EVIDENCE_ROOT/r1-offline-engineering-v1"
```

Its delivery grade is `R1_ENGINEERING_OFFLINE_ZERO_REWARD`: deterministic Movement plus verified
Event-to-Store plumbing, with learned Movement, semantic Reward, real-video policy, mobile control
and RL all explicitly false.

The final simulator hierarchy is packaged as a small evidence extension around that R1 package:

```bash
python -m hok_agent movement-mvp --mode package-hierarchical-rule \
  --source-run "$MOVEMENT_EVIDENCE_ROOT/r1-offline-engineering-v1" \
  --evidence-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/joystick-persistence-baseline-v1/report.json" \
  --evidence-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/movement-change-only-contract-v1/report.json" \
  --evidence-report "$MOVEMENT_EVIDENCE_ROOT/movement-change-pilot-v1/report.json" \
  --evidence-report "$MOVEMENT_EVIDENCE_ROOT/movement-change-replay-v1/report.json" \
  --evidence-report "$MOVEMENT_EVIDENCE_ROOT/movement-change-position-v2-pilot/report.json" \
  --evidence-report "$MOVEMENT_EVIDENCE_ROOT/movement-change-geometry-replay-v1/report.json" \
  --output-dir "$MOVEMENT_EVIDENCE_ROOT/r1-hierarchical-rule-v1"
python -m hok_agent movement-mvp --mode package-hierarchical-rule --verify-only \
  --output-dir "$MOVEMENT_EVIDENCE_ROOT/r1-hierarchical-rule-v1"
```

Its grade is `R1_HIERARCHICAL_RULE_OFFLINE`. It binds Macro goal geometry change, deterministic
direction persistence and Router STOP, but contains no checkpoint and grants no real-RGB, Reward,
mobile-control or RL capability.

`make check` now always binds this checkout's absolute `src` directory through the shared
`RUN_PYTHON` wrapper, preventing a neighbouring editable `hok_agent` installation from being
collected accidentally.

The next-cycle Houyi data audit is available through `movement-mvp --mode houyi-data-audit`.
It opens container metadata only after filtering to train/dev, decodes no frames, and checks the
cohort, hero profile and existing summaries for a complete real-session identity binding. The
current result is `HOUYI_BOUND_REAL_DATA_NOT_AVAILABLE`: 103 train and 23 dev metadata records have
zero hero keywords, the profile is an unconfigured template, and no real session binds Houyi with
immutable identity evidence. The one `hero=houyi` summary belongs to PixelArena Stage A.

An earlier exploratory metadata pass opened all 149 container headers, including 23 test headers,
before split filtering. It decoded zero test frames, found zero hero hits and was not used for
selection or tuning, but is recorded as a boundary deviation. The formal audit opens zero test
containers. Existing video must not be relabeled as Houyi from visual resemblance alone.

The next-cycle real-RGB observability preflight is also available through `movement-mvp`:

```bash
python -m hok_agent movement-mvp --mode real-rgb-preflight \
  --config configs/movement_real_rgb_preflight_v1.json \
  --target-root "$HOK_LARGE_ROOT/datasets/v5-target-file-atomic-v2" \
  --output-dir "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-observability-v1"
```

The frozen v1 run failed: overall self/target pair coverage was 0.4792, one train session reached
only 0.0938, and marker jumps reached 0.4128. It read no test frames, stored no RGB, trained no model
and sent no input. A new minimap detector/tracker contract is required before R2 or Movement
training; rerunning v1 with lower gates is not an accepted repair.

V2 removes unstable red-target selection from the canvas and accepts a semantic goal from the
deterministic Macro context:

```bash
python -m hok_agent movement-mvp --mode real-rgb-goal-canvas \
  --config configs/movement_real_rgb_goal_canvas_v2.json \
  --prior-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-observability-v1/report.json" \
  --target-root "$HOK_LARGE_ROOT/datasets/v5-target-file-atomic-v2" \
  --output-dir "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-goal-canvas-v2"
```

The frozen v2 run passed crop, repeatability and two-goal counterfactual checks on all 288 frames.
It does not verify player localization, the gameplay meaning of the fixed lane coordinate, or policy
performance, so training, promotion and R2 remain closed.

The next simulator-only learnability gate uses the same minimap-plus-goal-ring task:

```bash
python -m hok_agent movement-mvp --mode goal-canvas-overfit32-materialize \
  --config configs/movement_goal_canvas_overfit32_v1.json \
  --goal-canvas-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-rgb-goal-canvas-v2/report.json" \
  --output-dir "$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/goal-canvas-overfit32-v1"
python -m hok_agent movement-mvp --mode overfit32 \
  --config configs/movement_goal_canvas_overfit32_v1.json \
  --dataset "$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/goal-canvas-overfit32-v1/overfit32.npz" \
  --output-dir "$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/goal-canvas-overfit32-seed0-v1" \
  --device cuda --architecture task-specific
```

The frozen run passed accuracy 1.0 and loss 0.00614 with all nine recalls at 1.0. Its checkpoint is
diagnostic-only; formal trajectory training must initialize a fresh model.

The fresh 64/24 goal-canvas candidate has also completed. Teacher and exact-RGB geometry reached
24/24, but the learned epoch-20 checkpoint reached only 9/24 despite training loss 0.01414. It is
frozen failed; no holdout or R2 stage opened. The next learned correction must change only the
model's spatial-relation bias under a new contract, rather than add more epochs or data to this run.

The first two-slot relational correction then failed the earlier overfit32 gate at accuracy 0.625
and loss 1.258; its formal training was not started. Pure action supervision did not reliably assign
player/goal identity to the slots. Any next correction must use automatic simulator localization
targets under a new contract and must not reuse the diagnostic checkpoint.

The first joint automatic-localization diagnostic then reached 3.94-pixel mean slot error but failed
slot-cell accuracy (0.6875) and action accuracy/loss (0.4375/1.055). Formal training did not start.
The next bounded hypothesis is two-stage training: localize slots first, freeze them, then fit the
action temporal/head layers under a new contract.

Two-stage v2 passed localization at 1.0 within-one-cell accuracy and 3.97-pixel mean error, but the
frozen-localizer action stage reached only 0.8125 accuracy and 0.871 loss. Synthetic policy tuning is
now stopped. The next precondition is a high-resolution real-player-cue audit, not another toy-model
head or longer training run.

Player-localization audit v2 invalidates the old near-continuous cue: it was primarily a fixed
top-right UI marker. After excluding that region, only session 002 retains partial action-responsive
coverage (0.1401); sessions 003/005 have no usable track. The old cue and continuity reports remain
historical evidence and cannot authorize training or R2.

A bounded real-counterfactual learnability check now reuses five non-overlapping source windows
from session 002, changes only the hollow Macro goal ring, and forms the balanced 32-sample
nine-action diagnostic. The task-specific GroupNorm+GRU fits it at accuracy 1.0 and loss 0.00732.
This is a memorization result, not held-out-window or executed-action evidence. The subsequent
five-group evaluation confirms the limitation: full accuracy is 0.7778, versus 0.6889 with the
player masked and 0.6667 from the target ring alone; its control gains miss the frozen 0.15 gate.
The route is frozen as shortcut/no-generalization evidence, with no promoted checkpoint.

The bounded appearance tracker also failed to expand coverage: it tracks 155/1485 frames in 002
and none in 003/005. It is not a replacement for v2 and is not connected to navigation. Paired
main/minimap QA shows a fountain/edge observability problem in the sampled scenes; a moving green
marker elsewhere is not sufficient player identity. Next inspect whether existing train/dev source
views retain the complete minimap, especially its spawn corner, before changing detection again.
That inspection is now complete: the three teacher sessions cannot restore missing pixels from
their four saved crops. Two existing, identity-matched landscape train/dev source videos retain
the full minimap and lower-edge margin at native resolution. Use those as alternative data, not as
reconstructions of the teacher sessions. One shop-obscured sample is unusable; the portrait source
is deferred because the diagnostic rotation preview was upside down. No detector or model is
promoted. Next run only a bounded native-resolution localization pilot on the two landscape sources;
do not require new recording, reopen test, or re-tune the failed appearance tracker.

The native-resolution pilot is now implemented as `movement-mvp --mode native-player-pilot`, with
`--source-root`, `--cohort-dir`, `--pre-ingest` and `--output-dir`. It opens only the two selected
cohort-bound landscape train/dev videos, crops the map before resizing to 256x256, and retains
16 frames per source with integer-microsecond sampling. Main-view QA is separate from the map.
The current green-ring cue confirms 9/16 dev frames and 0/16 train frames; overlapping portraits
and background grass remain limitations. This is a partial visual cue, not verified player identity
or an action label. There is no training, device input, or navigation integration.

Cached background comparison is available through the same mode with `--source-run <native-run>`:
it reads no video and does not change RGB. Removing background candidates left confirmed counts
unchanged (train 0, dev 9). The bounded `--train-visibility-scan` option subsequently sampled only
the same train video's 5/10/15-percent windows, with confirmed counts 0/13/15 out of 16. Thus two
clearer train-side clips exist without detector retuning or using dev as training. These are
selected visual-cue examples from one source, not independent episodes or verified action data.
The 15-percent clip is the preferred next perception-QA fixture; no model training is opened.

Identity/coordinate QA on the two clear cached clips is now complete. All 112 supported
translation comparisons have zero equivariance error, with unchanged confirmation/unknown counts;
conversion back to the actual source sampling grid is exact. These are geometric checks, not
localization accuracy. Enlarged portraits are visually consistent across the two clips, but
controlled-player identity and Houyi identity remain unverified. Retain these as non-promoting
perception regression fixtures; do not repeat the same QA or treat their coordinates as action
labels or game-world ground truth.

The first fixed cross-session weak-anchor audit is also complete. It materializes 8 train and
4 dev landscape sessions at 10/30/60 percent, 36 independent session-window groups and 576 frames.
Train support passes at 7/8 sessions and 130 confirmed frames; dev reaches only 2/4 sessions despite
64 confirmed frames, failing the frozen 3-session requirement. The result is
`WEAK_VISUAL_ANCHOR_COHORT_INSUFFICIENT`: no training or checkpoint. QA also shows shop overlays
and non-universal ring presentation. One separately versioned coverage repair may add four new,
anonymous-order dev sessions without changing detection or windows; it cannot rewrite this result.

That one repair is complete. Four new anonymous-order dev sessions raise combined dev support to
6/8 sessions and 171 confirmed frames; train remains 7/8 and 130. All original gates now pass and
the result is `WEAK_VISUAL_ANCHOR_COHORT_SUPPORTED_QA_ONLY`. It still creates no action labels and
does not verify controlled-player or Houyi identity. The next step is a session-isolated data gate
for nine counterfactual target directions using only window-end confirmed anchors; model training
remains closed until that gate passes.

The counterfactual data gate now passes from cached evidence without decoding video: 8 train groups
from 7 sessions yield 72 balanced samples; 9 dev groups from 6 sessions yield 81. Every group has
`STOP + 8 directions`, session splits are disjoint, and current tracker output exactly reproduces
the frozen reports. This opens one small weak-anchor relation diagnostic only. Labels are synthetic
target directions, never human/executed actions; Movement deployment and identity claims remain closed.

Materialization is complete in deduplicated form: 17 source clips plus a 153-row relation index,
rather than nine RGB copies per group. Source RGB is unchanged; the fixed yellow target ring is
rendered deterministically after copying at training time, then nearest-sampled to 128x128.
Train/dev remain 72/81 rows with all nine classes at 17 total each. This dataset permits one
seed-0 relation diagnostic with full, anchor-masked and goal-only inputs; it is not deployable BC.

That diagnostic is now closed failed. The initial render drew radius 7 at 256px before resizing and
failed overfit36 at 0.111 accuracy / 2.203 cross-entropy. One explicit render-order repair resized
first and drew radius 7 at model resolution; overfit36 then passed at 1.0 / 0.00647. Formal
session-isolated dev nevertheless collapsed to one class: full, anchor-masked and goal-only each
reached only 0.111 accuracy and 0.0222 macro-F1. No checkpoint was saved. Do not add updates, models,
samples or threshold changes to this weak-anchor relation route.

The next observable-event cycle now has a complete offline engineering replay for the frozen
E1a death/respawn diagnostic. It replays 285 derived RGB frames into 284 causal, NOOP transitions,
reproduces 1 death, 1 respawn and 17 HP-delta events, and commits a final `VIDEO_EOF` transition.
All rewards are zero and all rows are explicitly non-training because semantic accuracy is still
unverified. SQLite integrity and every frame reference pass; this is Event-to-Store evidence only.

A follow-up inventory over all eight existing non-test operational sessions finds only one paired
death/respawn session and seven negative sessions. The frozen engine emits no death on session 002
despite six legacy hard-stop frames, confirming that hard-stop is not a death label. The required
three positive sessions are unavailable, so death Reward remains closed. Future work must source
additional dynamic candidates from existing train/dev video without treating bar absence as truth.

A two-cue consensus diagnostic is now available with `--banner-consensus`. It confirms death only
when the frozen health state is `DEAD` while the independently derived death-banner hard-stop is
active, then confirms respawn only after the banner clears and health stabilizes to `ALIVE`.
The known positive remains 1/1, while both session-002 hard-stop rising edges are rejected. This
improves candidate specificity but does not solve the one-positive-session data shortage.

The capped 12-session raw-video preflight is also complete and rejected. It decoded 652,190
train/dev frames and sampled 52,868 at 200 ms, but the normalized mobile banner ROI fires on
scoreboards, kill notifications and persistent red UI. Eleven automatic paired sessions and one
unpaired session are therefore not semantic death evidence. Developer QA shows active gameplay in
candidate frames. The result is `NATIVE_DEATH_CUE_PREFLIGHT_DOMAIN_MISMATCH`; no threshold repair,
clip materialization, Reward training or test access follows.

```bash
python -m hok_agent.hierarchical_e1 \
  --contract configs/hierarchical_event_e1_health.json \
  --replay-health-report "$HOK_LARGE_ROOT/audit/hierarchical-event-e1/health-engineering-v2-width-repair/report.json" \
  --replay-session "$HOK_LARGE_ROOT/runs/mobile-operation-base/death-stop-60s-v1" \
  --output-dir "$HOK_LARGE_ROOT/runs/hierarchical-event-e1/death-respawn-transition-replay-v1"
```

The older diagnostic entrypoint uses the existing v2 source bindings (new output directory required):

```bash
python -m hok_agent movement-mvp --mode real-player-tracking-audit \
  --config configs/movement_real_player_localization_audit_v2.json \
  --prior-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-localization-v2/report.json" \
  --session-root "$HOK_LARGE_ROOT/datasets/operation-movement-teacher-v1" \
  --output-dir "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-appearance-tracking-v1"
```

It emits an automatically extracted appearance template, paired QA sheets, and a self-hashed
diagnostic report. Correlation is not confidence calibrated against identity labels; missing
detections emit `unknown`, never extrapolated player positions. False-lock rate and reacquisition
latency remain unverified. The completed local run must not be rerun to tune its threshold.

The earlier frozen entrypoints are:

```bash
python -m hok_agent movement-mvp --mode real-player-localization-audit-v2 \
  --config configs/movement_real_player_localization_audit_v2.json \
  --prior-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-cue-v1/report.json" \
  --session-root "$HOK_LARGE_ROOT/datasets/operation-movement-teacher-v1" \
  --output-dir "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-localization-v2"
python -m hok_agent movement-mvp --mode real-counterfactual-grouped-eval \
  --config configs/movement_real_counterfactual_grouped_eval_v1.json \
  --prior-report "$HOK_LARGE_ROOT/audit/hierarchical-movement-mvp/real-player-localization-v2/report.json" \
  --session-root "$HOK_LARGE_ROOT/datasets/operation-movement-teacher-v1" \
  --dataset "$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/real-counterfactual-overfit32-v1/overfit32.npz" \
  --overfit-report "$HOK_LARGE_ROOT/datasets/hierarchical-movement-mvp/real-counterfactual-overfit32-v1/report.json" \
  --output-dir "$HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/real-counterfactual-grouped-v1" \
  --device cuda
```

The plan supersedes the future schedule and growth proposals in the historical sections below.
It does not change frozen results, reopen video-test, authorize phone control or start RL.
See [DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md) for executed state.

## Historical architecture and reusable components

```text
V1–V3  frozen deterministic and behavior-cloning regressions
V4     local video/V4L2 -> read-only RGB hypotheses
V5     PixelArena source teacher -> SimSiam -> pseudo labels -> Mean Teacher
V6     RGB-derived tracking and causal temporal diagnostics
V7     Rich PixelArena -> factorized ResNet-18 policy -> PixelArena-only loop
T8     mobile/video demonstrations -> causal policy diagnostics -> Shadow -> bounded gates
HP-v0  FrameBus -> RGB events + shared temporal policy -> deterministic Router -> unified Replay
```

V5/V6 base training uses no human action, frame, HUD, tracking, or temporal labels. T8 is a
separate lineage and may use only its own standardized observed or actually dispatched testbed
events. Legal actions, cooldown state, and structured truth never enter an RGB Actor.

The visual-policy organization was informed by
[ResnetGPT](https://github.com/FengQuanLi/ResnetGPT),
[WZCQ](https://github.com/FengQuanLi/WZCQ), and
[wzry_ai](https://github.com/myBoris/wzry_ai). This repository does not copy their device-control
code, data, weights, coordinates, assets, or recordings.

## Hierarchical Policy v0 historical development route

The next development route keeps one RGB PolicyBundle with a shared temporal representation and
three logical heads: Macro, Movement, and Combat. An independent, versioned VisualEventEngine
derives terminal, death/respawn, and self-health events from RGB for RewardHub. A deterministic
Router owns freshness, masks, pointer conflicts, and persistent-joystick execution. Demo, simulator,
controller, online, and failure rows share one episode-major TransitionStore with different source
tags and samplers.

The deliberately small E0 implementation is complete: an immutable latest-frame FrameBus,
VisualState/Event plus exact-once fusion, and a transactional SQLite UnifiedTransitionStore now
validate proposal freshness, action/capture ordering, terminal retention, and episode continuity.
E0 itself is offline-only and has no detector, trained Bundle, online RL, MoE, continuous joystick output,
PPO, or model-driven mobile input. Run its focused regression with `make hierarchical-e0-smoke`.
The full contracts, data roles, parameter tiers, training order, and 1/3/10-episode gates are in
[docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md](docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md).

E1a now adds an offline centered-health-bar diagnostic for HP-change candidates and temporal
death/respawn events. Its first cross-session width contract failed; the single preserved repair
passed the engineering gate with zero train/challenge false deaths and one dev death/respawn pair.
It remains explicitly non-promoting: numeric HP accuracy and independent semantic accuracy are not
verified, so Reward stays disabled. Run its synthetic regression with
`make hierarchical-e1-health-smoke`.

The frozen E1b OCR coverage audit failed: only 1/8 train and 2/4 dev sessions exposed a result
screen in the sampled tail, and neither split produced a reliable `WIN` or `LOSS` token. The report
contains only allowlisted token counts and anonymous IDs. This lineage is not repaired by changing
OCR confidence, tail duration, or sampling rate; the next terminal route must use short-video
crystal-destruction transitions. Run its local regression with
`make hierarchical-e1-terminal-smoke`.

The separate E1c anchor preflight then scanned the final decoded frame of every train/dev session
and found 34 train plus 7 dev result-page anchors with zero outcome conflicts. Video position was
not a label, anchor frames are forbidden from future model input, and video-test was not decoded.
This passes only the data-support gate for constructing preceding short clips; dynamic terminal
accuracy, WIN/LOSS, Reward, and promotion remain unverified. Run its regression with
`make hierarchical-e1c-anchor-smoke`.

E1c-clip materialization produced 34 train and 7 dev same-session triplets: one terminal-transition
candidate, one nearby late-game negative, and one earlier negative per anchored session. The model
shards contain only RGB sequences, labels, and split-only anonymous IDs; anchor frames, timestamps,
relative offsets, OCR, and paths are absent. This passes materialization only, not semantic
terminal detection. Run `make hierarchical-e1c-clip-smoke`.

The mandatory E1c-probe stopped before visual training: within-session materialization ordinal
predicts all three labels with train/dev macro-F1 `1.0/1.0`. Visual accuracy on this dataset could
not distinguish terminal semantics from match-time progression. The fixed-offset lineage is frozen;
no GPU model was trained. Run `make hierarchical-e1c-probe-smoke`.

E1d replaces fixed offsets with visual consensus: temporal change, white explosion, and persistent
structure change. One preserved repair produced 33 train and 7 dev weak candidates. WIN/LOSS and
Reward remain closed. Run `make hierarchical-e1d-crystal-smoke`.

E1d-clip built 33 train and 7 dev event-centered same-scene pairs. Hash-based pair reversal reduced
ordinal-only accuracy to 0.424/0.429. Run `make hierarchical-e1d-clip-smoke`; Reward remains closed.

The CPU seed-0 probe passed overfit32 and reached dev temporal macro-F1 1.0 versus 0.8571 for
last-frame and shuffled controls. This is learnability evidence only. Run
`make hierarchical-e1d-probe-smoke`.

The three pre-test checkpoints are frozen in safetensors with bundle hash
`55a679883119cdbf1a6a7703f945d61ce33408bad84013362e66355e83345c79`. Their dev metrics exactly
reproduce the probe. A one-shot 23-session test contract is frozen before test decoding; it permits
no retraining or threshold adjustment.

That one-shot test was consumed and failed at runtime on a test session without a usable
pre-result sequence. No metric report was produced, and the frozen failure explicitly sets
`rerun_allowed=false` and `integration_allowed=false`. Offline EventEngine integration was not
started.

The follow-up E1e diagnostic did not reopen test. It continued across all 85 unused unanchored
train/dev sessions, but only 6 train and 1 dev pairs were eligible versus 10/3 required; 63 sessions
lacked a usable pre-result sequence. E1e is frozen insufficient and cannot replace the formal test.

P0 then compared the existing epoch-3 SimSiam adapter with its exact source encoder and a random
ResNet-18 on frozen E1d features. All three reached dev macro-F1 1.0, so the adapter showed no
incremental value and is rejected as the new PolicyBundle initializer. Run
`make hierarchical-p0-adapter-smoke`.

The harder P0 temporal-order dataset contains 128 train and 32 dev pairs with identical frame sets
and endpoints. Adapter temporal macro-F1 was 0.3333 versus source 0.4687 and random 0.5142, so the
old adapter is rejected again and cannot initialize the new PolicyBundle.

The separate seed-0 temporal SSL pilot passed its 32-sample overfit and non-collapse checks, and
improved dev macro-F1 to 0.7031, 0.1889 above the best frozen baseline. It nevertheless missed the
pre-frozen 0.75 dev gate. The run is frozen failed without an encoder checkpoint; it cannot
initialize PolicyBundle or open policy training, Reward, test, capture, or input. Run the contract
regression with `make hierarchical-p0-ssl-smoke`.

The separately versioned P0 temporal SSL v2 then indexed 1,648 train-only pairs across all 103
video-train sessions without copying RGB or using video-dev for checkpoint selection. Its fixed
last epoch reached frozen dev macro-F1 0.7907 and passed every predeclared gate. The resulting
ResNet-18 plus GRU representation is allowed only as the P0 initializer; no policy head, Reward,
test, capture, or input is opened. Run `make hierarchical-p0-ssl-v2-data-smoke` and
`make hierarchical-p0-ssl-v2-smoke`.

P1 began with a raw-resolution Movement teacher audit over 103 train and 23 dev videos. A preserved
display-matrix repair recovered every rotated video, but the frozen teacher still covered only
18.44%/18.98% of sampled frames and dev produced only four stable west labels versus sixteen
required. The audit is frozen failed; no Movement Head was trained and its labels are explicitly
teacher recommendations, not observed human actions. Run `make hierarchical-p1-movement-audit-smoke`.

The separate P1 Macro data audit passed on project-owned PixelArena: 1,620 train and 412 dev causal
windows cover FARM_LANE, PUSH_STRUCTURE, and ENGAGE in every 40/10 episode. Class-prior and
time-only controls reached only 0.2019 and 0.3893 macro-F1. This permits one simulator-only Macro
Head learnability run from the frozen P0 representation; it is not real-video semantic evidence.
Run `make hierarchical-p1-macro-data-smoke`.

The frozen-P0 Macro Head run then failed: dev macro-F1 was 0.3868 versus 0.3893 for the time-only
control, and its gain over label shuffle was only 0.0845. Overfit32 also missed its loss gate.
No Macro Head checkpoint was saved. This does not invalidate P0 temporal-order learning, but it
does show that the frozen representation cannot directly supply the required simulator Macro
semantics through this fixed canvas/head. Run `make hierarchical-p1-macro-head-smoke`.

The P1 Combat data audit also stopped before training. Eight five-minute sessions contain adequate
button counts, but a 200 ms clock-only lookup reaches macro-F1 0.9331 and the sessions expose only
two positive action sequences. The artifacts also do not bind a verified Houyi identity. These
rows validate deterministic cooldown execution, not tactical action selection. Run
`make hierarchical-p1-combat-data-smoke`.

P1v2 therefore replaces the failed fully frozen transfer assumption with a bounded task-specific
adapter design. P0 through layer2 remains shared and frozen; Macro and Movement each own an
independently trained 8–14M layer3/layer4 plus GRU branch at 2 Hz and 10 Hz. Combat v0 remains the
10 Hz deterministic visual cooldown arbiter until non-clock, hero-bound data exists. Models emit
proposals only; the deterministic Router owns freshness, concurrent pointers, death, and hard stop.
Run `make hierarchical-p1v2-architecture-smoke`.

The first P1v2 Movement audit found only east/west moves in the Global Agent dataset: 376/226 train
and 115/45 dev, with all six other directions absent. Skill aim and wait were excluded rather than
misused as movement. A read-only check also found that historical V7 fit/acquisition move rows are
all ego-view east. No Movement branch was trained; a new balanced, visually conditioned 2D
PixelArena navigation source is required. Run `make hierarchical-p1v2-movement-data-smoke`.

The new source materializes 512 train and 128 dev RGB sequences with exact 64/16 support for each
direction, alternating blue/red ego views and zero group overlap. It permits one task-specific
Movement branch learnability run, but its claim is limited to approaching a locally visible target;
lane strategy and real-video transfer remain unverified. Run
`make hierarchical-p1v2-movement-2d-smoke`.

The first task-specific branch reached perfect eight-direction dev macro-F1 and recall, far above
the 0.1748 label-shuffle control, but failed its mandatory overfit32 gate after the sole batching
repair: accuracy 0.9375 and loss 0.1704 missed 0.95/0.05. No checkpoint was retained. The result is
learnability evidence only; a future normalization-stable branch must use a new versioned contract.
Run `make hierarchical-p1v2-movement-branch-smoke`.

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,events,events-ocr,bc,vision,shadow,preingest]'

make check
make accept
make accept-v2
make pixel-smoke
make shadow-live-smoke
make alignment-smoke
make temporal-smoke
make rich-smoke
```

The CI workflow uses CPU PyTorch. CUDA is required only for formal GPU acceptance and larger local
training runs; a sandbox that cannot see the GPU is not evidence that the host GPU is unavailable.

## External data storage

Large data is never committed. The public default is `.local-data/hok-agent-v5`, which is ignored
by Git:

```bash
make storage-init
make storage-show
```

Override it for another mounted data volume without changing code:

```bash
HOK_LARGE_ROOT=/absolute/private/path/hok-agent-v5 make storage-show
```

The external tree contains `datasets/`, `checkpoints/`, `runs/`, `cache/`, `audit/`, and
`staging/`. Manifests may store artifact basenames and anonymous hashes, never source-video
locators, account identifiers, raw touch dumps, or credentials.

## Read-only video

Offline input must be one non-symlink regular local recording:

```bash
python -m hok_agent shadow-video \
  --input /absolute/path/to/privacy-reviewed-recording.mp4 \
  --model /absolute/private/path/model.safetensors \
  --output-dir .local-data/hok-agent-v5/runs/shadow-offline-001
```

Live capture accepts only one explicitly selected non-symlink `/dev/videoN` V4L2 character
device. Numeric indexes, URLs, network streams, and automatic source selection are rejected.

## Mobile testbed: locked by default

The checked-in examples are intentionally invalid:

- `configs/mobile_testbed_identity.example.json`
- `configs/mobile_testbed_layout.example.json`

Create private local files only for a project-owned self-built app:

```bash
cp configs/mobile_testbed_identity.example.json configs/mobile_testbed_identity.local.json
cp configs/mobile_testbed_layout.example.json configs/mobile_testbed_layout.local.json
```

Fill the package, version, signing identity, APK SHA-256, owner attestation, date, display geometry,
and normalized control positions from your own build. Then calculate the canonical identity hash:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

path = Path("configs/mobile_testbed_identity.local.json")
payload = json.loads(path.read_text(encoding="utf-8"))
payload.pop("identity_sha256", None)
payload["identity_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
```

The default identity path may be overridden with `HOK_MOBILE_IDENTITY_PATH`. Identity and layout
files matching `*.local.json` are ignored by Git.

A read-only dry run still requires an explicit serial and capture source:

```bash
HOK_MOBILE_IDENTITY_PATH=configs/mobile_testbed_identity.local.json \
python -m hok_agent mobile-testbed \
  --serial YOUR_USB_SERIAL \
  --model /absolute/private/path/model.safetensors \
  --layout configs/mobile_testbed_layout.local.json \
  --output-dir .local-data/hok-agent-v5/runs/mobile-dry-run \
  --device cpu --run-seconds 60 --infer-hz 1
```

Device input additionally requires the explicit input flag and a finite action cap. Immediately
before every dispatched event, the runtime rechecks the serial, locally attested package,
foreground window, display identity, and guard watchdog. Missing identity, invalid identity hash,
package/version/signature drift, layout mismatch, disconnection, or backgrounding stops the run.
Accessibility, root, global hooks, HID/UHID, uinput, minitouch, generic shell execution, process or
memory inspection, protocol interception, evasion, and online learning are outside the supported
surface.

See [BOUNDARIES.md](BOUNDARIES.md) before enabling any mobile input.

## T8 status

T8-v2.7 is permanently frozen failed: its reports may be verified but not used to justify more
collection, threshold changes, or another four-class fit.

T8-v3 predicts five RGB-observable states from a V5-initialized ResNet-18 and 16-frame causal TCN:

- `enemy_visible`
- `attack_opportunity`
- `basic_ready`
- `skill1_ready`
- `skill2_ready`
- derived `confidence` and `abstain`

The single seed-0 run did not pass its frozen dev gates. Mean head macro-F1 was `0.450962`, minimum
positive recall was `0.314075`, normal-minus-shuffled margin was `0.023275`, and confidence
coverage was `0.039481`. Offline replay rejected the model; video-test, live Shadow, and all T8-v3
device stages remained unopened. These results are failure evidence, not a released policy.

T8-v4 narrowed the task to four local visual cues and used conservative dual-teacher weak targets.
Its RGB and temporal controls passed, but the selected model failed the frozen spatial-selectivity
gate. T8-v4 is therefore frozen failed with no replay, Shadow, or input permission. See
[docs/T8_V4_PROTOCOL.md](docs/T8_V4_PROTOCOL.md).

T8-v5 is a smaller offline successor that asks only whether each fixed correct ROI predicts the
frozen weak target better than time, a wrong ROI, and shuffled labels. Enemy, basic attack, and
skill1 are formal heads; skill2 is diagnostic-only because its frozen dev negative support is too
small. Basic attack passed, but enemy and skill1 failed the frozen wrong-ROI margin, so T8-v5 is
frozen without a TCN stage and remains non-promoting. See
[docs/T8_V5_ROI_PROTOCOL.md](docs/T8_V5_ROI_PROTOCOL.md).

The independent Basic-only MVP extracts only the passed basic ROI component and combines it with
the frozen enemy visual rule. Its complete video-dev replay produced six conservative candidates
with no invalid-screen or rate-limit violations. The admitted five-minute zero-control Shadow
completed all 1,500 cycles with low latency but produced zero candidates: the live basic score
never reached the frozen 0.80 threshold. The route is frozen without a probe or device input. See
[docs/T8_BASIC_MVP_PROTOCOL.md](docs/T8_BASIC_MVP_PROTOCOL.md).

A final deterministic rule fallback verified that the calibrated basic ROI appeared visually
ready. The first 0.80 smoke failed on capture variation; one fixed 0.75 engineering calibration
passed 100/100 cycles. Owner observation established that basic attack has no cooldown dimming, so
visual and execution coordinates were separated instead of requiring a false release event. The
corrected private touch point passed 20 actions, one minute, and five minutes without unexpected
input.

All four combat buttons were then moved to an acknowledged synchronous ADB tap sender. Two
independent 60-second mixed probes each executed basic attack, skill1, skill2, and skill3 exactly
five times, with 20/20 synchronous acknowledgements and zero unexpected actions per run. This is a
deterministic owner-testbed result; it is not model-driven gameplay and does not include movement,
aiming, or target selection.

The subsequent visual combat arbiter replaced the fixed button schedule with cooldown-aware
round-robin selection. Its 60-second gate executed 20 actions and its five-minute gate executed 58;
every command was synchronously acknowledged and owner-observed as normal. Skills must visibly
enter cooldown and recover before rearming. See
[docs/VISUAL_COMBAT_ARBITER_PROTOCOL.md](docs/VISUAL_COMBAT_ARBITER_PROTOCOL.md).

Its first formal event package has two diagnostic sessions and 78 synchronously bound actions.
Training remains blocked: the initial events have no RGB/features and only fixed-rate derived
timestamps. The next dataset gate requires twelve new timestamped feature sessions frozen as
8 train, 2 dev, and 2 test.

Mobile Operation Base v1 closes the first engineering part with one guarded two-pointer scrcpy
session. It continuously moves through eight directions while concurrently executing combat and
the single recommended-equipment purchase, observes the minimap, and stores four derived ROI
views. The five-minute gate and a live death/respawn hard-stop test passed. See
[docs/MOBILE_OPERATION_BASE_PROTOCOL.md](docs/MOBILE_OPERATION_BASE_PROTOCOL.md).

Operation Policy v1 starts the offline second part without reopening the failed T8 lineages. It
uses the frozen operation/combat evidence to train 200/500 ms inverse-dynamics heads, admits only
their high-confidence agreement on video-train/video-dev, and compares simple 16-frame movement
and combat policies. Purchase and hard-stop remain deterministic; the entire first contract is
zero-control and cannot connect to the phone. See
[docs/OPERATION_POLICY_V1_PROTOCOL.md](docs/OPERATION_POLICY_V1_PROTOCOL.md).
Its seed-0 inverse-dynamics gate is now frozen failed: spatial encoder features improved the
negative-control margin but did not recover reliable movement directions or combat classes. The
run stopped before video pseudolabels, policy fitting, Shadow, or phone input.

The separate Operation Direct Policy check used existing executed actions without connecting the
phone. It also failed: direction changes and combat classes were not predictable from automatic
round-robin schedules. Operation Base therefore closes the actuator and data-binding layer, not
tactical policy supervision. See
[docs/OPERATION_DIRECT_POLICY_V1_PROTOCOL.md](docs/OPERATION_DIRECT_POLICY_V1_PROTOCOL.md).

Operation Movement Teacher keeps the persistent joystick but replaces the fixed movement schedule
with the frozen nearest-target minimap rule. Its spatial v1.1 repair audits a pool of automatic
sessions, freezes exactly three train and one dev session only after real eight-direction coverage,
then runs a balanced 32-window overfit before one offline seed-0 pilot. It does not use human
labels, PPO, Shadow, or model-driven phone input. See
[docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md](docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md).
The read-only `mobile-operation-team-side` preflight detects blue versus red from the unique
yellow self-card highlight on the loading panel and fails closed on gameplay or ambiguous frames.
The formal collector then uses that result for a summary-only 20-second marksman opener derived
from the existing blue-bottom human trace, mirrored for red, before any training row is recorded.

The separate `mobile-marksman-lane-controller` command skips movement learning entirely. It uses
the verified side, the same opener, a deterministic advance/hold lane cycle, existing visual
combat and purchase rules, and opener replay after death. See
[docs/MARKSMAN_LANE_CONTROLLER_PROTOCOL.md](docs/MARKSMAN_LANE_CONTROLLER_PROTOCOL.md).

The frozen simulator foundation is Global Agent v1: a structured simulator rule teacher completes full games and
labels macro intent plus semantic target zone; an RGB student learns those high-level decisions,
while the existing deterministic navigation, combat, purchase, layout, hero-profile, and safety
modules execute them. Its five offline stages are complete: the frozen GlobalArena rule regression
is 19/20 (with 20/20 tower progress), seed-0 BC reached 7/10 pure-student simulator terminals, the
only DAgger round reached 9/10, and the fresh 20-seed holdout selected DAgger at 18/20. The adapter
improved video consistency but reduced simulator terminals, so it is not promoted. The selected
student then passed only 2/6 fixed challenge states. Two separately authorized zero-control mobile
diagnostics passed runtime safety but produced constant `DISENGAGE/OWN_BASE`; the 10-minute Shadow
and every input stage remain closed.
See [docs/GLOBAL_AGENT_V1_PROTOCOL.md](docs/GLOBAL_AGENT_V1_PROTOCOL.md).

The completed non-promoted repair is Human IfO Bridge v1. It uses complete human-match videos as observation-only
behavior demonstrations, learns Human/Sim temporal representations, trains inverse macro dynamics
from GlobalArena truth, and then fits Human-BC with frozen-Dagger distillation. Scenario cards are
diagnostic-only; direct pixel-similarity reward and early reinforcement learning are not admitted.
See [docs/HUMAN_IFO_V1_PROTOCOL.md](docs/HUMAN_IFO_V1_PROTOCOL.md).

The reproducible entrypoints are `global-agent-evaluate`, `global-agent-materialize`,
`global-agent-train`, `global-agent-dagger`, `global-agent-domain-adapt`, and
`global-agent-replay`, `global-agent-holdout`, and `global-agent-challenge`. Their Make targets use
`HOK_LARGE_ROOT`; dataset, checkpoint, video, and report artifacts are never committed. The public
summary is [docs/GLOBAL_AGENT_V1_OFFLINE_EVIDENCE.json](docs/GLOBAL_AGENT_V1_OFFLINE_EVIDENCE.json).

Adaptive Layout and Hero Profiles v1 separates device geometry from hero skill behavior. Button
groups are located by structure rather than skill-icon appearance; local hero profiles define how
the three fixed skill slots execute. Unknown heroes remain skill-disabled. See
[docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md](docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md).

## Project documents

- [AGENTS.md](AGENTS.md): implementation authority and module constraints.
- [BOUNDARIES.md](BOUNDARIES.md): permitted and forbidden execution surfaces.
- [DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md): concise current-state ledger.
- [docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md](docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md): active
  development architecture, contracts, training order, and acceptance gates.
- [docs/GLOBAL_AGENT_V1_PROTOCOL.md](docs/GLOBAL_AGENT_V1_PROTOCOL.md): frozen full-episode macro-policy foundation.
- [docs/GLOBAL_AGENT_V1_CONVERGENCE_ROADMAP.md](docs/GLOBAL_AGENT_V1_CONVERGENCE_ROADMAP.md): frozen v1 route, gates, and stop conditions.
- [docs/T8_V4_PROTOCOL.md](docs/T8_V4_PROTOCOL.md): frozen T8-v4 diagnostic and promotion protocol.
- [docs/T8_V5_ROI_PROTOCOL.md](docs/T8_V5_ROI_PROTOCOL.md): T8-v5 isolated-ROI evidence gate.
- [docs/T8_BASIC_MVP_PROTOCOL.md](docs/T8_BASIC_MVP_PROTOCOL.md): deterministic basic-only gates.
- [docs/VISUAL_COMBAT_ARBITER_PROTOCOL.md](docs/VISUAL_COMBAT_ARBITER_PROTOCOL.md): cooldown-aware four-button arbiter.
- [docs/MOBILE_OPERATION_BASE_PROTOCOL.md](docs/MOBILE_OPERATION_BASE_PROTOCOL.md): frozen movement, combat, purchase, minimap, and hard-stop base.
- [docs/OPERATION_POLICY_V1_PROTOCOL.md](docs/OPERATION_POLICY_V1_PROTOCOL.md): offline inverse-dynamics and causal movement/combat route.
- [docs/OPERATION_DIRECT_POLICY_V1_PROTOCOL.md](docs/OPERATION_DIRECT_POLICY_V1_PROTOCOL.md): frozen executed-action learnability check.
- [docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md](docs/OPERATION_MOVEMENT_TEACHER_PROTOCOL.md): active state-conditioned movement route.
- [docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md](docs/ADAPTIVE_LAYOUT_AND_HERO_PROFILES.md): device geometry and skill-behavior contracts.
- [docs/DELIVERY_HISTORY.md](docs/DELIVERY_HISTORY.md): sanitized historical ledger.

## License

Licensed under the [Apache License 2.0](LICENSE).
````

### Snapshot: docs/ENGINEERING_CONVERGENCE_PLAN.md

````text
# 工程收敛推进计划

## 当前接续：从真实摇杆反馈生成监督

零参数几何基线在v2数据达到train256/256、dev96/96、八类recall1.0，并在原多waypoint
路线完成10/10局、40次转向、40 KEEP、40 STOP、120 arena step。模型/GPU/输入均为0。
因此模拟层正式选择Macro目标几何规则+执行器持续方向+Router停止，所有learned Movement
checkpoint保持拒绝。下一步仅打包该离线证据；真实应用仍需新的可靠玩家定位和可观察
Macro目标，未解决前不再训练Movement模型。

位置held-out v2模型仍失败：train accuracy1.0，dev0.2083、macro-F10.1987，N/S/NW/SE
recall为0；epoch5到30准确率不变且loss继续上升。E/W三行覆盖并未让CNN学会相对几何，
仍表现为绝对位置记忆。本轮关闭同数据重训、模型扩展和关系架构调参，也不创建新replay。
下一步仅验证已有零参数goal-canvas几何规则在v2数据及原路线是否精确；若通过，模拟转向
继续使用确定性规则，真实应用仍等待可靠的玩家/目标视觉来源。

位置修正版v2数据已通过：保持256/96与八类平衡不变，E/W在train/dev都覆盖y2/3/4；
train当前位置x=6/7/8，dev x=5/9，位置集合交叉0，352个clip唯一。其他动作几何、目标环、
15旧目标+1新目标和无STOP边界均未改。下一步只fresh-init一次last-frame模型，复用原seed、
AdamW、30epochs和门槛；不重跑GRU，因为失败已定位为空间覆盖，不加载旧checkpoint。

多waypoint回放在正式预检失败20/40，arena step保持0。N/S和四个斜向全部正确，E/W均为
0/10：原数据把E/W与玩家固定在y=3绑定，模拟dev也复用了该模式；新路线在y=2/4时暴露
位置捷径。当前checkpoint与路线重试关闭。下一步只修模拟数据覆盖，让E/W分别出现在
y=2/3/4，并以位置分组构造held-out dev；先物化和验证，不调整模型、阈值或真实接口。

change-policy模拟对照通过：GRU与last-frame均达到train/dev accuracy1.0、macro-F11.0、八类
recall1.0。按预先简化规则选择587080参数last-frame，较GRU少99072参数且无需时序状态；
最佳epoch5。该checkpoint仅允许模拟集成。下一步离线运行多waypoint episode：仅Macro目标
变化时推理，其他step由执行器KEEP，目标到达/未知/终局由Router STOP。不得由此声称
真实RGB、手机或实战性能，也不加载未选GRU权重。

change-event模拟数据已完成：train64局256样本、dev24局96样本，八方向严格平衡、episode
交叉0、352个clip唯一。序列前15帧显示旧Macro目标，最后当前帧显示新目标；标签仅由玩家
标记与新目标环几何关系产生。输入没有previous_action、方向箭头或STOP。下一步固定比较
fresh 8-head GroupNorm+GRU与last-frame control；若效果相同选择更简单者。该实验只验证
simulator目标条件，不产生真实RGB、手机或实战能力声明。

change-only 合同已通过：模型只在Macro目标版本变化、卡住恢复或复活重置时调用，只输出
八方向；previous_action不进入Actor。稳态由执行器KEEP，目标到达/未知/终局由Router STOP。
8个方向变化、8个稳态、8个同方向新目标和3个STOP边界均通过。下一步使用现有目标环
画布生成独立simulator change-event数据集，先验证目标与标签因果关系，不训练模型、
不复用任何失败checkpoint，也不将模拟结果描述成真实游戏转向能力。

零参数 persistence 基线在冻结延续数据上达到 train203/203、dev80/80，八方向 recall1.0；
现有执行器对 previous=current 的八方向全部输出 KEEP，并已有 previous_action 保存/恢复。
这只证明动作保持，无新转向能力。由此关闭 learned continuation：持续移动由执行器状态
负责，STOP由Router负责。后续 Movement 模型只能学习“何时转向以及转到哪里”，且必须
由单独可观察的 Macro 目标提供条件；当前人类视频摇杆标签本身不包含这个目标。

八分类 continuation pilot 仍失败：train accuracy1.0，dev accuracy0.25、macro-F10.1897，
W/E/NW/SE recall为0，准确率略低于多数类0.2625。模型和同数据重训停止。由于延续标签
按定义已有两帧相同执行方向，确定性执行状态直接保持 previous_direction 可能天然满足
该任务。下一步仅从冻结 manifest 审计这一零参数基线；若其精确成立，持续移动交给
确定性控制器，学习模型只应处理带可观察目标的新转向意图，而非重复学习动作保持。

八方向延续数据已物化：train203/16来源、dev80/5来源，来源交叉0，283个clip唯一；输入
提前标签81–100ms并遮掉摇杆。动作顺序固定N/S/W/E/NW/NE/SW/SE，数据中没有STOP，
停止继续归确定性Router。下一步只运行一次fresh-init八分类continuation pilot，复用现有
spatial encoder和GRU结构并新建8输出head、类别平衡采样，不加载任何历史权重。该模型
只负责维持已建立的移动方向，不负责新转向、停止或战术意图。

起点/延续审计已通过新的可观察目标：只有同方向连续段的第3帧及以后才作为标签，使输入
结束前已有两帧同方向操作。固定16/5来源 split 下，八方向延续样本为 train203/dev80，
每类均有支持。STOP 不进入模型；回中画面仍混有死亡、遮挡、禁用和主动停止，继续由
确定性 Router 管理。下一步只物化这283个八方向因果遮罩样本，不训练；其能力边界是
维持既有移动方向，不包括人类转向起点或战术意图。

201/48规模对照仍失败：最佳epoch10的dev accuracy0.1667、macro-F10.1465，四类recall为0，
准确率低于多数类基线0.2708。相比73/25仅将F1从0.0974提升到0.1465，随后继续训练只会
提高train并降低dev。当前数据取每段动作的起点，而输入截止于起点前，可能要求模型预测
画面中尚不可见的人类转向意图；这只是待验证假设。下一步只用冻结JSON区分动作起点和
延续帧，统计其来源/类别支持，不解码RGB、不训练、不调整模型或提取器。

21来源分组数据已完成：16局训练201样本、5局内部开发48样本，两个 split 九类齐全、
来源交叉0、249个 clip 唯一。开发来源保留原2局，再按新增横屏匿名哈希首/中/末加入3局，
未按标签挑选。输入提前标签81–100ms且摇杆区域为零。下一步允许一次 fresh-init scale21
pilot，完全复用上次的模型、类别平衡采样、优化器、30 epochs和门槛；不得加载失败权重，
不得改提取器或 split，也不打开 video-dev/test。

24个新增 train 来源的严格运行发现11个竖屏输入并停止。唯一修复只按容器宽高和旋转
元数据保留13个横屏来源，不补新来源。新增2600帧、653候选、19次释放 STOP；合并原
8局后为21个来源、4160帧、1210候选、27次释放 STOP，八方向分别有8–16局支持。
数据规模已明显增加，来源扫描至此关闭。下一步固定21局来源分组并物化一次较大的
因果数据集；继续沿用冻结提取器，不调整已失败模型，不打开 video-dev/test。

固定24来源扩展首次因11个竖屏输入停止；唯一修复只按容器宽高/rotation排除它们，不补
新来源。13个横屏来源完成2600帧、653候选、19次释放STOP。合并此前8局后，共21来源、
4160帧、1210候选、27次释放；各方向获得8–16个来源支持，稳定段共192个方向样本。
来源扩展至此关闭。下一步固定21来源的train/internal-dev分组，再从每个稳定段起点和
释放STOP物化一次因果遮罩数据；不调提取器、不复训当前失败模型，也不打开video-dev/test。

首个 grouped pilot 已失败并冻结。类别平衡训练使 train accuracy 达1.0，但两局内部 dev
只有 accuracy0.16、macro-F10.0974，六类 recall 为0，两个来源 accuracy 分别0和0.235；
多数类基线 accuracy0.28。该证据说明当前73个训练样本没有跨来源泛化，但不能单独归因
于数据量、弱标签噪声、画面域差异或人类动作随机性。同数据重训和模型调参停止。
下一步只扩大冻结提取器在额外既有 train 视频中的覆盖证据，先检查来源数、方向与释放
STOP 支持，再另立训练合同；不打开 video-dev/test，不复用任何诊断 checkpoint。

固定分组数据已通过：6个来源训练73样本，2个来源内部开发25样本；两个 split 九类齐全、
来源交叉为0、98个 clip 唯一，输入提前标签95–100ms且摇杆区域为零。内部 dev 来自原
train cohort，并非未触碰的 video-dev/test；STOP/SW/W 各仅1例，因此后续必须报告
macro-F1和逐类 recall。下一步只运行一次 fresh-init、类别平衡采样的现有 GroupNorm+GRU
pilot，不加载 overfit32 权重；任何结果均不直接推广 checkpoint 或宣称语义准确率。

唯一一次真实 RGB overfit32 已通过：现有 GroupNorm+GRU、seed0、200 updates 在 RTX4090
达到 accuracy1.0、loss0.005445、九类 recall1.0，首步梯度和参数更新正常。该结果只证明
32个遮罩因果 clip 能被记忆；诊断权重不得用于正式初始化，唯一尝试已消耗。
下一步从已冻结候选中物化一次固定来源分组数据：六个 train 来源、两个内部 dev 来源，
不得打开原 video-dev/test。先做因果、遮罩、类别与来源隔离校验，再决定是否 fresh-init
训练第一个 grouped pilot；不在此阶段宣称真实动作语义或游戏性能。

首个真实 RGB 诊断数据集已物化并机械通过：32 个样本、16×128×128 RGB、STOP8、八方向
各3，覆盖8个来源。方向样本同类来自三个不同来源；所有输入截止于标签前96–99ms，
摇杆区域在缩放前后置零，标签帧和确认帧都不进入 Actor 输入，32个 clip 哈希唯一。
下一步只允许现有 task-specific GroupNorm+GRU 做一次 fresh-init overfit32，验证链路能否
记忆这些自动弱标签。通过也不代表跨局泛化、语义准确率或可部署策略，不启动正式训练。

最后三个 train 来源已按固定五个时段完成，train 扩展正式关闭。合并八局后共有1560帧、
557个候选；八方向稳定段分别获得3–7局支持，明确释放 STOP 恰好8次。该结果允许下一步
制作小型诊断数据集：只取连续两帧方向和释放 STOP，Actor 的16帧输入截止于标签前一帧，
左下角摇杆区域必须遮除，来源分组和哈希必须保留。当前仍不训练、不推广 checkpoint，
也不把自动 UI 候选描述为独立语义真值；完成数据因果/泄漏校验后再决定训练。

冻结提取器已在四个额外 train 来源完成迁移检查，每局固定 20/50/80% 三段。四局均有
候选，平均覆盖34.17%、最低14.17%；稳定方向跨局支持 N3/S2/W2/E1/NW2/NE3/SW1/SE1，
释放 STOP 仅2次。E、SW、SE和 STOP 未满足跨局支持，暂不物化策略数据。
更大 train-only 扫描只能增加自动候选数量，不能取代独立语义准确率；若继续，必须
保持提取器、来源和时间选择预先固定。规则 Movement 继续作为工程可用基线。

因果资格检查已完成。方向必须连续两帧一致，八方向均有支持，但 S/W/SW/SE 等只有
一个稳定段。STOP 仅接受 500ms 内存在明确方向的回中起点，剩 4 次，低于固定门槛8；
变暗画面中的长期回中被排除。未来输入窗口结束于标签前一帧，当前实际间隔 96–114ms；
下一帧只用于离线确认标签稳定，不能进入 Actor 输入。因此暂不物化样本或训练。
下一步用冻结 v3 提取器检查少量额外 train 来源的迁移和独立 session 支持。

固定提取器的十二窗口覆盖检查已完成：480 帧中 177 个候选，九类齐全，STOP62。
其中 STOP40 位于整体变暗的 f50/f70/f80/f90 窗口，尚不能证明对应正常对局中的主动
停止。其余 STOP22 也仍是 UI 状态候选。下一步在已有证据上明确场景适用性和画面—
动作时间对应，再考虑样本物化；不重调模板，不把单录像覆盖当作跨录像训练成功。

覆盖检查采用固定几何提取器，在同一 train 来源按 5/15/20/25/35/40/45/50/65/70/80/90%
时刻各取 4 秒、10 Hz。与已有 10/30/60% 窗口区分保存。报告同时统计帧数、连续动作段
和支持窗口数，避免把一次保持动作当作很多独立例子。STOP 与 unknown 分开统计。
只保存候选、实际 PTS、原图哈希和 QA，不重复存储原分辨率 RGB。方向齐全只是必要
数据条件，不自动解锁训练，也不替代跨 session 评估或独立标签准确率。

底座几何版本已完成：分别匹配四个标记，至少三个在预期十字几何位置一致；允许一个
被遮挡。train 候选 69/120，复用 dev 13/120（E、SE、NW、W、NE；无 STOP）。
9/9 缩放回归和遮挡测试通过，全部十三个 dev 候选已目视检查，但不宣称独立准确率。
当前应固定该提取器，转向已有 train 来源更多时间窗的方向与 STOP 覆盖检查。
不要继续局部模板/阈值搜索，也不把稀疏候选直接当作已验证的策略训练数据。

尺度归一化已实现：底座与圆点共享九档候选尺度，坐标还原到原图，STOP 半径按尺度
归一化。9/9 train 合成变换检查通过；真实 dev 中圆点匹配达阈值 72/120，底座仅
3/120，最终三个 W 候选。train 候选序列保持不变。几何检查不代表语义准确率。
下一个局部问题是底座方向标记的几何定位；不继续扫尺度或降低阈值。复用本次缓存，
train 优先，已检查的 dev 只作为说明清楚的回归集；当前不启动策略训练。

首个提取器已完成，结论 `JOYSTICK_FIXED_SCALE_EXTRACTOR_NOT_TRANSFERABLE`。
在 train 生成模板并冻结后，候选覆盖为 train 56/120、dev 0/120。UI 可见性成立，
但固定像素模板存在尺度不适配和半透明底座受背景干扰的问题。候选不进入训练。
后续先用 train 的合成缩放/平移检查控件尺度归一化；本次 dev 后续只作公开说明的
回归集，不再称为未触碰的独立评估。不要通过降低置信度把错误位置强行变成标签。

用户确认原录像包含半透明摇杆。六个固定 train/dev 原分辨率短片段已完成目视检查，
可见活动圆点偏移、方向切换、保持和回中外观；商店遮挡需要返回 unknown。
结论只到可观测性，不等于自动方向标签已经验证。旧小地图失败实验保持原结论。

下一步复用本次缓存实现底座中心、圆点中心及活动状态提取；只在 train 校准，dev
原样评估。先输出偏移向量、置信度及 unknown；仅明确回中/释放可记为 STOP。
漂移底座必须逐帧识别，不能假定全局固定中心。先核对实际 PTS 和操作变化的对应，
再决定监督窗口如何对齐。摇杆区域只能用于标签，必须排除出未来 Actor 输入。
不需要新增录制、人工标签、模型或手机连接；本轮尚未启动自动标签批量生产。

## 2026-09-07：短缺口光流批次收口

本次用户授权独立实验覆盖既有 002/003/005 派生小地图；后羿身份不作为通用定位前提。
仅用旧 v2 内区候选初始化 LK 光流，15×15 窗口、2 级金字塔、最多 20 个角点；
角点质量 0.01、最小间隔 2 像素、初始化半径 12 像素。至少 4 个角点满足前后向误差
≤1 像素、单步位移 ≤8 像素；从最后直接检测到重接最多 1000 ms，重接误差 ≤3 像素。
所有参数在正式运行前固定；未重接的补缺全部丢弃。动作不进入跟踪器。

验收要求连续有效帧增加 ≥50%、至少 10 个独立补缺、12 个 QA 片段；动作响应至少
10 个事件、正向比例 ≥0.75、投影中位数 ≥1 像素。跨 session 规划另需至少两局各有
20 个互不重叠 16 帧窗口。QA 属内部检查，不产生人工训练标签。

批次已完成，结论 `PLAYER_FLOW_GAP_INSUFFICIENT`：002 连续帧增幅 9.6%、9 个缺口、
8 个窗口；003/005 均无有效窗口。动作响应通过不覆盖数据不足。保留原规则底座和
历史失败证据，不启动训练、不改阈值重试；本轮没有新的待调参任务。
工程预算上限 4 小时，GPU 0，新增产物约 1 MiB，低于 50 MiB 上限。

日期：2026-09-05。状态：阶段 A/B 通过；阶段 C 首个候选失败，等待新版评估合同。
适用分支：`hierarchical-policy-v0-prep`。

同日补充：已核对 MOBA 原始工程资料，来源与适用差异见第 10 节。只借鉴工程组织，
不迁移环境、不安装外部框架、不下载权重；80 小时 / 24 GPU 小时预算不变。

## 1. 目标与优先级

首轮交付一个可重现、可恢复、可替换策略的**离线工程 MVP**，先证明目标条件下的连续移动。
这不是手机可用版、游戏强策略或强化学习效果证明。后续在单独授权后接已有自建 App 执行层，
再考虑真实反馈训练。不能保证训练一定成功；可以保证按预算停止、保留证据并交付可运行部分。

本计划替代旧层级协议的未来阶段排序、参数扩容路线和“每次失败只能修一次”的研发安排。
`DELIVERY_PROGRESS.md` 仍是唯一已执行状态来源。旧配置、数据、报告、权重、已消费 test 和
失败结论不改写、不冒充通过；旧命令不因本计划自动解禁。新研发运行另记 run ID 即可，
仅在任务定义、输入或标签含义改变时升级合同，不为普通优化器调试另建一套架构。

阶段 A 已完成最小离线动作闭环；未执行训练、视频解码、手机采集或输入，也未推送 GitHub。
后续阶段 C 仍完全离线；不新增人类录制、人工标注、输入通道或额外设备门禁。

## 2. 当前资产与真实缺口

| 资产 | 本轮用途 | 不代表什么 |
|---|---|---|
| FrameBus、视觉事件 schema、UnifiedTransitionStore | 复用观察绑定、连续记录、终局先写、恢复机制 | 不代表视觉终局已可靠 |
| 持续摇杆、双指针、分方/分路开局、视觉战斗规则 | 保留接口，离线用执行替身；未来复用原设备实现 | 历史执行通过不代表当前模型会操作 |
| P0 temporal SSL v2 checkpoint | 可选初始化；与任务表现一起评估 | 时序分类收益不等于策略收益 |
| 现有约 10.8M 可训练参数 Movement 分支 | 默认复用，不先扩容或重建三个大模型 | 参数量不是智慧或性能证明 |
| 512/128 静态八方向序列 | 保留为历史诊断，不做导航验收 | 位置没有随动作更新，F1=1 不是连续移动 |
| 既有视频 train/dev | 小规模视觉域检查；有收益再追加预训练 | 无动作的视频不是行为克隆真值 |
| E1d/E1e 终局证据 | 保留失败记录，不在首轮继续研究 | 已消费 test 不得重开，不能接终局奖励 |

当前最重要的未知是：动作驱动环境中的可学性、真实画面预处理是否正确，以及模拟到真实的差距。
BatchNorm 只是过拟合异常的候选原因；现有过拟合和全量训练还同时改变了优化器、学习率及批次。
不预设原因、不把修某个损失数字当成整个项目目标。

## 3. 最小应用结构

```text
离线 RGB / 模拟器 RGB → 既有 FrameBus
                         ├─ 规则 Macro：选择本轮目标
                         ├─ Movement：朝目标移动 / 停止（唯一学习模块）
                         └─ 规则 Combat / purchase / death-stop
                                      ↓
                          确定性 Router → 执行替身
                                      ↓
                           下一观察 → TransitionStore
```

- 固定后羿、射手、一套布局；首轮蓝方下路为主，红方镜像只做确定性配置回归，不另训一个模型。
- 三个逻辑策略保留接口，但不要求同时训练三个模型。Macro/Combat 规则是过渡组件，不能计为学习收益。
- Movement 输出八方向加 `STOP`；持续移动由既有指针生命周期维护，不靠重复长按实现。
- 目标条件必须明确。新任务拟使用由可见 RGB 目标生成的目标标记画布，模型仍收 RGB；
  同一场景换目标应改变动作。模拟器真值只给 teacher/评估，不能通过目标标记偷渡隐含位置或地图真值。
  实施 A 阶段先明确标记生成、目标选择、坐标变换和新 schema，再编写训练代码；旧 RGB 合同不静默改变。
  标记生成器只读取 FramePacket RGB 和显式目标类别，后者只选择规则，不进入 Actor 张量。
  记录生成器版本/来源；遮挡、无目标或无法匹配时返回 unknown，不能从模拟 state 补位置。
- 学习策略、规则基线和执行层共用同一动作接口；先在模拟器完成状态变化，录像回放只能验证推理与记录，不能验证动作效果。
- 按 10 Hz 调度目标设计，慢模型可降到 5 Hz 并明确记录；指针保持与策略频率解耦。
  报告实际吞吐、p50/p95 延迟、错过周期比例；不以配置中的 Hz 宣称实测性能。
  16 帧按固定时间间隔采样，不是任取最新 16 张；训练与推理绑定同一采样周期。
  10 Hz 降为 5 Hz 不得静默改变模型的时间跨度：优先保持观察采样率、减少推理次数。
- 首轮不新建 RewardHub、数据库、消息总线、插件框架、仪表盘或第二套设备执行器。
  所有 reward 分量保持 0。模拟终局使用环境评估真值且标明 synthetic；录像结束记为截断，不能伪造 GAME_END。

## 4. 首轮预算与工作顺序

这是单人等效约 10 个工作日、最多 80 个有效工程小时的工作量预算，不是保证十个自然日内训练成功。
模型运行等待、外部授权等待单独记录；每天/每阶段结尾检查实际耗时，达到任一预算先停再报告。
GPU 小时按单张 GPU 占用墙钟计，失败、对照和诊断也计费；CPU 任务不能用来绕过总工程预算。

| 阶段 | 工程小时上限 | GPU 小时上限 | 具体交付与退出条件 | 超时/失败处理 |
|---|---:|---:|---|---|
| A 收束与接口 | 8 | 0 | 确认复用入口，定义目标画布、STOP、截断/终局、预算记录；规则基线跑通一条真实状态变化轨迹 | 缺接口只做最小补齐；不开始全量训练 |
| B 诊断与数据正确性 | 12 | 2 | 修复 overfit 未通过仍启动全训的问题；保存诊断权重；32 样本配合动作/目标反事实检查；小批现有视频检查旋转、裁剪和尺度 | 定位不到原因则保留失败与规则轨迹，取消后续学习运行 |
| C 连续移动小闭环 | 24 | 12 | teacher 实际驱动环境生成轨迹；训练一套 Movement；完成下述固定回合 dev 验收 | 两个有证据的修正候选仍失败就结束学习阶段，不扩模型/数据 |
| D 离线应用集成 | 16 | 4 | 不依赖 C 通过也能集成规则基线；选定策略完整运行 1→3→10 个模拟 episode，回放和中断恢复可重现 | 无法通过就交付最小可复现故障，不称 MVP 完成 |
| E 打包与结论 | 12 | 2 | 单一入口/配置/报告；固定最终模拟 holdout 评估；标明规则与学习收益、真实域缺口、后续接入前提 | 不用最终 holdout 修模型；明确交付等级 |
| 机动 | 8 | 4 | 仅用于上述明确工程故障，阶段结束登记剩余额度 | 用完停止，不自动续期 |
| **总计** | **80** | **24** | **含诊断、对照、失败与最终复核** | **追加预算需用户确认** |

阶段剩余额度不能自动转换为新课题；挪用机动额度必须在进度账本写原因。
若 B/C 学习失败，D/E 使用规则基线完成数据/运行面的有限交付，不声称已经具备学习移动能力。
若域检查失败，模拟器研发最多继续到本轮预算结束，不启动下一轮模拟器扩展来回避真实域问题。

体量和资源预算：

- 同时只做 1 个交付任务，最多附带 1 个直接阻塞问题；最多 1 个 GPU 训练进程。
- 本轮完整训练最多 6 次：初始候选、最多 2 次定因修正、1 次标签打乱对照、1 次初始化对照、
  1 次选定候选复现。顺序按需执行，失败可提前取消剩余运行；每次最多 2 GPU 小时。
  32 样本诊断最多 4 次、合计 2 GPU 小时；GPU 总预算仍为上限，额度不是必须用完的指标。
- 复用现有分支，单一部署策略暂以总参数不超过 20M 为规划上限；先量实际延迟/显存，
  不因训练门未过扩到 60M/200M，也不为降低账面体量强行重写已有模型。
  A 阶段实数清点 total/trainable/frozen（20M 指包含冻结部分的 total）、权重字节与峰值显存；
  未计数前不宣称满足上限，超出先复盘而非默认重写。
- 新增 Python 生产模块目标不超过 4 个、测试模块不超过 4 个、新配置 1 个；
  如超过先解释不可复用的原因，不靠塞入巨型单文件规避。此为计划复盘线，不加运行时安全检查。
- 增量产物最多 50 GiB：数据/缓存 30、权重/恢复快照 10、报告/余量 10；历史数据不计入增量，
  不重复拷贝整套视频、不删除历史产物。开始前检查可用空间，不足则缩小新采样量。
- 不新增付费云/API 调用，新增云预算为 0。电费与人工现金成本未知，不编造金额。
- 每个 run 记录工程用时、GPU 用时、增量字节、候选编号、目的、结果和剩余预算；
  初期用一份外部运行汇总即可，不新建预算管理服务。

## 5. 训练与验收怎么避免卡住

### 5.1 调试不是正式评测

32 样本诊断先于全训，使用与正式训练一致的预处理和明确可比的训练配置，分别报告 train/eval 模式。
不通过就不执行长训练；始终保存诊断 checkpoint、配置、随机状态、loss 和梯度摘要，
与可部署 checkpoint 分开。保存失败权重不等于晋级。过拟合门在首次运行前确定，
新任务沿用 eval accuracy ≥0.95、交叉熵 ≤0.05，32 个样本须包含 STOP 和八方向；
两种模式结果分别记录。A 合同还需固定分组采样及目标反事实判定，再运行首次诊断。
本轮不得为通过而放宽；旧 0.95/0.05 门的失败仍是失败。

排查优先级：标签/坐标/时间对齐 → 梯度和参数更新 → train/eval 差异 → 优化器/归一化。
每次只改一个待验证原因；不能同时改变学习率、模型、采样后称已定位 BatchNorm。
相同原因连续两次没有改善，停止该假设，用剩余预算决策改任务或降级，而不是无限换版本。

### 5.2 数据是真轨迹，指标是真行为

初始生成 64 个 train、24 个 dev 模拟轨迹，最长 128 step；按整条场景/轨迹拆分，
真实 step 改变位置后重新渲染。覆盖起点、目标、距离、停止、少量障碍和目标切换。
种子、背景、血量和时间不能按方向分配；使用不同几何/路线划分，不能只证明匿名 ID 无交叉。
teacher 必须先证明能到达；失败轨迹单独保留，不冒充正确示范。
新增最终模拟 holdout 固定 40 个可行场景（不是已有视频 test），在选定模型前只存种子/规则，
不解码、不看结果、不参与调参。A 阶段固定生成和可行性规则。
只有 C 通过才消费最终 holdout；R0 集成使用独立开发种子，不消费学习策略的最终评测。
所有策略成对使用相同场景；固定方向基线为持续 E，随机基线为每步均匀九动作、固定单个随机种子，
两者同样需要 STOP 判定。A 合同绑定实现 hash、场景 IDs 与种子；差值是同场景成功回合数之差，
区间采用 Wilson 95% 区间。单随机种子对照是工程筛查，不是统计优越性证明。
另加一个**无训练的 RGB 几何导航基线**：复用目标/自身可见定位，按相对方向走、到达后 STOP；
不读模拟真值，不新建规划器。随机/固定方向只是低水平检查，不能证明学习的实际增量价值。
同一组 rollout 同时报学习与几何基线的到达、碰撞、耗时，复用已有评估代码，不增加训练次数。

首次 C 使用以下 v1 dev 工程门，原失败结论保留；v2 的预先固定约定见下方 C 实现说明：

- 24 个回合：规则 teacher 至少 23/24 到达；学习策略至少 21/24 到达，且比随机/固定方向中更好者高至少 8/24。
- 到达必须在限定步数内进入目标范围并连续正确 STOP 3 步；目标切换后应重新出发。
- 单报纯学习与混合结果。学习策略被规则接管的回合不计入纯学习到达成功数。
- 碰撞 step 比例不超过 5%；方向反复折返比例不超过 10%；计算窗口、分母与零分母规则在 A 阶段落实。
- 报告到达步数、相对 teacher 路径长度、卡住比例及每种场景结果。分类 F1 只用于查错。
- 同 RGB 换目标、错误目标标记、打乱标签做针对性检查，证明动作确实受任务条件影响。
  对照失败不追加大规模训练，回到目标/标签定义。

24 局只是小规模工程筛查，不是泛化统计证明。E 阶段只评估一次新 40 场景 holdout：
到达至少 34/40，teacher 至少 38/40，学习相对随机/固定方向更好者至少多 12/40；
碰撞、折返和 STOP 定义不变，报告原始计数及成功率区间。失败则保留模拟学习未晋级结论。
预算内可调 train/dev；不能用消费后的 holdout 或旧 E1d test 修模型。
R1 的通过仍只说明模拟学习任务可用。若学习策略相对 RGB 几何基线无收益，默认应用候选保留规则，
学习权重只做可替换候选；不为证明“模型一定更好”扩任务或新增模型。真实域未验证时两者都不部署。

### 5.3 工程闭环不被视觉终局绑架

离线验收区分：模拟任务自然终局、录像 EOF 截断、异常中断。EOF/超时不是胜负事件。
使用现有 schema 能表达的字段；若缺 truncated，A 阶段做最小明确扩展并测试，不滥用 `done`。
终局样本先存后停，所有 reward 为 0，记录 checkpoint 和事件实现的真实 hash。
恢复保存 optimizer、随机状态、序列状态和 replay 游标；跨进程加载后续接，无重复 step。
控制器-only 模式明确标无学习权重，不能拿随机权重充数。
复用/补齐逐步 requested_source、executed_source、fallback_reason 的等价记录，汇总学习实际执行比例、
接管回合数和 pure/hybrid/rule 成功数；学习不晋级时结论固定为 `R0_ONLY / learned_not_promoted`。

1→3→10 局分别验证完整记录、跨局 reset、长运行稳定；任何因果倒序、损坏引用、动作积压、
丢终局样本或恢复重复都阻止规模升级。latest-frame 丢弃旧帧是可计数的设计行为，不承诺传输零丢帧。
离线执行替身可验证指针事件次序，不得宣称已重新验证手机实际输入延迟。

## 6. 现有视频的预算化使用

本轮结束后启动的首个真实RGB预检已经执行并冻结失败。合同固定3个session、9段、288帧，
只使用train/dev。总体目标配对覆盖率0.4792，一局train仅0.0938，标记跳变率0.4128；因此
真实目标条件尚不可用。test、训练、手机和R2均未打开。下一步须另建小地图检测/跟踪合同，
在读取更多session前固定完整小地图裁剪、候选身份和时序连续性规则；不得降低v1门槛重跑。

后续v2按职责分离收窄问题：固定蓝方射手下路目标由Macro合同给出，小地图画布只负责完整裁剪
和空心目标环，不再从模糊红色像素猜目标。相同288帧的内容框、非黑裁剪、重复生成和双目标
反事实检查全部通过。该结果只证明目标条件能够进入RGB；自身定位、分路坐标语义和策略收益仍
未证明。下一步仅允许另建32样本模拟learnability合同，不直接进入R2或真实动作。

该独立32样本门已经通过：STOP 8、八方向各3，全部窗口有动作前状态变化，8组换目标反事实
成立；686,281参数GroupNorm+GRU达到accuracy 1.0、loss 0.00614和九类recall 1.0。诊断权重
不能用于正式训练。下一步可 fresh-init 生成并训练64/24条同输入模拟轨迹；仍不打开R2、test或
真实RGB训练。

64/24正式候选已完成并冻结失败。teacher和RGB几何均为24/24；fresh模型训练loss降到0.01414，
但epoch-20只到9/24，未过21/24门。碰撞、折返和效率门已过，失败局仍大量提前STOP。后续不再
增加epoch、数据或重调采样；若继续学习，只允许在同一数据/训练预算下另立空间关系模型合同。
真实玩家图标可观测性仍须独立解决，模拟关系模型通过也不能直接进入R2。

首个空间关系对照已在32样本门停止：93,611参数的双注意力槽＋坐标差＋GRU仅达到accuracy
0.625、loss 1.258，N/S/E/W recall为0，未启动64/24正式训练。仅靠动作loss不足以给两个槽
分配玩家/目标身份。若继续，须另立自动模拟定位辅助监督合同，同时检查定位误差和九动作门；
诊断权重不复用，真实玩家图标证据仍为R2独立前提。

联合定位辅助诊断也已在32样本门停止。平均槽误差3.94像素通过5像素门，但网格准确率0.6875、
动作accuracy 0.4375和loss 1.055均失败，未启动正式训练。后续若继续，只允许另立“两阶段：
先定位、冻结定位层、再拟合动作”的合同；不能复用失败权重或把坐标标签放入Actor输入。

两阶段v1因16x16精确格命中0.6699停止；只读诊断确认全部预测在相邻一格内。预先冻结的v2改用
“一格内≥0.95”并保留5像素误差门，定位以1.0/3.97像素通过，冻结层也保持不变；动作阶段仍仅
0.8125 accuracy、0.871 loss。至此停止继续优化模拟器head、更新数或采样。下一步必须先验证
高清真实小地图中的玩家线索，再决定是否值得重新设计策略模型。

既有高清裁剪后小地图session 002/003/005的玩家线索预检已通过：4,455帧中每局覆盖率均高于
0.977，单候选比例最低0.879，跳变p95最高1.32像素。它证明自动配色连通域是稳定工程线索，
但没有独立真值，不能宣称语义身份或方向准确。与Macro目标环的只读组合也已完成：原始方向
覆盖率最低0.977，三帧确认后最低0.976，方向变化稳定；但只观察到E/S/SE，且后两局全为S。
因此本轮冻结为组合工程通过、方向多样性不足，`policy_training_allowed=false`；不恢复模拟器
模型调参，也不进入R2。

随后新增的最小反事实可学性门不要求新录制：从session 002选择5个互不重叠的16帧窗口，只替换
Macro空心目标环，组成STOP 8及八方向各3的32样本。686,281参数GroupNorm+GRU在唯一一次
seed-0诊断中达到accuracy 1.0、loss 0.00732和九类recall 1.0。该结果只证明五个源窗口上的
目标条件可以被拟合；没有源窗口或session泛化，也不是执行动作标签。因此诊断权重不可复用，
正式训练和R2继续关闭，下一门必须先独立定义源窗口holdout。

定位v2复核随后确认旧高覆盖主要来自右上角固定UI。排除该区域后，session 002仅208/1485帧
可定位，但17个既有发送动作在1秒后的位移投影有15个为正，中位数2.01像素；session 003为0，
session 005仅4帧零散噪声。因此定位范围冻结为session 002局部可用。五个有效源窗口的一次性
五折×三输入评估也已完成：full accuracy/macro-F1为0.7778/0.7370，最差折0.3333；遮挡玩家和
仅目标环accuracy仍为0.6889/0.6667，full优势0.0889/0.1111均未过0.15门。当前模型路线冻结为
`REAL_COUNTERFACTUAL_MODEL_SHORTCUT_OR_NO_GENERALIZATION`，不再调模型；下一步只允许修复多局
自动玩家定位覆盖，正式训练、R2和设备输入继续关闭。

2026-09-06 的有界定位修复已做完一个候选：从002既有局部轨迹自动提取头像模板，加连续两帧
确认和丢失重捕获，不依赖红色配对。结果为002的155/1485帧、003/005均0，未扩大覆盖，不接入
导航，也不降阈值重跑。主画面/小地图并排QA显示003抽查画面在泉水、002后段和005多数抽查
画面也在泉水附近，自身头像存在边缘裁切迹象；005其他移动绿色标记不能直接认作自身。
这些是开发目检，不是逐帧真值；死亡、镜头跟随和完整ROI边界仍不能仅凭这些缩小图确认。

唯一下一步收敛为**已有源画面的完整小地图可恢复性检查**，最多半天：先查既有train/dev
画面的裁剪记录与少量代表帧，确认泉水角落是否保留。若原有源仍保留完整区域，只做一次
完整ROI重提取与并排验证；若源不含该区域，明确现有数据限制并停止此定位方案。不能从
已裁掉的128×128像素推造精确位置，不能把泉水状态替代地图坐标。不增加模型、训练或新录制，
不打开test，不连接手机；不再以增加候选覆盖率替代身份正确性。

该可恢复性检查已完成：三局teacher只有裁剪缓存，无法恢复同局缺失像素；原视频库中两条
已绑定的横屏train/dev视频则保留了完整小地图和下缘外侧区域。改用这两条作为替代定位来源，
不把它们说成原三局恢复版。九个时点的QA中，一条dev视频中段有商店遮挡；竖屏来源的旋转
预览方向错误，暂不纳入下一小批，不继续为其增加旋转诊断。下一步只处理两条横屏原分辨率
画面：各选一段短窗口，保留小地图边缘余量，先确认自身身份和可见时段，再检查连续定位。
不要从整体缩成128×128的旧缓存再次放大小地图；也不要把“完整ROI存在”当作“玩家定位通过”。
沿用半天上限，若身份仍无法区分则输出unknown并明确停止原因，不以覆盖率换取误识别。

原分辨率短窗口pilot已完成：每局16帧，先裁剪再缩为256×256，保留主画面对照。整数微秒采样
修复了浮点边界漏帧；两条真实窗口在修复前后内容相同。绿色圆环加邻近连续确认，在dev窗口
保留9帧候选，train窗口因头像重叠仅出现一次孤立候选，确认后为0。身份仍未独立验证，不开放
训练或导航。不要强求遮挡窗口输出位置，也不继续降低圆环颜色/形状阈值。本批检查到此收口，
后续先复用已保存32帧分离地图本体与外围背景、标出unknown的原因；若要增加定位数据，只从
已有train来源选可见片段，不能把dev窗口回填训练集，不能将规则建议视为人类按键。

缓存背景对照已经完成：dev候选55→36（保留边缘）或14（仅内部的诊断），但确认帧均为9；
train均为0。停止追加背景过滤。随后一次性检查同一train视频5/10/15%位置的三个16帧窗口，
不调检测阈值，分别确认0/13/15帧。15%窗口每帧唯一候选，首帧仅等待确认，可优先用于下一步
感知QA；10%窗口有一帧面板遮挡及重新确认。两个窗口来自同一视频，不是独立对局，也不证明
玩家身份、真实按键或轨迹泛化。保留5%和20%遮挡失败，不继续强求补全；不把这次样本筛选
当成定位准确率提升。下一步以现有清晰片段核对身份与坐标稳定性，不再改变圆环检测器，
暂不扩展到其他source、不训练、不接手机。

这两个清晰缓存的身份/坐标QA也已完成：四个固定±4像素平移下，13/15帧的112次坐标比较误差
为0且unknown不变；恢复到原画面采样像素的往返转换一致。主画面与头像放大图支持“两个片段
为同一绿色头像”，但不构成独立的自身身份或后羿身份验证。坐标稳定性、身份语义与真实动作
监督必须分开。到这里停止对同一批片段追加类似审查；它们作为固定的非推广感知回归样例。
后续若推进学习，应先把任务明确为“绿环视觉锚点”的弱监督任务，单独说明身份不确定性和
用途边界，而不是把当前候选升级成真值或重新开启失败Movement训练。本批不执行新的学习。

首次跨session弱锚点审计固定8个train、4个dev，使用与15%探索预检不重合的10/30/60%窗口，
共36组、576帧。Train为7/8 session、130确认帧，通过6/128门；dev为2/4 session、64确认帧，
未通过3-session门但通过48帧门。状态冻结为`WEAK_VISUAL_ANCHOR_COHORT_INSUFFICIENT`，不训练、
不改门槛。目检显示失败包括商店遮挡和不同录像中绿环表现不一致；简单蓝青视野框连通域又被
塔、路径和UI大量混淆，不作为当前实现。允许一次独立v2覆盖修复：按匿名顺序加入4个此前未
打开的横屏dev session，保持检测器、10/30/60%窗口、session支持定义和原4局结果不变。
合并dev若仍少于3个支持session就停止绿环弱监督路线；若达到，只开放定位数据审计，不直接
开放Movement训练或语义身份声明。

v2覆盖修复已按匿名哈希顺序加入4个新dev session，冻结的三个窗口和检测规则不变。合并结果
为train 7/8、130帧，dev 6/8、171帧，所有数据覆盖门通过；v1失败报告不修改。新增QA未见固定
右上角UI锁定，仍有商店遮挡和多头像重叠，故结论只到`WEAK_VISUAL_ANCHOR_COHORT_SUPPORTED_QA_ONLY`。
下一步允许做一次session隔离的九类反事实目标数据门：每个16帧窗口仅使用末帧已确认锚点，
目标环在锚点之后绘制，标签由目标相对方向确定；train/dev session不得交叉。该数据门通过也
只开放弱锚点关系诊断训练，不等于人类动作模仿、真实玩家身份或可执行Movement。

九类反事实数据门已通过：仅选择确认帧不少于8且末帧有锚点的窗口，使用256画布、24像素目标
偏移和7像素空心环。Train 8组/7 session/72样本，dev 9组/6 session/81样本；每组九类完整，
session零交叉，当前检测结果与冻结报告逐窗口一致。允许下一步物化这153个样本并运行一次小型
关系诊断训练；不得把它称为动作BC，也不得用dev选择检测阈值、身份规则或数据窗口。诊断若
失败，停止模型路线；若通过，也只证明弱绿环—目标环关系可跨session学习，不能直接接手机。

153条逻辑样本已采用去重结构物化：17个源窗口只存一次，样本索引保存group、九类label和目标
坐标；train/dev为72/81，每类全局17。源RGB保持不变，训练时复制后绘制固定黄环，再用固定偶数
索引缩至128。下一步唯一允许的训练是seed-0小诊断：full、锚点遮挡、仅目标三个输入使用相同
GroupNorm+GRU、优化器和更新数；先过小样本可学性，再看session隔离dev。Full需同时达到
accuracy/macro-F1 0.80、每类recall 0.60，并相对两个对照各领先0.15，否则冻结失败，不调环、
检测器、split或模型。即使通过也仅保存诊断证据，不开放手机执行或RL。

该唯一训练批次已结束。v1先在256画半径7目标环再缩放，overfit36为0.111/2.203；确认标签均衡、
153个full clip唯一且每帧仅约32个目标变化像素后，允许一次渲染顺序修复。v2先缩至128再画
半径7环，overfit36达到1.0/0.00647，但正式session隔离dev中full、锚点遮挡、仅目标全部为
accuracy 0.1111、macro-F1 0.0222并只预测SW。由此只能说明小样本可记忆，不能跨session学习
绿环—黄环关系；对照差值均为0。冻结`WEAK_ANCHOR_RELATION_DIAGNOSTIC_FAILED`，不增加epoch、
模型、窗口、伪标签或第二种定位特征，不保存checkpoint。真实Movement继续使用确定性规则底座；
若开启下一工程周期，应转向独立的可观测事件/Reward数据面或英雄绑定数据，不再沿本路线调参。

新的可观察事件周期已从`DEATH/RESPAWN`工程回放开始，不重开终局test或旧检测调参。冻结E1a
dev-death的285帧经现有HealthTemporalEventEngine生成284条NOOP transition，复现1次死亡、1次
复活和17次HP变化；最后一条以VIDEO_EOF截断，先写Store再结束。所有reward为0、所有样本
`training_eligible=false`，但284条因果校验全部有效，SQLite integrity为ok。为支持这种合法的
不可训练诊断链，Store只在上一条`causal_order_valid=false`时传播链错误；单纯主动禁用训练
不再污染后续因果状态。该结果只证明Event→Transition→Store工程闭环，不证明死亡语义准确、
HP数值准确或Reward可用。下一步应从既有train/dev派生数据只读盘点跨session死亡候选；在至少
三局一致前不赋死亡奖励，不接手机，不打开test。

现有operational派生数据的完整盘点已执行：共8局、7080帧，冻结引擎只在death-stop局产生
1组DEATH/RESPAWN，其余7局均为负对照且无死亡误报；session 002虽有6个legacy hard-stop帧，
健康条覆盖仍为1.0且没有死亡事件。因此`DEATH_RESPAWN_CANDIDATES_INSUFFICIENT`，3局正样本门
未过。不能用hard-stop、血条暂时不可见或已知session名称补语义标签。下一步若继续，只能对
既有train/dev原视频建立动态候选预检，先找死亡倒计时/复活回场等第二视觉线索；不读test，
不训练Reward模型，不修改E1a阈值。

第二线索共识已实现并冻结：Health状态为DEAD且既有全屏死亡横幅hard-stop仍有效时才输出死亡；
横幅解除后Health稳定为ALIVE才输出复活。健康消失单独和hard-stop单独均不输出死亡。8局重放
保留death-stop局的1组DEATH/RESPAWN，拒绝session 002的2个hard-stop上升沿，其余局无事件、
无单边事件。它提高候选特异性但正例仍只有1局，状态为`DEATH_BANNER_CONSENSUS_DATA_INSUFFICIENT`。
下一步只能在既有train/dev原视频上做最多12局的动态第二线索预检；先确认死亡横幅/倒计时布局
是否跨视频可见，再决定是否物化短片，不把本地hard-stop位直接当外部视频标签。

该12局原视频预检已全长完成：652,190解码帧、52,868个200ms样本，8 train/4 dev均完成三张
上下文QA。按mobile横幅像素数归一化后的顶部ROI在每局命中118–850个样本，主要是比分栏、
击杀提示和常驻红色UI；中央健康条规则也不具备跨布局身份稳定性。自动输出11个配对session和
1个单边session，但QA候选帧中英雄仍在移动/战斗，因此这些全部不能算死亡语义证据。冻结为
`NATIVE_DEATH_CUE_PREFLIGHT_DOMAIN_MISMATCH`，不调整ROI、颜色比例、健康裁剪或时序，不物化
短片、不训练Reward。现有50GB视频仍可用于无标签视觉预训练，但不能靠这套mobile布局规则
产生死亡标签。下一步项目应停止当前Reward感知扩展，保留零奖励Event-to-Store工程基线；
若无新语义来源，不再尝试塔伤、经济等更难的伪标签支线。

本轮现已按单入口交付收口为`R1_ENGINEERING_OFFLINE_ZERO_REWARD`。`package-cycle`嵌套复核过的
R0规则包、零奖励死亡/复活Event-to-Store回放和四份关键失败边界；外层manifest逐文件绑定，
Event SQLite通过只读backup且不含WAL/SHM，`verify-only`在新进程只读通过。最终包含R0 90条、
Event 284条transition和385个帧bundle，总reward与输入命令均为0，Event训练资格为0，无模型
checkpoint。该R1是工程能力整理，不是R0策略升级，也不授权真实视频策略或手机执行。没有新
语义来源时当前工程周期到此结束；后续立项应先明确可获得的标签/环境反馈，再选择Reward或
英雄专用动作数据，不继续从同一批无真值视频制造新支线。

Movement分层控制也已作为`R1_HIERARCHICAL_RULE_OFFLINE`独立冻结。外层包嵌套验证既有R1，
再绑定方向保持、换向合同、两次神经模型失败和最终几何回放共6份报告；不复制数据集或权重。
当前模拟器执行链固定为Macro精确目标几何换向、Executor保持方向、Router停止。该结果只证明
合成标记可见条件下的确定性工程组合，不解决真实玩家定位或Macro目标可观察性。

下一周期的后羿数据只读审计已完成。149个MP4的文件/目录名无后羿、Houyi、射手或发育路线索；
正式审计先按cohort过滤，仅打开103 train和23 dev容器元数据，英雄关键词仍为0，test容器与帧
均为0。hero profile仍是`TEMPLATE_NOT_CONFIGURED`，owner attestation没有hero字段；152份summary
只有PixelArena Stage A声明`hero=houyi`，完整真实session绑定为0。因此冻结
`HOUYI_BOUND_REAL_DATA_NOT_AVAILABLE`，不从现有50GB录像做后羿专用训练。此前一次探索性元数据
检查错误地在split过滤前打开全部149个容器头，包含23个test；未解码帧、命中0、未用于选择或
调参，但作为边界偏差保留。下一步必须是episode级owner声明加不可变加载面板/英雄参考证据，
且只适用于声明后的session；不能追认旧数据或用技能存在、射手外观替代身份。

交付级全量检查最终为448项pytest、Ruff、strict mypy和项目安全全部通过。首次未显式绑定
当前`src`时，共享环境导入了相邻仓库并在collection阶段报33个缺模块错误；未执行测试主体。
Makefile统一入口现固定`PYTHONPATH=$(CURDIR)/src`，随后原命令通过。该修复只消除环境歧义，
不改变任何模型、数据或运行证据。

不再要求用户录制或标注。先在既有 train/dev 中抽取至多 12 个 session、每局 3 段短片，
总计最多 36 段，跨旋转、尺度与场景检查。人工查看由开发者完成，仅做预处理 QA，不创建训练标签。
旋转矩阵正确性需结合图像确认；“成功解码”不能当作方向正确的证据。

既有录像用于视觉初始化和域差距诊断，不凭屏幕运动推定人类按键，不把 teacher 建议称为人类动作。
当前 P0 可先直接复用；只有初始化对照显示策略收益，后续才预算扩大 SSL。
录像 shadow 可测输出变化和目标敏感性；没有真值时不得报告真实动作准确率或真实到达率。
无法从当前 RGB 稳定生成目标标记时，真实域学习接入停止；交付模拟版并明确缺少的可观测证据。
上述 36 段逐段报告目标可生成帧比例、unknown 比例和标记跳变率；A 阶段固定这些计数定义。
这些无真值统计只用于提前发现域问题，不能单凭高覆盖解锁 R2，也不能凭空制定真实方向准确率门。

## 7. 交付等级和后续投资

| 等级 | 必须交付 | 不能宣称 |
|---|---|---|
| R0 离线工程基线 | 规则控制完整回合、可恢复 store、单一运行入口、报告和测试 | Movement 学会或手机可用 |
| R1 模拟学习 MVP | R0 + 纯学习 dev/新 holdout 达标、因果目标检查、独立 checkpoint | 真实场景有效、会对线或会赢 |
| R2 自建 App 有限应用（本轮不执行） | 单独授权；复用现有输入接口；读屏验证后 1→3→10 局，单报人工干预/规则接管 | 模拟器表现可以替代真机证据 |

R0 是可交付下限而非最终学习目标；C 失败不能自动宣布项目完成。
本轮结束必须报告“继续哪项、停止哪项、缺什么证据、下一轮多少钱/多久”，再由用户决定追加预算。

后续顺序：R1/真实域可观测性 → R2 固定场景应用 → 固定视觉反馈的有限后训练 → Macro 或 Combat 中
失败占比最高的一个学习模块。RL 范式保留，不在无可信 reward 时启动。
首轮不做 PPO、MoE、大规模超参搜索、多英雄、多布局、经济/经验检测或全量多事件 CV。
终局、HP、塔伤等按实际失败频率逐项补，不能把“全视觉系统完善”设为运行工程基线的前置条件。

后训练另立预算，先离线确认 reward，再冻结事件模型、单个策略更新、固定回合对照和可回滚权重。
不同时更新感知、reward 和三个策略。自动采集在当前离线范围外；未来如确有需要须另行授权，
不以用户拒绝人类录制为由默认启动采集。

后续仅保留两个**未授权的预算预估包**，不预先铺开代码：

| 后续包 | 启动条件 | 追加预算上限建议 | 到点交付/停止 |
|---|---|---|---|
| R2 固定自建 App 接入 | 用户单独授权、目标视觉可观测证据和现有设备条件齐全 | 24 工程小时、0 训练 GPU 小时、10 GiB | 复用接口做读屏/1→3→10 局；域失配则报告缺口，不继续模拟器调参 |
| 单策略有限后训练 | R2 稳定、独立可信反馈已有，不依赖补齐全套 CV | 40 工程小时、16 GPU 小时、20 GiB | 对固定初始化做成对完整回合比较；无收益回滚，事件系统未就绪不启动 |

若依次获批，三包累计规划为 144 工程小时、40 GPU 小时、80 GiB 增量；这是有条件的投资封顶建议，
不是当前授权，也不保证这点资源能解决域迁移或奖励识别。超过范围的感知研究/第三个策略学习重新决策，
不能隐含塞入这些预算。初始 80 小时到期先复盘，再决定是否值得投入下一包。

## 8. A–E 技术实施建议

以下是新实现的默认方案，不是已验证结果；A 开始时把必要数值写入一份配置，后续只记录差异。
不为每个子步骤写新协议、审批单或独立报告。阶段 A/B 共用小轨迹，C 才生成完整训练集。

### A：先让一条轨迹真正动起来（PASSED）

已实现 `movement-mvp --mode stage-a`，复用 RichPixelArena、FrameBus 和
UnifiedTransitionStore。固定蓝方后羿射手下路从 `(2,4)` 连续执行 6 次 `E` 到 `(8,4)`，
随后执行 `STOP`；7 条 transition 均有效，最后一条以
`NAVIGATION_GOAL_REACHED / TERMINATED` 先提交再退出。配置步长为 100 ms，reward 总和为 0，
`input_commands_sent=0`。正式产物 basename 为 `stage-a-seed0-v1`，占用 76 KiB；
summary SHA-256 为 `52f1227c3ddfa4c91d5d1b76ea1895eb2d94262eb9a86e879c92172627de8555`。
本次未单独计时，因此不虚构已耗工程小时；GPU 用时为 0。

1. 复用 `rich_arena.py` 的 `RichPixelArena.reset/step/observe` 和既有 RGB renderer；
   新增薄任务适配层，负责起终点、任务结束和动作映射，不重写物理/碰撞/渲染。
   先做无障碍单目标，跑通后再使用现有可碰撞结构；不要把“少量障碍”扩成地图编辑器。
   参考 hok_env 的环境/策略分工，只给本任务一个 reset/step/close 薄接口；
   不引入 Gym 兼容层、网络 RPC、Gamecore 或通用多环境注册表。
2. 定义 `reset → RGB → proposal → step(action) → next RGB`。
   一次动作调用一次环境 step，禁止只改 tick 后重画。teacher 可读 state，Actor 只收画布。
   导航目标到达是 `navigation_goal_reached` 任务终止，不冒充水晶终局或 WIN/LOSS。
3. 画布建议沿用 128×128 RGB、16 帧，颜色和归一化复用 P0；在可见目标周围画固定样式空心环，
   不画指向箭头、最短路径或带方向含义的颜色。先用模拟画面的简单颜色/连通域定位，
   不额外训练检测器；真实视频用已有 RGB 规则，失败返回 unknown，不能拿模拟规则冒充真实检测。
   两个可见候选目标中由规则选一个；目标变更重新构造窗口，避免旧目标标记混入新任务。
   必须定位可见自身锚点，不能把固定画布中心偷偷当作自身位置；第一组样本就改变自身位置。
   teacher 使用的障碍/通路若未在 Actor 视野中表现出来，该任务先删掉不可观测部分，
   不要求 RGB 学生拟合只有真值 teacher 才知道的最短路径。
4. 新动作顺序固定为 `STOP,N,NE,E,SE,S,SW,W,NW`，用小映射转换既有方向枚举，
   不重排旧八分类权重含义。unknown 由 Router 停止并记 fallback，不伪造成策略主动 STOP。
   时间建议模拟每 step 对应 100 ms，仅作逻辑时钟；CPU 批量生成不实际 sleep。
5. 定义碰撞率为被环境阻止的移动 step / 请求移动 step；折返为同目标下相邻两个非 STOP
   动作恰好反向的次数 / 可比较动作对；零分母记 0 并保留分母。卡住记连续 10 个移动请求位置未变。
   路径按实际位移累计，斜向用欧氏长度；起点已在目标内的场景不计算路径比，单列 STOP 检查。
6. `transition_store.py` 当前有 done、无显式 truncated。优先增加向后兼容的截断/任务结束原因字段，
   旧记录缺省只代表旧 schema，不回填历史含义；不要复制另一套 Store。
   自然任务完成、128-step 超时和录像 EOF 分别编码，reward 均为 0。
   参考 wzry_ai 固定提交的终局漏存问题，先写一条成功样本和一条超时样本，再连训练器。
   技术验证通过不构成导航训练通过；不把“脚本可启动”升级为“RL 效果已证明”。

最少验证：一条固定轨迹发生位移、换目标改变 teacher 方向、STOP 不移动、RGB 标记不读 state。
用参数化小测试覆盖动作坐标及红方镜像即可，不枚举所有英雄/布局。

### B：共用训练路径，定位问题（SIMULATOR LEARNABILITY PASSED）

已物化 32 个互不跨 episode 的动作前窗口：STOP 8 个，八方向各 3 个；每个窗口 16 帧，
历史状态均发生真实位移，标签动作在窗口结束后实际执行。模型输入只有目标标记 RGB；
数据集 SHA-256 为 `ea75e7ec0b4e85bdff217b326b11bfd0bd8bf3fb7fc0c6b5103041a6dd936328`。

同一 `train_step` 的三次有界诊断结果：

| 候选 | 结果 | 解释 |
|---|---|---|
| 旧 P0 Movement 分支 | eval accuracy 0.5938、loss 0.7872 | 失败；不进入全训 |
| 同分支冻结全部 BatchNorm | accuracy 0.25、loss 2.1224 | 更差，停止 BN 假设 |
| 686,281 参数 task-specific GroupNorm+GRU | accuracy 1.0、loss 0.00745，九类 recall 全 1.0 | 通过小样本可学性门 |

三次均为 200 更新、batch 8、lr 0.001、相同数据和门槛；诊断 checkpoint 与完整训练权重分开，
`full_training_called=false`。任务专用报告 SHA-256 为
`c16cca8157e9cc99b3e4c363df35ea914d928d2b37a6cece59a12a865f085981`，checkpoint SHA-256 为
`f7e0df57c947426e69f6a894fba5799d46a23710e247412201cfe48a7c27f918`。
首次任务专用运行在训练完成后的评估接口报错，最小修复后重试一次；失败运行未产出指标。
报告的 normalization 字段随后仅作事实性修正为 `group_norm`，metrics/checkpoint 未改。

当前目标检测颜色与标记规则明确是 PixelArena 专用，不能用于真实视频。为减少无效解码，
本阶段未执行真实视频 QA；这将 Stage C 限定为 simulator-only，并继续阻止 R2，而不是把模拟通过
冒充真实域通过。完整训练仍未开始。

训练链路已按 `CAUSAL_OVERFIT32_PASSED_TASK_SPECIFIC` 收口：任务专用模型成为后续正式训练
默认值，P0 仅保留为显式失败对照且无自动回退。包含一次评估接口失败及其唯一重试在内，
四次诊断额度已经耗尽，配置会拒绝第五次 overfit。正式训练必须从 seed 0 新建模型，不能加载
32样本诊断权重。共享训练步在第一次更新后立即记录 loss、梯度范数、有限性和参数变化；
CPU失败路径测试确认失败仍保存诊断 checkpoint 且不会调用完整训练。

1. 从少量真实 step 轨迹取 32 个因果窗口：8 个 STOP、八方向各 3 个。窗口只包含决策前帧；
   开始不足 16 帧重复首帧，禁止跨 episode 拼接。样本含到达前后，不能全取静止画面。
2. 复用现有 `MovementBranch` 的 encoder/GRU/head 结构，输出层改九类；旧 P0 初始化只加载
   shape 匹配的 encoder/GRU，head 新初始化。旧冻结运行入口与配置不改成“可重跑”。
3. 新任务用同一 `train_step` 跑 overfit 和正式训练，避免现有 Adam/AdamW 两套代码差异。
   建议起点 AdamW、lr=1e-3、weight_decay=0、batch=8、FP32、seed=0，
   overfit 最多 200 更新，全训最多 20 epoch，均受时间预算限制；这些是起点，不是最优参数宣称。
   冻结 trunk 保持 eval，其他归一化先保留原行为；没有证据不直接换 GroupNorm。
4. 首个更新检查 loss 有限、可训练参数确实更新；每 20 更新记录 loss，结束报告 train/eval
   accuracy 和 loss。只存梯度范数摘要，不逐参数输出或写大直方图。
   overfit 失败保存 last 诊断快照后立即 return，测试用 spy 确认正式训练函数没有被调用。
5. 若 train 好、eval 差，才在同一组输入比较 BN 模式；若两者都差，先查标签和梯度。
   诊断读取训练模式指标时保留/还原 running stats，不能让“评估”改变待比较权重。
   按预算最多四次小诊断，不增加独立多 seed 研究。
6. 视频 QA 先选 3 个不同旋转/比例 session，确认解码→旋转→裁剪→resize 顺序；
   只有出现未覆盖格式才增加，12 session/36 短片是上限，不是必须完成的审查配额。
   用内存预览复核，不另存原始截图或创建标签集；已有文件身份和 split 证据有效就复用。
   现有动作缺失视频只用于视觉用途；不要为接入 Hokoff 的离线 RL 流程补造 action/reward 字段。

最少验证：非对称 RGB 图案的轴/颜色正确、窗口不越界、一次参数更新、失败不全训、保存后可加载。
不要为 loss 公式、配置常数或第三方库行为写镜像测试。

### C：小数据 BC → 真实 rollout（24 小时 / 12 GPU 小时）

首个候选已完成并冻结为失败。64/24轨迹数据门通过，teacher 为64/64和24/24；fresh-init
686,281参数模型训练loss从1.8618降至0.0272。独立dev中epoch 10为9/24，epoch 20为15/24，
后者碰撞率0.2708，未达到21/24及0.05门槛。随机基线达到18/24，导致“多8局”在24局上
数学不可达；原报告仍按预设门槛失败，不事后放宽。holdout未打开。

第二轮采用 `configs/movement_mvp_stage_c_v2.json`，在训练前固定，旧配置不覆盖。
仍使用既有24个dev场景、128步上限、21/24到达、碰撞≤0.05、折返≤0.10。替换不可达的
“多8局”为：成功数不低于随机/固定方向，且失败计满128步的平均步数≤两者较小值的75%。
所有策略在相同条件下评估；该指标是有限预算工程检查，不宣称统计优越性。

只读定位发现：旧模型在教师轨迹上train为232/232、dev为71/87；偏航后既有错误STOP，也有
边界/死亡导致的强制STOP。原实现还遗漏了连续STOP三步。因此v2隔离导航任务，使用同一个
RichPixelArena，把tower_damage和minion_damage设为0，落实连续STOP三步；原模拟器默认值
不变。旧checkpoint在v2单独以reference-only重评，不能把跨环境的15/24差值解释为训练收益。

数据仍为64/24：16条train轨迹（每方向2条）加入4轮实际位移扰动及短暂停顿，再由teacher
走回目标并连续STOP。扰动动作写入轨迹但不作为学习标签，窗口保留这些动作产生的真实历史；
其余48条为普通teacher轨迹。dev不参与纠错标签。模型、优化器、学习率、batch和20epoch保持
不变，仍只评epoch10/20。新配置是为保留旧冻结配置增加的唯一版本文件，不增加运行入口。
训练前写入含配置/数据/源码hash的training-contract，checkpoint与dev报告绑定它。

本轮最多运行恢复数据候选和一次有证据的修正；合计最多2次新增完整训练，纳入原预算。
若仍达不到门槛，结束本轮学习调整，D继续规则基线的Store/连续episode集成。holdout不打开。
训练label失衡是待检验因素；本次保持均匀窗口采样，不同时修改模型或加入额外loss。

1. 生成既定 64/24 train/dev 轨迹。teacher 优先复用已有导航；若不足，用小网格 BFS 的下一步
   产生八方向标签，禁止斜向穿过阻挡角。只在 teacher 中用地图真值；Actor 输入仍来自 RGB。
   改变起点/目标/可行路线，而不是仅更换背景种子。先 teacher 检查，再允许学习训练。
   从 64 个 train 场景中预留 16 个用于偏离常规路线的起点/过冲恢复，数量不扩张；
   teacher 仍实际走出恢复轨迹。覆盖学生可能到达的状态，而不只重复完美直线示范。
2. 以 episode 保存 uint8 帧一次，训练窗口仅存索引；不把每个 16 帧窗口重复拷贝。
   动作标签是该窗口之后真实执行的 teacher 动作。teacher 失败轨迹隔离，不在训练时静默过滤计数。
3. 使用普通九类交叉熵，先均匀采样窗口并报告九类数量；只在明确 STOP 失衡时把采样调整记作
   一个修正候选。不同时引入 focal loss、多个 auxiliary loss 或新的时序模型。
4. 先训练初始候选，完整 dev rollout 后再决定是否运行修正与对照；六次完整训练是预算上限。
   按 dev 到达数优先、碰撞更少其次、到达步数更少再次选择 checkpoint，不按分类 F1 选。
   选择规则首次训练前写配置，同样用于所有候选；训练早期不反复打开 holdout。
   参考 Hokoff 把评估从训练循环拆开的经验，训练批次内不跑完整对局；
   默认只在 epoch 10 和 20 导出候选，训练完成后独立 evaluate 两者，时间上限提前停止则评 last。
   两个候选复用相同 dev 种子，评估耗时计入 C；不启动常驻 evaluator、模型池或并行 learner。
5. 换目标检查先用 8 对确定性场景：相同原始 RGB 上分别标记两个不同方向的可见目标。
   预期动作不同且朝向各自目标；错误标记也应暴露目标依赖。它只能证明任务条件被使用，
   不能证明真实游戏战术能力。标签打乱、初始化对照只各做一次，不重复全套消融。
6. 正式 dev 评估关闭 teacher 接管；混合模式另跑、另报。偏离轨迹时保留失败片段索引，
   不立即加入自动 DAgger/新增采集系统。若已有候选耗尽则按原预算降级。
   每局返回 success/timeout/runtime_error；三者都进总分母，不跳过失败进程或缺尾轨迹。
   最常见失败归到观察/目标、策略、执行/延迟、记录/恢复四类，下一次修正只针对最多的一类。

最少验证：teacher 真正驱动位置更新、整个场景 split 隔离、目标反事实和规定的 24 局 dev。
普通回归使用少量固定种子；40 场景 holdout 只在 E 的 R1 候选评估一次。

### D：低延迟推理、连续执行与恢复（16 小时）

先用现有 `movement-mvp --mode rule-batch` 接口，依次设置 `--episodes 1`、`3`、`10`完成最小数据面。
它复用固定Stage A规则和同一个Store；`--step-budget`在已提交transition处暂停，`--resume`
通过确定性重放从当前episode的下一step继续，具体命令在README。D0已用step 4中断版和连续版
各跑10局，两者transition及frame-view摘要相同。固定场景重复运行仍不能算模型泛化；学习失败时
不为交付再加训练或替换模型。当前实验结论只以进度账本为准。

1. 一个进程、一个 latest-frame FrameBus、一个同步 Store writer 即可。
   策略只返回 proposal；执行层维护 DOWN/MOVE/UP，连续同向不重复 DOWN。
   推理错过时刻不补发历史动作，记录 skipped cycle；复用现有有效期/停止逻辑，不加新输入门禁。
   分开记录观察采样、策略决策、动作派发三种频率，以及 observation age 和端到端 p95。
   capture/预处理、forward、执行、写盘分别计时，哪个慢才优化哪个；不先加线程。
   OpenAI Five 的 7.5 Hz 来自其引擎和接口，不直接设为本项目默认，更不复制其异步 action offset。
2. 首版直接推理 16 帧窗口并测延迟。只有超过 100/200 ms 目标时，缓存固定 encoder 的每帧
   embedding，用 16 项环形缓冲重跑 GRU。窗口重跑 GRU 时 hidden 每窗清零，
   不把上一窗口 hidden 再带入而重复计算历史。等价性用同一 RGB 序列的 logits 检查一次。
   目标身份/画布规则变更、权重版本变化和 episode reset 都清缓存；不先做 TensorRT/ONNX 导出。
   缓存 key 绑定每帧派生 RGB hash、预处理版本和 checkpoint。目标身份更换才整体重置窗口，
   同一目标在画面中正常移动只生成新帧、淘汰最旧项，不能每帧移动都清空历史。
3. 统一记录 `(s,a,r,s')`、请求/执行来源、动作起止和帧时间，复用 Store 的顺序及事务校验。
   全部 reward 为 0；任务终止先 append 再关闭。录像 replay 用 `executed_action=none` 的
   等价记录标明没有执行，不把预测动作记成设备动作。
4. 权重用 safetensors；恢复状态单独保存本地产物，包含 optimizer、RNG、数据游标和配置绑定，
   只加载本项目生成的可信状态。临时文件写完后原子替换，manifest 最后写作提交标记。
   hash 在加载/保存边界检查一次，不在每个 step 重新计算权重 hash。
5. 区分两类恢复：离线训练恢复（optimizer/RNG/采样游标）与模拟 rollout 恢复
   （环境快照或确定性重放、episode/step、窗口及 Store 游标）。恢复边界只能是已经提交的
   step事务；未提交尾部不能凭猜测续接。
   D0只实现模拟rollout恢复：SQLite transition为真值，从seed重放并校验RGB/动作/终止链；
   step 4恢复后的90条记录已与连续运行一致。训练恢复随本轮训练关闭，不再实现；不承诺未来
   手机物理状态可无缝恢复。

最少验证：同向持续与转向、终局先存、一次跨进程恢复、1→3→10 模拟回合。
10 局可在同一 runner 分段检查，第 1/4/14 局形成三个阶段报告，不为三档再建三条命令。

### E：一个入口交付，不再铺框架（12 小时）

本阶段已按R0收口。`movement-mvp --mode package`将D0中断版和连续版核验后写入一个原子目录，
`--verify-only`在独立进程只读复核。最终包为521,244字节，包含90条transition、10条终局、
100个派生帧和单一manifest，不含checkpoint。当前状态与hash以进度账本为准。

后续`package-cycle`与`package-hierarchical-rule`仍复用同一`movement-mvp`入口：前者冻结零奖励
Event工程链，后者只增加最终模拟器分层规则证据。两者均为不可覆盖目录、支持独立只读验证，
且不把失败checkpoint带入交付。

1. 为新任务只加一个 lazy CLI；当前可运行入口为`movement-mvp`，规则运行、训练诊断、评估、
   恢复与package均通过其mode选择。规则/学习策略由同一配置字段选择，
   不为每次候选新增 CLI/Make target。默认配置完全离线。
2. 保留一份 resolved config、一份 summary.json 和必要 step 记录；摘要自动汇总实际能力、
   pure/hybrid/rule 指标、耗时/峰值显存、预算、checkpoint 来源及失败码，直接供进度账本引用。
   不手抄多份哈希清单、不建立报告签名链；历史产物不删除。
   额外结果只加现有 summary 字段：RGB 规则对照、失败分类、三种频率与观察年龄。
   候选权重和当前应用选择分开记录：学习通过但规则更好时，两者可以不同。
3. C 通过才跑一次 40 场景学习 holdout；本轮C未通过，已经报告R0_ONLY且未消费holdout。
   一条干净命令可加载配置/权重完成离线运行；断网运行不应触发下载或手机接口导入。
4. 完成交付回归与一次全量 `make check`，写清安装环境、命令、恢复方式及应用限制；
   不为文档措辞再重训，不要求第二个代理重复全量验收。

### 后续两包的技术起点（仍不执行）

- 自建 App 接入：现有 transport 原样复用；先验证目标画布在真实 RGB 可生成且无坐标偏转，
  再按授权做有限闭环。离线程序与设备 adapter 之间只转换标准 proposal，不复制整套策略。
  遇到不明画面使用既有停止逻辑；手机中断默认结束本局，下局重新初始化，不恢复悬空指针。
- 有限后训练：可信视觉反馈缺失就不启动。准备就绪后优先给同一 Movement 表征增加一个九动作
  Q head，采用离散 Double-DQN 的单策略小试验，target network 与 reward 模型分别固定更新规则；
  BC checkpoint 留作回滚。先保留 16 帧窗口 replay，不同时引入 recurrent burn-in、优先 replay
  和多个 critic。自然终止不 bootstrap，超时截断在有有效下一状态时允许 bootstrap。
  这只是下一包的技术建议，不是已选择/实现算法，也不从零奖励日志宣称能学到胜负策略。
  BC logits 不是 Q 值：只复用表征，Q head 独立初始化；探索动作与 BC 参考策略显式区分。
  九动作单策略时才保留 Double-DQN 小试验建议；不能因为 hok_env/OpenAI Five 使用 PPO 就换算法，
  也不能把 wzry_ai 的存在当成当前 RGB 条件下训练已经有效的证据。

## 9. 精简检查与日常推进

整轮共用一个 CLI 和一份主配置，各阶段只增加自己的报告段落/运行摘要；复用模块，不新增同义协议和重复入口。
阶段报告写实际能力、失败、证据位置、预算已用/剩余及唯一下一步。不能只报测试数或 hash。
历史文件暂不移动删除，README 和进度只指向本计划；清理旧框架不作为新项目。

检查按风险分层，替代旧“每次提交全量检查＋所有历史 smoke”的要求：

| 改动 | 必做检查 | 不再默认做 |
|---|---|---|
| 仅文档 | diff、链接目标、预算/状态一致性；`git diff --check` | 训练、全量 pytest、mypy、子代理复审 |
| 局部代码 | 受影响的聚焦测试、Ruff；类型接口变化再跑 mypy | 全部历史实验/所有 smoke、重复 reviewer |
| Store/schema/CLI/依赖边界等跨模块变更 | 相关集成回归、strict mypy、项目安全检查 | 无关数据重解码、无关模型重训 |
| 可交付代码冻结（E，或中途实际交付版本） | 一次 `make check`＋`git diff --check`＋所交付能力运行证据 | 在相同代码上再次跑等价检查 |

已有全量结果可绑定代码 tree，后续仅改文档不使代码测试失效；代码再改则交付前重新验证。
本地过程提交采用对应层级，不强迫逐 commit 全量。B/C 的学习门只控制是否训练/晋级，
不妨碍 R0 工程集成。外部授权、test 隔离、因果顺序和恢复正确性不删减。

不默认召开多代理审查；只有证据冲突或跨模块关键问题需要时才用一次聚焦复核，
并计入原阶段预算。审查/文档的计划份额不超过总工程预算 10%（8 小时），
自动测试运行与必要缺陷修复另计原阶段；超出先删重复流程，不降低正确性要求。
正常配置选择和预算内定位问题由开发者继续处理，不逐项询问用户；
仅范围/预算扩大、设备授权或关键目标改变需要用户决定。

完整 12-session QA、全部六次训练、每项对照并不是必做配额；有失败证据可提前停止，
重复 checks 可复用。阶段 E 的 12 小时主要用于 CLI/恢复交付及运行验证，不是 12 小时写报告。
本计划不修改任何现有模型配置或训练门，也不增加输入保护；后续 A 阶段提交最小机器合同变更。

本轮资料搜索到此为止：已知四个工程来源足够支持当前取舍。除实现遇到具体缺口外，不继续
搜排行榜、比较框架或写文献综述；以下新增字段/规则基线纳入原步骤，不单开审查阶段。

C的有界修正结束后进入D的规则数据面、恢复和有限R0打包；不让学习调参无限阻塞离线应用交付。
当前已执行结果和唯一下一步以进度账本为准。

## 10. MOBA 工程来源与本地取舍

检索日期：2026-09-05。以下为原作者仓库/官方技术报告；已读公开说明和相关源码，
未在本机复现这些外部工程。动态 README 只作当日参考，不声称其依赖在本机可用。
表中“采用”是本项目的工程判断，并非来源证明这些改动会提高当前模型性能。

| 来源与核实事实 | 本计划采用 | 不采用/适用差异 |
|---|---|---|
| [Tencent hok_env](https://github.com/tencent-ailab/hok_env#introduction)：SDK/训练框架/PPO；引擎接口返回 observation、legal_action、reward、done，Gamecore 需申请 | A 的环境/策略职责分离，D 的观察/动作绑定 | 不安装 Gamecore/Wine/Docker，不把结构化状态或引擎 reward 输入本 RGB 项目 |
| [Tencent Hokoff](https://github.com/tencent-ailab/hokoff#evaluate)：区分采样、训练、评估；README 说明训练内评估影响效率，因此拆出评估；要求 hok_env/Gamecore | C 在有限 checkpoint 边界独立评估，E 复用 run 目录与汇总 | 不引入模型池、分布式采样或新数据格式；本地无动作录像不等同可直接用的离线 RL 轨迹 |
| [myBoris/wzry_ai 固定 train.py](https://github.com/myBoris/wzry_ai/blob/d5082c41ddea6b0c1d462fb6ef923a53cea29d05/train.py#L44)：step 后 done 分支先 break，随后才有经验存储；采集与训练分线程 | A/D 保留最短 transition 链，并测试 terminal 先写再停 | 不复制旧训练/设备代码，不恢复八头组合动作，不把源码存在当成有效性能证据 |
| [OpenAI Five 官方报告，附录 E/L](https://cdn.openai.com/dota-2.pdf)：使用结构化状态而非屏幕像素；30 Hz 引擎每四帧决策一次，有效 7.5 Hz | D 区分采样/决策/派发，持续动作不等于逐帧推理 | 不复制分布式系统、异步 action offset 或规模；它不证明本项目 RGB-to-action 的域迁移 |

据此优先修正的是：**可观测任务、强规则对照、训练/评估解耦、时间尺度一致、失败计入分母**。
目标标记、20M 上限、32 样本门、24/40 场景数、训练超参数仍是本地预算化设计，
不是这些来源给出的最佳实践数值。今后真实域无改善时，先看数据与任务差异，不再以更大框架替代诊断。
````
