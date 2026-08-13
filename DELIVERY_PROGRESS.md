# Delivery Progress

- Last update: 2026-08-13
- Current task: `V5-REAL-DATA-PRE_INGEST`
- Status: `CANDIDATE_INVENTORY_COMPLETE_STORAGE_ROUTED_WAITING_E_RW`
- Product Actor input: `RGB_ONLY`
- Closed-loop environment: `PIXELARENA_ONLY`
- Commercial-client action output: `false`
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
| V5 Visual Alignment | `PATH_BOUND_NON_PROMOTING_FRAMEWORK_IMPLEMENTED` | source producer/thresholds + 12 sessions + 300-clip sealed audit | `DATA_CANDIDATES_FOUND_WAITING_SOURCE_THRESHOLDS_AND_CURATION` |
| V6 Temporal Coach | `PATH_BOUND_FAIL_CLOSED_FRAMEWORK_IMPLEMENTED` | valid V5 release + V6 checkpoint + 300 tracking frames + 200-clip audit | `WAITING_V5_AND_DATA` |
| V7 Rich PixelArena V2 | `COMPLETE` | three-seed CUDA classification/closed-loop/replay/latency | `FORMAL_PASSED` |

The missing capture node and labels do not block contracts, simulated-source tests, CPU
smokes, annotation tooling, or Rich PixelArena. Read-only raw recording candidates now exist,
but privacy/session/overlap curation is incomplete. These gaps still block any live throughput,
real-domain precision, class release, tracking accuracy, or commercial-video advice claim.

## Frozen acceptance gates

V4 live: explicit non-symlink `/dev/videoN`, latest-frame-only bounded capture, 1080p60 input,
10 Hz inference, at least 99% scheduled cycles, end-to-end p95 at most 100 ms, zero raw frame
persistence and zero control output.

V5: source teacher has no tick/modulo shortcut; real training has zero human action labels;
session splits have no overlap; SimSiam avoids collapse; pseudo labels meet all frozen
agreement/OOD filters; one Mean Teacher round only. Advice release additionally needs two
blind annotators, overall accepted precision at least 85%, each released class at least 75%,
sealed coverage at least 30%, OOD false accept at most 5%, Cohen kappa at least 0.70, at
least five-point improvement over frozen baselines at matched coverage, and bounded source
regression. Audit labels never feed training or threshold selection.

V6: sealed tracking PCK@0.1 at least 85%, visibility F1 at least 90%, HP MAE at most 0.10,
skill-ready F1 at least 90%; temporal advice overall precision at least 85%, unlocked-class
precision at least 75%, coverage at least 20%, transition/OOD false advice at most 5%,
switch-rate reduction at least 50%, added-delay median/p95 at most 300/500 ms, and live chain
10 Hz with p95 at most 100 ms.

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
  Raw files were not changed, copied into Git, or persisted by path. This is not a formal V5
  manifest: partial overlap/re-encode grouping, privacy masking, independent-session proof,
  and source-rights documentation remain unresolved.
- New large training storage is routed through `HOK_LARGE_ROOT`, defaulting to
  `/media/hgdl1012/E/wzry-data/hok-agent-v5`, with dedicated datasets/checkpoints/runs/cache/
  audit/staging directories and a fail-closed read-write mount preflight. At configuration
  time `/dev/sda2` was actually mounted `fuseblk ro` despite an `rw` fstab entry, and a write
  probe failed with `Read-only file system`. No directory was created and no frozen run was
  moved. `make storage-show` resolved every large-output root under E; `make -n storage-init
  accept-v3 accept-v7` preserved that routing; `make storage-preflight` failed as designed;
  and the post-change `make check` passed Ruff, strict mypy, 83 tests, and the 40-file /
  31-Python-file / 8,999-line size gate. `make accept pixel-smoke shadow-live-smoke
  alignment-smoke temporal-smoke rich-smoke` also passed; these were CPU/fail-closed
  regressions and did not create GPU or hardware evidence. Run `make storage-init` only after
  the host remounts E read-write.
- 2026-08-13 pre-data closure replaced bare in-memory V5 promotion inputs with strict
  regular-file manifest/source/target/pseudo/model/ledger/audit loading, persisted the single
  Mean Teacher EMA model, and fixed formal V5 release creation and loading closed while
  collapse thresholds remain unspecified. V6 now binds actual artifact paths, exact model
  metadata, session-isolated 180/60/60 tracking evidence, and the same two blinded reviewers
  across exactly 200 temporal clips; absent evidence remains `ABSTAIN` without model forward.
- Pre-data closure checks observed: Ruff passed; strict mypy passed for 18 source files;
  full pytest `83 passed`; project gate passed with 40 files, 31 Python files, 8,999 nonblank
  Python lines, and four root Markdown files. V1 and V2 acceptance passed; V3, V5, V6, and V7
  CPU smokes passed; V4 capture tests passed `7/7`. No hardware, real recording, label, CUDA
  formal training, or real-domain capability evidence was created in this closure.

## Remaining blockers and next task

1. `V4-HARDWARE-ACCEPTANCE`: connect a read-only UVC capture card and run the frozen
   10-minute gate; no phone control connection is needed or allowed.
2. `V5-SOURCE-AND-DATA`: the strict path chain and final EMA persistence are implemented.
   A compact independent causal-source producer still needs 165–205 lines, but the frozen
   9,000-line gate has only single-digit capacity; first replace/compact existing code rather
   than relaxing the gate. Then freeze numeric collapse thresholds, curate at least 12
   independent connected components from the audited candidates, create privacy-masked
   shards, and supply the 300-clip audit. Formal release writing/loading remains disabled
   until all are present.
3. `V6-TRAIN-AND-AUDIT`: checkpoint/release loaders and recomputed raw evidence gates are
   implemented. Train/save the real checkpoint, then provide session-isolated 180/60/60
   tracking evidence and the two-reviewer 200-clip temporal audit. Current advice remains
   all-`ABSTAIN` because no valid V5 release can exist yet.
4. V7 has no remaining frozen acceptance blocker. Further RichArena changes require a new
   versioned ruleset, new data, and a fresh acceptance run rather than changing this result.

Current maximum claim: the project-owned Rich PixelArena V2 RGB-only factorized ResNet-18
behavior-cloning agents passed the frozen three-seed RTX 4090 classification, negative-control,
closed-loop, replay/tamper, side-symmetry, illegal-action, and latency gates. V4 live hardware,
V5 real-domain alignment, and V6 released temporal advice remain externally blocked or
fail-closed. This establishes no Honor of Kings, GameCore, commercial-client automation, or
out-of-environment capability.
