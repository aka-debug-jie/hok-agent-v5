# Delivery Progress

- Last update: 2026-08-12
- Current task: `COMPACT-V4-V7-IMPLEMENTATION`
- Status: `PARTIAL_DELIVERY`
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
| V5 Visual Alignment | `NON_PROMOTING_FRAMEWORK_IMPLEMENTED` | 12 sessions + path-bound gate + 300-clip sealed audit | `WAITING_DATA_AND_RELEASE_GATE` |
| V6 Temporal Coach | `FAIL_CLOSED_FRAMEWORK_IMPLEMENTED` | V6 checkpoint + 300 tracking frames + 200-clip audit | `WAITING_DATA_AND_RELEASE_BINDING` |
| V7 Rich PixelArena V2 | `CORE_RENDERER_TRAINER_IMPLEMENTED` | three-seed CUDA classification/closed-loop/latency | `FORMAL_NOT_PASSED` |

The missing capture node, recordings, and labels do not block contracts, simulated-source
tests, CPU smokes, annotation tooling, or Rich PixelArena. They do block any live throughput,
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
- V7 implementation, shared CLI/CI gates, and this ledger are the current Git `HEAD`.
- `make check`: Ruff passed; strict mypy passed for 18 source files; `84 passed`; project
  gate passed with 40 files, 31 Python files, 8,217 nonblank Python source lines, and four
  root Markdown files.
- Frozen regressions: V1 acceptance passed; V2 three-seed structured BC passed; V3 CPU
  smoke passed. V4 capture tests passed `7/7`; V5 contract smoke passed with no release;
  V6 smoke emitted only `ABSTAIN`; V7 CPU smoke passed with five factor heads.
- V7 trajectory-only preflight (no frame materialization/training): 512/512 unique complete
  public trajectories, 466/512 teacher crystal completions (`91.015625%`), and minimum
  per-split transition count for every syntax template of fit/acquisition/validation/test
  = `114/43/22/28`; two render variants would double these sample counts.
- First authorized host RTX 4090 V7 formal attempt reached seed-0 validation but failed with
  `CUDNN_STATUS_EXECUTION_FAILED` because validation was sent as one unbounded batch. The
  code now validates in fixed 256-frame batches.
- The single post-fix retry reached seed-0 sealed classification and failed its frozen
  threshold. No run directory or PASSED report was retained. Review then found and fixed
  red-side 180-degree self-view labels still using world directions; current ego-direction
  code has passed static/unit/CPU smoke gates but has not received another CUDA formal run.
- The original project shell exposed a conflicting CUDA library path. Direct pytest
  collection failed on `libcusparse`/`nvJitLink`; all Make targets now run with
  `LD_LIBRARY_PATH` unset and the pinned project Torch 2.5.1 environment passed.
- No `/dev/video*` was visible at planning time; this is not evidence about future host
  capture-card availability. No live V4 run was attempted.
- No real recording manifest, sealed action audit, or tracking-label set has been supplied.
  No V5/V6 real-domain accuracy or advice class is released.

## Remaining blockers and next task

1. `V4-HARDWARE-ACCEPTANCE`: connect a read-only UVC capture card and run the frozen
   10-minute gate; no phone control connection is needed or allowed.
2. `V5-ARTIFACT-PATH-GATE`: replace framework-only in-memory promotion inputs with strict
   path-loaded session/source/pseudo/model/audit artifacts, persist the final EMA model, and
   only then ingest the 12-session recording set. Until then formal release writing stays
   disabled.
3. `V6-RELEASE-BINDING`: train/save a V6 checkpoint and bind the 300-frame tracking and
   200-clip temporal audits; current advice intentionally remains all-`ABSTAIN`.
4. `V7-CUDA-RERUN-AFTER-EGO-DIRECTION-FIX`: perform a new three-seed formal run; a capability
   claim is forbidden until all classification, closed-loop, replay, control, and latency
   gates pass and a five-file run directory exists.

Current maximum claim: the repository contains runnable, tested V4–V7 implementation
frameworks and project-owned Rich PixelArena rules/rendering/data/training code. It does not
yet establish live-capture performance, real-domain advice accuracy, a released temporal
coach, or a passed Rich PixelArena learned policy. It establishes no Honor of Kings,
GameCore, commercial-client automation, or out-of-environment capability.
