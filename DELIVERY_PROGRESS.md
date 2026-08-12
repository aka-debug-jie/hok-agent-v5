# Delivery Progress

- Last update: 2026-08-12
- Current task: `V4-LIVE-SHADOW`
- Status: `IN_PROGRESS`
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
| V4 live Shadow | `IN_PROGRESS` | 10-minute 1080p60 UVC / 10 Hz run | `READY_FOR_HARDWARE` after code smoke |
| V5 Visual Alignment | `NOT_IMPLEMENTED` | 12 sessions + 300-clip sealed audit | `WAITING_DATA` after code smoke |
| V6 Temporal Coach | `NOT_IMPLEMENTED` | 300 tracking frames + 200-clip audit | `WAITING_DATA` after code smoke |
| V7 Rich PixelArena V2 | `NOT_IMPLEMENTED` | CPU/CUDA gates; no phone required | `READY_TO_IMPLEMENT` |

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

## Current work log

- Authority reset to the compact V4–V7 route: `IN_PROGRESS`; commit not yet recorded.
- No `/dev/video*` was visible at planning time; this is not evidence about future host
  capture-card availability.
- No real recording manifest, sealed action audit, or tracking-label set has been supplied.
- No V5/V6 real-domain accuracy or advice class is currently released.
- No V7 capability claim is allowed until its independent acceptance gates actually run.
