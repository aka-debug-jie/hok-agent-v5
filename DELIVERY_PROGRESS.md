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
The next T8 descendant would require a new label/state contract; it must not be presented as a
continuation of the failed v2.7 or v3 pilots by threshold relaxation.
