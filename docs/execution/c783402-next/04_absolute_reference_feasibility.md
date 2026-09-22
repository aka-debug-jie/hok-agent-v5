# Absolute-reference feasibility check (A1)

Status: `NOT_FEASIBLE`, read-only. The owner authorized a cue-independent absolute position
reference derived from the same RGB on 2026-09-22. This document records the read-only check that
was run before any contract or code was written, and its negative verdict.

## What was checked, and on what

Scope: the persisted derived views of one composed run, `placement-route-2`, which wrote four
128x128 views per step (`main`, `minimap`, `hud`, `equipment`); 30 placement steps and 51 route
steps. Only derived views exist - no raw frames - so this is a statement about the persisted views,
not about the full-resolution screen.

Method: pure read-only analysis (temporal statistics, connected green components, near-white
structure search, phase correlation and a brute-force integer shift search over the `main` view).
No device, no capture, no training, no threshold or contract change.

## Finding 1: the minimap is a fixed full-map view, so it gives a frame, not a position

Per-pixel temporal standard deviation over the placement episode:

| View | mean std | fraction < 2 | fraction < 5 | fraction > 15 |
|---|---|---|---|---|
| `minimap` | 3.36 | 0.692 | 0.884 | 0.058 |
| `main` | 15.80 | 0.040 | 0.048 | 0.501 |

The minimap background is static, so the map is drawn whole and fixed rather than scrolling with
the hero. Consecutive-frame mean absolute difference is 2.10 (median) on the minimap against 12.69
on `main`. A fixed map supplies a stable coordinate frame but carries no information about where
the hero is: the background does not move, so the position can only come from a marker.

## Finding 2: there is no second, structural hero indicator on the minimap

- Near-white pixels number only 3-7 per frame and all sit inside the moving green blob
  (step 0 at rows 58-63 / cols 56-64, step 10 at rows 52-58 / cols 62-72), with no row or column
  peaks. There is no viewport or camera rectangle.
- Red-dominant pixels number 636-752 per frame (median 729), because terrain is red. Red is not a
  clean marker channel.
- Green-dominant content is not unique. The largest components in the placement episode are a large
  bottom-right green region (size 69-151, spanning up to 44x38), a compact moving blob of roughly
  18x19 (the hero ring), and a fixed green UI element at `(121, 89)` of size 9 in every frame.
  This matches the earlier finding that most frames hold two or more green blobs and the single
  candidate tracker stays fragile.

So the only source of hero position is a hero-marker detector, and any such detector reads the same
pixels as the frozen green-ring cue.

## Finding 3: the camera view cannot supply an independent displacement either

If the world is fixed and the camera follows the hero, frame-to-frame translation of `main` would be
an independent displacement observation. It is not recoverable at this resolution:

- Phase correlation returns `(0, 0)` with an unstable peak sign, and a brute-force integer search
  over +/-12 px returns the best shift `(0, 0)` on every one of the sampled steps with correlation
  only 0.47-0.79, while the cue's own step deltas are of order 1-2 px.
- Half the `main` pixels change by more than 15 grey levels every frame, so the view is dominated by
  animation, effects and HUD rather than a rigidly translating scene.

## Verdict

`NOT_FEASIBLE` as a position or displacement reference. The fixed minimap can anchor a coordinate
frame but not a hero position; the only position source is a marker detector on the same pixels as
the frozen cue, which the R0 contract already forbids ("the same unvalidated detector must not both
author a reward and prove its own accuracy"); and the camera view carries no recoverable rigid
translation.

**Nothing was built on this.** Per the next-step pack, a source that cannot be obtained is recorded
once and is not turned into a stub, a fake-filled adapter or a second research line.

## The one honest thing the fixed map does give

Because the minimap background is static and fixed, it supports a **cue-independent registration
check**: verify that the minimap crop, scale and offset did not shift between sessions, by
registering each session's background against a reference background. That is a data-quality guard
for cross-session comparability. It is explicitly **not** a position or reward reference, and it does
not open F2 on its own.

## What this leaves open

- A2: a genuinely different source that the owner names and confirms is obtainable (for example an
  owner-side test export, or new capture with known truth). Only then is a minimal interface
  designed.
- B: the in-boundary F2 task - behavior preservation plus measured latency on the frozen composed
  task - which needs no external result reference at all.
