# Reuse gap: same-session placement then route

The question is only whether the already-passed Route B v15 can be turned into a reusable fixed
task whose start is placed in the same session, and what is missing to do that by reusing what
already exists. It is not a claim that the composed chain has succeeded on hardware.

Touchpoints inspected, and nothing else: `mobile_navigation_store.py`, `mobile_testbed.py`,
`transition_store.py`, `cli.py` and the `Makefile`, and the focused navigation-store tests.

## The five items

| # | Item | Status | Code location | The minimal test that would fail if this layer were silently unwired |
|---|---|---|---|---|
| 1 | Placement and route run in one session, with no unrecorded manual movement between them | **to implement** | `run_mobile_navigation_episode` `mobile_navigation_store.py:2155`; `run_mobile_navigation_episodes` `:2195`; guard opened per runtime at `_prepare_navigation_runtime` `:1136`; `_open_runtime` `:2121`; `_close_runtime` `:2128` | A fake-runtime test that composes both phases and asserts the device guard/session is entered exactly once and that the route phase's first localised position is the placement phase's last. Today the only batch tests are the bad-count guard (`tests/test_mobile_navigation_store.py:341`) and the episode-id ordering (`:354`); neither would fail if a second guard were opened |
| 2 | The start is re-checked before entering the route, a bad start refuses rather than retrying forever | **to implement** | the step loop begins at `waypoint_index = 0` `:1366` and drives at `runtime.targets[0]`; arrival is set only when every target is done `:1951`-`:1954`; `_episode_outcome` `:1027` has no setup-failure class | A test that feeds a start sample outside a declared start tolerance and asserts the route does not start, that a distinct setup outcome is reported, and that at most the declared number of placement attempts is issued. No such outcome class or test exists |
| 3 | Placement is not counted as route success | **to implement** (the arrival semantics already exclude it; the reporting does not) | `arrived` requires all targets `:1953`; `arrivals` / `arrival_rate` are computed over route episodes at `:2280`-`:2281`; no placement fields exist | A test that composes placement and route and asserts four separate denominators (`session_attempts`, `placement_successes`, `route_started`, `route_successes`) and that a placement-only arrival leaves route successes unchanged. No such fields or test exist |
| 4 | Pointer release and an explainable Store on every exit | **implemented in code, unproven by test** | `finally: _close_runtime(runtime, …)` at `:2182`-`:2183` (single) and `:2261`-`:2262` (batch); `_open_runtime` `:2121`; the joystick is adopted rather than assumed released at `:1371` | An abort/exception test asserting the joystick is released and the Store is left explainable. Not found: the lifecycle tests cover only the bad-count guard and the id ordering |
| 5 | The start check is an RGB pre-check and does not claim independent truth | **implemented** | position comes from `_goal_navigation_tracked_cue` `:1465`, the frozen green-ring cue on the minimap view; unknown position routes to `wait` / `unknown_position` `:483`; the R0 contract marks position `rgb_derived` with `independent_map_reference_available = false` | A displaced-cue frame yields a setup failure rather than a pass. Also missing, but the code path is already single-source |

## The one-scalar constraint that shapes the design

`arrival_tolerance_pixels` is read once, at `:1148`, and applied to every target. Route B v15
declares 4.0 px and placement v4 declares 1.5 px. A single composed contract therefore cannot
carry both, so the minimal design is a **session-level wrapper around two contracts** (one guard
and one session, two phases), not a five-target contract.

## What is already reusable (do not rebuild)

- the runtime assembly, the guard and the scrcpy session (`_prepare_navigation_runtime`,
  `_open_runtime` / `_close_runtime`);
- the single `UnifiedTransitionStore` and its reload verifier;
- the RGB cue localiser and the per-step arrival evaluation;
- the existing executor, Router, joystick and action vocabulary;
- the existing CLI entries (`mobile-navigation-store`, `mobile-navigation-store-batch`). No new
  CLI alias, no second FrameBus, no second scheduler.

## What this table does not claim

- It does not claim the composed chain works on device; items 1-3 have no implementation yet.
- It does not claim placement solved cold start. The corridor limit and the cold-start point are
  unchanged and stay closed.
- It does not claim any start-position robustness. Reproducing the pass still depends on the
  declared 1.5 px placement.
