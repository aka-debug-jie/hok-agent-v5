# Reuse gap: same-session placement then route

The question is only whether the already-passed Route B v15 can be turned into a reusable fixed
task whose start is placed in the same session, and what is missing to do that by reusing what
already exists. It is not a claim that the composed chain has succeeded on hardware.

Touchpoints inspected, and nothing else: `mobile_navigation_store.py`, `mobile_testbed.py`,
`transition_store.py`, `cli.py` and the `Makefile`, and the focused navigation-store tests.

## The five items

| # | Item | Status | Code location | The minimal test that would fail if this layer were silently unwired |
|---|---|---|---|---|
| 1 | Placement and route run in one session, with no unrecorded manual movement between them | **implemented (offline)** | `_NavigationDevice` `mobile_navigation_store.py:1072`; `device=` reuse in `_prepare_navigation_runtime` `:1135`; `run_mobile_navigation_placement_route` `:2403` | `test_placement_route_opens_one_device_and_starts_the_route` asserts the second phase receives the first phase's `_NavigationDevice` and that the run opens once and closes once. It would fail if a second guard were opened |
| 2 | The start is re-checked before entering the route, a bad start refuses rather than retrying forever | **implemented (offline)** | `_placement_start_gate` `:2330`; one placement attempt per invocation by construction | `test_placement_start_gate_refuses_an_unarrived_placement`, `_refuses_a_start_outside_the_declared_tolerance`, `_refuses_a_missing_position`, and `test_placement_route_refuses_and_never_starts_the_route_on_a_bad_start`, which asserts the route phase is never entered |
| 3 | Placement is not counted as route success | **implemented (offline)** | `_placement_route_summary` `:2374` | `test_placement_route_summary_keeps_placement_out_of_route_success`; a placement-only arrival leaves `route_started`, `route_successes` and `end_to_end_successes` at zero |
| 4 | Pointer release and an explainable Store on every exit | **implemented and tested** | `finally: _close_runtime(runtime, …)` `:2210` (single), `:2289` (batch), and the composed run's single close `:2501` | the composed-run test asserts exactly one close even when the route phase is skipped |
| 5 | The start check is an RGB pre-check and does not claim independent truth | **implemented** | position from `_goal_navigation_tracked_cue` `:1492`; the gate reads only the placement episode's recorded `final_position`; unknown position routes to `wait` / `unknown_position` `:484`; the R0 contract marks position `rgb_derived` with `independent_map_reference_available = false` | `test_placement_start_gate_refuses_a_missing_position` pins that a lost marker refuses rather than passes |

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

## What was added, and what it deliberately did not touch

Added, all offline:

- `_NavigationDevice` and the `device=` reuse path, so both phases share one guard, session,
  joystick and watchdog;
- `_placement_start_gate`, a pure pre-action check over the placement episode's recorded final
  position, the declared start and the declared tolerance;
- `_placement_route_summary`, the separated denominators;
- `run_mobile_navigation_placement_route`, the composed run, plus the thin CLI entry
  `mobile-navigation-placement-route`.

Deliberately not touched: the route contracts and their 4.0 px gate, the ROIs, the layouts, the
thresholds, the executor, the Router and the Store schema. `training_eligible` stays `false`.
A refused start is a run-level status, not a new Store terminal reason, so no schema change was
needed: the composed run reports `status` as `PASSED`, `SETUP_FAILED` or `FAILED`, and
`setup_failure` names the gate reason. `SETUP_FAILED` means the route never ran, so it is scored
neither as a route success nor as a route failure. Phase identity is carried by the episode id
(`…-placement` / `…-route`), which is recoverable from the Store without a new column.

## What this table does not claim

- It does not claim a staged admission or start-position independence. After the void first attempt,
  four composed runs (`placement-route-2` to `-5`) passed end to end, but only `-2` began with a
  real placement walk; the later runs already stood inside the placement tolerance. The repeatability
  is of the composition and its lifecycle.
- It does not claim placement solved cold start. The corridor limit and the cold-start point are
  unchanged and stay closed.
- It does not claim any start-position robustness. Reproducing the pass still depends on the
  declared 1.5 px placement, and the one composed run landed at 1.08 px.

## The device run, and the defect it found first

The first attempt dispatched no input: `_run_episode` read `persistence_applied` on every step while
the name was first assigned only inside the declared masked-persistence block, so a contract without
that block raised `UnboundLocalError` on its first step. That regression would equally have broken
the placement and the bound-route contracts, and the offline suite missed it because no test ran the
step loop. It is fixed, and a fake-device step-loop test now pins both branches. The composed run
then passed 1 of 1 on the second attempt.
