# The single next step

## Status: `DEVICE_PASSED_1_OF_1`, staged continuation pending

The design was implemented offline, pinned by tests, and then run once on device after a real
defect was found and fixed. This document records the implementation, the defect and the run.

**The implemented step:** `run_mobile_navigation_placement_route` runs the declared placement
contract and then the unchanged Route B v15 contract on one guard, one scrcpy session and one
Store, with a start gate between the phases and separate placement/route accounting. It is exposed
as `hok-agent mobile-navigation-placement-route`. Nothing in the route contract, the ROIs, the
layouts, the thresholds, the executor, the Router or the Store schema changed.

## The device run, and the defect it found first

The first attempt dispatched no input at all: `_run_episode` read `persistence_applied` on every
step while the name was first assigned only inside the declared masked-persistence block, so a
contract without that block raised `UnboundLocalError` on its first step. That is a regression
introduced with the persistence work and it would equally have broken the placement and bound-route
contracts. The offline suite missed it because no test ran the step loop; a fake-device step-loop
test now pins both branches.

The second attempt (`placement-route-2`) passed end to end, 1 of 1 composed attempt:

| Field | Value |
|---|---|
| placement | arrived at `(54.58, 69.36)`, `1.08 px` from the declared start `(53.5, 69.4)`, gate `1.5 px` unchanged, 29 transitions, 21.97 s |
| route | all four waypoints under the unchanged contract `ba46e2b2`, `NAVIGATION_GOAL_REACHED`, 50 transitions, 49.90 s, `traversability_masked_steps` 1 |
| session / store | one guard, one session, one Store, 79 transitions across both phase ids |
| denominators | `session_attempts` 1, `placement_successes` 1, `route_started` 1, `route_successes` 1, `end_to_end_successes` 1, `setup_failure` null |
| dispatch | 194 input commands, zero retries, zero unacked dispatches |
| verifier | both episodes recoverable, `store_integrity=ok`, no findings |

**This is one composed run, not a staged admission.** The chain changed, so continuing to 3 needs
the owner's word; re-running 1 -> 3 -> 10 is not implied.

## What is in place

- One session, two phases, one process: both phases share a `_NavigationDevice`, so the guard and
  session are opened once and closed once.
- Start gate: `_placement_start_gate` refuses the route when the placement did not arrive, lost its
  marker, or stopped outside the declared start tolerance. One placement attempt per invocation, so
  a bad start cannot loop.
- Separate denominators: `session_attempts`, `placement_successes`, `route_started`,
  `route_successes`, `end_to_end_successes`, plus placement and route wall times.
- Status is `PASSED`, `SETUP_FAILED` (the route never ran, so it is scored neither way) or
  `FAILED` (the route ran and did not arrive), with `setup_failure` naming the gate reason.
- Phase identity from the episode id (`…-placement` / `…-route`), with no Store schema change.
- Offline coverage: eleven tests, `tests/test_mobile_navigation_store.py` - the composed-run cases
  and two fake-device step-loop regression tests. They prove scheduling, the refusal (including a
  lost marker and a placement that failed), the accounting and the single release only.

## The command that was run

```text
hok-agent mobile-navigation-placement-route \
  --serial <authorized serial> \
  --placement-config configs/movement_goal_navigation_reposition_v4.json \
  --config configs/movement_goal_navigation_route_b_v15.json \
  --visual-layout ../hok-agent-v5/configs/mobile_testbed_layout_calibrated_v3.json \
  --execution-layout ../hok-agent-v5/configs/mobile_testbed_layout_all_actions_corrected.local.json \
  --observation-rois ../hok-agent-v5/configs/mobile_observation_rois.local.json \
  --output-dir $HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/placement-route-2 \
  --enable-input
```

Always reported: total session attempts, placement successes, route starts, route arrivals,
end-to-end successes, and the placement and route times separately. A refused start is never folded
into a route success rate.

## What remains

- `placement-route-1` is void (0 input, 0 steps) and `placement-route-2` is 1 of 1. The chain
  changed, so the staged rule would continue to 3 next, and that needs the owner's word.
- The composed run is still read-only with respect to learning: `training_eligible` stays `false`
  and a placement start is not an independent reference.

## Milestones, relabelled

Reachable now:

- **F0 baseline binding** - done; skippable.
- **F1 same-session lifecycle** - implemented offline and passed one device run
  (`placement-route-2`, 1 of 1). This is the only engineering increment that was reachable under
  the current boundaries.

**Blocked on scope, not queued work:**

- **F2 one learning feedback** - blocked: no independent result reference exists
  (`independent_map_reference_available = false`).
- **F3 minimal trainable candidate** - blocked: F2.
- **F4 small post-training comparison** - blocked: F3.
- **F5 full automatic match** - blocked: F4, and it also needs an authorized action range and a
  scenario that actually supports departure, combat, death/recovery and a terminal.

## The ceiling, stated plainly

After F1 the system can do nothing it could not do before. It moves from *unmeasurable* to
*measurable*. The passed 14 of 14 is currently self-referential - the contract judges itself by
the gate it declares - and under no-source there is no independent reference to check it against.
So this step improves how reliably the passed baseline can be used and reported, and it does not
move the project closer to playing the game. That needs a change of data-source boundary, which is
the owner's decision and is not requested here.

## Closing

```text
已复用：
新确认的工程缺口：
仍缺的数据/权限：
下一件事：
当前是否可执行：
本次没有执行：设备输入 / 新采集 / 训练 / push
```
