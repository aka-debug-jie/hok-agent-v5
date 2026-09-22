# The single next step

## Status: `FROZEN` at `5c17846` - delivered, 4 of 4, not a staged admission

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

The second attempt (`placement-route-2`) passed end to end, and three further authorized runs
(`-3`, `-4`, `-5`) passed too, so the composed chain stands at **4 of 4 passed attempts**:

| Run | Start gate | Placement | Route | Verifier |
|---|---|---|---|---|
| `-2` | `PLACEMENT_CONFIRMED` 1.08 px | 29 steps, 21.97 s, 87 cmds | 4 wp, 50 steps, 49.90 s, mask 1 | 2 episodes recoverable, ok |
| `-3` | `PLACEMENT_CONFIRMED` 0.32 px | 1 step, 0.8 s, 3 cmds | 4 wp, 59 steps, 55.4 s, mask 1 | 2 episodes recoverable, ok |
| `-4` | `PLACEMENT_CONFIRMED` 0.60 px | 1 step, 0.8 s, 3 cmds | 4 wp, 52 steps, 49.9 s, mask 2 | 2 episodes recoverable, ok |
| `-5` | `PLACEMENT_CONFIRMED` 0.90 px | 1 step, 0.8 s, 3 cmds | 4 wp, 62 steps, 57.7 s, mask 2 | 2 episodes recoverable, ok |

Every run: `session_attempts` 1, `placement_successes` 1, `route_started` 1, `route_successes` 1,
`end_to_end_successes` 1, `setup_failure` null, zero retries, zero unacked dispatches, route
contract `ba46e2b2` unchanged, `NAVIGATION_GOAL_REACHED`.

**What the 4 of 4 does and does not show.** It shows the composition and its lifecycle repeat: the
device is opened once and closed once per run, the gate confirms, both phases land in one Store and
the reload verifier recovers every episode. It does **not** show start-position independence: runs
`-3` to `-5` began with the hero already inside the placement tolerance, so their placement was one
step; only `-2` began with a real placement walk. And it is **not a staged admission** for the route
contract, which keeps its own 14 of 14 at its own commit.

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

- The owner froze the delivered scope on 2026-09-22. No further device work is scheduled and no new
  rule is to be built; the composed chain stands at 4 of 4 passed runs (`placement-route-2` to `-5`),
  with `placement-route-1` void at 0 input and 0 steps. The repeatability is of the composition, not
  of an independent start, and it is not a staged admission for the route contract.
- The composed run is still read-only with respect to learning: `training_eligible` stays `false`
  and a placement start is not an independent reference.
- If work resumes, the next move is the owner's: fund a different kind of source (a corridor measured
  over its full length, or an acceptance signal that can be confirmed) or extend the boundary so an
  independent result reference can exist. Neither is requested here.

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
