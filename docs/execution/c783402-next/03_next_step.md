# The single next step

## Status: `IMPLEMENTED_OFFLINE`, device acceptance pending

The design was implemented offline and its lifecycle is pinned by tests. What remains is a device
run, which needs its own authorization. This document records both.

**The implemented step:** `run_mobile_navigation_placement_route` runs the declared placement
contract and then the unchanged Route B v15 contract on one guard, one scrcpy session and one
Store, with a start gate between the phases and separate placement/route accounting. It is exposed
as `hok-agent mobile-navigation-placement-route`. Nothing in the route contract, the ROIs, the
layouts, the thresholds, the executor, the Router or the Store schema changed.

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
- Offline coverage: nine tests, `tests/test_mobile_navigation_store.py`, using fake runtimes and
  counter monkeypatches; they prove scheduling, the refusal (including a lost marker and a
  placement that failed), the accounting and the single release only.

## The device acceptance, when it is authorized

One run first, under the unchanged staged rule (1, then 3 only if the chain changed to justify a
rebind, not for a documentation update). The placement phase uses the same 1.5 px contract that
produced the bound pass:

```text
hok-agent mobile-navigation-placement-route \
  --serial <authorized serial> \
  --placement-config configs/movement_goal_navigation_reposition_v4.json \
  --config configs/movement_goal_navigation_route_b_v15.json \
  --visual-layout ../hok-agent-v5/configs/mobile_testbed_layout_calibrated_v3.json \
  --execution-layout ../hok-agent-v5/configs/mobile_testbed_layout_all_actions_corrected.local.json \
  --observation-rois ../hok-agent-v5/configs/mobile_observation_rois.local.json \
  --output-dir $HOK_LARGE_ROOT/runs/hierarchical-movement-mvp/placement-route-1 \
  --enable-input
```

Reported, always: total session attempts, placement successes, route starts, route arrivals,
end-to-end successes, and the placement and route times separately. A refused start must not be
folded into a route success rate.

## Milestones, relabelled

Reachable now:

- **F0 baseline binding** - done by this pass; skippable.
- **F1 same-session lifecycle** - the design above. This is the only engineering increment that is
  reachable under the current boundaries.

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
