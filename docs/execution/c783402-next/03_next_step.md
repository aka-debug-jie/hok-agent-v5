# The single next step

## Status: `READY_OFFLINE_DESIGN`

The existing interfaces are reusable, so the next step is a design, not a new runner and not a
new contract. This document stops at the design points; implementing them needs a separate
decision.

**The next step:** a session-level wrapper that runs the declared placement contract and then the
unchanged Route B v15 contract on one guard, one scrcpy session and one Store, with a start gate
between the phases and separate placement/route accounting.

## Design points (reuse only)

1. **One session, two phases, one process.** Assemble the runtime once, open the guard and session
   once, run the placement phase, then the route phase, then release once. The whole point is that
   the placement and the route share the session instead of being two invocations.
2. **A start gate between the phases.** Compare the placement phase's last localised position
   against a declared start tolerance. On failure, emit a distinct setup outcome and stop without
   starting the route; bound the placement attempts so a bad start cannot loop.
3. **Separate denominators.** Report `session_attempts`, `placement_successes`, `route_started`,
   `route_successes`, `end_to_end_successes`, and the placement and route wall times separately. A
   placement arrival must not increment route successes.
4. **Two contracts, one session.** Because `arrival_tolerance_pixels` is a single scalar per
   contract, keep placement (1.5 px) and route (4.0 px) as two declared contracts rather than one
   five-target contract.
5. **Phase identity in the record.** Each transition row should carry which phase it belongs to,
   so a placement step is never read as a route step.

## Offline test cases to write first

Verified against fake runtimes and already-allowed fixtures; no phone. These prove software logic
only, not cold start and not game ability.

| Scenario | Expected |
|---|---|
| Placement reaches the declared start and identity/screen are valid | one session created; the route starts only after placement ends |
| Placement does not meet the start condition | the route does not start; report a setup failure, not a route success or failure |
| The guard is invalid or the observation is stale between phases | release the pointer and stop; do not inherit the previous command |
| The route reaches all four waypoints | record task completion, then exit; do not emit the arrival reward twice |
| Timeout, capture failure or action failure | keep distinct exit semantics; never masquerade as a win or a loss |
| A second run | state explicitly whether it re-places; do not treat the previous end point as an already-valid start |

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
