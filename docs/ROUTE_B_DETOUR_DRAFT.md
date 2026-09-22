# Route B declared detour planning - draft protocol

**Status: DRAFT FOR REVIEW. Nothing here is implemented, no contract is versioned, and no device
session is authorized by this document. It exists so the owner can review the semantics before any
code is written.** It also does not change the recorded claim: the current scope limits still stand
and the cold-start limit stays named as measured.

## Why this is the next capability

The recorded cold-start failures end with the hero stalled about ten pixels from waypoint `(50, 50)`
needing to travel north, and the closest stall sits in cell `14:12`, where the frozen v15 grid
removes north at `0.080` px per 100 ms over 36 bounded samples while north-east reads `0.183` in the
same cell. The route fails on the mask's own decision.

The mask can only remove a bearing. The only better bearing at `14:12` is north-east, which moves
away from the goal, and the greedy objective will not take a detour such as north-east then north.
Two declared attempts to close this with commitment and hysteresis both failed for recorded reasons
(v16's stall commitment never engaged because the stall is inside the approach band, and v17's band
hysteresis engaged while the hero genuinely cannot go north), so the gap is a missing capability, not
a mis-tuned parameter.

## Definitions

- **Obstructed `(cell, bearing)`.** The frozen grid holds at least `minimum_samples` for that cell and
  bearing, and its normalised rate is below `minimum_rate_per_100ms`. This is exactly the predicate
  the mask already uses; the detour must not invent a second one.
- **Unknown `(cell, bearing)`.** The grid does not hold enough samples. Existing mask semantics are
  fail-open, so the detour treats unknown as usable but records how many of its steps relied on
  unknown cells, so a detour that only works by travelling through unmeasured space is visible rather
  than hidden.
- **Detour.** A bounded, declared, deterministic sequence of bearings that starts by not decreasing
  the goal distance, is chosen to route around obstructed `(cell, bearing)` pairs, and ends when the
  goal-directed bearing is usable again. It is a plan, not a search at execution time.
- **Goal-directed bearing.** The bearing the existing `geometry_rule` already picks toward the current
  waypoint (or the path lookahead), before the mask is applied.
- **Stall.** Declared, not inferred: the same flat-localised-position condition the `no_advance_guard`
  already measures, over a declared window.

## Trigger

The detour is a candidate only when all of these hold, in the Router's precedence order:

1. the goal is not reached and the marker is known and inside the free movement region;
2. the mask removes the goal-directed bearing (or reduces it to a bearing whose rate is below the
   floor), so the route cannot simply continue;
3. the hero has stalled for the declared trigger window, **including inside the approach band** -
   this is the v16 lesson, and it is why the detour must outrank the approach commitment rather than
   defer to it;
4. the declared per-episode detour attempt budget is not spent.

If condition 2 does not hold, the existing rules run unchanged. The detour never replaces the mask;
it consumes the mask's verdict.

## Plan semantics

- Search over the **frozen grid only**, from the current cell to the current waypoint's cell, with
  BFS and the declared bearing order as the deterministic tie-break. No new measurement, no model, no
  device read.
- A move from a cell in a bearing is allowed when that `(cell, bearing)` is not obstructed and the
  destination cell is inside the free movement region.
- The plan length is capped. If no path exists within the cap, that is a recorded outcome, not a
  silent continuation.
- Unknown cells are walkable (fail-open) but counted. A plan whose every alternative needs unknown
  cells is still returned, with its unknown count, so the evidence states its own weakness.
- The plan is only allowed to *start* by not decreasing the goal distance; after the first step it may
  approach the goal normally, so the detour is "step aside then go around", not "go backwards for a
  while".

## Budget

No new budget line. The detour spends out of the contract's existing per-episode budget, so a
failed detour can never extend an episode:

- `maximum_detour_steps` - a sub-cap inside `maximum_commands`;
- `maximum_detour_attempts_per_episode` - how many detours may be tried;
- the detour must fit inside the same `maximum_duration_seconds` and the localisation-gap guard, so a
  failed detour cannot extend the episode the way the v7 recovery was explicitly prevented from
  doing;
- byte cost is the same per-step observation cost already paid, so no new large output.

## Acceptance criteria

A detour attempt is **effective** only when, after it completes, the goal-directed bearing is usable
again and the goal distance decreases over the declared confirmation steps. Anything else is a
recorded failure, never a success by timeout:

- effective -> normal planning resumes; the episode continues toward the waypoint and arrival is
  still required by the contract's own gate;
- ineffective -> the attempt is counted, a bounded second attempt is allowed while the budget lasts,
  and when the attempts are spent the episode terminates with a declared reason such as
  `detour_exhausted` (an explicit failure, distinct from `arrival` and from `unknown`).

## Failure handling

- No threshold changes after the run, no observation-window shortening, no session substitution.
- No re-run as a substitute for a failed attempt; existing failed runs stay failed.
- The step rows and episode summary carry the detour counters (attempts, steps, unknown-cell steps,
  outcome) so a detour is never invisible in the record.
- A detour that is never triggered, or never effective, is reported as such; it is not dressed up as
  a partial pass.

## Interaction with existing declared rules

- **Mask:** consumed, not bypassed. A detour step that the mask would remove is still removed; the
  detour is only allowed to pick a *different* bearing the mask permits.
- **Approach commitment:** the detour outranks it at the trigger (the v16 lesson). Once the detour is
  done, the approach commitment resumes unchanged.
- **Progress guard:** backward motion inside a detour is a detour step, not a stall; the guard's
  baseline is rescored the same way the bounded-retrace recovery already rescored it, so a detour is
  never read as a stall.
- **Unknown-position recovery:** unchanged; it handles a lost marker, the detour handles a present
  marker with no usable goal bearing.
- **`no_advance_guard`:** unchanged; a detour cannot be used to keep a frozen screen alive.

## Validation before any device session

1. Offline replay of the declared plan over the recorded cold-start failures
   (`route-b-coldstart-v14-start-1-batch`, `-v16-1-batch`, `-v17-1-batch`): does the trigger fire,
   what path does BFS return at `14:12`, and does it avoid the obstructed `(14:12, N)` pair while
   staying inside the region?
2. A focused test that pins the trigger precedence, the BFS tie-break, the unknown-cell counting, the
   budget caps, and the "ineffective is a recorded failure" rule.
3. Only then a new versioned route contract (a v18 digest) and a new declared experiment, at stage 1
   first, with the ordinary staged-admission rule.

## Must not do

- No model, no learned policy, no online RL, no device read beyond the existing per-step frame.
- No new input transport, no new coordinate source, no internal game state.
- No removal or weakening of the hard stop, the region filter, the mask, or any gate.
- No reopening of the frozen detector or of v16/v17.
- No claim that a detour exists until it is run and passes its own gate.

## Open decisions for the owner

1. **Plan scope.** Is a bounded BFS over the frozen grid the intended detour, or does the owner want a
   simpler declared sidestep vocabulary? BFS is the version that directly matches the `14:12` evidence
   and is offline-testable; a sidestep vocabulary is smaller but re-creates the v16/v17 guessing.
2. **Unknown-cell policy.** Keep fail-open (walk unknown, count it), or refuse to plan through unknown
   cells when a known alternative exists? Fail-open matches the mask; the count keeps it honest.
3. **Rank relative to the death-box fix.** The 2026-09-22 rebind showed the enlarged death box
   false-positives on an in-match announcement, so no device run on this route is trustworthy until
   that is fixed. The detour is offline-only until then, so the two do not conflict, but the owner
   should choose whether the death-box discriminator or this draft is implemented first.
