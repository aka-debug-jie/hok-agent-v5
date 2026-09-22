# Route B contract index

Route B is the second declared device route in the no-source movement chain, run on the
owner-authorized self-built mobile testbed app through `mobile_testbed.py`. It is a versioned
lineage: each `configs/movement_goal_navigation_route_b_vN.json` is a frozen declaration with its
own `contract_sha256`, and a version is only ever superseded by a new file. Old versions and old
runs stay intact and reproducible at their recorded digest.

This page is an index, not evidence. The evidence — measured results, causes and verdicts — lives in
the ledger sections of [DELIVERY_PROGRESS.md](../DELIVERY_PROGRESS.md).

## Versions

| # | digest | declared change | device batch and outcome |
| --- | --- | --- | --- |
| v1 | `634c26f524e9` | first declared four-waypoint rectangle `(44,44) (88,44) (88,84) (44,84)` inside the free-movement region | `route-b-batch-1` 0/3 arrived, best 2 of 4, `TIMEOUT` |
| v2 | `c6cbbb517771` | adds the declared progress guard | `route-b-batch-2` 0/3; `route-b-batch-3` 0/3 with a `SAFETY_STOP` |
| v3 | `b56351901b59` | guard value revision | `route-b-batch-5` 0/3, reached 2/3/3 waypoints |
| v4 | `e33f7def7b1f` | adds the declared `final_approach` block | `route-b-batch-6` 0/3, reached 2/2/2 |
| v5 | `d06b1821d7d5` | moves the targets to a diagnostic inset `(50,50) (80,50) (80,80) (50,80)` | `route-b-batch-7` 0/3 |
| v6 | `fcb4d8086a91` | replaces the override stack with the single-arbiter `planner` | `route-b-batch-9` and `route-b-batch-10` 0/3, both `CAPTURE_FAILURE` — this is the batch pair the terminal-blind-run review is built on |
| v7 | `13465a1194a6` | adds `unknown_recovery` = `bounded_retrace` | never spent on a session: the offline replay over all six recorded blind runs shows the approved retrace bearing acts in one run for one step |
| v8 | `3c41d5bbbfb1` | recovery becomes `bounded_retrace_or_waypoint` | `route-b-batch-11-stage1` 0/1, 3 of 4. The recovery fired on hardware for the first time (marker lost steps 73–76, retrace applied, step 77 localised); the leg still failed on convergence |
| v9 | `03339a7b57e5` | explicit press/release in the approach band plus the declared backlog bound | `route-b-batch-12-stage1` 0/1, 3 of 4. Bounded press steps fall from 3.5–8.5 px to a 1.21 px median |
| v10 | `43297e0dd42a` | recovery re-aims at the current waypoint within a declared 16 px band | `route-b-batch-13-stage1` 0/1, 3 of 4 |
| v11 | `41f80184500d` | direct final approach, but the direction function is fed the previous joystick, so one sector of hysteresis locks NE/NW | `route-b-batch-14-stage1` 0/1, **0 of 4** — a recorded regression, kept as evidence rather than deleted |
| v12 | `4a18c686b464` | same declared values, hysteresis no longer fed back | `route-b-batch-15-stage1` 0/1, back to 3 of 4 |
| v13 | `393dd82ba7c6` | last target `(50,80)` becomes `(50,70)`, because `(50,80)` measured unobservable | `route-b-batch-16-stage1` **1/1 arrived, 4 of 4** — the first complete four-waypoint pass. `route-b-batch-17-stage3x` then 0/3 on the same digest, so the pass is start-position dependent |
| v14 | `caa0a58d67b3` | adds `no_advance_guard` (`flat_localised_position`, 12 steps, 1.0 px) so a frozen screen is an `ACTION_FAILURE` rather than a `TIMEOUT` | `route-b-batch-18-stage3` 0/3, 0 of 4, `TIMEOUT`, `no_advance_events=0` — the guard correctly did not fire because the world was still advancing |
| v15 | `ba46e2b21624` | adds the measured traversability mask and a final-approach commitment of three steps | `route-b-batch-19-stage3` and `-20-stage10` passed 3/3 and 10/10 but with the mask **inert** (a vocabulary defect, see below). After the fix, `route-b-batch-22-stage1`, `-23-stage3` and `-21-stage10` passed **1/1, 3/3 and 10/10** with the mask live. The 2026-09-22 rebind of the same digest passed stage 1 (1/1) but failed stage 3 (1/3) on a corrected-death-box false positive; the detector then gained a declared confirmation policy and the pass was **re-bound to the current code** by `route-b-rebind3-stage1`, `-stage3` and `-stage10` at **1/1, 3/3 and 10/10** on the same contract but with ROIs `876adf7626a1` and a declared 1.5 px placement |

| v18 | `f6ae612becc7` | adds a declared bounded grid detour on top of the v15 grid and mask: a breadth-first search over the frozen grid, triggered only when the mask has just removed the goal-directed bearing and the hero has stalled, with two attempts per episode and `detour_exhausted` as its own outcome | `route-b-detour-stage1` 0/1: the trigger asked for no motion, the failure oscillates, so `detour_attempts=0` and it timed out at `14:12` |
| v19 | `a7945b2c` | same detour with the trigger corrected to a no-progress rule (goal-distance improvement over six localised steps) | `route-b-detour3-stage1` 0/1 but the detour fired: two attempts at `14:12`, plan `NE NW`, ended `detour_exhausted` at twenty steps. It moved the hero **east**, and the measured displacements show every bounded press in that cell is a wall slide (north = 1.70 px west against 0.20 px north), so no available bearing goes north and no cell-transition plan is sound (mean bounded displacement under 2 px against 4 px cells) |

## The v15 traversability grid

The grid is frozen inside the v15 contract and aggregated from these exact source runs:

```text
route-b-batch-1   route-b-batch-2   route-b-batch-3   route-b-batch-5
route-b-batch-6   route-b-batch-7   route-b-batch-9   route-b-batch-10
route-b-batch-11-stage1  route-b-batch-12-stage1  route-b-batch-13-stage1
route-b-batch-14-stage1  route-b-batch-15-stage1  route-b-batch-16-stage1
route-b-batch-17-stage3x route-b-batch-18-stage3
```

That list is pinned, not discovered by glob. The first freeze was built by globbing
`route-b-batch-*`, and the five v15 batches added afterwards moved 55 of its 208 cells and added
three more, so the frozen grid had become unreproducible. Pinning the sources — and failing closed
when a contract records no source list — is what makes `make traversability-check` meaningful.

```sh
make traversability-check PYTHON=../hok-agent-v5/.venv/bin/python \
  HOK_LARGE_ROOT=/path/to/hok-agent-v5 WZRY_DATA_ROOT=/path/to/wzry-data
```

The check rebuilds from the pinned list and fails on any cell difference. It is bound to the
contract in `MOBILE_NAV_ROUTE`, so it verifies whichever route contract is selected, not only v15.
`make traversability-build` regenerates the block itself, and the block records its own
`source_runs`, so a later contract can be regenerated without consulting this page.

## Scope limits that travel with the pass

- **Start position, now measured and named rather than merely open.** Every passing run began near
  `(53, 68)` or `(62.9, 61.9)`, and the pass therefore holds from the measured starting positions
  only. The recorded failing start `(78.4, 55.4)` was then re-run deliberately, in three consecutive
  episodes from that start with the contract unchanged, and it failed all three: the route does not
  fit its declared 90 s duration budget from there. Reproducing the pass also depends on placement:
  a start 2.8-3.9 px from the recorded passing starts timed out, while a start delivered within
  0.66-0.79 px by the declared 1.5 px placement passed, so the tolerance on the measured start is
  about one pixel rather than the four pixels the earlier coarse placement assumed.
- **The cause is a real obstruction, and it agrees with the mask.** All three of those episodes end
  with the hero stalled about ten pixels from waypoint `(50, 50)`, needing to travel north. The
  closest one stalls in cell `14:12`, which is exactly a cell where the frozen grid removes north,
  at 0.080 px per 100 ms over 36 bounded samples against 0.183 for north-east in the same cell. The
  route is not failing because a rule is wrong; it must go north through a cell where north does not
  work. That is the strongest evidence so far that the mask measures something real, and it arrives
  from a direction nobody designed.
- **What that limit is not, and what would close it.** It is not a commitment or hysteresis problem:
  two declared fixes were tried and both failed for reasons now recorded (v16's stall commitment
  never engaged because the stall is inside the approach band, and v17's band hysteresis engaged but
  the hero genuinely cannot go north). The mask can only remove a bearing, it cannot plan a detour:
  the only better bearing at `14:12` is north-east, which moves away from the goal, and the greedy
  objective will not take a two-step detour such as west then north. Closing the limit needs declared
  detour planning, which the route does not have and which is not built here. That planning is now
  built as contract v18 (`f6ae612becc7`): a bounded breadth-first search over the same frozen grid,
  triggered when the mask has just removed the goal-directed bearing and the hero has stalled, with
  re-masking of every planned bearing, a declared attempt and confirmation budget, and
  `detour_exhausted` as its own failure. It was then run, and it did not open the limit: v18 never
  fired because its trigger asked for no motion while the failure oscillates, and v19 (`a7945b2c`)
  fired twice at `14:12` but its plan moved the hero east. Measuring the actual displacements corrects
  the explanation the earlier text carried: the grid stores only the projection onto the commanded
  bearing, and at `14:12` a bounded north press is a 1.70 px **west** slide against 0.20 px north, so
  the cell is a wall slide rather than a slow-but-passable bearing. No bearing available there goes
  north, and no cell-transition plan is sound because the mean bounded displacement is under two
  pixels against four-pixel cells. Opening the limit needs a displacement-vector record per
  `(cell, bearing)` and evidence of a lateral gap, neither of which exists.
- **The route is not the original rectangle.** The fourth waypoint is `(50, 70)`, not `(50, 80)`,
  because `(50, 80)` measured unobservable: across six runs and 211 localised samples the closest
  any localised position came to it was exactly 6.00 px against a 4.0 px tolerance.
- **The mask's criterion is the projection, and it can remove the most-moving bearing.** Measured from
  the pinned runs at the Router's bounded tier, the mask makes thirty-eight decisions over forty-seven
  covered cells: five removals and thirty-three keeps, and two of the five removals are the
  largest-magnitude motion in their own cell. At `14:12` north is removed on a `0.0548` projection
  while its measured mean displacement is `(-0.18, -1.73)` px, the largest motion in the cell, and at
  `14:11` the same pattern is `0.0569` against `0.3860`. That is correct by the mask's own criterion -
  the press did not move the hero along the commanded bearing - but it means a route that has to travel
  along a wall while pressed against it is the case where the mask removes the only thing that moves.
  The recorded ablation agrees: with the mask inert the same contract passed thirteen of thirteen.
- **The thin cells stay unmeasured, and the reason is measured.** A declared bounded traverse of the wall
  corridor (probe contract `da70e99fa632`, region moved to include `15:13`/`16:13`, cell cap 8 to 12)
  was placed at `(63.0, 54.0)` and ran its two sessions: 96 pulses and 24 control windows each, zero
  hard stops, 776 and 778 of 780 frames localised. Both sessions passed their own gates, but the analysis
  failed on `region_conformant` and `concentration_conformant`, because the measured idle drift is
  `0.4551` px per 100 ms against bounded-tier press rates of `0.05` to `0.18`. Only 12.7 percent of the 79
  cell-bearing observations clear that idle bound, and the thin cells return one to three samples per
  bearing at rates at or near zero. Filling them needs many incidental visits rather than a short
  traverse, which is why the mask's bounded-tier contribution is only measurable where the route already
  goes, and why the earlier probe line used a 2500 ms instrument that measures a tier the Router never
  issues.
- **The corridor is narrow, not blocked, and the cold-start point is a mask-floor question.** A probe
  contract may now declare the subset it presses (`press_directions`, order `declared-subset-repeating`),
  because a balanced eight-direction schedule cannot measure a wall: south is responsive and north is
  not, so its own pulses walk the hero away from the wall. Two directional traverses were placed at
  `14:12` and both passed every gate (region and concentration conformant, zero violations, idle drift
  `0.085` and `0.0437`). `movement_active_probe_v15` pressed north, north-east and north-west six times
  each and covered `14:12` to `14:10`, where nothing upward clears the `0.05` floor.
  `movement_active_probe_v16` pressed twelve times each and covered `14:12` north-west to `9:12`, about
  five cells and seventy-two observations, where four cell-bearing pairs sit at or above the floor with
  at least three samples: `10:12` north `0.2018` (n=3), `13:12` north-east `0.0773` (n=5), `13:12`
  north-west `0.0575` (n=5), `12:12` north-west `0.0602` (n=7). So the hero creeps north-west about five
  cells, or twenty to twenty-four pixels, over thirty-six bounded presses, an implied `0.6` px per press,
  while the mask removes the bearing that produces that creep at the stall cell (`0.0548` against the
  `0.08` floor). The displacement record and the lateral-gap evidence the detour named are both present,
  and the mechanism the result points to is a bounded declared persistence on a masked bearing whose
  creep is measured. The probe reads `13:12` north at `0.0209` (n=5) against the incidental `0.2808`
  (n=15); the disagreement is recorded, not smoothed.
- **The probe now refuses a moving hero, and that is measured to matter.** The corridor traverse's
  failure was diagnosed as a walking hero: the analyser subtracts the idle rate it measures, so an idle
  drift of `0.4551` px per 100 ms raised the floor above every bounded press and only 12.7 percent of
  observations cleared it. Contract `3588ae62866a` declares an optional `stationarity_preflight` - eight
  frames, 1.5 px - and abandons the session before any pulse if the localised hero moves further. The
  same traverse re-run records travel of `0.0` and `0.136` px, idle drift down to `0.1047`, observations
  clearing the idle bound up from 12.7 to 42.7 percent, a second session that is fully region-conformant
  with zero violations where the earlier run had nine, and the thin cell `16:13` resolved to three to six
  samples on eight bearings. It is still not enough for the corridor: the analysis is FAILED because the
  first session drifted out during the run, the traverse covers `16:12` through `23:12` so its own pulses
  walked the hero away from the wall, and `15:13` still holds a single sample, because the contract
  requires all eight directions and south is more responsive than north there (`0.1527` against
  `0.0000`).
- **Mask attribution.** The mask is live and fires on 1 to 5 steps per episode, but its isolated
  contribution is not measured by a controlled ablation. The commitment is the driver: an earlier
  run of the same contract with the mask inert passed 13 of 13 episodes.
- **The pass is re-bound to the current code, from a declared 1.5 px placement.** On 2026-09-22 the
  staged admission was re-run against the unchanged contract `ba46e2b2` to turn the inference into a
  fact, and the first attempt failed: it passed stage 1 but stopped stage 3 on a corrected-death-box
  false positive, and after the detector was fixed a second attempt timed out from a start 2.8-3.9 px
  off the recorded passing starts. Placement was the obstacle, so
  `movement_goal_navigation_reposition_v4.json` (`51cadbc07470`) declares a 1.5 px arrival tolerance, a
  6.0 px deceleration entry with the 200 ms tier and a 0.5 px no-advance travel; the route's own 4.0 px
  arrival gate is untouched. It landed 1.41 px from its target and delivered a start measured 0.79 px
  from the recorded passing start by the green-ring centre and 0.66 px by the route run's first fix.
  `route-b-rebind3-stage1`, `-stage3` and `-stage10` then passed **1/1, 3/3 and 10/10** with the death
  policy live and ROIs `876adf7626a1`, every store verified. The pass is now bound to the checked-out
  code, and the start sensitivity is sharper rather than gone: it depends on that 1.5 px placement.
- **The death-box defect and its fix, kept because they bound the pass's history.** The enlarged
  `(683, 0, 937, 46)` box from `5031c82` reads only about 100-120 red and 0 white on ordinary frames
  but overlaps the top-centre in-match announcement region, where a red banner with white text clears
  the unchanged 2000/80 thresholds; it stopped `route-b-rebind-stage3` episode 1 while the hero was
  walking. The stop is proven false from the stop frame: the green-ring cue centroid equals the logged
  position, the main view shows a full health bar with no death overlay, the hud skills are coloured
  rather than greyed, and the next episode begins beside the stop rather than at the fountain. The fix
  relaxes nothing: the box and minima are unchanged and the stop now also requires the colour test to
  hold for `confirmation_steps` 2 consecutive observations and the localised hero to travel at most
  `maximum_travel_pixels` 1.0 over `stationary_window_steps` 2, with a lost marker counting as no
  travel. That refuses the recorded stop step on both halves (streak one against two, travel 4.24 px
  against 1.0) while the owner death reference still reads death-visible, and the negative set is
  preserved under `audit/hierarchical-movement-mvp/death-box-negatives-v1/`. The confirmation is shown
  to do real work rather than merely never firing: in the ten-episode stage `death_banner_steps` is 1
  in episodes 1, 4 and 6 while `death_confirmed_steps` is 0 in all of them, so three raw colour hits
  were rejected while the hero kept walking and arrived. The offending non-death banner itself is still
  unphotographed, because the runner persists no life-state strip; the counted rejections are direct
  evidence that a rejection happened but weaker than a picture.
