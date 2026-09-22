# Baseline binding: `c783402`

Method: read-only. Version guard first, then each object bound through the existing loader or by
reading the already-persisted run report. Nothing was re-run on device, and no historical test
suite was re-executed.

## Step 1 - version guard

| Check | Command | Result |
|---|---|---|
| branch | `git branch --show-current` | `hierarchical-policy-v0-prep` |
| HEAD | `git rev-parse HEAD` | `c783402c50a04fa3bd5685a22325a2896a164c15` |
| baseline present | `git cat-file -e c783402…^{commit}` | present (exit 0) |
| worktree | `git status --short` | clean (no output) |
| whitespace | `git diff --check` | clean |
| last commit | `git log -1` | `2026-09-22T21:46:43+08:00`, "docs: bring the ledger header in line with the last three commits" |
| commit count | `git rev-list --count HEAD` | 252 |

## Step 2 - the objects, bound first-hand

### Contracts and layout (read through the existing loader)

| Object | Path | Bound value |
|---|---|---|
| Route B v15 contract | `configs/movement_goal_navigation_route_b_v15.json` | digest `ba46e2b216249c8325eaf7ac50e78e519ae9ae75b1c7606cdc52c1c5e8a24d0e`; `targets_minimap_xy` `[[50,50],[80,50],[80,80],[50,70]]`; `arrival_tolerance_pixels` `4.0`; `maximum_duration_seconds` `90` |
| Placement contract v4 | `configs/movement_goal_navigation_reposition_v4.json` | digest `51cadbc0747018857a9c14b6f707f8147dfd6bd382165c0ad7403b29a433596e`; `targets_minimap_xy` `[[53.5,69.4]]`; `arrival_tolerance_pixels` `1.5` |
| Observation ROIs (local, git-ignored) | `../hok-agent-v5/configs/mobile_observation_rois.local.json` | digest `876adf7626a1736a57b1960fdbadf724c1148b49128b8d4fdc3e75d5081a8eff` |
| Visual layout | `../hok-agent-v5/configs/mobile_testbed_layout_calibrated_v3.json` | digest `13570674923272236fa43389f04cbb1f6e05c5d7d35e57a2b3fed2f49f32406f` |
| Execution layout | `../hok-agent-v5/configs/mobile_testbed_layout_all_actions_corrected.local.json` | digest `a4c077b317a216d1c0c1a33faf53a11c3c6bdbff9370b0ad9c4e984853fcbc51` |

The coordinate convention is whatever the frozen contract already uses; this pass does not infer
xy versus yx. The 1.5 px placement value is the declared placement gate. It is **not** a proven
robustness radius for the route, whose own gate remains 4.0 px and is untouched.

### The three bound runs (read from their own `batch-summary.json`)

| Run | Status | Episodes / arrivals | Transitions (terminal) | Input commands | Backlog | Retries | Findings | Integrity |
|---|---|---|---|---|---|---|---|---|
| `route-b-rebind3-stage1` | PASSED | 1 / 1 | 57 (1) | 123 | free | 0 | none | ok |
| `route-b-rebind3-stage3` | PASSED | 3 / 3 | 175 (3) | 375 | free | 0 | none | ok |
| `route-b-rebind3-stage10` | PASSED | 10 / 10 | 557 (10) | 1208 | free | 0 | none | ok |

Every batch carries `contract_sha256 = ba46e2b2…`, `observation_rois_sha256 = 876adf76…`,
`visual_layout_sha256 = 13570674…`, `execution_layout_sha256 = a4c077b3…`,
`policy_binding_versions = 1`, `binding_stable = true`, `max_actions_per_step = 3`,
`steps_without_ack = 0`.

Per-episode, all fourteen: `arrived = true`, `waypoints_reached = 4`,
`terminal_reason = NAVIGATION_GOAL_REACHED`, `episode_end_kind = TERMINATED`, duration
46.96-62.42 s against the 90 s budget.

### Independent verifier re-run in this pass

`hok-agent mobile-navigation-verify --store <run>/transitions.sqlite3 --frame-root <run>/frames --all`
for all three stores returned `store_integrity = ok`, `findings = []`, `recoverable = true` for
every episode, 57 / 175 / 557 transitions, and `terminal_reason = NAVIGATION_GOAL_REACHED` on
all fourteen.

### Death-confirmation effect (the discriminator doing real work)

In `route-b-rebind3-stage10`, episodes 1, 4 and 6 record `death_banner_steps = 1` while
`death_confirmed_steps = 0`, and all ten episodes record `death_confirmed_steps = 0` and arrive.
Three raw colour hits were rejected while the hero kept walking.

### Placement, bound first-hand

`runs/hierarchical-movement-mvp/reposition-to-passing-start-2/summary.json` used contract
`51cadbc07470…`, arrived, and stopped at `(54.8367, 69.8367)`, which is 1.41 px from the declared
target `(53.5, 69.4)`. This is the 1.41 px figure the ledger reports, now bound to a named run.

The other placement runs used different contracts (`05a0b689ac51`, `d3174df095ab`,
`66137712670f`, `d7cca3e64b76`, plus two earlier ones), and
`reposition-to-coldstart-start-6` (contract `d3174df095ab`) is the one that did not arrive
(`NOT_DONE`). Each placement run is written to its own directory with its own summary, which is
the evidence that placement is currently a separate invocation rather than a phase of the route
run.

## Step 2b - second-hand (reported, not re-executed in this pass)

| Fact | Source | Why not first-hand here |
|---|---|---|
| 650 tests pass, Ruff passes, strict mypy over 76 files | user report | the full `make check` was not run in this read-only pass |
| local == origin == `c783402`, 0 to push | user report | no `git fetch` was performed in this pass, so the remote ref was not re-read |
| cold start closed after three declared escape mechanisms | ledger and `docs/ROUTE_B_CONTRACT_INDEX.md` | the underlying runs exist on disk, but their analyses were not re-audited here |
| mask bounded-tier contribution (47 cells, 38 decisions, 5 removals) | ledger and contract index | same |
| corridor measurement (`14:12` to `9:12`) | ledger and contract index | same |

Full `make check` is a deliverable-freeze step, not a documentation step; it was deliberately not
run for this pass.

## Step 2c - `NOT_FOUND_LOCALLY`

| Missing object | Status |
|---|---|
| App source, internal API or reference sidecar | absent by design: the branch is no-source. A matching package name is not proof of ownership, and no source is held to modify |
| An independent game reference | none. The R0 contract records `independent_map_reference_available = false` and `comparison_available = false` |
| A training-data inventory or manifest document | none found under `docs/`. The persisted transitions are described only by their own run summaries |

## Step 4 - one-pass data availability registration

Reused `game_rules/r0_feedback_contract_v1.json` rather than writing a new registry. Fields are
the pack's, values are the existing contract's plus the batch facts above.

| Field | Value |
|---|---|
| task | `goal_directed_navigation_with_recovery`: four declared waypoints in order, then stop |
| source_kind | `rule_outcome` (terminal), `rgb_derived` (position, motion, death/ended), `execution_stream` (action dispatch) |
| source_exists | verified: action dispatch against the Store record, hero position from the frozen green-ring cue; unavailable: an independent game reference; unverified: death/ended |
| use_allowed | recording only. `training_allowed = false`, `reward_allowed = false`, `independent_map_reference_available = false` |
| time_alignment | `observation_period_ms = 300`; each transition carries its observation ids and capture timestamps; the settled-store path binds the dispatched action to the following observation |
| semantic_scope | terminal is arrival within the declared tolerance for every target in order. It says nothing about a win, a death, or a healthy game state |
| dependence | terminal's reference is `same_source`; position's second derivation is `implementation_independent_only`, and the duplication guard already failed it (it agreed exactly on 100% of the compared steps) |
| negative_coverage | non-death frames exist for the banner (`audit/hierarchical-movement-mvp/death-box-negatives-v1/`, 83 read-only frames); the banner false positives are counted, not photographed |
| decision | usable for recording; **not** usable for supervision or for a training/evaluation target |

Data classes actually present, from the batch summaries:

- RGB-only derived views: `derived_views_persisted = true`, `raw_frames_persisted = false`.
  Usable for offline perception work only; not an action demonstration.
- RGB plus actually dispatched actions and timestamps: present, but every batch records
  `training_eligible = false`, `events_claimed = 0` and
  `event_engine_version = mobile-navigation-no-visual-event-v1`, so these are not yet
  training-eligible transitions.
- A result reference different from the failed panel or the duplicating position derivation:
  does not exist. Recorded once here; per the pack this is not turned into a second long project.
