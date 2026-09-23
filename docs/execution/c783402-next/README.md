# Next-step pack for `c783402`: baseline binding and reuse gap

This directory holds the read-only result of the first task in
`hok_next_execution_c783402_20260922.zip`, executed against the checkout at
`c783402c50a04fa3bd5685a22325a2896a164c15` on the branch `hierarchical-policy-v0-prep`
(worktree `hok-agent-v5-movement-diversity`).

## What this is

- `01_baseline_binding.md` - which of the reported facts are verified first-hand in this pass,
  which are second-hand, and which are not found locally; plus the one-pass data availability
  registration.
- `02_reuse_gap.md` - the five-item reuse-gap table for "same-session placement then route",
  each with its code location and the minimal test that would fail if the layer were silently
  unwired.
- `03_next_step.md` - the single next step, its status, the milestone relabelling, and the
  explicit ceiling.

## Update: the gap was then closed offline

After the read-only pass, the same-session lifecycle was implemented offline as
`run_mobile_navigation_placement_route` (CLI `mobile-navigation-placement-route`) in
`src/hok_agent/mobile_navigation_store.py`, with seven offline tests in
`tests/test_mobile_navigation_store.py`. The route contract, the ROIs, the layouts, the
thresholds, the executor, the Router and the Store schema are unchanged and `training_eligible`
stays `false`. The first authorized device attempt found a real defect instead of a result: the step
loop read `persistence_applied` before it was bound for any contract without a declared
masked-persistence block, so it dispatched no input; that is fixed and pinned offline. The composed
chain then passed 4 of 4 authorized runs (`placement-route-2` to `-5`) on one session and one Store
each, with every episode recoverable and `store_integrity=ok`. That repeatability is of the
composition, not of an independent start: only `-2` began with a real placement walk, because the
later runs already stood inside the placement tolerance. The cold-start limit, the mask scope and
the bound pass are unchanged, and this is not a staged admission for the route contract.
`02_reuse_gap.md` and `03_next_step.md` record the implemented status and the runs.

## B: awaiting an owner-named truth source

`06_truth_source_interface_decision.md` sets out what an external truth source must supply (hero
position in a fixed frame, and/or match terminal), the three candidate shapes - app-side export, app
source tree, or a controlled ground-truth capture - and what would and would not become provable with
each. Nothing is implemented and no adapter is written until the owner names a source.

## Frozen

The owner froze the delivered scope on 2026-09-22. `F0` and `F1` are delivered, the delivered scope
is frozen at commit `5c17846`, no further device work is scheduled, and `F2`-`F5` stay blocked on
scope because no independent result reference exists.

## What this is not

- Not a runtime configuration. No contract, layout, ROI or threshold is added or changed here.
- Not a permission. It does not authorize device input, capture, training, annotation, paid
  services, push or merge.
- Not a re-validation of the passed admission. The 14 of 14 remains the evidence recorded at
  its own commit; this pass only re-reads and binds it.
- Not a design for cold start, corridor escape, or a reward reference. Those stay closed as
  recorded in the ledger.

## Scope held during this pass

Read-only over the repository and the already-persisted local reports. No phone connection, no
capture, no input, no training, no test/holdout opening, no threshold or route change. Only the
top of `DELIVERY_PROGRESS.md` is mutable execution state, and it gains one pointer line to this
directory.

## Hash rule used throughout

A repository blob SHA, a raw-file SHA-256, and a contract's internal normalised digest are three
different values. Every digest below was obtained by calling the existing loader
(`_goal_navigation_contract`, `load_observation_rois`, `load_layout`), not by padding a prefix
and not by hashing the file bytes.
