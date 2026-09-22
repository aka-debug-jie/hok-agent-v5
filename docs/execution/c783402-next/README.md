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
masked-persistence block, so it dispatched no input; that is fixed and pinned offline. The second
attempt passed end to end as 1 of 1 composed run on one session and one Store. The cold-start limit,
the mask scope and the bound pass are unchanged; continuing to a staged 3 needs the owner's word.
`02_reuse_gap.md` and `03_next_step.md` record the implemented status and the run.

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
