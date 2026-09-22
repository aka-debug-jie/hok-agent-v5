# HoK-Agent V5

[![CI](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml/badge.svg)](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

RGB-only MOBA research software with project-owned PixelArena, recorded-video analysis and an
existing bounded mobile testbed. It is not affiliated with a game publisher.
The public tree contains no device identity, calibrated layout, recordings, dataset or checkpoint.

## Current work

Follow the one active task in [DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md#current-execution-state).
That ledger is the authority for actual results; this section only orientates.

The narrow no-source chain is delivered on the owner-authorized self-built test app, with no
internal API, backend or internal reference:

- No-source identity and control passed with batch `active-probe-v25` under contract v9
  (`92594112`): 704/704 localisation, direction consistency 1.0, zero identity switches.
- Declared-target navigation passed, including the staged `1 -> 3 -> 10` admission with
  `goal-navigation-a3-staged-5` under contract `6e5401d7`: 14 of 14 rounds arrived, arrival rate
  1.0, arrival errors 1.69-4.00 px, zero identity switches.
- The chain is bound into the single `UnifiedTransitionStore` (L1 and L2) with
  `mobile-navigation-store-2` and `mobile-navigation-store-batch-2` under contract `0e603fcc`:
  terminal transition written before the episode ends, every step causal-order valid, a reload
  verifier that reports the episodes recoverable, and three consecutive episodes with no action
  backlog, no frame-reference damage and store integrity ok.
- The separately declared four-waypoint route B passed its staged `1 -> 3 -> 10` admission
  under contract `ba46e2b21624`: 14 arrivals in 14 attempts, four of four waypoints each,
  `arrival_rate` 1.0, no action backlog and store integrity ok on the independent reload
  verifier. Its two declared rules are a measured traversability mask, built from the
  already-recorded transitions and normalised by the declared press duration, and a
  three-step final-approach commitment. Three limits travel with it. The pass holds from the
  measured starting positions only: the recorded failing start `(78.4, 55.4)` was re-run in
  three consecutive episodes and failed all three, stalling about ten pixels short of waypoint
  `(50, 50)` in cell `14:12`, where the route must go north and the mask removes north - so the
  route's own failure lands on the mask's decision and agrees with it, and the limit is a real
  obstruction rather than a rule defect that further tuning would close. The fourth waypoint is
  `(50, 70)` rather than `(50, 80)` because `(50, 80)` measured unobservable. Finally, the pass is
  scoped to its own commit: a 2026-09-22 rebind of the same contract passed stage 1 (1/1) but
  failed stage 3 (1/3), because the corrected death box now covers the top-centre in-match
  announcement region and stopped a run whose hero was demonstrably alive. The box and its
  thresholds were then left unchanged and the stop was given a declared confirmation policy: the
  banner must hold for two consecutive observations and the hero must not travel in the same window,
  since a dead hero cannot move. Versions and digests are indexed in
  [docs/ROUTE_B_CONTRACT_INDEX.md](docs/ROUTE_B_CONTRACT_INDEX.md).

Single-policy post-training is closed as `DATA_SOURCE_LIMITED`, not left open: R0 tested three
feedback lines and all three failed for a reportable reason - the navigation feedback has no
independent map reference and its second derivation is a duplicate, the commanded-response reward
is structurally independent but keeps a sub-pixel tail (p95 5.63 px), and the discrete panel signal
is periodic but not resolvable at the 1.2 s observation cadence. No visual event engine is permitted
on this route. Reopening needs an external independent reference; the deterministic chain above is
the delivered result. Historical model results are retained separately and remain unproven.

## Environment and commands

Python >=3.11 is required. This worktree currently shares the existing environment at
`../hok-agent-v5/.venv`; it reports Python 3.11.15. Explicitly bind this checkout's source:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" ../hok-agent-v5/.venv/bin/python -m hok_agent movement-mvp --help
make PYTHON=../hok-agent-v5/.venv/bin/python check
```

The second command is for a code-delivery check, not each documentation update.
For a fresh checkout, use a project-local environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,events,vision,shadow,preingest]'
```

Dependencies and available optional groups are defined in [pyproject.toml](pyproject.toml).
GPU visibility in a sandbox is not proof of host GPU availability.

## Frozen delivery verification

Set HOK_LARGE_ROOT to the existing private artifact root. This command only verifies the local
frozen package and runs no policy, training or device input:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" ../hok-agent-v5/.venv/bin/python -m hok_agent movement-mvp \
  --mode package-hierarchical-rule --verify-only \
  --output-dir "${HOK_LARGE_ROOT:?set HOK_LARGE_ROOT}/runs/hierarchical-movement-mvp/r1-hierarchical-rule-v1"
```

Completed N1/N2 artifacts are local under `$HOK_LARGE_ROOT`:

```text
audit/hierarchical-movement-mvp/n1-real-navigation-demo-v1
runs/hierarchical-movement-mvp/n2-multigoal-recovery-v1
runs/hierarchical-movement-mvp/n2-multigoal-continuous-v1
runs/hierarchical-movement-mvp/session002-partial-navigation-shadow-v1
audit/hierarchical-movement-mvp/action-response-identity-v1
```

Earlier experiment commands and exact planning snapshots are in
[DELIVERY_HISTORY.md](docs/DELIVERY_HISTORY.md#planning-snapshot-20260909); their presence is not
an instruction to rerun consumed experiments.

## Data and execution scope

Artifacts stay under HOK_LARGE_ROOT; historical data and weights are retained.
The default storage path is Git-ignored .local-data/hok-agent-v5. Existing raw videos are not
copied into the repository. Public manifests contain anonymous identities and artifact references,
not source-video locators, credentials or device identifiers.

This batch used existing offline evidence and PixelArena. Recorded-video suggestions are not
executed actions. All rewards remain zero. No phone connection, old test/holdout, RL, new human
recordings or annotation was used.
Future device tasks use the existing locally attested testbed and its unchanged authorization
requirements in [BOUNDARIES.md](BOUNDARIES.md).

## Project documents

- [DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md#current-execution-state): current task and actual results.
- [ENGINEERING_CONVERGENCE_PLAN.md](docs/ENGINEERING_CONVERGENCE_PLAN.md): goals, budgets and exits.
- [AGENTS.md](AGENTS.md): implementation and verification rules.
- [BOUNDARIES.md](BOUNDARIES.md): data, Actor and device boundaries.
- [DELIVERY_HISTORY.md](docs/DELIVERY_HISTORY.md): historical evidence and superseded instructions.

Legacy modules remain reusable components, not parallel active workstreams.
The frozen hierarchical, Global Agent and mobile protocols remain in docs/ and apply only when
their specific historical lineage is being inspected.

## License

Licensed under the [Apache License 2.0](LICENSE).
