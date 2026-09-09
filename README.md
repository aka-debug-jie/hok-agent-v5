# HoK-Agent V5

[![CI](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml/badge.svg)](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

RGB-only MOBA research software with project-owned PixelArena, recorded-video analysis and an
existing bounded mobile testbed. It is not affiliated with a game publisher.
The public tree contains no device identity, calibrated layout, recordings, dataset or checkpoint.

## Current work

Follow the one active task in [DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md#current-execution-state).
The [engineering plan](docs/ENGINEERING_CONVERGENCE_PLAN.md) defines the next 12-hour offline batch:

- N1: make existing real-ROI localization, goals, suggestions and unknown intervals visible.
- N2: connect multi-goal rules to the existing Store and mid-episode recovery in one runtime.
- One integration check and handoff. No new training or packaging milestone.

The frozen hierarchical package contains a 10-episode synthetic geometry report alongside the
older recoverable rule/Event evidence. Its 352/352 decisions and 10/10 routes are narrow simulator
results; packaging does not prove the newer strategy already uses the older recovery runtime.
The current Movement learning route has no promoted checkpoint. Real-player/goal observability
and real gameplay improvement remain unproven. Historical model results are retained separately.

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

N1/N2 commands will be documented after implementation. Earlier experiment commands and exact
planning snapshots are in [DELIVERY_HISTORY.md](docs/DELIVERY_HISTORY.md#planning-snapshot-20260909);
their presence is not an instruction to rerun consumed experiments.

## Data and execution scope

Artifacts stay under HOK_LARGE_ROOT; historical data and weights are retained.
The default storage path is Git-ignored .local-data/hok-agent-v5. Existing raw videos are not
copied into the repository. Public manifests contain anonymous identities and artifact references,
not source-video locators, credentials or device identifiers.

This batch uses existing offline evidence and PixelArena. Recorded-video suggestions are not
executed actions. All rewards remain zero. No phone connection, old test/holdout, RL, new human
recordings or annotation is part of the active batch.
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
