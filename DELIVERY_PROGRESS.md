# Delivery Progress

- Last update: 2026-08-12
- Current task: `MINIMAL-V1`
- Status: `COMPLETE`
- Claim scope: `pixelarena_engineering`
- HoK capability claim: `false`
- GameCore equivalence claim: `false`

## Reset decision

The former platform-first M0-M8 plan has been retired from the active tree. GameCore is
not available and is not on the main route. The only active target is a deterministic,
abstract 1v1 PixelArena vertical slice with a local service, three test policies, public
JSONL recording, and fresh-process replay.

Reference measurements were made without running their code:

| Repository snapshot | Tracked files | Python files | Python lines |
|---|---:|---:|---:|
| ResnetGPT `5981806` | 31 | 15 | 1,575 |
| WZCQ `2f16e0d` | 27 | 18 | 2,723 |
| wzry_ai `996ac49` | 54 | 11 | 1,256 |

The formal 10x cap uses the smallest value per metric: 270 files, 110 Python files, and
12,560 Python lines. The local gate is stricter: 22 files, 14 Python files, and 1,400
Python lines including tests.

## Acceptance ledger

- Code refactor: `COMPLETE`
- `make check`: `PASSED` — Ruff clean; strict mypy clean in 8 source files; 22 tests passed
- `accept-minimal-v1`: `PASSED` — seed 101
- Size gate: `PASSED` — 22 files; 13 Python files; 1,069 Python lines; 4 root Markdown files
- Safety scan: `PASSED` — zero findings
- Commit: `this delivery commit; resolve with git rev-parse HEAD`

Acceptance details: scripted blue and scripted red each destroyed the opposing abstract
crystal in 12 ticks with 7 moves, 5 attacks, and 5 structure-damage events. Seeded random
ended at the 32-tick limit. All three public traces reproduced tick-for-tick in fresh
spawned processes, and a modified observation hash was rejected.

Commands run from the repository root:

```text
.venv/bin/python -m pip install -e .
make check
.venv/bin/python -m hok_agent accept-minimal-v1 --seed 101
git diff --check
```

Only infrastructure behavior may be reported in this task. No training, strategy
performance, GameCore, Honor of Kings, or real-client capability conclusion is allowed.
