# Delivery Progress

- Last update: 2026-08-12
- Current task: `MINIMAL-V2-STRUCTURED-BC`
- Status: `COMPLETE`
- Claim scope: `pixelarena_engineering`
- HoK capability claim: `false`
- GameCore equivalence claim: `false`

## Reset decision

The former platform-first M0-M8 plan has been retired from the active tree. GameCore is
not available and is not on the main route. Minimal V1 froze a deterministic abstract
1v1 PixelArena service, three test policies, public JSONL recording, and fresh-process
replay. Minimal V2 adds only a small supervised structured-policy slice on top of it.

Reference measurements were made without running their code:

| Repository snapshot | Tracked files | Python files | Python lines |
|---|---:|---:|---:|
| ResnetGPT `5981806` | 31 | 15 | 1,575 |
| WZCQ `2f16e0d` | 27 | 18 | 2,723 |
| wzry_ai `996ac49` | 54 | 11 | 1,256 |

The formal 10x cap uses the smallest value per metric: 270 files, 110 Python files, and
12,560 Python lines. The frozen V1 gate was 22 files, 14 Python files, and 1,400 Python
lines. Minimal V2 uses the still much stricter gate of 24 files, 15 Python files, and
1,800 Python lines including tests.

## Acceptance ledger

Minimal V1 remains frozen at commit `4267c202f2205237fc48e889a9ddd2b31c7f632d`.
Minimal V2 implements a CPU-only 550-parameter MLP that imitates the scripted test
driver from public pre-action observations. The model consumes ten fixed public
features; legal actions never enter `forward` and are consulted only at evaluation or
execution boundaries.

- V2 code: `COMPLETE`
- V2 focused tests: `PASSED` — 4 tests
- V2 full checks: `PASSED` — Ruff clean; strict mypy clean in 9 source files; 27 tests
- V2 size gate: `PASSED` — 24 files; 15 Python files; 1,787 Python lines; 4 root Markdown files
- V2 safety scan: `PASSED` — zero findings; generated `runs/` excluded from source gates
- V2 acceptance: `PASSED`
- V2 run directory: `runs/minimal-v2-bc-v1` — Git ignored
- V2 commit: `this delivery commit; resolve with git rev-parse HEAD`

### Minimal V2 result

Collection completed all 256 episodes and 3,232 scripted-side pre-action transitions.
Canonical public-observation SHA-256 deduplication produced 495 unique samples and zero
label conflicts. The fixed class-stratified hash split contains 345 train, 74 validation,
and 76 test samples with zero hash overlap; each of the three demonstrated classes occurs
in every split. The public dataset contains no legal-action set, reward, teacher, truth,
or privileged field.

| Seed | Test CE / initial CE | Exact top-1 | Balanced accuracy | Majority baseline | Illegal top-1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.049954 | 98.684% | 99.074% | 47.368% | 1.316% |
| 1 | 0.044323 | 100.000% | 100.000% | 47.368% | 0.000% |
| 2 | 0.048433 | 100.000% | 100.000% | 47.368% | 0.000% |

All three training seeds passed every frozen metric. Seed 2 had the lowest validation
cross-entropy and was selected. It destroyed the opposing abstract crystal from both
blue and red in 12 ticks; raw illegal actions and execution-time mask corrections were
zero in both runs. Runtime: Python 3.11.15, Torch 2.5.1+cpu, one thread, CPU only.

The run contains exactly `dataset.jsonl`, three JSON model files, and `report.json`.
Observed SHA-256 values:

```text
dataset.jsonl     609179c1cd903f486cd8b37e3aacd17c363ba3229c48cd9c3338389f92d2e388
model-seed-0.json f767d3088b317fbce0a4791295608382a2595532001eb2812385bdb2ee7a6a85
model-seed-1.json 78fb3399db08e464ba342208541b0a1edc4678902e6f3a4cd506c967c94d3eab
model-seed-2.json aadecccadfd9e7a155e41b8b9b446273ef078d68c6dacceea020372207a370fb
report.json       a8def25d33caa81ce79610575ea87d840e2ec3ec241e2b7d976bae58a793746f
```

### Minimal V1 frozen result

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
.venv/bin/python -m pip install -e '.[dev,bc]'
make check
.venv/bin/python -m hok_agent accept-minimal-v1 --seed 101
.venv/bin/python -m hok_agent accept-minimal-v2-bc --output-dir runs/minimal-v2-bc-v1
PYTHONPATH=src .venv/bin/python -S -c 'import hok_agent, hok_agent.arena, hok_agent.cli'
git diff --check
```

The maximum claim is supervised imitation of the scripted policy under the fixed
PixelArena rules. No policy superiority, environment-seed generalization, recurrent or
reinforcement learning, GameCore, Honor of Kings, transfer, or real-client capability
conclusion is allowed.
