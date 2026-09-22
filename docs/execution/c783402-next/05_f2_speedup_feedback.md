# F2: the feedback for one speed task

Status: `OPENED_AND_IMPLEMENTED`, offline. The owner chose route B: keep the source boundary and put
F2 on the "perception / runtime speedup" task instead. This document is the declaration F2 asks for
(before any code), and it records the first-hand measurements that calibrate its gate. It trains no
candidate; that is F3.

## The task

Make one already-trained Global Agent component cheaper while its behaviour is preserved, on the
frozen Global Agent v1 simulator task. The F2 definition in the next-step pack is explicit about what
judges it and what cannot substitute for it:

| | Declared |
|---|---|
| task | perception / runtime speedup of one learnable component |
| judging reference | the same task's behaviour is preserved **and** the measured latency drops |
| forbidden substitutes | a self-supervised loss going down, a parameter count going down |
| first-round shape | one head or one encoder changes, everything else stays fixed |

## Source registration (one row, per the pack)

| Field | Value |
|---|---|
| task | speedup of the frozen Global Agent v1 macro policy on its frozen simulator task |
| source_kind | existing actual model + existing frozen offline evaluation (no new source) |
| source_exists | verified: the frozen selected checkpoint and the frozen dev/train cohort are on disk |
| use_allowed | offline measurement only. No device input, no online learning, no capture |
| time_alignment | not applicable: the comparison is over already-persisted windows, not live frames |
| semantic_scope | it can show "the same decisions got cheaper". It says nothing about whether the game goes better |
| dependence | behaviour and latency come from the same frozen checkpoint and the same frozen windows; the candidate shares them, which is the point |
| negative_coverage | the frozen 20-seed holdout (18 of 20 non-timeout for the baseline) remains the acceptance for any candidate that changes control |
| decision | usable for evaluation of a speed candidate; **not** usable as a game-state reward |

## The frozen baseline, measured first-hand (2026-09-22)

| Quantity | Value |
|---|---|
| checkpoint | DAgger `selected.safetensors`, sha256 `c033264f83d1c667c4dff5f93dec02183535386cfcde1a527a60303647a3e39e` |
| parameters | **11,383,694**, of which the single `main` resnet18 view is **11,168,832 (98.1%)**; minimap 7,968; hud 7,968; temporal + heads 198,926 |
| behaviour | dev intent macro-F1 **0.8891**, dev zone macro-F1 **0.8446** (510 dev windows) |
| latency | interleaved CPU, batch 8, 30 repetitions: median **480.8 ms**, p95 **526.9 ms** per batch; ≈60 ms per decision |

The recorded 0.8465 intent macro-F1 in `docs/GLOBAL_AGENT_V1_OFFLINE_EVIDENCE.json` belongs to the BC
checkpoint `88c2faed…`, not to this selected DAgger checkpoint, so the two numbers do not conflict.

## The gate, and its calibrated noise floor

A single isolated latency measurement on this host is not usable: the same frozen model measured a
median batch time of about 0.47 s in one context and about 2.8 s in another, and within one process
five rounds spread 1.27x. The harness therefore **interleaves** the baseline and the candidate in the
same process, alternating rounds, and judges on the interleaved medians.

Noise floor, measured by comparing the frozen checkpoint against itself:

| Quantity | Self-comparison result |
|---|---|
| intent macro-F1 delta | **0.0** (identical weights reproduce identical behaviour exactly) |
| median latency reduction | **-0.13 %** |
| p95 latency reduction | **-2.98 %** |

So the declared threshold - at least **25 %** reduction at both the median and the p95 - sits roughly
an order of magnitude above the measured noise. Parameters are reported, and are never a pass reason.

## What was implemented

- `measure_global_speed(...)` in `global_policy.py`: one checkpoint's parameters by part, behaviour
  on a declared split, and latency; read-only, refuses a missing or symlinked checkpoint.
- `compare_global_speed(...)`: the interleaved baseline/candidate measurement plus the verdict.
- `global_speed_verdict(...)`: passes only when behaviour holds and **both** the median and the p95
  drop by the declared fraction.
- CLI `global-agent-speed-report --checkpoint … [--baseline-checkpoint …]`, writing
  `speed-report.json` under `HOK_LARGE_ROOT`.
- Ten focused tests in `tests/test_global_policy_speed.py`, including that a smaller model with no
  latency win must fail.

Reports written (read-only inputs): `audit/global-agent-speed/baseline-dagger-v1/` (866 B) and
`audit/global-agent-speed/self-comparison-dagger-v1/` (2,744 B).

## Boundaries held

No training, no new source, no device, no capture. The frozen detector, the Router, the executor, the
action order, the Store semantics and every frozen lineage are untouched; `global_policy.py` is
inside the existing Torch allowlist. A parameter reduction and a self-supervised loss are not
substitutes for preserved behaviour and a measured latency drop.

## What is deliberately not claimed

- Not that any speedup has been achieved. This is the feedback, and only the baseline and its noise
  floor have been measured; a candidate is F3.
- Not that this task improves gameplay. It is the compression question the pack allows, and the pack
  says plainly it must not be called a policy-level improvement.
- Not a re-run of the 20-seed holdout. That stays the acceptance for a candidate that changes control.
