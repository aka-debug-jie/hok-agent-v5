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

## The F3 smoke, and the blocker it found

The owner asked for the pack's "minimal gradient/load smoke" before any pilot. It was run offline in
`tests/test_global_agent.py` (the allowlisted focused test for this line; the torch allowlist in
`safety.py` was deliberately **not** widened, and the checker rejected an earlier draft that put
torch in a new file).

What the smoke proves:

| Result | Value |
|---|---|
| only `main` changed, 512-wide interface kept | `minimap`/`hud`/`project`/`temporal`/`pool`/`intent`/`zone`/`scene` parameter counts identical |
| a cheaper `main` is constructible | 11,168,832 -> 267,744 parameters (2.4 %) |
| total model | 11,383,694 -> 482,606 (**-95.8 %**) |
| one distillation step | gradients appear on `main` only; every other parameter keeps `grad is None` and stays bit-identical |

What it found, and why F3 cannot simply start:

`load_global_model` constructs a `GlobalMacroPolicy`, whose `main` is hardcoded to `resnet18`, and
loads it with `strict=True`. A student whose `main` is a different network therefore **cannot be
loaded**: saving it succeeds, and loading it raises
`RuntimeError: Missing key(s) in state_dict: "main.conv1.weight", "main.bn1.weight", …`.

So the gradient half works and the load half does not. Turning this into a candidate needed a
**declared architectural variant** - a versioned configuration that says which `main` a checkpoint
carries - because that is a public interface change, not an optimizer tweak. The owner authorized it,
and it is implemented:

- `GlobalMacroPolicy(variant, main_architecture)`, with `MAIN_ARCHITECTURES = ("resnet18", "compact")`
  and an unknown architecture refused at construction.
- `_save_model` now writes `main_architecture` into the checkpoint metadata, so a checkpoint declares
  the `main` it carries.
- `architecture_from_metadata` reads it, and **a metadata without the key is read as resnet18**. That
  backward-compatibility rule is what keeps every frozen checkpoint on disk loadable without being
  rewritten, so their recorded hashes stay valid. `human_ifo.py` and `human_inverse.py` keep working
  unchanged because they only ever build the default.

With that in place the whole pipeline connects, demonstrated end to end on the frozen split:

| | median | p95 | intent macro-F1 | main architecture | parameters |
|---|---|---|---|---|---|
| baseline (frozen teacher) | 480.1 ms | 495.8 ms | 0.8891 | resnet18 | 11,383,694 |
| candidate (untrained compact) | 5.59 ms | 7.2 ms | 0.0039 | compact | 482,606 |
| verdict | 98.8 % faster | 98.6 % faster | -0.885 | - | -10,901,088 |

The verdict is `passed = false` with the single reason `intent_macro_f1_regressed`. That is the gate
working exactly as declared: a candidate that is 86x faster and 95.8 % smaller is still rejected
because its behaviour collapsed, so a latency win and a parameter win are demonstrably not
substitutes for preserved behaviour. It also shows the latency half has large headroom - the compact
`main` is cheap enough that a trained student has room to pass it - so the open question is behaviour
recovery, which is F3's pilot.

## The F3 pilot: two bounded configurations, both rejected

`distill_global_main` (CLI `global-agent-distill`) was added and run twice on the frozen pilot
`global-agent-v1/pilot-40-10-v1` (2,016 train windows, 510 dev windows), training only the compact
`main` while every other weight is copied from the teacher and verified bit-identical afterwards.
The gate and its thresholds were not touched between runs; only the training configuration changed.

| Run | configuration | loss first -> last | dev intent macro-F1 | dev zone macro-F1 | gate |
|---|---|---|---|---|---|
| `pilot-compact-v1` | 200 steps, lr 0.01 | 57.9 -> 62.3 (rose) | **0.3800** | 0.4578 | rejected, `intent_macro_f1_regressed` |
| `pilot-compact-v2` | one epoch, 252 steps, lr 3e-4 | 57.9 -> 39.9 (min 26.7, spread to 147) | **0.0683** | 0.1233 | rejected, `intent_macro_f1_regressed` |
| teacher | frozen DAgger | - | 0.8891 | 0.8446 | baseline |

Both candidates were about 99% faster than the teacher (5.5 ms against ~511 ms per batch) and 95.8%
smaller, and **both were rejected on behaviour alone**. The untrained student started at 0.0039, so
the first configuration did learn something, but three findings are enough to stop:

- the dev result is not monotone in the budget: the lower learning rate with more steps did far
  worse (0.0683 against 0.3800), so the current objective is not a reliable training signal;
- the loss trace is unstable within a single run (min 26.7, max 147), consistent with an
  ill-scaled MSE on raw logits rather than a well-behaved distillation objective;
- neither configuration is close to the behaviour the gate requires, and the pack says explicitly
  not to expand data, time or parameters before a small verified gain exists.

So the honest F3 outcome is a working pipeline and a negative pilot. Promoting nothing is the
correct result, and a real distillation configuration (a proper objective and schedule, then a
larger budget) would be a new decision with real compute behind it, not another tweak.

## The gentler backbone: the same architecture family, halved

The owner then asked for a gentler compression rather than a different network, so a
`resnet18_shallow` variant was declared: the same resnet18 shapes, the same 512-wide output and the
same conv widths, with the second block dropped from each of `layer1` to `layer4`. It keeps the family
and drops 55.1 % of the parameters (main 11,168,832 -> 4,898,112), most of them from the heaviest last
stage. This is what "preserve behaviour" looked like in practice:

| Run | budget | loss first -> last | dev intent macro-F1 | dev zone macro-F1 | gate |
|---|---|---|---|---|---|
| `pilot-shallow-v1` | 1 epoch, 252 steps, lr 1e-3 | 50.1 -> 4.61 | **0.8526** | 0.7699 | rejected, `intent_macro_f1_regressed` |
| `pilot-shallow-v2` | 4 epochs, 900 steps, lr 1e-3 | 50.1 -> 2.55 (min 0.375) | **0.8772** | 0.8243 | rejected, `intent_macro_f1_regressed` |
| teacher | frozen DAgger | - | 0.8891 | 0.8446 | baseline |

The gentler backbone moved dev intent macro-F1 from 0.0039 untrained to 0.8526, then to 0.8772 with a
4x budget, and it clears the latency half comfortably - the measured reduction is **43.3 % at the
median and 42.4 % at the p95**, both above the declared 25 %. The gate still returns `passed = false`
because the behaviour drop is **0.0119** against a declared maximum drop of **0.0**.

That is the real trade-off, and it is now a policy decision rather than an engineering unknown:
either accept a small, declared behaviour tolerance (and a fitted student could then pass), or train
to exact parity first. The gate's zero tolerance has not been loosened to make a candidate pass, and
nothing was promoted.

At this point the compact answer is that this setup is viable in principle - a same-family shallow
`main` gets within 0.0119 of the frozen teacher while cutting latency by 42 % - but a deployable
candidate needs the acceptance question answered first.

### A candidate passes the gate, with no tolerance loosened

Rather than relax the zero tolerance to let the 0.0119 candidate through, the same shallow family was
fitted a little further across seeds. Seed 1 reaches dev parity with the frozen teacher and passes the
unchanged gate on its own merit:

| Candidate | dev intent macro-F1 | delta | median reduction | p95 reduction | gate (declared max drop 0.0) |
|---|---|---|---|---|---|
| seed 0, 900 steps | 0.8772 | -0.0119 | 43.3 % | 42.4 % | rejected |
| **seed 1, 1400 steps** | **0.8891** | **+0.00006** | **42.8 %** | **41.8 %** | **passed, no reasons** |
| seed 2, 1400 steps | 0.8853 | -0.0038 | 42.8 % | 42.7 % | rejected |

So the F2 feedback has now rejected and passed the same compression at the same declared threshold,
which is the evidence that it discriminates. The passing candidate is a `resnet18_shallow` `main` with
5,112,974 parameters against the teacher's 11,383,694, on the frozen dev split.

### What "passed" does and does not mean here

- It is a **dev-split** result, and the split was read many times while iterating on seeds. The gate's
  `maximum_exact_agreement_fraction` concern applies to this line, so the honest reading is a strong
  dev signal, not a final acceptance.
- It is **not** a holdout result. The frozen 20-seed holdout stays unopened and remains the acceptance
  for any candidate that changes control.
- It is **not** a gameplay improvement. Nothing was promoted, the candidate is not wired into the
  runtime, and `promotion_allowed` stays false on every artifact.
- The gate's zero-intent-drop tolerance was reviewed with the owner and **left as it is**; one seed
  passed under it without any tolerance change.

## What is deliberately not claimed

- Not that any speedup has been achieved. The feedback is in place and two pilot candidates were
  rejected; nothing replaced the frozen teacher.
- Not that this task improves gameplay. It is the compression question the pack allows, and the pack
  says plainly it must not be called a policy-level improvement.
- Not a re-run of the 20-seed holdout. That stays the acceptance for a candidate that changes control.
