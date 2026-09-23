# B: the interface decision for a truth source (not built)

Status: `AWAITING_OWNER_SOURCE`. Nothing here is implemented, and no adapter is written until the owner
names a source that is actually obtainable. This document exists so the choice is concrete.

## Why this document exists

Every remaining question about the main objective - whether the cold start is really impassable,
whether a transition's terminal is real, whether the mask helps or hurts - needs one thing the project
does not have: a **truth that does not come from the same RGB detector the chain already uses**. The R0
contract forbids letting one unvalidated detector both author a reward and prove its own accuracy, and
the A1 feasibility check already established that no such reference can be derived from the same
captured pixels (the minimap is a fixed full map with no second indicator, and the camera view carries
no recoverable rigid translation).

So the source has to be external. The only question is which external source is genuinely obtainable,
and that is the owner's call, not something an engineer may assume. This document sets out what each
candidate source would have to supply, so naming one is a decision with a known shape.

## What the source must supply

Any usable source must answer, for a given run, at least one of these with an authority independent of
the RGB cue:

| # | Fact | Why it is needed | Independent because |
|---|---|---|---|
| F1 | hero position in a fixed frame, per sampled instant | to score localization without the cue grading itself | it comes from game state, not from reading the screen |
| F2 | match terminal: win / loss / draw, and the ending tick | to tell a real terminal from a timeout | it is decided by the game, not inferred from pixels |
| F3 | structure state: tower/crystal health, per instant | to score objective progress independently | same |
| F4 | whether the hero is alive, and the death/recovery moments | to validate the death confirmation policy | same |

The minimum viable source is **F1 plus F2**. F3 and F4 sharpen conclusions but are not required to
break the deadlock.

## The three candidate sources, and what each must provide

### B-1 - app-side export (best fit if available)

The identity file attests the testbed app is owner-self-built (`owner_attested_self_built: true`,
`package` `com.tencent.tmgp.sgame`, base APK sha `cea8693d33e6...`, signature `9a7b53ca`), so an
export is plausible in principle. To design a read-only importer, the owner must provide:

| Needed | Example shape (illustrative, not assumed) |
|---|---|
| the on-device path the app writes | e.g. `/sdcard/.../state.jsonl` |
| the schema: field names and units | position units and frame, tick units, terminal vocabulary |
| a one-line real sample | so no field name is invented |
| the cadence | one row per game tick, per frame, or on change |

I would then write an importer that **reads** that file, binds each row to an observation by timestamp,
and refuses anything the schema does not declare. No adapter is written first.

### B-2 - the app source tree

If the owner holds the source, providing its location lets me read the real export surface instead of
guessing, then define the interface from what the app can actually emit.

### B-3 - a ground-truth capture under a controlled scenario

If a known-truth scenario can be run - a hero placed at declared coordinates, or a scripted stationary
period, or a scripted terminal - the owner must provide the **procedure** and the **truth values**. I
would turn it into a labelled fixture and use it to check the detector, not to train on.

## What becomes provable, and what does not

Even with the best of these, the honest gains are bounded:

| Would become provable | Would still not be provable |
|---|---|
| whether the cue-localised position is accurate, against an external frame | whether the policy is good, on its own |
| whether a recorded terminal was real or a timeout | a match-level win rate, from one or few runs |
| whether the cold start is genuinely impassable, not just impassable to three mechanisms | full-match competence, which needs a scenario with departure, combat, death/recovery and a terminal |
| whether the mask's removals cost progress, scored against F1/F3 | that the candidate compression improves play |

## What I will not do without a named source

- write a stub importer, or define a schema for a source that does not exist;
- treat a matching package name, or a self-built attestation, as proof a source is obtainable;
- fill a missing field with a guess and later describe the condition as met.

## The acceptance is declared, and nothing is built

The acceptance a source would be judged by reuses the existing R0 gates rather than redefining them,
and is recorded as a proposal at
[`game_rules/truth_source_acceptance_v1.json`](../../../game_rules/truth_source_acceptance_v1.json)
(`status: PROPOSED_NOT_IMPLEMENTED`). Its duplication guard is byte-equal to the R0 contract's, and its
`implementation_gate.may_start_only_after` is the owner's declaration below. So the contract exists, the
importer does not, and the two cannot be confused.

## The decision requested

One line from the owner is enough:

```text
source: B-1 | B-2 | B-3
what it is: <path / repo / procedure>
what it provides: F1 and/or F2 (and optionally F3, F4)
how it is obtained: <command or step>
```

With that, the next step is a minimal read-only importer plus a paired check against the frozen cue -
which is the first thing in this project that could score the detector without the detector scoring
itself.
