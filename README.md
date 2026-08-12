# HoK-Agent Minimal V2

This repository now has one goal: a small, deterministic, project-owned 1v1 lane
environment called PixelArena. It exposes a local process lifecycle, three non-learning
policies, public JSONL recording, exact action replay, and one CPU behavior-cloning
experiment.

It is not an Honor of Kings implementation, bot, or capability claim. No GameCore is
available or required. No real phone or commercial client is connected by this project.

## Runnable slice

```text
PixelArena state -> legal actions -> NULL/random/scripted action
                 -> local health/reset/step/close service
                 -> public JSONL trace -> fresh-process replay

public pre-action state -> fixed 10 features -> 10x32x6 tanh MLP
                        -> raw prediction -> execution-boundary legal mask
```

The fixed abstract ruleset has two sides, one lane, one tower and one crystal per side.
An episode ends when a crystal is destroyed or the tick limit is reached. Actions are
factorized into macro, type, target, direction, skill, upgrade, and auxiliary fields.
Legal actions are returned at the execution boundary and are not part of observations.

## Commands

```bash
.venv/bin/python -m pip install -e '.[dev,bc]'
make check
.venv/bin/python -m hok_agent accept-minimal-v1 --seed 101
.venv/bin/python -m hok_agent accept-minimal-v2-bc \
  --output-dir runs/minimal-v2-bc-v1
.venv/bin/python -m hok_agent record \
  --blue scripted --red null --seed 101 --output /tmp/blue.jsonl
.venv/bin/python -m hok_agent replay /tmp/blue.jsonl
.venv/bin/python -m hok_agent check
```

The V1 gate runs scripted play from both sides, a seeded random episode, and
fresh-process replay checks. The V2 gate collects 256 scripted-versus-random episodes,
deduplicates and splits public observations, trains three 550-parameter CPU models, and
evaluates raw predictions before applying legal actions only at the execution boundary.
If the output directory exists when the command starts, it refuses to modify it. A run
is built in a sibling temporary directory and then published by one atomic rename with:

- `dataset.jsonl`
- `model-seed-0.json`, `model-seed-1.json`, and `model-seed-2.json`
- `report.json`

The maximum valid claim after both gates pass is:

> Under the fixed project-owned PixelArena rules, a small supervised CPU model can
> imitate the scripted test policy from ten public structured features.

## Why this is intentionally small

The project learned the useful structural lesson from
[ResnetGPT](https://github.com/FengQuanLi/ResnetGPT),
[WZCQ](https://github.com/FengQuanLi/WZCQ), and
[wzry_ai](https://github.com/myBoris/wzry_ai): prove one short end-to-end path before
building a platform. Their real-device control code is specifically not reused.

The mechanical 10x ceiling uses the smallest value observed per metric: 270 files
(WZCQ: 27), 110 Python files and 12,560 Python lines (wzry_ai: 11 and 1,256).
This repository applies a much stricter Minimal V2 budget of 24 files, 15 Python files,
and 1,800 Python lines including tests.

Recurrent RL, multiple abstract archetypes, 3v3, rendered pixels, and a separately
reviewed real-client read-only coach remain future work. None is implemented here.
