# HoK-Agent Minimal V1

This repository now has one goal: a small, deterministic, project-owned 1v1 lane
environment called PixelArena. It exposes a local process lifecycle, three non-learning
policies, public JSONL recording, and exact action replay.

It is not an Honor of Kings implementation, bot, or capability claim. No GameCore is
available or required. No real phone or commercial client is connected by this project.

## Runnable slice

```text
PixelArena state -> legal actions -> NULL/random/scripted action
                 -> local health/reset/step/close service
                 -> public JSONL trace -> fresh-process replay
```

The fixed abstract ruleset has two sides, one lane, one tower and one crystal per side.
An episode ends when a crystal is destroyed or the tick limit is reached. Actions are
factorized into macro, type, target, direction, skill, upgrade, and auxiliary fields.
Legal actions are returned at the execution boundary and are not part of observations.

## Commands

```bash
python -m pip install -e '.[dev]'
make check
python -m hok_agent accept-minimal-v1 --seed 101
python -m hok_agent record --blue scripted --red null --seed 101 --output /tmp/blue.jsonl
python -m hok_agent replay /tmp/blue.jsonl
python -m hok_agent check
```

The acceptance command runs scripted play from both sides, a seeded random episode,
and fresh-process replay checks. Its maximum valid claim is:

> The project-owned abstract PixelArena deterministic 1v1 service, three non-learning
> baselines, and public JSONL replay vertical slice are runnable.

## Why this is intentionally small

The project learned the useful structural lesson from
[ResnetGPT](https://github.com/FengQuanLi/ResnetGPT),
[WZCQ](https://github.com/FengQuanLi/WZCQ), and
[wzry_ai](https://github.com/myBoris/wzry_ai): prove one short end-to-end path before
building a platform. Their real-device control code is specifically not reused.

The mechanical 10x ceiling uses the smallest value observed per metric: 270 files
(WZCQ: 27), 110 Python files and 12,560 Python lines (wzry_ai: 11 and 1,256).
This repository applies a much stricter local budget of 22 files, 14 Python files, and
1,400 Python lines.

Future work is deliberately unordered until this slice passes: a small behavior-cloning
experiment, then recurrent RL, multiple abstract archetypes, 3v3, rendered pixels, and
finally a separately reviewed real-client read-only coach. None is implemented here.
