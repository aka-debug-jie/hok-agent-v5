# HoK-Agent Pixel V3

This repository has one product goal: train a compact visual policy that consumes only
project-owned PixelArena RGB frames and emits the existing six abstract 1v1 actions.
Closed-loop actions run only inside PixelArena. A commercial client may later provide
read-only video for a separately reviewed coach, but it never receives automated input.

```text
PixelArena public state -> deterministic 128x128 RGB renderer
                         -> ResNet-18 visual policy -> six action logits
                         -> execution-boundary legal filter -> PixelArena step
```

The design extends the useful common shape of
[ResnetGPT](https://github.com/FengQuanLi/ResnetGPT),
[WZCQ](https://github.com/FengQuanLi/WZCQ), and
[wzry_ai](https://github.com/myBoris/wzry_ai): visual observation, trainable policy,
structured action, and replayable runner. Their device-control code, coordinates,
weights, assets, and platform-specific setup are not reused.

## Active delivery

`MINIMAL-V3-PIXEL-BC` is in progress. It adds one deterministic renderer, one
ResNet-18-from-scratch model family, one bounded behavior-cloning run, and at most one
DAgger acquisition pass. It does not add PPO, DQN, GRU, Transformer, a general training
framework, multiple archetypes, or 3v3.

The Actor is RGB-only. Structured state is confined to PixelArena, the deterministic
teacher, rendering, and evaluation. Legal actions never enter the model encoder or
`forward`; they are used only for teacher selection, audit, and execution.

The formal command will be:

```bash
python -m hok_agent accept-pixel-v3 \
  --device cuda \
  --output-dir runs/pixel-v3-v1
```

CPU CI uses `python -m hok_agent accept-pixel-v3 --smoke --device cpu`, which is a
non-promoting lifecycle check and not performance evidence.

## Frozen baselines

- Minimal V1: deterministic PixelArena lifecycle, NULL/random/scripted policies, public
  JSONL trace, fresh-process replay, and tamper rejection.
- Minimal V2: a 550-parameter structured MLP that imitates the scripted policy. It
  remains a reproducible teacher/baseline, not the product Actor.

## Maximum claim

After the V3 formal gate actually passes, the maximum claim is that an RGB-only visual
imitation policy completes fixed abstract 1v1 tasks in the project-owned PixelArena.
No result establishes Honor of Kings, GameCore, commercial-client control, transfer,
or environment-external capability.

The active source gate is deliberately small: at most 36 project files, 22 Python files,
4,000 Python lines including tests, and these four root Markdown authority files.
