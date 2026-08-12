# HoK-Agent Pixel V4

This repository has one product goal: train a compact visual policy that consumes only
project-owned PixelArena RGB frames and emits the existing six abstract 1v1 actions.
Closed-loop actions run only inside PixelArena. The V4 Shadow path accepts a
privacy-reviewed local recording for offline diagnosis, always abstains on that
unvalidated domain, and never sends automated input.

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

`MINIMAL-V4-SHADOW-READONLY` adds one offline bridge from a local recording to the
frozen V3 RGB Actor. It writes only `predictions.jsonl` and `summary.json`; every
commercial-domain row has `advisory_action=ABSTAIN`, because no real-domain calibration
has been established. It accepts no camera, URI, device node, live stream, or phone
control surface.

```bash
python -m hok_agent shadow-video \
  --input /absolute/path/to/privacy-reviewed-recording.mp4 \
  --model runs/pixel-v3-v1/model-seed-0.safetensors \
  --output-dir runs/shadow-v4-recording-001
```

`MINIMAL-V3-PIXEL-BC` remains complete and frozen. It adds one deterministic renderer, one
ResNet-18-from-scratch model family, one bounded behavior-cloning run, and at most one
DAgger acquisition pass. It does not add PPO, DQN, GRU, Transformer, a general training
framework, multiple archetypes, or 3v3.

The Actor is RGB-only. Structured state is confined to PixelArena, the deterministic
teacher, rendering, and evaluation. Legal actions never enter the model encoder or
`forward`; they are used only for teacher selection, audit, and execution.

The formal command is:

```bash
make install
env -u LD_LIBRARY_PATH .venv/bin/python -m hok_agent accept-pixel-v3 \
  --device cuda \
  --output-dir runs/pixel-v3-v1
```

The host exports an unrelated CUDA 12.0 library path. The command removes that inherited
path so the pinned Torch 2.5.1 CUDA 12.1 wheel resolves only its project-local runtime.

CPU CI uses `python -m hok_agent accept-pixel-v3 --smoke --device cpu`, which is a
non-promoting lifecycle check and not performance evidence.

## Frozen baselines

- Minimal V1: deterministic PixelArena lifecycle, NULL/random/scripted policies, public
  JSONL trace, fresh-process replay, and tamper rejection.
- Minimal V2: a 550-parameter structured MLP that imitates the scripted policy. It
  remains a reproducible teacher/baseline, not the product Actor.

## Maximum claim

The maximum capability claim remains that an RGB-only visual
imitation policy completes fixed abstract 1v1 tasks in the project-owned PixelArena.
V4 additionally proves only that a local recording can be decoded and passed through
that frozen model without any client action output. It does not establish useful real
client advice, Honor of Kings ability, GameCore equivalence, transfer, or automation.

The active source gate is deliberately small: at most 36 project files, 22 Python files,
4,000 Python lines including tests, and these four root Markdown authority files.
