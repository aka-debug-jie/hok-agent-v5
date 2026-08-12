# AGENTS.md

`DELIVERY_PROGRESS.md` is the only current-state ledger. Read it, `README.md`, and
`BOUNDARIES.md` before changing this repository.

## Active implementation rules

- Build only `MINIMAL-V3-PIXEL-BC`: one renderer, one ResNet-18 model family, one
  bounded BC run, and at most one DAgger acquisition pass.
- Preserve Minimal V1 and V2 as secondary regression baselines. Do not present the
  structured MLP as the product Actor.
- The V3 Actor is RGB-only. Legal actions and structured observations must not enter
  its `forward` method.
- Closed-loop actions are allowed only inside project-owned PixelArena. The commercial
  client remains read-only and outside this delivery.
- Do not add PPO, DQN, recurrent/Transformer policy code, a model registry, distributed
  training, multiple archetypes, 3v3, device input, hooks, protocol access, or evasion.
- Do not copy source, weights, data, coordinates, assets, or device setup from reference
  projects.

## Mechanical gates

- At most 36 project files, excluding `.git`, `.venv`, caches, and ignored run outputs.
- At most 22 Python files and 4,000 total Python lines including tests.
- Exactly four root Markdown authority files.
- Base PixelArena import remains free of Torch/torchvision. ML imports are allowed only
  in the structured BC module, the pixel training module, and their focused tests.
- `torchvision` and `safetensors` are permitted only for the pixel model path. Device,
  network, shell-runner, and client-control imports remain denied.

Before completion, run `make check`, both frozen baseline gates, CPU pixel smoke, the
pinned CUDA formal acceptance, and `git diff --check`. Record only observed results and
hashes in `DELIVERY_PROGRESS.md`.
