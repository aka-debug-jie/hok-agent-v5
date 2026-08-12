# AGENTS.md

`DELIVERY_PROGRESS.md` is the only current-state ledger. Read it, `README.md`, and
`BOUNDARIES.md` before changing this repository.

## Active implementation rules

- Build only `MINIMAL-V4-SHADOW-READONLY`: one offline local-video decoder, the frozen
  V3 RGB model, and JSON-only diagnostic output that always abstains on unvalidated
  commercial-client footage.
- Preserve Minimal V1 and V2 as secondary regression baselines. Do not present the
  structured MLP as the product Actor; preserve V3 as the frozen RGB Actor.
- The V3 Actor is RGB-only. Legal actions and structured observations must not enter
  its `forward` method.
- Closed-loop actions are allowed only inside project-owned PixelArena. V4 may read only
  a non-symlink local recording; it accepts no URI, camera/device node, or live stream.
- Do not add PPO, DQN, recurrent/Transformer policy code, a model registry, distributed
  training, multiple archetypes, 3v3, device input, hooks, protocol access, or evasion.
- Do not copy source, weights, data, coordinates, assets, or device setup from reference
  projects.

## Mechanical gates

- At most 36 project files, excluding `.git`, `.venv`, caches, and ignored run outputs.
- At most 22 Python files and 4,000 total Python lines including tests.
- Exactly four root Markdown authority files.
- Base PixelArena import remains free of Torch/torchvision. ML imports are allowed only
  in the structured BC module, the pixel module, and their focused tests. PyAV is allowed
  only in the Shadow module and its focused test.
- `torchvision` and `safetensors` are permitted only for the pixel model path. Device,
  network, shell-runner, and client-control imports remain denied.

Before completion, run `make check`, both frozen baseline gates, CPU pixel smoke, a
generated-local-video Shadow smoke, and `git diff --check`. Record only observed results
and hashes in `DELIVERY_PROGRESS.md`.
