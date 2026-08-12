# AGENTS.md

`DELIVERY_PROGRESS.md` is the only current-state ledger. Read it, `README.md`, and
`BOUNDARIES.md` before changing this repository.

## Implementation rules

- Build only the smallest runnable slice described in the ledger.
- Closed-loop actions are allowed only inside the project-owned PixelArena.
- The commercial client is outside the executable surface. Never add device input,
  client control, account automation, hooks, protocol access, or evasion features.
- GameCore is unavailable and is not a dependency or active integration track.
- Use abstract archetypes and rules. Do not copy game heroes, skills, values, assets,
  code, weights, action maps, or device configuration from reference projects.
- Keep legal actions separate from actor-facing observations. Persist public replay
  data only; do not introduce privileged, teacher, truth, or training-only fields.
- Baselines are test drivers, not learned agents. Do not claim HoK ability, training
  readiness, win-rate performance, or transfer to a real client.

## Size gates

- Minimal V2: at most 24 project files, excluding `.git`, `.venv`, caches, and output.
- Minimal V2: at most 15 Python files and 1,800 total Python lines, including tests.
- At most four root Markdown authority files.
- Base PixelArena is standard-library only. Torch is optional and may appear only in
  `bc.py` and its focused test; V2 is CPU-only.

Run `make check`, both minimal acceptance commands, and record only observed results in
`DELIVERY_PROGRESS.md` before marking the task complete.
