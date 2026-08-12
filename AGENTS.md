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

- At most 22 project files, excluding `.git`, `.venv`, caches, and generated output.
- At most 14 Python files and 1,400 total Python lines, including tests.
- At most four root Markdown authority files.
- Python standard library only at runtime; `pytest`, Ruff, and mypy are development tools.

Run `make check` and `python -m hok_agent accept-minimal-v1 --seed 101` before marking
the task complete. Record only observed results in `DELIVERY_PROGRESS.md`.
