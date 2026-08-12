# Delivery Progress

- Last update: 2026-08-12
- Current task: `MINIMAL-V3-PIXEL-BC`
- Status: `IN_PROGRESS`
- Product Actor input: `RGB_ONLY`
- Closed-loop environment: `PIXELARENA_ONLY`
- Commercial-client action output: `false`
- HoK capability claim: `false`
- GameCore equivalence claim: `false`

## Goal reset

The project is now a compact visual-Agent extension of ResnetGPT, WZCQ, and wzry_ai.
Their common useful target is visual observation -> trainable policy -> structured action
-> replayable runner. Their device-control implementations and unverified performance
claims are not inherited.

Minimal V1 and V2 are frozen foundations, not the final product. The active target is a
128x128 RGB-only ResNet-18 policy that completes abstract 1v1 episodes inside the
project-owned PixelArena. The first visual delivery permits one behavior-cloning model
family and at most one bounded DAgger pass. PPO, DQN, GRU, Transformer, multi-archetype,
3v3, and real-client implementation are not active tasks.

The V3 source gate is 36 project files, 22 Python files, 4,000 Python lines including
tests, and four root Markdown files.

## Current V3 ledger

- Authority-document reset: `COMPLETE`
- TacticalTeacher with six-action coverage: `NOT_IMPLEMENTED`
- Deterministic RGB renderer: `NOT_IMPLEMENTED`
- Grouped pixel dataset: `NOT_IMPLEMENTED`
- ResNet-18 pixel BC: `NOT_IMPLEMENTED`
- Bounded DAgger: `NOT_IMPLEMENTED`
- CPU smoke: `NOT_RUN`
- RTX 4090 formal acceptance: `NOT_RUN`
- V3 run directory: `NOT_CREATED`
- V3 commit: `NOT_CREATED`

No V3 capability claim is currently allowed.

## Frozen baseline evidence

### Minimal V1

- Commit: `4267c202f2205237fc48e889a9ddd2b31c7f632d`
- Acceptance: `PASSED` at seed 101
- Scripted blue/red each destroyed the opposing crystal in 12 ticks.
- Seeded random reached the 32-tick limit.
- All traces replayed in fresh processes and a modified hash was rejected.

### Minimal V2

- Commit: `c61fb47a994bf5d584be06ba46e43f0c6550a4df`
- Acceptance: `PASSED`
- Dataset: 256 episodes, 3,232 raw transitions, 495 unique public observations,
  zero label conflicts, and split counts 345/74/76.
- Exact top-1 across seeds 0/1/2: 98.684%, 100%, 100%.
- Best structured model: seed 2; blue/red each destroyed the opposing crystal in
  12 ticks with zero raw illegal actions and zero mask corrections.
- Run directory: `runs/minimal-v2-bc-v1` (Git ignored).

These results prove only deterministic simulator and structured imitation behavior. They
do not prove visual perception, strategy superiority, Honor of Kings, GameCore, transfer,
or real-client capability.

## Required V3 evidence

The final ledger must record real commands and results for static gates, V1/V2
regressions, CPU pixel smoke, six-class grouped data, all three formal training seeds,
sealed-test controls, matched closed-loop evaluation, CUDA latency, artifact hashes,
DAgger disposition, and the final commit. A failed formal run remains `FAILED`; it does
not unlock PPO or permit threshold changes.
