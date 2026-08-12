# HoK-Agent Compact Pixel Route

This project is a safe, compact extension of the visual-policy shape demonstrated by
ResnetGPT, WZCQ, and wzry_ai. It keeps the useful chain—pixels to a trainable policy to a
structured action—but does not copy their device-control code, data, weights, coordinates,
or assets. Automatic actions exist only in project-owned PixelArena.

```text
V4  local video/UVC -> frozen RGB hypothesis -> terminal + JSON (no control)
V5  causal source teacher -> SimSiam -> filtered pseudo labels -> one Mean Teacher round
V6  RGB hero/HUD tracking -> causal 8-frame TCN -> stable advice or ABSTAIN
V7  Rich PixelArena RGB -> factorized ResNet-18 policy -> PixelArena-only closed loop
```

V1 deterministic traces, V2 structured BC, V3 six-class RGB BC, and V4 offline Shadow are
frozen regression baselines. No PPO, general RL framework, GRU, Transformer, distributed
trainer, GameCore adapter, or real-client action surface is part of this route.

## Commands

Frozen gates:

```bash
make check
make accept
make accept-v2
make pixel-smoke
```

Offline V4:

```bash
python -m hok_agent shadow-video \
  --input /absolute/path/to/privacy-reviewed-recording.mp4 \
  --model runs/pixel-v3-v1/model-seed-0.safetensors \
  --output-dir runs/shadow-offline-001
```

Live V4 (only an explicitly selected V4L2 capture node):

```bash
python -m hok_agent shadow-live \
  --input /dev/video10 \
  --model runs/pixel-v3-v1/model-seed-0.safetensors \
  --output-dir runs/shadow-live-001 \
  --device cuda --capture-size 1920x1080 --capture-fps 60 --infer-hz 10
```

V5/V6 stage commands operate on privacy-masked session manifests and Git-ignored run
directories. They fail closed until the required 12 sessions, sealed audit, and tracking
labels exist. V7 has an independent CPU smoke and a separate CUDA acceptance command; it
never changes the six-class Shadow vocabulary.

## Frozen V5/V6 evidence contract

- Real-video action training uses zero human action labels.
- At least 12 independent sessions are split before processing: at least 8 train, 2 dev,
  and 2 sealed test. Re-encodes, overlapping clips, and near duplicates stay in one group.
- The V5 source teacher removes tick-progress shortcuts and uses only visible public state.
- SimSiam adapts shallow encoder layers; pseudo labels require source/student, view, temporal,
  and OOD agreement; Mean Teacher runs once.
- Two blinded annotators audit 500 clips: 300 for V5 frame advice and 200 reserved for V6.
  Audit labels produce evidence and `release.json` only; they never train or tune the model.
- V6 may use 300 session-isolated keyframes for hero centers/visibility and HUD status, but
  these are not action labels.

## Rich PixelArena V2

Rich V2 is an independent `pixelarena-rich-1v1-v2` 15x7 single-lane simulator with eight
movement directions, minions, tower/crystal objectives, respawn, basic attack, directional
dash/projectile skills, and one target skill. It uses snapshot-based simultaneous resolution
and a dedicated renderer/model hash. Its Actor predicts factor heads; legal domains are
consulted only at execution. V1/V3 renderers, models, hashes, and traces remain byte-stable.

## Maximum claim

The maximum automated capability claim is RGB-only completion of fixed abstract 1v1 tasks
inside project-owned PixelArena. Real-client video remains read-only. A passed sealed audit
may support only the released abstract host-side advice classes, never Honor of Kings skill,
optimal play, GameCore equivalence, real-client control, or unmeasured transfer.
