# Delivery Progress

- Last update: 2026-08-12
- Current task: `MINIMAL-V4-SHADOW-READONLY`
- Status: `COMPLETE`
- Product Actor input: `RGB_ONLY`
- Closed-loop environment: `PIXELARENA_ONLY`
- Commercial-client action output: `false`
- HoK capability claim: `false`
- GameCore equivalence claim: `false`
- Next task: privacy-reviewed recording validation; no direct phone connection is required

## Delivered route

The project is now a compact visual-Agent extension of ResnetGPT, WZCQ, and wzry_ai:
project-owned PixelArena RGB -> ResNet-18 -> six abstract actions -> PixelArena step.
Their device-control code and capability claims are not inherited. Minimal V1 and V2
remain frozen simulator/structured baselines; the product Actor is the V3 RGB path.

V3 contains a deterministic 128x128 renderer, TacticalTeacher, grouped public pixel
dataset, three-seed ResNet-18 behavior cloning, a sealed negative-control/test gate,
an execution-boundary legal filter, and at most one acquisition-only DAgger round. This
delivery adds no PPO, recurrent model, Transformer, general RL framework, multi-archetype,
3v3, GameCore, or commercial-client control path.

## V4 read-only Shadow status

V4 adds one narrow offline path: a non-symlink local recording is opened read-only,
decoded by PyAV 15.1.0, resized to the frozen 128x128 RGB contract, and passed to the
strictly loaded V3 safetensors model. It publishes exactly `predictions.jsonl` and
`summary.json` through a same-parent atomic rename. No input path, frame, image, audio,
or video is persisted.

The decoder binds each suffix to a fixed container demuxer and passes an empty nested
protocol whitelist. Input and model are opened once with `O_NOFOLLOW`, verified by
`fstat`, and reused through checked descriptors. The promoted V3 SHA-256 is mandatory,
and sampling is bounded to 30-frame stride and 300 retained frames.

Commercial-domain predictions are explicitly raw model hypotheses. Every row has
`advisory_action=ABSTAIN`, reason `UNVALIDATED_COMMERCIAL_DOMAIN`, and
`control_output=false`; the summary fixes real-domain validation, promotion eligibility,
commercial-client action output, and HoK capability to false. URI, numeric camera,
`/dev`, directory, symlink, and FIFO inputs are rejected before decoding.

Observed V4 commands and results:

```text
pip install -e '.[dev,bc,vision,shadow]'        -> PASSED (PyAV 15.1.0)
make check                                      -> PASSED (45 tests)
make accept                                     -> PASSED
accept-minimal-v2-bc                            -> PASSED
make pixel-smoke                                -> PASSED, non-promoting
shadow-video generated local MP4, CPU, 3 frames -> PASSED, 3 ABSTAIN / 0 advice
git diff --check                                -> PASSED
```

- Project gate: 30 files, 21 Python files, 3,969 Python lines, four root Markdown files
- Generated source SHA-256: `cae2a16164b1fbfdbdabf157bbe4f312cf4bdd838b6660147724c4a50a7ebe9b`
- Frozen model SHA-256: `df511e9b19327886da359400055dcc99aad6520a495c6d5e0495031c86b44eed`
- Predictions SHA-256: `82fbedd5056a4084941eac42294415bb60e6ca11bc0dd0f7233f01e43f7cbbef`
- Summary SHA-256: `431461b877292feeb20821ef69fa92ec6c35d33619484562ae6958ce6222047a`
- V4 feature commit: the commit containing this ledger; resolve with `git rev-parse HEAD`

The V4 smoke used generated local pixels, not a phone or commercial-client recording.
It proves only decoder/model/output isolation. A privacy-reviewed recording is the next
input needed to measure domain mismatch; even then, this route remains read-only and
cannot emit or execute a commercial-client action.

## Final V3 status

- Authority-document reset: `COMPLETE` in commit `1f7d034`
- TacticalTeacher and deterministic RGB renderer: `COMPLETE`
- Grouped public pixel dataset: `COMPLETE`
- Three-seed ResNet-18 behavior cloning: `COMPLETE`
- DAgger disposition: `NOT_TRIGGERED` (`0` rounds)
- CPU smoke: `PASSED`, non-promoting
- RTX 4090 formal acceptance: `PASSED`
- V3 run: `runs/pixel-v3-v1` (Git ignored)
- Failed precursor retained: `runs/pixel-v3-v1-failed-closed-loop-20260812`
- Superseded pre-mirror-fix run retained: `runs/pixel-v3-v1-pre-mirror-fix-20260812`
- V3 feature commit: the commit containing this ledger; resolve with `git rev-parse HEAD`

## Observed acceptance evidence

Commands executed:

```text
make check
make accept
make accept-v2
make pixel-smoke
env -u LD_LIBRARY_PATH .venv/bin/python -m hok_agent accept-pixel-v3 \
  --device cuda --output-dir runs/pixel-v3-v1
git diff --check
```

The host exports an unrelated CUDA 12.0 library path, so the formal command removed that
inherited path and used only the pinned project runtime. Observed runtime: Python 3.11.15,
Torch 2.5.1+cu121, torchvision 0.20.1+cu121, CUDA runtime 12.1, NVIDIA GeForce RTX 4090.

Static and regression results:

- Ruff: `PASSED`
- strict mypy: `PASSED`
- pytest: `41 passed`
- project gate: 28 files, 19 Python files, 3,607 Python lines, four root Markdown files
- Minimal V1: `PASSED`; both scripted sides won in 12 ticks, random terminated at 32
- Minimal V2: `PASSED`; test exact accuracy 98.684% / 100% / 100%, closed loop passed
- Pixel CPU smoke: `PASSED`; 151 samples, one seed/epoch, non-promoting

Formal data:

- 256/256 episodes terminal; teacher crystal completion 98.828%
- 10,616 RGB samples and 256 trajectory groups
- split samples: fit 5,948; acquisition 1,494; validation 1,506; test 1,668
- every split contained all six actions with at least 70 samples per action
- dataset fields are only frames, action, group, tick, render seed, split, frame hash, source

Formal training and sealed test:

| seed | best epoch | exact top-1 | balanced accuracy | minimum recall | raw illegal |
|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 99.760% | 99.591% | 97.778% | 0.120% |
| 1 | 19 | 99.880% | 99.630% | 97.778% | 0.000% |
| 2 | 21 | 99.700% | 99.572% | 97.778% | 0.180% |

- promoted model: seed 0, selected only by validation cross-entropy
- black-frame exact accuracy: 8.633%; mismatched-frame exact accuracy: 0%; both failed
  the main classification gate
- acquisition actor/teacher completion: 97.222% / 97.222%; correction rate 0%; no DAgger
- all three models: NULL completion 100%, matched random completion 100%, blue/red 100%/100%
- closed-loop raw illegal/correction rates: 0.000%, 0.156%, 0.039%; executed illegal: 0
- promoted RTX 4090 forward-only batch-1 FP32 p95: 1.932 ms

Formal artifacts and SHA-256:

| file | SHA-256 |
|---|---|
| `dataset.npz` | `e516ce312c02df43194274bd581eef07f6eb7f001d6061b7a34ff72fd9fe806f` |
| `model-seed-0.safetensors` | `df511e9b19327886da359400055dcc99aad6520a495c6d5e0495031c86b44eed` |
| `model-seed-1.safetensors` | `692bece1377df3c0a925ab9bd628487b2e0f6579d765a0b9a19d978040fcbe67` |
| `model-seed-2.safetensors` | `7e20245e7b25a8d5194afe9df6fcf978e350a8a58b24b3906603f851f21ed465` |
| `report.json` | `005b6ec0d405ce1f409fb4e000f95620bf46e4f15aa5f1a6c046d9ab387992a7` |

The successful run directory contains exactly those five files. The dataset reload with
`allow_pickle=False`, frame hashes, split constraints, all three safetensors metadata,
tensor shapes, and file hashes were verified after publication.

## Failure record and claim boundary

The first formal attempt failed honestly at the closed-loop gate because seed 2 completed
63/64 NULL episodes. Its `FAILED` report was retained. Renderer color variation was then
restricted to contrast-preserving non-semantic bands; no model, data-count, training, or
acceptance threshold was reduced. A fresh three-seed run passed all frozen gates.
A final regression then exposed one-pixel midpoint rounding asymmetry in the red-side
self-view. The coordinate normalization and its direct blue/red equality test were fixed;
the pre-fix success was superseded and another fresh three-seed run produced the hashes
and metrics above.

Maximum claim: an RGB-only visual imitation Agent completes the fixed abstract 1v1 task
inside this project-owned PixelArena. This does not establish Honor of Kings, GameCore,
commercial-client automation, transfer, generalization outside the fixed rules, or real
client capability.
