# HoK-Agent V5

[![CI](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml/badge.svg)](https://github.com/aka-debug-jie/hok-agent-v5/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

HoK-Agent V5 is an RGB-only MOBA research testbed built around project-owned PixelArena
environments, causal video models, offline Shadow evaluation, and fail-closed mobile-testbed
interfaces. It does not call a game-internal API and is not affiliated with or endorsed by a game
publisher.

The public repository intentionally contains no authorized mobile build identity, calibrated
coordinates, device serial, recording, dataset, checkpoint, or run artifact. Mobile input is
disabled unless the operator supplies local private evidence for a self-built test app and every
runtime guard passes.

## Architecture

```text
V1–V3  frozen deterministic and behavior-cloning regressions
V4     local video/V4L2 -> read-only RGB hypotheses
V5     PixelArena source teacher -> SimSiam -> pseudo labels -> Mean Teacher
V6     RGB-derived tracking and causal temporal diagnostics
V7     Rich PixelArena -> factorized ResNet-18 policy -> PixelArena-only loop
T8     mobile/video demonstrations -> causal policy diagnostics -> Shadow -> bounded gates
```

V5/V6 base training uses no human action, frame, HUD, tracking, or temporal labels. T8 is a
separate lineage and may use only its own standardized observed or actually dispatched testbed
events. Legal actions, cooldown state, and structured truth never enter an RGB Actor.

The visual-policy organization was informed by
[ResnetGPT](https://github.com/FengQuanLi/ResnetGPT),
[WZCQ](https://github.com/FengQuanLi/WZCQ), and
[wzry_ai](https://github.com/myBoris/wzry_ai). This repository does not copy their device-control
code, data, weights, coordinates, assets, or recordings.

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,bc,vision,shadow,preingest]'

make check
make accept
make accept-v2
make pixel-smoke
make shadow-live-smoke
make alignment-smoke
make temporal-smoke
make rich-smoke
```

The CI workflow uses CPU PyTorch. CUDA is required only for formal GPU acceptance and larger local
training runs; a sandbox that cannot see the GPU is not evidence that the host GPU is unavailable.

## External data storage

Large data is never committed. The public default is `.local-data/hok-agent-v5`, which is ignored
by Git:

```bash
make storage-init
make storage-show
```

Override it for another mounted data volume without changing code:

```bash
HOK_LARGE_ROOT=/absolute/private/path/hok-agent-v5 make storage-show
```

The external tree contains `datasets/`, `checkpoints/`, `runs/`, `cache/`, `audit/`, and
`staging/`. Manifests may store artifact basenames and anonymous hashes, never source-video
locators, account identifiers, raw touch dumps, or credentials.

## Read-only video

Offline input must be one non-symlink regular local recording:

```bash
python -m hok_agent shadow-video \
  --input /absolute/path/to/privacy-reviewed-recording.mp4 \
  --model /absolute/private/path/model.safetensors \
  --output-dir .local-data/hok-agent-v5/runs/shadow-offline-001
```

Live capture accepts only one explicitly selected non-symlink `/dev/videoN` V4L2 character
device. Numeric indexes, URLs, network streams, and automatic source selection are rejected.

## Mobile testbed: locked by default

The checked-in examples are intentionally invalid:

- `configs/mobile_testbed_identity.example.json`
- `configs/mobile_testbed_layout.example.json`

Create private local files only for a project-owned self-built app:

```bash
cp configs/mobile_testbed_identity.example.json configs/mobile_testbed_identity.local.json
cp configs/mobile_testbed_layout.example.json configs/mobile_testbed_layout.local.json
```

Fill the package, version, signing identity, APK SHA-256, owner attestation, date, display geometry,
and normalized control positions from your own build. Then calculate the canonical identity hash:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

path = Path("configs/mobile_testbed_identity.local.json")
payload = json.loads(path.read_text(encoding="utf-8"))
payload.pop("identity_sha256", None)
payload["identity_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
```

The default identity path may be overridden with `HOK_MOBILE_IDENTITY_PATH`. Identity and layout
files matching `*.local.json` are ignored by Git.

A read-only dry run still requires an explicit serial and capture source:

```bash
HOK_MOBILE_IDENTITY_PATH=configs/mobile_testbed_identity.local.json \
python -m hok_agent mobile-testbed \
  --serial YOUR_USB_SERIAL \
  --model /absolute/private/path/model.safetensors \
  --layout configs/mobile_testbed_layout.local.json \
  --output-dir .local-data/hok-agent-v5/runs/mobile-dry-run \
  --device cpu --run-seconds 60 --infer-hz 1
```

Device input additionally requires the explicit input flag and a finite action cap. Immediately
before every dispatched event, the runtime rechecks the serial, locally attested package,
foreground window, display identity, and guard watchdog. Missing identity, invalid identity hash,
package/version/signature drift, layout mismatch, disconnection, or backgrounding stops the run.
Accessibility, root, global hooks, HID/UHID, uinput, minitouch, generic shell execution, process or
memory inspection, protocol interception, evasion, and online learning are outside the supported
surface.

See [BOUNDARIES.md](BOUNDARIES.md) before enabling any mobile input.

## T8 status

T8-v2.7 is permanently frozen failed: its reports may be verified but not used to justify more
collection, threshold changes, or another four-class fit.

T8-v3 predicts five RGB-observable states from a V5-initialized ResNet-18 and 16-frame causal TCN:

- `enemy_visible`
- `attack_opportunity`
- `basic_ready`
- `skill1_ready`
- `skill2_ready`
- derived `confidence` and `abstain`

The single seed-0 run did not pass its frozen dev gates. Mean head macro-F1 was `0.450962`, minimum
positive recall was `0.314075`, normal-minus-shuffled margin was `0.023275`, and confidence
coverage was `0.039481`. Offline replay rejected the model; video-test, live Shadow, and all T8-v3
device stages remained unopened. These results are failure evidence, not a released policy.

## Project documents

- [AGENTS.md](AGENTS.md): implementation authority and module constraints.
- [BOUNDARIES.md](BOUNDARIES.md): permitted and forbidden execution surfaces.
- [DELIVERY_PROGRESS.md](DELIVERY_PROGRESS.md): concise current-state ledger.
- [docs/DELIVERY_HISTORY.md](docs/DELIVERY_HISTORY.md): sanitized historical ledger.

## License

Licensed under the [Apache License 2.0](LICENSE).
