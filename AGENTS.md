# AGENTS.md

`DELIVERY_PROGRESS.md` is the only current-state ledger. Read it, `README.md`, and
`BOUNDARIES.md` before changing this repository.

## Active route

- V4: read a privacy-reviewed local recording or an explicitly selected Linux V4L2 UVC
  capture node and emit host-side JSON/terminal hypotheses. Never emit device input.
- V5: train a separate causal PixelArena source teacher, then run unlabeled real-video
  SimSiam adaptation, conservative pseudo-label filtering, and one Mean Teacher round.
- V6: add RGB-derived dual-hero/HUD tracking and a causal eight-frame temporal coach.
- V7: add an independent Rich PixelArena identity with 2D movement and factorized skills.
- Preserve V1/V2/V3 and the offline V4 route as frozen regressions. Never overwrite their
  configs, renderers, models, hashes, schemas, or run evidence.

## Non-negotiable boundaries

- Automatic closed-loop actions run only inside project-owned PixelArena.
- A commercial client is read-only video input. A host-side suggestion is never a phone,
  client, account, protocol, or operating-system input event.
- Never add ADB, minitouch, scrcpy control, HID/UHID, uinput, Accessibility, root, hooks,
  injection, memory/process inspection, protocol interception, evasion, or real-match
  automation. Do not add a generic shell runner or network capture input.
- Live capture accepts only an explicit, non-symlink `/dev/videoN` Linux V4L2 character
  device. Numeric camera indexes, other device nodes, URIs, and network streams fail closed.
- Real-domain training uses no action labels. Human action labels are sealed audit evidence
  only and must not affect weights, thresholds, or hyperparameters. V6 may use the frozen
  300-keyframe non-action tracking labels described in `README.md`.
- RGB Actors receive RGB only. Legal actions and structured state may be used by a
  PixelArena teacher or execution boundary, never by an encoder, temporal hidden state, or
  commercial-domain input.
- Do not copy code, weights, action maps, coordinates, assets, recordings, or device setup
  from the three reference repositories.

## Dependency and module allowlist

- Base PixelArena imports remain free of Torch, torchvision, safetensors, PyAV, OpenCV,
  Tk, and device APIs; CLI imports optional stages lazily.
- Torch/torchvision/safetensors are allowed only in `bc.py`, `pixel.py`, `alignment.py`,
  `temporal.py`, `rich_pixel.py`, and their focused tests.
- PyAV is allowed only in `shadow.py`, `capture.py`, `alignment.py`, and focused tests.
- Tk/Pillow annotation UI is allowed only in `alignment.py`; it must be offline and must
  never open a capture node or feed audit labels into training.
- CUDA is a formal V3/V5/V6/V7 acceptance surface; CPU CI smoke is non-promoting. A sandbox
  GPU probe failure is not evidence that the host RTX 4090 is unavailable.

## Mechanical gates

- At most 48 project files, excluding `.git`, `.venv`, caches, and ignored run outputs.
- At most 32 Python files and 9,000 Python source lines excluding blank separator lines.
- Exactly four root Markdown authority files.
- Keep the outer reference-size gate at 270 files, 110 Python files, and 12,560 Python source
  lines using the same blank-line exclusion.
- Before each commit run Ruff, strict mypy, full pytest, the project size/safety gate,
  relevant frozen regression gates, the stage CPU smoke, and `git diff --check`.
- Never manufacture GPU, hardware, recording, annotation, or accuracy evidence. Missing UVC
  hardware is `READY_FOR_HARDWARE`; missing recordings/labels remain explicit external waits.
