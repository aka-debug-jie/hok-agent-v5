# Boundaries

## Allowed execution surfaces

- Deterministic structured and RGB closed-loop actions inside the project-owned
  PixelArena process.
- A 128x128 public renderer, scripted/tactical teachers, RGB-only policy training,
  offline evaluation, replay, and bounded DAgger inside PixelArena.
- CPU CI smoke and a pinned single-GPU formal training run on the local RTX 4090.
- A future, separately reviewed Shadow Coach that consumes user-provided or
  privacy-reviewed commercial-client video and emits advice on the host only.

## Never part of the executable action surface

- Commercial-client or phone input through ADB, scrcpy control, minitouch, HID/UHID,
  uinput, Accessibility, macros, mechanical input, or account automation.
- Root, hooks, injection, memory/process inspection, protocol interception or changes,
  anti-cheat detection/evasion, or automated real matches.
- Online learning, reward adaptation, policy promotion, or action execution against a
  commercial client.
- GameCore assumptions, license probing, unknown binary/weight downloads, or external
  authorization gates.

## Actor, data, and legal actions

The V3 Actor accepts only RGB tensors. It receives no public-state dictionary, legal
mask, reward, teacher identity, truth, privileged value, account, device, or training
entity identifier. Structured public state may be used before rendering and in the
teacher/evaluator, but it is not an Actor input.

The persisted pixel dataset may contain uint8 frames, executed action labels, episode
group hashes, ticks, render seeds, split/source codes, and frame hashes. It must not
persist legal sets, rewards, raw structured observations, truth, or privileged state.
Legal actions are transient and may be used only for teacher choice, loss-side audit,
and the final execution filter. Raw model predictions and filter corrections are
reported separately.

## Claims and reference isolation

All outputs remain within a PixelArena engineering scope, with HoK capability,
GameCore equivalence, transfer, and commercial-client control explicitly false. The
three reference repositories are architectural references only. No source, weights,
coordinates, action maps, screenshots, assets, device identifiers, or control setup are
copied into this project.
