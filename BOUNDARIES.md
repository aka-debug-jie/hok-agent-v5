# Boundaries

## Allowed now

- Deterministic closed-loop actions inside the project-owned PixelArena process.
- Abstract 1v1 rules, NULL/random/scripted test policies, public trace recording,
  replay, tests, and local static checks.
- Standard-library local multiprocessing with no network or device capability.

## Never part of this executable project

- Phone or commercial-client input by ADB, scrcpy control, minitouch, HID/UHID,
  uinput, Accessibility, macros, or mechanical input.
- Root, hooks, injection, memory/process inspection, protocol interception or changes,
  anti-cheat detection, evasion, account automation, or automated real matches.
- Online learning, reward adaptation, or action execution against a commercial client.
- GameCore assumptions, license probing, binary download, or external authorization gates.

A future Shadow Coach may consume user-provided, privacy-reviewed video or replay data
and emit advice on the host only. It must remain separately reviewed and read-only.

## Data and claims

Actor-facing observations contain public abstract state only. Legal actions are separate.
The Minimal V1 trace contains actions, public state hashes, public events, terminal state,
and outcome; it contains no account, device, image, teacher, truth, private reward, or
training identifier.

All outputs use `claim_scope=pixelarena_engineering`, with HoK capability and GameCore
equivalence explicitly false. Results cannot be transferred across those scopes.

The three public reference repositories are design references only. No source, weights,
coordinates, action maps, assets, or device setup are copied. In particular, the absence
of an explicit license in a reference repository is treated as no permission to reuse it.
