# Boundaries

## Allowed now

- Deterministic closed-loop actions inside the project-owned PixelArena process.
- Abstract 1v1 rules, NULL/random/scripted test policies, public trace recording,
  replay, tests, and local static checks.
- CPU-only supervised imitation of the scripted policy from ten public structured
  PixelArena features. Torch is optional and isolated to the BC module.
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

Actor-facing observations contain public abstract state only. Legal actions are separate
from the model input and are consulted only during evaluation audit or execution. The
Minimal V1 trace contains actions, public state hashes, public events, terminal state,
and outcome. The Minimal V2 dataset contains pre-action public observations and executed
scripted actions. Neither contains legal-action sets, accounts, devices, images, rewards,
teacher/truth state, privileged state, or training-only entity identifiers.

All outputs use `claim_scope=pixelarena_engineering`, with HoK capability and GameCore
equivalence explicitly false. V2 demonstrates supervised imitation only under the fixed
PixelArena rules; it does not establish environment-seed generalization, policy
superiority, RL readiness, transfer, or real-client capability.

The three public reference repositories are design references only. No source, weights,
coordinates, action maps, assets, or device setup are copied. In particular, the absence
of an explicit license in a reference repository is treated as no permission to reuse it.
