# Boundaries

## Allowed execution surfaces

- Deterministic structured and RGB closed loops inside project-owned PixelArena V1 and the
  independently versioned Rich PixelArena V2.
- Offline analysis of a non-symlink regular local recording.
- Read-only capture from one explicitly named, non-symlink `/dev/videoN` Linux V4L2
  character device through PyAV. Capture is latest-frame-only, bounded, local, and emits
  terminal/JSON diagnostics on the host.
- Offline synthetic training, unlabeled real-video representation learning, conservative
  pseudo-label research, one Mean Teacher round, and RGB-derived tracking/temporal advice.
- A sealed human audit may unlock individual abstract advice classes. Audit labels never
  train the model or tune thresholds. V6 tracking alone may use 300 non-action keyframes.

## Never part of the executable surface

- Any commercial-client or phone input through ADB, scrcpy control, minitouch, HID/UHID,
  uinput, Accessibility, macros, mechanical input, or account automation.
- Root, hooks, injection, process/memory access, protocol interception/change, anti-cheat
  detection/evasion, automated real matches, or any attempt to conceal automation.
- Numeric camera indexes, arbitrary device nodes, symlinked devices, URLs, RTSP/TCP/UDP or
  other network streams, generic shell commands, device enumeration, or automatic source
  selection.
- Online learning from a commercial client, reward adaptation, real-client policy
  promotion, or mapping Rich PixelArena skills to a real-client control surface.
- GameCore assumptions, license probing, unknown binaries/weights, or credential storage.

## Data, Actor, and output boundary

Commercial video is read-only. V4 writes only bounded JSONL events and a summary; it never
stores raw frames, thumbnails, audio, or video. Privacy-masked 128x128 derived frames may be
stored only in Git-ignored, session-partitioned NPZ shards for V5/V6. Original recordings
stay outside the repository. Persisted real-domain records must not contain paths, account
identifiers, legal masks, rewards, structured state, truth, or privileged fields.

New large derived datasets, caches, checkpoints, audit media, and formal training runs live
under the external `HOK_LARGE_ROOT`; the default is
`/media/hgdl1012/E/wzry-data/hok-agent-v5`. Manifests may contain only artifact basenames and
anonymous hashes, never raw-video locators. Repository symlinks are not a substitute for
strict regular-file loading. Existing frozen local run evidence is preserved in place.

V3/V5/V6 Actors accept RGB tensors or RGB sequences only. Tracking values used by V6 must
be produced internally from RGB. A caller cannot supply legal actions, structured state,
teacher identity, reward, truth, account, or device state. PixelArena legal domains are
transient teacher/execution-boundary data and never enter an encoder or hidden state.

Before a V5 release passes its sealed audit, every real-domain row has
`advisory_action=ABSTAIN` and `control_output=false`. After an overall audit passes, only
classes listed in a hash-bound `release.json` may appear as host-side advice; failed classes
remain `ABSTAIN`. Advice is an imitative abstract hypothesis, not proof of optimal play and
never an executable client action.

## Claim boundary

V7 may prove only that an RGB factorized policy completes the fixed project-owned Rich
PixelArena task. V4–V6 may prove only read-only decoding, domain-alignment diagnostics, and
audited abstract host-side suggestions. Nothing in this repository establishes Honor of
Kings skill, GameCore equivalence, real-client automation, tactical optimality, or transfer
outside the audited recordings and fixed abstract vocabulary.
