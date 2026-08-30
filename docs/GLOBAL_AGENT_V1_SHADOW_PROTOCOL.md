# Global Agent v1 Stage 6A: zero-control Shadow

Stage 6A observes one locally attested, owner-authorized self-built test App through an explicit
V4L2 node. It is a transport and runtime-stability check, not a semantic-accuracy claim and not a
promotion to input. The promoted DAgger checkpoint is fixed by SHA-256 in
`GLOBAL_AGENT_V1_SHADOW_AUTHORIZATION.json`.

The only data flow is `RGB → frozen real_video_views → 16 causal frames → MacroCommand → candidate
mode log`. The runtime starts a foreground/display `GuardWatchdog`, but contains no input sender,
scrcpy control, ADB input, HID, online learning, or model update. `input_commands_sent=0`,
`control_output=false`, and `device_input_allowed=false` are invariants.

## Run sequence

First run 60 seconds. It passes only when cycle coverage is at least 95%, p95 end-to-end latency is
under 500 ms, guard failures are zero, no backlog develops, and no input is sent. The 600-second
run uses the exact same checkpoint, preprocessing, thresholds, and device binding. A foreground
loss, stale capture, decoder failure, or display drift immediately stops logging and emits no
candidate after the stop.

The output directory contains only `events.jsonl` and `summary.json`; it stores frame hashes,
anonymous configuration hashes, timestamped probabilities, candidate modes, latency, abstention,
and stop reasons. It never stores frames, video, serials, package names, coordinates, identity
contents, or device paths.

Challenge-pack failure remains a block on every input stage. A passing Stage 6A cannot authorize
Stage 6B; it can only establish that the real-RGB zero-control path is stable.

## First smoke result

The first 60-second run completed 114/114 macro cycles with p95 end-to-end latency 37.2 ms, zero
hard stops, and `input_commands_sent=0`. Its candidates were exclusively `DISENGAGE/OWN_BASE`.
The source App was paused for that run. The result therefore closes only the paused-scene transport
smoke; it cannot establish either real-domain semantic degeneration or candidate diversity.

## Active-scene diagnostic result

The subsequent 60-second zero-control run completed 114/114 cycles at p95 end-to-end latency
33.1 ms, with zero hard stops and `input_commands_sent=0`. All 114 logged frame hashes were
distinct, so the capture source was visually dynamic. Its candidates nevertheless remained
exclusively `DISENGAGE/OWN_BASE`. This is a completed runtime diagnostic with candidate diversity
not demonstrated; it is not a claim of verified real-video semantic error. The run stops here: no
threshold change, retraining, or 600-second Shadow is admitted. The fixed challenge failure
continues to block every input stage.

## Local-ROI repair result

Stage 6A-v1.1 preserved the frozen checkpoint but used the local, observation-only main-view,
minimap, and HUD crops before resizing them to the model contract. The 60-second run again completed
114/114 cycles with p95 end-to-end latency 32.9 ms, zero hard stops, and zero input. It recorded the
local ROI configuration hash but no coordinates or images. Its 114 dynamic frames again produced
only `DISENGAGE/OWN_BASE`. The geometry repair therefore rules out the earlier whole-screen resize
path as a sufficient explanation; no further preprocessing variant is admitted in this lineage.
