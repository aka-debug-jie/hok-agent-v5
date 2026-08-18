# Operation Movement Teacher v1

Operation Movement Teacher v1 is the state-conditioned movement-data route downstream of the
frozen Mobile Operation Base. It replaces the fixed eight-direction actuator schedule with a
visual minimap teacher while retaining the same persistent joystick pointer, transient
combat/purchase pointer, build identity, foreground, display, watchdog, death, unknown-screen,
duration, rate, and cleanup guards.

The teacher operates on the private high-resolution minimap ROI. A compact green component and a
spatially overlapping compact red component identify the player marker. The nearest separate red
component is the current target. Their relative vector is quantized to eight directions. Three
consistent frames are required before a direction change, every direction is held for at least one
second, and a missing detection keeps the current direction for at most one second before `wait`.
No fixed patrol or random fallback is permitted.

The frozen offline audit over 1,485 existing minimap frames passed: detection coverage was 0.7838,
all eight directions were present, and player-position jump P95 was 3.59 pixels against a 5-pixel
limit. No phone or control output was used for this audit.

The live sequence is strictly staged: sixty-second zero-input observation, sixty-second bounded
input smoke, four automatic five-minute pilot sessions, seed-0 movement training, then—only after
the pilot gates pass—eight additional sessions and an 8/2/2 formal split. Every sample stores only
derived main/minimap/HUD/recommended RGB, continuous movement state, teacher confidence, actual
execution events, timestamps, and hashes. Raw video, source paths, serials, and coordinates are
not persisted.

The movement model uses V5/SimSiam ResNet-18 initialization and 16 past main/minimap frames. It is
compared with time-only, last-frame, pooled MLP, causal TCN, and label-shuffle controls. The
existing selected T8-v2.6 seed-1 combat model remains immutable and is bound by hash; it is not
retrained. Purchase, skill3 handling, cooldowns, touch lifecycles, and hard stops remain
deterministic.

No movement model, fusion Shadow, or device-input stage is admitted until the four-session pilot
passes every frozen movement and transition gate. A missing foreground package or build identity
stops before any touch.
