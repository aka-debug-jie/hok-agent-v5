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

Before collection, one read-only loading-panel check may locate the unique yellow self-name
highlight. A top-row highlight means blue side and a bottom-row highlight means red side. The
loading-screen cue and highlight margin must both pass; otherwise side is unknown. The frame is not
persisted, the result is not a movement label, and this check sends no input.

For the fixed marksman role, the verified side selects one deterministic continuous opening
prelude: blue holds east toward bottom lane and red holds west toward top lane for ten seconds.
The prelude runs before formal rows begin, is recorded only in the session summary, and never uses
the minimap-teacher label. A missing side disables collection rather than choosing a route.

The frozen offline audit over 1,485 existing minimap frames passed: detection coverage was 0.7838,
all eight directions were present, and player-position jump P95 was 3.59 pixels against a 5-pixel
limit. No phone or control output was used for this audit.

The live sequence is strictly staged: sixty-second zero-input observation, sixty-second bounded
input smoke, four automatic five-minute pilot sessions, seed-0 movement training, then—only after
the pilot gates pass—eight additional sessions and an 8/2/2 formal split. Every sample stores only
derived main/minimap/HUD/recommended RGB, continuous movement state, teacher confidence, actual
execution events, timestamps, and hashes. Raw video, source paths, serials, and coordinates are
not persisted.

The direction-diversity repair does not change the teacher or its nearest-target rule. Failed and
extra automatic sessions form an auditable candidate pool; exactly three train sessions and one
dev session are selected deterministically only when both splits contain usable causal windows for
all eight directions. Unselected sessions remain audit-only. The current bounded repair may add at
most four five-minute attempts and stops without a split if real visual labels remain incomplete.

The movement model uses V5/SimSiam ResNet-18 initialization and 16 past main/minimap frames. It is
compared with time-only, last-frame, pooled MLP, causal TCN, and label-shuffle controls. The
existing selected T8-v2.6 seed-1 combat model remains immutable and is bound by hash; it is not
retrained. Purchase, skill3 handling, cooldowns, touch lifecycles, and hard stops remain
deterministic.

An opening-level teacher session may observe skill3 still locked during warmup. In that case only
the Movement Teacher disables skill3 for the complete bounded session; movement, basic attack,
skill1, skill2, purchase, and hard stops continue unchanged. The frozen Mobile Operation Base
still requires skill3 to be ready and retains its original failure behavior.

The separately versioned spatial pilot keeps the same frozen encoder and thresholds but uses eight
past frames and a 2x4 grid from each main/minimap feature map. Before the full seed-0 pilot it must
memorize exactly four stable automatic-teacher windows per non-wait direction. This repair follows
the preserved 32-sample diagnosis that global pooling discarded direction layout; it does not add
human labels, PPO, Shadow, or model-driven phone input.

No movement model, fusion Shadow, or device-input stage is admitted until the four-session pilot
passes every frozen movement and transition gate. A missing foreground package or build identity
stops before any touch.
