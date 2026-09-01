# Deterministic Marksman Lane Controller v1

This is a non-learning owner-testbed controller. It consumes the read-only loading-panel side
result and uses no Movement Teacher, movement checkpoint, human label, reward, or PPO output.

Blue replays the frozen north-east/east bottom-lane opener; red uses its center-mirrored
south-west/west top-lane route. After reaching lane, blue advances north and red advances south for
eight seconds, then holds for four seconds while the existing visual combat and recommended
purchase rules continue. The cycle repeats for the finite run.

Death, ended, or unknown screens release both pointers immediately. After three valid recovery
frames the controller replays the same side-specific opener before returning to the lane cycle.
Missing side, foreground, build identity, display identity, layout, or watchdog freshness stops
without a fallback.

The controller writes bounded derived ROI tensors, deterministic phase/action events, timestamps,
and hashes under `HOK_LARGE_ROOT`. It stores no raw video, source path, serial, coordinate, account,
or training label. Its output cannot train or promote a movement model.
