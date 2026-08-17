# Mobile Operation Base v1

Mobile Operation Base v1 is the frozen first-part engineering surface for the owner-authorized
self-built App. One pinned scrcpy 1.25 session carries both RGB and exactly two touch pointers:
pointer 0 owns the persistent joystick and pointer 1 owns transient combat or recommended-purchase
taps. Every frame remains subject to build identity, foreground package, display orientation,
watchdog, duration, vocabulary, rate, and count guards.

The observation layout is private and Git-ignored. Each 5 Hz sample stores four derived 128x128 RGB
views: main view, minimap, combat HUD, and the single recommended-equipment region. It also stores
actual elapsed timestamps, continuous nine-class movement state, sparse five-class combat event,
sparse recommended-purchase event, minimap hash/change, hard-stop state, and rejection evidence.
Full raw frames, source paths, serials, and execution coordinates are not persisted.

Movement uses a real pointer lifecycle: DOWN at the joystick center, MOVE for eight-direction
changes without releasing, and UP only for stop, hard-stop, or final cleanup. Combat and purchase
use pointer 1 and may execute while pointer 0 remains active. Recommended purchase requires three
blink confirmations, a two-second refractory period, a five-per-minute cap, and a changed or
disappeared recommendation before rearming.

The top-center death-replay banner is a hard stop for death, respawn, and the owner-testbed ended
state. Unknown/invalid screens use the same stop path. Appearance immediately releases pointer 0,
blocks movement/combat/purchase, and records only wait. Operations resume only after three
consecutive valid frames with the banner absent.

The 60-second v2 smoke passed with all eight directions, 20 combat actions, five purchases, 285
minimap observations, and 25 movement-parallel action cycles. The formal five-minute run passed
with 59 movement transitions, 60 combat actions, 25 purchases, 1,485 samples, 85 parallel action
cycles, and zero unexpected actions. The death test detected 12 hard-stop cycles, released the
joystick once, emitted zero actions during the stop, waited two additional recovery frames, and
then resumed movement.

This freezes the operation and automatic-supervision base only. It does not implement enemy
semantics, target selection, aiming, learned movement, or tactical intent. Those capabilities must
use this interface rather than adding another input path.
