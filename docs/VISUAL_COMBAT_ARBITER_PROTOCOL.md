# Visual combat arbiter protocol

The visual combat arbiter is a deterministic owner-testbed controller, not a learned policy. It
uses the original fixed icon locations for visual state and a separate Git-ignored execution
layout for acknowledged synchronous taps. It can select only basic attack, skill1, skill2, and
skill3. Movement, aiming, target selection, arbitrary shell commands, and commercial clients are
outside its surface.

Basic attack uses a frozen 0.75 visual-ready threshold and a global action interval. Skill1 and
skill2 use frozen absolute ROI readiness. Skill3 uses a session warmup brightness baseline because
the older teacher has no skill3 calibration. Every skill is disarmed after a tap and may rearm only
after three cooldown frames followed by three ready frames. Basic attack has no false cooldown
release requirement.

Selection is round-robin over the first visually admitted action. Every tap passes the local build
identity, foreground package, display orientation, layout, watchdog, vocabulary, count, and rate
guards, then uses synchronous `input touchscreen tap` with command acknowledgement.

The 60-second gate allowed at most 20 total actions and passed with 4 skill1, 5 skill2, 1 skill3,
and 10 basic actions. The five-minute gate allowed at most 60 and passed with 10 skill1, 10 skill2,
8 skill3, and 30 basic actions. All 78 commands were synchronously acknowledged, and the owner
observed no wrong action, cooldown repetition, missed tap, coordinate error, or uncontrolled input.
The lower skill3 count reflects its longer visual cooldown; the arbiter did not bypass it to fill
the cap.

The output schema supports timestamped frame hashes, visual probabilities, selected action,
cooldown state, synchronous execution state, and rejection reason without storing RGB, serial,
source paths, or coordinates. These events may be used later as automatic `executed_action`
supervision only under a separately frozen dataset contract. They do not establish tactical skill
or semantic enemy understanding.

The first formal event collection contains one 60-second and one five-minute session: 1,770 event
rows, 78 synchronously bound actions, and 1,692 wait rows. Every executed row has a selected action,
synchronous acknowledgement, and `synchronous_executed_action` source; cooldown-unarmed and wait
states are both covered. These two diagnostic sessions have only derived fixed-rate timestamps and
no RGB/features, so training is explicitly blocked. The frozen dataset contract requires twelve
new sessions with actual elapsed timestamps and 16-frame derived RGB or frozen-encoder features,
then an 8/2/2 session split.
