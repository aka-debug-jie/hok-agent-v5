# Basic-only deterministic MVP protocol

This is an independent delivery route after the frozen T8-v4 and T8-v5 research failures. It may
extract only the T8-v5 basic-attack correct-ROI head because that component independently passed
time, wrong-ROI, and shuffle controls. The failed T8-v5 combined decision does not authorize
control.

The candidate vocabulary is exactly `wait` and `candidate_basic_attack`. A candidate requires a
valid, non-frozen screen, frozen red enemy-rule probability at least 0.80, basic ROI probability at
least 0.80, three consecutive 5 Hz confirmations, a 1.5-second refractory period, no more than ten
candidates per minute, no more than twenty per session, and no more than three uninterrupted
identical candidates.

The first gate is offline video-dev replay. It must process all 23 sessions, produce at least five
candidates, produce zero invalid-screen candidates and rate violations, pass black/gray controls,
open no video-test, persist no RGB/source paths, and send no input.

Only a passing hash-bound offline summary admits one five-minute read-only Shadow. Shadow requires
95% cycle coverage, p95 scheduled-to-decision latency at most 150 ms, at least one and at most ten
candidates per minute, zero invalid-screen candidates, and zero input commands. Failure stops the
route. A separately frozen 20-action probe contract does not exist until Shadow passes.

This route tests bounded candidate generation in the owner-authorized self-built app. It does not
establish semantic accuracy, tactical correctness, commercial-game capability, or authorization
outside that test app.

## Frozen result

Offline replay processed 116,703 sampled video-dev frames and passed with six conservative
candidates, zero invalid-screen candidates, and zero rate violations. The admitted five-minute
read-only Shadow completed 1,500/1,500 cycles with p95 scheduled-to-decision latency 15.6 ms and
zero input commands, but produced zero candidates. Basic probability remained near 0.30 and never
reached 0.80; enemy probability reached 0.80 on only 9 frames, with no overlap. Shadow failed and
the 20-action probe was not created or opened. Threshold relaxation and repeat Shadow are not
permitted in this lineage.

## Deterministic rule fallback

A separate engineering fallback removed the failed learned basic head and used the frozen HUD ROI
score directly. Contract v1 at 0.80 failed because all 100 smoke scores were 0.767--0.794. One
fixed capture-domain calibration lowered the engineering-only threshold to 0.75 and added a
mandatory release guard: after every tap, the ROI must fall below 0.75 before another tap can be
armed. The second smoke passed 100/100 cycles.

The initial bounded probe sent one basic tap after warmup. Across 225 cycles the ROI probability
never fell below 0.795. Owner confirmation then established that basic attack has no cooldown
dimming, so the release condition was invalid for that action rather than evidence of a failed
tap. A private, Git-ignored touch sample corrected the execution point while the original icon
center remained the visual ROI. The separated path passed 20 actions, one minute, and five minutes.

The same visual/execution separation was applied to skills. Skill1 and skill2 touch points were
confirmed, and skill1 demonstrated a visible cooldown cycle. Skill3 required synchronous ADB
acknowledgement rather than the unacknowledged interactive pipe. A tap-only synchronous sender was
then frozen for all four combat buttons. Two 60-second mixed probes each executed every button five
times with 20/20 acknowledged commands, zero unexpected actions, no raw frames, and no persisted
coordinates. Movement, aiming, target selection, and model-driven control remain outside this
result.
