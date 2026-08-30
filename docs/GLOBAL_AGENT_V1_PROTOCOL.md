# Global Agent v1

Global Agent v1 is the active project route for one fixed hero in a project-owned or explicitly
authorized complete-match test environment. It learns only high-level intent and target region
from RGB. Existing deterministic navigation, combat, purchase, hero-profile, layout, and safety
modules execute the command. It does not add another T8 lineage or train direct touch coordinates.

## Architecture

The full enum reserves nine intents and ten semantic target zones. The first executable checkpoint
enables only `FARM_LANE`, `PUSH_STRUCTURE`, `ENGAGE`, `DISENGAGE`, and `RECALL`, plus
`OWN_BASE`, `MID_LANE`, `ENEMY_BASE`, and `HOLD_CURRENT_ZONE`. Its input is 16 causal RGB frames
at 5 Hz from main view, minimap, and HUD. `ABSTAIN` is derived by confidence and hard safety
guards; a five-class scene head is training evidence only.

The Global Planner emits `MacroCommand(intent, target_zone, confidence, valid_for_ms)` every 500
ms. TargetZoneNavigator converts the visible player position and selected zone center into an
eight-direction movement command. Skill Router maps the macro intent to deterministic combat modes
such as wave clear, structure, hero combat, objective, escape, and no-combat. The Mobile Operation
Base remains the only mobile touch lifecycle and safety authority.

## Rule teacher and simulator

The first milestone wraps Rich PixelArena as a middle-lane `GlobalArena`. It reuses periodic waves,
towers, crystals, combat, death/respawn, recall, a scripted opponent, and match termination rather
than adding a second physics engine. Three lanes, jungle, objectives, and economy simulation are
deferred. The structured teacher may read current simulator observations only to generate labels
and fallback decisions; the RGB student never receives structured state.

The causal chain must remain complete: farming, wave clearing, combat, tower push, high ground,
crystal, and terminal outcome. Stage 1A requires one non-timeout crystal terminal with zero manual
interventions and zero safety violations. Stage 1B then requires at least 16 of 20 seeded games
end normally, safety violations are zero, death recovery is 100%, stuck time is below 10%, and at
least 70% of games make tower progress.

## Data and training

The current pilot is exactly 40 train episodes and 10 dev episodes, split by whole episode with no
test cohort. Every row binds derived RGB, anonymous episode, tick, scene, teacher intent, target
zone, executed low-level action, and outcome. Only RGB enters the student.

The first model uses ResNet-18 for main view, small CNNs for minimap/HUD, and a small causal TCN.
Loss is intent cross-entropy plus zone cross-entropy plus 0.25 scene auxiliary
cross-entropy. It does not learn exact skills, aim coordinates, target units, combos, a value
network, or full state reconstruction.

Exactly one simulator DAgger round may add low-confidence, teacher-disagreement, stuck, and
recovery states reached by the RGB student. The round starts at 25% student authority and cannot
open another round. PPO is outside the current engineering route.

## Frozen five-stage result

- Rule teacher: Stage 1A passed; Stage 1B reached 20/20 non-timeout crystal terminals with zero
  safety or illegal-action events.
- Pilot data: 40 train and 10 dev episodes, 3,276 rows; manifest
  `5f46380464d4dd0b479e754dbf9ef58fbef51dd1ec10698849a93ff5a327712f`.
- Seed-0 BC: TCN selected; dev intent macro-F1 `0.8465`, zone macro-F1 `0.7257`, and 7/10 pure
  student simulator terminals.
- One DAgger round: 765 boundary samples; 9/10 terminals, mean tower damage `11.1→12.0`, fallback
  `0.0828→0.0525`, zero safety or illegal-action events. The original incomplete-metric failure
  report is preserved and a hash-bound read-only acceptance adds mean tower damage without
  retraining.
- Video adaptation and replay: the first candidate failed simulator non-regression and remains
  preserved. Dev-selected v2 passed on 103/23 sessions, with consistency `0.01581→0.00780` and
  simulator terminals `9/10→8/10`. The 23-session offline replay emitted 391 candidate rows,
  passed all deterministic limits and the static negative control, and sent zero input.

## Evaluation and mobile boundary

Primary metrics are episode termination, win rate against fixed scripts, tower progress, crystal
damage, stuck-time ratio, death recovery, fallback rate, human interventions, and invalid actions.
Intent/zone metrics are secondary. Real-video adaptation reads only the frozen 103 video-train and
23 video-dev sessions; video-test stays unopened. Offline replay emits candidate logs and sends
zero input. Mobile progression is not part of these five stages and requires a separately reviewed
ten-minute read-only Shadow contract.

The first version controls one hero only. Multi-hero profiles, five-agent cooperation, large-scale
self-play, offline RL on current logs, PPO before imitation closure, and commercial-client control
are outside Global Agent v1.

The stage order, stop conditions, and recorded results are maintained in
[`GLOBAL_AGENT_V1_CONVERGENCE_ROADMAP.md`](GLOBAL_AGENT_V1_CONVERGENCE_ROADMAP.md).
