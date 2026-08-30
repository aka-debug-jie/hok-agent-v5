# Global Agent v1

Global Agent v1 is the active project route for one fixed hero in a project-owned or explicitly
authorized complete-match test environment. It learns only high-level intent and target region
from RGB. Existing deterministic navigation, combat, purchase, hero-profile, layout, and safety
modules execute the command. It does not add another T8 lineage or train direct touch coordinates.

## Architecture

The input is 16 causal frames at 5 Hz from main view, minimap, and HUD. The model outputs one of
nine intents—`FARM_LANE`, `FARM_JUNGLE`, `CLEAR_WAVE`, `PUSH_STRUCTURE`, `DEFEND`,
`CONTEST_OBJECTIVE`, `ENGAGE`, `DISENGAGE`, or `RECALL`—and one of ten semantic target zones:
own base, three lanes, own jungle, two river objectives, enemy jungle, enemy base, or hold current
zone. `ABSTAIN` is derived by confidence and hard safety guards. A six-class scene head is training
auxiliary evidence only.

The Global Planner emits `MacroCommand(intent, target_zone, confidence, valid_for_ms)` every 500
ms. TargetZoneNavigator converts the visible player position and selected zone center into an
eight-direction movement command. Skill Router maps the macro intent to deterministic combat modes
such as wave clear, structure, hero combat, objective, escape, and no-combat. The Mobile Operation
Base remains the only mobile touch lifecycle and safety authority.

## Rule teacher and simulator

The first milestone is a simplified complete `GlobalArena` built from Rich PixelArena: three
lanes, periodic waves, towers, bases, death/respawn, recall/purchase, one neutral objective,
scripted teammates/opponents, match termination, and a 10–12 minute cap. The structured rule
teacher may read simulator truth only to generate labels and fallback decisions; the RGB student
never receives structured state.

The causal chain must remain complete: farming, wave clearing, combat, tower push, high ground,
crystal, and terminal outcome. The teacher is accepted after at least 16 of 20 seeded games end
normally, safety violations are zero, death recovery is 100%, stuck time is below 10%, and at
least 70% of games make tower progress.

## Data and training

The initial frozen cohort is 200 teacher games for train, 50 for dev, and 50 sealed test games,
split by whole episode. Every 200 ms row binds derived RGB, timestamp, episode, scene, teacher
intent, target zone, executed low-level actions, safety mask, progress, death, and final outcome.
Only RGB enters the student. Training batches are scene-balanced across navigation, lane farm,
jungle farm, combat, structure push, and return/defend.

The first model uses ResNet-18 for main view, small CNNs for minimap/HUD, and one small causal TCN
or GRU. Loss is intent cross-entropy plus zone cross-entropy plus 0.25 scene auxiliary
cross-entropy. It does not learn exact skills, aim coordinates, target units, combos, a value
network, or full state reconstruction.

After behavior cloning, at most three simulator DAgger rounds may add states reached by the RGB
student while the structured teacher labels or safely takes over. PPO is blocked until the rule
teacher is stable, the student completes at least 70% of sealed simulator games, safety violations
are zero, and the student materially beats time-only and class-prior controls.

## Evaluation and mobile boundary

Primary metrics are episode termination, win rate against fixed scripts, tower progress, crystal
damage, stuck-time ratio, death recovery, fallback rate, human interventions, and invalid actions.
Intent/zone metrics are secondary. Real-video encoder adaptation begins only after the simulator
student can complete games. Mobile progression is then ten-minute read-only Shadow, three
independent ten-minute bounded runs, and at least three complete test episodes. Any failure stops
the next stage.

The first version controls one hero only. Multi-hero profiles, five-agent cooperation, large-scale
self-play, offline RL on current logs, PPO before imitation closure, and commercial-client control
are outside Global Agent v1.

The stage order, stop conditions, and current first implementation target are maintained in
[`GLOBAL_AGENT_V1_CONVERGENCE_ROADMAP.md`](GLOBAL_AGENT_V1_CONVERGENCE_ROADMAP.md).
