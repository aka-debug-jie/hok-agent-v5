# Human IfO Bridge v1 protocol

Status: `FROZEN_NON_PROMOTED`. This document preserves the evaluated repair contract; the current
development route is `docs/HIERARCHICAL_POLICY_V0_PROTOCOL.md`.

## Objective

Human IfO Bridge v1 learns macro behavior from complete human-match videos without requiring human
action logs or manual frame labels. Human videos teach when high-level behavior changes; GlobalArena
teaches which `MacroCommand` commonly explains a latent state transition; the frozen DAgger policy
protects complete-episode behavior from regression.

The frozen evaluated chain was:

```text
human full-match videos
→ shared Human/Sim temporal representation
→ GlobalArena inverse macro dynamics
→ human-video macro pseudolabels
→ Human-BC
→ optional transition imitation
→ offline replay
→ zero-control mobile Shadow
```

The model input remains main-view, minimap, and HUD RGB at 5 Hz in 16-frame causal windows. The
first pilot retains the existing five intents and four target zones. Exact movement, skills,
purchase, layout, hero behavior, and safety remain deterministic modules.

## Gate A: cohort and shared representation

- Preserve the frozen DAgger checkpoint, 20-seed holdout, challenge results, video adapter, and both
  mobile Shadow diagnostics unchanged.
- Keep the existing 103 video-train and 23 video-dev episode split; video-test remains unopened.
- Select 20 train and 5 dev complete matches from one hero and one role/mode only. Every match must
  be human-controlled, uncut, have stable HUD, a complete timeline, no large overlay, and usable
  main/minimap/HUD crops. Model output cannot influence selection. If this narrow cohort is not
  available, stop rather than mix heroes or roles.
- The only selection input is the Git-ignored `configs/human_ifo_cohort.local.json`, created by
  `human-ifo-cohort-template`. Each row binds an anonymous session hash to its frozen split and
  declares hero, role, mode, human-control, complete-match, HUD, overlay, and ROI eligibility.
  Preflight rejects missing rows, split drift, duplicates, qualification failure, or identity mix.
- An optional unlabeled `human-ifo-cohort-propose` audit may rank technically usable train/dev sessions
  by deterministic RGB temporal signatures. Its output is explicitly non-promoting:
  `semantic_identity_verified=false`, `human_control_verified=false`, and `gate_a_allowed=false`.
  It may guide later selection but cannot replace the local cohort manifest or pass Gate A.
- Exactly one `human-ifo-unsupervised-repair` may add a fixed 0.1-weight latent mean/covariance
  alignment term to the existing within-domain temporal and simulator auxiliary losses. It remains
  diagnostic even if all numeric criteria pass; no second repair, Gate B, or phone work follows from it.

The completed unlabeled repair improved maximum Human-to-Sim nearest-scene concentration from
0.94375 to 0.83125 without simulator macro regression, but Human and Sim temporal-neighbor AUC were
0.665 and 0.6436. It therefore stopped at Gate A with no repair remaining. These numbers describe
the technical candidate cohort only and do not verify semantic identity or human-level behavior.

After that bounded repair closed, one separately versioned broad diagnostic may use all 103
video-train and 23 video-dev sessions without labels. It samples one deterministic representative
shard per session, trains one seed-0 epoch, and keeps video-dev diagnostic-only. This tests whether
cohort diversity improves general temporal structure; it cannot pass formal Gate A, open Gate B,
or weaken the same-identity cohort requirement.

The completed broad diagnostic reached corrected Human/Sim temporal-neighbor AUC of 0.6815/0.7847,
reduced maximum nearest-scene concentration to 0.6675, and preserved simulator macro performance.
Human temporal AUC remained below 0.70, so the zero-label representation route is frozen without
further tuning. Earlier time-bucket metrics are superseded because they mixed episode identities;
the final metric uses the nearest temporal neighbor within the same simulator episode.
- The scenario-card contract is diagnostic-only and cannot provide Human IfO training labels.

## H1: shared temporal representation

Reuse the current ResNet-18 main-view encoder, minimap/HUD CNNs, and causal TCN. Train one shared
encoder with within-domain temporal contrastive learning on Human and GlobalArena windows while
retaining a GlobalArena macro-classification auxiliary loss. The representation must encode temporal
behavior rather than raw pixel similarity.

Gate A passes only when Human and Sim temporal-neighbor AUC are each at least 0.70, correct temporal
order beats shuffled order, frozen simulator macro-F1 degrades by no more than 0.03, and Human
transitions have non-collapsed nearest-neighbor support among simulator transition scenes. A frozen
linear Human/Sim domain probe is reported as a diagnostic; it cannot by itself pass the gate.
Human-dev is diagnostics-only and cannot enter the optimizer. Failure permits one representation
repair only; it cannot open inverse dynamics or phone work.

## Gate B: inverse macro dynamics

Train `InverseMacroModel(z_t, z_t+Δ)` only from GlobalArena transitions and their exact intent/zone
labels. Compare current-state-only, `Δ=0.4s`, `Δ=0.8s`, `Δ=1.2s`, and label-shuffle controls, then
freeze the one transition horizon with the greatest admitted transition gain.

Gate B requires intent and zone macro-F1 of at least 0.80, transition gain of at least 0.05 over the
current-state-only control, and a margin of at least 0.20 over label shuffle. If current-state-only
is best, stop Human IfO rather than treating a classifier as inverse dynamics. It predicts macro
intent and zone only; it never reconstructs taps, joystick coordinates, aiming, or skill timing.

An explicit engineering acceptance of the broad 0.681 Human temporal AUC permitted one
simulator-only Gate B run without changing the original Gate A evidence. The selected 0.8-second
model reached intent/zone macro-F1 of 0.9323/0.9056 and exceeded label shuffle by 0.7045, but its
mean gain over state-only was only 0.0282. Gate B therefore failed the transition-causality gate and
cannot produce human-video pseudolabels without a separately reviewed boundary change.

## Gate C: human-video macro pseudolabels

Apply the accepted inverse model to consecutive Human latent transitions. Only human-train may
produce Human-BC samples; human-dev is diagnostics-only and is mechanically rejected by the training
loader. Confidence at or above 0.75 enters Human-BC; confidence from 0.50 through 0.75 remains
representation-only; lower confidence is discarded. A 1.5-second minimum segment hold suppresses
frame-level jitter.

Gate C requires at least 40% high-confidence coverage, no intent above 70% of accepted samples, at
least four accepted intents, and stable per-episode segment duration and switching rates. These are
pseudolabel-collapse guards, not a claim that human behavior must be class-balanced or semantically
verified. One failed run permits one temperature-only calibration on simulator dev; it cannot retrain
the inverse model, inspect human truth, or lower any gate. A second failure closes the pilot.

The engineering-accepted Gate B produced one Gate C audit. Before calibration, train/dev coverage
was 65.7%/68.5%, but accepted only 3/2 intents and had maximum shares of 83.5%/88.2%. The single
simulator-dev temperature calibration selected 1.1; the final audit retained 60.1%/63.2% coverage,
still accepted only 3/2 intents, and increased maximum shares to 87.9%/93.0%. Gate C is therefore
frozen as pseudolabel collapse. Human-BC, further calibration, threshold changes, and mobile stages
remain closed.

## Gate D: Human-BC and read-only validation

Train seed 0 first with 50% GlobalArena truth and 50% accepted Human pseudolabels. The loss is Human
macro cross-entropy plus `0.5 ×` simulator macro cross-entropy plus `0.5 ×` frozen-Dagger
distillation. Select only on simulator dev and video-dev diagnostics; video-test remains unopened.

Promotion requires at least 17/20 fresh simulator terminals, zero safety and illegal actions,
challenge performance of at least 4/6, video-dev maximum intent share below 70%, at least four
non-abstaining intents, and clear margins over time-only and label-shuffle controls. Only then may
one separately authorized 60-second zero-control Shadow run. A constant Shadow output stops the
route before the 10-minute stage and before every input stage.

## H5: not planned transition imitation

H5 is skipped by default. Only after Gate D passes but leaves a clear human-behavior deficit may one
bounded GlobalArena policy-improvement experiment use a discriminator over
`(z_t, z_t+1)` transitions. Its reward is fixed to:

```text
1.0 × human-transition imitation
+ 0.5 × terminal
+ 0.2 × structure progress
- 0.2 × stuck
```

Raw-RGB similarity is never a reward. H5 cannot weaken complete-episode, challenge, safety, or
Shadow gates and cannot authorize mobile input by itself.

A separately authorized attempt tested the latent-transition reward before any policy update. The
raw transition-pair discriminator reached dev AUC 0.995 but forward-vs-reverse Human margin only
0.0067. A single domain-normalized-delta repair still reached AUC 0.983 with margin 0.0071. Both
models identify Human/Sim domain rather than temporal behavior. The style reward is rejected, no
policy-gradient update is run, and the frozen Dagger policy remains selected.

The final bounded fallback froze the Human-adapted visual encoder and rebound only the causal and
macro heads using GlobalArena rule-teacher truth plus 0.5 frozen-Dagger distillation. It matched the
baseline at 18/20 non-timeout terminals, 20/20 tower progress, and mean tower damage 12.0; fallback
improved from 0.0563 to 0.0539, while stuck time rose from 0.0499 to 0.0569. Both models passed only
2/6 challenge scenarios. The rebound checkpoint is not promoted. Human video remains useful only as
visual encoder adaptation evidence in this lineage; strategy supervision remains simulator-only.

A final simulator-only parameterized curriculum used 48 training variants and 24 disjoint RGB
holdout variants around the six challenge families. The stronger v1 reached 4/6 canonical cards but
only 12/24 parameter holdout, increased stuck time from 0.0499 to 0.0641, and reduced tower damage
from 12.0 to 11.85. A single generic conservative repair reduced curriculum weight, learning rate,
and epochs while increasing Dagger distillation; it returned to 2/6 canonical, remained 12/24 on
parameter holdout, and still regressed episode metrics. Both candidates are rejected. No additional
curriculum weighting or card-specific tuning is allowed; frozen Dagger remains selected.

The final frozen-latent probe tested health bucket, own-base presence, enemy-distance bucket, push
condition, and ordinary lane advance on disjoint simulator state grids. Only push reached 0.861;
the other factors were 0.219–0.486 and none cleared both the 0.85 and shuffle-margin gates. The one
authorized auxiliary update unfroze only ResNet layer4 and the fusion projection. Final factor F1
was health 0.233, base 0.753, distance 0.439, push 0.929, and ordinary lane 0.735, while complete
episodes fell from 18/20 to 12/20, stuck rose to 0.0641, tower damage fell to 11.85, canonical
challenge stayed 2/6, and parameter holdout fell to 8/24. The candidate is rejected permanently;
no more representation training or model-family expansion is allowed in Global Agent v1.

## Interfaces and storage

Implementation is limited to `human_ifo.py`, `human_inverse.py`, one configuration, this protocol,
and focused tests. It reuses `GlobalMacroPolicy`, `GlobalArena`, the frozen DAgger checkpoint,
`real_video_views`, and existing train/dev manifests. All features, pseudolabels, checkpoints, and
reports remain below `HOK_LARGE_ROOT`; outputs contain anonymous episode hashes and never persist
source paths, raw video, device identity, or video-test references.

Phone access is prohibited in Gate A-D training and evaluation. The first permitted phone use is the
separately reviewed Gate D zero-control Shadow; model-driven input remains closed.
