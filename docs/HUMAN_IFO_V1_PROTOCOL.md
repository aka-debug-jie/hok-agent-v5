# Human IfO Bridge v1 protocol

## Objective

Human IfO Bridge v1 learns macro behavior from complete human-match videos without requiring human
action logs or manual frame labels. Human videos teach when high-level behavior changes; GlobalArena
teaches which `MacroCommand` commonly explains a latent state transition; the frozen DAgger policy
protects complete-episode behavior from regression.

The active chain is:

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

## H0: cohort and lineage freeze

- Preserve the frozen DAgger checkpoint, 20-seed holdout, challenge results, video adapter, and both
  mobile Shadow diagnostics unchanged.
- Keep the existing 103 video-train and 23 video-dev episode split; video-test remains unopened.
- Select a deterministic pilot cohort of 24 train and 6 dev complete matches using only technical
  QC: stable HUD, complete timeline, no large overlay, consistent mode, and usable main/minimap/HUD
  crops. Model output cannot influence selection.
- The scenario-card contract is diagnostic-only and cannot provide Human IfO training labels.

## H1: shared temporal representation

Reuse the current ResNet-18 main-view encoder, minimap/HUD CNNs, and causal TCN. Train one shared
encoder with within-domain temporal contrastive learning on Human and GlobalArena windows while
retaining a GlobalArena macro-classification auxiliary loss. The representation must encode temporal
behavior rather than raw pixel similarity.

H1 passes only when Human and Sim temporal-neighbor AUC are each at least 0.70, correct temporal
order beats shuffled order, and frozen simulator macro-F1 degrades by no more than 0.03. Failure
permits one representation repair only; it cannot open inverse dynamics or phone work.

## H2: inverse macro dynamics

Train `InverseMacroModel(z_t, z_t+1)` only from GlobalArena transitions and their exact intent/zone
labels. Compare it with current-state-only and label-shuffle controls.

H2 requires intent and zone macro-F1 of at least 0.80, transition gain of at least 0.05 over the
current-state-only control, and a margin of at least 0.20 over label shuffle. It predicts macro
intent and zone only; it never reconstructs taps, joystick coordinates, aiming, or skill timing.

## H3: human-video macro pseudolabels

Apply the accepted inverse model to consecutive Human latent transitions. Confidence at or above
0.75 enters Human-BC; confidence from 0.50 through 0.75 remains representation-only; lower confidence
is discarded. A 1.5-second minimum segment hold suppresses frame-level jitter.

H3 requires at least 40% high-confidence coverage, no intent above 70% of accepted samples, at least
four accepted intents, and stable per-episode segment duration and switching rates. These are
pseudolabel-distribution checks, not a claim of real semantic accuracy. One failed H3 run closes the
pilot without lowering thresholds.

## H4: Human-BC

Train seed 0 first with 50% GlobalArena truth and 50% accepted Human pseudolabels. The loss is Human
macro cross-entropy plus `0.5 ×` simulator macro cross-entropy plus `0.5 ×` frozen-Dagger
distillation. Select only on simulator dev and video-dev diagnostics; video-test remains unopened.

Promotion requires at least 17/20 fresh simulator terminals, zero safety and illegal actions,
challenge performance of at least 4/6, video-dev maximum intent share below 70%, at least four
non-abstaining intents, and clear margins over time-only and label-shuffle controls. Only then may
one separately authorized 60-second zero-control Shadow run. A constant Shadow output stops the
route before the 10-minute stage and before every input stage.

## H5: optional transition imitation

H5 is skipped when H4 already meets the engineering target. Otherwise, and only after H4 passes,
one bounded GlobalArena policy-improvement experiment may use a discriminator over
`(z_t, z_t+1)` transitions. Its reward is fixed to:

```text
1.0 × human-transition imitation
+ 0.5 × terminal
+ 0.2 × structure progress
- 0.2 × stuck
```

Raw-RGB similarity is never a reward. H5 cannot weaken complete-episode, challenge, safety, or
Shadow gates and cannot authorize mobile input by itself.

## Interfaces and storage

Implementation is limited to `human_ifo.py`, `human_inverse.py`, one configuration, this protocol,
and focused tests. It reuses `GlobalMacroPolicy`, `GlobalArena`, the frozen DAgger checkpoint,
`real_video_views`, and existing train/dev manifests. All features, pseudolabels, checkpoints, and
reports remain below `HOK_LARGE_ROOT`; outputs contain anonymous episode hashes and never persist
source paths, raw video, device identity, or video-test references.

Phone access is prohibited in H0-H4 training and evaluation. The first permitted phone use is the
separately reviewed H4 zero-control Shadow; model-driven input remains closed.
