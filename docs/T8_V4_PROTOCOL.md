# T8-v4 zero-human-label weak-supervision protocol

## Purpose and lineage

T8-v4 is a new, non-promoting, offline research lineage. It does not relax, overwrite, or
reinterpret the frozen failures of T8-v2.7 or T8-v3. It asks whether two automatic teachers agree
on four directly observable visual states, whether RGB predicts those weak targets better than
time and shuffled-label controls, and whether the prediction uses the expected image regions.

T8-v4 uses no human frame or action labels, creates no annotation interface, opens no phone or
device-input path, and never reads video-test. Outputs may contain offline candidate-action logs,
but `promotion_allowed=false`, `control_output=false`, and `device_input_allowed=false` are frozen.

## Observation contract

The learned outputs are exactly:

- `main_view_enemy_cue_visible`
- `basic_attack_button_visual_enabled`
- `skill1_button_visual_ready`
- `skill2_button_visual_ready`

Every output describes the final frame of a 16-frame causal window represented by 512-dimensional
features. `main_view_enemy_cue_visible=false` means only that the current main view has no accepted
enemy visual cue; it does not mean that no enemy exists outside vision. Button outputs describe
appearance only, not executability, range, safety, targeting, or tactical value.

`attack_opportunity`, `target_attackable`, `safe_to_attack`, skill3, movement, aiming, holding, and
target intent are excluded. The only candidates are `candidate_basic_attack`, `candidate_skill1`,
`candidate_skill2`, and `none`; they are written to offline logs only.

The machine authorities are:

- `game_rules/observation_contract_v2.json`
- `game_rules/candidate_action_contract_v1.json`
- `configs/t8_v4_weak_supervision_v1.json`
- `configs/t8_v4_experiment_plan_v1.json`

Each uses canonical-JSON self-hashing. Missing fields, hash changes, threshold changes, split
changes, old-lineage inputs, and any test access fail closed.

## Independent automatic teachers

The rule teacher is the frozen red enemy-cue and fixed HUD-ROI detector. Its implementation,
thresholds, layout identity, and report hash are versioned and cannot be tuned during T8-v4.

The synthetic-transfer teacher is trained from PixelArena RGB with automatically generated
structured truth. Its ResNet-18 starts from the existing V5 visual adapter and may use the existing
unlabeled real-video SimSiam representation adaptation. It never reads rule-teacher predictions or
artifacts. PixelArena dev alone selects it. Every head must reach macro-F1 at least 0.90 on the
frozen synthetic dev set before real-video inference.

These sources are intentionally different: one is a fixed real-image visual rule, the other is a
learned synthetic-domain detector. Agreement is evidence of teacher consensus, not proof of real
semantic correctness.

## Automatic pseudolabel rule

For each real-video frame, each teacher receives five deterministic views:

1. original;
2. brightness +8%;
3. brightness -8%;
4. contrast 0.9;
5. contrast 1.1.

A head is accepted only when both teachers give the same binary decision, both original-view
confidences are at least 0.80, and each teacher keeps the same decision across all five views.
Every other result is `uncertain` and has zero training-loss mask. Negative accepted rows are
deterministically downsampled per session to at most three times the accepted positives.

The anonymous manifest and feature shards may contain only session hashes, timestamps, 16x512
features, automatic labels, masks, confidences, and contract hashes. They contain no raw or
duplicated RGB/video, device path, source path, or human label.

Train contains exactly 103 frozen video-train sessions and dev contains exactly 23 frozen
video-dev sessions. A source session belongs to one split. Video-test may not be parsed, listed,
or opened.

The weak route passes only if each head has accepted coverage at least 0.15 and accepted-label
perturbation stability at least 0.90. A failed head allows at most one rule-teacher repair; the
confidence threshold cannot be lowered. Continued failure records weak-supervision evidence as
insufficient and stops expansion.

## Seed-0 diagnostic ladder

All models use the identical split, seed 0, eight epochs, masked weak targets, and dev-only
selection:

| Model | Diagnostic |
|---|---|
| Class prior | class imbalance |
| Time-only | temporal/session progress shortcut |
| Last-frame linear | single-frame visual signal |
| Pool-MLP | unordered temporal aggregation |
| Causal TCN | ordered 16-frame value |
| Label shuffle | negative control |

The best RGB model must exceed time-only weak-target macro-F1 by at least 0.10 and its matched
label-shuffle run by at least 0.15. The TCN is retained only if it exceeds the better of
last-frame linear and Pool-MLP by at least 0.05; otherwise the simpler model is selected. Metrics
are summarized per session and uncertainty is estimated by session-level bootstrap.

## Causal controls

Temporal controls are static last-frame repetition, reversed order, and a two-second shift.
Spatial controls are gameplay mask, HUD mask, gameplay/HUD swap, and semantic-preserving sham
swap. Enemy-cue confidence should react primarily to gameplay interventions; button confidence
should react primarily to HUD interventions. The correct-region mean confidence drop must be at
least 0.15 and the unrelated-region drop at most 0.05.

A spatial report is valid only when generated from the selected student on video-dev with the
same model, dataset, contract, and intervention hashes. Teacher responses alone cannot satisfy the
student spatial-selectivity gate.

## Decision and claim boundary

The run writes exactly one `decision.json` with at least:

```json
{
  "human_labels_used": false,
  "synthetic_teacher_passed": false,
  "teacher_consensus_usable": false,
  "rgb_signal_against_weak_targets_demonstrated": false,
  "spatial_selectivity_demonstrated": false,
  "temporal_order_adds_value": false,
  "semantic_accuracy_verified": false,
  "promotion_allowed": false,
  "control_output": false,
  "next_required_action": "run_contract_checks"
}
```

Without human truth, the project reports teacher agreement, perturbation stability, weak-target
learnability, spatial selectivity, temporal controls, and cross-session behavior. It does not
report real-video semantic precision, recall, or accuracy. A positive weak-target result does not
authorize offline replay, Shadow, device input, or a claim of game skill.

## Mechanical gate order

1. Verify canonical contract hashes and frozen lineage.
2. Train and admit the independent PixelArena source teacher.
3. Materialize train/dev consensus pseudolabels and anonymous QC.
4. Stop if coverage or stability fails after the single permitted repair.
5. Run the seed-0 ladder, shuffle, temporal controls, and student spatial interventions.
6. Write and freeze `decision.json`.

Any source-teacher failure, repeated coverage failure, RGB-vs-time failure, spatial-selectivity
failure, or inability to reach a decision within one week stops the route. Failure does not add a
larger model, Transformer, third teacher, ordinary video collection, human annotation, or phone
input. It is recorded as `weak supervision evidence insufficient`.
