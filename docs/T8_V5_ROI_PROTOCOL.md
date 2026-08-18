# T8-v5 ROI-isolated zero-label diagnostic protocol

## Scope

T8-v5 is a new offline lineage created after T8-v4 failed spatial selectivity. It does not reopen,
repair, or reinterpret T8-v4. It reuses only the frozen T8-v4 repair-1 consensus labels and the
existing V5-initialized ResNet-18 adapter. It collects no video, uses no human labels, opens no
video-test, and has no replay, Shadow, phone, or control stage.

The four visual-state names remain unchanged. The formal gate applies only to enemy cue, basic
attack, and skill1. Skill2 is diagnostic-only because the frozen dev consensus contains only four
accepted negative samples; it cannot support a stable admission claim.

## ROI isolation

- Enemy correct ROI: the fixed gameplay crop.
- Enemy wrong ROI: the fixed HUD crop.
- Each button correct ROI: a fixed square centered on its external-layout coordinate.
- Each button wrong ROI: the next combat button's ROI in a fixed cycle.

Every crop is made from the detected content box normalized to 128x128, then encoded by the frozen
adapter. Only 512-dimensional correct-ROI and wrong-ROI features, weak labels, loss masks,
timestamps, anonymous session hashes, and lineage hashes are stored. RGB, video, source paths, and
device paths are not stored.

## Seed-0 ladder

The first run compares class prior, time-only, correct-ROI linear, wrong-ROI linear, and
correct-ROI label shuffle under the same 103/23 split, seed 0, eight epochs, and class-balanced
masked loss. No TCN or larger model is part of this gate.

Each formal head must independently satisfy all three conditions:

- correct ROI minus time-only macro-F1 at least 0.10;
- correct ROI minus wrong ROI macro-F1 at least 0.15;
- correct ROI minus label shuffle macro-F1 at least 0.15.

A mean score cannot hide a failed head. Skill2 is always reported but never contributes to the
decision. Only if all three formal heads pass may a later, separately frozen experiment test
whether a 16-frame ROI TCN adds at least 0.05 over the single-frame ROI model.

## Claim and stop boundary

A passing result means only that the correct fixed ROI predicts conservative automatic weak
targets better than time, wrong-region, and shuffled controls. It does not establish real-video
semantic accuracy, action validity, game skill, or transfer.

`human_labels_used=false`, `semantic_accuracy_verified=false`, `promotion_allowed=false`,
`control_output=false`, and `device_input_allowed=false` are immutable. Failure of any formal head
freezes T8-v5 as insufficient ROI evidence; it does not add a larger model, adjust thresholds,
collect data, or reconnect a device.
