# Operation Policy v1

Operation Policy v1 is the offline second-part learning route downstream of the frozen Mobile
Operation Base v1. It does not reopen or overwrite T8-v2.7, T8-v3, T8-v4, T8-v5, or Basic MVP.
Its first engineering target is deliberately limited to continuous nine-class movement and sparse
five-class combat. Recommended purchase and death/unknown-screen stops remain deterministic
Mobile Operation Base responsibilities.

The first stage trains seed-0 inverse-dynamics heads from already frozen owner-testbed evidence.
The five-minute and sixty-second Operation Base sessions provide eight-direction movement and
concurrent combat examples. Eight existing visual-combat sessions provide stationary `wait` and
four-button combat examples, split six train and two dev. The selected V5/SimSiam ResNet-18 stays
frozen. Movement uses the encoder's 4x4 main-view feature map; combat uses 4x4 main-view plus HUD
feature maps so directional evidence is not erased by global pooling. Source pairs use the frozen
5 Hz scheduled sample clock while reports retain their actual elapsed timestamp evidence. Two independently
evaluated 200 ms and 500 ms heads must pass dev and shuffled-label gates before video labels exist.

The second stage opens only the 103 video-train and 23 video-dev shards. A label is accepted only
when the 200 ms and 500 ms inverse models agree and both reach confidence 0.80. Other rows remain
`uncertain`. Accepted `none`/`wait` rows are deterministically capped at three times the positive
count. Output contains anonymous session hashes, timestamps, float16 frozen features, automatic
labels, confidences, and contract hashes; it contains no repeated RGB, video, source path, device
identity, private coordinate, or layout value. Video-test shards are never opened.

The final offline stage compares time-only, last-frame, pooled MLP, causal TCN, and label-shuffle
baselines on 16 past frames. TCN is retained only if its dev score exceeds the best simpler model
by 0.03. One fixed minimap ablation is retained only if movement macro-F1 improves by 0.02. These
metrics measure learnability against automatic inverse-dynamics targets, not real semantic action
accuracy.

Every artifact keeps `semantic_accuracy_verified=false`, `promotion_allowed=false`,
`control_output=false`, and `device_input_allowed=false`. A passing offline movement decision may
request a separately reviewed read-only Shadow contract. It does not itself authorize Shadow or
phone input. Failure stops the route without lowering thresholds, growing the model, collecting
more ordinary video, or reopening a frozen T8 lineage.

## Frozen result

The pooled-feature seed-0 run failed and was preserved. One preprocessing repair used the frozen
scheduled source clock and the same encoder's 4x4 spatial map. The repaired 200/500 ms movement
macro-F1 values were 0.2472/0.2059 and combat values were 0.1714/0.2124, below the immutable
0.70/0.55 gates. The 500 ms movement shuffle margin was also below 0.15. The route stopped before
video pseudolabels and policy training and may not be reopened by threshold reduction or a larger
model.
