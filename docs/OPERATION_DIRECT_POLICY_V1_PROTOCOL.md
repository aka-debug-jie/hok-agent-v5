# Operation Direct Policy v1

Operation Direct Policy v1 is a zero-phone offline learnability check over the already frozen
Mobile Operation Base and visual-combat RGB/event sessions. It uses the selected frozen
V5/SimSiam ResNet-18, 16 past frames at 5 Hz, and main-view, HUD, and minimap features. Continuous
movement state and sparse combat events are the automatically recorded executed-action targets;
purchase and hard-stop remain deterministic and are not learned.

The five-minute Operation Base session and six visual-combat sessions form train. The independent
sixty-second Operation Base session and two visual-combat sessions form dev. `wait` and `none`
rows are deterministically capped, and time-only, last-frame, pooled MLP, causal TCN, and
label-shuffle controls use the same split. Direction-change rows are a hard gate so persistence
cannot masquerade as a useful movement policy.

The seed-0 run selected Pool-MLP. Movement macro-F1 was 0.1618 and transition accuracy was 0.0909
(1/11), below the frozen 0.55 and 0.40 gates. Combat macro-F1 was 0.1913 versus 0.45. TCN was
worse than the best simple model. The route is therefore frozen failed without Shadow or input.

This result distinguishes actuator evidence from policy evidence: the frozen movement directions
and cooldown-aware round-robin combat actions verify concurrent execution, but they are not chosen
from gameplay state. Their labels cannot teach when or why a tactical action should be selected.
Every artifact remains `semantic_accuracy_verified=false`, `promotion_allowed=false`,
`control_output=false`, and `device_input_allowed=false`.
