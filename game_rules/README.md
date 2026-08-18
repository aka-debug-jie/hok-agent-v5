# Game Rules v1

本目录描述标准 5v5 对局的最小分层知识接口。它不是攻略全文，也不是手机控制脚本。

设计原则：

1. 只把稳定、不可违反的游戏事实写成硬规则。
2. 版本相关时间点只作为可替换先验，不直接决定设备动作。
3. 局势、目标和战术由 RGB 模型学习；规则层只提供输出词表、无效组合约束和低置信度弃权。
4. 训练 Actor 仍只接收 RGB/因果 RGB 序列，不接收结构化真值、奖励或规则状态。
5. 本目录不扩大 `AGENTS.md` 与 `BOUNDARIES.md` 规定的执行权限。

文件职责：

- `ruleset_v1.json`：稳定硬事实、版本先验及明确非目标。
- `observation_contract_v1.json`：允许模型从 RGB 推断的少量状态。
- `observation_contract_v2.json`：T8-v4 使用的四状态候选视觉约定。
- `candidate_action_contract_v1.json`：T8-v4 的候选动作派生约定。
- `intent_contract_v1.json`：高层意图词表及建议决策周期。
- `state_machine_v1.json`：最小对局阶段与意图约束，不规定具体连招。
- `source_registry_v1.json`：规则来源和核验状态。

以下信息流是远期架构，不是 T8-v4 已实现能力：

```text
causal RGB
  -> learned observations with confidence
  -> learned high-level intent
  -> learned low-level action policy
  -> existing safety and legality boundary
```

`ABSTAIN` 是合法输出。任何低置信度、画面不完整或未知界面都不应由规则层猜测。

T8-v4 当前采用更窄的只读路径（双教师弱监督、零人工标注）：

```text
causal RGB
  -> four local visual-cue probabilities
  -> quality/confidence/stability abstention
  -> deterministic candidate-action rule
  -> read-only event log
```

它只覆盖固定布局、固定技能槽语义的局部视觉线索，不实现本目录 v1 合同中的完整观察、
高层意图或学习型低层动作策略。完整定义见
[`docs/T8_V4_PROTOCOL.md`](../docs/T8_V4_PROTOCOL.md)。
