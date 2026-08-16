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
- `intent_contract_v1.json`：高层意图词表及建议决策周期。
- `state_machine_v1.json`：最小对局阶段与意图约束，不规定具体连招。
- `source_registry_v1.json`：规则来源和核验状态。

推荐的信息流：

```text
causal RGB
  -> learned observations with confidence
  -> learned high-level intent
  -> learned low-level action policy
  -> existing safety and legality boundary
```

`ABSTAIN` 是合法输出。任何低置信度、画面不完整或未知界面都不应由规则层猜测。
