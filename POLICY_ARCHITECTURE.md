# V5 策略架构

## 1. 架构目标

V5 的策略必须：

- 在结构化 public observation 上先学会完整获胜；
- 支持长时序记忆；
- 支持多实体和目标注意力；
- 直接建模因子化动作；
- 不依赖 K96 作为硬候选瓶颈；
- 能扩展到多英雄和 3v3；
- 能被 RGB/public belief student 蒸馏。

## 2. V5-Actor-1

```text
Public Observation
  ├─ self/HUD encoder
  ├─ scalar/economy encoder
  ├─ entity set encoder
  ├─ map/global encoder
  ├─ hero/side embedding
  └─ previous action encoder
             │
             ▼
      recurrent core (GRU first)
             │
       shared policy state
     ┌───────┼────────┬────────┐
     ▼       ▼        ▼        ▼
   Macro   Action   Target   Parameters
   head    head     attention  heads
             │
             ▼
      factorized distribution
```

第一版优先使用 GRU，而不是立即使用大 Transformer。原因：

- 环境吞吐更重要；
- hidden 语义更容易审计；
- 单 4090 上更新成本低；
- 后续可将 recurrent core 替换为 causal Transformer。

## 3. Entity Set Encoder

输入为 visible/public entities：

- entity category；
- team relation；
- relative position；
- health/resource；
- structure/minion/resource public flags；
- visibility/age；
- optional public track key。

采用 permutation-invariant attention 或 Deep Sets。

禁止输入：

- 不可见实体；
-真实 Core entity ID；
- teacher target；
- future event；
- legal mask。

## 4. Target Attention

target head 直接在当前 public entity set 上归一化：

```text
query = f(policy_state, macro, action_type)
key_i = g(entity_i)
P(target=i) = softmax(query · key_i)
```

另有 `no-target` token。

target identity 只在当前 public proposal/entity set 中解释；不得依赖跨 branch 的 candidate index。

## 5. 动作分解

建议第一版：

1. `macro_intent`
2. `action_type`
3. `target`
4. `direction`
5. `skill_slot`
6. `item/upgrade`
7. `stop/hold duration`（如上游需要）

条件关系必须显式编码，避免把无关参数共同归一化。

## 6. Legal mask

legal mask：

- 不进入 encoder；
- 不进入 recurrent hidden；
- 不作为 target attention feature；
- 在 sampling 时按 action branch 应用；
- 记录 mask 前后概率和 action；
- 全 illegal 时 fail-closed。

训练中可以使用 legal subset 验证和 masked log-prob，但不能把“当前动作是否合法”伪装成可部署 public feature。

## 7. Critic

1v1 第一版：

- shared public encoder；
- 可选 privileged state encoder；
- multi-head value：
  - terminal win；
  - structure；
  - economy；
  - combat；
  - survival。

3v3：

- centralized Critic；
- team/global privileged observation；
- individual/team value heads；
- Actor 仍 decentralized。

Critic 在导出时完全删除。

## 8. BC 与 RL 共用接口

Actor 必须同时支持：

```text
forward(observation, hidden)
sample(observation, legal_mask, hidden)
evaluate_actions(observation, action, legal_mask, hidden)
```

BC 和 PPO 不得分别实现两套 action decode。

## 9. K96 的保留范围

允许建立 `debug_top_actions(k=96)`：

- 展示最高概率完整动作；
- teacher equivalence audit；
- 人工解释；
- offline error analysis。

禁止：

- 先用规则生成 96 个候选再让模型只能从中选择；
- 为每次 projection miss 增加 reservation；
- 将 teacher action 注入 debug list；
- 把 debug list 作为部署动作空间。

## 10. 多英雄扩展

采用：

- shared encoder；
- hero embedding；
- hero-specific action availability adapter；
- shared recurrent core；
- 必要时 small hero-specific output adapters。

优先验证参数共享，再考虑按英雄独立模型。

## 11. 3v3 扩展

Actor：

- 同一参数共享给多个受控英雄；
- 每个 hero 独立 hidden；
- public teammate features可见；
- role embedding。

Critic：

- centralized global/team state；
- joint action/value context。

## 12. 最小自动测试

- entity permutation invariance；
- target mask；
- no-target；
- action branch normalization；
- legal mask不改变 encoder/hidden；
- all masked fail-closed；
- BC/PPO action log-prob parity；
- hidden terminal reset；
- chunked/unrolled numerical parity within tolerance；
- privileged randomization Actor delta = 0；
- export graph denylist；
- K96 debug关闭后行为不变。
