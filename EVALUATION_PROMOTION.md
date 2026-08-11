# 评测、Checkpoint 晋升与回滚合同

## 1. 核心原则

V5 的 checkpoint 只能由闭环证据晋升。

以下不能单独晋升：

- loss 下降；
- cached Top-1/Top-3；
- reward 上升；
- teacher recall；
- action margin；
- 参数发生变化；
- 单个成功 episode；
- 训练 cohort 上的胜率。

## 2. 评测层级

### E0 — 工程 smoke

目的：验证环境和执行链。

- 5–10 个 episode；
- random/constant policy；
- 检查 reset/step/terminal；
- 不构成能力结论。

### E1 — Development mini

目的：快速发现方向性问题。

- train-disjoint；
- 固定 20–40 局；
- 允许用于开发选择；
- 不得反复消费到失去独立性。

### E2 — Candidate promotion

目的：决定 candidate 是否成为 active。

- 完全预注册；
- 固定 200 局或合同指定规模；
- side、opponent、hero 平衡；
- evaluator 在 checkpoint freeze 后运行；
- 结果不能反向选择 update。

### E3 — Independent benchmark

目的：形成可对外描述的项目结论。

- 新 seed；
- 新 opponent 或 unseen lineup；
- 多训练 seed；
- 不用于调参。

## 3. 1v1 初始固定评测

建议初版 `eval-suite-1v1-v1`：

- 200 局；
- 100 红方、100 蓝方；
- 固定英雄；
- 至少两个 opponent profiles；
- seed 预先冻结；
- 不进入训练。

### 初始 promotion 门

| 指标 | 门 |
|---|---:|
| 总胜率 | ≥ 0.70 |
| 任一 side 胜率 | ≥ 0.60 |
| 完整结束率 | ≥ 0.99 |
| protocol/decode/illegal | 0 |
| hero tower/crystal damage | > 0 且覆盖多数胜局 |
| 最大 active action 占比 | < 0.90 |
| 相对 active Wilson 下界 | 严格改善或不退化+其他主指标严格改善 |
| 独立训练复现 | ≥ 2/3 seeds |

阈值可在第一次正式 candidate 前版本化修订；正式运行后不能降低。

## 4. 统计

### 4.1 胜率

报告：

- numerator/denominator；
- point estimate；
- Wilson 95% CI；
- side/opponent/hero 分层；
- pairwise same-seed comparison（可用时）。

### 4.2 Elo

- league Elo 只在固定更新规则下比较；
- 保存对手池快照；
- 报告对每个历史对手的逐项结果；
- 不用单一 Elo 隐藏 matchup 崩溃。

### 4.3 多 seed

报告每个 training seed 和聚合，不把 seed 当作普通 episode 混在一起。

## 5. 必须报告的闭环指标

### 5.1 Outcome

- win/loss/draw；
- terminal reason；
- episode length；
- forced close；
- side；
- opponent；
- hero/lineup。

### 5.2 Objective

- tower damage/last hit；
- crystal damage/last hit；
- economy/experience；
- objective/resource；
- K/D/A；
- hero attributable progress。

### 5.3 Safety/engineering

- illegal action；
- action decode；
- stale/duplicate tick；
- RPC failure；
- replay failure；
- NaN/Inf；
- timeout；
- environment restart。

### 5.4 Collapse

- action distribution；
- macro distribution；
- target distribution；
- entropy；
- no-target share；
- idle/move streak；
- dominant action share。

## 6. 晋升状态机

```text
TRAINED_CANDIDATE
  ↓ E0/E1 pass
EVAL_FROZEN
  ↓ E2
  ├─ PASS → ACTIVE
  └─ FAIL → REJECTED
```

若 active 已存在：

```text
candidate vs active
  ├─ capability/safety strict pass → replace active
  ├─ tie → keep active
  └─ regression → reject candidate
```

`best` 与 `active` 可不同：

- `active`：下一轮训练 parent；
- `best`：benchmark/league 最佳。

## 7. 回滚

以下立即回滚：

- illegal/protocol/replay 非零；
- reward/terminal schema 漂移；
- active 老 benchmark 严重退化；
- 动作坍塌；
- hidden/action log-prob 不一致；
- NaN/Inf；
- checkpoint/optimizer artifact 不完整。

回滚必须恢复：

- Actor；
- Critic；
- optimizer；
- scheduler；
- normalization state；
- league active pointer；
- curriculum state。

## 8. Failure Taxonomy

评测失败只能优先归为一个 primary category：

1. `ENVIRONMENT_BOUNDARY_FAILURE`
2. `ACTION_ADAPTER_FAILURE`
3. `NUMERICAL_OR_TRANSACTION_FAILURE`
4. `REWARD_GOODHART`
5. `TERMINAL_CREDIT_FAILURE`
6. `POLICY_COLLAPSE`
7. `SIDE_BIAS`
8. `OPPONENT_OVERFIT`
9. `HERO_GENERALIZATION_FAILURE`
10. `MULTI_AGENT_CREDIT_FAILURE`
11. `VISION_STATE_FAILURE`
12. `SIM_TO_REAL_FAILURE`
13. `CAPABILITY_NO_GO`

secondary evidence可以多个。

## 9. 评测数据隔离

- promotion suite 不进入 gradient；
- independent benchmark 不用于 curriculum；
- eval replay 不进入 BC/DAgger；
- 失败 eval 不自动并入训练；
- 同一 source 的 rerender 保持同 split。

## 10. 报告

每次 E2/E3 必须生成符合 `schemas/evaluation_report.schema.json` 的 JSON，并包含：

- code/config/checkpoint/environment hash；
- suite identity；
- seed registry；
- per-episode rows或独立明细 artifact；
- metrics；
- confidence intervals；
- gate derivation；
- disposition；
- report hash。
