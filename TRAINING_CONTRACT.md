# V5 训练合同

## 1. 总原则

```text
环境先可用
→ BC 建立可玩初始化
→ on-policy PPO 学会赢
→ opponent league 提升稳健性
→ 多英雄
→ 3v3
→ 视觉蒸馏
```

任何阶段都不能用更多 GPU 时间绕过未通过的前置门。

## 2. 运行前检查

每个训练任务必须按顺序通过：

1. config/schema；
2. unit；
3. fixed-seed environment smoke；
4. action/reward/terminal audit；
5. 32-sample overfit（supervised）或 single-env reward check（RL）；
6. 10–30 分钟 mini run；
7. 只有 mini 无工程错误才允许 full run。

## 3. 行为克隆

### 3.1 数据

来源：

- common AI；
- 官方/授权 teacher；
- 规则基线；
- 已通过的 league policy；
- 完整胜局和失败局。

保存完整 episode，不只保存单步标签。

### 3.2 Loss

至少分解：

- macro CE；
- action-type CE；
- target set CE/attention CE；
- parameter CE；
- value/return；
- terminal/opportunity 可选辅助。

每个 head 报告：

- 样本数；
- loss；
- accuracy/Top-k；
- gradient norm；
- entropy；
- class/action coverage。

### 3.3 BC 关闭条件

- overfit 失败：实现或 schema bug；
- static 好、closed-loop 全零结构进展：不得继续堆 BC update，进入访问状态/动作语义诊断；
- 动作标签不足：补完整 episode 数据，不做无限 class-weight patch。

## 4. PPO

### 4.1 初始化

- 从通过闭环门的 BC checkpoint 启动；
- 不从随机或 legacy 零胜 parent 启动完整任务；
- Critic 可以预热。

### 4.2 Rollout

- 以完整 episode 为权威；
- learner 可 chunk，但 return/GAE 跨 chunk 连续；
- terminal 与 time-limit truncated 分开；
- RNN hidden 只在真正 episode boundary 重置；
- 保存 raw/masked/executed action 和 log-prob。

### 4.3 PPO 监控

- win/terminal；
- reward components；
- policy/value loss；
- KL；
- clip fraction；
- entropy；
- gradient norm；
- explained variance；
- action distribution；
- side/opponent 分层；
- tower/crystal events；
- illegal/decode/protocol errors。

## 5. Curriculum

课程必须改变任务难度，而不是事后改评测门。

推荐：

```text
合法动作与生存
→ 补刀
→ 局部战斗
→ 第一座塔
→ 单路结构推进
→ 水晶终局
→ 完整 1v1
```

promotion 由固定 curriculum evaluator 决定。

当完整胜局已经稳定，逐步降低 dense shaping 权重；terminal win 始终存在。

## 6. Reward 规则

- 每个分量命名；
- 总和可复算；
- 终局胜负优先级最高；
- 塔/水晶优先于纯英雄伤害；
- 小兵自然推进不能被误记为 hero 可归因能力；
- 训练 reward 不作为 checkpoint 唯一晋升指标；
- 不根据正式 eval 结果反调 reward。

固定审计策略：

- random；
- NULL/idle；
- common AI；
- winning teacher；
- active candidate。

若 NULL 获得高结构正奖励，reward 设计直接 No-Go。

## 7. Opponent League

对手池：

- common AI；
- current；
- best；
- recent historical；
- diverse historical；
- exploiter。

每个 checkpoint 保存：

- policy identity；
- hero set；
- training opponents；
- Elo；
- evaluation report；
- active/best/retired 状态。

不得删除导致 current 失败的历史对手。

## 8. 多英雄

阶段顺序：

1. 单英雄通过；
2. 同 archetype 2–3 英雄；
3. 不同 archetype；
4. 扩大英雄池。

每次扩展必须运行旧英雄 retention suite。

## 9. 3v3

采用 CTDE/MAPPO：

- decentralized actor；
- centralized critic；
- team/individual reward；
- role metrics；
- opponent/lineup holdout；
- lazy-agent 检查。

## 10. 超参数治理

允许：

- 在 train-only mini cohort 上做一次预注册 bounded sweep；
- 每个变量 2–4 个候选；
- 使用固定选择规则；
- sweep 结束后冻结 full run 配置。

禁止：

- 正式 eval 失败后继续追加第 N+1 个 learning rate；
- 每个 failure 创建新 loss；
- 用 seed mining 挑成功 checkpoint；
- 选择偶然胜局作为 active；
- 反复消费同一 holdout。

## 11. 随机种子

- development：至少 1 seed；
- candidate promotion：至少 3 independent training seeds；
- 主要结论至少 2/3 复现；
- eval 使用固定、版本化 seed registry；
- training/eval seed 不重叠。

## 12. Checkpoint

只有两类：

- `candidate`：训练完成但未晋升；
- `active`：通过 promotion suite；
- `best`：league/benchmark 最优；
- `retired`：保留但不再 active。

失败 candidate 不能作为下一 full run 的默认 parent，除非协议明确说明其工程结果仍有效且能力未回退。

## 13. 计算预算

单 RTX 4090 的优先级：

1. 环境并行；
2. 小型 structured recurrent policy；
3. BC/PPO；
4. 多英雄共享；
5. 3v3；
6. 视觉模型。

环境吞吐未测前，不承诺总训练步数和完成时间。
