# V5 里程碑与晋升门

## 总依赖

```text
M0 Legacy freeze + repository bootstrap
  ↓
M1 Authorized environment adapter + throughput
  ↓
M2 Fixed-hero 1v1 behavior cloning
  ↓
M3 Fixed-hero 1v1 recurrent PPO
  ↓
M4 Opponent league + multi-hero 1v1
  ↓
M5 3v3 centralized training / decentralized execution
  ↓
M6 RGB → public belief state
  ↓
M7 Pixel-derived policy in PixelArena
  ↓
M8 Real-client read-only shadow advisor
```

复杂度只能由前一阶段的闭环能力证据解锁。

---

## M0 — Legacy freeze 与 V5 仓库启动

### 目标

建立与 Pixel-MOBA v4.x 解耦的新仓库和最小工程合同。

### 交付物

- 本启动包落库；
- legacy 仓库只读 tag/commit 记录；
- Python 3.11 learner skeleton；
- 独立 environment-service 接口；
- mock environment；
- run/evaluation schema；
- CI、安全扫描和 secret scan；
- `TASK_000` 完成报告。

### 通过门

- 单元测试和 schema validation 全通过；
- mock env 固定 seed 可重复；
- service/learner 不共享私有内存；
- safety scan 没有真实客户端输入路径；
- legacy repo 无代码/报告被改写；
- `DELIVERY_PROGRESS.md` 更新完整。

### 失败含义

基础设施或边界尚未建立；不得接入真实 GameCore 或启动训练。

---

## M1 — 授权 GameCore adapter 与吞吐

### 目标

证明环境真实可用、可完整结束对局、可稳定收集训练数据。

### 交付物

- GameCore service；
- versioned RPC；
- 1v1 adapter；
- random baseline；
- common-AI baseline；
- ABS/replay artifact 索引；
- throughput benchmark；
- 100-episode stability report。

### 通过门

- 许可证和二进制实际可用；
- `reset → step → terminal → close` 连续完成；
- 固定 seed/environment version 的结构化 replay 精确；
- 100 个 episode：
  - protocol errors = 0；
  - action decode errors = 0；
  - non-terminal forced close ≤ 1%，且原因可解释；
  - reward component finite；
  - frame/tick 单调；
- 明确记录：
  - 单实例 env-steps/s；
  - 4/8/16 实例吞吐或可达到的最大并行数；
  - CPU、RAM、GPU、RPC 延迟；
- 不把 GameCore/许可证提交 Git。

### 失败决策

- 许可证缺失：`WAITING_EXTERNAL`；
- SDK/schema 问题：只修 adapter；
- 吞吐过低：只做服务批处理/并行优化；
- 不因此退回 PixelArena 宣称完成官方环境目标。

---

## M2 — 固定英雄 1v1 BC

### 目标

用完整 common-AI/teacher 轨迹让 Agent 获得可闭环运行的初始能力。

### 交付物

- 因子化 recurrent Actor；
- value heads；
-完整 episode dataset；
- 32-sample overfit；
- mini/full BC；
- episode-disjoint evaluation；
- first playable checkpoint。

### 数据门

- train/dev/eval 按 episode、seed、side、opponent 隔离；
- 动作、legal mask、reward 和 terminal 对齐；
- 每类生效 action head 有非零监督；
- 胜局和失败局均保留；
- 不进行 adjacent-frame 随机拆分。

### 通过门

诊断门：

- 32-sample overfit ≥ 99% teacher-equivalent；
- 所有 action heads 有有限且非零梯度；
- no NaN/Inf；
- action decode 和 legal sampling 通过。

闭环门：

- 100 个固定 episode 完整结束率 ≥ 99%；
- illegal/decode/protocol errors = 0；
- 不允许全 WAIT、全 MOVE 或单动作坍塌；
- 对塔/水晶存在非零可归因进展；
- 相比 random 有显著改善；
- 至少形成可继续 PPO 的稳定 checkpoint。

BC 不要求立即达到最终胜率，但如果闭环完全零结构进展，不进入 M3。

---

## M3 — 固定英雄 1v1 recurrent PPO

### 目标

在 on-policy 访问状态上学会完整获胜。

### 训练原则

- 从通过 M2 的 BC checkpoint 开始；
- recurrent PPO/GAE；
- legal action mask 仅用于采样；
-完整 episode rollout；
- curriculum 从局部目标逐步过渡到完整胜利；
- promotion 不读取训练 cohort。

### 建议 MVP 通过门

在预注册 200 局 fixed-eval suite 上：

- 总胜率 ≥ 70%；
- 任一 side 胜率 ≥ 60%；
- Wilson 95% 下界高于当前 active checkpoint；
- 完整 episode 结束率 ≥ 99%；
- illegal/decode/protocol errors = 0；
- hero tower/crystal damage 明确非零；
- 最大 active action 占比 < 90%；
- 与 NULL/random/common-AI 基线相比结构结果改善；
- 3 次独立 seed 中至少 2 次复现主要结论。

阈值是 V5 初始工程门，可在**正式训练前**版本化调整；运行后不得事后降低。

### 失败分类

- reward 上升、win 不升：reward Goodhart；
- BC 好、PPO 退化：on-policy/optimization 或 catastrophic forgetting；
- side 差异大：side bias；
- fixed AI 好、其他对手崩：opponent overfit；
- 不能完整结束：long-horizon/terminal credit；
- illegal 非零：action/adapter 工程问题。

---

## M4 — Opponent league 与多英雄 1v1

### 目标

从固定对手、固定英雄扩展到稳定自博弈和共享多英雄策略。

### 交付物

- league manager；
- current/best/historical/exploiter/common-AI 对手池；
- Elo；
- hero embedding；
- shared encoder/recurrent core；
- hero-conditioned heads；
- distillation/retention 机制。

### 通过门

- active checkpoint 对历史池 Elo 提升；
- 无单一 cyclic exploit 主导；
- 至少 3 个英雄达到各自预注册胜率；
- 任一英雄不因新增英雄退化超过容差；
- side、hero、opponent 分层均有胜局；
- 多 seed 复现；
- 失败 checkpoint 不进入 league active set。

### 禁止

- 根据同一对手的评测反复调参；
- 用多余英雄样本填补某英雄零能力；
- 在固定英雄尚未通过 M3 时提前做大英雄池。

---

## M5 — 3v3 MAPPO/CTDE

### 目标

学习可归因的团队协作，而不是三个共享 reward 的独立挂机 Agent。

### 架构

- shared Actor；
- hero/role embedding；
- decentralized recurrent hidden；
- centralized Critic；
- MAPPO 或等价 CTDE；
- team/individual/mixed value heads。

### 通过门

- 3v3 环境 100-episode stability；
- 预注册对手池中有稳定胜率/Elo；
- 每个受控英雄均有非零角色贡献；
- 不出现长期 lazy agent；
- team reward、individual reward 和 terminal zero-sum 分开记录；
- unseen lineup 和 side 独立 holdout；
- 不读取不可见敌方 truth。

### 失败处理

若共享团队奖励导致 lazy agent，优先修 credit assignment、value decomposition 和角色指标；不通过隐藏 privileged hint 修复。

---

## M6 — RGB → Public Belief State

### 目标

把视觉问题从策略问题中独立出来。

### 输出

- scene；
- self/HUD；
- visible entities；
- public target proposals；
- map/minimap；
- cooldown；
- temporal public belief；
- confidence、age 和 OOD。

### 通过门

- 对齐的 research/sandbox 数据；
- source-trajectory-disjoint；
- seen/unseen renderer 分开报告；
- 关键 public state 指标达到合同；
- perception failure 可触发 ABSTAIN；
- 不宣称 RGB 直接获得不可见 truth。

---

## M7 — Pixel-derived closed loop

### 目标

使用 M6 输出驱动冻结或受控微调的 winning strategy。

### 顺序

1. belief-state substitution；
2. teacher policy action agreement；
3. value distillation；
4. closed-loop PixelArena；
5. on-policy pixel DAgger；
6. 可选 end-to-end fine-tuning。

### 通过门

- public-only export；
- privileged randomization delta = 0；
- PixelArena 完整 episode 胜率；
- 相对 structured teacher 的能力保持率；
- OOD/低置信失败关闭；
- 不能用静态 imitation accuracy 代替闭环结果。

---

## M8 — 真实客户端只读 Shadow Advisor

### 目标

把视觉和策略输出转化为有用建议与复盘。

### 通过门

- V4L2/回放只读；
- 无输入设备和控制依赖；
- latency、staleness、confidence 可观测；
- unsupported/invalid/OOD → ABSTAIN；
- 建议与事实回放可复核；
- 用户人工验收；
- 长期安全报告。

M8 永不授权真实客户端闭环。
