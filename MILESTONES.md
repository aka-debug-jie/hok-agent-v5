# V5 里程碑与晋升门

## 总依赖

```text
M0 Legacy freeze + repository bootstrap + no-GameCore route reset
  ↓
M1 PixelArena-Structured ruleset/service + throughput
  ↓
M2 Fixed-archetype baseline + episode dataset
  ↓
M3 Fixed-archetype 1v1 behavior cloning
  ↓
M4 Fixed-archetype 1v1 recurrent PPO
  ↓
M5 Opponent league + multi-archetype 1v1
  ↓
M6 3v3 centralized training / decentralized execution
  ↓
M7 RGB → public belief + PixelArena RGB-only closed loop
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
- legacy 只读来源记录；若工作副本没有 `.git`，明确记录无法取得 commit identity；
- Python 3.11 learner skeleton；
- 独立 environment-service 接口；
- mock environment；
- run/evaluation schema；
- CI、安全扫描和 secret scan；
- `TASK_000` 完成报告；
- no-GameCore 路线、scoped claim 和 `TASK-010` 合同。

### 通过门

- 单元测试和 schema validation 全通过；
- mock env 固定 seed 可重复；
- service/learner 不共享私有内存；
- safety scan 没有真实客户端输入路径；
- legacy repo 无代码/报告被改写；
- `DELIVERY_PROGRESS.md` 更新完整。

### 失败含义

基础设施或边界尚未建立；不得启动 PixelArena 训练，也不得接入真实客户端动作。

---

## M1 — PixelArena-Structured ruleset、service 与吞吐

### 目标

从零建立 V5 原生项目内环境，证明 ruleset 可完整结束对局、可精确复核并可稳定收集训练数据。

### 交付物

- versioned PixelArena ruleset 与 local-process service；
- public/privileged observation、factorized action、reward 和 terminal 合同；
- versioned RPC、snapshot/restore 和 canonical replay；
- random baseline；
- NULL 与 scripted objective baseline；
- replay artifact 索引；
- throughput benchmark；
- 100-episode stability report。

### 通过门

- `reset → step → terminal → close` 连续完成；
- 固定 seed/environment/ruleset version 的结构化 replay 精确；
- 100 个 episode：
  - protocol errors = 0；
  - action decode errors = 0；
  - non-terminal forced close = 0；
  - reward component finite；
  - frame/tick 单调；
- 明确记录：
  - 单实例 env-steps/s；
  - 4/8/16 实例吞吐或可达到的最大并行数；
  - CPU、RAM、GPU、RPC 延迟；
- 移动、伤害、死亡/复活、兵线、塔/水晶与终局都有因果测试；
- NULL 不获得 Agent 可归因结构奖励；
- `claim_scope=pixelarena_engineering`，不作策略或 HoK 能力结论。

### 失败决策

- ruleset/schema 未闭合：只修环境与合同；
- public/privileged 泄漏或 side 不对称：M1 No-Go；
- 吞吐过低：只做服务批处理/并行优化；
- 不用训练补偿环境缺陷，不把内部结果宣称为官方环境目标。

---

## M2 — 固定原型 1v1 baseline 与 episode dataset

### 目标

冻结 PixelArena-Structured 的 NULL、deterministic random 和 scripted objective baseline，并生成与训练/评测隔离合同一致的完整 episode 数据。

### 交付物

- scripted teacher 的版本化、确定性策略；
- public、teacher、reward/privileged 三通道物理或逻辑隔离；
- episode/seed/side/opponent source-disjoint split；
- raw、legal、executed factorized action 对齐；
- reward/terminal/outcome 可复算 validator；
- dataset manifest、lineage 和 hash；
- NULL/random/scripted reward audit。

### 通过门

- 每个 split 均按完整 episode 隔离；
- Actor-facing public loader 无 truth、teacher、reward、不可见敌方或训练 entity ID；
- legal mask 仅在瞬时执行/审计路径；
- teacher 与 reward 可由 fresh replay 重算；
- 胜局和失败局均保留，每个生效 action head 有非零覆盖；
- NULL 不获得可归因结构正奖励；scripted baseline 有可解释结构进展；
- 不训练神经策略，不产生 checkpoint promotion。

---

## M3 — 固定原型 1v1 BC

### 目标

用完整 PixelArena scripted-teacher 轨迹让 Agent 获得可闭环运行的初始能力。

### 交付物

- 因子化 recurrent Actor；
- value heads；
-完整 episode dataset；
- 32-sample overfit；
- mini/full BC；
- episode-disjoint evaluation；
- first PixelArena BC bootstrap checkpoint。

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
- illegal/decode/protocol/replay/nonfinite errors = 0；
- 不允许全 WAIT、全 MOVE 或单动作坍塌；
- 对塔/水晶存在非零可归因进展；
- 相比 random 有显著改善；
- 至少形成可继续 PPO 的稳定 checkpoint。

BC 不要求立即达到最终胜率，但如果闭环完全零结构进展，不进入 M4。

---

## M4 — 固定原型 1v1 recurrent PPO

### 目标

在 on-policy 访问状态上学会完整获胜。

### 训练原则

- 从通过 M3 的 BC checkpoint 开始；
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
- illegal/decode/protocol/replay/nonfinite errors = 0；
- Agent 可归因 tower/crystal damage 明确非零；
- 最大 active action 占比 < 90%；
- 与 NULL/random/scripted baseline 相比结构结果改善；
- 3 次独立 seed 中至少 2 次复现主要结论。

阈值是 V5 初始工程门，可在**正式训练前**版本化调整；运行后不得事后降低。

### 失败分类

- reward 上升、win 不升：reward Goodhart；
- BC 好、PPO 退化：on-policy/optimization 或 catastrophic forgetting；
- side 差异大：side bias；
- fixed baseline 好、其他对手崩：opponent overfit；
- 不能完整结束：long-horizon/terminal credit；
- illegal 非零：action/adapter 工程问题。

---

## M5 — Opponent league 与多原型 1v1

### 目标

从固定对手、固定原型扩展到稳定自博弈和共享多原型策略。

### 交付物

- league manager；
- current/best/historical/exploiter/scripted 对手池；
- Elo；
- archetype embedding；
- shared encoder/recurrent core；
- archetype-conditioned heads；
- distillation/retention 机制。

### 通过门

- active checkpoint 对历史池 Elo 提升；
- 无单一 cyclic exploit 主导；
- 至少 3 个 PixelArena 原型达到各自预注册胜率；
- 任一原型不因新增原型退化超过容差；
- side、archetype、opponent 分层均有胜局；
- 多 seed 复现；
- 失败 checkpoint 不进入 league active set。

### 禁止

- 根据同一对手的评测反复调参；
- 用多余原型样本填补某原型零能力；
- 在固定原型尚未通过 M4 时提前做大原型池。

---

## M6 — 3v3 MAPPO/CTDE

### 目标

学习可归因的团队协作，而不是三个共享 reward 的独立挂机 Agent。

### 架构

- shared Actor；
- archetype/role embedding；
- decentralized recurrent hidden；
- centralized Critic；
- MAPPO 或等价 CTDE；
- team/individual/mixed value heads。

### 通过门

- 3v3 环境 100-episode stability；
- 预注册对手池中有稳定胜率/Elo；
- 每个受控角色均有非零角色贡献；
- 不出现长期 lazy agent；
- team reward、individual reward 和 terminal zero-sum 分开记录；
- unseen lineup 和 side 独立 holdout；
- 不读取不可见敌方 truth。

### 失败处理

若共享团队奖励导致 lazy agent，优先修 credit assignment、value decomposition 和角色指标；不通过隐藏 privileged hint 修复。

---

## M7 — RGB → Public Belief State 与 RGB-only closed loop

### M7A — RGB → Public Belief State

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

### M7B — Pixel-derived closed loop

### 目标

使用 M7A 输出驱动冻结或受控微调的 winning strategy。

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

---

## AX — 可选授权 GameCore 校准（非依赖）

AX 不属于 M0–M8 的关键路径。只有新授权合同、非 Git 证据引用、运行时有效许可证、独立任务和安全复核同时存在时才可启动。AX 使用独立 suite、report 和 registry；不得继承或重命名 PixelArena active，不得解锁真实客户端动作。
