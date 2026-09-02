# Hierarchical Policy v0 开发协议

## 1. 状态与目的

当前状态：`E1B_WEAK_LABEL_COVERAGE_FAILED`。E1a工程诊断通过但不晋级；E1b终局OCR覆盖
失败；完整E1门尚未通过。

本协议把项目现有的 RGB 感知、完整 episode、双指针执行和离线训练能力收束成一条新的
分层策略开发线。它是开发合同，不是实现、训练结果或能力证明。Global Agent、Human IfO、
T8、Operation Policy 和 Movement Teacher 的冻结结论保持不变；本路线不通过调低旧门槛、
重用旧失败标签或扩大旧模型来重新开启它们。

长期方向保留：离线视觉预训练、Macro/Movement/Combat 三个策略、独立视觉事件系统、统一
Replay 和真实闭环后训练。首版只实现长期架构中形成自动训练数据闭环所必需的部分。

## 2. v0 的能力边界

v0 只面向预先声明的后羿、固定射手角色和一条固定分路。蓝方走下路，红方走上路。英雄、角色、分路和分方
在 episode 开始前完成绑定，用于选择配置、英雄 adapter 和确定性开局模板；这些元数据不作为
RGB Actor 的张量输入，也不进入视觉编码器或时序隐藏状态。自动分方结果一旦绑定，在当前
episode 内不可漂移。

v0 学习输出限定为：

- Macro：`FARM_LANE`、`PUSH_STRUCTURE`、`ENGAGE`、`DISENGAGE`、`RECALL`、`HOLD`。
- Movement：`NONE` 加八方向 `N/NE/E/SE/S/SW/W/NW`。
- Combat：`WAIT`、`BASIC_ATTACK`、`SKILL1`、`SKILL2`、`SKILL3`。

发送信号、连续瞄准、目标单位选择、组合连招、连续摇杆向量、MoE、多英雄和多智能体不属于
v0。购买、技能可用性、动作掩码、开局模板、死亡停止、指针冲突和所有输入保护继续由确定性
模块负责。

开发准备阶段不授权手机模型控制、在线梯度更新或新增输入通道。以后即使进入真实闭环，也
只能复用现有 Mobile Operation Base 在所有身份和运行时保护通过后连接项目所有者的自建测试
App；本协议不允许把该接口扩展到其他客户端、包或账号。

## 3. 目标架构

```text
                         +-----------------------+
RGB capture ------------>| FrameBus              |
                         | immutable observation |
                         +-----------+-----------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
      +----------------------+                 +----------------------+
      | VisualEventEngine    |                 | PolicyBundle v0      |
      | RGB-only, frozen     |                 | RGB-only             |
      +----------+-----------+                 +----------+-----------+
                 |                                        |
                 |                             +----------+----------+
                 |                             | shared temporal     |
                 |                             | representation      |
                 |                             +----+--------+-------+
                 |                                  |        |
                 |                              Macro     Movement  Combat
                 |                                  +--------+-------+
                 |                                           |
                 |                                deterministic Router
                 |                                           |
                 |                                dual-pointer Executor
                 |                                           |
                 +-------------------+-----------------------+
                                     v
                        RewardHub + UnifiedTransitionStore
                                     |
                              sequence Replay sampler
                                     |
                         offline pretrain / online learner
```

三个策略在逻辑上独立训练和评估，但不是三个完整 RGB 大模型。主画面、小地图和 HUD 可以使用
各自的轻量视觉 stem；融合后的时序表征由三个 Head 共享。一次 observation 只产生一个
`observation_id`。每个 proposal 分别记录产生它的 source observation 和本步 applied
observation；未复用的三个 Head 必须使用当前 source ID，跨频率复用必须显式标记
`carried_forward=true`、仍在 `valid_until_ns` 内，并保持同一 `policy_bundle_version`。

## 4. 模块合同

### 4.1 FrameBus

FrameBus 是唯一帧入口，采用 latest-frame-only，不允许无界队列。它为每次观察产生不可变的
`FramePacket`：

```text
observation_id
capture_start_ns / capture_end_ns
main / minimap / hud derived-frame hash
frame_bundle_ref
capture_source_class
capture_latency_ms
```

`frame_bundle_ref` 只能是外部数据根下的匿名 basename 或内容 ID，不能记录源视频路径、设备
序列号、账号、坐标或凭据。推理、事件检测和 transition 记录共享同一 FramePacket，不允许
各模块自行再次截图。

### 4.2 VisualEventEngine

VisualEventEngine 与 Actor 完全独立。在线输入只能是 RGB FramePacket 和固定 ROI 配置；OCR、
颜色、几何和时序模型都属于 RGB 派生方法。执行动作时间可以用于离线候选片段检索和弱标签
审计，但不能作为在线事件模型输入，也不能改变事件置信度。

E0 先实现 Frame/State/Event 协议、事件去重和版本/hash。E1 只实现：

```text
GAME_END / WIN / LOSS
DEATH / RESPAWN
SELF_HP_DELTA
```

`TOWER_DAMAGE / TOWER_DESTROYED` 为 E1.5。只有固定布局下的离线片段证明其身份、可见性和
时序去抖可靠，才加入 reward；它不阻塞第一条完整 transition。

E1a 使用128x128 main RGB中央区域的绿色血条候选，连续三帧确认死亡与复活。首个宽度上限
16像素的诊断在challenge session产生7次假死亡，报告已保留；唯一一次实现修复把允许宽度
改为24像素，没有改变确认帧数或通过门槛。修复后两个正常train session零假死亡，dev产生
一次死亡和一次复活，challenge零假死亡。通过的报告为
`HOK_LARGE_ROOT/audit/hierarchical-event-e1/health-engineering-v2-width-repair/report.json`，
report SHA-256为`454f28198d6f388f3975eadd3770d256920a967d02e921e91bafb9dcfc2d4a90`。
这是dev已见的工程诊断，不是独立语义准确率。`SELF_HP_DELTA`只保留候选事件，数值精度、
Reward和promotion仍为false。

E1b按冻结hash顺序选择8个train和4个dev完整视频，只在末24秒以1 Hz挖掘候选帧，时间不
作为标签。PP-OCR仅保留胜利、失败和结果页白名单命中，不保存任意OCR文字。GAME_END覆盖为
train `0.125`、dev `0.5`，WIN/LOSS覆盖均为`0`，因此冻结失败。报告位于
`HOK_LARGE_ROOT/audit/hierarchical-event-e1/terminal-coverage-v1/report.json`，report SHA-256为
`dfbae056925e179f93cd475d081ac4731aba86d3c37dbf422e0ba95efa363b4f`。多数录像在水晶爆炸阶段
结束，没有结果页或胜负文字；不得通过调低OCR置信度、延长tail或提高采样率重开该路线。

动态事件由短视频窗口判断，静态状态由按视觉状态分层采样的截图判断。禁止按固定对局时间
抽取静态样本，以免模型学习时间先验。每个事件必须包含 `event_id`、起止时间、旧值、新值、
变化量、置信度、ROI 和去重键；同一事实只能产生一次 reward。

Reward Engine 在一批 Policy 训练期间保持冻结。更新 EventEngine 必须产生新版本/hash，完成
独立离线审核，并从下一个 episode 边界开始生效。Actor 与 Reward 模型不得在同一在线循环中
同时更新。

### 4.3 PolicyBundle

首个物理实现是一套 Bundle：视觉 stem、共享 Temporal Core、三个策略 Head 和英雄 adapter。
Critic 只在训练时存在，v0 先使用一个标量 `V(s)`。奖励分量始终单独记录，但不预先建立五个
Value Head。

建议的第一档参数预算是 Policy 35–60M、EventEngine 5–15M。它只是闭环验证档，不是永久
上限。扩容规则固定为：

| 档位 | Policy 参数 | 进入条件 |
|---|---:|---|
| v0-S | 35–60M | 首个实现，验证因果、可学性和吞吐 |
| v0-M | 80–120M | 同数据同训练预算下，v0-S 明确欠拟合且延迟合格 |
| v1-L | 150–220M | v0-M 仍有容量瓶颈，并通过完整 episode Benchmark |

模型变大、MoE、额外 Critic 或更高分辨率不能代替数据覆盖与因果正确性。任何 Bundle 只允许
在 episode 边界原子切换；切换、死亡、复活和新 episode 都必须清空或重置 recurrent state。

### 4.4 多频率策略

FrameBus 的开发目标是 10 Hz，5 Hz 是首轮最低可测运行频率。Macro 约 1 Hz，Movement 和
Combat 约 10 Hz。未到调度时刻的 Head 可以复用仍在 `valid_until_ns` 内的 proposal，但必须
保留原 source observation，并绑定当前 applied observation；过期 proposal 必须变为
`HOLD/NONE/WAIT`，不能静默复用。

Movement 的离散分类不等于间歇移动。执行器保持 pointer 0 的真实生命周期：第一次移动发
DOWN，方向改变只发 MOVE，方向未变不重复发送，`NONE`、硬停止或清理才发 UP。方向滞回和
最小保持时间由确定性执行器管理，防止 `E/NE` 抖动。持续时间和摇杆幅度不由 v0 Head 输出。

Combat 只提出按钮类别。技能亮度、冷却、英雄固定槽位行为和 pointer 1 可用性由 action mask
和英雄 adapter 决定。被 mask 的 proposal 记录为 rejected/no-op，不自动替换为另一个学习
动作。

### 4.5 Deterministic PolicyRouter

Router 没有可训练参数，只执行以下合同：

- 检查 source/applied observation ID、carried-forward 状态、Bundle 版本、proposal 时效和置信度。
- 应用死亡/终局/未知画面、技能可用性和动作词表 mask。
- 解决 pointer 资源、移动锁和执行优先级冲突。
- 将 Macro 的约束映射为 Movement/Combat 的允许范围。
- 保留 requested、masked 和 executed 三种动作，禁止把最终执行动作反写成模型预测。

如果未来需要学习“战斗时保持、减速、锁定或覆盖移动”，该决策由 Macro 或 Combat Head 输出
有限的 movement policy proposal；Router 仍保持确定性。

### 4.6 串行因果 step

每个训练 step 必须遵守：

```text
capture observation
→ run scheduled heads on the same observation_id
→ route proposals
→ synchronously apply/acknowledge action-state update
→ wait the frozen settle interval
→ capture next observation
→ update VisualEventEngine
→ calculate reward components
→ append transition atomically
→ if done: end episode
```

pointer 0 可以在两个 step 之间持续按下；`action_dispatch_ack_ns` 表示方向状态更新已经被确认，
不是角色已经走完。下一帧必须晚于该确认和稳定窗口。终局 transition 必须先写入 Replay，
随后才能结束 episode。

首次输入失败、一次重试及最终状态都进入 executed-action 记录。只有最终确认成功才计入成功
动作；第二次仍失败时 transition 记录为不可训练并终止 episode。

### 4.7 UnifiedTransitionStore

项目只建立一个物理 TransitionStore。E0 使用标准库 SQLite WAL 保存事务式元数据，帧本体仍在
外部数据根，通过匿名 basename 和 hash 引用。`demo`、`sim`、`controller`、`online` 和
`offline_video` 是 source，`failure` 是 tag，不是多套 schema。Sampler 在统一数据上提供
demo、online、failure 和 joint 视图。
序列 replay 必须按完整 episode 保存，支持 recurrent burn-in，并机械排除：

- 因果时间顺序失败；
- FramePacket 缺失或 hash 不匹配；
- Bundle/EventEngine 版本缺失；
- terminal transition 未写入；
- 相邻 step 不连续；
- 明确标记 `training_eligible=false` 的诊断数据。

checkpoint 包含模型、optimizer、scheduler、replay cursor、随机种子和 schema/hash 绑定。写入采用
临时文件加原子替换；只在 episode 边界加载新 PolicyBundle。

机器可读字段见
[`game_rules/hierarchical_transition_contract_v0.json`](../game_rules/hierarchical_transition_contract_v0.json)。

## 5. 数据与训练用途

| 数据 | v0 允许用途 | 不允许的推断 |
|---|---|---|
| 约 50 GB 历史 RGB | SSL、时序一致性、域适配、EventEngine 候选片段 | 没有同步动作时不得直接 BC |
| 已有近似对齐的人类录像 | 表征与定性诊断 | 不得宣称可靠 Movement/Combat 标签 |
| 合格 RGB + executed action | 低层 Movement/Combat BC 或 DQfD warm-start | 自动固定日程不得当作战术专家 |
| GlobalArena/PixelArena | Macro、Critic 和完整 episode 预训练 | 结构化真值不得进入 RGB Actor |
| 自建测试 App 新 episode | 通过闭环门后进入 online/failure replay | EventEngine 未冻结时不得在线 RL |

三个策略先分开训练：Macro 使用完整仿真 episode；Movement 只使用因果同步且方向覆盖合格的
动作数据；Combat 使用固定英雄、固定技能槽位的同步动作和结果事件。共享视觉编码器先冻结，
分别证明 Head 可学，再组装同一 Bundle。联合训练只允许低学习率更新共享 Temporal Core，且
必须与分头 checkpoint 做完整 episode 对照。

首个在线算法不预先同时实现多个家族。v0 保持离散动作后，首选候选是 recurrent off-policy
Q-learning（R2D2-style）。DQfD 只在存在合格同步示范时使用；P-DQN 只在后来引入离散加连续
参数动作时评估；PPO 只有在并行环境吞吐足够且离线/离策略路线明确受限时再评估。

## 6. 开发阶段与退出门

| 阶段 | 实现内容 | 退出门 |
|---|---|---|
| D0 | 本协议、示例配置、机器合同 | JSON 可解析，权威文件同步，`make check` 通过 |
| E0（PASSED） | FrameBus、VisualState/Event、Transition validator | 16项聚焦测试通过；全仓329项测试通过 |
| E1a（诊断通过，不晋级） | 中心血条、死亡/复活、HP变化候选 | 一次宽度修复后工程门通过；Reward仍关闭 |
| E1b（FROZEN FAILED） | 终局OCR弱标签与结果页覆盖 | GAME_END 0.125/0.5，WIN/LOSS 0/0；不重调 |
| E1c（下一候选） | 水晶摧毁动态短视频转场 | 新合同后才可实现；不能替代WIN/LOSS真值 |
| E1（未通过） | 合并终局、死亡/复活、自身血量 | 独立语义与HP精度证据齐全后才能接RewardHub |
| L0 | 离线录像 replay：事件→reward→transition | terminal 先存后停；无重复事件、断步或版本缺失 |
| L1 | 自建测试 App 一个完整 episode，模型不更新 | Capture→Action→Event→Reward→Replay 完整可恢复 |
| L2 | 连续 3 局 | 无动作积压、帧引用损坏、checkpoint 损坏 |
| L3 | 连续 10 局和固定 Benchmark | 延迟、终局率、失败码和动作有效率可重复 |
| P0 | 50 GB RGB SSL 和 episode-disjoint probe | 小样本 overfit 先过；dev 优于时间/打乱基线 |
| P1 | Macro/Movement/Combat 分头预训练 | 每个 Head 独立通过自己的可学性与负对照 |
| P2 | PolicyBundle-v0-S 组装 | 同 observation/version，多频率调度无陈旧动作 |
| R0 | Head-only online RL | 固定 Benchmark 明显优于初始化 Bundle |
| R1 | 解冻 Temporal Core | 完整 episode 不退化后才保留 |
| S0 | 容量复盘 | 只有欠拟合和延迟证据才进入 v0-M |

事件权重、RL 序列长度、burn-in、更新频率和模型宽度在相应阶段的训练合同中冻结，不在本开发
准备协议中凭经验硬编码。

## 7. 最小开发文件图

代码按以下责任拆分；前三项已在 E0 实现：

```text
src/hok_agent/frame_bus.py              immutable FramePacket / latest-frame transport
src/hok_agent/visual_events.py          VisualState/Event and E0 fusion
src/hok_agent/transition_store.py       schema validation and atomic append
src/hok_agent/reward_hub.py             versioned event-to-component mapping
src/hok_agent/hierarchical_policy.py    shared temporal bundle and three heads
src/hok_agent/policy_router.py          deterministic proposal arbitration
tests/test_frame_bus.py
tests/test_visual_events.py
tests/test_transition_store.py
```

E0 没有创建 RewardHub、模型或在线入口。E1a在`hierarchical_e1.py`实现离线中心血条诊断、
时序事件和自校验报告；RewardHub继续等待完整E1门通过。

## 8. 当前停止条件与不确定性

- 32 样本过拟合失败：停止扩数据或扩模型，先修标签、空间信息和时序对齐。
- E1 在 session-disjoint dev 上不能稳定工作：不得生成 RL reward。
- 单局存在时间倒序、动作积压、terminal 丢失或 checkpoint 无法恢复：不得进入三局。
- v0-S 训练损失仍高是容量候选；训练很低而 dev 差是数据/泛化问题，不能靠扩模型解决。
- 塔血量、敌人血量、经济和经验仍缺少稳定身份与时序证据，当前只能列为后续事件。
- 现有 50 GB 数据足够开始视觉预训练，但没有证据证明它足以训练成熟的三策略闭环。

当前唯一下一开发任务：设计E1c水晶摧毁动态短视频转场合同，先区分GAME_END转场与普通
高亮战斗；不开test，不接RewardHub，不把视频结束时间作为标签。
