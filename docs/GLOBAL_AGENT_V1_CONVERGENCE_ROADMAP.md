# Global Agent v1 收束路线

## 唯一主线

```text
现有安全执行底座
→ 规则驱动的完整仿真对局
→ 完整episode数据与宏观标签
→ RGB高层意图模仿学习
→ 一轮仿真内DAgger修正
→ Human IfO共享时序表征
→ 仿真逆宏观动力学与人类视频伪标签
→ Human-BC
→ 移动端只读Shadow
→ 受限单英雄完整对局
```

Global Agent v1 只控制一个固定英雄。学习模型只输出宏观意图和目标区域；移动、战斗、购买、英雄技能行为、布局和安全停止继续由确定性模块执行。

## 北极星与WIP限制

所有实验按以下字典序评价，而不是平均局部指标：`safety_violations=0`、
`non_timeout_terminal`、`tower_progress`、`stuck_time_ratio`、
`teacher_fallback_rate`、`win_rate`，最后才是局部F1。每个新任务必须说明它将改善哪一个
一级episode指标；否则不进入主线。

同时只允许一个全局功能任务和一个最高频阻塞bug。每日台账固定为：

```text
CURRENT GOAL:
BLOCKING FAILURE:
NEXT ACCEPTANCE COMMAND:
DO NOT WORK ON:
```

## 阶段与通过条件

| 阶段 | 工作 | 通过条件 | 未通过时的处理 |
|---|---|---|---|
| 0 | 冻结现有T8/Operation失败与通过证据 | 新旧lineage互不混用 | 不再扩展局部T8路线 |
| 1A | First Full Game：中路单线规则闭环 | `PASSED`：seed 17 水晶终局，安全违规0 | 已关闭 |
| 1B | 20局规则稳定性 | `PASSED`：19/20水晶终局、20/20塔推进，死亡恢复100%，卡住8.97% | 已冻结精确回归 |
| 2 | 固定40 train/10 dev完整episode | `PASSED`：3,276行，episode级split，无test | 已冻结manifest |
| 3 | Seed-0 RGB宏观行为克隆 | `PASSED`：TCN优于time-only和shuffle，纯学生7/10终局 | 已冻结BC |
| 4 | 唯一一轮DAgger | `PASSED`：765边界样本，9/10终局，平均塔伤提升，安全违规0 | 禁止第二轮 |
| 5 | 103/23真实视频适配与离线回放 | `PASSED`：适配非退化，391行回放，输入0 | 等待独立Shadow审查 |
| 5.5 | Fresh 20-seed holdout 与模型选择 | `PASSED`：Dagger 18/20，adapted 17/20；选择Dagger | 已冻结，禁止重训 |
| 5.6 | 最小challenge pack | `FAILED`：学生仅通过2/6固定语义状态 | 阻止所有输入；不调阈值或重训 |
| 6A（诊断） | 移动端只读Shadow | v1 与本地ROI v1.1均`RUNTIME_PASSED`；候选多样性未证明 | 10分钟与输入关闭 |
| 6B（未开放） | 受限单英雄完整对局 | 必须另行修复challenge并审查 | 当前禁止 |
| Gate A | 窄Human cohort与跨域共享表征 | 时序可学且Human→Sim邻域不坍缩 | 只允许一次表征修复 |
| Gate B | 多horizon逆宏观动力学 | 真实transition gain；current-only不得胜出 | 否则停止IfO |
| Gate C | 伪标签坍缩门 | 覆盖、类别、持续时间通过；dev不可训练 | 仅一次Sim calibration |
| Gate D | Human-BC与只读验证 | Sim保持、challenge、真实候选多样性通过 | 60秒后才审查10分钟Shadow |
| H5（非计划） | latent transition imitation | 仅Gate D通过仍有明确行为缺口 | 不作为主线阻塞项 |

## 固定模型边界

- RGB学生输入：主画面、小地图、HUD的16帧因果序列。
- RGB学生输出：9类`intent_id`、10类`target_zone_id`、6类辅助`scene_id`。
- `ABSTAIN`由安全和置信度派生，不作为首版学习标签。
- 规则教师可以使用仿真结构化真值；RGB学生、移动端模型和Shadow不能使用。
- 当前移动教师改为`玩家位置 + target_zone中心 → 八方向`，不再以最近红色目标决定宏观目的地。
- TargetZoneNavigator只追踪下一个语义waypoint，不直线冲区域中心。第一版仅启用
  `OWN_BASE → MID_ENTRY → MID_CENTER → ENEMY_MID_TOWER → ENEMY_HIGH_GROUND → ENEMY_BASE`。
- 当前战斗执行器改为由`combat_mode`路由，不再以固定轮转作为策略标签。

## 第一版能力开关

9类意图和10类区域枚举保持不变，但首个checkpoint只启用：

```text
FARM_LANE, PUSH_STRUCTURE, ENGAGE, DISENGAGE, RECALL
OWN_BASE, MID_LANE, ENEMY_BASE, HOLD_CURRENT_ZONE
```

多线、野区和中立目标不属于当前五阶段。规则教师的首版标签优先级固定为：

```text
ABSTAIN → RECALL → DISENGAGE → ENGAGE → PUSH_STRUCTURE → FARM_LANE
```

教师决策必须是RGB可观察状态的函数；近似相同的可观察状态不得因隐藏seed、脚本编号、
未来事件或未渲染真值获得相反标签。

## 进展监视与失败分类

每个宏观命令记录目标、最小保持时间、最大无进展时间、进展信号和恢复次数。默认：
2 Hz决策、1.5秒最小保持、3秒最大无进展、每意图最多2次重规划、每局最多3次恢复。
普通恢复依次为备用waypoint、全局重规划、`NAV_STUCK`安全结束；死亡、未知画面、
基地受威胁和紧急撤退可以立即覆盖。

每个非正常终局只能归入一个主失败码：

```text
NAV_STUCK, NO_WAVE_PROGRESS, COMBAT_LOOP, RECALL_LOOP, DEFEND_LOOP,
DEATH_RECOVERY_FAILED, NO_TOWER_DAMAGE, CRYSTAL_NOT_REACHED,
SAFETY_STOP, TIMEOUT_OTHER
```

每10或20局自动统计失败码；只修最高频失败。局部模块只有在阻塞First Full Game、
造成至少20%失败、安全违规或可明显改善一级episode指标时才准入。

## 唯一一轮DAgger授权

```text
Teacher only → 25% student authority → student-primary simulator dev
```

DAgger只收集学生低置信、与教师分歧、卡住和恢复状态，不重复容易导航帧，不启动第二轮。

## 当前不做的事

- 不再新增敌人可见、按钮就绪、目标可攻击等局部研究lineage。
- 不训练直接点击坐标、瞄准坐标、目标单位或完整连招。
- 不把现有手机动作日志直接用于离线强化学习。
- Human-BC通过前不运行PPO；H5仅为可跳过的仿真内提升。
- 不在仿真完成前让移动端模型输出控制动作。
- 不在单英雄闭环前做多英雄或五智能体协作。

## 当前状态

```text
CURRENT GOAL: freeze Global Agent v1 model evidence
BLOCKING FAILURE: key observable factors are weak, while the only authorized encoder update regresses full episodes to 12/20
NEXT ACCEPTANCE COMMAND: none; frozen Dagger is the permanent Global Agent v1 policy
DO NOT WORK ON: scenario-card training, preprocessing variants, second DAgger, early PPO, 10m Shadow or phone input
```

独立授权的活跃场景只读Shadow已验证传输与运行时安全，但未证明候选多样性。它没有开放
手机输入。任何受限完整对局仍必须先以独立合同解决challenge语义泛化失败。

当前修复主线是Human IfO Bridge v1：人类视频教行为分布，GlobalArena教状态转移对应的
宏观动作，冻结DAgger保护完整终局能力。场景卡片仅保留为备用诊断。权威协议见
`docs/HUMAN_IFO_V1_PROTOCOL.md`。
