# V5 无 GameCore 主路线

## 1. 路线决定

本项目已确认没有 GameCore、许可证或对应服务。它们不再是 M0–M8 的依赖，项目也不再以 `WAITING_EXTERNAL` 作为全局状态。

V5 的可执行目标是：在项目自有 PixelArena 中建立可复核的结构化 MOBA 策略研究线，依次完成固定原型 1v1、多原型 1v1、3v3、RGB-only 闭环，最后将真实商业客户端的只读视频或回放用于智能教练。仓库名保留为 `hok-agent-v5`，但任何报告标题和能力表述必须写明环境范围。

GameCore 只保留为默认关闭的可选外部校准轨。以后即使获得授权，也必须新建立项、独立 registry 和独立报告；它不会重写或阻塞本路线。

## 2. 环境与证据层级

| 层级 | 环境 | 允许的结论 | 禁止的结论 |
|---|---|---|---|
| A0 | deterministic mock | RPC、schema、hash、replay、错误关闭 | 策略能力、胜率、checkpoint 晋升 |
| A1 | PixelArena-Structured | 指定 ruleset 下的 1v1/3v3、Elo、项目内部晋升 | HoK、GameCore 或商业客户端能力 |
| A2 | PixelArena-RGB | 指定 renderer 下的感知和 RGB-only 闭环 | 真实客户端闭环能力 |
| A3 | 真实客户端只读视频/回放 | 域审计、建议延迟、置信度、人工有用性 | 自动动作、在线 RL、胜率、策略晋升 |
| AX | 可选授权 GameCore | 未来独立授权后的外部校准 | 自动继承 PixelArena 结论 |

所有 E1–E3 报告必须绑定：

- `claim_scope`；
- environment family 与 identity hash；
- ruleset ID/major version；
- observation/action/reward schema version；
- mode、原型或 lineup registry；
- renderer family/version（像素路径）；
- evaluation suite version。

不同 environment family、ruleset major 或 claim scope 的胜率和 active checkpoint 不得合并。

## 3. 任务图

```text
TASK-006  路线纠正与合同同步
   ↓
TASK-010  PixelArena-Structured 规则集、服务与吞吐
   ↓
TASK-020  固定原型 1v1 scripted baseline 与 episode dataset
   ↓
TASK-030  因子化 GRU Actor 与行为克隆
   ↓
TASK-040  固定原型 recurrent PPO 与内部晋升
   ↓
TASK-050  opponent league 与多原型 1v1
   ↓
TASK-060  3v3 ruleset 与 CTDE/MAPPO
   ↓
TASK-070  RGB → public belief 与蒸馏
   ↓
TASK-080  PixelArena RGB-only closed loop
   ↓
TASK-090  真实客户端只读 Shadow Coach
```

每项任务只能由前一项的验收报告解锁。`TASK-010` 不训练，`TASK-020` 不训练神经策略，`TASK-030` 只做 BC，PPO 从 `TASK-040` 开始。

## 4. 里程碑定义

| 里程碑 | 任务 | 可交付能力 | 进入下一阶段的硬门 |
|---|---|---|---|
| M0 | TASK-000 + TASK-006 | bootstrap 与路线治理 | mock/合同/安全通过，权威文件一致 |
| M1 | TASK-010 | 可完整结束对局的 PixelArena-Structured | 100 局稳定、确定性/重放、因果规则、吞吐报告 |
| M2 | TASK-020 | scripted/NULL/random baseline 与完整数据集 | source-disjoint、公开/特权隔离、教师和奖励可复算 |
| M3 | TASK-030 | fixed-archetype BC bootstrap | 32 样本 overfit、100 局闭环、结构进展、无动作坍塌 |
| M4 | TASK-040 | fixed-archetype PPO active | 预注册 200 局、胜率/side/工程/多 seed 门 |
| M5 | TASK-050 | league 与至少 3 个原型 | 每原型独立通过、retention、matchup 无隐藏崩溃 |
| M6 | TASK-060 | 3v3 CTDE baseline | 每角色非零贡献、无 lazy agent、unseen lineup holdout |
| M7 | TASK-070/080 | public belief 与 RGB-only PixelArena 策略 | renderer-disjoint、无 privileged side channel、闭环保持率 |
| M8 | TASK-090 | 真实客户端只读教练 | 无控制依赖、OOD/ABSTAIN、隐私、可追溯人工验收 |

## 5. 初始评测门

### M1 工程门

- 100/100 episode 完整结束；
- protocol、decode、illegal-executed、replay、nonfinite error 均为 0；
- 同 build/config/seed/action 序列的 transition 与终局精确重放；
- 移动、伤害、死亡/复活、兵线、结构、水晶和终局均有可测试因果；
- NULL、random、scripted baseline 结果可解释；
- 单实例及可达到的并行吞吐有真实报告。

### M3 BC 门

- 32 样本 teacher-equivalent 至少 99%；
- 所有生效 action head 有有限非零梯度；
- 100 局完整结束率至少 99%；
- illegal/decode/protocol 为 0；
- 不允许全 WAIT、全 MOVE 或单动作坍塌；
- 相比 NULL/random 有非零可归因结构进展。

### M4 PPO 内部晋升门

在首个正式 candidate 前冻结阈值和 suite；初始目标为：

- 200 局，side 与 opponent 平衡；
- 总胜率至少 70%，任一 side 至少 60%；
- 完整结束率至少 99%；
- protocol/decode/illegal/replay/nonfinite 为 0；
- Wilson 95% 下界优于 active，或不退化且其他主指标严格改善；
- 塔/水晶可归因进展非零，最大 active action 占比低于 90%；
- 3 个独立训练 seed 中至少 2 个复现。

这些门只产生 `claim_scope=pixelarena_internal` 的 active checkpoint。

## 6. 配置与 artifact 迁移

`TASK-010` 的第一切片必须先引入 scoped evidence schema，取代含糊的裸 `formal` 语义：

```yaml
evaluation_tier: E0|E1|E2|E3
promotion_eligible: true|false
claim:
  scope: pixelarena_internal
  environment_family: pixelarena
  ruleset_id: pixelarena_1v1_v1
  hok_capability_claim: false
  gamecore_equivalence_claim: false
```

在 schema v2 和独立 PixelArena registry 完成前，现有 v1 schema 继续 fail-closed：mock 和 PixelArena 均不能晋升。这是迁移锁，不是长期路线定义。

## 7. Legacy 复用边界

legacy 已有 deterministic core、snapshot/replay、public/privileged 三通道、episode split、raw/executed action 和 team-core 等工程模式。V5 只能按思想和小型纯 Python 合同重新实现并写新测试。

禁止迁移：

- checkpoint、optimizer、cache、K96、scorer、grammar、旧阈值和能力结论；
- 旧 ruleset 数值、placeholder renderer、设备配置和真实客户端路径；
- 整体复制 legacy `src/`；
- 把 legacy 的 canary、静态 CE 或工程通过解释为 V5 能力。

## 8. 永久边界

- 真实商业客户端永不接收自动动作；
- 不实现 ADB input、scrcpy control、HID/UHID、uinput、Accessibility、root/hook、内存或协议自动化、反检测；
- 不用真实账号自动对局或在线 RL；
- legal mask 不进入 Actor encoder/hidden/普通 observation；
- truth、teacher、reward、不可见敌方和训练 entity ID 不进入导出 Actor；
- 不把 mock 或 PixelArena 结果称为 HoK/GameCore 能力；
- 不因 GameCore 不存在而暂停本地主线，也不猜测、下载或伪造其许可证与二进制。

## 9. 可选 GameCore 轨

可选轨默认：`enabled=false`、`dependency_of_main_route=false`、transport/evaluation/promotion 全锁、registry 独立。只有新的书面授权、非 Git 证据引用、运行时有效许可证、独立任务和安全复核同时存在时才可启动。它永不授权真实客户端控制。
