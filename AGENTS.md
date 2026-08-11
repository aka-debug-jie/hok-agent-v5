# AGENTS.md — HoK-Agent V5 仓库执行规则

本文件适用于整个仓库，优先级仅低于平台安全规则和适用法律/许可证。

## 1. 必须先读

开始任何实现前，按顺序读取：

1. `PROGRAM_CHARTER.md`
2. `SAFETY_BOUNDARIES.md`
3. `DELIVERY_PROGRESS.md`
4. `ROUTE_NO_GAMECORE_V1.md`
5. `MILESTONES.md`
6. 与当前任务相关的合同文件
7. `DELIVERY_PROGRESS.md` 指定的当前任务文件

`DELIVERY_PROGRESS.md` 是当前状态唯一权威。历史文档只能解释理由，不能覆盖当前状态。

## 2. 项目主路线

V5 的可执行主路线固定为：

```text
项目自有 PixelArena-Structured
→ 固定原型 1v1 baseline/dataset
→ 固定原型 1v1 BC
→ 1v1 recurrent PPO
→ opponent league 与多原型
→ 3v3 centralized-training/decentralized-execution
→ RGB→public belief state
→ RGB-only PixelArena 闭环
→ 真实客户端只读 shadow advisor
```

GameCore 不是主线依赖。它只可作为未来单独授权、默认关闭、独立 registry 的外部校准轨；缺失时不得把整个项目置为 `WAITING_EXTERNAL`。PixelArena 结果只能产生 `pixelarena_internal` 范围的结论，不能表述为 HoK/GameCore 能力。

不得把旧 Pixel-MOBA v4.x 的 K96/scorer/grammar patch 链恢复为 V5 主路线。

## 3. 不可降低的安全边界

- 闭环动作只允许发送到：
  - 腾讯明确授权的研究 GameCore；
  - 项目自有、明确隔离的 PixelArena/测试环境。
- 真实商业客户端只允许视频/回放读取和 Linux 端建议。
- 禁止实现或调用 ADB 触控、scrcpy 控制、HID/UHID、`/dev/uinput`、Accessibility 自动化、root、hook、内存读写、协议拦截、客户端修改、反检测或绕过反作弊。
- 不得使用真实账号进行自动化对局。
- GameCore 许可证、二进制、密钥、账号和设备标识不得进入 Git。
- privileged state 可以进入训练 Critic、teacher、loss 和 evaluator，但不得进入导出 Actor 的 observation、hidden 或运行时排序。

发生边界冲突时，改成 sandbox-only 或 read-only advisory 等价方案；不能通过“仅用于测试”绕过。

## 4. 研发规则

- Python 主训练栈使用 3.11+；上游 SDK 若限制旧版本，放进独立服务/容器，不污染 learner 环境。
- public API 必须有类型标注。
- 环境、策略、训练、评测、数据、服务边界必须分层。
- 每个长作业前依次通过：
  - unit；
  - environment smoke；
  - 小样本 overfit 或 fixed-seed reward check；
  - 10–30 分钟 mini run；
  - 才允许 full run。
- 一次实验最多改变一个主要因果变量；允许小规模预注册 sweep，但禁止失败后无限扩展搜索空间。
- 不根据正式评测结果选择 checkpoint、调 reward 或重写阈值。
- 神经网络数值使用容差、KL、margin 和 rank stability 评估；不要求不同 CUDA batch 形状逐字节相同。
- 环境 transition、reward、terminal、action decode、split 和 artifact hash 必须精确可复核。

## 5. 证据语义

以下指标只能诊断，不能晋升：

- train loss；
- cached Top-1/Top-3；
- teacher action recall；
- 单个 margin；
- reward 曲线；
- 参数 hash 变化。

checkpoint 晋升必须依赖闭环：

- 完整 episode 结束率；
- 胜率和置信区间；
- Elo/league 结果；
- 红蓝方平衡；
- 塔/水晶与可归因进展；
- illegal/协议错误；
- 动作坍塌与对手过拟合检查。

## 6. 工作流

开始任务时：

1. 在 `DELIVERY_PROGRESS.md` 把任务置为 `IN_PROGRESS`；
2. 记录范围、输入 authority、明确不做什么；
3. 实现代码、测试、配置和文档；
4. 运行与任务风险相称的检查；
5. 写入真实命令、结果、artifact 路径、剩余风险和下一任务；
6. 只有验收全部通过才标记 `DONE`。

外部许可证或二进制缺失时，仅可选 GameCore 轨使用 `WAITING_EXTERNAL` 或 `NOT_AVAILABLE`；本地 PixelArena 主线继续按自身门控推进。不得描述成代码失败，也不得伪造外部结果。

## 7. 源码控制

- 一个 commit 对应一个里程碑切片或一个可解释修复。
- 不把 runtime、checkpoint、视频、ABS、许可证、密钥或真实账号信息提交到 Git。
- 所有实验配置必须版本化。
- 不覆盖失败报告；失败结果是保留证据。
- V5 不从 legacy v4.x checkpoint、optimizer 或 K96 配置恢复训练。

## 8. 当前任务

当前任务只取自 `DELIVERY_PROGRESS.md`。`TASK_000` 和 `TASK_005` 是历史任务，不得覆盖当前路线。

当前 `TASK-010` 只允许完成：

- scoped evidence schema；
- V5 原生 PixelArena ruleset 与 service；
- public/privileged projection；
- replay/snapshot、baseline、stability 与 throughput；
- M1 验收报告。

不允许在 TASK-010 启动 BC/PPO，不允许宣称已训练会玩王者的 Agent。
