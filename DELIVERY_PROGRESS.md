# HoK-Agent V5 交付进度

> 本文件是当前状态唯一权威。  
> 历史讨论、提示词、旧仓库台账和实验文档不能覆盖本文件。

- 最后更新：`2026-08-11`
- 当前阶段：`V5-M0-BOOTSTRAP`
- 当前任务：`TASK-000 REPOSITORY_AND_ENVIRONMENT_BOOTSTRAP`
- 当前状态：`IN_PROGRESS`
- 当前 promotion：`无 checkpoint 可晋升`
- 真实客户端边界：`READ_ONLY_SHADOW_ONLY`
- GameCore 连接状态：`UNKNOWN / 需要现场预检`
- GameCore 许可证状态：`UNKNOWN / 不得假设已获得`

## 1. 已冻结决定

1. Pixel-MOBA v4.x 保留为 legacy evidence，不再创建 v4.4.1。
2. V5 不从 legacy checkpoint、optimizer 或 K96 profile 恢复。
3. 结构化 Strategy Agent 先于 Vision Student。
4. checkpoint 只由闭环评测晋升。
5. 真实商业客户端不执行自动动作。
6. PixelArena 降级为 test double、视觉课程和 counterfactual 环境。
7. 授权 GameCore 是最终策略能力权威；接入失败时不能伪造替代结论。

## 2. 当前任务范围

见 `TASK_000_BOOTSTRAP.md`。

当前只允许：

- 新仓库骨架；
- Python 3.11 learner；
- 独立旧版 SDK service 契约；
- mock environment；
- run/evaluation schema；
- upstream 和许可证 preflight；
- random/common-AI 基准 harness；
- 环境吞吐基准框架；
- unit/CI。

当前不允许：

- 长 BC/PPO；
- 多英雄；
- 3v3；
- pixel policy；
- 真实客户端动作；
- 旧 v4.x 继续训练。

## 3. 外部阻塞

| 项目 | 状态 | 处理 |
|---|---|---|
| GameCore license | 未确认 | 实际检查；缺失则标记 `WAITING_EXTERNAL` |
| GameCore binaries | 未确认 | 不提交 Git；按上游条款获取 |
| 支持系统 | 未确认 | 现场确认 Windows/WSL/Docker |
| 可并行实例数 | 未测 | M1 吞吐 benchmark |
| upstream hero/action schema | 未冻结 | 接通后生成 adapter snapshot |

## 4. 里程碑摘要

| 里程碑 | 状态 | 下一门 |
|---|---|---|
| M0 Legacy freeze + V5 bootstrap | `READY` | repo、schema、mock、CI 完成 |
| M1 GameCore adapter + throughput | `BLOCKED_BY_M0` | 100 episodes 稳定、0 协议错误 |
| M2 1v1 BC baseline | `LOCKED` | M1 通过 |
| M3 1v1 PPO + fixed eval | `LOCKED` | BC 闭环基线通过 |
| M4 League + multi-hero | `LOCKED` | 1v1 PPO promotion |
| M5 3v3 MAPPO | `LOCKED` | multi-hero 与 3v3 env 门 |
| M6 Vision belief state | `LOCKED` | winning teacher 与对齐数据 |
| M7 Pixel closed loop | `LOCKED` | perception 和 distillation 门 |
| M8 Real shadow advisor | `LOCKED` | pixel/public-state 证据 |

## 5. 更新模板

每个任务完成后追加：

```text
日期：
任务：
状态：
变更文件：
运行命令：
验证结果：
生成 artifact：
已知风险：
外部操作：
下一任务：
```

不得用“基本完成”“应该通过”替代真实结果。
