# HoK-Agent V5 交付进度

> 本文件是当前状态唯一权威。  
> 历史讨论、提示词、旧仓库台账和实验文档不能覆盖本文件。

- 最后更新：`2026-08-11`
- 当前阶段：`V5-M1-EXTERNAL-ACCESS-WAITING`
- 当前任务：`TASK-005 EXTERNAL_ACCESS_WAITING / MOCK-ONLY CONTROL PLANE`
- 当前状态：`WAITING_EXTERNAL`
- 当前 promotion：`无 checkpoint 可晋升`
- 真实客户端边界：`READ_ONLY_SHADOW_ONLY`
- GameCore 连接状态：`WAITING_EXTERNAL / 未发现项目内 hok_env checkout，未进行真实服务握手`
- GameCore 许可证状态：`WAITING_EXTERNAL / 未提供路径，未读取任何许可证内容`

## 1. 已冻结决定

1. Pixel-MOBA v4.x 保留为 legacy evidence，不再创建 v4.4.1。
2. V5 不从 legacy checkpoint、optimizer 或 K96 profile 恢复。
3. 结构化 Strategy Agent 先于 Vision Student。
4. checkpoint 只由闭环评测晋升。
5. 真实商业客户端不执行自动动作。
6. PixelArena 降级为 test double、视觉课程和 counterfactual 环境。
7. 授权 GameCore 是最终策略能力权威；接入失败时不能伪造替代结论。

## 2. 当前任务范围

`TASK-000` 已完成；TASK-005 的 mock-only 控制面已实现并复验，当前仅等待授权环境输入。

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
| GameCore license | `WAITING_EXTERNAL` | 用户/授权方提供路径；不得上传、读取或猜测内容 |
| GameCore binaries | `WAITING_EXTERNAL` | 按上游条款获取；不下载或提交二进制 |
| 支持系统 | Linux + Docker daemon 已预检 | 真实服务接入时再现场确认兼容模式 |
| 可并行实例数 | 未测 | M1 吞吐 benchmark |
| upstream hero/action schema | 未冻结 | 接通后生成 adapter snapshot |

## 4. 里程碑摘要

| 里程碑 | 状态 | 下一门 |
|---|---|---|
| M0 Legacy freeze + V5 bootstrap | `DONE` | mock 基础设施门已通过；不构成 GameCore 或能力结论 |
| M1 GameCore adapter + throughput | `WAITING_EXTERNAL` | 需获授权的 GameCore、license 与真实 health/reset/step/close 握手 |
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

## 6. TASK-000 实际记录

日期：`2026-08-11`

任务：`TASK-000 REPOSITORY_AND_ENVIRONMENT_BOOTSTRAP`

状态：`DONE（仅 M0 mock 基础设施）`

变更文件：Python 3.11+ package、typed contracts、deterministic mock、spawned-process JSON RPC service、artifact verifier、CLI、tests、CI、preflight 与 M0 报告；详见 `reports/m0/M0_ACCEPTANCE_REPORT.md`。

运行命令：Ruff、strict mypy、pytest、`env-smoke --episodes 100`、`env-benchmark --episodes 100`、`verify-artifact`、`safety-scan`、`preflight --probe-upstream`。

验证结果：`make check PYTHON=.venv/bin/python` 通过：Ruff、strict mypy（19 source files）、pytest（18 passed）和 safety scan（67 files，0 finding）均通过。最终 mock 100/100 complete，`terminal_rate=1.0`，deterministic replay=true，protocol/action-decode/illegal/replay errors 均为 0；manifest 与 evaluation report 均为 schema+self-hash+artifact-hash valid。mock service 运行于独立 spawned process；client 对 identity/license/tick/legal-action 违规 fail-closed。

生成 artifact：`artifacts/runs/m0-mock-smoke-20260811T090534Z/` 与 `artifacts/runs/m0-mock-benchmark-20260811T090549Z/`（均被 Git 忽略）；runtime source commit=`4fe32d622de24523bcc7dfad6f0935b568c23773`，`dirty=false`。

已知风险：mock 仅证明基础设施；尚无 GameCore 二进制、许可证、真实 schema、真实服务握手或任何策略能力证据。

外部操作：未下载 GameCore，未读取许可证，未修改 legacy，未对真实客户端发送动作。

下一任务：继续 `TASK-005 EXTERNAL_ACCESS_WAITING`；若用户/授权方提供 Git 外的获授权 GameCore 与 license 接入方式，先执行真实服务 identity/license/schema/health/reset/step/close preflight，再进入 `TASK-010 GAMECORE_ADAPTER_AND_THROUGHPUT`。

## 7. TASK-005 mock-only control-plane 实际记录

日期：`2026-08-11`

任务：`TASK-005 EXTERNAL_ACCESS_WAITING / MOCK-ONLY CONTROL PLANE`

状态：`IMPLEMENTED / WAITING_EXTERNAL`；没有启动训练、没有 checkpoint、没有连接或探测真实 GameCore。

变更文件：`TASK_005_EXTERNAL_ACCESS_WAITING.md`、`configs/program_v1.yaml`、环境/RPC 合同、control-plane gate、artifact schema/verifier、CLI、mock server stub 与相应测试；runtime source commit=`57125d4f8ff4376bf7d0eed58821df31c1b24908`，`dirty=false`。

运行命令：`make check PYTHON=.venv/bin/python`；`env-smoke --config configs/run_smoke_v1.yaml --episodes 100`；两份 `verify-artifact`；三次 `access-gate --runtime-license-status valid`；`preflight --probe-upstream`。

验证结果：Ruff、strict mypy（20 source files）、pytest（35 passed）、safety scan（73 files，0 finding）均通过。mock 100/100 complete，`terminal_rate=1.0`，deterministic replay=true，protocol/action-decode/illegal/replay errors 均为 0；新 manifest 与 evaluation report 均 schema+self-hash+artifact-hash valid。当前配置对 GameCore transport、formal evaluation、promotion 均返回预期 `WAITING_EXTERNAL`（exit=2），并由计数 test 证明不调用 test-double 的 health/reset/step。

生成 artifact：`artifacts/runs/m0-mock-smoke-20260811T095514Z/`（Git 忽略）；最新项目内 preflight 报告为 `reports/m0/upstream_gamecore_preflight.json`。

已知风险：新增 reset/step 关联字段仍是未对外发布的 local protocol v1；进入 TASK-010 前必须完成显式协议版本/兼容性决议。artifact verifier 的通过仅表示技术 schema/hash 完整，不表示腾讯外部授权。

外部阻塞：GameCore binary、license 接入方式、获授权范围与真实 service 启动方式均未在 Git 外提供；preflight 仍为 `WAITING_EXTERNAL`。远程 upstream 探测可达（HTTP 200），但项目内未发现 checkout；未读取 license 内容、未记录 license 路径、未进行真实握手。

下一任务：等待上述外部输入；外部授权证据、明确 service 方式和许可证接入方式齐全后，以新的隔离 transport/service factory 执行 TASK-010 preflight。不得以 mock 或 service 自报 runtime `valid` 代替授权。
