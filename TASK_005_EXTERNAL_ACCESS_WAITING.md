# TASK-005 — External Access Waiting / Mock-only Control Plane

## 状态

- 阶段：`M1` 前置等待期
- 当前状态：`IMPLEMENTED / WAITING_EXTERNAL`
- 环境权威：仅项目自有 `mock`；不连接、探测或启动真实 GameCore。

## 目标

在腾讯明确授权的 GameCore、许可证路径和 service 接入方式尚未提供时，完成不会产生能力结论的控制面硬化：状态对齐、mock-only artifact gate、RPC 请求/会话关联和机械 fail-closed 测试。

“Mini”在本任务中只表示受限的 mock 诊断规模，不是一种新的环境授权类别，也不允许产生 checkpoint、promotion 或 HoK 能力结论。

## 本任务允许

- `mock` service、RPC、schema、CLI、单元测试与安全扫描；
- mock/PixelArena artifact 的 non-formal / diagnostic-only gate；
- 请求、episode、tick、legal-action、identity 与 runtime license 的 fail-closed 校验；
- learner、loss、checkpoint、resume 的接口设计和机械测试，但不运行训练。

## 本任务禁止

- 下载、读取、镜像或提交 GameCore、license、ABS、私有 replay；
- 启动/探测真实 GameCore，或把服务自报 license 当作外部授权证明；
- 长 BC/PPO、FULL_RUN、checkpoint promotion、真实 common-AI/teacher 收集；
- 多英雄、3v3、pixel policy、legacy 恢复或任何真实客户端自动动作。

## 交付物与验收

1. `configs/program_v1.yaml` 与 `DELIVERY_PROGRESS.md` 机器可读状态一致：`M1`、`TASK-005`、`WAITING_EXTERNAL`。
2. run/evaluation schema 只接受 typed environment kinds；mock/PixelArena 不可 formal 或晋升；`hok_gamecore` artifact 必须包含运行时 `license_status=valid`。这只是内部运行时门，不替代腾讯的外部授权证据。
3. `reset` 回显 `request_id`，`step` 回显 `episode_id`；任何错配 response 必须关闭已知本地 session、毒化 client/transport，并拒绝后续 health/reset/step，直至重建隔离 transport。
4. GameCore transport、formal evaluation 与 promotion 必须通过同一运行时控制面 gate：`EXTERNAL_ACCESS_CONFIRMED`、非 Git 外部授权证据引用、显式解锁、服务 runtime `license_status=valid` 缺一不可。transport/service factory 必须在创建 socket、进程或 SDK service 前完成本地 gate；当前配置下三者均返回 `WAITING_EXTERNAL`，且不触发 service health/reset/step。
5. `make check PYTHON=.venv/bin/python`、mock smoke、artifact verification、safety scan 均通过。

协议注记：当前 RPC 是未对外发布的本地 pre-release v1。TASK-010 之前必须完成一次显式的协议版本升级或兼容性决议，才能把新增的 reset/step 关联字段交给任何非 mock peer。

## 退出条件

仅当用户/授权方在 Git 外提供并确认：获授权的 GameCore、许可证接入方式、服务启动方式和允许的研究范围，并将控制面状态改为 `EXTERNAL_ACCESS_CONFIRMED` 后，才可转入 `TASK-010 GAMECORE_ADAPTER_AND_THROUGHPUT`。路径存在、schema-valid artifact 或 service 自报 `valid` 均不能单独满足该条件。
