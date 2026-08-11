# TASK-006 — 无 GameCore 路线纠正

## 状态与目标

- 阶段：M0 治理修正
- 状态：`DONE`
- 目标：解除 GameCore 对本地主线的结构性死锁，建立 PixelArena 内部权威和完整 M0–M8 计划。

## 允许范围

- 更新权威文档、入口提示词、机器控制面配置和相应机械测试；
- 将 GameCore 迁移为可选外部校准轨并保持 fail-closed；
- 定义 scoped claim、任务依赖和 `TASK-010` 验收；
- 只读审阅 legacy 的可复用工程模式。

## 明确不做

- 不实现 PixelArena gameplay；
- 不运行 BC/PPO、生成 checkpoint 或作能力结论；
- 不连接 GameCore、手机、真实客户端或输入设备；
- 不读取许可证、不修改 legacy、不迁移旧模型/K96/cache。

## 验收

1. `DELIVERY_PROGRESS.md`、章程、安全、里程碑、环境、训练、评测、数据和 runbook 一致；
2. `program_v1.yaml` 主权威为 PixelArena，可选 GameCore 轨为 `NOT_AVAILABLE` 且不阻塞主线；
3. GameCore gate 仍在任何 runtime `valid` 自报下 fail-closed；
4. 现有 mock schema 和 replay 继续有效；
5. Ruff、strict mypy、pytest、配置验证、安全扫描和 package integrity 全通过；
6. 没有训练、checkpoint、设备或外部服务操作；
7. 完成后当前任务切换为 `TASK-010 / READY`。
