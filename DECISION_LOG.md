# 决策记录

只记录会改变长期架构、边界或里程碑的决定。普通实验结果进入 run/evaluation artifact。

## D-001 — Strategy First

- 日期：2026-08-11
- 状态：Superseded by D-004（保留历史原文）
- 决定：先在授权 GameCore 中训练 structured winning policy，再做视觉蒸馏。
- 原因：旧路线无法区分 perception、candidate support、ranking、visitation 与 long-horizon credit。
- 后果：Vision Track 不再阻塞 1v1 Strategy Track。

## D-002 — K96 不再作为主动作空间

- 日期：2026-08-11
- 状态：Accepted
- 决定：V5采用因子化动作策略；K96仅可作为debug top-actions。
- 原因：旧K96逐步成为手工策略和hard support bottleneck。
- 后果：action adapter和factorized log-prob成为M0/M2核心合同。

## D-003 — Commercial Client Read-only

- 日期：2026-08-11
- 状态：Accepted / Irreversible
- 决定：真实商业客户端不执行自动动作。
- 原因：安全、合规、公平和研究边界。
- 后果：最终产品是shadow advisor，不是商业服机器人。

## D-004 — No-GameCore PixelArena Mainline

- 日期：2026-08-11
- 状态：Accepted
- 决定：确认项目没有 GameCore；PixelArena 成为 M1–M7 的项目内部策略权威，GameCore 降为默认关闭且不阻塞主线的可选外部校准轨。
- 原因：原路线将不存在的外部依赖设为唯一晋升权威，导致 M1–M8 永久死锁。
- 后果：原计划的 `TASK-010 GAMECORE_ADAPTER_AND_THROUGHPUT` 未执行并取消；编号改用于 `TASK-010 PIXELARENA_STRUCTURED_FOUNDATION`。所有 PixelArena 结论强制限定为 `pixelarena_internal`。

## D-005 — Scoped Promotion

- 日期：2026-08-11
- 状态：Accepted
- 决定：checkpoint promotion 必须绑定 claim scope、environment family、ruleset major 和 schema/suite identity；不同 scope 使用独立 registry。
- 原因：裸 `formal` 或裸胜率无法区分内部环境能力与外部游戏能力。
- 后果：现有 v1 schema 在 TASK-010 完成 scoped schema 前继续拒绝 PixelArena promotion。
