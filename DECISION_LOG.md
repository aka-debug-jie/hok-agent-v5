# 决策记录

只记录会改变长期架构、边界或里程碑的决定。普通实验结果进入 run/evaluation artifact。

## D-001 — Strategy First

- 日期：2026-08-11
- 状态：Accepted
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
