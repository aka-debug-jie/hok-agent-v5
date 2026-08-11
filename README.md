# HoK-Agent V5 — PixelArena MOBA Research Agent

> 路线：`pixelarena-local-v1`
> 日期：`2026-08-11`
> 目标：在项目自有 PixelArena 中先建立可复核的结构化策略，再扩展到 3v3、RGB-only 闭环和真实客户端只读教练。

本项目已确认没有 GameCore。GameCore 不是主线依赖，只保留为未来默认关闭的可选外部校准轨；当前工作不等待许可证或二进制。

## 1. 本启动包解决什么问题

旧项目把自建环境、视觉感知、目标跟踪、K96 候选、长序列策略、教师学习、DAgger、强化学习和真实域迁移串成一条主链。任何一层失败，最终都可能表现为“没有攻击、没有结构伤害、没有胜局”，导致归因困难和不断追加局部修复。

V5 将问题拆成三条可独立验收的路线：

1. **Strategy Track**：在 PixelArena-Structured 中训练能够完成完整对局的结构化策略。
2. **Vision Track**：把画面转换成 public belief state，再蒸馏已会获胜的策略。
3. **Shadow Track**：真实商业客户端只读，输出 Linux 端建议、分析和复盘，不发送动作。

## 2. 权威文件与读取顺序

模型、开发者和审阅者进入仓库后，必须按以下顺序读取：

1. `AGENTS.md`
2. `PROGRAM_CHARTER.md`
3. `SAFETY_BOUNDARIES.md`
4. `DELIVERY_PROGRESS.md`
5. `MILESTONES.md`
6. `ROUTE_NO_GAMECORE_V1.md`
7. `ENVIRONMENT_CONTRACT.md`
8. `POLICY_ARCHITECTURE.md`
9. `TRAINING_CONTRACT.md`
10. `EVALUATION_PROMOTION.md`
11. `DATA_AND_ARTIFACT_CONTRACT.md`
12. `AUTOMATION_RUNBOOK.md`
13. `DELIVERY_PROGRESS.md` 指定的当前任务文件

其中：

- `PROGRAM_CHARTER.md` 是长期目标权威；
- `SAFETY_BOUNDARIES.md` 是不可降低边界权威；
- `DELIVERY_PROGRESS.md` 是当前状态唯一权威；
- `MILESTONES.md` 是阶段进入和晋升权威；
- 每次运行的事实只来自程序生成的 `run_manifest.json` 和 `evaluation_report.json`。

## 3. 本包包含的文件

| 文件 | 用途 |
|---|---|
| `AGENTS.md` | Codex/开发模型的仓库级执行规则 |
| `PROGRAM_CHARTER.md` | 项目目标、成功定义、非目标 |
| `SAFETY_BOUNDARIES.md` | 商业客户端、授权环境、数据和特权信息边界 |
| `DELIVERY_PROGRESS.md` | 当前任务与状态台账 |
| `MILESTONES.md` | M0–M8 阶段与硬门 |
| `ROUTE_NO_GAMECORE_V1.md` | 当前无 GameCore 主路线、claim scope 与任务图 |
| `ENVIRONMENT_CONTRACT.md` | PixelArena 服务、RPC、环境和重放合同 |
| `POLICY_ARCHITECTURE.md` | 不依赖 K96 硬筛选的因子化策略架构 |
| `TRAINING_CONTRACT.md` | BC、PPO、自博弈、MAPPO、蒸馏训练合同 |
| `EVALUATION_PROMOTION.md` | 评测、checkpoint 晋升和回滚 |
| `DATA_AND_ARTIFACT_CONTRACT.md` | 数据分层、manifest、hash 和隐私 |
| `AUTOMATION_RUNBOOK.md` | 自动化状态机、前两周执行顺序 |
| `LEGACY_MIGRATION_PLAN.md` | Pixel-MOBA v4.x 冻结和迁移方案 |
| `RISK_REGISTER.md` | 风险、触发器和止损规则 |
| `TASK_000_BOOTSTRAP.md` | 已完成的历史 bootstrap 任务 |
| `TASK_010_PIXELARENA_STRUCTURED_FOUNDATION.md` | 当前可直接实施的 M1 任务 |
| `CODEX_START_PROMPT.md` | 带解释的模型启动提示词 |
| `PROMPT_V5_BOOTSTRAP.txt` | 可直接整段复制的纯提示词 |
| `configs/*.yaml` | 项目、smoke 和 1v1 评测配置模板 |
| `schemas/*.json` | 运行与评测报告 JSON Schema |
| `REFERENCES.md` | 官方环境与研究依据 |

## 4. 最短启动路径

1. 从 `DELIVERY_PROGRESS.md` 确认当前任务。
2. 当前实施 `TASK_010_PIXELARENA_STRUCTURED_FOUNDATION.md`，不启动训练。
3. 先完成 scoped schema、ruleset、service、replay、baseline 和 M1 stability/throughput。
4. 只有 M1 报告通过后才进入 dataset/BC；GameCore 缺失不阻塞这些步骤。

## 5. 核心原则

- 先证明 PixelArena 结构化策略能够赢，再训练视觉学生。
- checkpoint 只由闭环胜率、Elo、结构/水晶结果和安全指标晋升。
- K96 可以保留为诊断视图，但不能再成为策略动作空间的硬信息瓶颈。
- 真实商业客户端永远只读。
- 所有结论必须绑定 claim scope、environment identity 和 ruleset；PixelArena 结果不是 HoK/GameCore 结果。
- 每个失败都必须缩小问题空间，不能自动派生下一轮局部调参。

## 6. M0 可运行接口

在 Python 3.11+ 环境安装开发依赖后，可运行：

```text
make check
make validate
make integrity
python -m hok_agent env-smoke
python -m hok_agent env-benchmark --episodes 100
python -m hok_agent mock-replay-record --output /tmp/mock_public_replay.json --seed 101 --side blue --max-steps 12
python -m hok_agent mock-replay-verify /tmp/mock_public_replay.json
python -m hok_agent preflight --probe-upstream
python -m hok_agent verify-artifact <run_manifest.json>
python -m hok_agent access-gate --operation gamecore_transport --runtime-license-status valid
```

`env-smoke` 与 `env-benchmark` 永远标记为 `mock`，只证明服务、合同、schema 和
artifact 基础设施；它们不构成任何 Honor of Kings 或 GameCore 能力结论。

`mock-replay-record`/`mock-replay-verify` 只处理内建 mock 的 public-only diagnostic trace：它保存因子化动作、tick、终局和公开 observation hash，并在新的 mock service process 中重放。它不保存 reward、legal mask、teacher、truth、privileged state 或内部 replay hash；不是训练 replay、formal evaluation 或 promotion 证据。

`access-gate` 不连接服务。它只管理默认关闭的可选 GameCore 轨，并继续以 `WAITING_EXTERNAL` 拒绝 transport、GameCore evaluation 和 GameCore promotion；这不影响 PixelArena 主线。

所有 V5 YAML 都遵循根级 `version` + 默认 `configs/` 路径约定。`preflight` 和 `configs/runtime_inputs_v1.yaml` 只属于可选 GameCore 轨；路径、许可证和密钥值不进入 Git，也不会改变 PixelArena 状态。
