# HoK-Agent V5 启动包

> 版本：`v5.0-bootstrap-1`  
> 日期：`2026-08-11`  
> 目标：从 Pixel-MOBA v4.x 的局部补丁链迁移到“官方/授权 GameCore 中先学会赢，再进行视觉蒸馏”的可持续路线。

## 1. 本启动包解决什么问题

旧项目把自建环境、视觉感知、目标跟踪、K96 候选、长序列策略、教师学习、DAgger、强化学习和真实域迁移串成一条主链。任何一层失败，最终都可能表现为“没有攻击、没有结构伤害、没有胜局”，导致归因困难和不断追加局部修复。

V5 将问题拆成三条可独立验收的路线：

1. **Strategy Track**：在腾讯授权的 Honor of Kings GameCore 或兼容研究环境中，先训练能够完成完整对局的结构化策略。
2. **Vision Track**：把画面转换成 public belief state，再蒸馏已会获胜的策略。
3. **Shadow Track**：真实商业客户端只读，输出 Linux 端建议、分析和复盘，不发送动作。

## 2. 权威文件与读取顺序

模型、开发者和审阅者进入仓库后，必须按以下顺序读取：

1. `AGENTS.md`
2. `PROGRAM_CHARTER.md`
3. `SAFETY_BOUNDARIES.md`
4. `DELIVERY_PROGRESS.md`
5. `MILESTONES.md`
6. `ENVIRONMENT_CONTRACT.md`
7. `POLICY_ARCHITECTURE.md`
8. `TRAINING_CONTRACT.md`
9. `EVALUATION_PROMOTION.md`
10. `DATA_AND_ARTIFACT_CONTRACT.md`
11. `AUTOMATION_RUNBOOK.md`
12. `TASK_000_BOOTSTRAP.md`

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
| `ENVIRONMENT_CONTRACT.md` | GameCore 服务、RPC、环境和重放合同 |
| `POLICY_ARCHITECTURE.md` | 不依赖 K96 硬筛选的因子化策略架构 |
| `TRAINING_CONTRACT.md` | BC、PPO、自博弈、MAPPO、蒸馏训练合同 |
| `EVALUATION_PROMOTION.md` | 评测、checkpoint 晋升和回滚 |
| `DATA_AND_ARTIFACT_CONTRACT.md` | 数据分层、manifest、hash 和隐私 |
| `AUTOMATION_RUNBOOK.md` | 自动化状态机、前两周执行顺序 |
| `LEGACY_MIGRATION_PLAN.md` | Pixel-MOBA v4.x 冻结和迁移方案 |
| `RISK_REGISTER.md` | 风险、触发器和止损规则 |
| `TASK_000_BOOTSTRAP.md` | 第一个可直接实施的任务 |
| `CODEX_START_PROMPT.md` | 带解释的模型启动提示词 |
| `PROMPT_V5_BOOTSTRAP.txt` | 可直接整段复制的纯提示词 |
| `configs/*.yaml` | 项目、smoke 和 1v1 评测配置模板 |
| `schemas/*.json` | 运行与评测报告 JSON Schema |
| `REFERENCES.md` | 官方环境与研究依据 |

## 4. 最短启动路径

1. 将本目录复制到一个**全新的仓库**，不要覆盖旧 Pixel-MOBA 仓库。
2. 把 `DELIVERY_PROGRESS.md` 中的仓库路径和负责人占位符补全。
3. 把 `PROMPT_V5_BOOTSTRAP.txt` 整段发给 Codex。
4. 首先完成 `TASK_000_BOOTSTRAP.md`，不启动长训练。
5. GameCore 许可证或二进制未获得时，状态必须是 `WAITING_EXTERNAL`；可以实现 mock adapter、RPC、schema 和基准框架，但不得伪造已接通官方环境。

## 5. 核心原则

- 先证明结构化策略能够赢，再训练视觉学生。
- checkpoint 只由闭环胜率、Elo、结构/水晶结果和安全指标晋升。
- K96 可以保留为诊断视图，但不能再成为策略动作空间的硬信息瓶颈。
- 真实商业客户端永远只读。
- 每个失败都必须缩小问题空间，不能自动派生下一轮局部调参。

## 6. M0 可运行接口

在 Python 3.11+ 环境安装开发依赖后，可运行：

```text
make check
make validate
python -m hok_agent env-smoke
python -m hok_agent env-benchmark --episodes 100
python -m hok_agent preflight --probe-upstream
python -m hok_agent verify-artifact <run_manifest.json>
python -m hok_agent access-gate --operation gamecore_transport --runtime-license-status valid
```

`env-smoke` 与 `env-benchmark` 永远标记为 `mock`，只证明服务、合同、schema 和
artifact 基础设施；它们不构成任何 Honor of Kings 或 GameCore 能力结论。

`access-gate` 不连接服务。当前它会以 `WAITING_EXTERNAL` 拒绝 GameCore transport、formal evaluation 和 promotion；服务自报 `valid` 不是腾讯外部授权证明。

所有 V5 YAML 都遵循 legacy 的根级 `version` + 默认 `configs/` 路径约定。`preflight` 会自动读取 `configs/runtime_inputs_v1.yaml`，其中只保存非秘密环境变量名；路径、许可证和密钥值不进入 Git，也不会改变 control-plane 的 `WAITING_EXTERNAL` 锁。
