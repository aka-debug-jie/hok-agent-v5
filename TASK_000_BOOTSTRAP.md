# TASK-000 — V5 仓库与环境接入骨架

> 历史已完成任务。第 8 节记录的是当时的后续路线，已由 D-004 与 `ROUTE_NO_GAMECORE_V1.md` 取代；当前任务以 `DELIVERY_PROGRESS.md` 为准。

## 1. 状态

- 任务：`TASK-000`
- 阶段：`M0`
- 初始状态：`READY`
- 性质：工程启动，不训练长模型，不产生能力结论。

## 2. 目标

创建一个与 Pixel-MOBA v4.x 解耦的新仓库，并交付：

1. Python 3.11 learner/evaluator 骨架；
2. 独立 environment service 的版本化接口；
3. mock environment；
4. run/evaluation manifest；
5. safety 和 secret scan；
6. upstream `hok_env` 接入 preflight；
7. 可运行的 smoke/benchmark harness；
8. 第一份 M0 自验报告。

## 3. 输入

- 本 V5 启动包；
- legacy repository URL/commit，仅作只读参考；
- 官方 `tencent-ailab/hok_env` 文档；
- 当前宿主机环境信息；
- 用户明确提供的 GameCore/许可证状态。

禁止假设 GameCore、license 或 hero/action schema 已存在。

## 4. 建议仓库结构

```text
hok-agent-v5/
  AGENTS.md
  PROGRAM_CHARTER.md
  SAFETY_BOUNDARIES.md
  DELIVERY_PROGRESS.md
  MILESTONES.md
  docs/
  configs/
  schemas/
  src/hok_agent/
    cli.py
    contracts/
    envs/
      base.py
      mock.py
      rpc_client.py
    artifacts/
    evaluation/
    safety/
  services/hok_gamecore/
    README.md
    protocol/
    server_stub/
  tests/
  pyproject.toml
  Makefile
  .github/workflows/ci.yml
  .gitignore
```

## 5. 实施切片

### A. 仓库初始化

- 复制本启动包；
- 创建 Python package；
- Python 3.11；
- Ruff、mypy、pytest；
- 基础 Make targets；
- GitHub Actions 或等价 CI；
- `DELIVERY_PROGRESS.md` → `IN_PROGRESS`。

### B. 合同类型

实现：

- `EnvironmentIdentity`
- `HealthResponse`
- `ResetRequest/Response`
- `StepRequest/Response`
- `FactorizedAction`
- `PublicObservation`
- `RewardVector`
- `EpisodeOutcome`
- `RunManifest`
- `EvaluationReport`

所有类型必须版本化、可 JSON 序列化、输入严格验证。

### C. Mock Environment

最小 deterministic 环境：

- reset(seed)；
- 单调 tick；
- 至少 3 个 action branches；
- legal action；
- 命名 reward；
- terminal；
- fixed-seed replay；
- optional injected error 测试；
- 明确标记 `environment_kind=mock`。

Mock 只验证基础设施，不模拟王者能力。

### D. RPC

定义独立服务协议：

- health；
- reset；
- step；
- close。

第一版可使用本地 IPC/ZeroMQ/gRPC 中一种，但协议层必须与传输层分离。

### E. Upstream Preflight

程序化检查：

- `hok_env` repo/version；
- upstream Python要求；
-当前 OS/WSL/Docker；
- GameCore path；
- license path；
-不要读取/打印 license内容；
-支持 mode/hero只有实际连接后才记录；
-缺失时生成 `WAITING_EXTERNAL` 报告。

### F. Smoke 与 benchmark

实现：

```text
env-smoke
env-benchmark
verify-artifact
```

mock必须可运行；真实 GameCore 只有实际可用时运行。

### G. 安全

测试：

- 禁止 API/字符串；
- no real-client input；
- secret/许可证不入库；
- privileged字段不在 Actor schema；
- `.gitignore` 完整；
- service identity fail-closed。

## 6. 验收

M0 通过要求：

- `make check` 或等价命令成功；
- tests覆盖合同、mock determinism、RPC、schema、safety；
- mock 100 episodes：
  - protocol errors=0；
  - deterministic replay=true；
  - manifests schema-valid；
- environment preflight生成明确状态；
- GameCore缺失时状态为 `WAITING_EXTERNAL`，而非假成功；
- legacy repo没有改动；
-没有任何训练 checkpoint；
-没有 K96主动作空间；
- `DELIVERY_PROGRESS.md` 记录真实结果。

## 7. 明确不做

- 不训练BC/PPO；
- 不接入真实客户端动作；
- 不复制v4.x checkpoint；
- 不移植K96；
- 不伪造common-AI结果；
- 不下载来源不明GameCore；
- 不把mock称为HoK环境。

## 8. 完成后的下一任务

- GameCore可用：`TASK-010 GAMECORE_ADAPTER_AND_THROUGHPUT`
- GameCore不可用：`TASK-005 EXTERNAL_ACCESS_WAITING + TRAINING_INFRASTRUCTURE_ON_MOCK/MINI`
