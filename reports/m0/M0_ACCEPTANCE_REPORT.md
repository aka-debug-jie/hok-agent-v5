# M0 验收报告 — HoK-Agent V5

日期：2026-08-11
状态：`DONE（仅 V5 mock 基础设施）`

## 结论边界

`TASK-000` 的 M0 工程门已通过：独立 Python 3.11+ 项目、严格 typed contracts、deterministic mock、独立进程 environment service、RPC client/server stub、artifact integrity、CLI、CI、安全扫描和 upstream preflight 已实际实现并验证。

这不是 GameCore 可用性、Honor of Kings 策略能力、1v1 胜率、BC/PPO 训练或真实客户端自动化结论。mock 成功只证明基础设施。当前 GameCore/许可证状态保持 `WAITING_EXTERNAL`。

旧 `pixel-moba-codex-starter` 仅作为只读 legacy evidence：未改写其代码、报告、checkpoint、optimizer 或 K96 profile；本仓库没有从 legacy 恢复任何训练状态，也没有 K96 主动作空间。

## 实现交付

| 项目 | 位置 |
|---|---|
| Python package / CLI | `src/hok_agent/` |
| typed contracts 与 Actor denylist | `src/hok_agent/contracts/types.py` |
| deterministic mock | `src/hok_agent/envs/mock.py` |
| 独立 spawned-process RPC transport | `src/hok_agent/envs/process_transport.py` |
| RPC client/server 与 fail-closed gate | `src/hok_agent/envs/rpc.py` |
| artifact hashing / verification | `src/hok_agent/artifacts/` |
| smoke / benchmark harness | `src/hok_agent/evaluation/smoke.py` |
| safety / secret / license scan | `src/hok_agent/safety/` |
| upstream/GameCore preflight | `src/hok_agent/preflight.py` |
| CI | `.github/workflows/ci.yml` |

环境 service 使用 `multiprocessing` 的 `spawn` context。learner/client 只持有 Pipe transport；`MockEnvironment` 和 RPC dispatcher 在子进程内构造。测试确认 service PID 与 learner PID 不同。

`LocalRpcClient` 在构造、`reset` 与每次 `step` 前检查 health identity；预期 `hok_gamecore` 时要求 `license_status=valid`。identity 变化、无效许可证、stale tick、非法动作、非单调 tick、非终止状态的空 legal action 或 RPC/contract 错误都会拒绝动作并关闭当前 session。legal action 只保留在 sampling/execution/loss-audit 边界；Actor observation mapping 拒绝 `legal_*`、reward、teacher、truth 和其他 privileged 字段。

manifest/report self hash 为 schema required；manifest verifier 还核验 repo-relative artifact 路径和 SHA-256。mock evaluation 只能是 `DIAGNOSTIC_ONLY`，不能用 mock artifact 晋升 checkpoint。

## 实际验证

验证解释器是项目内 `.venv` 的 Python `3.11.15`；它由已有解释器创建，安装了本项目 editable + dev dependencies，且由 `.gitignore` 排除。项目在 `pyproject.toml` 声明 `requires-python >=3.11`。本轮没有修改既有环境，也没有启动长训练任务。

| 命令 | 实际结果 |
|---|---|
| `make check PYTHON=.venv/bin/python` | Ruff pass；strict mypy 19 source files 无 issue；pytest `18 passed`；safety scan 67 files、0 finding |
| `PYTHONPATH=src .../python -m hok_agent env-smoke --config configs/run_smoke_v1.yaml --episodes 100` | pass，100/100 complete |
| `PYTHONPATH=src .../python -m hok_agent verify-artifact .../run_manifest.json --root .` | schema+self-hash+referenced artifact hash pass |
| `PYTHONPATH=src .../python -m hok_agent verify-artifact .../evaluation_report.json --root .` | schema+self-hash+mock disposition/gate semantics pass |
| `PYTHONPATH=src .../python -m hok_agent env-benchmark --config configs/run_smoke_v1.yaml --episodes 100` | mock-only `531.77 env-steps/s` |

最终 smoke artifact（runtime source commit=`4fe32d622de24523bcc7dfad6f0935b568c23773`，`dirty=false`）：

- `artifacts/runs/m0-mock-smoke-20260811T090534Z/run_manifest.json`
  - file SHA-256：`sha256:9c160d9be13c5ead7b9238932b7f9119ded73896ea057ee24eee554bf570b676`
  - self hash：`sha256:f8ef59c09537b0d7f11ed9db8107245cdabcd31e74a06a17ea45c4590312d70c`
- `artifacts/runs/m0-mock-smoke-20260811T090534Z/evaluation_report.json`
  - file SHA-256：`sha256:717276a9a65bbbb67f9d359892c23eb3834b6746af66db8dd75b65ce94eb7c86`
  - self hash：`sha256:9724990ab7a3e2a4929d22b2431eb4d7e47e4b55c2ecf841391e704520484ae8`
- `artifacts/runs/m0-mock-smoke-20260811T090534Z/episodes.jsonl`
  - file SHA-256：`sha256:fef734aa0d6deabe72e9e1a5db5d183ec9c65309d7e5fe910a4f118a8a45ade5`

Smoke metrics：`completed=100`、`terminal_rate=1.0`、`deterministic_replay=true`、`protocol_errors=0`、`action_decode_errors=0`、`illegal_executed_actions=0`、`replay_errors=0`、`tick_monotonic=true`。evaluation disposition=`DIAGNOSTIC_ONLY`。

Benchmark artifact：`artifacts/runs/m0-mock-benchmark-20260811T090549Z/benchmark.json`，file SHA-256=`sha256:959b35aee5988d0a95415d926b787f61ea3009002553ade829eeaab696fdb0cb`，self hash=`sha256:10d00514bde08652b360f29267cb4f12250f7a31903acaf95bf9cd645400de6d`。该数值不代表 GameCore 吞吐或任何策略性能。

## 授权环境预检

报告：[upstream_gamecore_preflight.json](upstream_gamecore_preflight.json)。

- 上游公开仓库探测可达：`https://github.com/tencent-ailab/hok_env`（HTTP 200）。
- 上游 service Python 约束记录为 `>=3.6,<3.10`；V5 learner 因此不直接依赖旧 SDK。
- 当前宿主机：Linux，非 WSL，Docker daemon 可达（29.1.3）。
- 有界项目内检查未发现 `hok_env` checkout；未提供 GameCore 或 license path；没有读取 license 内容、记录路径值、下载二进制或执行真实 service handshake。
- 结论：`WAITING_EXTERNAL`。这不是“无许可证”或“GameCore 未安装”的全局事实判断。

## 未解决阻塞与下一任务

需要用户或授权方在 Git 外提供/确认腾讯明确授权的 GameCore、license 路径和 service 接入方式。收到后先进行实际 identity、license、schema、health/reset/step/close 和吞吐 preflight；不要直接启动 BC/PPO。

下一任务：`TASK-005 EXTERNAL_ACCESS_WAITING + TRAINING_INFRASTRUCTURE_ON_MOCK/MINI`。仅当真实授权服务通过预检后，进入 `TASK-010 GAMECORE_ADAPTER_AND_THROUGHPUT`。
