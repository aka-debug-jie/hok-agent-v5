# M0 验收报告 — HoK-Agent V5

日期：2026-08-11
状态：`IN_PROGRESS（M0 isolation/fail-closed 修复中）`

## 范围与结论

本轮已建立独立 V5 仓库的工程启动基础：Python 3.11+ package、严格 JSON contracts、deterministic mock、health/reset/step/close JSON RPC stub、artifact hash/schema verifier、CLI、CI、safety/secret/license scan 和上游预检。

没有训练 BC/PPO，没有读取或恢复 v4 checkpoint/optimizer/K96 profile，没有创建 K96 硬候选空间，也没有对真实商业客户端发送任何动作。mock 成功只证明 M0 基础设施；不构成 Honor of Kings、GameCore、1v1 胜率或策略能力结论。

独立审阅发现当前 JSON RPC stub 仍是同进程实现，且 client 的 identity/license gate 尚未强制执行；因此本报告暂不能作为 M0 通过证明。修复、重新运行 100-episode mock 和冻结证据后才可结案。

旧 `pixel-moba-codex-starter` 未被修改。本仓库中的 `PACKAGE_MANIFEST.json` 是启动 ZIP 的历史清单；其文件哈希只适用于打包时的原始内容，不能用于验证本工作树的后续 M0 编辑。

## 实现交付

| 项目 | 位置 |
|---|---|
| Python package / CLI | `src/hok_agent/` |
| typed contracts | `src/hok_agent/contracts/types.py` |
| mock + RPC boundary | `src/hok_agent/envs/` |
| artifact hashing / verification | `src/hok_agent/artifacts/` |
| safety / secret scan | `src/hok_agent/safety/` |
| upstream/GameCore preflight | `src/hok_agent/preflight.py` |
| tests | `tests/` |
| CI | `.github/workflows/ci.yml` |

## 实际验证

验证解释器为已有的 Python `3.11.15` 环境；项目自身在 `pyproject.toml` 中声明 `requires-python = >=3.11`。本轮未修改该既有环境。

| 命令 | 实际结果 |
|---|---|
| `/home/hgdl1012/miniconda3/bin/ruff check src tests` | pass |
| `PYTHONPATH=/tmp/hok-v5-mypy /home/hgdl1012/miniconda3/envs/phase7-figs/bin/python -m mypy src`（strict） | pass，18 source files，无 issue |
| `PYTHONPATH=src /home/hgdl1012/miniconda3/envs/phase7-figs/bin/python -m pytest` | pass，10 tests |
| `PYTHONPATH=src /home/hgdl1012/miniconda3/envs/phase7-figs/bin/python -m hok_agent safety-scan --root .` | pass，0 finding |
| `PYTHONPATH=src /home/hgdl1012/miniconda3/envs/phase7-figs/bin/python -m hok_agent env-smoke --config configs/run_smoke_v1.yaml --episodes 100` | pass，100/100 complete |
| `PYTHONPATH=src /home/hgdl1012/miniconda3/envs/phase7-figs/bin/python -m hok_agent verify-artifact artifacts/runs/m0-mock-smoke-20260811T084847Z/run_manifest.json --root .` | schema+self-hash pass |
| `PYTHONPATH=src /home/hgdl1012/miniconda3/envs/phase7-figs/bin/python -m hok_agent verify-artifact artifacts/runs/m0-mock-smoke-20260811T084847Z/evaluation_report.json --root .` | schema+self-hash pass |
| `PYTHONPATH=src /home/hgdl1012/miniconda3/envs/phase7-figs/bin/python -m hok_agent env-benchmark --episodes 100` | mock-only 3,588.55 env-steps/s；不代表 GameCore 吞吐 |

Mock smoke artifact：

- `artifacts/runs/m0-mock-smoke-20260811T084847Z/run_manifest.json`
  - file SHA-256：`sha256:5343fd4a09000dd780d98e9233e50d4c9457ef472d5c6fbe39e3cf9808940ae1`
  - self hash：`sha256:ff7f684c83f0ed626dd3d82063bf492772bcbddba7720f13db70497bb062cd2d`
- `artifacts/runs/m0-mock-smoke-20260811T084847Z/evaluation_report.json`
  - file SHA-256：`sha256:c5f36f979b8081b4ae9bd1926db902ecc66d95ffe1dbe58f6d9c1e5a13eaea87`
  - self hash：`sha256:b6b0d9475327f84a6fa02ba8d94882bd427fc07bf5b3301ae3242aa1ebbf0d38`
- runtime source identity：`1aa2ba3ae83e4d93a1d1c5ec46e8f250deb126cf`，`dirty=false`

Smoke metrics：`completed=100`，`terminal_rate=1.0`，`protocol_errors=0`，`action_decode_errors=0`，`illegal_executed_actions=0`，`replay_errors=0`，`fixed_seed_replay_exact=true`。evaluation disposition 为 `DIAGNOSTIC_ONLY`。

Mock benchmark artifact：`artifacts/runs/m0-mock-benchmark-20260811T084935Z/benchmark.json`，file SHA-256=`sha256:7cda627569d2c721c9bfae1696054f06b986c3ff3c6f8db3bb481e6060cb6af1`。

## 授权环境预检

预检报告：[upstream_gamecore_preflight.json](upstream_gamecore_preflight.json)。

- 上游公开仓库可达：`https://github.com/tencent-ailab/hok_env`。
- 上游 Python 约束记录为 `>=3.6,<3.10`，因此 V5 learner 与上游 service 继续保持进程/环境隔离。
- 本机为 Linux；Docker daemon 可达。
- 限定项目路径中未发现 `hok_env` checkout；未提供 GameCore 或 license path；未读取任何 license 内容；未进行实际 service health handshake。
- 结论为 `WAITING_EXTERNAL`，不是代码失败，也不是“无许可证”的最终事实判断。

## 未解决阻塞与下一任务

需要用户或授权方在 Git 外提供/确认获授权的 GameCore、license 路径和接入方式。届时先完成真实环境 identity、license、schema、health/reset/step/close handshake 与吞吐预检；不要启动 BC/PPO。

下一任务固定为 `TASK-005 EXTERNAL_ACCESS_WAITING + TRAINING_INFRASTRUCTURE_ON_MOCK/MINI`；只有真实授权服务通过后，才可进入 `TASK-010 GAMECORE_ADAPTER_AND_THROUGHPUT`。
