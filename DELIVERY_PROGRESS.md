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

## 2.1 已完成：mock public transition trace

状态：`IMPLEMENTED / WAITING_EXTERNAL`。

范围：为 deterministic mock 增加独立、公开字段限定的逐 transition 诊断 trace 与 fresh-process 重放/篡改验证。该 trace 不是训练 replay、不是 GameCore/ABS replay、不能保存 reward、teacher、truth、privileged state、legal mask、内部 replay identity/hash 或真实客户端数据。

固定边界：只接受内建 mock profile；不接受 GameCore、许可证、手机、control-plane 或任意外部 service 参数；输出固定标记为 `mock`、`formal=false`、`capability_claim=none`，且不是 evaluation report。

验收目标：trace 自哈希、逐行 hash chain、公开 observation hash、factorized action、tick/terminal/truncated/public outcome 的 fresh-process exact replay，以及篡改拒绝。mock 成功仍只证明基础设施。

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

## 8. TASK-005 legacy 配置约定接入记录

日期：`2026-08-11`

任务：按 legacy 的非敏感配置约定接入 V5，不迁移 legacy 的真实客户端、设备或训练设置。

状态：`IMPLEMENTED / WAITING_EXTERNAL`。

变更文件：所有 V5 YAML 根级固定为 `version: 1`；新增 `configs/runtime_inputs_v1.yaml`；`env-smoke`、`preflight` 和 `validate-config` 使用项目内 `configs/` 默认路径；Makefile 延续 `SYSTEM_PYTHON`/`VENV` 可覆盖约定。runtime YAML 只允许三个环境变量名，不能保存路径值、许可证、密钥、授权状态或 lock。

运行命令：`make check PYTHON=.venv/bin/python`；`make validate PYTHON=.venv/bin/python`；`.venv/bin/python -m hok_agent env-smoke --episodes 100`；两次 `verify-artifact`；`.venv/bin/python -m hok_agent preflight --root . --output reports/m0/upstream_gamecore_preflight.json --probe-upstream`。

验证结果：Ruff、strict mypy（20 source files）、pytest（42 passed）和 safety scan（75 files，0 finding）均通过。配置验证确认 4 份 YAML 与 2 份 JSON schema 有效；未知 `version: 2` 会被拒绝。mock 100/100 complete，`terminal_rate=1.0`、deterministic replay=true、protocol/action-decode/illegal/replay errors 均为 0；新 manifest 和 evaluation report 的 schema、self-hash 与 artifact hash 均有效。runtime source commit=`e575f4f77d9bdaa63cc8d1b452288814a4807f3a`，`dirty=false`。

生成 artifact：`artifacts/runs/m0-mock-smoke-20260811T103131Z/`（Git 忽略）；run manifest hash=`sha256:cca7ed475a44796dee0e3c46aabea7f44b5d4a0cc0f57aa2e4321d2a8edf5606`；evaluation report hash=`sha256:59f02a8e699031cd728420bf7ba0442318e6669a77b1d6143be6adb93f7a45f4`。

外部操作：只读检查 legacy 的 YAML/CLI/Make 约定；未修改 legacy，未迁移 V4 checkpoint/K96、设备设置或真实客户端脚本。upstream 可达（HTTP 200），但项目内未发现 `hok_env` checkout；GameCore 与 license 均未提供，未读取 license 内容、未记录路径值、未执行真实 handshake。

已知风险：legacy 没有可复用的 GameCore/license/service 配置，因此不能凭“配置一致”推断外部授权。当前 preflight 仍为 `WAITING_EXTERNAL`。

下一任务：等待 Git 外的获授权 GameCore、license 接入方式和明确的服务启动/协议资料；随后按 TASK-010 只在隔离 service 中执行真实 health/reset/step/close preflight。

## 9. TASK-005-PACKAGE-INTEGRITY-01 启动记录

日期：`2026-08-11`

状态：`DONE（本地交付完整性切片）/ WAITING_EXTERNAL`。

范围：基于 legacy 已完成的 source-only delivery integrity 模式，修复 V5 `PACKAGE_MANIFEST.json` 的文件集合与 hash 漂移，并把只读校验接入本地检查。只实现 V5 原生 checker 与机械篡改测试；不复制 legacy 代码，不迁移模型、checkpoint、K96、训练缓存、设备配置或真实客户端路径。

输入 authority：legacy 根台账、只读安全合同、source-only integrity 脚本和 V5 当前 `PACKAGE_MANIFEST.json`。legacy 的旧顶层 manifest/validation report 已过期，不能用于能力结论。

明确不做：不运行训练、不生成 checkpoint、不连接 GameCore、不读取 license、不连接手机、不改变 `WAITING_EXTERNAL` 或任何 control-plane lock。

实际变更：新增 `src/hok_agent/package_integrity.py` 与 `tests/test_package_integrity.py`；新增默认只读 `package-integrity` CLI、显式 `--write` 刷新模式和 `make integrity`，并把 integrity 纳入 `make check`。新的 manifest 固定记录 `manifest_version`、package、受控文件的相对路径/size/SHA-256 与 canonical self-hash；受限目录和文件被排除，受控 symlink、畸形项、重复/绝对/`..` 路径、缺失/新增/篡改均 fail-closed。

legacy 只读审阅结论：唯一 current-state authority 是 legacy `DELIVERY_PROGRESS.md`；v4.4 只是 non-promoting engineering canary，v4.4.1/R5/RL/MICRO-GATE/G10 均未完成或锁定。可借鉴的仅是 source-only integrity、公开/特权隔离、deterministic replay、失败关闭和事务回滚模式；任何 legacy checkpoint、optimizer、K96 profile、cache、阈值、设备配置、真实客户端路径或能力结论均未迁入。legacy 顶层旧 manifest/validation report 的日期与当前台账不一致，因此未作为能力证据。

运行命令：`.venv/bin/python -m pytest tests/test_package_integrity.py`；`.venv/bin/python -m hok_agent package-integrity --root . --write`；`make check PYTHON=.venv/bin/python`；`make validate PYTHON=.venv/bin/python`；`.venv/bin/python -m hok_agent env-smoke --episodes 100`；两次 `verify-artifact`；`.venv/bin/python -m hok_agent package-integrity --root .`。

验证结果：package-integrity focused tests `7 passed`；全量 `make check` 通过：Ruff、strict mypy（21 source files）、pytest（49 passed）、safety scan（77 files，0 finding）和 package integrity（68/68 controlled files）均通过；4 份 YAML 和 2 份 JSON schema 有效。100 局 mock `100/100` complete，`terminal_rate=1.0`、deterministic replay=true、protocol/action-decode/illegal/replay errors 均为 0；run manifest 与 evaluation report 的 schema/self-hash/artifact hash 均有效。runtime source commit=`d62a46c20ace14e17eda3508c18d395df9a89dbb`，`dirty=false`。

生成 artifact：`artifacts/runs/m0-mock-smoke-20260811T105706Z/`（Git 忽略）；run manifest hash=`sha256:021a617a62236089882ec2533baa0fa0941cd7d3186573c20195e107a8bcb57a`；evaluation report hash=`sha256:8ca784bd893099a026ee372fb84464817e8f165a3671260da480631f7e6ad617`。

外部操作与阻塞：未运行训练、未生成 checkpoint、未连接 GameCore 或手机、未读取 license、未修改 legacy。GameCore/license/service 方式仍未提供，控制面和所有外部操作继续为 `WAITING_EXTERNAL`。

下一任务：保持 TASK-005 waiting；未来有意改变 V5 受控源码时，先显式运行 `package-integrity --write`，再运行 `make check`。只有 Git 外的获授权 GameCore、license 接入方式和服务协议资料齐全后，才进入 TASK-010 的隔离 service preflight。

## 10. TASK-005-MOCK-PUBLIC-TRACE-01 实际记录

日期：`2026-08-11`

任务：`TASK-005-MOCK-PUBLIC-TRACE-01 / public-only mock diagnostic trace`

状态：`DONE（mock 基础设施切片）/ WAITING_EXTERNAL`。trace 不是训练 replay、canonical GameCore/ABS replay、formal evaluation 或 promotion 证据。

变更文件：`TASK_005_MOCK_REPLAY_01.md`、`DATA_AND_ARTIFACT_CONTRACT.md`、`ENVIRONMENT_CONTRACT.md`、`schemas/mock_public_replay.schema.json`、`src/hok_agent/contracts/mock_replay.py`、`src/hok_agent/evaluation/mock_replay.py`、CLI、generic artifact verifier、tests 和使用说明。只复用了 legacy 的 canonical hash、append-only row chain 与 fresh-process replay 思想；未复用 legacy replay、reward、teacher、truth、entity ID、K96、模型、checkpoint 或训练 journal。

运行命令：`make check PYTHON=.venv/bin/python`；focused `pytest`（mock replay/RPC/contracts）；`validate-config`；`mock-replay-record --output artifacts/runs/task005-mock-public-trace-20260811T111700Z/mock_public_replay.json --seed 101 --side blue --max-steps 12`；对应 `mock-replay-verify` 与 `verify-artifact`；`env-smoke --episodes 100`。

验证结果：全量 `make check` 通过：Ruff、strict mypy（23 source files）、pytest（64 passed）、safety scan（83 files，0 finding）和 package integrity（74/74 controlled files）均通过。focused replay/RPC/contracts suite `25 passed`，3 份 JSON schema 与 4 份 YAML 有效。record 与 fresh verify 使用不同 child PID（412/415），2 条 transition 完整终局；artifact self-hash/schema 通过。测试还覆盖同 seed/profile 字节一致、exclusive-create、rehashed action 篡改、私密字段、非公开 target、行链/终局/tick 篡改、控制面不得触发和 verifier 不改输入。

生成 artifact：`artifacts/runs/task005-mock-public-trace-20260811T111700Z/mock_public_replay.json`（Git 忽略），artifact hash=`sha256:0db29e47552678b2393283e88a9a7b9cbc9e9a34ec2b54eead0c0bbcfa049c32`，file hash=`sha256:3daf6c9e3002abca47d6982a95c839463798cb4426336373d1f6a254bf0bae97`。100 局 mock artifact：`artifacts/runs/m0-mock-smoke-20260811T111656Z/`，`100/100` complete、terminal rate=`1.0`、deterministic replay=`true`、protocol/action-decode/illegal/replay errors=`0`；runtime source commit=`fc395707e25bee0482e5792a82e2d18203ca674c`，`dirty=false`。

已知风险：自哈希只证明内容完整性，不证明作者身份；攻击者能重算全部 hash 时可形成另一个形式有效的 mock trace。trace 也不保存 reward、legal mask、privileged state 或 raw observation，故不能代替训练/正式 replay 要求。

外部操作与阻塞：未读取或连接 GameCore/license，未调用 external control-plane，未连接手机或真实商业客户端，未修改 legacy。GameCore binary、license 接入方式、外部授权证据、真实 service 协议仍未提供；状态继续为 `WAITING_EXTERNAL`。

下一任务：保持 `TASK-005 EXTERNAL_ACCESS_WAITING`，等待 Git 外的获授权 GameCore、license 接入方式和 service 启动/协议资料；齐全后先进入 `TASK-010` 的隔离 service preflight。不得以本 trace 或 mock smoke 替代该输入。
