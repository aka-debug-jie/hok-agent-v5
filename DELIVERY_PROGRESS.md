# HoK-Agent V5 交付进度

> 本文件是当前状态唯一权威。  
> 历史讨论、提示词、旧仓库台账和实验文档不能覆盖本文件。

- 最后更新：`2026-08-11`
- 当前阶段：`V5-M1-PIXELARENA-STRUCTURED`
- 当前任务：`TASK-010 PIXELARENA_STRUCTURED_FOUNDATION`
- 当前状态：`READY`
- 当前 promotion：`无 checkpoint 可晋升`
- 真实客户端边界：`READ_ONLY_SHADOW_ONLY`
- 可选 GameCore 轨：`NOT_AVAILABLE / DISABLED / NON_BLOCKING`
- GameCore 许可证状态：`UNKNOWN / 未提供路径，未读取任何许可证内容`

## 1. 已冻结决定

1. Pixel-MOBA v4.x 保留为只读 legacy evidence；不修改、不训练、不创建 v4.4.1。
2. V5 不从 legacy checkpoint、optimizer、cache、阈值或 K96 profile 恢复。
3. 项目没有 GameCore；GameCore 不是 M0–M8 依赖，只是默认关闭的可选外部校准轨。
4. PixelArena-Structured 是项目内部策略权威；结果只可声明 `pixelarena_internal`。
5. structured strategy 先于 vision student；先固定原型 1v1，再 league/多原型，再 3v3。
6. checkpoint 只在同一 claim scope/environment/ruleset major 内由闭环评测晋升。
7. mock 永远 diagnostic-only；现有 v1 schema 在 scoped schema v2 完成前继续拒绝 PixelArena promotion。
8. 真实商业客户端只读，不执行自动动作或在线 RL。

## 2. 当前任务范围

`TASK-006` 已完成路线纠正。当前唯一实施任务是 `TASK-010_PIXELARENA_STRUCTURED_FOUNDATION.md`。

当前允许：

- scoped evidence schema v2 与独立 PixelArena registry；
- V5 原生 PixelArena ruleset、local-process service 和版本化 RPC；
- public/privileged projection、factorized action、reward、terminal；
- replay/snapshot、NULL/random/scripted baseline；
- 100 局 stability、规则因果测试和 throughput benchmark。

当前不允许：

- BC/PPO、可晋升 checkpoint、promotion、多原型、3v3 或 pixel policy；
- GameCore、手机、真实客户端、输入设备或许可证操作；
- 修改 legacy 或迁移旧 runtime/model/K96/cache。

## 2.1 已完成的 M0 基础设施

typed contracts、deterministic mock、spawned-process RPC、artifact/schema verifier、package integrity、safety scan 和 public mock trace 均保留。它们只证明 M0 基础设施，不证明 PixelArena、HoK 或策略能力。

## 3. 当前阻塞与非阻塞外部项

| 项目 | 状态 | 影响 |
|---|---|---|
| V5 PixelArena ruleset/service | `NOT_IMPLEMENTED` | TASK-010 当前工程目标 |
| scoped evidence schema v2 | `NOT_IMPLEMENTED` | 完成前 PixelArena promotion 继续 fail-closed |
| PixelArena 可并行实例数 | `UNMEASURED` | TASK-010 throughput 验收 |
| GameCore/license/binaries | `NOT_AVAILABLE / OPTIONAL` | 不阻塞主线；不得猜测、下载或伪造 |
| 真实客户端 | `READ_ONLY_SHADOW_ONLY` | M8 前不接入；永不允许动作 |

## 4. 里程碑摘要

| 里程碑 | 状态 | 下一门 |
|---|---|---|
| M0 bootstrap + route reset | `DONE` | mock 基础设施和路线治理通过；不构成策略能力 |
| M1 PixelArena-Structured foundation | `READY` | TASK-010 100 局稳定、重放、规则因果与吞吐 |
| M2 baseline + episode dataset | `LOCKED` | M1 通过 |
| M3 fixed-archetype BC | `LOCKED` | M2 dataset/reward audit 通过 |
| M4 fixed-archetype PPO | `LOCKED` | M3 closed-loop BC 通过 |
| M5 league + multi-archetype | `LOCKED` | M4 scoped promotion |
| M6 3v3 CTDE/MAPPO | `LOCKED` | M5 retention/league 门 |
| M7 belief + RGB-only PixelArena | `LOCKED` | M6 与 renderer/data 门 |
| M8 real-client read-only coach | `LOCKED` | M7、隐私和 shadow isolation 门 |
| AX optional GameCore calibration | `NOT_AVAILABLE / NON_BLOCKING` | 仅未来新授权任务 |

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

## 6. 历史实际记录

以下第 6–10 节保留当时的真实命令、状态和结论，其中 `WAITING_EXTERNAL` 与旧 `TASK-010 GAMECORE...` 只表示当时路线，已由 D-004 取代，不是当前任务或当前阻塞。

### 6.1 TASK-000 实际记录

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

### 6.2 TASK-005 mock-only control-plane 实际记录

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

### 6.3 TASK-005 legacy 配置约定接入记录

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

### 6.4 TASK-005-PACKAGE-INTEGRITY-01 启动记录

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

### 6.5 TASK-005-MOCK-PUBLIC-TRACE-01 实际记录

日期：`2026-08-11`

任务：`TASK-005-MOCK-PUBLIC-TRACE-01 / public-only mock diagnostic trace`

状态：`DONE（mock 基础设施切片）/ WAITING_EXTERNAL`。trace 不是训练 replay、canonical GameCore/ABS replay、formal evaluation 或 promotion 证据。

变更文件：`TASK_005_MOCK_REPLAY_01.md`、`DATA_AND_ARTIFACT_CONTRACT.md`、`ENVIRONMENT_CONTRACT.md`、`schemas/mock_public_replay.schema.json`、`src/hok_agent/contracts/mock_replay.py`、`src/hok_agent/evaluation/mock_replay.py`、CLI、generic artifact verifier、tests 和使用说明。只复用了 legacy 的 canonical hash、append-only row chain 与 fresh-process replay 思想；未复用 legacy replay、reward、teacher、truth、entity ID、K96、模型、checkpoint 或训练 journal。

运行命令：`make check PYTHON=.venv/bin/python`；focused `pytest`（mock replay/RPC/contracts）；`validate-config`；`mock-replay-record --output artifacts/runs/task005-mock-public-trace-20260811T111700Z/mock_public_replay.json --seed 101 --side blue --max-steps 12`；对应 `mock-replay-verify` 与 `verify-artifact`；`env-smoke --episodes 100`。

验证结果：最终全量 `make check` 通过：Ruff、strict mypy（23 source files）、pytest（65 passed）、safety scan（83 files，0 finding）和 package integrity（74/74 controlled files）均通过。focused artifact/replay suite `27 passed`，3 份 JSON schema 与 4 份 YAML 有效。record 与 fresh verify 使用不同 child PID（418/421），2 条 transition 完整终局；任意输出文件名也由 `artifact_kind` 自动绑定 mock schema 与 typed row-chain 合同，artifact self-hash/schema 通过。测试还覆盖同 seed/profile 字节一致、exclusive-create、rehashed action 篡改、私密字段、非公开 target、行链/终局/tick 篡改、控制面不得触发、verifier 不改输入和自定义文件名的语义篡改拒绝。

生成 artifact：`artifacts/runs/task005-mock-public-trace-20260811T112400Z/custom-trace-name.json`（Git 忽略），artifact hash=`sha256:0db29e47552678b2393283e88a9a7b9cbc9e9a34ec2b54eead0c0bbcfa049c32`，file hash=`sha256:3daf6c9e3002abca47d6982a95c839463798cb4426336373d1f6a254bf0bae97`。100 局 mock artifact：`artifacts/runs/m0-mock-smoke-20260811T112426Z/`，`100/100` complete、terminal rate=`1.0`、deterministic replay=`true`、protocol/action-decode/illegal/replay errors=`0`；runtime source commit=`0062deda7f737a008cc1fe46aa3d4778fb55c897`，`dirty=false`。

已知风险：自哈希只证明内容完整性，不证明作者身份；攻击者能重算全部 hash 时可形成另一个形式有效的 mock trace。trace 也不保存 reward、legal mask、privileged state 或 raw observation，故不能代替训练/正式 replay 要求。

外部操作与阻塞：未读取或连接 GameCore/license，未调用 external control-plane，未连接手机或真实商业客户端，未修改 legacy。GameCore binary、license 接入方式、外部授权证据、真实 service 协议仍未提供；状态继续为 `WAITING_EXTERNAL`。

下一任务：保持 `TASK-005 EXTERNAL_ACCESS_WAITING`，等待 Git 外的获授权 GameCore、license 接入方式和 service 启动/协议资料；齐全后先进入 `TASK-010` 的隔离 service preflight。不得以本 trace 或 mock smoke 替代该输入。

## 7. TASK-006 无 GameCore 路线纠正实际记录

日期：`2026-08-11`

任务：`TASK-006 NO_GAMECORE_ROUTE_RESET`

状态：`DONE`。没有启动训练、没有 checkpoint、没有实现 PixelArena gameplay，也没有连接外部服务或设备。

路线结果：GameCore 从唯一主线权威改为 `enabled=false / dependency_of_main_route=false / NOT_AVAILABLE` 的可选外部校准轨；项目主线改为 PixelArena-Structured → baseline/dataset → BC → recurrent PPO → league/多原型 → 3v3 → RGB-only PixelArena → 真实客户端只读 Shadow Coach。所有内部能力结论必须绑定 `claim_scope=pixelarena_internal`、environment/ruleset/schema/suite identity，不能称为 HoK/GameCore 能力。

变更文件：新增 `ROUTE_NO_GAMECORE_V1.md`、`TASK_006_NO_GAMECORE_ROUTE_RESET.md`、`TASK_010_PIXELARENA_STRUCTURED_FOUNDATION.md` 和 `configs/pixelarena_local_v1.yaml`；同步章程、安全、里程碑、环境、策略、训练、评测、数据、runbook、legacy 迁移、风险、决策、README/启动提示、program/eval config、可选 GameCore service 说明、control-plane、tests 和 package manifest。

控制面结果：`strategy_authority.kind=pixelarena`；可选 GameCore gate 与项目 `program.current_status` 解耦，并严格要求 optional track `enabled=true`、external status/connection、attestation、非秘密 evidence ref、显式 operation unlock 和 runtime valid license。操作名限定为 `gamecore_transport/gamecore_evaluation/gamecore_promotion`。测试证明 disabled-but-ready 仍在 service health 前拒绝，也证明 PixelArena identity 不消费该外部 gate。

运行命令：

- `.venv/bin/python -m pytest -q tests/test_task005_control_plane.py tests/test_access_gate_cli.py tests/test_config_validation.py`；
- `.venv/bin/python -m hok_agent env-smoke --config configs/run_smoke_v1.yaml --episodes 100`；
- 三次 `.venv/bin/python -m hok_agent access-gate ... --runtime-license-status valid`，分别覆盖 GameCore transport/evaluation/promotion；
- `.venv/bin/python -m hok_agent package-integrity --root . --write`；
- `make check PYTHON=.venv/bin/python`；
- `make validate PYTHON=.venv/bin/python`；
- `git diff --check`；
- `git -C ../pixel-moba-codex-starter status --short` 与 `rev-parse HEAD`（只读；源目录无 `.git`，命令按预期无法提供 commit/status）。

验证结果：focused route/control-plane/config suite `14 passed`；最终 Ruff 通过，strict mypy 对 23 个 source files 通过，pytest `67 passed`，safety scan 扫描 87 files、0 finding，package integrity `78/78` controlled files 通过；5 份 YAML 与 3 份 JSON schema 有效；`git diff --check` 通过。100 局 mock regression 为 `100/100` completed、terminal rate=`1.0`、deterministic replay=`true`、protocol/action-decode/illegal/replay errors 全为 0，manifest/report verification 均为 true。三个可选 GameCore gate 均以 `WAITING_EXTERNAL`、exit=`2` 拒绝；runtime 自报 `valid` 没有解锁作用。

生成 artifact：`artifacts/runs/m0-mock-smoke-20260811T115415Z/`（Git 忽略）。它只证明 M0 mock 基础设施未被路线变更破坏，不证明 PixelArena 或任何策略能力。

legacy：只读审阅确认可借鉴 deterministic core、snapshot/replay、public/privileged 数据隔离、raw/executed action 和 episode audit 思想；没有迁移代码、checkpoint、optimizer、K96、scorer、grammar、cache、阈值、设备配置或能力结论。当前 legacy 源目录没有 `.git`，因此本轮不能提供其 commit identity；未从文件名猜测。

已知风险与未实现项：PixelArena ruleset/service、scoped evidence schema v2、canonical gameplay replay、baseline、Actor/Critic、trainer、league、renderer 和 Shadow Coach 均尚未实现。现有 v1 schema 继续把 PixelArena 锁为 diagnostic-only，`eval_suite_1v1_v1.yaml` 保持 disabled，直到 TASK-010 先完成 scoped schema 与 registry。没有 checkpoint 可晋升。

外部操作：未探测或连接 GameCore，未读取许可证，未联网下载二进制，未连接手机/客户端/输入设备，未修改 legacy。

下一任务：`TASK-010 PIXELARENA_STRUCTURED_FOUNDATION / READY`。只实现 scoped schema、V5 原生 PixelArena ruleset/service、public/privileged projection、replay/snapshot、NULL/random/scripted baseline、100 局 stability 和 throughput；不启动 BC/PPO。
