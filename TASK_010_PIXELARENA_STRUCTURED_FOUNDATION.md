# TASK-010 — PixelArena-Structured 基础环境与吞吐

## 1. 状态

- 阶段：M1
- 初始状态：`READY`
- 性质：环境工程；不训练神经策略，不生成可晋升 checkpoint。

## 2. 目标

从零实现 V5 原生 `pixelarena-structured-v1`：一个项目自有、确定性、无头、可完成完整固定原型 1v1 对局的研究环境，并通过既有 `health/reset/step/close` RPC 提供服务。

它是 PixelArena 内部策略研究的环境权威，不是 HoK/GameCore 仿真等价性声明。

## 3. 实施切片

### A. Scoped evidence 与 registry

- 新版本 run/evaluation schema 强制 `evaluation_tier`、`promotion_eligible`、`claim_scope`、environment/ruleset identity；
- mock 永远 diagnostic-only；
- PixelArena 和可选 GameCore 使用独立 registry；
- 不同 ruleset major 不继承 active 或合并胜率。

### B. V5 规则合同

版本化定义：

- 2D 坐标、tick 和边界；
- 固定原型、属性、冷却、死亡与复活；
- 移动、基础攻击、至少一个定向技能和一个目标技能；
- 兵线、塔、水晶、伤害归因、经济/经验；
- terminal win/loss/draw 与 time-limit truncated；
- public observation、privileged sidecar、legal action、reward components；
- factorized action 到执行动作的唯一映射。

### C. Core 与服务

- fresh V5 implementation，不整体复制 legacy runtime；
- seed、snapshot/restore、branch replay；
- public/privileged 物理或逻辑隔离；
- local spawned-process service；
- NULL、deterministic random、scripted objective baseline；
- canonical gameplay replay、environment identity 和 mapping hash。

### D. 评测与吞吐

- 规则 property tests 与 side-symmetry audit；
- 100-episode stability；
- fixed-seed exact replay；
- 1/4/8/16 或宿主可达上限的 throughput；
- CPU/RAM、RPC p50/p95、steps/s、episodes/hour；
- M1 acceptance report，固定 `claim_scope=pixelarena_engineering`。

## 4. 硬验收

- `health → reset → step → terminal/truncated → close` 100/100 完成；
- protocol、decode、illegal-executed、replay、nonfinite error 全为 0；
- 同 identity/seed/actions 精确重放，snapshot 分支不改变原 session；
- 每一类动作分支有 round-trip 和因果测试；
- NULL 不获得英雄可归因结构奖励；scripted baseline 能产生可解释结构进展和终局；
- public loader 看不到 truth、teacher、reward、不可见敌方或训练 entity ID；
- privileged sidecar 随机化后 Actor-facing projection 不变；
- legal mask 仅在 sampling/execution/audit 路径；
- 无 K96 硬候选空间；
- full checks、安全扫描和 package integrity 通过；
- 报告不作 HoK/GameCore 或商业客户端能力结论。

## 5. 停止条件

- 规则合同未闭合、确定性失败、public/privileged 泄漏或 side 严重不对称时先修 M1；
- 不用训练补偿环境缺陷；
- 不下载 GameCore 或移植旧 checkpoint/K96；
- 不因手机存在而创建真实客户端动作路径。

## 6. 完成后的下一任务

`TASK-020 PIXELARENA_FIXED_1V1_BASELINES_AND_DATASET`：冻结 scripted teacher、完整 episode dataset、split/lineage 和 reward audit，仍不启动 PPO。
