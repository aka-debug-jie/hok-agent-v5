# 环境与服务合同

## 1. 环境权威

V5 使用四个相互隔离的环境范围。当前可执行主线是项目自有 PixelArena；GameCore 不是 fallback 或依赖。

| 环境 | 权威范围 |
|---|---|
| deterministic mock | 单元测试、RPC、CI 与诊断；永不产生策略能力或 promotion |
| PixelArena-Structured | 指定 ruleset 的项目内部 1v1/3v3 训练、评测和 scoped promotion |
| PixelArena-RGB | 指定 renderer 的感知、蒸馏与 RGB-only 项目内部闭环 |
| 可选授权 Honor of Kings GameCore / `hok_env` | 未来独立授权后的外部校准；独立 registry，不阻塞或继承主线 |

同一报告必须明确 environment family、identity、ruleset 与 `claim_scope`，不得合并不同环境或 ruleset major 的胜率。PixelArena 结果不得称为 HoK/GameCore 能力。

## 2. 进程隔离

PixelArena 和 learner 均使用 Python 3.11+，但仍以进程和 RPC 分离，确保环境与训练可独立复核：

```text
┌──────────────────────────────┐
│ pixelarena-service           │
│ Python 3.11+ / V5 ruleset    │
│ isolated local process       │
└──────────────┬───────────────┘
               │ versioned RPC
┌──────────────▼───────────────┐
│ learner/evaluator            │
│ Python 3.11+ / PyTorch 2.x   │
└──────────────────────────────┘
```

transport 与 contract 必须分离；learner 不直接读取环境内部状态。任何未来 GameCore transport/service factory 仍必须先通过可选外部访问 gate，才可创建 socket、进程或 SDK service；它不得复用 PixelArena registry。

## 3. RPC 最小接口

### 3.1 `Health`

请求：

```json
{}
```

响应至少包含：

```json
{
  "service_version": "string",
  "environment_kind": "mock|pixelarena|hok_gamecore",
  "sdk_version": "string",
  "gamecore_build": "string|null",
  "license_status": "valid|missing|invalid|unknown|not_applicable",
  "supported_modes": ["1v1"],
  "supported_heroes": ["..."],
  "ready": true
}
```

当前 protocol v1 的 `supported_heroes`、`hero_config` 字段为已发布前的兼容名称。TASK-010 必须作显式协议版本决议，并在 PixelArena v2 使用原型语义；不得把这些字段解释为真实 HoK 英雄支持。

### 3.2 `Reset`

输入：

- `request_id`
- `mode`
- `seed`
- `side`
- `hero_config`
- `opponent_config`
- `reward_config_version`
- `evaluation_mode`

输出：

- 原样回显 `request_id`；
- `episode_id`
- `tick/frame`
- observation；
- legal action；
- reward component names；
- environment identity；
- initial replay identity。

### 3.3 `Step`

输入：

- `episode_id`
- `expected_tick`
- factorized action；
- action schema version。

输出：

- 原样回显 `episode_id`；
- next observation；
- reward vector；
- terminal/truncated；
- outcome；
- legal action；
- tick；
- replay hash；
- error code。

client 必须将 reset 的 `request_id` 和 step 的 `episode_id` 与本地请求关联；任何错配 response 都不得继续执行动作，并必须关闭已知本地 session、毒化当前 client/transport。毒化后只能进行 best-effort `close` 清理，必须重建隔离 transport 后才能恢复调用。

### 3.4 `Close`

必须幂等，并输出 stop reason。

## 4. Observation 合同

环境内部 observation 可以是扁平数组，但 adapter 必须保存：

1. 原始环境 observation（仅内部 replay/adapter 路径）；
2. feature names/version；
3. 规范化 public observation；
4. optional privileged critic observation；
5. mapping hash。

不得仅保存无语义的向量而丢失 schema identity。

规范化 public observation建议包含：

- self state；
- visible allies/enemies；
- minions/structures/resources；
- public spatial features；
- cooldown/economy/level；
- action history；
- side/hero public identity；
- visibility mask。

不可见敌方真值只能进入 privileged Critic/teacher 路径。

## 5. Factorized Action 合同

禁止把 K96 作为 V5 主动作空间。

统一动作结构：

```text
macro_intent
action_type
target_type
target_index/public_target_key
direction
skill_slot
item_slot
auxiliary_parameter
```

分布：

```text
P(macro)
× P(action_type | macro)
× P(target | macro, action_type)
× P(direction/skill/item | selected branches)
```

adapter 负责把因子化动作转换成环境 executed action。

## 6. Legal action

- 环境 legal action 保留原始值和 mapping hash；
- legal mask 只在动作采样/执行前使用；
- Actor forward 不读取 legal mask；
- action 被 mask 后仍必须记录 raw sample、masked sample 和最终 executed action；
- 任何 mapping 不确定时拒绝执行。

## 7. Reward 合同

报告必须记录命名 reward components，不允许只有总和。

最低建议：

- win/crystal；
- tower/structure；
- economy；
- experience；
- last hit；
- hero damage；
- death；
- objective/resource；
- time/step；
- environment raw reward；
- shaped total。

`shaped total` 必须等于命名分量求和。

promotion 始终优先使用真实 terminal、胜负和结构结果，不允许只看 shaped reward。

## 8. Determinism 合同

必须精确：

- seed registry；
- environment/build/config identity；
- action decode；
- reward decomposition；
- terminal/outcome；
- episode split；
- saved replay identity。

神经网络 logits 不要求不同硬件逐字节相同，但必须：

- finite；
- 相同 checkpoint 的 action distribution 在容差内；
- fixed evaluation 能给出统计一致结论。

### 8.1 Mock public diagnostic trace

TASK-005 可用一个由 verifier 自己启动的 fresh `ProcessJsonTransport` 对单局 `mock_public_transition_replay` 重放。它只使用内建 mock profile，不读取 `program_v1.yaml`、不创建 GameCore transport、不接收 license/service/phone 参数，也不接触 `ExternalAccessGate`。

legal action 在 record/replay 进程内瞬时用于 sampling/execution audit，不能写入 trace；trace 只保留 public observation 的 canonical hash。任何 non-mock、formal、capability claim、未知/私密字段、hash chain 断裂或逐步不一致都必须 fail-closed。

## 9. 吞吐 benchmark

M1 对 PixelArena-Structured 必须测：

| 并行实例 | env-steps/s | episode/hour | p50 RPC | p95 RPC | CPU | RAM | GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | |
| 4 | | | | | | | |
| 8 | | | | | | | |
| 16 或上限 | | | | | | | |

benchmark 中 policy 使用固定轻量随机/常数模型，避免把模型速度混入环境吞吐。

## 10. GameCore 缺失时的状态

GameCore 和许可证当前不存在。可选外部轨保持 `NOT_AVAILABLE`；本地主线继续允许：

- mock 基础设施；
- 实现和测试项目自有 PixelArena ruleset/service；
- 在相应里程碑解锁后开展 PixelArena 内部 BC/PPO/league/3v3 与 scoped promotion；
- PixelArena-RGB 与真实客户端只读 Shadow；
- 保留可选外部轨状态，不主动猜测或下载受限输入。

禁止：

- 把 mock 胜率用作任何策略能力；
- 把 PixelArena 胜率称为 HoK/GameCore 胜率；
- 假设 GameCore 支持英雄和 action schema；
- 把网上二进制或许可证镜像进仓库。

## 11. PixelArena identity 最低字段

每个 PixelArena run 必须绑定：package/code hash、ruleset ID/version、map ID、tick rate、observation/action/reward schema、原型/lineup registry、max episode steps、seed registry 和 service version。像素 run 还必须绑定 renderer family/version。identity 任一变化都不得静默复用旧 active。
