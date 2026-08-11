# 环境与服务合同

## 1. 环境权威

V5 使用三类环境：

| 环境 | 权威范围 |
|---|---|
| 授权 Honor of Kings GameCore / `hok_env` | 1v1/3v3 策略能力与胜负权威 |
| Tencent Mini 或其他轻量研究环境 | 分布式 RL、MAPPO、credit assignment 练兵 |
| PixelArena | 单元测试、counterfactual、视觉课程、CI |

同一报告必须明确 `environment_kind`，不得合并不同环境的胜率。

## 2. 进程隔离

上游 `hok_env` 的运行版本约束可能与主训练栈不同，因此固定为服务化：

```text
┌──────────────────────────────┐
│ hok-gamecore-service         │
│ Python 3.8 / upstream SDK    │
│ Windows/WSL/Docker boundary  │
└──────────────┬───────────────┘
               │ versioned RPC
┌──────────────▼───────────────┐
│ learner/evaluator            │
│ Python 3.11+ / PyTorch 2.x   │
└──────────────────────────────┘
```

不得把旧版 SDK 直接安装进 learner 环境后长期耦合。

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
  "environment_kind": "hok_gamecore|mock|pixelarena",
  "sdk_version": "string",
  "gamecore_build": "string|null",
  "license_status": "valid|missing|invalid|unknown",
  "supported_modes": ["1v1"],
  "supported_heroes": ["..."],
  "ready": true
}
```

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

- next observation；
- reward vector；
- terminal/truncated；
- outcome；
- legal action；
- tick；
- replay hash；
- error code。

### 3.4 `Close`

必须幂等，并输出 stop reason。

## 4. Observation 合同

上游 observation 可以是扁平数组，但 adapter 必须保存：

1. 原始 upstream observation；
2. upstream feature names/version；
3. 规范化 public observation；
4. optional privileged critic observation；
5. mapping hash。

不得仅保存无语义的向量而丢失 upstream schema identity。

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

adapter 负责把因子化动作转换成 upstream action。

## 6. Legal action

- upstream legal action 保留原始值和 mapping hash；
- legal mask 只在动作采样/执行前使用；
- Actor forward 不读取 legal mask；
- action 被 mask 后仍必须记录 raw sample、masked sample 和最终 upstream action；
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
- upstream raw reward；
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

## 9. 吞吐 benchmark

M1 必须测：

| 并行实例 | env-steps/s | episode/hour | p50 RPC | p95 RPC | CPU | RAM | GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | |
| 4 | | | | | | | |
| 8 | | | | | | | |
| 16 或上限 | | | | | | | |

benchmark 中 policy 使用固定轻量随机/常数模型，避免把模型速度混入环境吞吐。

## 10. 无许可证时的 fallback

许可证或 GameCore 未获得时，只允许：

- 实现 mock service；
- 实现 RPC；
- 完成 schema/tests；
- 接入开源 Mini/PixelArena；
- 完成 learner 和 evaluator skeleton；
- 记录申请/外部阻塞。

禁止：

- 把 mock/PixelArena 胜率称为 HoK GameCore 胜率；
- 假设支持英雄和 action schema；
- 把网上二进制或许可证镜像进仓库。
