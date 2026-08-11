# 自动化训练与运维 Runbook

## 1. 目标

建立统一控制器，使实验不再依赖“每次失败手写一套 runner 和协议”。

## 2. 状态机

```text
CREATED
  ↓
PREFLIGHT
  ↓
ENV_SMOKE
  ↓
UNIT_GATE
  ↓
OVERFIT_OR_REWARD_GATE
  ↓
MINI_RUN
  ↓
FULL_RUN
  ↓
CHECKPOINT_FROZEN
  ↓
PROMOTION_EVAL
  ├─ PROMOTED
  ├─ REJECTED
  └─ INVALID_ROLLBACK
```

可选 GameCore 轨缺失：

```text
OPTIONAL_EXTERNAL_PREFLIGHT → NOT_AVAILABLE|WAITING_EXTERNAL
```

这不改变 PixelArena 主状态机。PixelArena 的每个阶段由 ruleset、dataset、mini/full 和 scoped promotion 门控制。任何未来 GameCore transport/evaluation/promotion 入口都必须先调用同一可选外部访问 gate；`access-gate` 只检查该外部轨并绝不连接服务。

## 3. 组件

### Orchestrator

- 读取 resolved config；
- 创建 run ID；
- 调度 worker/learner/evaluator；
- 维护状态；
- 生成 manifest；
- 自动停止；
- 不自行改超参数。

### Rollout Worker

- 连接 environment service；
- 收集完整 episode；
- 保存 action/log-prob/reward/terminal；
- 处理 restart；
- 不写 checkpoint。

### Learner

- 从 replay/rollout读取；
- 执行 BC/PPO/MAPPO；
- 原子 checkpoint；
- finite/gradient检查。

### Evaluator

- 冻结 checkpoint后启动；
- 不共享 optimizer；
- 使用固定 suite；
- 生成正式报告；
- 推导 gate。

### League Manager

- opponent registry；
- Elo；
- active/best/historical/exploiter；
- 不删除失败对手。

## 4. 建议 CLI

```text
python -m hok_agent preflight  # optional GameCore track only
python -m hok_agent validate-config
python -m hok_agent package-integrity
python -m hok_agent access-gate --operation gamecore_transport --runtime-license-status valid
python -m hok_agent env-smoke --config configs/run_smoke_v1.yaml
python -m hok_agent env-benchmark
python -m hok_agent mock-replay-record --output /tmp/mock_public_replay.json
python -m hok_agent mock-replay-verify /tmp/mock_public_replay.json
python -m hok_agent pixelarena-smoke
python -m hok_agent pixelarena-benchmark
python -m hok_agent collect-bc
python -m hok_agent train-bc
python -m hok_agent train-ppo
python -m hok_agent evaluate --suite configs/eval_suite_1v1_v1.yaml
python -m hok_agent promote --report <evaluation_report.json>
python -m hok_agent league status
python -m hok_agent verify-artifact <path>
```

初始任务不要求所有命令都实现，但接口应保持一致。

## 5. 两周执行顺序

### Day 1–3

- 新仓库；
- 文档落库；
- Python 3.11 package；
- schema；
- mock env；
- CI；
- safety/secret scan；
- legacy freeze记录。

### TASK-010 / M1

- scoped evidence schema；
- PixelArena-Structured ruleset/service；
- RPC health/reset/step/close；
- public/privileged projection；
- replay/snapshot 和一个完整 episode。

### TASK-010 关闭门

- 规范化 observation/action adapter；
- NULL/random/scripted baseline；
- replay index；
- throughput benchmark；
- 100-episode stability准备。

### TASK-020/030 前置

- V5-Actor-1 skeleton；
- factorized action/log-prob tests；
- BC dataset contract；
- scripted-teacher collection mini；
- 32-sample overfit；
- BC闭环 smoke；
- 不在两周内承诺长 PPO。

以上是依赖顺序，不是完成时间承诺。只有前一任务报告通过后才进入下一任务。

## 6. 长作业规则

- 每个长作业必须有明确输出目录；
- 支持 graceful stop；
- 定期写 health/metrics；
- checkpoint原子替换；
- resume验证 run/config/environment identity；
- 不因交互终端断开损坏结果；
- 最终状态由报告决定，不由进程退出码单独决定。

## 7. 自动停止

以下触发自动 stop/rollback：

- NaN/Inf；
- illegal/protocol/replay error；
- reward schema漂移；
- loss或KL越过硬上限；
- action collapse持续超过窗口；
- environment error率超过门；
- artifact写入失败；
-安全锁不完整。

## 8. 自动 Promotion

`promote` 命令只能：

1. 读取 schema-valid evaluation report；
2. 重新推导 gate；
3. 校验 checkpoint hash；
4. 原子更新 registry；
5. 保留旧 active；
6. 写 promotion record。

不得接受手工布尔 `pass=true` 而不重新计算。

## 9. 可观测性

最小 dashboard：

- env steps/s；
- episodes/hour；
- win rate rolling；
- episode length；
- reward components；
- action distribution；
- policy/value loss；
- KL/entropy；
- invalid/protocol；
- CPU/RAM/GPU；
- current opponent mix；
- active/best checkpoint。

## 10. 失败后的下一步

失败报告必须给：

- primary failure category；
- first failing gate；
- supporting metrics；
-建议调查模块；
-明确禁止的下一步；
- 是否允许新的 bounded experiment。

控制器不得自动创建“vN+1调参实验”。
