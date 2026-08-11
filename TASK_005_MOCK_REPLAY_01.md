# TASK-005-MOCK-PUBLIC-TRACE-01

## 状态

- 阶段：`M1` 前置等待期
- 当前状态：`IN_PROGRESS / WAITING_EXTERNAL`
- 环境：仅内建 deterministic `mock`

## 目标

实现单文件、public-only、non-formal 的完整 mock transition trace，并在新 spawned mock service 中独立重放和篡改拒绝。

## 范围

允许：typed trace contract、JSON schema、record/verify CLI、单元测试、公开字段 hash 和 mock RPC。

禁止：训练、checkpoint、GameCore、许可证、control-plane、手机、PixelArena、legacy、真实客户端、reward/teacher/truth/privileged/legal-mask 持久化。

## 验收

1. record 以 exclusive create 写单个 JSON；同一 seed/profile 的输出逐字节一致。
2. trace 有 canonical artifact hash、每行 hash chain，且必须完整终局。
3. verify 在 fresh spawned mock process 中逐 transition 比较 action legality、tick、公开 observation hash、terminal/truncated 与公开 outcome。
4. 删除、重排、重复、hash/action/tick/outcome 篡改和注入私密字段均 fail-closed。
5. record/verify 不读取或改写 `configs/program_v1.yaml`，不调用 external access gate，也不产生 capability/promotion 结论。

## 退出

完成后仍回到 `TASK-005 EXTERNAL_ACCESS_WAITING`。只有 Git 外的外部授权、GameCore、license 和 service 方式齐全后才可进入 TASK-010。
