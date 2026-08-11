# Pixel-MOBA v4.x 冻结与 V5 迁移计划

## 1. 冻结决定

旧仓库：

`aka-debug-jie/pixel-moba-codex-starter`

保留为 legacy evidence。

应创建：

- 只读 tag，例如 `legacy-v4.4-freeze-20260811`；
- freeze commit；
- `LEGACY_FINAL_STATUS.md`；
- 失败/成功 artifact索引；
- 不再创建 v4.4.1。

## 2. 先修正状态权威

旧仓库中 `DELIVERY_PROGRESS.md` 必须与真实最新结果一致，至少写明：

- v4.3.5 已运行的实际 disposition；
- rollback；
- v4.4 是否只处于 protocol/implementation；
- runtime artifact未上传的边界；
- V5另开仓库；
-旧仓库不再是 active训练项目。

只修状态和索引，不重解释历史结果。

## 3. 迁移清单

### 可以迁移

按思想/接口重写或抽取：

- canonical hashing；
- artifact verification；
- dataset split；
- replay identity；
- atomic checkpoint；
- safety/secret scan；
- read-only capture；
- state replay；
- shadow UI；
- public/privileged isolation tests；
- PixelArena snapshot/counterfactual；
- structured logging。

### 谨慎迁移

需写新 contract：

- reward component logger；
- recurrent chunk/burn-in工具；
- renderer；
- entity set编码器；
- environment vectorization。

### 不迁移到 V5 主线

- v2/v3/v4 checkpoint；
- optimizer；
- residual parent chain；
- K96 hard candidate generator；
- macro reservation；
- grammar closure；
- v4.x scorer；
-旧 frozen hash authority；
-每轮实验专用 runner；
-过时的行动分类和映射。

## 4. 迁移方式

禁止把整个旧 `src/` 复制到新仓库。

采用：

```text
列出需要的能力
→ 写最小独立接口
→ 从旧实现提取或重写
→ 新测试证明
→ 记录 provenance
```

## 5. PixelArena 新角色

保留：

- mock环境；
- CI；
- snapshot/restore；
- effectful事件测试；
-视觉随机化；
-稀有场景；
-counterfactual；
-像素蒸馏。

不再：

- 证明官方王者能力；
- 驱动最终策略架构；
- 为每个 teacher miss 扩展候选 grammar；
- 替代 GameCore promotion suite。

## 6. v4.4 最终处理

可选地只运行一次 `collect-only`，目的：

-统计 baseline student访问状态上的 effectful structure opportunities；
-形成旧路线 postmortem；
-不训练；
-不创建 checkpoint；
-不解锁后续。

若运行，结果只进入 `legacy/postmortem/`。

## 7. 完成门

- legacy repo有明确 freeze；
-新仓库不依赖旧 runtime path；
-没有 v4 checkpoint输入；
-新 CI 不需要旧报告；
-迁移模块有独立测试；
-状态台账不会跨仓库互相覆盖。
