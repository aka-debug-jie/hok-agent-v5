# Pixel-MOBA v4.x 冻结与 V5 迁移计划

## 1. 冻结决定

旧仓库：

`aka-debug-jie/pixel-moba-codex-starter`

保留为 legacy evidence。

只在 V5 记录：

- 只读检查时的 legacy commit；若源目录没有 `.git`，如实记录 unavailable，不从文件名猜测；
- 可复用工程模式和禁止迁移清单；
- 旧状态只能引用 legacy 自身当前台账；
- 不再创建 v4.4.1。

不得为了 V5 冻结而修改、tag 或提交 legacy 仓库。

## 2. 先修正状态权威

V5 只读取旧仓库 `DELIVERY_PROGRESS.md` 的当前状态，不替它修正台账，也不把旧报告当作 V5 能力证据。V5 自己必须写明：

- legacy 只读；
- 没有迁移 checkpoint/optimizer/K96/cache/阈值；
- 只借鉴了哪些接口或测试思想；
- legacy 的陈旧 manifest、文档和台账冲突不能成为 V5 authority。

不重解释或重写历史结果。

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

V5 重新实现并承担：

- structured 1v1/3v3 内部训练与评测权威；
- scoped checkpoint promotion；
- snapshot/restore；
- effectful事件测试；
-视觉随机化；
-稀有场景；
-counterfactual；
-像素蒸馏。

仍然禁止：

- 证明官方王者能力；
- 驱动最终策略架构；
- 为每个 teacher miss 扩展候选 grammar；
- 将内部 promotion 表述为 GameCore/商业游戏能力。

## 6. Legacy 最终处理

不再运行 collect、训练、评测、tag 或提交。若未来需要更多证据，只做范围明确的只读审阅，并将新结论写入 V5；不得向 legacy 写入 postmortem 或索引。

## 7. 完成门

- V5 记录 legacy 只读来源；有 `.git` 时记录 commit，无 `.git` 时记录 unavailable；
-新仓库不依赖旧 runtime path；
-没有 v4 checkpoint输入；
-新 CI 不需要旧报告；
-迁移模块有独立测试；
-状态台账不会跨仓库互相覆盖。
