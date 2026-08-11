# 数据、Artifact 与 Lineage 合同

## 1. 数据平面

### 1.1 Strategy training

来自授权 GameCore/研究环境：

- structured public observation；
- optional privileged Critic state；
- legal action；
- actions；
- reward components；
- terminal/outcome；
- replay identity。

### 1.2 Strategy evaluation

与训练 episode/seed/opponent/side 分离。

### 1.3 Vision training

- PixelArena rendered RGB；
- 可合法对齐的研究 replay；
- public-state labels；
- teacher action/value distillation；
- renderer lineage。

### 1.4 Real unlabelled

真实客户端只读视频：

- 自监督；
- temporal consistency；
- OOD；
- shadow diagnostics。

不用于 policy online exploration。

### 1.5 Real labeled dev

- 感知、renderer 和 threshold 开发；
- 不进入 strategy gradient；
- 不进入 checkpoint selection 的未声明路径。

### 1.6 Real frozen audit

- 最终 shadow evaluation；
- 不进入训练、适配、阈值、选择或人工反复调试。

## 2. 分割规则

按完整 source 分割：

- episode；
- seed；
- opponent；
- side；
- hero/lineup；
- source trajectory；
- renderer family；
- real session/clip。

禁止 adjacent-frame random split。

同一 trajectory 的所有 RGB rerender 必须继承同一 split。

## 3. 目录建议

```text
artifacts/
  runs/<run_id>/
    run_manifest.json
    config.resolved.yaml
    metrics.jsonl
    checkpoints/
    evaluation/
    logs/
    replay_index.json
  datasets/<dataset_id>/
    manifest.json
    public/
    privileged/
  league/
    registry.json
    ratings.jsonl
```

大型 artifact 默认不进入 Git。

## 4. Run Manifest

每次训练/评测都必须生成：

- run ID；
- created time；
- git commit；
- dirty state；
- environment identity；
- SDK/GameCore build；
- config hash；
- seed registry；
- algorithm；
- input checkpoint；
- output checkpoint；
- dataset/opponent suite；
- compute；
- safety locks；
- status；
- artifacts。

字段合同见 `schemas/run_manifest.schema.json`。

## 5. Checkpoint Payload

必须包含：

- artifact version；
- Actor state；
- Critic state（训练用）；
- optimizer/scheduler；
- normalization；
- architecture config；
- action schema；
- observation schema；
- training step；
- parent checkpoint；
- run ID；
- model hash；
- safety flags；
- promotion status。

导出部署 Actor 时必须移除：

- Critic；
- privileged encoder；
- optimizer；
- reward normalization中不必要的训练私有信息。

## 6. Hash 语义

必须 hash：

- config；
- source code commit；
- environment/build；
- dataset manifest；
- checkpoint；
- evaluation suite；
- report。

不要求把所有 CUDA logits做文件级 hash authority。

## 7. Replay

保存：

- episode ID；
- environment identity；
- seed；
- actions；
- reward；
- terminal；
- outcome；
- upstream replay/ABS identity；
- optional observation hashes。

若上游不能完全确定性重放，必须明确记录可重复层级，不能伪装为 exact。

### 7.1 Mock public diagnostic trace（非 canonical replay）

`mock_public_transition_replay` 是 TASK-005 中只用于 deterministic `mock` 的公开诊断 trace，不是本节 canonical replay，也不满足上文 reward、upstream replay/ABS identity 的保存要求。

它必须固定为：

- `environment_kind=mock`；
- `formal=false`；
- `capability_claim=none`；
- 内建固定 mock profile；
- 单局完整 transition、canonical self-hash 与逐行 hash chain；
- 仅保存 seed、side、mode、mock episode 关联符、环境 identity hash、factorized action、tick、terminal/truncated、公开 outcome 和公开 observation 的 hash。

它不得保存：

- reward 或 reward component；
- legal action set/mask；
- teacher、truth、privileged/Core state；
- raw observation、upstream/ABS identity、internal replay identity/hash；
- 账号、设备、路径、许可证、GameCore 或真实客户端数据；
- 训练 entity ID 或不可见目标 ID。

验证器必须在 fresh spawned mock process 中重放已记录 action，并比较逐 transition 的公开 observation hash、tick、terminal/truncated 与公开 outcome。该验证只证明 mock 基础设施的公开路径可重放；不得用于训练、formal evaluation、promotion 或任何 HoK/GameCore 能力结论。

## 8. Public / Privileged 分离

建议物理分离：

```text
episode/public.*
episode/privileged.*
```

public loader无法看到 privileged目录。

Actor unit test随机置乱 privileged内容后输出必须不变。

## 9. Artifact 保留

永久保留：

- active/best checkpoint；
-正式 E2/E3报告；
- failed promotion reports；
- dataset manifests；
- league registry；
- safety incident evidence。

可清理：

-中间 tensor dump；
-重复 mini checkpoint；
-临时 renderer cache；
-非权威 debug视频。

清理不得破坏正式报告可解释性。

## 10. Git 忽略

至少忽略：

```text
.venv/
.env
secrets/
licenses/
gamecore/
artifacts/
datasets/
checkpoints/
replays/
*.abs
*.mp4
*.mkv
*.pt
*.pth
*.onnx
*.engine
license.dat
```

## 11. 数据隐私

真实视频写盘前：

- 去除账号名/聊天/通知；
- 不保存联系人和支付信息；
- 记录用途和保留期；
- 不上传公开仓库；
- 用户可删除。

## 12. 报告诚信

- 不覆盖失败 artifact；
- 不把 smoke称为 formal；
- 不把 mock环境称为 GameCore；
- 不把 PixelArena胜率合并进 HoK胜率；
- 不从文件名猜测事实；
- 只有 schema-valid、自哈希报告才可作为自动 promotion 输入。
