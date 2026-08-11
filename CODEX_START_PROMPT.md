# Codex 继续执行提示词

下面内容可直接交给新的 Codex 线程：

```text
接管 `/media/hgdl1012/84f6bd42-d506-47ff-8bf3-7b354ba37618/media/hgdl1012/program/hok-agent-v5`。

项目已确认没有 GameCore；不要等待、询问或尝试接入 GameCore/license。主线是项目自有 PixelArena-Structured → 固定原型 1v1 baseline/dataset → BC → recurrent PPO → league/多原型 → 3v3 → RGB-only PixelArena → 真实客户端只读 Shadow Coach。GameCore 仅是默认关闭、不阻塞主线的未来可选校准轨。

先完整读取 AGENTS.md、PROGRAM_CHARTER.md、SAFETY_BOUNDARIES.md、DELIVERY_PROGRESS.md、ROUTE_NO_GAMECORE_V1.md、MILESTONES.md、所有核心合同和 DELIVERY_PROGRESS 指定的任务文件。DELIVERY_PROGRESS 是当前状态唯一权威。

当前只实施 TASK_010_PIXELARENA_STRUCTURED_FOUNDATION.md：scoped evidence schema、V5 原生 PixelArena ruleset/service、public/privileged projection、factorized action、replay/snapshot、NULL/random/scripted baseline、100 局 stability 与 throughput。不要启动 BC/PPO，不生成可晋升 checkpoint，不作 HoK/GameCore 能力结论。

legacy `pixel-moba-codex-starter` 只读。只能借鉴 deterministic core、snapshot/replay、数据隔离与 episode audit 思想；不得复制旧 runtime，不迁移 checkpoint/optimizer/K96/scorer/grammar/cache/阈值/设备配置或能力结论。

真实商业客户端永远只读。禁止 ADB input、scrcpy control、HID/UHID、uinput、Accessibility、root/hook、内存/协议自动化、反检测和真实账号自动对局。

开始时把台账任务设为 IN_PROGRESS；完成代码、测试、配置、报告后运行 Ruff、strict mypy、pytest、validate-config、safety-scan、package-integrity 和任务级 stability/benchmark；只有全部验收通过才标 DONE，并记录真实命令、结果、artifact、风险和下一任务。
```
