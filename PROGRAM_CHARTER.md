# HoK-Agent V5 项目章程

## 1. 使命

构建一个可复核、可扩展的 MOBA 智能体研究系统：

1. 在项目自有、版本化的 **PixelArena** 中，训练能够自主完成完整对局的策略；
2. 将获胜策略逐步扩展到多原型和 3v3；
3. 将策略蒸馏到仅依赖公开像素及其派生状态的学生；
4. 在真实商业客户端上只提供 read-only 的战术建议、分析和复盘。

## 2. North Star

V5 的第一 North Star 不是静态动作准确率，而是：

> 固定原型的结构化 1v1 Agent，在预注册的 PixelArena 完整 episode 评测中，能够稳定结束对局、对结构和水晶产生可归因进展，并在双方都以统计上可信的优势击败固定基线对手。

长期 North Star：

> 多原型、可自博弈、可在 3v3 中体现团队协作，并能通过视觉学生在 PixelArena-RGB 中重现主要能力。

以上均为 `pixelarena_internal` 范围，不是 HoK/GameCore 能力主张。

## 3. 三层产品定义

### 3.1 Strategy Agent

- 输入：研究环境提供的结构化 public observation 与历史。
- 输出：因子化合法动作。
- 环境：项目自有 PixelArena-Structured。
- 目标：完整对局胜率、Elo、结构和水晶结果。

### 3.2 Pixel Student

- 输入：RGB、pixel-derived public belief state、历史动作。
- 输出：与 Strategy Agent 同构的因子化动作分布。
- 环境：PixelArena、可对齐的研究 replay。
- 目标：在不读取 privileged truth 的情况下重现 Strategy Agent 能力。

### 3.3 Shadow Advisor

- 输入：真实客户端只读视频或回放。
- 输出：Linux 端建议、风险、解释和复盘。
- 目标：有用、低延迟、置信度可校准、失败关闭。
- 不输出真实客户端控制事件。

## 4. 主要成功标准

### 4.1 三个月目标

- PixelArena-Structured ruleset/service 可稳定跑完整 episode；
- 固定原型 1v1 recurrent policy 建立；
- 对项目内 scripted/league 固定基线达到预注册胜率门；
- 自动训练、评测、checkpoint 晋升和回滚可运行。

### 4.2 六个月目标

- 多原型 1v1 共享策略；
- opponent league 与 Elo；
- 至少一个可复现的 3v3 学习基线；
- RGB→public belief state 的第一版；
- strategy/vision 误差能够独立归因。

### 4.3 长期目标

- 3v3 有稳定团队能力；
- PixelArena 中 pixel-derived closed loop；
- 真实客户端只读智能教练与复盘工具。

## 5. 明确非目标

- 不在商业服、排位或普通账号上运行自动闭环 Agent；
- 不实现触控注入、反检测、客户端修改或协议自动化；
- 不以单张 RTX 4090 复现职业级全英雄 5v5 为近期承诺；
- 不把自建 PixelArena 的胜率描述为官方王者 GameCore 胜率；
- 不把真实王者英雄名、技能或数值当作未获授权 PixelArena 的默认 ruleset；
- 不再把 K96/scorer 局部修复作为项目主线；
- 不以训练 reward、静态 Top-k 或单个 canary 代替完整对局能力。

## 6. 核心研究问题

1. 在可承受算力下，固定原型 1v1 的最小可行 policy/algorithm 是什么？
2. 哪些 PixelArena 结构化 public observation 足以支撑完整对局？
3. opponent league、课程和多头价值对稳定性分别有多大贡献？
4. 多原型共享表示如何避免灾难性遗忘？
5. 3v3 中如何分解团队 reward 和个人贡献？
6. pixel-derived belief state 的信息损失如何限制 strategy transfer？
7. 真实视频上的域差可以缩小到何种程度？

## 7. 决策原则

- Strategy first, vision later。
- 闭环结果优先于代理指标。
- 先固定原型，再多原型；先 1v1，再 3v3；先 3v3，再讨论更大规模。
- 复杂度必须由已通过的门解锁。
- 每个失败都应产生一个可验证的失败分类，而不是自动派生下一版 patch。
- 结论必须绑定环境与 ruleset；项目内部胜率不外推为商业游戏能力。

## 8. 可选外部校准

GameCore 当前不存在，也不属于 M0–M8 依赖。未来若获得明确授权，只能新建立项、使用独立报告和 registry 进行外部校准；不得自动继承 PixelArena promotion，也不得改变真实客户端只读边界。
