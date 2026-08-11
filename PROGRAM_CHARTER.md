# HoK-Agent V5 项目章程

## 1. 使命

构建一个可复核、可扩展的 MOBA 智能体研究系统：

1. 在**腾讯授权的 Honor of Kings GameCore 或兼容研究环境**中，训练能够自主完成完整对局的策略；
2. 将获胜策略逐步扩展到多英雄和 3v3；
3. 将策略蒸馏到仅依赖公开像素及其派生状态的学生；
4. 在真实商业客户端上只提供 read-only 的战术建议、分析和复盘。

## 2. North Star

V5 的第一 North Star 不是静态动作准确率，而是：

> 固定英雄的结构化 1v1 Agent，在预注册的完整 episode 评测中，能够稳定结束对局、对塔和水晶产生可归因进展，并在红蓝双方都以统计上可信的优势击败固定基线对手。

长期 North Star：

> 多英雄、可自博弈、可在 3v3 中体现团队协作，并能通过视觉学生在项目自有像素环境中重现主要能力。

## 3. 三层产品定义

### 3.1 Strategy Agent

- 输入：研究环境提供的结构化 public observation 与历史。
- 输出：因子化合法动作。
- 环境：授权 GameCore/研究环境。
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

- 官方/授权环境接通并可稳定跑完整 episode；
- 固定英雄 1v1 recurrent policy 建立；
- 对 common AI 或等价固定基线达到预注册胜率门；
- 自动训练、评测、checkpoint 晋升和回滚可运行。

### 4.2 六个月目标

- 多英雄 1v1 共享策略；
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
- 不再把 K96/scorer 局部修复作为项目主线；
- 不以训练 reward、静态 Top-k 或单个 canary 代替完整对局能力。

## 6. 核心研究问题

1. 在可承受算力下，固定英雄 1v1 的最小可行 policy/algorithm 是什么？
2. 哪些结构化 public observation 足以支撑完整对局？
3. opponent league、课程和多头价值对稳定性分别有多大贡献？
4. 多英雄共享表示如何避免灾难性遗忘？
5. 3v3 中如何分解团队 reward 和个人贡献？
6. pixel-derived belief state 的信息损失如何限制 strategy transfer？
7. 真实视频上的域差可以缩小到何种程度？

## 7. 决策原则

- Strategy first, vision later。
- 闭环结果优先于代理指标。
- 先固定英雄，再多英雄；先 1v1，再 3v3；先 3v3，再讨论 5v5。
- 复杂度必须由已通过的门解锁。
- 每个失败都应产生一个可验证的失败分类，而不是自动派生下一版 patch。
- 能用成熟、授权环境验证的问题，不再由自建近似环境重复承担。
