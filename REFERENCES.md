# 参考资料

> 核验日期：2026-08-11。  
> GameCore 当前不是主线依赖。以下外部资料只服务于未来可选校准轨或研究背景，不表示项目已授权、接通或需要等待它们。

## 1. Tencent Honor of Kings AI Open Environment

- Repository: https://github.com/tencent-ailab/hok_env
- GameCore application: https://aiarena.tencent.com/aiarena/en/open-gamecore
- Upstream documentation: https://aiarena.tencent.com/hok/doc/

核验时 upstream README 说明：

- 环境用于与 Honor of Kings GameCore 交互；
- 提供 SDK、RL framework 和 PPO 示例；
- 已提供 1v1 与 3v3 路径；
- Python要求为 3.6–3.9；
- GameCore运行涉及 Windows，Linux容器通常通过 WSL2/Docker 等方式连接；
- GameCore需要单独申请 license；
- 提供 ABS replay工具说明；
- README列出约20个目标英雄/英雄配置（实际使用时应重新查询）。

这些事实只说明 upstream 能力，不说明本项目已经获得许可证或已接通。

## 2. Honor of Kings Arena

Hua Wei et al.  
“Honor of Kings Arena: an Environment for Generalization in Competitive Reinforcement Learning.”  
NeurIPS Datasets and Benchmarks 2022.  
https://arxiv.org/abs/2209.08483

研究背景用途：

- 环境、observation/action/reward 设计；
- 英雄和对手泛化；
- 研究基线。

## 3. Full MOBA RL

Deheng Ye et al.  
“Towards Playing Full MOBA Games with Deep Reinforcement Learning.”  
https://arxiv.org/abs/2011.12692

可借鉴：

- curriculum self-play；
- policy distillation；
- off-policy adaptation；
- multi-head value；
- opponent/hero scalability。

不能据此承诺单张 RTX 4090 复现职业级完整 5v5。

## 4. Supervised MOBA Policy

Deheng Ye et al.  
“Supervised Learning Achieves Human-Level Performance in MOBA Games: A Case Study of Honor of Kings.”  
https://arxiv.org/abs/2011.12582

用途：

- 说明完整行为轨迹和统一宏观/微观建模可作为 RL warm-start；
- 不意味着本项目拥有同规模人类数据。

## 5. Legacy Project

- https://github.com/aka-debug-jie/pixel-moba-codex-starter

仅作只读历史证据和工程模式参考。V5 重新实现自己的 PixelArena ruleset，不复制旧 runtime、模型、K96、配置或能力结论。
