# V5 风险登记表

| ID | 风险 | 概率 | 影响 | 触发器 | 缓解 | 止损 |
|---|---|---:|---:|---|---|---|
| R1 | GameCore许可证/二进制无法获得 | 中 | 极高 | preflight缺失 | 提前申请；Mini/PixelArena开发基础设施 | 状态 `WAITING_EXTERNAL`，不伪造官方结果 |
| R2 | Windows/WSL/Docker兼容性 | 中 | 高 | reset/step不稳定 | 独立Python 3.8服务；容器化；版本锁定 | 不污染learner；冻结adapter版本 |
| R3 | GameCore吞吐过低 | 中 | 高 | env-steps/s不足 | 多实例、RPC batching、异步rollout | 未测吞吐不启动大PPO |
| R4 | Upstream action schema不明确 | 中 | 高 | decode/illegal | schema snapshot、round-trip test | action mapping未闭合则M1 No-Go |
| R5 | Reward Goodhart | 高 | 高 | reward升、win不升 | terminal/塔/水晶promotion；NULL审计 | 冻结失败reward，重新预注册 |
| R6 | BC闭环分布偏移 | 高 | 中 | static高、闭环差 | PPO/on-policy；完整episode | 不继续堆IID CE |
| R7 | PPO策略坍塌 | 中 | 高 | dominant action/entropy异常 | BC retention、KL、curriculum、rollback | 自动回滚 |
| R8 | 固定对手过拟合 | 高 | 高 | common AI强、其他崩 | opponent league、historical/exploiter | 不晋升 |
| R9 | Side bias | 中 | 中 | 红蓝差异 | 平衡数据和评测 | 任一side低于门不晋升 |
| R10 | 多英雄灾难性遗忘 | 高 | 高 | 新英雄加入后旧英雄退化 | shared+hero embedding、retention suite、distillation | 回滚英雄扩展 |
| R11 | 3v3 lazy agent/credit assignment | 高 | 高 | 某hero长期零贡献 | centralized critic、individual/team heads、role metrics | 3v3不晋升 |
| R12 | Structured→pixel信息不足 | 高 | 高 | belief probe ceiling低 | 先state probe；改visual representation | 不调policy scorer掩盖 |
| R13 | Replay和RGB无法精确对齐 | 中 | 高 | ABS对齐失败 | 单独技术验证；PixelArena辅助 | 不宣称端到端蒸馏 |
| R14 | 项目再次文档/版本膨胀 | 高 | 中 | 每失败新增协议 | 五份权威合同+程序化manifest | 禁止无任务的版本文档 |
| R15 | CUDA数值脆弱性 | 中 | 中 | 小扰动改变动作 | margin/rank robustness，不要求byte exact | 低margin policy不晋升 |
| R16 | 商业客户端边界滑坡 | 低 | 极高 | 出现输入/反检测代码 | CI扫描、读写隔离 | 立即停止并删除违规路径 |
| R17 | 真实数据隐私泄漏 | 中 | 高 |账号/聊天入库 | 去标识、私有存储、retention | 隔离和删除 |
| R18 | 单4090不足以支持最终3v3 | 中 | 中高 |训练速度不可接受 | 小模型、共享策略、异步rollout | 缩小英雄池/任务，不虚报5v5 |
| R19 | 上游项目停止维护 | 中 | 高 |依赖不可安装 | adapter隔离、fork SDK层、接口测试 | 固定可用版本 |
| R20 | 自动实验无界消耗 | 中 | 中 |连续失败自动重试 | 预算、gate、人工批准 | 超预算自动停止 |

## 风险审阅规则

- 每个里程碑开始前更新风险。
- 触发器出现后，`DELIVERY_PROGRESS.md` 必须记录。
- 高影响风险不能仅以“后续关注”关闭。
- 涉及安全、许可证或真实客户端的风险没有性能补偿。
