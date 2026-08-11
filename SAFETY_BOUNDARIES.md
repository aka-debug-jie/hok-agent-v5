# 安全、合规与使用边界

本文件是不可降低的边界权威。性能目标、用户要求、开发便利和研究兴趣均不能覆盖本文件。

## 1. 环境分级

| 环境 | 读取 | 自动动作 | 训练 | 允许用途 |
|---|---:|---:|---:|---|
| 腾讯明确授权的 GameCore/AI 开放环境 | 是 | 是 | 是 | 研究、训练、评测、回放 |
| 项目自有 PixelArena/测试模拟器 | 是 | 是 | 是 | 单测、课程、视觉、counterfactual |
| 真实商业客户端/普通手机客户端 | 只读视频/回放 | **否** | 不做在线 RL | Shadow 建议、离线分析、感知评测 |
| 商业服、排位、真人对局账号 | 只读观察也应谨慎 | **绝对禁止自动动作** | 否 | 不作为自动化测试场 |

## 2. 禁止能力

真实客户端路径中禁止：

- `adb shell input`、tap、swipe、keyevent；
- scrcpy control mode；
- `/dev/uinput`、`/dev/input` 写入；
- HID、UHID、OTG 键鼠模拟；
- Accessibility 跨应用触控或自动化；
- root、hook、注入、内存读写；
- 网络协议拦截、重放、修改；
- 客户端文件、资源、进程或设备指纹修改；
- 反作弊检查、干扰、隐藏或规避；
- 为避免检测而进行行为随机化；
- 自动登录、匹配、排位、领取或账号经营。

不得提供“仅作为研究”“只在自己账号”“低频率”等例外。

## 3. 许可证和上游条款

- `hok_env` SDK 的开源许可证不自动等于 GameCore 使用授权。
- GameCore、ABS 工具、许可证文件和相关二进制必须按照腾讯提供的申请和条款使用。
- 未获得明确授权时，项目状态必须是 `WAITING_EXTERNAL`。
- 不得在仓库中镜像或再分发受限 GameCore、许可证、私有资产或 replay 工具。
- 上游条款与本边界冲突时，采用更严格的一方。

## 4. 特权信息隔离

允许进入 training-only 路径：

- 完整 Core state；
- entity identity/kind；
- legal action；
- teacher action；
- reward component；
- terminal/outcome；
- centralized Critic observation。

禁止进入导出 Actor：

- 非公开真值；
- 不可见敌方状态；
- 真实 entity ID；
- teacher label；
- reward；
- opponent 内部状态；
- 仅训练时 legal mask 以外的 privileged hint。

必须自动验证：

1. Actor input schema denylist；
2. export graph 输入审计；
3. 随机化 privileged sidecar 时 Actor logits/hidden 不变；
4. Critic、teacher 和 Actor 参数/输入图分离。

## 5. Legal action 的特殊规则

研究环境提供的 `legal_action` 可以：

- 用于采样或执行前 mask；
- 用于 loss-side 校验；
- 用于 evaluator 统计；
- 用于错误拒绝。

它不应作为普通 representation 特征输入 Actor，也不能被隐藏编码后用于真实客户端。

## 6. 数据与隐私

禁止提交到 Git：

- `license.dat`；
- API key、账号和 cookie；
- GameCore 私有二进制；
- 设备 serial、真实账号 ID；
- 原始真人语音、聊天和个人信息；
- 大型视频、ABS 和 checkpoint。

真实视频如用于感知研究：

- 必须只读；
- 先去除账号名、聊天、好友、定位和通知；
- 以 session 为单位分割；
- real labeled dev 不进入 policy 梯度；
- frozen audit 不进入阈值、checkpoint 或模型选择。

## 7. Fail-closed

以下任一情况必须拒绝动作或停止研究环境 session：

- 环境身份不匹配；
- 许可证无效；
- observation/action schema 版本不匹配；
- tick/frame 非单调；
- legal action 为空或损坏；
- RPC stale、重复或超时；
- action decode 失败；
- 真实商业客户端被错误识别为 sandbox；
- safety lock 不完整。

## 8. CI 安全门

每次 release 需要通过：

- 禁止 API/字符串扫描；
- secret scan；
- Actor export denylist；
- GameCore/PixelArena package identity test；
- real-client shadow isolation；
- no-input-device mount test；
- network allowlist；
- artifact privacy test。

任何安全门失败都不能通过降低测试、改名、删除扫描项或改成 warning 处理。
