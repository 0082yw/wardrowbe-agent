# Agent 双模式配置

只从可信启动进程读取，不读取用户请求，不修改 `.env`。默认 `local_cpu`；显式选择 `remote_gpu` 后，可信默认模型为 `qwen3:4b-instruct-2507-q4_K_M`，可由启动环境显式覆盖；空值仍拒绝。

```bash
export WARDROBE_AGENT_MODEL_MODE=local_cpu
```

远程模式必须另外指定已在服务器安装的本地权重模型名；下列命令只示范配置，不会建立隧道：

```bash
export WARDROBE_AGENT_MODEL_MODE=remote_gpu
export WARDROBE_AGENT_REMOTE_MODEL=qwen3:4b-instruct-2507-q4_K_M
```

- CPU 地址固定 `http://127.0.0.1:11434/v1/chat/completions`，保留 `qwen3:0.6b-cpu`、禁云及 CPU 核验。
- 远程地址固定 `http://127.0.0.1:11435/v1/chat/completions`。SSH 认证、隧道、服务器禁云和设备选择由另行授权的部署管理；应用不会登录、下载或启动模型服务。
- 两种模式都禁用环境代理、HTTP 重定向、端点回退和自动重试。用户不能选择模式、端点或模型。
- CPU Windows/WSL 门槛仍为 2.5/1.25 GiB。远程模式基本运行门槛为 1/0.5 GiB；仍拒绝 PSI full avg10 超过 1%、新 Swap 活动、缺失或超过 10 秒的 Windows 采样。
- Windows 采样路径通过既有 `WARDROWBE_MEMORY_FILE` 注入；不伪造可用内存。
- `tunnel_status=connected` 只说明回环隧道入口返回了目标已安装模型的元数据，不能证明 SSH 身份、服务器位置或 GPU 执行。需要运维核对 SSH 主机密钥、隧道所有者及远端监听。
- 全局状态检查不提供远端物理 GPU 证明；`gpu_verified` 保持 `unknown`。请求级证明见第58节说明，HTTP 200 本身仍不能升级为 GPU 已验证。
- 数据库、用户身份、Session、衣物归属、组合、图片签名均留在本地。远端只生成文本。

## 受保护启动器与部署前提

第53节经授权迁移现有私有启动包：备份原文件及摘要，网络和 HTTP 操作按模式分别只允许 11434 或 11435；数据库5432、Redis6379、绑定8001、认证和 SQL 保护保持不变。
模式策略位于 `app/services/agent_runtime_policy.py`，其 SHA256 固定在私有启动包内；修改策略必须重新核验完整性。
`local_agent.py start` 默认保持 CPU。远程模式只检查人工建立的隧道及目标模型，缺失时拒绝，不启动本地 Ollama，也不创建 SSH 连接。
重复 start 不切换已有进程的模式或模型；变更必须显式 reload。本节不执行模式切换或本地服务重启。
只有服务器模型、禁云/设备配置及人工隧道都准备好后，才能另行授权实际接入；离线测试不等于真实 GPU 验证。

## 状态和审计

原预览 API 不变。状态保留旧 CPU 字段，增加 execution_mode、tunnel_status、gpu_verified；远程模式使用 remote_ollama。
审计在原请求 ID 下记录可信模式/模型、请求次数、时延、token、工具次数、候选/预览数量及 Schema 结果。
缺失指标记录 unavailable，不保存提示词、ToolMessage、图片签名 URL 或认证材料。

## 思考控制兼容性

生成接口保持 OpenAI Chat Completions，模型只由可信启动配置决定：
- Hybrid：要求 GGUF 身份、thinking 元数据及模板控制分支一致，发送 `reasoning_effort="none"`。
- Instruct-only：核验 Qwen3-Instruct-2507 的 GGUF 身份、completion 能力和无思考模板；不发送 think、reasoning_effort 或 /no_think。
- Thinking-only 与证据不足的模型：拒绝生成，不回退 CPU、不自动选择其他权重。

元数据缺少 thinking 字段时，仅已核验的 Qwen3-Instruct-2507 身份及模板证据可支持 Instruct-only；其他未知情况拒绝。
服务端只读预检使用 `/api/tags` 与 `POST /api/show`，不会加载模型。
私有保护只新增 127.0.0.1:11435 的 POST /api/show，JSON 必须恰为可信模型名；其他路径、端口、代理与重定向规则保留。
正式生成仍限制单并发、max_tokens=256、固定回环入口；CPU 仍要求原关闭思考参数。
Instruct-only 省略思考开关，但仍拒绝意外思考字段、思考标记、空正文及 finish_reason=length。

保留的 qwen3:4b 权重是 Thinking-2507，不能冒充已关闭思考；新目标为 qwen3:4b-instruct-2507-q4_K_M。
旧权重不删除，下载及 GPU 验证只在另行明确授权时进行。当前默认运行模式不因配置目标变更而切换。
正式接口仍为受认证约束的只读预览，不自动保存；模型不得决定可信衣物 ID 或图片。

## 第58节：请求级 GPU 证明（可选、默认未配置）

全局 GET /agent/status 只说明配置和服务就绪，gpu_verified 仍不能为 true。
预览响应可带 execution：本次 request_id、执行模式、模型、文案回退字段及 GPU 证明。
前端只在本次响应具有完整且一致的证明时显示“本次请求已验证”，不会沿用旧预览或其他用户的结果。

可信启动者可提供 WARDROBE_AGENT_GPU_EVIDENCE_FILE；本节不设置该变量、不修改 .env，也不启动采样生产者。
既有独立采样流程必须把真实记录原子发布到本机当前用户拥有、权限0600的普通文件；禁止符号链接、FIFO、超过64KiB的文件。
文件格式为 observations 数组（1至32条），每条必须由实际采样提供：
model、model_digest、process_id、process_started_at、gpu_uuid、observed_at、layers_loaded、layers_total、model_size_bytes、gpu_size_bytes。
不能手工填写“成功”数据，也不能由终端用户或模型提交证明。模型名/摘要须匹配本次只读能力预检；物理 UUID 须为已批准 GPU 1。
记录必须位于本次真实 HTTP 推理时间窗口内且不超过10秒；进程须在观测前启动，全部模型层和模型运行大小须驻留 GPU。
后端在真实 stop 响应后将有效记录绑定自己的请求 ID；MockTransport、HTTP 200本身、旧记录和不完整证明均不能升级 GPU 状态。
evidence_id 是记录完整性摘要，并非第三方数字签名；信任边界是现有可信启动者及其0600采样文件。
缺少生产者时保持 unknown/unavailable；存在不匹配记录时标记 rejected，不伪造成功。当前代码不会为此登录SSH或增加网络白名单。

第57节旧记录不足以自动充当新请求证明：旧请求 ID 和时间窗口不同，且旧采样未逐条保存进程启动时间。
下一次真实验收前，需要另行确认采样生产者按此契约发布记录；仅部署这些代码不能保证页面立即显示 GPU 已验证。


## 第59节：只读验收采样生产者

sample_remote_gpu.py 仅使用已有认证 SSH master，BatchMode 和失败的 ProxyCommand
确保连接失效时不会自动登录或建立隧道。它不发送生成请求、不加载模型、不改变设备或服务。
采样只读取目标 Ollama 的 api/ps、当前账号目标权重进程及新日志；按实际端口、
权重路径、进程启动时间匹配完整层加载，再与物理 GPU UUID 和全部驻留大小交叉核验。
本机采集端将严格验证的最近32条记录原子写入当前用户0600文件。
采集进程有120秒上限；缺少记录不会伪造证明，也不会阻断原有模型请求。
每次验收使用新证据文件与日志，不复用旧请求记录。仅在独立授权的真实验收前运行。

日常运行和验收边界见 [部署指南](../../docs/DEPLOYMENT.md)；历史记录不代表当前进程或新请求已取得GPU证明。
