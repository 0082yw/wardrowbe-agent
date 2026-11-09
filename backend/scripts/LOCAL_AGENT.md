# WSL 本地 Agent 运行管理

在项目根目录执行，使用已有 `backend/.venv`、Node.js 和生产构建。不会安装、构建、下载模型或执行迁移。

历史首次接管示例（当前电脑已完成接管，不要重复执行）：

```bash
backend/.venv/bin/python -B backend/scripts/local_agent.py start \
  --adopt-running --reload --build-dir frontend/.cache/agent59-build
```

日常三个入口：

```bash
backend/.venv/bin/python -B backend/scripts/local_agent.py start
backend/.venv/bin/python -B backend/scripts/local_agent.py status
backend/.venv/bin/python -B backend/scripts/local_agent.py stop
```

当前已验证的本机运行约定为 `remote_gpu`，新终端执行上述命令前明确设置：

```bash
export WARDROBE_AGENT_MODEL_MODE=remote_gpu
export WARDROBE_AGENT_REMOTE_MODEL=qwen3:4b-instruct-2507-q4_K_M
```

代码仍保留缺省 `local_cpu` 兼容路径；这里不会改变已运行进程的模式，也不允许失败回退。`remote_gpu` 只检查人工准备的回环11435入口，不登录服务器或创建隧道。配置与部署前提见 [AGENT_MODEL_MODES.md](AGENT_MODEL_MODES.md)。

新版本已有构建时，显式指定更新；重复 `start` 不产生重复进程：

```bash
backend/.venv/bin/python -B backend/scripts/local_agent.py start \
  --reload --build-dir frontend/.cache/agent59-build
```

## 范围与前置条件

- 先手动打开 Docker Desktop。仅检查/启动已经存在且端口映射正确的 `wardrobe-db`、`wardrobe-redis`；缺失时停止，不创建容器或卷。
- 保留专用开发库 `wardrowbe_agent_dev`、原项目外 0600 凭据文件和既有图片目录。不会创建账号或衣物。
- 前端只监听 `127.0.0.1:3001`，代理到 `127.0.0.1:8001`；复用根目录 `.env` 中原认证密钥，不写入源码或运行清单。
- 仅CPU模式会按需启动本机Ollama，要求既有 `qwen3:0.6b-cpu`、禁云及 `num_gpu=0`；远程模式只核对现有隧道与服务，不启动本机模型。缺少条件时停止，不拉取或创建模型。
- 启动不运行模型；模型由正式预览请求按需加载。已有 Ollama 属于借用服务，`stop` 不会关闭它。模型可能在 Ollama 的既有保留时长内继续驻留，本工具不擅自改动该配置。
- 更新前采集60秒资源基线：CPU模式Windows/WSL至少2.5/1.25 GiB；远程模式至少1/0.5 GiB。两者均要求PSI full avg10不超过1%、无新增Swap、Windows快照年龄0至10秒。失败后停止，不循环等待或降低门槛。
- 优先复用已有 Windows 内存采样；缺少时使用仓库内只读 PowerShell 采样脚本，后台隐藏运行。自己的采样器通过停止标记正常退出，不终止借用的采样器。

## 进程与保护

首次初始化从当前已经验证的私有启动快照迁移认证、SQL、回环网络和内存保护，保存在 `~/.local/state/wardrowbe-agent/local-runtime/`（0700，文件0600）。配置位于 `~/.config/wardrowbe-agent/local-runtime.json`，只含路径、非敏感运行设置及校验摘要，不保存整个进程环境。

该迁移只在当前电脑第一次执行；以后不依赖 `/tmp` 的原快照。现有 Test User 只允许正常登录同步更新登录时间，其他业务写入仍被阻止。一次性衣物新增例外不会迁移。

每个进程记录 PID、启动时间、可执行文件、工作目录和命令摘要。只有身份匹配、明确归属于本工具的进程才能收到 SIGTERM；PID 被其他进程复用时停止并报告，不强杀。未登记端口冲突不会自动接管。

停止顺序为前端、后端、工具自己启动的 Ollama、工具自己的采样器。PostgreSQL、Redis、Docker Desktop、借用的 Ollama/采样器、Codex、VS Code 及其他项目始终保留。不会启动旧 3000/8000 或 Worker。

日志位于上述用户私有运行目录。错误提示不会输出密码、完整连接字符串或进程环境。

## 当前边界

- 这是本机专用的受保护开发启动器，不是公网部署方案。
- 继续保留原验证启动保护的每个后端进程最多 3 次真实生成预算；耗尽时明确报错，需要显式更新进程。本节不擅自扩大这个预算。
- 缺少已有配置/凭据、构建或模型时会报告阻塞，不安装或猜测替代配置。
- 目前穿搭意图使用受控的简体中文句式；例如 `帮我搭配三套通勤穿搭，包含白色衬衫和蓝色牛仔裤`。正式场景需要实际正式程度记录，不根据名称推断。

完整架构、部署前提和运维边界见 [部署指南](../../docs/DEPLOYMENT.md)。
