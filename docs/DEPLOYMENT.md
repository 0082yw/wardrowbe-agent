# 本地部署与远程 GPU 运维

## 准备环境

已验收的环境是 Windows + WSL Ubuntu、Python 3.11、`frontend/.nvmrc` 指定的 Node.js，以及 Docker Desktop 中已配置的 PostgreSQL 和 Redis。新工作区按 [开发指南](../CONTRIBUTING.md#local-development) 准备依赖，使用专用数据库、Redis 和图片目录。部署还需准备有效的私有启动保护包、匹配的摘要、独立 Next.js 生产构建及既有认证配置。缺任一条件时停止。数据库凭据、SSH 凭据和签名链接保留在私有环境。

前端绑定 `127.0.0.1:3001`，通过运行时 `BACKEND_URL` 代理至 `127.0.0.1:8001`。现有数据库结构原样使用。应用更新前离线检查目标构建、源码哈希及启动保护包。开发和测试数据库不得指向真实运行库。

## Ollama、Qwen3 与 SSH 隧道

| 模式 | 模型与入口 | Windows / WSL 可用内存门槛 |
|---|---|---|
| `local_cpu` | `qwen3:0.6b-cpu`，本机回环 11434，核验 CPU 配置 | 2.5 / 1.25 GiB |
| `remote_gpu` | `qwen3:4b-instruct-2507-q4_K_M`，本机回环 11435 转发至远端回环 11434 | 1 / 0.5 GiB |

远端已验收版本为 Ollama 0.40.13。部署前由获准的运维方核对 Instruct 模型摘要和模板、禁云、单并发、2048 上下文及批准的物理 GPU 1 UUID。Ollama 保持仅监听远端回环。凭据和真实服务器地址从私有部署资料获取。需要建立隧道且获准时，可从 WSL 使用已有主机密钥记录执行：

```bash
ssh -N -T -M -S "$WARDROWBE_SSH_CONTROL" -p 22 \
  -L 127.0.0.1:11435:127.0.0.1:11434 \
  -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$WARDROWBE_SSH_TARGET"
```

以上变量由操作者从私有部署资料设置，控制套接字路径须为本用户可访问的独立路径。密码仅在终端交互输入，现有健康隧道直接复用。应用使用固定回环模型地址，拒绝代理、重定向、外部端点及 CPU 静默回退。切换只接受可信启动环境。

## 启动、检查与停止

在 WSL 的仓库根目录执行。以下是已具备私有保护包和构建的本机用法，`start`、`stop` 仅在获准操作时执行。

```bash
export WARDROBE_APP_MODE=full_app
export WARDROBE_AGENT_MODEL_MODE=remote_gpu
export WARDROBE_AGENT_REMOTE_MODEL=qwen3:4b-instruct-2507-q4_K_M
backend/.venv/bin/python -B backend/scripts/local_agent.py status
backend/.venv/bin/python -B backend/scripts/local_agent.py start
```

- 检查 `3001/login`、`3001/api/v1/health`、`8001/api/v1/health/ready` 和 `3001/api/v1/capabilities`。远程入口通过 `11435/api/version` 及目标模型元数据只读检查。
- 启动器复用健康的数据库和 Redis，检查 Windows 快照年龄不超过 10 秒、WSL memory PSI full avg10 不超过 1%、基线期间无新增 Swap 活动。内存下探、快照过期或隧道断开时按保护停止。
- 旋转 Worker 通过同一管理器的 `worker-start` 单独恢复。需单独获准并检查可信配置、资源和受限队列，现有实例健康时复用。旧 Docker 应用和 Worker 保持停止。
- `local_agent.py stop` 只处理已登记且身份匹配的自有进程，保留数据库、Redis 和借用服务。运维入口与私有配置说明见 [本机运行管理](../backend/scripts/LOCAL_AGENT.md)。

## 更新、回滚与证据

1. 保存旧构建、校验值、可信配置、启动环境及进程登记。备份中的认证环境保持私有权限。
2. 离线构建到独立目录，验证 BUILD_ID、standalone 入口、静态资源和权限契约。只在授权后定向加载发生变化的服务。不要用会同时停止前后端和 Worker 的 `--reload` 替代仅前端更新。
3. 检查健康、代理和能力清单。若失败，在确认进程归属及旧构建完整性后，用原配置和旧构建定向回滚。回滚材料由部署者在私有环境保管。
4. GPU 取证需要事先授权和同次请求内的新鲜采样。复用已有 SSH master，按 [采样契约](../backend/scripts/AGENT_MODEL_MODES.md) 使用有界只读采样。无新鲜证明时保持 `unknown`。
