"""WSL 用户级开发运行管理；不安装、不迁移、不下载，不管理其他项目。"""

import argparse
import ast
import base64
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path.home() / ".local/state/wardrowbe-agent/local-runtime"
CONFIG = Path.home() / ".config/wardrowbe-agent/local-runtime.json"
LEGACY = Path("/tmp/wardrowbe-auth-runtime-nr8welhb/state.json")
CONTAINERS = {"database": ("wardrobe-db", 5432), "redis": ("wardrobe-redis", 6379)}


class RuntimeErrorSafe(Exception):
    """只输出可公开的错误，不回显 subprocess 环境或连接字符串。"""


def require(condition, message):
    if not condition:
        raise RuntimeErrorSafe(message)


def private_json(path):
    require(not path.is_symlink() and path.is_file(), "私有配置文件不存在或为符号链接")
    info = path.stat()
    require(info.st_uid == os.getuid() and info.st_mode & 0o777 == 0o600, "私有配置必须由当前用户拥有且权限为0600")
    return json.loads(path.read_text())


def write_private(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(not path.is_symlink(), "拒绝覆盖符号链接")
    temporary = path.with_name(path.name + ".new")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_record(pid):
    p = Path(f"/proc/{pid}")
    try:
        parts = (p / "stat").read_text().rsplit(")", 1)[1].split()
        if parts[0] == "Z":
            return None
        uid = int(re.search(r"(?m)^Uid:\s+(\d+)", (p / "status").read_text())[1])
        return {"pid": int(pid), "ticks": parts[19], "exe": os.readlink(p / "exe"),
                "cwd": os.readlink(p / "cwd"), "uid": uid,
                "command_sha256": hashlib.sha256((p / "cmdline").read_bytes()).hexdigest()}
    except (OSError, ValueError):
        return None


def same_process(record):
    live = process_record(record["pid"]) if record else None
    # Next.js changes process.title after startup. PID reuse is rejected by start ticks/exe/cwd.
    return bool(live and live["uid"] == os.getuid()
                and all(live[k] == record[k] for k in ("pid", "ticks", "exe", "cwd", "uid")))


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def verify_ollama_process(binary):
    result = subprocess.run(["ss", "-lntp", "sport = :11434"], capture_output=True, text=True, check=True)
    owners = re.findall(r'pid=(\d+)', result.stdout)
    require(len(set(owners)) == 1, "无法确认本机 Ollama 监听进程，不继续")
    pid = int(owners[0])
    record = process_record(pid)
    require(record and record["uid"] == os.getuid() and record["exe"] == str(Path(binary).resolve()),
            "11434 不是预期的本账号 Ollama 进程")
    env = dict(v.split("=", 1) for v in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0") if "=" in v)
    require(env.get("OLLAMA_NO_CLOUD") == "true" and env.get("OLLAMA_HOST") == "127.0.0.1:11434",
            "运行中的 Ollama 未确认禁云或回环监听；不修改其全局配置")


def request(path, *, port, payload=None):
    import httpx
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=5) as client:
        url = f"http://127.0.0.1:{port}{path}"
        response = client.get(url) if payload is None else client.post(url, json=payload)
        require(response.status_code == 200, f"本机服务 {port} 的只读检查失败")
        return response


def wait_ready(port, path, record):
    for _ in range(40):
        require(same_process(record), f"{port} 进程提前退出；查看用户私有目录日志")
        try:
            request(path, port=port)
            return
        except Exception:
            time.sleep(.5)
    raise RuntimeErrorSafe(f"{port} 未在期限内就绪，不自动重启")


def docker_dependencies():
    """仅启动已存在、映射明确的两个容器；不操作卷、密码或其他容器。"""
    probe = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True)
    require(probe.returncode == 0, "Docker Desktop 不可用，请先手动打开；不会安装或创建替代服务")
    for name, port in CONTAINERS.values():
        result = subprocess.run(["docker", "inspect", "--format",
            '{{json .State}}|{{json .HostConfig.PortBindings}}', name], capture_output=True, text=True)
        require(result.returncode == 0, f"既有容器 {name} 不存在；停止，不创建新容器")
        state, ports = map(json.loads, result.stdout.strip().split("|", 1))
        require(any(p.get("HostPort") == str(port) for p in ports.get(f"{port}/tcp", [])),
                f"{name} 端口映射不符，停止")
        if not state["Running"]:
            subprocess.run(["docker", "start", name], check=True, capture_output=True)
        for _ in range(30):
            check = subprocess.run(["docker", "inspect", "--format", "{{.State.Health.Status}}", name],
                                   capture_output=True, text=True, check=True)
            if check.stdout.strip() == "healthy":
                break
            time.sleep(1)
        else:
            raise RuntimeErrorSafe(f"{name} 健康检查失败；不会重建容器")
    import redis
    client = redis.Redis(host="127.0.0.1", port=6379, db=15, socket_timeout=3)
    try:
        require(client.ping(), "Redis DB15 不可连接")
    finally:
        client.close()


def extract_wrappers(entry):
    """一次性迁移已核验的保护源码，不重新实现 SQL、认证和网络限制。"""
    outer = next(n.value for n in ast.walk(ast.parse(entry))
                 if isinstance(n, ast.Constant) and isinstance(n.value, str) and "def _attest_loaded" in n.value)
    auth = next(n.value for n in ast.walk(ast.parse(outer))
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and "def install_and_run" in n.value)
    return outer, auth


def prepare(build):
    """初始化当前电脑的私有保护包，凭据继续从原可信文件读取，绝不保存进程环境。"""
    legacy = private_json(LEGACY)
    require(Path(legacy["root"]).resolve() == ROOT, "历史运行配置属于其他项目")
    original = Path(legacy["original_bootstrap"])
    entry = Path(legacy["backend_entry_path"])
    require(digest(original) == legacy["original_bootstrap_sha256"] and digest(entry) == legacy["backend_entry_sha256"],
            "既有保护源码校验失败")
    window = private_json(Path(legacy["upload_runtime_directory"]) / "window.json")
    require(window["state"] == "closed_success", "一次性上传权限尚未关闭，停止初始化")
    fe = dict(v.split("=", 1) for v in Path(f"/proc/{legacy['new_frontend_pid']}/environ").read_bytes().decode().split("\0") if "=" in v)
    from dotenv import dotenv_values
    trusted = dotenv_values(ROOT / ".env")
    require(fe.get("NEXTAUTH_SECRET") and fe["NEXTAUTH_SECRET"] == trusted.get("NEXTAUTH_SECRET"),
            "不能从原可信配置复原认证密钥，停止；不保存进程环境")
    require(fe.get("DEV_MODE") == "true" and fe.get("BACKEND_URL") == "http://127.0.0.1:8001", "前端认证/代理配置不符")
    PRIVATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    outer, auth = extract_wrappers(entry.read_text())
    tree = ast.parse(original.read_text())
    heartbeat = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HEARTBEAT" for t in node.targets):
            heartbeat = node.value.args[0].value
            node.value = ast.parse("Path(os.environ['WARDROWBE_MEMORY_FILE'])", mode="eval").body
    require(heartbeat is not None, "未找到原有内存保护；不跳过保护")
    bootstrap = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    bootstrap_path = PRIVATE / "backend.py"
    write_private(bootstrap_path, bootstrap)
    auth = auth.replace(str(LEGACY), str(PRIVATE / "auth-state.json"))
    write_private(PRIVATE / "auth-state.json", {
        "policy": legacy["policy"], "original_bootstrap": str(bootstrap_path),
        "original_bootstrap_sha256": digest(bootstrap_path),
    })
    old_auth = extract_wrappers(entry.read_text())[1]
    class ReplaceAuth(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(auth), node) if node.value == old_auth else node
    write_private(PRIVATE / "backend-entry.py", ast.unparse(ast.fix_missing_locations(ReplaceAuth().visit(ast.parse(outer)))) + "\n")
    # The one-use upload wrapper is deliberately not carried forward; baseline rejects POST /items.
    write_private(PRIVATE / "frontend-guard.mjs", legacy["updated_frontend_guard"])
    node = os.readlink(f"/proc/{legacy['new_frontend_pid']}/exe")
    config = {"root": str(ROOT), "node": node, "build": str(build), "legacy_heartbeat": heartbeat,
              "heartbeat": str(Path(heartbeat).parent.parent / "wardrowbe-agent-managed-memory/memory.json"),
              "ollama": str(Path.home() / ".local/opt/ollama/v0.40.12/bin/ollama"),
              "models": str(Path.home() / ".ollama/models"),
              "frontend_internal_url": fe.get("NEXTAUTH_URL_INTERNAL", "http://127.0.0.1:3001"),
              "guard_hashes": {p.name: digest(p) for p in PRIVATE.iterdir() if p.name in
                               ("backend.py", "backend-entry.py", "auth-state.json", "frontend-guard.mjs")}}
    write_private(CONFIG, config)
    return config


def build_path(value):
    path = Path(value).resolve()
    require(path.is_relative_to(ROOT / "frontend") and (path / "BUILD_ID").is_file()
            and (path / "standalone/server.js").is_file(), "需要本项目已有的完整生产构建；不会自动构建或安装")
    return path


def fresh_memory(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if 0 <= time.time() - value["unix_seconds"] <= 10 else None
    except (OSError, ValueError, KeyError):
        return None


def resources(heartbeat, *, execution_mode="local_cpu"):
    require(execution_mode in ("local_cpu", "remote_gpu"), "资源保护模式无效")
    windows_min, wsl_min = (2.5, 1.25) if execution_mode == "local_cpu" else (1, 0.5)
    w = fresh_memory(heartbeat)
    require(w is not None, "Windows 内存采样不可用；不放宽保护")
    m = int(re.search(r"(?m)^MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) * 1024
    psi = float(re.search(r"(?m)^full avg10=([\d.]+)", Path("/proc/pressure/memory").read_text())[1])
    swaps = {k: int(v) for k, v in (line.split() for line in Path("/proc/vmstat").read_text().splitlines()) if k in ("pswpin", "pswpout")}
    require(w["available_bytes"] >= windows_min * 2**30 and m >= wsl_min * 2**30 and psi <= 1,
            f"现有资源门槛未满足：Windows {w['available_bytes'] / 2**30:.3f} GiB（需{windows_min:g}），"
            f"WSL {m / 2**30:.3f} GiB（需{wsl_min:g}），PSI {psi:.2f}%（最多1%）；停止，不重试")
    return {"windows_gib": round(w["available_bytes"] / 2**30, 3), "wsl_gib": round(m / 2**30, 3), "psi": psi, **swaps}


class Manager:
    def __init__(self, config):
        self.config = config
        self.guard_dir = Path(config.get("guard_directory", PRIVATE))
        self.state_path = PRIVATE / "processes.json"
        self.state = private_json(self.state_path) if self.state_path.exists() else {"processes": {}}
        self.heartbeat = self.state.get("heartbeat") or config["legacy_heartbeat"]

    def save(self):
        write_private(self.state_path, self.state)

    def spawn(self, name, command, cwd, env):
        log = PRIVATE / (name + ".log")
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "ab", buffering=0) as stream:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        record = process_record(proc.pid)
        require(record is not None, f"{name} 启动失败")
        self.state["processes"][name] = {"record": record, "owned": True}
        self.save()
        print(json.dumps({"started": name, "pid": proc.pid}), flush=True)
        return record

    def clear_worker_ready(self, record):
        path = PRIVATE / "rotation-worker-ready.json"
        if path.exists():
            require(process_record(record["pid"]) is None, "Worker尚未退出，不清理就绪记录")
            info = private_json(path)
            require(info.get("record") == record, "Worker就绪记录归属不符，不清理")
            path.unlink()

    def stop_one(self, name):
        entry = self.state["processes"].get(name)
        if not entry or not entry["owned"]:
            return
        if not same_process(entry["record"]):
            require(process_record(entry["record"]["pid"]) is None, f"{name} PID身份不符；拒绝终止")
            if name == "rotation_worker":
                self.clear_worker_ready(entry["record"])
                self.state["processes"].pop(name, None)
                self.save()
            return
        if name == "rotation_worker":
            require(process_record(entry["record"]["pid"])["command_sha256"] == entry["record"]["command_sha256"],
                    "Worker命令行身份不符，拒绝终止")
        if name == "monitor":
            Path(self.state["monitor_stop"]).touch(exist_ok=False)
        else:
            os.kill(entry["record"]["pid"], signal.SIGTERM)
        for _ in range(40):
            if not same_process(entry["record"]):
                if name == "rotation_worker":
                    self.clear_worker_ready(entry["record"])
                self.state["processes"].pop(name, None)
                self.save()
                return
            time.sleep(.5)
        raise RuntimeErrorSafe(f"{name} 未正常退出；不强杀、不继续重启")

    def start_worker(self):
        require(application_policy(self.config).mode == "full_app", "Worker要求可信full_app")
        require(os.environ.get("WARDROBE_AGENT_MODEL_MODE") == "remote_gpu", "Worker模式必须为remote_gpu")
        backend = self.state["processes"].get("backend")
        require(backend and same_process(backend["record"])
                and self.state.get("application_mode") == "full_app", "先完成full_app后端部署，再启动Worker")
        env = dict(v.split("=", 1) for v in Path(f"/proc/{backend['record']['pid']}/environ").read_bytes().decode().split("\0") if "=" in v)
        require(env.get("WARDROBE_APP_MODE") == "full_app"
                and env.get("WARDROBE_AGENT_MODEL_MODE") == "remote_gpu",
                "实际后端模式与Worker不一致")
        source = ROOT / "backend/scripts/rotation_worker.py"
        existing = self.state["processes"].get("rotation_worker")
        if existing and same_process(existing["record"]):
            require(existing.get("entry_sha256") == digest(source)
                    and existing.get("config_sha256") == digest(CONFIG), "现有Worker版本不同，拒绝重复启动")
            print("受保护旋转Worker已运行，不创建重复实例")
            return
        require(not existing or process_record(existing["record"]["pid"]) is None,
                "Worker PID身份异常，拒绝接管")
        self.current_resources()
        child_env = {**os.environ, "WARDROWBE_MEMORY_FILE": self.heartbeat}
        result = subprocess.run([str(ROOT / "backend/.venv/bin/python"), "-B", str(source), "--check-only"],
                                cwd=ROOT, env=child_env, capture_output=True, timeout=30)
        require(result.returncode == 0, "Worker凭据/SQL/队列/资源预检失败，不启动")
        ready = PRIVATE / "rotation-worker-ready.json"
        require(not ready.exists(), "存在Worker就绪文件，先只读核对归属；不清理或覆盖")
        try:
            record = self.spawn("rotation_worker", [str(ROOT / "backend/.venv/bin/python"), "-B", str(source)],
                                ROOT, child_env)
            for _ in range(30):
                require(same_process(record), "Worker提前退出，不重试")
                self.current_resources()
                if ready.exists():
                    info = private_json(ready)
                    require(info.get("record") == record and info.get("config_sha256") == digest(CONFIG)
                            and info.get("entry_sha256") == digest(source), "Worker就绪身份不符")
                    self.state["processes"]["rotation_worker"].update(
                        entry_sha256=digest(source), config_sha256=digest(CONFIG))
                    self.save()
                    return
                time.sleep(.5)
            raise RuntimeErrorSafe("Worker未就绪，不重试")
        except Exception:
            self.stop_one("rotation_worker")
            raise

    def stop_worker(self):
        self.stop_one("rotation_worker")

    def ensure_monitor(self):
        for candidate in (self.heartbeat, self.config["legacy_heartbeat"], self.config["heartbeat"]):
            if fresh_memory(candidate):
                self.heartbeat = candidate
                self.state["heartbeat"] = candidate
                self.save()
                return
        existing = self.state["processes"].get("monitor")
        require(not existing or not same_process(existing["record"]), "内存监视器已运行但无新采样；停止，不重复启动")
        target = Path(self.config["heartbeat"])
        target.parent.mkdir(parents=True, exist_ok=True)
        stop = target.parent / f"stop-{time.time_ns()}"
        powershell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        require(Path(powershell).is_file(), "现有 Windows PowerShell 不可用，不安装替代工具")
        def windows_path(path):
            return subprocess.run(["wslpath", "-w", str(path)], check=True, capture_output=True, text=True).stdout.strip().replace("'", "''")
        script = (ROOT / "backend/scripts/windows_memory.ps1").read_text()
        command = "& {" + script + "} -OutputPath '" + windows_path(target) + "' -StopPath '" + windows_path(stop) + "'"
        encoded = base64.b64encode(command.encode("utf-16le")).decode()
        self.state["monitor_stop"] = str(stop)
        self.spawn("monitor", [powershell, "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-EncodedCommand", encoded], ROOT, os.environ.copy())
        self.heartbeat = str(target)
        for _ in range(20):
            if fresh_memory(target):
                self.state["heartbeat"] = str(target)
                self.save()
                return
            time.sleep(.5)
        raise RuntimeErrorSafe("Windows 内存监视器未就绪；不继续启动模型或应用")

    def baseline(self):
        first = None
        for index in range(13):
            value = self.current_resources()
            if first is None:
                first = value
            require(all(value[k] == first[k] for k in ("pswpin", "pswpout")), "Swap 出现新活动；停止，不重试")
            print(json.dumps({"baseline_sample": index + 1, **value}), flush=True)
            if index < 12:
                time.sleep(5)

    def adopt(self):
        legacy = private_json(LEGACY)
        require(Path(legacy["root"]).resolve() == ROOT, "拒绝接管其他项目")
        for name, port in (("backend", 8001), ("frontend", 3001)):
            pid = legacy["new_" + name + "_pid"]
            record = process_record(pid)
            require(record and record["ticks"] == str(legacy["new_" + name + "_start_ticks"])
                    and record["uid"] == os.getuid()
                    and Path(record["cwd"]).is_relative_to(ROOT), "既有项目进程身份校验失败")
            require(port_open(port), "既有项目进程端口未就绪")
            if name == "backend":
                require(hashlib.sha256(Path(legacy["backend_entry_path"]).read_bytes()).hexdigest() == legacy["backend_entry_sha256"], "原启动入口已变更")
                args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                require(str(legacy["backend_entry_path"]).encode() in args, "后端入口不符")
            else:
                env = dict(x.split("=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0") if "=" in x)
                require(env.get("NODE_ENV") == "production" and env.get("PORT") == "3001"
                        and env.get("BACKEND_URL") == "http://127.0.0.1:8001", "前端配置不符")
            self.state["processes"][name] = {"record": record, "owned": True}
        self.save()
        self.state["build_id"] = legacy.get("lesson49_build_id")
        self.save()

    def ensure_ollama(self):
        cloud = json.loads((Path.home() / ".ollama/server.json").read_text())
        require(cloud.get("disable_ollama_cloud") is True, "Ollama 禁云配置未确认，不修改或继续")
        if not port_open(11434):
            existing = self.state["processes"].get("ollama")
            require(not existing or not same_process(existing["record"]), "已有 Ollama 进程尚未就绪，不重复启动")
            binary = Path(self.config["ollama"])
            require(binary.is_file(), "已有 Ollama 不存在；不安装或下载")
            env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_NO_CLOUD": "true",
                   "OLLAMA_MODELS": self.config["models"],
                   "CUDA_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1", "GGML_VK_VISIBLE_DEVICES": "-1"}
            record = self.spawn("ollama", [str(binary), "serve"], ROOT, env)
            wait_ready(11434, "/api/ps", record)
        sys.path.insert(0, str(ROOT / "backend"))
        verify_ollama_process(self.config["ollama"])
        from app.services.local_agent_model import _validate_model_config, _validate_runtime
        _validate_model_config(request("/api/show", port=11434, payload={"model": "qwen3:0.6b-cpu"}).json())
        _validate_runtime(request("/api/ps", port=11434).json(), require_loaded=False)

    def current_resources(self):
        mode = os.environ.get("WARDROBE_AGENT_MODEL_MODE", "local_cpu")
        return resources(self.heartbeat) if mode == "local_cpu" else resources(
            self.heartbeat, execution_mode=mode,
        )

    def configured_model(self):
        sys.path.insert(0, str(ROOT / "backend"))
        from app.services.agent_model_config import AgentModelConfig

        return AgentModelConfig.from_env().model_name

    def ensure_model(self):
        sys.path.insert(0, str(ROOT / "backend"))
        from app.services.agent_model_config import AgentModelConfig

        config = AgentModelConfig.from_env()
        if config.execution_mode == "local_cpu":
            return self.ensure_ollama()
        import asyncio
        from app.services.remote_agent_model import RemoteAgentTextModel

        model = RemoteAgentTextModel(model=config.model_name, memory_file=self.heartbeat)
        try:
            asyncio.run(model.check_ready())
        except Exception:
            raise RuntimeErrorSafe("SSH 隧道入口或远程模型服务未就绪；不登录、不建隧道、不回退 CPU") from None

    def preflight_backend(self):
        for mode in ("--auth-check-only", "--lesson48-no-bind"):
            result = subprocess.run([str(ROOT / "backend/.venv/bin/python"), "-B",
                str(self.guard_dir / "backend-entry.py"), mode], cwd=ROOT,
                env={**os.environ, "WARDROWBE_MEMORY_FILE": self.heartbeat}, capture_output=True, timeout=30)
            require(result.returncode == 0, "后端凭据/认证/SQL/路由保护自检失败；保留当前服务")
        print("认证/SQL保护31项及路由启动自检通过；生成请求0次", flush=True)

    def start(self, *, reload=False):
        access_policy = application_policy(self.config)
        mode = os.environ.get("WARDROBE_AGENT_MODEL_MODE", "local_cpu")
        require(mode in ("local_cpu", "remote_gpu"), "模型运行模式无效，不启动服务")
        for name, expected in self.config["guard_hashes"].items():
            require(digest(self.guard_dir / name) == expected, "私有启动保护包哈希不符；拒绝启动")
        docker_dependencies()
        for name, port in (("backend", 8001), ("frontend", 3001)):
            entry = self.state["processes"].get(name)
            if port_open(port):
                require(entry and same_process(entry["record"]), f"{port} 被未登记进程使用；不接管或终止")
        if not reload and all(self.state["processes"].get(n) and same_process(self.state["processes"][n]["record"])
                              for n in ("backend", "frontend")):
            require(self.state.get("application_mode", "read_only_demo") == access_policy.mode,
                    "已运行应用权限模式不同；需要显式reload")
            requested_model = self.configured_model()
            require(self.state.get("execution_mode", "local_cpu") == mode
                    and self.state.get("model_name", "qwen3:0.6b-cpu") == requested_model,
                    "已运行服务的推理模式或模型不同；必须显式 start --reload，不会静默切换")
            require(self.state.get("build_id") == (Path(self.config["build"]) / "BUILD_ID").read_text().strip(),
                    "已有服务属于其他构建，请显式使用 start --reload，不会静默替换")
            self.ensure_monitor()
            self.ensure_model()
            print("本项目服务已经运行，不创建重复实例")
            return
        self.ensure_monitor()
        self.baseline()
        self.ensure_model()
        self.preflight_backend()
        if reload:
            self.stop_one("rotation_worker")
            self.stop_one("frontend")
            self.stop_one("backend")
        for name, port in (("backend", 8001), ("frontend", 3001)):
            if port_open(port):
                continue
            self.current_resources()
            if name == "backend":
                env = {**os.environ, "WARDROWBE_MEMORY_FILE": self.heartbeat}
                record = self.spawn(name, [str(ROOT / "backend/.venv/bin/python"), "-B", str(self.guard_dir / "backend-entry.py")], ROOT, env)
                wait_ready(port, "/api/v1/health/ready", record)
                self.state["execution_mode"] = mode
                self.state["model_name"] = self.configured_model()
                self.save()
            else:
                from dotenv import dotenv_values
                trusted = dotenv_values(ROOT / ".env")
                require(bool(trusted.get("NEXTAUTH_SECRET")), "原认证密钥不存在，停止")
                env = {k: v for k, v in os.environ.items() if not k.startswith(("OIDC_", "FORWARD_AUTH_", "NEXTAUTH_", "AUTH_"))}
                env.update(NODE_ENV="production", HOSTNAME="127.0.0.1", PORT="3001", DEV_MODE="true",
                    BACKEND_URL="http://127.0.0.1:8001", NEXTAUTH_URL="http://localhost:3001",
                    NEXTAUTH_URL_INTERNAL=self.config["frontend_internal_url"], NEXTAUTH_SECRET=trusted["NEXTAUTH_SECRET"],
                    NEXT_TELEMETRY_DISABLED="1", NODE_OPTIONS="--import=" + (self.guard_dir / "frontend-guard.mjs").as_uri())
                record = self.spawn(name, [self.config["node"], "server.js"], Path(self.config["build"]) / "standalone", env)
                wait_ready(port, "/login", record)
        request("/api/v1/health", port=3001)
        self.state["application_mode"] = access_policy.mode
        self.state["build_id"] = (Path(self.config["build"]) / "BUILD_ID").read_text().strip()
        self.save()
        print("本机 Agent 已就绪；模型按请求加载，本次生成请求0次")

    def stop(self):
        for name in ("rotation_worker", "frontend", "backend", "ollama", "monitor"):
            self.stop_one(name)
        print("仅停止已登记且身份匹配的项目进程；数据库、Redis和借用服务保持不变")

    def status(self):
        result = {"processes": {n: {"pid": p["record"]["pid"], "owned": p["owned"],
                  "running": same_process(p["record"])} for n, p in self.state["processes"].items()},
                  "ports": {str(p): port_open(p) for p in (3001, 8001, 11434, 5432, 6379)},
                  "build_id": self.state.get("build_id"), "logs": str(PRIVATE)}
        result["generation_budget_per_backend_process"] = 3
        try:
            result["resources"] = self.current_resources()
        except RuntimeErrorSafe as exc:
            result["resources"] = str(exc)
        try:
            containers = subprocess.run(["docker", "ps", "--format", "{{.Names}}: {{.Status}}"],
                                        capture_output=True, text=True, timeout=5)
            result["containers"] = [line for line in containers.stdout.splitlines()
                                    if line.split(":", 1)[0] in {n for n, _ in CONTAINERS.values()}]
        except (OSError, subprocess.SubprocessError):
            result["containers"] = "unavailable"
        if result["ports"]["11434"]:
            try:
                result["loaded_models"] = [{"model": m.get("name"), "gpu_vram_bytes": m.get("size_vram")}
                                          for m in request("/api/ps", port=11434).json().get("models", [])]
            except Exception:
                result["loaded_models"] = "unavailable"
        print(json.dumps(result, ensure_ascii=False, indent=2))


def application_policy(config):
    sys.path.insert(0, str(ROOT / "backend"))
    from app.access_policy import validate_launch_config

    try:
        return validate_launch_config(config, os.environ.get("WARDROBE_APP_MODE", "read_only_demo"), owner=os.getuid())
    except (ValueError, KeyError, OSError):
        raise RuntimeErrorSafe("应用权限策略未准备完整；拒绝切换或扩大写入") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status", "worker-start", "worker-stop"))
    parser.add_argument("--build-dir", help="已有生产构建目录；不自动构建")
    parser.add_argument("--adopt-running", action="store_true", help="仅首次接管已核验的既有项目进程")
    parser.add_argument("--reload", action="store_true", help="仅定向更新已登记项目进程")
    args = parser.parse_args()
    try:
        require(sys.platform == "linux", "请在 WSL Ubuntu 中运行")
        PRIVATE.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (PRIVATE / "manager.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if CONFIG.exists():
                config = private_json(CONFIG)
                require(config["root"] == str(ROOT), "运行配置不属于当前项目")
                if args.build_dir:
                    require(args.command == "start", "仅 start 可以选择新构建")
                    config["build"] = str(build_path(args.build_dir))
                    write_private(CONFIG, config)
            else:
                require(args.command == "start" and args.adopt_running and args.build_dir,
                        "首次请使用 start --adopt-running --build-dir <已验证构建目录>")
                config = prepare(build_path(args.build_dir))
            manager = Manager(config)
            if args.adopt_running:
                require(args.command == "start", "只有 start 允许首次接管")
                manager.adopt()
            if args.command == "start":
                manager.start(reload=args.reload)
            elif args.command == "worker-start":
                manager.start_worker()
            elif args.command == "worker-stop":
                manager.stop_worker()
            else:
                getattr(manager, args.command)()
        return 0
    except Exception:
        # Never print raw exception arguments: they can contain SQL/credentials or command bodies.
        print("本机运行管理未完成：" + (str(sys.exc_info()[1]) if isinstance(sys.exc_info()[1], RuntimeErrorSafe)
                                   else "检查失败（" + type(sys.exc_info()[1]).__name__ + "）；未扩大权限，请检查私有日志和已有服务"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
