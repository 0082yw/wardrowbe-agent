"""正式 Agent 的本机文本入口：固定 CPU 别名，复用已有适配器，不接管数据库。

构造时不读文件、不联网；只有实际文本请求才检查本地配置及模型元数据。
本模块只检查 WSL 资源；Windows 可用内存须由开发启动/验证流程另外检查。
禁云配置文件仅证明配置已设置，不能证明 Ollama 进程绝不会访问外网。
"""

import asyncio
import json
import re
from pathlib import Path
from threading import Lock

import httpx
from langchain_core.messages import AIMessage, BaseMessage

from app.services.agent_audit import current_audit
from app.services.ollama_agent_model import OllamaAgentModel, OllamaModelError

LOCAL_MODEL = "qwen3:0.6b-cpu"
LOCAL_BASE_URL = "http://127.0.0.1:11434"
MODEL_TIMEOUT_SECONDS = 60.0
_REQUEST_GATE = Lock()


def _fail(code: str) -> OllamaModelError:
    return OllamaModelError("本机模型运行条件未满足，本次未生成预览。", code=code)


def _check_cloud_disabled() -> None:
    """只读当前服务账号的既有配置，不创建或更新配置。"""
    try:
        config = json.loads((Path.home() / ".ollama/server.json").read_text())
        if not isinstance(config, dict) or config.get("disable_ollama_cloud") is not True:
            raise ValueError
    except (OSError, ValueError):
        raise _fail("cloud_not_disabled") from None


def _check_resources() -> None:
    """沿用独立示例的 1.25 GiB / full avg10 1% 门槛，不修改系统配置。"""
    try:
        fields = {
            key: int(value.split()[0])
            for key, value in (
                line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
            )
            if key == "MemAvailable"
        }
        if fields["MemAvailable"] < 1.25 * 1024 * 1024:
            raise _fail("wsl_memory_low")
        pressure = Path("/proc/pressure/memory")
        if pressure.is_file():
            match = re.search(r"^full avg10=([0-9.]+)", pressure.read_text(), re.MULTILINE)
            if match is None:
                raise ValueError
            if float(match.group(1)) > 1:
                raise _fail("wsl_memory_pressure")
    except (OSError, ValueError, KeyError):
        raise _fail("wsl_resources_unavailable") from None


def _validate_model_config(payload: dict) -> None:
    """别名必须显式禁用 GPU；不根据模型名称推测实际参数。"""
    text = payload.get("parameters")
    if not isinstance(text, str):
        raise _fail("cpu_config_unconfirmed")
    parameters = {}
    try:
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0] in {"num_gpu", "num_ctx"}:
                if len(parts) != 2 or parts[0] in parameters:
                    raise ValueError
                parameters[parts[0]] = int(parts[1])
    except ValueError:
        raise _fail("cpu_config_unconfirmed") from None
    if parameters != {"num_gpu": 0, "num_ctx": 2048}:
        raise _fail("cpu_config_unconfirmed")


def _validate_runtime(payload: dict, *, require_loaded: bool) -> None:
    """size_vram=0 核验 GPU 显存为零；不操作其他模型或服务。"""
    models = payload.get("models")
    if not isinstance(models, list):
        raise _fail("cpu_runtime_unconfirmed")
    if not models:
        if require_loaded:
            raise _fail("cpu_runtime_unconfirmed")
        return
    if len(models) != 1 or not isinstance(models[0], dict):
        raise _fail("local_model_busy")
    model = models[0]
    if model.get("name") != LOCAL_MODEL:
        raise _fail("local_model_busy")
    if (
        type(model.get("size_vram")) is not int or model["size_vram"] != 0
        or type(model.get("context_length")) is not int or model["context_length"] != 2048
    ):
        raise _fail("cpu_runtime_unconfirmed")


class LocalAgentTextModel:
    """只向固定本机模型生成文本，禁止模型发起工具调用或选择其他端点。

    管理请求仅 /api/show 与 /api/ps；不拉取模型、不启停服务、不自动重试。
    MockTransport 是唯一传输注入点，供独立测试离线使用。
    单进程同时只允许一个文本请求，不排队占用更多资源。
    """

    def __init__(self, *, transport: httpx.MockTransport | None = None) -> None:
        if transport is not None and not isinstance(transport, httpx.MockTransport):
            raise ValueError("只允许注入离线 MockTransport")
        self._transport = transport
        self._adapter = OllamaAgentModel(
            model=LOCAL_MODEL, timeout_seconds=55, max_tokens=256,
            disable_thinking=True, transport=transport,
        )

    async def _metadata(
        self, client: httpx.AsyncClient, path: str, *, post: bool = False,
    ) -> dict:
        try:
            response = (
                await client.post(path, json={"model": LOCAL_MODEL})
                if post else await client.get(path)
            )
            if response.status_code != 200:
                raise ValueError
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError
            return payload
        except (httpx.RequestError, ValueError):
            raise _fail("local_metadata_unavailable") from None

    async def check_ready(self) -> None:
        """只读检查本机配置及管理接口；不加载模型、不发送生成请求。"""
        try:
            async with asyncio.timeout(10.0):
                _check_cloud_disabled()
                _check_resources()
                async with httpx.AsyncClient(
                    base_url=LOCAL_BASE_URL, transport=self._transport,
                    trust_env=False, follow_redirects=False,
                    timeout=httpx.Timeout(5.0, connect=5.0),
                ) as client:
                    _validate_model_config(await self._metadata(client, "/api/show", post=True))
                    _validate_runtime(await self._metadata(client, "/api/ps"), require_loaded=False)
        except TimeoutError:
            raise _fail("local_metadata_unavailable") from None

    async def _generate(self, messages: list[BaseMessage]) -> AIMessage:
        _check_resources()
        task = asyncio.create_task(self._adapter(messages, tools=[]))
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                _check_resources()
            return await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def __call__(self, messages: list[BaseMessage], *, tools: list[dict]) -> AIMessage:
        if not isinstance(tools, list) or tools:
            raise ValueError("本地文本模型不得接收可执行工具")
        audit = current_audit.get()
        if audit is not None:
            audit.bind_execution("local_cpu", LOCAL_MODEL)
        if not _REQUEST_GATE.acquire(blocking=False):
            raise _fail("local_model_busy")
        try:
            # 总 deadline 包含前后元数据；HTTP 各自的 read timeout 不能代替总时限。
            async with asyncio.timeout(MODEL_TIMEOUT_SECONDS):
                _check_cloud_disabled()
                _check_resources()
                async with httpx.AsyncClient(
                    base_url=LOCAL_BASE_URL, transport=self._transport,
                    trust_env=False, follow_redirects=False,
                    timeout=httpx.Timeout(5.0, connect=5.0),
                ) as client:
                    _validate_model_config(await self._metadata(client, "/api/show", post=True))
                    _validate_runtime(await self._metadata(client, "/api/ps"), require_loaded=False)
                    successful = cancelled = False
                    audit = current_audit.get()
                    if audit is not None:
                        audit._cpu_config = True
                    try:
                        answer = await self._generate(messages)
                        successful = True
                        return answer
                    except asyncio.CancelledError:
                        cancelled = True
                        raise
                    finally:
                        # 取消/截止时 _generate 已收拢任务，不再发元数据请求延长时限。
                        # 内存保护失败同样立即停止，不为补充诊断继续请求。
                        if not cancelled:
                            _check_resources()
                            runtime = await self._metadata(client, "/api/ps")
                            try:
                                _validate_runtime(runtime, require_loaded=successful)
                            except OllamaModelError:
                                if audit is not None:
                                    audit.cpu_rejected(runtime)
                                raise
                            if audit is not None:
                                audit.cpu_verified(loaded=bool(runtime["models"]))
        except TimeoutError:
            raise OllamaModelError("本地 Ollama 请求超时。", code="timeout") from None
        finally:
            _REQUEST_GATE.release()
