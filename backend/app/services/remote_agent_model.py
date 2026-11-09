"""仅访问人工建立的回环 SSH 隧道；不登录、不拉模型、不回退 CPU。

HTTP 可达只能说明隧道入口/模型服务可达，不能证明服务物理位置或 GPU 执行。
仅在可信本地采样证明与本次真实请求匹配时确认 GPU；缺少证明仍为 unknown。
"""

import asyncio
import time
from threading import Lock

import httpx
from langchain_core.messages import AIMessage, BaseMessage

from app.services.agent_audit import current_audit
from app.services.agent_resources import ResourceSnapshot, check_remote_resources
from app.services.ollama_agent_model import (
    OllamaAgentModel, OllamaModelError,
)

REMOTE_BASE_URL = "http://127.0.0.1:11435"
_REQUEST_GATE = Lock()


class RemoteAgentTextModel:
    def __init__(self, *, model: str, transport: httpx.MockTransport | None = None,
                 memory_file: str | None = None, gpu_evidence_file: str | None = None) -> None:
        self._adapter = OllamaAgentModel(
            model=model, execution_mode="remote_gpu", max_tokens=256,
            timeout_seconds=55, disable_thinking=True, transport=transport,
        )
        self.model_name = model
        self._transport = transport
        self._memory_file = memory_file
        self.tunnel_status = "unknown"
        self.gpu_verified = "unknown"
        self._gpu_evidence_file = gpu_evidence_file
        self._model_digest = None

    def _resources(self, baseline: ResourceSnapshot | None = None) -> ResourceSnapshot:
        return check_remote_resources(baseline, memory_file=self._memory_file)

    async def _check_service(self) -> None:
        self._adapter._non_thinking_profile = "unverified"
        # /api/tags 只读、不加载模型；不拿本机禁云文件冒充服务器配置证明。
        try:
            async with httpx.AsyncClient(
                transport=self._transport, trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(5.0, connect=5.0),
            ) as client:
                response = await client.get(REMOTE_BASE_URL + "/api/tags")
            if response.status_code != 200:
                raise OllamaModelError("SSH 隧道入口或远程模型服务不可用。", code="remote_service_unavailable")
            payload = response.json()
            models = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(models, list):
                raise ValueError
            matches = [m for m in models if isinstance(m, dict)
                       and m.get("name") == self.model_name]
            if len(matches) != 1:
                raise OllamaModelError("远程服务没有指定的已安装模型。", code="remote_model_missing")
            model = matches[0]
            if (model.get("remote_host") or model.get("remote_model")
                    or type(model.get("size")) is not int or model["size"] <= 0
                    or not isinstance(model.get("details"), dict)
                    or model["details"].get("format") != "gguf"):
                raise ValueError
            self.tunnel_status = "connected"
            # /api/tags 不包含 thinking 控制值；/api/show 只读、不加载模型。
            # 私有运行保护若尚未授权此管理接口，应明确失败，不能绕过保护继续生成。
            async with httpx.AsyncClient(
                transport=self._transport, trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(5.0, connect=5.0),
            ) as client:
                metadata = await client.post(REMOTE_BASE_URL + "/api/show", json={"model": self.model_name})
            if metadata.status_code != 200:
                raise OllamaModelError(
                    "远程模型思考控制元数据不可用，本次未发送生成请求。",
                    code="thinking_metadata_unavailable",
                )
            self._adapter.configure_non_thinking(metadata.json(), model_digest=model.get("digest"))
            self._model_digest = model.get("digest")
        except PermissionError:
            raise OllamaModelError(
                "运行保护尚未允许远程模型的只读元数据检查，本次未发送生成请求。",
                code="model_metadata_blocked",
            ) from None
        except httpx.TimeoutException:
            self.tunnel_status = "disconnected"
            raise OllamaModelError("SSH 隧道入口检查超时。", code="timeout") from None
        except httpx.RequestError:
            self.tunnel_status = "disconnected"
            raise OllamaModelError("SSH 隧道入口无法连接。", code="tunnel_unavailable") from None
        except (ValueError, TypeError):
            self.tunnel_status = "disconnected"
            raise OllamaModelError("远程模型元数据无效或不是本地权重模型。", code="remote_metadata_invalid") from None
        except OllamaModelError as exc:
            if exc.code not in {
                "thinking_metadata_unavailable", "thinking_disable_unsupported", "remote_metadata_invalid",
            } or self.tunnel_status != "connected":
                self.tunnel_status = "disconnected"
            raise
        finally:
            audit = current_audit.get()
            if audit is not None:
                audit.tunnel_status = self.tunnel_status

    async def check_ready(self) -> None:
        self._resources()
        try:
            async with asyncio.timeout(10):
                await self._check_service()
        except TimeoutError:
            self.tunnel_status = "disconnected"
            raise OllamaModelError("SSH 隧道入口检查超时。", code="timeout") from None

    async def __call__(self, messages: list[BaseMessage], *, tools: list[dict]) -> AIMessage:
        if not isinstance(tools, list) or tools:
            raise ValueError("远程文本模型不得接收可执行业务工具")
        audit = current_audit.get()
        if audit is not None:
            audit.bind_execution("remote_gpu", self.model_name)
        if not _REQUEST_GATE.acquire(blocking=False):
            raise OllamaModelError("远程模型请求正在处理中。", code="remote_model_busy")
        task = None
        try:
            async with asyncio.timeout(60):
                baseline = self._resources()
                await self._check_service()
                self._resources(baseline)
                self.gpu_verified = "unknown"
                started_at = time.time()
                task = asyncio.create_task(self._adapter(messages, tools=[]))
                while not task.done():
                    await asyncio.wait({task}, timeout=.1)
                    self._resources(baseline)
                answer = await task
                self._resources(baseline)
                finished_at = time.time()
                if audit is not None and self._transport is None:
                    from app.services.gpu_evidence import read_gpu_evidence

                    proof, evidence_status = read_gpu_evidence(
                        self._gpu_evidence_file, request_id=audit.request_id,
                        model=self.model_name, digest=self._model_digest,
                        started_at=started_at, finished_at=finished_at, now=time.time(),
                    )
                    audit.gpu_evidence_status = evidence_status
                    if proof is not None:
                        audit.record_gpu_evidence(proof)
                        self.gpu_verified = audit.gpu_verified
                return answer
        except TimeoutError:
            if audit is not None:
                audit.execution_error("timeout")
            raise OllamaModelError("远程模型请求超时，本次未生成预览。", code="timeout") from None
        except OllamaModelError as exc:
            if audit is not None:
                audit.execution_error(exc.code)
            raise
        finally:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            _REQUEST_GATE.release()
