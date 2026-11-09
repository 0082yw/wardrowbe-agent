"""仅供可信调用者注入的本机 Ollama 适配器；不注册路由、不读取模型凭据。

采用 /v1/chat/completions，复用 LangChain 的消息转换器和现有 AgentService。
模型能力、量化规格及本地权重尚需实测；使用前必须在 Ollama 服务端禁用云端功能。
固定回环地址只能约束本进程的请求，不能证明本机服务自身绝不会访问外网。
"""

import json
import math
import re
from copy import deepcopy
from typing import Any

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)

LOCAL_CHAT_URL = "http://127.0.0.1:11434/v1/chat/completions"
REMOTE_CHAT_URL = "http://127.0.0.1:11435/v1/chat/completions"
_FUNCTION_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_-]{0,63}")
_MODEL_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*(?::[a-zA-Z0-9][a-zA-Z0-9_.-]*)?")


class OllamaModelError(RuntimeError):
    """稳定的脱敏错误；不携带请求内容、服务端响应或原始连接异常。"""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def validate_model_name(model: str) -> None:
    if (not isinstance(model, str) or len(model) > 128
            or not _MODEL_NAME.fullmatch(model) or "cloud" in model.lower()):
        raise ValueError("必须明确指定本地权重模型名称，禁止 URL、命名空间和云端模型")


def validate_non_thinking_metadata(
    data: Any, *, model_name: str | None = None, model_digest: str | None = None,
) -> str:
    """多证据识别；thinking.values 只辅助判断，不能单独认定 Thinking-only。"""
    from app.services.ollama_model_identity import (
        INSTRUCT_MODEL, forced_thinking_prefix, verify_instruct_identity,
    )

    if not isinstance(data, dict) or data.get("error") or data.get("remote_host") or data.get("remote_model"):
        raise OllamaModelError("远程模型元数据无效。", code="remote_metadata_invalid")
    details = data.get("details", {})
    info = data.get("model_info", {})
    capabilities = data.get("capabilities")
    template = data.get("template")
    known_qwen = (
        isinstance(details, dict) and details.get("format") == "gguf"
        and details.get("family") == "qwen3" and isinstance(info, dict)
        and info.get("general.architecture") == "qwen3"
        and isinstance(capabilities, list) and all(isinstance(v, str) for v in capabilities)
        and "completion" in capabilities and isinstance(template, str) and bool(template.strip())
    )
    thinking = data.get("thinking")
    values = None
    if thinking is not None:
        values = thinking.get("values") if isinstance(thinking, dict) else None
        if (
            not isinstance(values, list) or not values
            or any(type(v) not in (bool, str) or v == "" for v in values)
            or len({(type(v), v) for v in values}) != len(values)
            or type(thinking.get("default")) not in (bool, str)
            or not any(type(v) is type(thinking["default"]) and v == thinking["default"] for v in values)
        ):
            raise OllamaModelError("远程模型思考控制元数据无效。", code="remote_metadata_invalid")
    if known_qwen and (info.get("general.finetune") == "Instruct" or model_name == INSTRUCT_MODEL):
        try:
            verify_instruct_identity(data, model_name=model_name, model_digest=model_digest)
        except ValueError:
            raise OllamaModelError("远程模型身份或官方模板证据不一致。", code="remote_metadata_invalid") from None
        # 0.40.13 可从历史消息处理推导出 thinking 能力；可信的 Instruct 身份优先。
        return "instruct_only"
    if (
        known_qwen and info.get("general.finetune") == "Thinking"
        and info.get("general.version") == "2507" and "thinking" in capabilities
        and forced_thinking_prefix(template)
        and (values is None or not any(v is False for v in values))
    ):
        raise OllamaModelError(
            "当前远程模型的身份和模板确认只能思考，不支持关闭思考；本次未发送生成请求。",
            code="thinking_disable_unsupported",
        )
    if (
        known_qwen and "thinking" in capabilities and values is not None
        and any(v is False for v in values) and any(v is True for v in values)
        and not (info.get("general.version") == "2507"
                 and info.get("general.finetune") in ("Thinking", "Instruct"))
        and re.search(r"\.Think\b", template)
    ):
        return "hybrid"
    raise OllamaModelError(
        "远程模型能力与模板证据不足或不一致，本次未发送生成请求。",
        code="thinking_metadata_unavailable" if thinking is None else "remote_metadata_invalid",
    )


def _tool_names(tools: list[dict]) -> set[str]:
    names = set()
    if not isinstance(tools, list):
        raise ValueError("工具定义必须是列表")
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if (
            not isinstance(function, dict)
            or tool.get("type") != "function"
            or not isinstance(function.get("name"), str)
            or not _FUNCTION_NAME.fullmatch(function["name"])
            or function["name"] in names
            or not isinstance(function.get("parameters"), dict)
            or function["parameters"].get("type") != "object"
        ):
            raise ValueError("工具定义格式无效")
        names.add(function["name"])
    return names


def _messages(messages: list[BaseMessage], names: set[str]) -> list[dict]:
    """只接受文本消息及完整的工具请求/结果关联；序列化交给现有 LangChain。"""
    if not isinstance(messages, list) or not messages:
        raise ValueError("需要非空消息列表")
    pending: dict[str, str] = {}
    seen = set()
    for message in messages:
        if not isinstance(message, (SystemMessage, HumanMessage, AIMessage, ToolMessage)):
            raise ValueError("仅支持系统、用户、模型和工具消息")
        if not isinstance(message.content, str):
            raise ValueError("本适配器仅支持文本")
        if isinstance(message, ToolMessage):
            if (
                message.tool_call_id not in pending
                or (message.name is not None and message.name != pending[message.tool_call_id])
            ):
                raise ValueError("工具结果与调用 ID 不匹配")
            pending.pop(message.tool_call_id)
            continue
        if pending:
            raise ValueError("工具结果尚未完整回传")
        if isinstance(message, AIMessage):
            if message.invalid_tool_calls:
                raise ValueError("历史消息包含无效工具调用")
            for call in message.tool_calls:
                call_id = call["id"]
                if (
                    not isinstance(call_id, str)
                    or not call_id.strip()
                    or call_id in seen
                    or call["name"] not in names
                ):
                    raise ValueError("历史工具调用的名称或 ID 无效")
                try:
                    json.dumps(call["args"], allow_nan=False)
                except (TypeError, ValueError):
                    raise ValueError("历史工具参数必须是有效 JSON 对象") from None
                seen.add(call_id)
                pending[call_id] = call["name"]
    if pending:
        raise ValueError("工具结果尚未完整回传")
    return convert_to_openai_messages(messages)


def _reject_constant(value: str) -> None:
    raise ValueError("工具参数必须使用有限 JSON 数值")


def _response(
    data: Any, names: set[str], previous_ids: set[str], *, disable_thinking: bool = False,
) -> AIMessage:
    """延续 HTTP 示例的严格结束标记检查，不猜测缺失的调用 ID 或参数。"""
    try:
        if not isinstance(data, dict) or data.get("error"):
            raise ValueError
        choices = data["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        message = choice["message"]
        if message["role"] != "assistant":
            raise ValueError
        if choice.get("finish_reason") == "length":
            raise OllamaModelError(
                "模型输出达到 token 上限，截断结果已拒绝，本次未生成有效回答。",
                code="response_truncated",
            )
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise ValueError
        calls = []
        ids = set(previous_ids)
        for call in raw_calls:
            call_id = call["id"]
            function = call["function"]
            if (
                call["type"] != "function"
                or not isinstance(call_id, str)
                or not call_id.strip()
                or call_id in ids
                or function["name"] not in names
                or not isinstance(function["arguments"], str)
            ):
                raise ValueError
            arguments = json.loads(function["arguments"], parse_constant=_reject_constant)
            if not isinstance(arguments, dict):
                raise ValueError
            ids.add(call_id)
            calls.append(
                {"id": call_id, "name": function["name"], "args": arguments, "type": "tool_call"}
            )
        finish = choice["finish_reason"]
        if calls:
            if finish != "tool_calls":
                raise ValueError
        elif finish != "stop" or not content.strip():
            raise ValueError
        if disable_thinking:
            for key in ("reasoning", "reasoning_content", "thinking"):
                value = message.get(key)
                if value is not None and not isinstance(value, str):
                    raise ValueError
                if isinstance(value, str) and value.strip():
                    raise OllamaModelError(
                        "模型仍返回思考内容，无法确认关闭思考，本次响应已拒绝。",
                        code="thinking_not_disabled",
                    )
            if "<think>" in content or "</think>" in content:
                raise OllamaModelError(
                    "模型正文包含思考标记，本次响应已拒绝。", code="thinking_not_disabled",
                )
        return AIMessage(
            content=content, tool_calls=calls, response_metadata={"finish_reason": finish}
        )
    except (KeyError, TypeError, ValueError):
        raise OllamaModelError("本地模型响应或工具调用格式无效。", code="invalid_response") from None


class OllamaAgentModel:
    """符合 AgentModel 的异步可调用对象；构造时不联网，单次调用只发一个本机请求。

    只接收消息和可信工具定义，不接收用户、Session、数据库或可变端点。
    transport 仅支持离线 MockTransport 测试；默认传输禁用代理及 HTTP 重定向。
    不自动重试、不拉取模型、不切换端点，也不复用 AIService 的回退列表。
    disable_thinking 仅请求关闭思考，不宣称已生效；远程生产入口另行核验模型元数据。
    不切换原生 API，不叠加 think 或 /no_think；默认保留模型原有的思考行为。
    """

    api_type = "openai_chat_completions"

    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: float = 60.0,
        max_tokens: int = 1024,
        disable_thinking: bool = False,
        transport: httpx.MockTransport | None = None,
        execution_mode: str = "local_cpu",
    ) -> None:
        validate_model_name(model)
        if execution_mode not in ("local_cpu", "remote_gpu"):
            raise ValueError("推理模式无效")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 300
        ):
            raise ValueError("超时必须是 0 至 300 秒之间的有限正数")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens 必须是 1 至 4096 的整数")
        if type(disable_thinking) is not bool:
            raise ValueError("disable_thinking 必须是布尔值")
        if transport is not None and not isinstance(transport, httpx.MockTransport):
            raise ValueError("只允许注入离线 MockTransport")
        self.model = model
        self.timeout = httpx.Timeout(float(timeout_seconds), connect=min(timeout_seconds, 5.0))
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self._transport = transport
        self.execution_mode = execution_mode
        self._non_thinking_profile = "unverified" if execution_mode == "remote_gpu" and disable_thinking else "hybrid"

    def configure_non_thinking(self, metadata: Any, *, model_digest: str | None = None) -> str:
        """仅供可信远程入口在读取 /api/show 后使用，不接受用户请求中的配置。"""
        if self.execution_mode != "remote_gpu":
            raise ValueError("本机 CPU 模式保留原有关闭思考配置")
        self._non_thinking_profile = validate_non_thinking_metadata(
            metadata, model_name=self.model, model_digest=model_digest,
        )
        return self._non_thinking_profile

    def prepare_request(
        self, messages: list[BaseMessage], *, tools: list[dict],
    ) -> tuple[dict, set[str], set[str]]:
        """复用同一消息/工具校验构造请求；不联网、不选择其他端点。"""
        if self.execution_mode not in ("local_cpu", "remote_gpu"):
            raise ValueError("推理模式无效，不执行请求或回退")
        if self.disable_thinking and self._non_thinking_profile == "unverified":
            raise OllamaModelError("远程模型能力尚未核验，本次未发送请求。", code="thinking_metadata_unavailable")
        names = _tool_names(tools)
        body = {
            "model": self.model,
            "messages": _messages(messages, names),
            "tools": deepcopy(tools),
            "stream": False,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking and self._non_thinking_profile == "hybrid":
            body["reasoning_effort"] = "none"
        try:
            json.dumps(body, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("消息或工具定义必须是有效 JSON 数据") from None
        previous_ids = {
            call["id"]
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        }
        return body, names, previous_ids

    async def __call__(self, messages: list[BaseMessage], *, tools: list[dict]) -> AIMessage:
        body, names, previous_ids = self.prepare_request(messages, tools=tools)
        from app.services.agent_audit import current_audit

        audit = current_audit.get()
        attempt = None
        error_code = "unavailable"
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                trust_env=False,
                follow_redirects=False,
                timeout=self.timeout,
            ) as client:
                if audit is not None:
                    attempt = audit.model_started(
                        self.model, transport="mock" if self._transport is not None else "local_http",
                        execution_mode=self.execution_mode,
                    )
                endpoint = REMOTE_CHAT_URL if self.execution_mode == "remote_gpu" else LOCAL_CHAT_URL
                response = await client.post(endpoint, json=body)
                if audit is not None and attempt is not None:
                    audit.model_response(attempt, response.status_code)
            if response.status_code != 200:
                raise OllamaModelError("本地 Ollama 返回非成功状态。", code="http_error")
            try:
                data = response.json()
            except ValueError:
                raise OllamaModelError("本地 Ollama 未返回有效 JSON。", code="invalid_response") from None
            if audit is not None and attempt is not None:
                audit.model_response(attempt, response.status_code, data)
            return _response(data, names, previous_ids, disable_thinking=self.disable_thinking)
        except httpx.TimeoutException:
            error_code = "timeout"
            raise OllamaModelError("本地 Ollama 请求超时。", code="timeout") from None
        except httpx.RequestError:
            error_code = "connection_failed"
            raise OllamaModelError("无法连接本地 Ollama。", code="connection_failed") from None
        except OllamaModelError as exc:
            error_code = exc.code
            raise
        finally:
            if audit is not None and attempt is not None:
                audit.model_finished(attempt, error=error_code)
