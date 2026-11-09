"""第 47 节：复用 AgentService 的真实本地模型集成示例，不连接数据库。

在 backend 目录显式运行：
.venv/bin/python -B -m examples.langgraph_ollama_agent_demo --run-local
测试默认使用 MockTransport；查询由后端控制，显式运行时最多 1 次本地文本模型请求。
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import httpx
from langchain_core.messages import AIMessage, ToolMessage

from app.services.ollama_agent_model import OllamaAgentModel
from app.services.wardrobe_query_agent import (
    DEFAULT_REASONING as DEFAULT_REASONING,
    DEFAULT_STYLE_NOTES as DEFAULT_STYLE_NOTES,
    TEXT_MODEL_INSTRUCTIONS as TEXT_MODEL_INSTRUCTIONS,
    WardrobeQueryAgentModel,
    default_query_answer as default_query_answer,
    grounded_query_text as grounded_query_text,
    parse_model_text as parse_model_text,
    query_call as query_call,
    read_candidates as read_candidates,
    route_item_type as route_item_type,
    unwrap_json_fence as unwrap_json_fence,
)

MODEL = "qwen3:0.6b-cpu"
BASE_URL = "http://127.0.0.1:11434"
QUERY = "请查询我的衬衫"
GREETING = "你好"

QUERY_MAX_TOKENS = 128
FINAL_MAX_TOKENS = 256


class OllamaTextModel:
    """沿用现有本地适配器，只执行工具完成后的文本请求，不增加预算或重试。"""

    def __init__(self, *, transport=None):
        self._model = OllamaAgentModel(
            model=MODEL, timeout_seconds=55, max_tokens=FINAL_MAX_TOKENS,
            disable_thinking=True, transport=transport,
        )

    async def __call__(self, messages, *, tools):
        if tools:
            raise ValueError("文本模型不能获得工具执行权限")
        return await self._model(messages, tools=[])


class DeterministicWardrobeModel(WardrobeQueryAgentModel):
    """示例仅注入观察器和替身；查询、文案及候选校验复用正式控制器。"""

    def __init__(self, text_model, trace):
        super().__init__(text_model, trace=trace, safety_check=text_model._check_safety, max_previews=1)

    def _query_call(self, item_type):
        return query_call(item_type)


@dataclass
class RequestBudget:
    maximum: int = 3
    used: int = 0


@dataclass
class Trace:
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    seconds: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    toolnode_calls: int = 0
    safety_failed: bool = False
    graph_inputs: list = field(default_factory=list)
    graph_outputs: list = field(default_factory=list)
    text_fallbacks: list = field(default_factory=list)


class DemoSafetyError(RuntimeError):
    """可报告的稳定保护代码，不包含底层异常、配置或凭据。"""

    def __init__(self, code):
        super().__init__("示例运行条件未满足")
        self.code = code


class ObservedModel:
    """只观察既有适配器；不生成回答、不修补模型输出、不实现工具循环。"""

    def __init__(
        self, model, trace, budget, *, after_request=None, safety_check=None, timeout=60,
    ):
        self.model = model
        self.trace = trace
        self.budget = budget
        self.after_request = after_request
        self.safety_check = safety_check
        self.timeout = timeout

    def _check_safety(self):
        if self.safety_check is not None:
            try:
                self.safety_check()
            except Exception:
                self.trace.safety_failed = True
                raise

    async def __call__(self, messages, *, tools):
        if self.budget.used >= self.budget.maximum:
            raise RuntimeError("示例真实请求预算已耗尽")
        self._check_safety()
        self.budget.used += 1
        self.trace.inputs.append([m.model_copy(deep=True) for m in messages])
        start = time.monotonic()
        task = asyncio.create_task(self.model(messages, tools=tools))
        try:
            while not task.done():
                if time.monotonic() - start >= self.timeout:
                    raise TimeoutError("模型请求超过示例时限")
                await asyncio.wait({task}, timeout=min(0.1, self.timeout))
                self._check_safety()
            answer = await task
            self.trace.outputs.append(answer.model_copy(deep=True))
            return answer
        except Exception as exc:
            self.trace.errors.append(getattr(exc, "code", type(exc).__name__))
            raise
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self.trace.seconds.append(round(time.monotonic() - start, 3))
            if self.after_request is not None:
                try:
                    await self.after_request()
                except Exception:
                    self.trace.safety_failed = True
                    raise


async def run_scenario(
    model, message, *, budget=None, after_request=None, safety_check=None,
    timeout=60, max_model_calls=2, max_tool_calls=1, fixture_items=None, user_id=None,
):
    """复用原 Service 图和工具工厂；仅查询服务与 Session 为隔离替身。"""
    from langgraph.prebuilt import ToolNode
    from sqlalchemy.ext.asyncio import AsyncSession

    from app import agent_tools
    from app.models.item import ItemStatus
    from app.schemas.agent import AgentPreviewResponse
    from app.services import agent_service

    trace = Trace()
    user = SimpleNamespace(id=user_id or UUID("10000000-0000-4000-8000-000000000099"))
    db = Mock(spec=AsyncSession)
    items = [
        SimpleNamespace(
            id=UUID(f"10000000-0000-4000-8000-00000000000{i}"),
            name=name, type="shirt", status=ItemStatus.ready, needs_wash=False,
        )
        for i, name in enumerate(["测试白色衬衫", "测试蓝色衬衫"], 1)
    ] if fixture_items is None else fixture_items

    async def lookup(*, user_id, filters, page, page_size):
        assert user_id == user.id
        matches = [item for item in items if item.type == filters.type]
        offset = (page - 1) * page_size
        return matches[offset:offset + page_size], len(matches)

    service = Mock(spec=agent_tools.ItemService)
    service.get_list = AsyncMock(side_effect=lookup)
    observed = ObservedModel(
        model, trace, budget or RequestBudget(), after_request=after_request,
        safety_check=safety_check, timeout=timeout,
    )
    controller = DeterministicWardrobeModel(observed, trace)

    class RecordingToolNode(ToolNode):
        async def ainvoke(self, input, config=None, **kwargs):
            trace.toolnode_calls += 1
            result = await super().ainvoke(input, config=config, **kwargs)
            trace.tool_results.extend(result["messages"])
            return result

    with (
        patch.object(agent_tools, "ItemService", return_value=service) as constructor,
        patch.object(agent_service, "ToolNode", RecordingToolNode),
    ):
        preview = await agent_service.AgentService(
            model=controller, max_model_calls=max_model_calls, max_tool_calls=max_tool_calls,
        ).generate_preview(message, current_user=user, db=db)
    assert db.mock_calls == [], "不得操作真实 Session 或接管事务"
    response = AgentPreviewResponse.from_preview(preview, items=controller.display_items)
    return SimpleNamespace(
        preview=preview, response=response, trace=trace, service=service, db=db,
        user=user, items=items, controller=controller, constructor=constructor,
    )


def verify_roundtrip(run):
    """工具请求来自后端；一轮真实文本模型不控制候选 ID、展示信息或状态。"""
    trace = run.trace
    assert not trace.errors and not trace.safety_failed
    assert run.preview.status == run.response.status == "success"
    assert len(trace.inputs) == len(trace.outputs) == 1
    assert len(trace.graph_inputs) == len(trace.graph_outputs) == 2
    assert trace.toolnode_calls == run.service.get_list.await_count == 1
    request, final = trace.graph_outputs
    assert len(request.tool_calls) == 1 and not final.tool_calls
    call = request.tool_calls[0]
    assert call == query_call("shirt")
    results = [m for m in trace.graph_inputs[1] if isinstance(m, ToolMessage)]
    assert len(results) == len(trace.tool_results) == 1
    result = results[0]
    assert result.status == "success" and result.name == call["name"]
    assert result.tool_call_id == trace.tool_results[0].tool_call_id == call["id"]
    payload = json.loads(result.content)
    assert all(
        str(item_id) in {item["id"] for item in payload["items"]}
        for outfit in run.preview.outfits for item_id in outfit.item_ids
    )
    assert run.service.get_list.await_args.kwargs["user_id"] == run.user.id
    run.constructor.assert_called_once_with(run.db)
    assert run.response.outfits[0].items[0].id == run.preview.outfits[0].item_ids[0]


def check_memory():
    values = {
        key: int(value.split()[0])
        for key, value in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        if key in {"MemAvailable"}
    }
    if values["MemAvailable"] < 1.25 * 1024 * 1024:
        raise DemoSafetyError("wsl_memory_low")
    pressure = Path("/proc/pressure/memory")
    if pressure.is_file():
        match = re.search(r"^full avg10=([0-9.]+)", pressure.read_text(), re.MULTILINE)
        if match and float(match.group(1)) > 1:
            raise DemoSafetyError("wsl_memory_pressure")


def _cli_env():
    return {
        "HOME": str(Path.home()), "USER": os.environ.get("USER", "aiagent"),
        "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
        "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_NO_CLOUD": "1",
    }


async def _ollama_command(*args):
    process = await asyncio.create_subprocess_exec(
        str(Path.home() / ".local/bin/ollama"), *args, env=_cli_env(),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
    if process.returncode:
        raise RuntimeError("本地 Ollama 管理命令失败")
    return stdout.decode(errors="replace")


def _install_network_guard():
    def audit(event, args):
        if event == "socket.connect":
            address = args[1]
            if (
                not isinstance(address, tuple) or address[0] != "127.0.0.1"
                or address[1] != 11434
            ):
                raise PermissionError("示例仅允许连接本机 Ollama")
        if event == "socket.getaddrinfo" and args[0] not in ("127.0.0.1", b"127.0.0.1"):
            raise PermissionError("示例禁止外部 DNS")
    sys.addaudithook(audit)


async def run_local(*, query_only=False):
    """显式运行；后端先查询，最多一次文本请求；任何失败均不重试。"""
    config = Path.home() / ".ollama/server.json"
    assert json.loads(config.read_text()).get("disable_ollama_cloud") is True
    check_memory()
    budget = RequestBudget(1)
    async with httpx.AsyncClient(
        base_url=BASE_URL, trust_env=False, follow_redirects=False, timeout=5,
    ) as client:
        response = await client.post("/api/show", json={"model": MODEL})
        response.raise_for_status()
        parameters = {}
        for line in response.json().get("parameters", "").splitlines():
            parts = line.split()
            if len(parts) > 1 and parts[0] in {"num_gpu", "num_ctx", "num_predict"}:
                parameters[parts[0]] = int(parts[1])
        assert parameters == {"num_gpu": 0, "num_ctx": 2048, "num_predict": 128}
        response = await client.get("/api/ps")
        response.raise_for_status()
        assert not response.json()["models"], "已有运行模型，先停止而不干扰其他任务"

        async def cpu_check():
            listing = await _ollama_command("ps")
            line = next((x for x in listing.splitlines() if x.split() and x.split()[0] == MODEL), "")
            assert re.search(r"\b100%\s+CPU\b", line) and "GPU" not in line
            runtime = await client.get("/api/ps")
            runtime.raise_for_status()
            loaded = next(x for x in runtime.json()["models"] if x["name"] == MODEL)
            assert loaded["size_vram"] == 0 and loaded["context_length"] == 2048
            print("CPU_CHECK", line.strip(), flush=True)

        model = OllamaTextModel()
        try:
            query = await run_scenario(
                model, QUERY, budget=budget, after_request=cpu_check, safety_check=check_memory,
            )
            print("QUERY_RESULT", query.response.model_dump_json(), flush=True)
            print("QUERY_TRACE", json.dumps({
                "model_calls": len(query.trace.inputs), "graph_steps": len(query.trace.graph_inputs),
                "seconds": query.trace.seconds,
                "toolnode_calls": query.trace.toolnode_calls, "errors": query.trace.errors,
            }), flush=True)
            verify_roundtrip(query)
            greeting_seconds = []
            if not query_only:
                greeting = await run_scenario(
                    model, GREETING, budget=budget, after_request=cpu_check,
                    safety_check=check_memory,
                )
                assert greeting.preview.status == "needs_clarification"
                assert not greeting.trace.errors and not greeting.trace.safety_failed
                assert len(greeting.trace.inputs) == 0 and greeting.trace.toolnode_calls == 0
                assert greeting.service.get_list.await_count == 0
                greeting_seconds = greeting.trace.seconds
                print("GREETING_RESULT", greeting.preview.model_dump_json(), flush=True)
            print("REAL_INTEGRATION_PASSED", json.dumps({
                "requests": budget.used, "tool_executions": query.trace.toolnode_calls,
                "tool_call_id": query.trace.tool_results[0].tool_call_id,
                "tool_request_source": "trusted_backend",
                "greeting_verified": not query_only, "greeting_seconds": greeting_seconds,
                "database_operations": 0,
            }), flush=True)
        finally:
            await _ollama_command("stop", MODEL)
            runtime = await client.get("/api/ps")
            runtime.raise_for_status()
            assert not runtime.json()["models"]
            print("MODEL_UNLOADED", MODEL, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-local", action="store_true", help="显式允许最多1次本地文本模型请求")
    parser.add_argument("--query-only", action="store_true", help="只验证后端查询和文本生成流程，最多1次请求")
    args = parser.parse_args()
    if not args.run_local:
        parser.error("必须显式传入 --run-local；默认不调用模型")
    isolated = "postgresql+asyncpg://isolated:test-only@127.0.0.1:1/wardrowbe_demo"
    os.environ.update({
        "DEBUG": "true", "SECRET_KEY": "isolated-example-only-signing-key",
        "DATABASE_URL": isolated, "TEST_DATABASE_URL": isolated,
        "REDIS_URL": "redis://127.0.0.1:1/15", "AI_INTERNAL_ENABLED": "false",
        "AI_TEXT_ENABLED": "false", "AI_VISION_ENABLED": "false",
    })
    for key in list(os.environ):
        if key.startswith(("OIDC_", "FORWARD_AUTH_")):
            os.environ.pop(key)
    _install_network_guard()
    from app.config import Settings
    with patch.dict(Settings.model_config, {"env_file": None}):
        asyncio.run(run_local(query_only=args.query_only))


if __name__ == "__main__":
    main()
