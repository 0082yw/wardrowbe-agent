"""本机 Ollama 适配器隔离测试；仅 MockTransport 和 Mock ItemService，无真实连接。"""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from sqlalchemy.ext.asyncio import AsyncSession

from app import agent_tools
from app.agent_tools import create_query_my_wardrobe_items_tool
from app.models.item import ItemStatus
from app.services import ollama_agent_model as adapter
from app.services.agent_service import AgentService
from app.services.item_service import ItemService
from app.services.ollama_model_identity import OFFICIAL_DIGEST

pytestmark = pytest.mark.asyncio
MODEL = "qwen3:0.6b-q4_K_M"  # 仅测试标签；不表示本机已安装或已验证此模型。


def completion(content=None, *, calls=None, finish=None):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {
        "choices": [{
            "message": message,
            "finish_reason": finish or ("tool_calls" if calls else "stop"),
        }]
    }


def tool_call(call_id="call_1", *, arguments=None, name="query_my_wardrobe_items"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps({"item_type": "shirt"} if arguments is None else arguments),
        },
    }


def final_content(ids=None):
    return json.dumps({
        "status": "success" if ids else "needs_clarification",
        "answer": "基于已确认衣物的预览。" if ids else "请补充穿搭需求。",
        "outfits": [{
            "item_ids": [str(item_id) for item_id in ids],
            "reasoning": "来自本次衣橱工具查询。",
            "style_notes": "请按实际场景调整。",
        }] if ids else [],
    })


@pytest.fixture
def context(monkeypatch):
    db = Mock(spec=AsyncSession)
    user = SimpleNamespace(id=uuid4())
    items = [
        SimpleNamespace(
            id=uuid4(), name="测试衬衫", type="shirt", user_id=user.id,
            status=ItemStatus.ready, needs_wash=False,
        ),
        SimpleNamespace(
            id=uuid4(), name="测试长裤", type="pants", user_id=user.id,
            status=ItemStatus.ready, needs_wash=False,
        ),
    ]
    ctx = SimpleNamespace(db=db, user=user, items=items, active=0, peak=0)

    async def lookup(*, user_id, filters, page, page_size):
        assert user_id == user.id
        ctx.active += 1
        ctx.peak = max(ctx.peak, ctx.active)
        try:
            await asyncio.sleep(0)
            selected = [item for item in items if item.type == filters.type]
            return selected[(page - 1) * page_size:page * page_size], len(selected)
        finally:
            ctx.active -= 1

    ctx.service = Mock(spec=ItemService)
    ctx.service.get_list = AsyncMock(side_effect=lookup)
    ctx.constructor = Mock(return_value=ctx.service)
    monkeypatch.setattr(agent_tools, "ItemService", ctx.constructor)
    tool = create_query_my_wardrobe_items_tool(current_user=user, db=db)
    definition = convert_to_openai_tool(tool)
    definition["function"]["parameters"] = tool.get_input_schema().model_json_schema()
    ctx.tools = [definition]
    yield ctx
    assert db.mock_calls == [], "适配器及工作流不得直接操作数据库或管理事务"


async def test_plain_answer_uses_fixed_local_endpoint_and_ignores_proxy_settings(context, monkeypatch):
    for key in ["AI_BASE_URL", "OPENAI_BASE_URL", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        monkeypatch.setenv(key, "https://external.invalid")
    seen = []
    client_options = []
    real_client = httpx.AsyncClient

    def make_client(**kwargs):
        client_options.append(kwargs)
        return real_client(**kwargs)

    monkeypatch.setattr(adapter.httpx, "AsyncClient", make_client)

    def handle(request):
        assert str(request.url) == adapter.LOCAL_CHAT_URL
        assert request.method == "POST"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == MODEL
        assert body["stream"] is False
        assert "reasoning_effort" not in body
        assert body["tools"] == context.tools
        assert body["messages"] == [
            {"role": "system", "content": "本地助手"},
            {"role": "user", "content": "你好"},
        ]
        assert request.extensions["timeout"]["read"] == 60.0
        seen.append(body)
        return httpx.Response(200, json=completion("你好"))

    messages = [SystemMessage(content="本地助手"), HumanMessage(content="你好")]
    before = deepcopy((messages, context.tools))
    model = adapter.OllamaAgentModel(model=MODEL, transport=httpx.MockTransport(handle))
    answer = await model(messages, tools=context.tools)
    assert isinstance(answer, AIMessage)
    assert answer.content == "你好"
    assert not answer.tool_calls
    assert len(seen) == 1
    assert (messages, context.tools) == before
    assert client_options[0]["trust_env"] is False
    assert client_options[0]["follow_redirects"] is False


@pytest.mark.parametrize("disable_thinking", [False, True])
async def test_thinking_control_comes_only_from_trusted_constructor(context, disable_thinking):
    seen = []

    def handle(request):
        body = json.loads(request.content)
        if disable_thinking:
            assert body["reasoning_effort"] == "none"
        else:
            assert "reasoning_effort" not in body
        assert body["max_tokens"] == 128
        assert body["tools"] == context.tools
        seen.append(body)
        return httpx.Response(200, json=completion("你好"))

    user_content = json.dumps({
        "disable_thinking": not disable_thinking,
        "reasoning_effort": "high" if disable_thinking else "none",
    })
    model = adapter.OllamaAgentModel(
        model=MODEL, max_tokens=128, disable_thinking=disable_thinking,
        transport=httpx.MockTransport(handle),
    )
    answer = await model([HumanMessage(content=user_content)], tools=context.tools)
    assert answer.content == "你好"
    assert len(seen) == 1
    assert seen[0]["messages"] == [{"role": "user", "content": user_content}]


@pytest.mark.parametrize("content,finish", [
    (None, "stop"), (" ", "stop"), ("partial", "length"), (None, "length"),
])
async def test_disabled_thinking_still_rejects_empty_or_truncated_responses(
    context, content, finish,
):
    seen = []

    def handle(request):
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "none"
        seen.append(body)
        payload = completion(content, finish=finish)
        payload["choices"][0]["message"]["reasoning"] = "private-thinking"
        return httpx.Response(200, json=payload)

    model = adapter.OllamaAgentModel(
        model=MODEL, max_tokens=128, disable_thinking=True,
        transport=httpx.MockTransport(handle),
    )
    with pytest.raises(adapter.OllamaModelError) as exc:
        await model([HumanMessage(content="简短回答")], tools=context.tools)
    assert exc.value.code == ("response_truncated" if finish == "length" else "invalid_response")
    assert "private-thinking" not in str(exc.value)
    assert len(seen) == 1
    context.service.get_list.assert_not_awaited()


@pytest.mark.parametrize("kwargs", [
    {"model": ""},
    {"model": "https://paid.invalid/model"},
    {"model": "registry.example/model"},
    {"model": "qwen3:cloud"},
    {"model": "qwen3:latest-cloud"},
    {"model": "qwen3\n"},
    {"model": None},
    {"timeout_seconds": 0},
    {"timeout_seconds": True},
    {"timeout_seconds": float("inf")},
    {"timeout_seconds": float("nan")},
    {"max_tokens": 0},
    {"max_tokens": True},
    {"max_tokens": 4097},
    {"base_url": "https://external.invalid"},
    {"transport": object()},
    {"disable_thinking": None},
    {"disable_thinking": 0},
    {"disable_thinking": "true"},
])
async def test_invalid_configuration_is_rejected_before_any_request(kwargs):
    with pytest.raises((ValueError, TypeError)):
        adapter.OllamaAgentModel(**{"model": MODEL, **kwargs})


@pytest.mark.parametrize("disable_thinking", [False, True])
async def test_tool_arguments_and_results_preserve_call_ids(context, disable_thinking):
    calls = [tool_call()]
    seen = []

    def handle(request):
        body = json.loads(request.content)
        assert body.get("reasoning_effort") == ("none" if disable_thinking else None)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json=completion(None, calls=calls))
        assistant, result = body["messages"][-2:]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"] == calls
        assert isinstance(assistant["tool_calls"][0]["function"]["arguments"], str)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert json.loads(result["content"]) == {"items": []}
        return httpx.Response(200, json=completion(final_content()))

    model = adapter.OllamaAgentModel(
        model=MODEL, disable_thinking=disable_thinking, transport=httpx.MockTransport(handle),
    )
    history = [HumanMessage(content="查询衬衫")]
    request = await model(history, tools=context.tools)
    assert request.tool_calls[0]["args"] == {"item_type": "shirt"}
    assert request.tool_calls[0]["id"] == "call_1"
    result = await model(
        history + [request, ToolMessage(
            content='{"items": []}', tool_call_id="call_1", name="query_my_wardrobe_items",
        )],
        tools=context.tools,
    )
    assert result.response_metadata["finish_reason"] == "stop"
    assert len(seen) == 2


@pytest.mark.parametrize("arguments", ["{", "[]", "null", '"shirt"', "123", '{"page":NaN}', {}])
async def test_invalid_wire_arguments_never_become_executable_calls(context, arguments):
    call = tool_call()
    call["function"]["arguments"] = arguments
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=completion(calls=[call]))
    )
    model = adapter.OllamaAgentModel(model=MODEL, transport=transport)
    with pytest.raises(adapter.OllamaModelError, match="格式无效") as exc:
        await model([HumanMessage(content="查询")], tools=context.tools)
    assert exc.value.code == "invalid_response"
    context.service.get_list.assert_not_awaited()


@pytest.mark.parametrize("kind", [
    "empty_choices", "provider_error", "wrong_role", "wrong_content", "empty",
    "truncated", "wrong_finish", "unknown_tool", "missing_id", "duplicate_id",
    "wrong_call_type", "calls_not_list", "thinking_only", "bad_json",
])
async def test_invalid_responses_fail_closed_without_echoing_private_details(context, kind):
    call = tool_call()
    payload = completion("private-response")
    if kind == "empty_choices":
        payload["choices"] = []
    elif kind == "provider_error":
        payload["error"] = {"message": "private-response"}
    elif kind == "wrong_role":
        payload["choices"][0]["message"]["role"] = "tool"
    elif kind == "wrong_content":
        payload["choices"][0]["message"]["content"] = False
    elif kind == "empty":
        payload = completion(" ")
    elif kind == "truncated":
        payload = completion("partial", finish="length")
    elif kind == "wrong_finish":
        payload = completion(calls=[call], finish="stop")
    elif kind == "unknown_tool":
        payload = completion(calls=[tool_call(name="query_other_user")])
    elif kind == "missing_id":
        call["id"] = ""
        payload = completion(calls=[call])
    elif kind == "duplicate_id":
        payload = completion(calls=[call, call])
    elif kind == "wrong_call_type":
        call["type"] = "other"
        payload = completion(calls=[call])
    elif kind == "calls_not_list":
        payload["choices"][0]["message"]["tool_calls"] = "private-response"
    elif kind == "thinking_only":
        payload = completion(None)
        payload["choices"][0]["message"]["reasoning"] = "private-response"

    def handle(request):
        if kind == "bad_json":
            return httpx.Response(200, text="private-response")
        return httpx.Response(200, json=payload)

    model = adapter.OllamaAgentModel(model=MODEL, transport=httpx.MockTransport(handle))
    with pytest.raises(adapter.OllamaModelError) as exc:
        await model([HumanMessage(content="查询")], tools=context.tools)
    assert exc.value.code == ("response_truncated" if kind == "truncated" else "invalid_response")
    assert "private-response" not in str(exc.value)


@pytest.mark.parametrize("kind,code", [
    ("read_timeout", "timeout"), ("connect_timeout", "timeout"),
    ("connection", "connection_failed"), ("http", "http_error"), ("redirect", "http_error"),
])
async def test_network_failures_do_not_retry_redirect_or_fallback(context, kind, code):
    seen = []

    def handle(request):
        seen.append(str(request.url))
        if kind == "read_timeout":
            raise httpx.ReadTimeout("private-error", request=request)
        if kind == "connect_timeout":
            raise httpx.ConnectTimeout("private-error", request=request)
        if kind == "connection":
            raise httpx.ConnectError("private-error", request=request)
        if kind == "redirect":
            return httpx.Response(307, headers={"Location": "https://paid.invalid/model"})
        return httpx.Response(503, json={"error": "private-error"})

    model = adapter.OllamaAgentModel(model=MODEL, transport=httpx.MockTransport(handle))
    with pytest.raises(adapter.OllamaModelError) as exc:
        await model([HumanMessage(content="查询")], tools=context.tools)
    assert exc.value.code == code
    assert "private-error" not in str(exc.value)
    assert seen == [adapter.LOCAL_CHAT_URL]


async def test_incomplete_or_mismatched_history_is_rejected_before_http(context):
    seen = []
    model = adapter.OllamaAgentModel(
        model=MODEL,
        transport=httpx.MockTransport(lambda request: seen.append(request)),
    )
    call = AIMessage(content="", tool_calls=[{
        "name": "query_my_wardrobe_items", "args": {"item_type": "shirt"},
        "id": "call_1", "type": "tool_call",
    }])
    for messages in [
        [call],
        [HumanMessage(content=[{"type": "text", "text": "not-pure-text"}])],
        [ToolMessage(content="result", tool_call_id="unknown")],
        [call, ToolMessage(content="result", tool_call_id="call_1", name="wrong_name")],
        [call, HumanMessage(content="missing result")],
    ]:
        with pytest.raises(ValueError):
            await model(messages, tools=context.tools)
    assert seen == []


@pytest.mark.parametrize("batch", [False, True])
async def test_actual_langgraph_loop_reuses_bound_tools_and_executes_sequentially(context, batch):
    seen = []
    first = tool_call("shirt_call")
    second = tool_call("pants_call", arguments={"item_type": "pants"})

    def handle(request):
        assert str(request.url) == adapter.LOCAL_CHAT_URL
        body = json.loads(request.content)
        assert str(context.user.id) not in json.dumps(body)
        parameters = body["tools"][0]["function"]["parameters"]
        assert parameters["additionalProperties"] is False
        assert set(parameters["properties"]) == {"item_type", "page", "page_size"}
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json=completion(calls=[first, second] if batch else [first]))
        if len(seen) == 2 and not batch:
            assert body["messages"][-1]["tool_call_id"] == "shirt_call"
            return httpx.Response(200, json=completion(calls=[second]))
        tool_results = [message for message in body["messages"] if message["role"] == "tool"]
        assert [message["tool_call_id"] for message in tool_results] == ["shirt_call", "pants_call"]
        assert [
            json.loads(message["content"])["items"][0]["id"] for message in tool_results
        ] == [str(item.id) for item in context.items]
        return httpx.Response(200, json=completion(final_content([item.id for item in context.items])))

    model = adapter.OllamaAgentModel(model=MODEL, transport=httpx.MockTransport(handle))
    result = await AgentService(model=model).generate_preview(
        "请搭配已有衣物", current_user=context.user, db=context.db,
    )
    assert result.status == "success"
    assert result.outfits[0].item_ids == [item.id for item in context.items]
    assert context.service.get_list.await_count == 2
    assert context.peak == 1
    assert len(seen) == (2 if batch else 3)


@pytest.mark.parametrize("parameters", [
    {"page": 0}, {"page_size": 101}, {"user_id": "other"},
    {"current_user": "other"}, {"db": "other"}, {"session": "other"},
])
async def test_service_still_rejects_invalid_or_identity_overriding_arguments(context, parameters):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=completion(
            calls=[tool_call(arguments={"item_type": "shirt", **parameters})]
        ))

    model = adapter.OllamaAgentModel(model=MODEL, transport=httpx.MockTransport(handle))
    result = await AgentService(model=model).generate_preview(
        "查询衬衫", current_user=context.user, db=context.db,
    )
    assert result.status == "invalid_result"
    assert result.outfits == []
    assert len(seen) == 1
    context.service.get_list.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local_cpu", "remote_gpu"])
@pytest.mark.parametrize("field", ["reasoning", "reasoning_content", "thinking"])
async def test_reasoning_returned_despite_none_is_not_a_success(mode, field):
    requests = []
    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "none"
        payload = completion("简短回答")
        payload["choices"][0]["message"][field] = "private reasoning"
        return httpx.Response(200, json=payload)
    model = adapter.OllamaAgentModel(model=MODEL, execution_mode=mode, disable_thinking=True,
                                    transport=httpx.MockTransport(handler))
    if mode == "remote_gpu":
        model.configure_non_thinking({
            "details": {"format": "gguf", "family": "qwen3"},
            "model_info": {"general.architecture": "qwen3"},
            "capabilities": ["completion", "thinking"],
            "thinking": {"values": [True, False], "default": True},
            "template": "{{ if .Think }}<think>{{ end }}assistant",
        })
    with pytest.raises(adapter.OllamaModelError) as error:
        await model([HumanMessage(content="hello")], tools=[])
    assert error.value.code == "thinking_not_disabled"
    assert "private reasoning" not in str(error.value) and len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["<think>private reasoning</think>answer", "private</think>answer"])
async def test_no_thinking_response_cannot_expose_think_markers(content):
    model = adapter.OllamaAgentModel(model=MODEL, disable_thinking=True,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=completion(content))))
    with pytest.raises(adapter.OllamaModelError) as error:
        await model([HumanMessage(content="hello")], tools=[])
    assert error.value.code == "thinking_not_disabled"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local_cpu", "remote_gpu"])
async def test_native_payload_is_not_mistaken_for_openai_completion(mode):
    requests = []
    def handler(request):
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"},
                                        "done": True, "done_reason": "stop"})
    model = adapter.OllamaAgentModel(model=MODEL, execution_mode=mode,
                                    transport=httpx.MockTransport(handler))
    with pytest.raises(adapter.OllamaModelError) as error:
        await model([HumanMessage(content="hello")], tools=[])
    assert error.value.code == "invalid_response" and len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("has_calls", [False, True])
async def test_length_rejects_even_valid_json_or_valid_tool_call(has_calls):
    payload = completion(final_content(), calls=[tool_call()] if has_calls else None, finish="length")
    model = adapter.OllamaAgentModel(model=MODEL, disable_thinking=True,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    tools = [{"type": "function", "function": {"name": "query_my_wardrobe_items", "parameters": {"type": "object"}}}]
    with pytest.raises(adapter.OllamaModelError) as error:
        await model([HumanMessage(content="hello")], tools=tools)
    assert error.value.code == "response_truncated"


@pytest.mark.parametrize("kind", ["unknown", "unqueried", "no_query"])
async def test_service_still_rejects_unconfirmed_clothing_ids(context, kind):
    seen = []

    def handle(request):
        seen.append(request)
        if len(seen) == 1 and kind != "no_query":
            return httpx.Response(200, json=completion(calls=[tool_call()]))
        selected = (
            uuid4() if kind == "unknown"
            else context.items[1].id if kind == "unqueried"
            else context.items[0].id
        )
        return httpx.Response(200, json=completion(final_content([selected])))

    model = adapter.OllamaAgentModel(model=MODEL, transport=httpx.MockTransport(handle))
    result = await AgentService(model=model).generate_preview(
        "查询衬衫", current_user=context.user, db=context.db,
    )
    assert result.status == "invalid_result"
    assert result.outfits == []
    assert context.service.get_list.await_count == int(kind != "no_query")


async def test_existing_model_and_tool_budgets_remain_effective(context):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=completion(calls=[tool_call(f"loop_{len(seen)}")]))

    model = adapter.OllamaAgentModel(model=MODEL, transport=httpx.MockTransport(handle))
    result = await AgentService(model=model, max_model_calls=2).generate_preview(
        "查询衬衫", current_user=context.user, db=context.db,
    )
    assert result.status == "invalid_result"
    assert len(seen) == 2
    assert context.service.get_list.await_count == 2

    context.service.get_list.reset_mock()
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, json=completion(calls=[tool_call("one"), tool_call("two")]),
    ))
    result = await AgentService(
        model=adapter.OllamaAgentModel(model=MODEL, transport=transport), max_tool_calls=1,
    ).generate_preview("查询衬衫", current_user=context.user, db=context.db)
    assert result.status == "invalid_result"
    context.service.get_list.assert_not_awaited()


async def test_normal_final_answer_does_not_request_or_execute_more_tools(context):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=completion(final_content()))

    model = adapter.OllamaAgentModel(model=MODEL, transport=httpx.MockTransport(handle))
    result = await AgentService(model=model).generate_preview(
        "你好", current_user=context.user, db=context.db,
    )
    assert result.status == "needs_clarification"
    assert len(seen) == 1
    context.service.get_list.assert_not_awaited()


def instruct_metadata():
    return json.loads((Path(__file__).parent / "fixtures/qwen3_instruct_2507_metadata.json").read_text())


@pytest.mark.asyncio
async def test_instruct_tool_calls_and_tool_results_keep_ids_without_thinking_switches():
    seen = []
    def handler(request):
        body = json.loads(request.content); seen.append(body)
        assert "reasoning_effort" not in body and "think" not in body
        if len(seen) == 1:
            return httpx.Response(200, json=completion(calls=[tool_call("instruct_call")]))
        assert body["messages"][-1]["role"] == "tool"
        assert body["messages"][-1]["tool_call_id"] == "instruct_call"
        return httpx.Response(200, json=completion("白色衬衫"))
    model = adapter.OllamaAgentModel(model="qwen3:4b-instruct-2507-q4_K_M", execution_mode="remote_gpu",
        disable_thinking=True, transport=httpx.MockTransport(handler))
    model.configure_non_thinking(instruct_metadata(), model_digest=OFFICIAL_DIGEST)
    tools = [{"type": "function", "function": {"name": "query_my_wardrobe_items", "parameters": {"type": "object"}}}]
    human = HumanMessage(content="query")
    call = await model([human], tools=tools)
    assert call.tool_calls[0]["id"] == "instruct_call" and isinstance(call.tool_calls[0]["args"], dict)
    result = await model([human, call, ToolMessage(content='[{"name":"白色衬衫"}]', tool_call_id="instruct_call")], tools=tools)
    assert result.content == "白色衬衫" and len(seen) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("content,finish,code", [
    ("", "stop", "invalid_response"), (" ", "stop", "invalid_response"),
    ("partial", "length", "response_truncated"), ("<think>private</think>", "stop", "thinking_not_disabled"),
])
async def test_instruct_response_validation_stays_strict(content, finish, code):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=completion(content, finish=finish))
    model = adapter.OllamaAgentModel(model="qwen3:4b-instruct-2507-q4_K_M", execution_mode="remote_gpu",
        disable_thinking=True, transport=httpx.MockTransport(handler))
    model.configure_non_thinking(instruct_metadata(), model_digest=OFFICIAL_DIGEST)
    with pytest.raises(adapter.OllamaModelError) as error:
        await model([HumanMessage(content="hello")], tools=[])
    assert error.value.code == code and len(seen) == 1


@pytest.mark.asyncio
async def test_unknown_remote_capability_cannot_send_generation():
    seen = []
    model = adapter.OllamaAgentModel(model="unknown", execution_mode="remote_gpu", disable_thinking=True,
        transport=httpx.MockTransport(lambda request: seen.append(request)))
    with pytest.raises(adapter.OllamaModelError) as error:
        await model([HumanMessage(content="hello")], tools=[])
    assert error.value.code == "thinking_metadata_unavailable" and seen == []
