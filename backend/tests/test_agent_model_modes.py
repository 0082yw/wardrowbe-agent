"""双模式离线验证：真实 LangGraph/ToolNode，Mock HTTP、业务服务与 Session。"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app import agent_tools
from app.api import agent as api
from app.database import get_db
from app.schemas.agent import AgentPreviewResponse, AgentRequest, AgentRuntimeStatus
from app.services import agent_resources as resources
from app.services.agent_audit import AgentAudit, current_audit
from app.services.agent_model_config import AgentModelConfig, create_agent_text_model, DEFAULT_REMOTE_MODEL
from app.services.local_agent_model import LocalAgentTextModel
from app.services.ollama_agent_model import OllamaAgentModel, OllamaModelError
from app.services.remote_agent_model import RemoteAgentTextModel
from app.utils.auth import get_current_user
from app.services.ollama_model_identity import OFFICIAL_DIGEST

_READ_RESOURCES = resources.read_resources


def snapshot(windows=1.5, wsl=.75, psi=0, swap_in=0, swap_out=0):
    return resources.ResourceSnapshot(int(windows * 2**30), int(wsl * 2**30), psi, swap_in, swap_out)


def completion(content='{"answer":"已完成衬衫查询。"}', *, finish="stop", calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                    "tool_calls": calls}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30}}


def model_metadata(profile="hybrid"):
    data = {
        "details": {"format": "gguf", "family": "qwen3"},
        "model_info": {"general.architecture": "qwen3"},
        "capabilities": ["completion", "tools", "thinking"],
        "thinking": {"values": [True, False], "default": True},
        "template": "{{ if .Think }}<think>{{ end }}assistant",
    }
    if profile == "instruct_only":
        data = json.loads((Path(__file__).parent / "fixtures/qwen3_instruct_2507_metadata.json").read_text())
        data["thinking"] = {"values": [False], "default": False}
    elif profile == "thinking_only":
        data["model_info"].update({"general.finetune": "Thinking", "general.version": "2507"})
        data["thinking"] = {"values": [True], "default": True}
        data["template"] = "<think>"
    return data


class OfflineRemote:
    def __init__(self, *, answer=None, error=None, tags_status=200, tags=None, show=None, show_status=200, model="qwen3:4b"):
        self.answer = completion() if answer is None else answer
        self.error = error
        self.tags_status = tags_status
        self.model = model
        self.tags = {"models": [{"name": model, "size": 2500000000,
                                "details": {"format": "gguf"}}]} if tags is None else tags
        # 这是支持关闭思考的隔离替身，不代表服务器当前 thinking-only 权重。
        if tags is None and model == DEFAULT_REMOTE_MODEL:
            self.tags["models"][0]["digest"] = OFFICIAL_DIGEST
        self.show = model_metadata() if show is None else show
        self.show_status = show_status
        self.requests = []
        self.chats = 0

    async def __call__(self, request):
        self.requests.append(request)
        assert request.url.host == "127.0.0.1" and request.url.port == 11435
        assert "authorization" not in request.headers
        if request.url.path == "/api/tags":
            if self.error == "tunnel":
                raise httpx.ConnectError("secret connection detail", request=request)
            return httpx.Response(self.tags_status, json=self.tags,
                                  headers={"Location": "https://external.invalid/private"})
        if request.url.path == "/api/show":
            assert request.method == "POST" and json.loads(request.content) == {"model": self.model}
            if self.error == "show_blocked":
                raise PermissionError("private runtime policy detail")
            if self.error == "show_timeout":
                raise httpx.ReadTimeout("private metadata timeout", request=request)
            return httpx.Response(self.show_status, json=self.show)
        assert request.url.path == "/v1/chat/completions"
        self.chats += 1
        data = json.loads(request.content)
        assert data["model"] == self.model and data["tools"] == []
        if self.show.get("model_info", {}).get("general.finetune") == "Instruct":
            assert "reasoning_effort" not in data and "think" not in data
        else:
            assert data["reasoning_effort"] == "none"
        assert data["max_tokens"] == 256
        if self.error == "timeout":
            raise httpx.ReadTimeout("secret timeout detail", request=request)
        if self.error == "http":
            return httpx.Response(503, json={"error": "private failure"})
        return httpx.Response(200, json=self.answer)


@pytest.fixture(autouse=True)
def offline_resources(monkeypatch):
    monkeypatch.setattr(resources, "read_resources", lambda memory_file=None: snapshot())
    monkeypatch.delenv("WARDROBE_AGENT_MODEL_MODE", raising=False)
    monkeypatch.delenv("WARDROBE_AGENT_REMOTE_MODEL", raising=False)


def test_default_cpu_remains_compatible_without_io():
    assert AgentModelConfig.from_env({}) == AgentModelConfig()
    assert isinstance(create_agent_text_model(), LocalAgentTextModel)
    assert AgentModelConfig.from_env({"WARDROBE_AGENT_REMOTE_MODEL": "qwen3:4b"}).execution_mode == "local_cpu"


def test_remote_initialization_uses_only_trusted_process_configuration(monkeypatch):
    monkeypatch.setenv("WARDROBE_AGENT_MODEL_MODE", "remote_gpu")
    monkeypatch.setenv("WARDROBE_AGENT_REMOTE_MODEL", "qwen3:4b")
    model = create_agent_text_model()
    assert isinstance(model, RemoteAgentTextModel)
    assert model.model_name == "qwen3:4b" and model.tunnel_status == "unknown"
    assert model.gpu_verified == "unknown" and model._adapter._transport is None


@pytest.mark.parametrize("env", [
    {"WARDROBE_AGENT_MODEL_MODE": "remote_gpu", "WARDROBE_AGENT_REMOTE_MODEL": ""},
    {"WARDROBE_AGENT_MODEL_MODE": "unknown"},
    {"WARDROBE_AGENT_MODEL_MODE": "LOCAL_CPU"},
    {"WARDROBE_AGENT_MODEL_MODE": "remote_gpu", "WARDROBE_AGENT_REMOTE_MODEL": "https://external.invalid/model"},
    {"WARDROBE_AGENT_MODEL_MODE": "remote_gpu", "WARDROBE_AGENT_REMOTE_MODEL": "qwen:cloud"},
    {"WARDROBE_AGENT_MODEL_MODE": "remote_gpu", "WARDROBE_AGENT_REMOTE_MODEL": "private/model"},
])
def test_invalid_startup_configuration_is_not_replaced_by_cpu(env):
    with pytest.raises(ValueError):
        AgentModelConfig.from_env(env)


@pytest.mark.parametrize("name", ["user_id", "db", "model", "endpoint", "execution_mode"])
def test_api_cannot_select_identity_or_inference_configuration(name):
    with pytest.raises(ValidationError):
        AgentRequest.model_validate({"message": "查询衬衫", name: "remote_gpu"})


@pytest.mark.parametrize("endpoint", ["https://external.invalid", "http://inference.example.invalid:11434", "http://localhost:11435"])
def test_endpoints_are_not_constructor_configuration(endpoint):
    with pytest.raises(TypeError):
        RemoteAgentTextModel(model="qwen3:4b", base_url=endpoint)


def test_adapter_rejects_unknown_mode_and_non_mock_transport():
    with pytest.raises(ValueError):
        OllamaAgentModel(model="qwen3:4b", execution_mode="external")
    with pytest.raises(ValueError):
        RemoteAgentTextModel(model="qwen3:4b", transport=object())


def test_mode_specific_thresholds_do_not_leak_into_each_other():
    resources.validate_resources(snapshot(), "remote_gpu")
    with pytest.raises(OllamaModelError, match="Windows"):
        resources.validate_resources(snapshot(), "local_cpu")
    resources.validate_resources(snapshot(2.5, 1.25), "local_cpu")
    resources.validate_resources(snapshot(1, .5, 1), "remote_gpu")


@pytest.mark.parametrize("value", [snapshot(.99), snapshot(wsl=.49), snapshot(psi=1.01)])
@pytest.mark.asyncio
async def test_remote_severe_resource_pressure_rejects_before_network(monkeypatch, value):
    monkeypatch.setattr(resources, "read_resources", lambda memory_file=None: value)
    fake = OfflineRemote()
    model = RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))
    with pytest.raises(OllamaModelError):
        await model([HumanMessage(content="hello")], tools=[])
    assert fake.requests == []


@pytest.mark.asyncio
async def test_new_swap_activity_rejects_before_generation(monkeypatch):
    values = iter([snapshot(), snapshot(swap_in=1)])
    monkeypatch.setattr(resources, "read_resources", lambda memory_file=None: next(values))
    fake = OfflineRemote()
    with pytest.raises(OllamaModelError) as error:
        await RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))(
            [HumanMessage(content="hello")], tools=[])
    assert error.value.code == "swap_activity" and fake.chats == 0


@pytest.mark.asyncio
async def test_successful_remote_http_never_claims_gpu_and_audit_has_safe_metrics(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://external.invalid:8888")
    monkeypatch.setenv("HTTPS_PROXY", "http://external.invalid:8888")
    fake = OfflineRemote()
    audit = AgentAudit(); token = current_audit.set(audit)
    try:
        model = RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))
        answer = await model([HumanMessage(content="secret private prompt")], tools=[])
    finally:
        current_audit.reset(token)
    assert isinstance(answer, AIMessage) and answer.response_metadata["finish_reason"] == "stop"
    data = audit.snapshot()
    assert data["execution_mode"] == "remote_gpu" and data["model_name"] == "qwen3:4b"
    assert data["tunnel_status"] == "connected" and data["gpu_verified"] == "unknown"
    assert data["input_tokens"] == 21 and data["output_tokens"] == 9
    assert data["inference_latency"] >= 0 and fake.chats == 1
    assert "secret private prompt" not in json.dumps(data)
    assert all(r.url.port == 11435 for r in fake.requests)


@pytest.mark.asyncio
async def test_http_client_disables_environment_proxy_and_redirects(monkeypatch):
    real = httpx.AsyncClient
    flags = []
    def spy(**kwargs):
        flags.append((kwargs.get("trust_env"), kwargs.get("follow_redirects")))
        return real(**kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", spy)
    fake = OfflineRemote()
    await RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))(
        [HumanMessage(content="hello")], tools=[])
    assert flags and all(pair == (False, False) for pair in flags)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_redirects_are_not_followed(status):
    fake = OfflineRemote(tags_status=status)
    with pytest.raises(OllamaModelError):
        await RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake)).check_ready()
    assert len(fake.requests) == 1 and fake.chats == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error,code", [("tunnel", "tunnel_unavailable"), ("timeout", "timeout"), ("http", "http_error")])
async def test_failures_have_no_retry_or_cpu_fallback(monkeypatch, error, code):
    cpu = AsyncMock(side_effect=AssertionError("cannot fall back"))
    monkeypatch.setattr(LocalAgentTextModel, "__call__", cpu)
    fake = OfflineRemote(error=error)
    audit = AgentAudit(); token = current_audit.set(audit)
    try:
        with pytest.raises(OllamaModelError) as failure:
            await RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))(
                [HumanMessage(content="hello")], tools=[])
    finally:
        current_audit.reset(token)
    assert failure.value.code == code
    assert fake.chats == (0 if error == "tunnel" else 1)
    cpu.assert_not_called()
    assert "secret" not in json.dumps(audit.snapshot())
    assert audit.snapshot()["gpu_verified"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("tags", [{"models": []}, {"models": [{"name": "qwen3:4b", "size": 0}]},
    {"models": [{"name": "qwen3:4b", "size": 100, "remote_host": "https://cloud.invalid", "details": {"format": "gguf"}}]}])
async def test_missing_or_cloud_stub_models_are_never_generated(tags):
    fake = OfflineRemote(tags=tags)
    with pytest.raises(OllamaModelError):
        await RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))(
            [HumanMessage(content="hello")], tools=[])
    assert fake.chats == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [completion("", finish="stop"), completion("partial", finish="length"),
    {"choices": []}])
async def test_invalid_or_truncated_responses_are_not_successful(answer):
    fake = OfflineRemote(answer=answer)
    with pytest.raises(OllamaModelError) as error:
        await RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))(
            [HumanMessage(content="hello")], tools=[])
    expected = "response_truncated" if (answer.get("choices") or [{}])[0].get("finish_reason") == "length" else "invalid_response"
    assert error.value.code == expected and fake.chats == 1


@pytest.mark.asyncio
async def test_remote_adapter_preserves_tool_call_and_tool_message_association():
    count = 0
    async def handler(request):
        nonlocal count
        count += 1
        assert request.url.port == 11435
        body = json.loads(request.content)
        if count == 1:
            return httpx.Response(200, json=completion(None, finish="tool_calls", calls=[{
                "id": "call_remote_test", "type": "function",
                "function": {"name": "query_test", "arguments": '{"item_type":"shirt"}'},
            }]))
        assert body["messages"][-1]["role"] == "tool"
        assert body["messages"][-1]["tool_call_id"] == "call_remote_test"
        return httpx.Response(200, json=completion("done"))
    model = OllamaAgentModel(model="qwen3:4b", execution_mode="remote_gpu", transport=httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "query_test", "parameters": {"type": "object"}}}]
    human = HumanMessage(content="query")
    request = await model([human], tools=tools)
    assert request.tool_calls[0]["args"] == {"item_type": "shirt"}
    answer = await model([human, request, ToolMessage(content="[]", tool_call_id="call_remote_test")], tools=tools)
    assert answer.content == "done" and count == 2


@pytest.fixture
def api_context(monkeypatch):
    from app.services.wardrobe_query_agent import WardrobeQueryAgentModel
    user = SimpleNamespace(id=uuid4(), family_id=None)
    db = Mock(spec=AsyncSession)
    item = SimpleNamespace(id=uuid4(), user_id=user.id, name="private wardrobe name", type="shirt",
        status="ready", needs_wash=False, is_archived=False, thumbnail_path=None, image_path=None,
        processing_kind=None, colors=[], primary_color=None, formality=None)
    service = Mock(spec=agent_tools.ItemService)
    service.get_list = AsyncMock(return_value=([item], 1))
    service.get_by_id = AsyncMock(return_value=item)
    monkeypatch.setattr(agent_tools, "ItemService", Mock(return_value=service))
    monkeypatch.setattr(api, "ItemService", Mock(return_value=service))
    monkeypatch.setattr(api, "rate_limit_by_user", AsyncMock())
    fake = OfflineRemote()
    remote = RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))
    controller = WardrobeQueryAgentModel(remote)
    app = FastAPI();app.include_router(api.router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[api.get_agent_model] = lambda: controller
    ctx = SimpleNamespace(user=user, db=db, item=item, service=service, fake=fake,
                          remote=remote, controller=controller, app=app)
    yield ctx
    assert db.mock_calls == [], "不能执行 SQL 写入或保存穿搭"


async def api_request(ctx, *, status=False):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=ctx.app), base_url="http://test") as client:
        return await client.get('/api/v1/agent/status') if status else await client.post(
            '/api/v1/agent/preview', json={"message": "查询我的衬衫"})


@pytest.mark.asyncio
async def test_remote_api_uses_same_real_graph_scope_images_and_final_schema(api_context, monkeypatch, caplog):
    from app.utils import signed_urls
    ctx = api_context
    monkeypatch.setattr(signed_urls, 'get_settings', lambda: SimpleNamespace(secret_key='isolated-test-signing'))
    ctx.item.thumbnail_path = f'{ctx.user.id}/shirt.jpg'
    response = await api_request(ctx)
    assert response.status_code == 200
    result = AgentPreviewResponse.model_validate_json(response.content)
    assert result.status == 'success' and result.outfits[0].item_ids == [ctx.item.id]
    assert result.outfits[0].items[0].name == ctx.item.name
    assert result.outfits[0].items[0].thumbnail_url.startswith(f'/api/v1/images/{ctx.user.id}/shirt.jpg?')
    assert ctx.service.get_list.await_count == ctx.fake.chats == 1
    assert ctx.service.get_list.await_args.kwargs['user_id'] == ctx.user.id
    assert all(c.kwargs['user_id'] == ctx.user.id for c in ctx.service.get_by_id.await_args_list)
    messages = ctx.controller.trace.graph_inputs[-1]
    tool = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool.tool_call_id == ctx.controller.call['id']
    entry = next(json.loads(r.message) for r in caplog.records if 'agent_preview_audit' in r.message)
    for name in ['execution_mode','model_name','tunnel_status','inference_latency','input_tokens',
                 'output_tokens','gpu_verified','tool_execution_count','candidate_count','preview_count','final_schema_status']:
        assert name in entry
    assert entry['tool_execution_count'] == entry['candidate_count'] == entry['preview_count'] == 1
    assert entry['final_schema_status'] is True and entry['gpu_verified'] == 'unknown'
    assert ctx.item.name not in json.dumps(entry) and 'sig=' not in json.dumps(entry)


@pytest.mark.asyncio
async def test_remote_status_is_metadata_only_not_gpu_proof(api_context):
    response = await api_request(api_context, status=True)
    value = AgentRuntimeStatus.model_validate_json(response.content)
    assert value.execution_mode == 'remote_gpu' and value.available and not value.cpu_only
    assert value.tunnel_status == 'connected' and value.gpu_verified == 'unknown'
    assert api_context.fake.chats == 0
    api_context.service.get_list.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', ['foreign', 'archived', 'wash', 'status'])
async def test_remote_query_never_accepts_unowned_or_unwearable_items(api_context, bad):
    ctx = api_context
    if bad == 'foreign':ctx.item.user_id = uuid4()
    elif bad == 'archived':ctx.item.is_archived = True
    elif bad == 'wash':ctx.item.needs_wash = True
    else:ctx.item.status = 'processing'
    result = (await api_request(ctx)).json()
    assert result['status'] != 'success' and result['outfits'] == []
    assert ctx.fake.chats == 0


@pytest.mark.asyncio
async def test_remote_image_cannot_point_at_another_owner(api_context):
    api_context.item.thumbnail_path = f'{uuid4()}/private.jpg'
    result = (await api_request(api_context)).json()
    assert result['status'] == 'success' and result['outfits'][0]['items'][0]['thumbnail_url'] is None


@pytest.mark.asyncio
async def test_tunnel_failure_is_clear_and_audited_not_mock_success(api_context, caplog):
    api_context.fake.error = 'tunnel'
    value = (await api_request(api_context)).json()
    assert value['status'] == 'invalid_result' and value['outfits'] == []
    assert 'SSH' in value['answer'] and '未回退' in value['answer']
    entry = next(json.loads(r.message) for r in caplog.records if 'agent_preview_audit' in r.message)
    assert entry['tunnel_status'] == 'disconnected' and entry['model_request_count'] == 0
    assert entry['error'] == 'tunnel_unavailable' and entry['gpu_verified'] == 'unknown'


@pytest.mark.asyncio
async def test_api_rejects_model_forged_ids(api_context):
    api_context.fake.answer = completion(json.dumps({'answer':'ok','item_ids':[str(uuid4())]}))
    value = (await api_request(api_context)).json()
    assert value['status'] == 'invalid_result' and value['outfits'] == []
    assert api_context.fake.chats == 1


@pytest.mark.parametrize('changes', [{'gpu_verified':True}, {'cpu_only':True},
    {'execution_mode':'local_cpu'}, {'tunnel_status':'disconnected'}, {'model':'https://external.invalid'}])
def test_remote_status_rejects_unproven_gpu_or_contradictory_fields(changes):
    with pytest.raises(ValidationError):
        AgentRuntimeStatus.model_validate({
            'mode':'remote_ollama','model':'qwen3:4b','available':True,'cpu_only':False,
            'execution_mode':'remote_gpu','tunnel_status':'connected','gpu_verified':'unknown',**changes})


@pytest.mark.parametrize('sample', [None, {'unix_seconds':0,'available_bytes':4*2**30},
    {'unix_seconds':999999999999,'available_bytes':4*2**30}, {'unix_seconds':0,'available_bytes':True}])
def test_unavailable_or_stale_windows_samples_fail_closed(tmp_path, monkeypatch, sample):
    monkeypatch.delenv('WARDROWBE_MEMORY_FILE',raising=False)
    p=tmp_path/'memory.json'
    if sample is not None:p.write_text(json.dumps(sample))
    with pytest.raises(OllamaModelError) as error:
        _READ_RESOURCES(str(p))
    assert error.value.code=='resources_unavailable'


@pytest.mark.asyncio
async def test_resource_pressure_cancels_inflight_request_without_retry(monkeypatch):
    calls=0;cancelled=False
    sequence=iter([snapshot(),snapshot(),snapshot(windows=.2)])
    monkeypatch.setattr(resources,'read_resources',lambda memory_file=None:next(sequence))
    async def handler(request):
        nonlocal calls,cancelled
        if request.url.path=='/api/tags':
            return httpx.Response(200,json=OfflineRemote().tags)
        if request.url.path=='/api/show':
            return httpx.Response(200,json=OfflineRemote().show)
        calls+=1
        try:await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled=True;raise
    with pytest.raises(OllamaModelError) as error:
        await RemoteAgentTextModel(model='qwen3:4b',transport=httpx.MockTransport(handler))(
            [HumanMessage(content='hello')],tools=[])
    assert error.value.code=='windows_memory_low' and calls==1 and cancelled


def test_launcher_remote_resources_have_basic_gate_and_failed_tunnel_never_loads_cpu(tmp_path,monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('mode_runtime',Path(__file__).parents[1]/'scripts/local_agent.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'fresh_memory',lambda p:{'available_bytes':int(1.5*2**30)})
    original=Path.read_text
    def read(p,*args,**kwargs):
        if str(p)=='/proc/meminfo':return 'MemAvailable: 786432 kB\n'
        if str(p)=='/proc/pressure/memory':return 'full avg10=0.00 avg60=0.00 total=0\n'
        if str(p)=='/proc/vmstat':return 'pswpin 0\npswpout 0\n'
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    assert module.resources('unused',execution_mode='remote_gpu')['windows_gib']==1.5
    with pytest.raises(module.RuntimeErrorSafe,match='需2.5'):
        module.resources('unused')
    monkeypatch.setenv('WARDROBE_AGENT_MODEL_MODE','remote_gpu')
    monkeypatch.setenv('WARDROBE_AGENT_REMOTE_MODEL','qwen3:4b')
    docker=Mock();monkeypatch.setattr(module,'docker_dependencies',docker)
    monkeypatch.setattr(module,'PRIVATE',tmp_path)
    manager=module.Manager({'legacy_heartbeat':'unused','guard_hashes':{}})
    monkeypatch.setattr(module,'port_open',lambda p:False)
    monkeypatch.setattr(manager,'ensure_monitor',Mock())
    monkeypatch.setattr(manager,'baseline',Mock())
    cpu=Mock();monkeypatch.setattr(manager,'ensure_ollama',cpu)
    monkeypatch.setattr(RemoteAgentTextModel,'check_ready',AsyncMock(side_effect=OllamaModelError('offline',code='tunnel_unavailable')))
    with pytest.raises(module.RuntimeErrorSafe,match='SSH'):
        manager.start()
    docker.assert_called_once()
    cpu.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("values,default", [([True, False], True), ([False], False)])
async def test_remote_selects_controls_from_verified_profile(values, default):
    show = model_metadata("instruct_only" if values == [False] else "hybrid")
    show["thinking"] = {"values": values, "default": default}
    name = DEFAULT_REMOTE_MODEL if values == [False] else "qwen3:4b"
    fake = OfflineRemote(show=show, model=name)
    model = RemoteAgentTextModel(model=name, transport=httpx.MockTransport(fake))
    answer = await model([HumanMessage(content="简短回答")], tools=[])
    assert answer.content and answer.response_metadata["finish_reason"] == "stop"
    assert [r.url.path for r in fake.requests] == ["/api/tags", "/api/show", "/v1/chat/completions"]
    body = json.loads(fake.requests[-1].content)
    assert model._adapter.api_type == "openai_chat_completions"
    assert body["max_tokens"] == 256
    if values == [False]:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == "none"
    assert "think" not in body and "num_predict" not in body and "options" not in body
    assert "/no_think" not in json.dumps(body) and fake.chats == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking,code", [
    ({"values": [True], "default": True}, "remote_metadata_invalid"),
    ({"values": ["none", "high"], "default": "none"}, "remote_metadata_invalid"),
    ({"values": ["false"], "default": "false"}, "remote_metadata_invalid"),
    (None, "thinking_metadata_unavailable"),
    ({}, "remote_metadata_invalid"),
    ({"values": [0, True], "default": True}, "remote_metadata_invalid"),
    ({"values": [False, False], "default": False}, "remote_metadata_invalid"),
    ({"values": [False], "default": True}, "remote_metadata_invalid"),
    ({"values": [False], "default": 0}, "remote_metadata_invalid"),
    ({"values": [False, {}], "default": False}, "remote_metadata_invalid"),
])
async def test_thinking_metadata_not_http_200_decides_compatibility(thinking, code, monkeypatch):
    show = {} if thinking is None else {"thinking": thinking}
    fake = OfflineRemote(show=show)
    cpu = AsyncMock(side_effect=AssertionError("no fallback"))
    monkeypatch.setattr(LocalAgentTextModel, "__call__", cpu)
    audit = AgentAudit(); token = current_audit.set(audit)
    try:
        model = RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))
        with pytest.raises(OllamaModelError) as error:
            await model([HumanMessage(content="private request")], tools=[])
    finally:
        current_audit.reset(token)
    assert error.value.code == code
    assert fake.chats == 0 and len(fake.requests) == 2
    assert model.tunnel_status == "connected" and model.gpu_verified == "unknown"
    assert audit.snapshot()["model_request_count"] == 0
    assert audit.snapshot()["error"] == code
    assert "private request" not in json.dumps(audit.snapshot())
    cpu.assert_not_called()




def test_approved_remote_default_does_not_switch_cpu_mode():
    assert AgentModelConfig.from_env({}).model_name == "qwen3:0.6b-cpu"
    assert AgentModelConfig.from_env({"WARDROBE_AGENT_MODEL_MODE": "remote_gpu"}).model_name == DEFAULT_REMOTE_MODEL
    assert DEFAULT_REMOTE_MODEL == "qwen3:4b-instruct-2507-q4_K_M"


@pytest.mark.parametrize("profile", ["hybrid", "instruct_only"])
def test_profiles_require_metadata_and_template_evidence(profile):
    from app.services.ollama_agent_model import validate_non_thinking_metadata
    name = DEFAULT_REMOTE_MODEL if profile == "instruct_only" else "qwen3:4b"
    assert validate_non_thinking_metadata(model_metadata(profile), model_name=name,
                                          model_digest=OFFICIAL_DIGEST) == profile


def test_verified_instruct_can_omit_thinking_descriptor():
    from app.services.ollama_agent_model import validate_non_thinking_metadata
    data = model_metadata("instruct_only"); data.pop("thinking")
    assert validate_non_thinking_metadata(data, model_name=DEFAULT_REMOTE_MODEL,
                                         model_digest=OFFICIAL_DIGEST) == "instruct_only"


@pytest.mark.parametrize("change", ["unknown_identity", "missing_template", "blank_template", "think_prefill",
    "think_history", "wrong_digest", "wrong_name"])
def test_instruct_profile_never_accepts_unknown_or_contradictory_evidence(change):
    from app.services.ollama_agent_model import validate_non_thinking_metadata
    data = model_metadata("instruct_only")
    if change == "unknown_identity": data["model_info"] = {}
    elif change == "missing_template": data.pop("template")
    elif change == "blank_template": data["template"] = "  "
    elif change == "think_prefill": data["template"] += "<think>"
    elif change == "think_history": data["template"] += "{{ .Thinking }}"
    name = "wrong" if change == "wrong_name" else DEFAULT_REMOTE_MODEL
    digest = "0" * 64 if change == "wrong_digest" else OFFICIAL_DIGEST
    with pytest.raises(OllamaModelError):
        validate_non_thinking_metadata(data, model_name=name, model_digest=digest)


@pytest.mark.asyncio
@pytest.mark.parametrize("descriptor", [True, False])
async def test_instruct_production_path_omits_all_thinking_controls(descriptor):
    metadata = model_metadata("instruct_only")
    if not descriptor: metadata.pop("thinking")
    fake = OfflineRemote(show=metadata, model=DEFAULT_REMOTE_MODEL)
    model = RemoteAgentTextModel(model=DEFAULT_REMOTE_MODEL, transport=httpx.MockTransport(fake))
    answer = await model([HumanMessage(content="简短回答")], tools=[])
    assert answer.content and model._adapter._non_thinking_profile == "instruct_only"
    body = json.loads(fake.requests[-1].content)
    assert "reasoning_effort" not in body and "think" not in body and "/no_think" not in json.dumps(body)
    assert fake.chats == 1 and model.gpu_verified == "unknown"


@pytest.mark.asyncio
async def test_verified_instruct_still_rejects_reasoning_output():
    reply = completion("private answer"); reply["choices"][0]["message"]["reasoning"] = "private reasoning"
    fake = OfflineRemote(show=model_metadata("instruct_only"), model=DEFAULT_REMOTE_MODEL, answer=reply)
    with pytest.raises(OllamaModelError) as error:
        await RemoteAgentTextModel(model=DEFAULT_REMOTE_MODEL, transport=httpx.MockTransport(fake))(
            [HumanMessage(content="hello")], tools=[])
    assert error.value.code == "thinking_not_disabled" and fake.chats == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("digest", [None, "0" * 64])
async def test_instruct_tag_digest_mismatch_prevents_any_generation(digest):
    fake = OfflineRemote(show=model_metadata("instruct_only"), model=DEFAULT_REMOTE_MODEL)
    fake.tags["models"][0]["digest"] = digest
    with pytest.raises(OllamaModelError) as error:
        await RemoteAgentTextModel(model=DEFAULT_REMOTE_MODEL, transport=httpx.MockTransport(fake))(
            [HumanMessage(content="hello")], tools=[])
    assert error.value.code == "remote_metadata_invalid" and fake.chats == 0


@pytest.mark.asyncio
async def test_instruct_auxiliary_true_does_not_send_thinking_switch():
    metadata = model_metadata("instruct_only")
    metadata["thinking"] = {"values": [True], "default": True}
    fake = OfflineRemote(show=metadata, model=DEFAULT_REMOTE_MODEL)
    answer = await RemoteAgentTextModel(model=DEFAULT_REMOTE_MODEL, transport=httpx.MockTransport(fake))(
        [HumanMessage(content="hello")], tools=[])
    assert answer.content and fake.chats == 1
    assert "reasoning_effort" not in json.loads(fake.requests[-1].content)


@pytest.mark.asyncio
async def test_instruct_api_retains_authentication_ids_images_and_preview_contract(api_context):
    ctx = api_context; ctx.fake.show = model_metadata("instruct_only"); ctx.fake.model = DEFAULT_REMOTE_MODEL
    ctx.fake.tags["models"][0]["name"] = DEFAULT_REMOTE_MODEL
    ctx.fake.tags["models"][0]["digest"] = OFFICIAL_DIGEST
    ctx.remote = RemoteAgentTextModel(model=DEFAULT_REMOTE_MODEL, transport=httpx.MockTransport(ctx.fake))
    from app.services.wardrobe_query_agent import WardrobeQueryAgentModel
    ctx.controller = WardrobeQueryAgentModel(ctx.remote)
    ctx.app.dependency_overrides[api.get_agent_model] = lambda: ctx.controller
    result = AgentPreviewResponse.model_validate_json((await api_request(ctx)).content)
    assert result.status == "success" and result.outfits[0].item_ids == [ctx.item.id]
    assert result.outfits[0].items[0].name == ctx.item.name
    assert ctx.service.get_list.await_args.kwargs["user_id"] == ctx.user.id
    assert ctx.db.mock_calls == [] and ctx.fake.chats == 1
@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 307, 308, 404, 500])
async def test_show_failure_cannot_be_ignored_or_redirected(status):
    fake = OfflineRemote(show_status=status)
    model = RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))
    with pytest.raises(OllamaModelError) as error:
        await model.check_ready()
    assert error.value.code == "thinking_metadata_unavailable"
    assert fake.chats == 0 and len(fake.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,code", [("show_blocked", "model_metadata_blocked"), ("show_timeout", "timeout")])
async def test_metadata_runtime_guard_or_timeout_prevents_generation(failure, code):
    fake = OfflineRemote(error=failure)
    with pytest.raises(OllamaModelError) as error:
        await RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))(
            [HumanMessage(content="hello")], tools=[])
    assert error.value.code == code and fake.chats == 0
    assert "private" not in str(error.value)


@pytest.mark.asyncio
async def test_thinking_only_status_and_preview_do_not_pretend_success(api_context, caplog):
    ctx = api_context
    ctx.fake.show = model_metadata("thinking_only")
    response = await api_request(ctx, status=True)
    state = AgentRuntimeStatus.model_validate_json(response.content)
    assert not state.available and state.tunnel_status == "connected"
    assert state.gpu_verified == "unknown" and ctx.fake.chats == 0
    result = AgentPreviewResponse.model_validate_json((await api_request(ctx)).content)
    assert result.status == "invalid_result" and result.outfits == []
    assert "不支持关闭思考" in result.answer
    entry = next(json.loads(r.message) for r in caplog.records if "agent_preview_audit" in r.message)
    assert entry["error"] == "thinking_disable_unsupported" and entry["model_request_count"] == 0
    assert ctx.fake.chats == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,code", [("length", "response_truncated"), ("reasoning", "thinking_not_disabled")])
async def test_truncation_or_unexpected_reasoning_has_safe_audit_code(failure, code):
    answer = completion("private final", finish="length" if failure == "length" else "stop")
    if failure == "reasoning":
        answer["choices"][0]["message"]["reasoning"] = "private reasoning"
    fake = OfflineRemote(answer=answer)
    audit = AgentAudit(); token = current_audit.set(audit)
    try:
        with pytest.raises(OllamaModelError) as error:
            await RemoteAgentTextModel(model="qwen3:4b", transport=httpx.MockTransport(fake))(
                [HumanMessage(content="private prompt")], tools=[])
    finally:
        current_audit.reset(token)
    value = audit.snapshot()
    assert error.value.code == code and value["error"] == code
    assert value["models"][0]["error"] == code and value["model_request_count"] == 1
    assert value["models"][0]["finish_reason"] == ("length" if failure == "length" else "stop")
    assert "private" not in json.dumps(value) and fake.chats == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,expected", [("length", "截断"), ("reasoning", "思考内容")])
async def test_api_returns_explicit_failure_without_accepting_json_or_saving(api_context, failure, expected):
    ctx = api_context
    ctx.fake.answer = completion('{"answer":"private answer"}', finish="length" if failure == "length" else "stop")
    if failure == "reasoning":
        ctx.fake.answer["choices"][0]["message"]["reasoning"] = "private reasoning"
    result = AgentPreviewResponse.model_validate_json((await api_request(ctx)).content)
    assert result.status == "invalid_result" and result.outfits == []
    assert expected in result.answer and "private" not in result.answer
    assert ctx.fake.chats == 1
    assert ctx.db.mock_calls == []


