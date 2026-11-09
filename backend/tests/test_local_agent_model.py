"""正式本地文本入口隔离测试；全部请求由 MockTransport 处理。"""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from app.services import local_agent_model as local
from app.services.ollama_agent_model import OllamaModelError

pytestmark = pytest.mark.asyncio
_REAL_CHECK_CLOUD = local._check_cloud_disabled
_REAL_CHECK_RESOURCES = local._check_resources


async def test_audit_records_failed_post_generation_cpu_check_without_claiming_success():
    from app.services.agent_audit import AgentAudit, current_audit

    audit = AgentAudit()
    token = current_audit.set(audit)
    try:
        offline = OfflineOllama(after=[running_model(size_vram=123)])
        with pytest.raises(OllamaModelError) as error:
            await local.LocalAgentTextModel(transport=httpx.MockTransport(offline))(
                [HumanMessage(content="查询事实")], tools=[],
            )
        assert error.value.code == "cpu_runtime_unconfirmed"
        record = audit.snapshot()
        assert record["cpu_only_verified"] is False
        assert record["models"][0]["cpu_only_verified"] is False
        assert record["models"][0]["gpu_vram_bytes"] == 123
        assert record["schema_passed"] == "unavailable" and offline.chats == 1
    finally:
        current_audit.reset(token)


def completion(content='{"answer":"本次衬衫查询已完成。"}', *, finish="stop", calls=None):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": finish}]}


def running_model(**changes):
    return {
        "name": local.LOCAL_MODEL, "size_vram": 0, "context_length": 2048, **changes,
    }


class OfflineOllama:
    def __init__(self, *, parameters=None, before=None, after=None, reply=None):
        self.parameters = (
            "num_gpu 0\nnum_ctx 2048\nnum_predict 128"
            if parameters is None else parameters
        )
        self.before = [] if before is None else before
        self.after = [running_model()] if after is None else after
        self.reply = completion() if reply is None else reply
        self.requests = []
        self.chats = 0

    def __call__(self, request):
        assert request.url.host == "127.0.0.1" and request.url.port == 11434
        assert request.url.scheme == "http" and "authorization" not in request.headers
        self.requests.append(request)
        if request.url.path == "/api/show":
            assert request.method == "POST"
            assert json.loads(request.content) == {"model": local.LOCAL_MODEL}
            return httpx.Response(200, json={"parameters": self.parameters})
        if request.url.path == "/api/ps":
            assert request.method == "GET"
            return httpx.Response(200, json={"models": self.after if self.chats else self.before})
        assert request.url.path == "/v1/chat/completions"
        self.chats += 1
        body = json.loads(request.content)
        assert body["model"] == local.LOCAL_MODEL
        assert body["tools"] == []
        assert body["max_tokens"] == 256 and body["reasoning_effort"] == "none"
        assert body["stream"] is False
        assert request.extensions["timeout"]["read"] == 55.0
        return httpx.Response(200, json=self.reply)


@pytest.fixture(autouse=True)
def isolated_local_checks(monkeypatch):
    monkeypatch.setattr(local, "_check_cloud_disabled", lambda: None)
    monkeypatch.setattr(local, "_check_resources", lambda: None)


async def test_constructing_dependency_does_not_read_files_or_connect(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("构造时不应读取环境或发起请求")

    monkeypatch.setattr(local.Path, "read_text", forbidden)
    monkeypatch.setattr(local.httpx, "AsyncClient", forbidden)
    model = local.LocalAgentTextModel()
    assert model._adapter.model == local.LOCAL_MODEL
    assert model._adapter.max_tokens == 256
    assert model._adapter.disable_thinking is True


async def test_fixed_local_request_and_cpu_metadata_without_proxy_or_redirect(monkeypatch):
    offline = OfflineOllama()
    original_client = httpx.AsyncClient
    options = []

    def client(**kwargs):
        options.append(kwargs)
        return original_client(**kwargs)

    monkeypatch.setattr(local.httpx, "AsyncClient", client)
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "AI_BASE_URL", "OPENAI_BASE_URL"]:
        monkeypatch.setenv(key, "https://external.invalid")
    answer = await local.LocalAgentTextModel(transport=httpx.MockTransport(offline))(
        [HumanMessage(content="查询事实")], tools=[],
    )
    assert answer.content == json.loads(json.dumps(offline.reply))["choices"][0]["message"]["content"]
    assert offline.chats == 1
    assert [request.url.path for request in offline.requests] == [
        "/api/show", "/api/ps", "/v1/chat/completions", "/api/ps",
    ]
    assert all(item["trust_env"] is False and item["follow_redirects"] is False for item in options)


@pytest.mark.parametrize("kwargs", [
    {"model": "external-model"},
    {"base_url": "https://external.invalid"},
    {"timeout_seconds": 120},
    {"max_tokens": 4096},
    {"disable_thinking": False},
    {"transport": object()},
])
async def test_untrusted_configuration_cannot_change_fixed_runtime(kwargs):
    with pytest.raises((TypeError, ValueError)):
        local.LocalAgentTextModel(**kwargs)


@pytest.mark.parametrize("tools", [[{"type": "function"}], None, {}])
async def test_text_runtime_cannot_expose_executable_tools(tools):
    offline = OfflineOllama()
    with pytest.raises(ValueError):
        await local.LocalAgentTextModel(transport=httpx.MockTransport(offline))(
            [HumanMessage(content="查询事实")], tools=tools,
        )
    assert offline.requests == []


@pytest.mark.parametrize("parameters", [
    "num_gpu 1\nnum_ctx 2048", "num_ctx 2048",
    "num_gpu 0\nnum_ctx 4096", "num_gpu 0\nnum_gpu 0\nnum_ctx 2048",
    "num_gpu nope\nnum_ctx 2048", "num_gpu 0 injected\nnum_ctx 2048",
])
async def test_unconfirmed_cpu_alias_stops_before_generation(parameters):
    offline = OfflineOllama(parameters=parameters)
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(offline))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == "cpu_config_unconfirmed"
    assert offline.chats == 0


@pytest.mark.parametrize("models,code", [
    ([running_model(size_vram=4096)], "cpu_runtime_unconfirmed"),
    ([running_model(size_vram=False)], "cpu_runtime_unconfirmed"),
    ([running_model(context_length=4096)], "cpu_runtime_unconfirmed"),
    ([running_model(name="other:latest")], "local_model_busy"),
    ([running_model(), running_model()], "local_model_busy"),
    (None, "cpu_runtime_unconfirmed"),
])
async def test_loaded_gpu_or_other_model_stops_without_interference(models, code):
    offline = OfflineOllama()
    offline.before = models
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(offline))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == code
    assert offline.chats == 0
    assert all(request.url.path in {"/api/show", "/api/ps"} for request in offline.requests)


@pytest.mark.parametrize("models", [
    [], [running_model(size_vram=1)], [running_model(context_length=1024)],
])
async def test_response_is_rejected_when_cpu_runtime_cannot_be_verified(models):
    offline = OfflineOllama(after=models)
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(offline))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == "cpu_runtime_unconfirmed"
    assert offline.chats == 1


@pytest.mark.parametrize("status", [302, 404, 500])
async def test_metadata_failures_are_sanitized_without_redirect_or_retry(status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status, headers={"location": "https://external.invalid"},
            json={"error": "private-server-detail"},
        )

    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(handler))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == "local_metadata_unavailable"
    assert "private" not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize("reply", [
    completion(None), completion(" "), completion("partial", finish="length"),
    completion("ok", calls=[{
        "id": "untrusted", "type": "function",
        "function": {"name": "query_my_wardrobe_items", "arguments": "{}"},
    }], finish="tool_calls"),
])
async def test_existing_strict_adapter_validation_is_preserved(reply):
    offline = OfflineOllama(reply=reply)
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(offline))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    expected = "response_truncated" if reply["choices"][0]["finish_reason"] == "length" else "invalid_response"
    assert error.value.code == expected
    assert offline.chats == 1
    assert offline.requests[-1].url.path == "/api/ps"


@pytest.mark.parametrize("exception,code", [
    (httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "connection_failed"),
])
async def test_model_timeout_or_connection_failure_has_no_retry(exception, code):
    offline = OfflineOllama()

    def handler(request):
        if request.url.path == "/v1/chat/completions":
            offline.requests.append(request)
            offline.chats += 1
            raise exception("private-error")
        return offline(request)

    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(handler))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == code
    assert "private" not in str(error.value)
    assert offline.chats == 1


async def test_resource_failure_prevents_any_http(monkeypatch):
    offline = OfflineOllama()

    def low():
        raise local._fail("wsl_memory_low")

    monkeypatch.setattr(local, "_check_resources", low)
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(offline))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == "wsl_memory_low"
    assert offline.requests == []


async def test_resource_failure_during_request_cancels_without_retry(monkeypatch):
    offline = OfflineOllama()
    cancelled = []

    async def handler(request):
        if request.url.path != "/v1/chat/completions":
            return offline(request)
        offline.requests.append(request)
        offline.chats += 1
        try:
            await asyncio.sleep(2)
        finally:
            cancelled.append(True)

    def resources():
        if offline.chats:
            raise local._fail("wsl_memory_pressure")

    monkeypatch.setattr(local, "_check_resources", resources)
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(handler))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == "wsl_memory_pressure"
    assert offline.chats == len(cancelled) == 1


async def test_outer_deadline_cancels_slow_mock_transport(monkeypatch):
    offline = OfflineOllama()
    cancelled = []

    async def handler(request):
        if request.url.path != "/v1/chat/completions":
            return offline(request)
        offline.requests.append(request)
        offline.chats += 1
        try:
            await asyncio.sleep(2)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(local, "MODEL_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(handler))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == "timeout"
    assert offline.chats == len(cancelled) == 1


async def test_concurrent_generation_is_rejected_instead_of_loading_twice():
    offline = OfflineOllama()
    assert local._REQUEST_GATE.acquire(blocking=False)
    try:
        with pytest.raises(OllamaModelError) as error:
            await local.LocalAgentTextModel(transport=httpx.MockTransport(offline))(
                [HumanMessage(content="查询事实")], tools=[],
            )
        assert error.value.code == "local_model_busy"
        assert offline.requests == []
    finally:
        local._REQUEST_GATE.release()


async def test_external_cancellation_does_not_leave_generation_running():
    offline = OfflineOllama()
    started = asyncio.Event()
    cancelled = []

    async def handler(request):
        if request.url.path != "/v1/chat/completions":
            return offline(request)
        offline.requests.append(request)
        offline.chats += 1
        started.set()
        try:
            await asyncio.sleep(2)
        finally:
            cancelled.append(True)

    model = local.LocalAgentTextModel(transport=httpx.MockTransport(handler))
    task = asyncio.create_task(model([HumanMessage(content="查询事实")], tools=[]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled == [True]
    assert not local._REQUEST_GATE.locked()


@pytest.mark.parametrize("raw,code", [
    ('{"disable_ollama_cloud":true}', None),
    ('{"disable_ollama_cloud":false}', "cloud_not_disabled"),
    ('{"disable_ollama_cloud":"true"}', "cloud_not_disabled"),
    ('{}', "cloud_not_disabled"), ("not-json", "cloud_not_disabled"),
])
async def test_cloud_disabled_configuration_requires_explicit_true(monkeypatch, raw, code):
    monkeypatch.setattr(local.Path, "read_text", lambda *args, **kwargs: raw)
    if code:
        with pytest.raises(OllamaModelError) as error:
            _REAL_CHECK_CLOUD()
        assert error.value.code == code
    else:
        _REAL_CHECK_CLOUD()


@pytest.mark.parametrize("available,full,code", [
    (1310720, "1.00", None), (1310719, "0.00", "wsl_memory_low"),
    (2000000, "1.01", "wsl_memory_pressure"), (2000000, "0.00", None),
])
async def test_original_wsl_resource_thresholds_are_retained(monkeypatch, available, full, code):
    def read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return f"MemAvailable: {available} kB\n"
        assert str(path) == "/proc/pressure/memory"
        return f"some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10={full} avg60=0.00\n"

    monkeypatch.setattr(local.Path, "read_text", read)
    monkeypatch.setattr(local.Path, "is_file", lambda *args: True)
    if code:
        with pytest.raises(OllamaModelError) as error:
            _REAL_CHECK_RESOURCES()
        assert error.value.code == code
    else:
        _REAL_CHECK_RESOURCES()


async def test_total_deadline_includes_hanging_metadata(monkeypatch):
    requests = []
    cancelled = []

    async def handler(request):
        requests.append(request)
        assert request.url.path == "/api/show"
        try:
            await asyncio.sleep(2)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(local, "MODEL_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(handler))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == "timeout"
    assert len(requests) == len(cancelled) == 1
    assert not local._REQUEST_GATE.locked()


async def test_total_deadline_includes_post_request_cpu_check(monkeypatch):
    offline = OfflineOllama()
    cancelled = []

    async def handler(request):
        if request.url.path == "/api/ps" and offline.chats:
            offline.requests.append(request)
            try:
                await asyncio.sleep(2)
            finally:
                cancelled.append(True)
        return offline(request)

    monkeypatch.setattr(local, "MODEL_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(handler))(
            [HumanMessage(content="查询事实")], tools=[],
        )
    assert error.value.code == "timeout"
    assert offline.chats == len(cancelled) == 1
    assert len(offline.requests) == 4
    assert not local._REQUEST_GATE.locked()


async def test_readiness_status_only_uses_metadata_and_does_not_load_model():
    offline = OfflineOllama()
    await local.LocalAgentTextModel(transport=httpx.MockTransport(offline)).check_ready()
    assert offline.chats == 0
    assert [request.url.path for request in offline.requests] == ["/api/show", "/api/ps"]


async def test_readiness_status_refuses_gpu_config_without_generation():
    offline = OfflineOllama(parameters="num_gpu 1\nnum_ctx 2048")
    with pytest.raises(OllamaModelError) as error:
        await local.LocalAgentTextModel(transport=httpx.MockTransport(offline)).check_ready()
    assert error.value.code == "cpu_config_unconfirmed" and offline.chats == 0
