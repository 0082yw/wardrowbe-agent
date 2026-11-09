"""Agent API 隔离测试；--noconftest，不使用建库、迁移、Redis 或真实模型夹具。"""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

import app.database as database
from app import agent_tools
from app.api import agent as agent_api
from app.database import get_db
from app.models.item import ItemStatus
from app.schemas.agent import AgentPreviewResponse
from app.services.agent_service import AgentPreview
from app.services.item_service import ItemService
from app.services.mock_agent_model import mock_agent_model
from app.utils.auth import get_current_user

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def context(monkeypatch):
    db = Mock(spec=database.AsyncSession)
    user = SimpleNamespace(id=uuid4(), family_id=None)
    other = SimpleNamespace(id=uuid4(), family_id=None)
    items = [
        SimpleNamespace(
            id=uuid4(), user_id=user.id, type="shirt", name="真实查询的衬衫",
            status=ItemStatus.ready, needs_wash=False, is_archived=False,
            image_path="/internal/private.jpg",
        ),
        SimpleNamespace(
            id=uuid4(), user_id=user.id, type="pants", name="本用户未查询的长裤",
            status=ItemStatus.ready, needs_wash=False, is_archived=False,
        ),
        SimpleNamespace(
            id=uuid4(), user_id=other.id, type="shirt", name="另一个账号的衬衫",
            status=ItemStatus.ready, needs_wash=False, is_archived=False,
        ),
    ]
    ctx = SimpleNamespace(
        db=db, user=user, other=other, items=items, real_transaction=False
    )

    async def lookup(*, user_id, filters, page, page_size):
        matches = [
            item for item in ctx.items
            if item.user_id == user_id and item.type == filters.type and not item.is_archived
        ]
        offset = (page - 1) * page_size
        return matches[offset : offset + page_size], len(matches)

    async def by_id(item_id, *, user_id):
        return next(
            (item for item in ctx.items if item.id == item_id and item.user_id == user_id),
            None,
        )

    ctx.service = Mock(spec=ItemService)
    ctx.service.get_list = AsyncMock(side_effect=lookup)
    ctx.service.get_by_id = AsyncMock(side_effect=by_id)
    ctx.constructor = Mock(return_value=ctx.service)
    monkeypatch.setattr(agent_tools, "ItemService", ctx.constructor)
    monkeypatch.setattr(agent_api, "ItemService", ctx.constructor)
    from app.utils import signed_urls

    monkeypatch.setattr(
        signed_urls, "get_settings", lambda: SimpleNamespace(secret_key="isolated-image-test-key")
    )
    ctx.limit = AsyncMock()
    monkeypatch.setattr(agent_api, "rate_limit_by_user", ctx.limit)

    async def authenticated_user():
        return ctx.user

    async def mock_db():
        yield db

    ctx.app = FastAPI()
    ctx.app.include_router(agent_api.router, prefix="/api/v1")
    ctx.app.dependency_overrides[get_current_user] = authenticated_user
    ctx.app.dependency_overrides[get_db] = mock_db
    # 旧规则模型回归显式注入；新默认本机控制器由下方独立用例覆盖。
    ctx.app.dependency_overrides[agent_api.get_agent_model] = lambda: mock_agent_model
    # ASGITransport 只在进程内分发请求；不打开监听端口、不触发生命周期启动。
    async with AsyncClient(
        transport=ASGITransport(app=ctx.app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        ctx.client = client
        yield ctx

    db.execute.assert_not_awaited()
    db.add.assert_not_called()
    db.add_all.assert_not_called()
    db.flush.assert_not_awaited()
    db.delete.assert_not_awaited()
    if not ctx.real_transaction:
        assert db.mock_calls == [], "路由不得自行执行 SQL、保存、提交或关闭 Session"


async def post(context, payload=None):
    return await context.client.post(
        "/api/v1/agent/preview",
        json={"message": "用我的衬衫生成预览"} if payload is None else payload,
    )


def model_for(context, selected_ids, *, extra=None, query_parameters=None):
    payload = AgentPreview(
        status="success",
        answer="模拟模型的回答。",
        outfits=[{
            "item_ids": selected_ids,
            "reasoning": "基于工具结果。",
            "style_notes": "模拟提示。",
        }],
    ).model_dump(mode="json")
    if extra:
        payload["outfits"][0].update(extra)
    model = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "query_my_wardrobe_items",
                    "args": {"item_type": "shirt", **(query_parameters or {})},
                    "id": "api_query",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content=json.dumps(payload)),
        ]
    )
    context.app.dependency_overrides[agent_api.get_agent_model] = lambda: model
    return model


async def test_valid_request_uses_real_workflow_and_trusted_display(context):
    response = await post(context)
    assert response.status_code == 200
    result = AgentPreviewResponse.model_validate_json(response.content)
    assert result.status == "success"
    assert "模拟" in result.answer
    assert result.outfits[0].item_ids == [context.items[0].id]
    assert result.outfits[0].items[0].name == "真实查询的衬衫"
    assert result.outfits[0].items[0].type == "shirt"
    assert result.outfits[0].reasoning
    assert result.outfits[0].style_notes
    assert "/internal/" not in response.text
    assert "user_id" not in response.text
    assert set(response.json()["outfits"][0]) == {
        "item_ids", "items", "reasoning", "style_notes"
    }
    context.service.get_list.assert_awaited_once()
    context.service.get_by_id.assert_awaited_once_with(
        context.items[0].id, user_id=context.user.id
    )
    assert context.service.get_list.await_args.kwargs["user_id"] == context.user.id
    assert all(entry.args == (context.db,) for entry in context.constructor.call_args_list)
    context.limit.assert_awaited_once_with(
        context.user.id, "agent_preview", max_requests=10, window_seconds=60
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": " \n\t "},
        {"message": "\u200b"},
        {"message": None},
        {"message": 123},
        {"message": "衣" * 10001},
        {"message": "衬衫", "user_id": "other"},
        {"message": "衬衫", "session": "untrusted"},
        {"message": "衬衫", "endpoint": "untrusted"},
    ],
)
async def test_invalid_input_is_422_without_querying_items(context, payload):
    response = await post(context, payload)
    assert response.status_code == 422
    context.service.get_list.assert_not_awaited()
    context.service.get_by_id.assert_not_awaited()


async def test_missing_auth_uses_original_dependency_and_returns_401(context):
    context.app.dependency_overrides.pop(get_current_user)
    response = await post(context)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    context.limit.assert_not_awaited()
    context.service.get_list.assert_not_awaited()
    context.service.get_by_id.assert_not_awaited()


async def test_mock_model_clarifies_other_natural_language_requests(context):
    response = await post(context, {"message": "周末徒步，想穿得舒服一点。"})
    assert response.status_code == 200
    assert response.json()["status"] == "needs_clarification"
    assert response.json()["outfits"] == []
    assert "模拟" in response.json()["answer"]
    context.service.get_list.assert_not_awaited()
    context.service.get_by_id.assert_not_awaited()


@pytest.mark.parametrize("kind", ["empty", "needs_wash", "processing"])
async def test_empty_or_unusable_query_does_not_hydrate_or_save(context, kind):
    if kind == "empty":
        context.items = []
    elif kind == "needs_wash":
        context.items[0].needs_wash = True
    else:
        context.items[0].status = ItemStatus.processing
    response = await post(context)
    assert response.status_code == 200
    assert response.json()["status"] == "no_items"
    assert response.json()["outfits"] == []
    context.service.get_list.assert_awaited_once()
    context.service.get_by_id.assert_not_awaited()


@pytest.mark.parametrize("kind", ["unknown", "unqueried", "other_user"])
async def test_forged_or_unconfirmed_ids_are_rejected_before_display_queries(context, kind):
    selected = {
        "unknown": uuid4(),
        "unqueried": context.items[1].id,
        "other_user": context.items[2].id,
    }[kind]
    model = model_for(context, [selected])
    response = await post(context)
    assert response.status_code == 200
    assert response.json()["status"] == "invalid_result"
    assert response.json()["outfits"] == []
    assert str(selected) not in response.text
    assert model.await_count == 2
    context.service.get_list.assert_awaited_once()
    context.service.get_by_id.assert_not_awaited()


async def test_model_cannot_supply_its_own_display_metadata(context):
    model_for(
        context, [context.items[0].id],
        extra={"items": [{"id": str(context.items[0].id), "name": "forged", "type": "hat"}]},
    )
    response = await post(context)
    assert response.status_code == 200
    assert response.json()["status"] == "invalid_result"
    assert "forged" not in response.text
    context.service.get_by_id.assert_not_awaited()


async def test_model_cannot_override_bound_identity(context):
    model = model_for(
        context, [context.items[0].id], query_parameters={"user_id": str(context.other.id)}
    )
    response = await post(context)
    assert response.status_code == 200
    assert response.json()["status"] == "invalid_result"
    assert model.await_count == 1
    context.service.get_list.assert_not_awaited()
    context.service.get_by_id.assert_not_awaited()


@pytest.mark.parametrize("kind", ["missing", "wrong_owner", "wrong_id", "archived", "wash", "processing"])
async def test_display_query_rechecks_permissions_and_current_item_state(context, kind):
    original_lookup = context.service.get_by_id.side_effect

    async def changed_item(item_id, *, user_id):
        item = await original_lookup(item_id, user_id=user_id)
        if kind == "missing":
            return None
        item = SimpleNamespace(**vars(item))
        if kind == "wrong_owner":
            item.user_id = context.other.id
        elif kind == "wrong_id":
            item.id = uuid4()
        elif kind == "archived":
            item.is_archived = True
        elif kind == "wash":
            item.needs_wash = True
        else:
            item.status = ItemStatus.processing
        return item

    context.service.get_by_id.side_effect = changed_item
    response = await post(context)
    assert response.status_code == 200
    assert response.json()["status"] == "invalid_result"
    assert response.json()["outfits"] == []
    context.service.get_by_id.assert_awaited_once_with(
        context.items[0].id, user_id=context.user.id
    )


@pytest.mark.parametrize("source", ["service", "display", "invalid_display"])
async def test_operational_failures_are_503_and_do_not_leak_details(context, source):
    if source == "service":
        fake = Mock()
        fake.generate_preview = AsyncMock(side_effect=RuntimeError("/internal/private"))
        context.app.dependency_overrides[agent_api.get_agent_service] = lambda: fake
    elif source == "display":
        context.service.get_by_id.side_effect = RuntimeError("/internal/private")
    else:
        context.items[0].name = "衣" * 101
    response = await post(context)
    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "AGENT_PREVIEW_UNAVAILABLE"
    assert "/internal/" not in response.text


async def test_rate_limit_429_is_not_converted_to_503(context):
    context.limit.side_effect = HTTPException(status_code=429, detail="Too many requests")
    response = await post(context)
    assert response.status_code == 429
    context.service.get_list.assert_not_awaited()
    context.service.get_by_id.assert_not_awaited()


async def test_each_user_queries_only_their_own_items(context):
    first_user = context.user
    first = await post(context)
    context.user = context.other
    second = await post(context)
    assert first.status_code == second.status_code == 200
    assert first.json()["outfits"][0]["item_ids"] == [str(context.items[0].id)]
    assert second.json()["outfits"][0]["item_ids"] == [str(context.items[2].id)]
    assert str(context.items[0].id) not in second.text
    assert [
        entry.kwargs["user_id"] for entry in context.service.get_list.await_args_list
    ] == [first_user.id, context.other.id]
    assert [
        entry.kwargs["user_id"] for entry in context.service.get_by_id.await_args_list
    ] == [first_user.id, context.other.id]


@pytest.mark.parametrize("fail", [False, True])
async def test_original_db_session_dependency_manages_read_only_transaction(context, monkeypatch, fail):
    @asynccontextmanager
    async def session_maker():
        yield context.db

    monkeypatch.setattr(database, "async_session_maker", session_maker)
    context.app.dependency_overrides.pop(get_db)
    context.real_transaction = True
    if fail:
        context.service.get_by_id.side_effect = RuntimeError("private failure")
    response = await post(context)
    assert response.status_code == (503 if fail else 200)
    assert context.db.commit.await_count == int(not fail)
    assert context.db.rollback.await_count == int(fail)
    context.db.close.assert_awaited_once()
    context.db.execute.assert_not_awaited()
    context.db.add.assert_not_called()
    context.db.flush.assert_not_awaited()


async def test_preview_signs_only_the_queried_owners_thumbnail(context):
    from urllib.parse import parse_qs, urlsplit

    from app.utils.signed_urls import verify_signature

    item = context.items[0]
    item.thumbnail_path = f"{context.user.id}/shirt_thumb.jpg"
    item.image_path = f"{context.user.id}/shirt.jpg"
    response = await post(context)
    assert response.status_code == 200
    url = response.json()["outfits"][0]["items"][0]["thumbnail_url"]
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert parts.path == f"/api/v1/images/{context.user.id}/shirt_thumb.jpg"
    assert verify_signature(
        parts.path.removeprefix("/api/v1/images/"), query["expires"][0], query["sig"][0]
    )
    assert not verify_signature(
        f"{context.other.id}/shirt_thumb.jpg", query["expires"][0], query["sig"][0]
    )
    assert "image_path" not in response.text
    assert "thumbnail_path" not in response.text
    context.service.get_by_id.assert_awaited_once_with(item.id, user_id=context.user.id)


@pytest.mark.parametrize("kind", ["missing", "processing", "absolute", "external", "other_user", "traversal"])
async def test_unavailable_or_unsafe_image_does_not_invalidate_text_preview(context, kind):
    item = context.items[0]
    item.thumbnail_path = f"{context.user.id}/shirt_thumb.jpg"
    item.image_path = f"{context.user.id}/shirt.jpg"
    if kind == "processing":
        item.processing_kind = "rotate"
    elif kind == "missing":
        item.thumbnail_path = item.image_path = None
    else:
        paths = {
            "absolute": "/internal/private.jpg",
            "external": "https://example.com/private.jpg",
            "other_user": f"{context.other.id}/private.jpg",
            "traversal": f"{context.user.id}/../private.jpg",
        }
        item.thumbnail_path = item.image_path = paths[kind]
    response = await post(context)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["outfits"][0]["items"][0]["thumbnail_url"] is None
    assert result["outfits"][0]["items"][0]["name"] == item.name
    assert "/internal/" not in response.text
    assert str(context.other.id) not in response.text


async def test_missing_thumbnail_falls_back_to_the_owners_original_image(context):
    context.items[0].image_path = f"{context.user.id}/shirt.jpg"
    response = await post(context)
    assert response.json()["outfits"][0]["items"][0]["thumbnail_url"].startswith(
        f"/api/v1/images/{context.user.id}/shirt.jpg?"
    )


@pytest.mark.parametrize("field", ["items", "thumbnail_url", "image_url"])
async def test_model_image_metadata_cannot_authorize_other_users_images(context, field):
    url = f"/api/v1/images/{context.other.id}/private.jpg?expires=9999999999&sig={'a' * 32}"
    value = (
        [{"id": str(context.items[0].id), "name": "forged", "type": "shirt", "thumbnail_url": url}]
        if field == "items" else url
    )
    model_for(context, [context.items[0].id], extra={field: value})
    response = await post(context)
    assert response.status_code == 200
    assert response.json()["status"] == "invalid_result"
    assert response.json()["outfits"] == []
    assert str(context.other.id) not in response.text
    context.service.get_by_id.assert_not_awaited()


@pytest.mark.parametrize("authenticated", [False, True])
async def test_retargeted_image_signature_cannot_read_another_users_file(context, monkeypatch, authenticated):
    from app.api import images as image_api
    from app.utils.auth import get_current_user_optional
    from app.utils.signed_urls import sign_optional

    context.app.include_router(image_api.router, prefix="/api/v1")
    context.app.dependency_overrides[get_current_user_optional] = (
        lambda: context.user if authenticated else None
    )
    image_service = Mock(side_effect=AssertionError("Unauthorized file must not be opened"))
    monkeypatch.setattr(image_api, "ImageService", image_service)
    url = sign_optional(f"{context.user.id}/shirt.jpg")
    response = await context.client.get(url.replace(str(context.user.id), str(context.other.id)))
    assert response.status_code == 401
    image_service.assert_not_called()


@pytest.fixture
def local_context(context, monkeypatch):
    """保留默认正式控制器，只替换本地HTTP传输和机器资源观测。"""
    from langgraph.prebuilt import ToolNode

    from app.services import local_agent_model

    context.app.dependency_overrides.pop(agent_api.get_agent_model)
    monkeypatch.setattr(local_agent_model, "_check_cloud_disabled", lambda: None)
    monkeypatch.setattr(local_agent_model, "_check_resources", lambda: None)
    context.tool_messages = []
    original = ToolNode.ainvoke

    async def observed(self, *args, **kwargs):
        output = await original(self, *args, **kwargs)
        context.tool_messages.extend(output["messages"])
        return output

    monkeypatch.setattr(ToolNode, "ainvoke", observed)
    install_local_response(context)
    return context


def install_local_response(context, *, content=None, finish="stop", failure=None, usage=None):
    import httpx

    from app.services.local_agent_model import LocalAgentTextModel

    if content is None:
        content = json.dumps({
            "answer": "本次衬衫查询已完成，可以先查看这件衬衫的单品预览。",
            "reasoning": "这件衬衫来自本次查询结果。",
            "style_notes": "当前仅完成衬衫查询；请补充下装需求。",
        }, ensure_ascii=False)
    context.local_requests = []
    context.local_paths = []
    completed = False

    def handler(request):
        nonlocal completed
        context.local_paths.append(request.url.path)
        assert request.url.host == "127.0.0.1" and request.url.port == 11434
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"parameters": "num_gpu 0\nnum_ctx 2048"})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{
                "name": "qwen3:0.6b-cpu", "size_vram": 0, "context_length": 2048,
            }] if completed else []})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        context.local_requests.append(body)
        assert context.service.get_list.await_count >= 1
        assert body["model"] == "qwen3:0.6b-cpu"
        assert body["tools"] == [] and body["max_tokens"] == 256
        assert body["reasoning_effort"] == "none"
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        model_input = json.dumps(body["messages"], ensure_ascii=False)
        assert str(context.user.id) not in model_input
        for candidate in context.items:
            assert str(candidate.id) not in model_input
        if failure == "timeout":
            raise httpx.ReadTimeout("private timeout details", request=request)
        if failure == "connection":
            raise httpx.ConnectError("private connection details", request=request)
        completed = True
        payload = {"choices": [{
            "message": {"role": "assistant", "content": content, "tool_calls": []},
            "finish_reason": finish,
        }]}
        if usage is not None:
            payload["usage"] = usage
        return httpx.Response(200, json=payload)

    context.app.dependency_overrides[agent_api.get_agent_text_model] = lambda: (
        LocalAgentTextModel(transport=httpx.MockTransport(handler))
    )


async def post_local(context, message="请查询我的衬衫"):
    return await post(context, {"message": message})


async def test_local_default_dependencies_construct_without_io_and_are_request_scoped():
    from app.services.local_agent_model import LocalAgentTextModel
    from app.services.wardrobe_query_agent import MAX_QUERY_TYPES, WardrobeQueryAgentModel

    text_model = agent_api.get_agent_text_model()
    assert isinstance(text_model, LocalAgentTextModel)
    first = agent_api.get_agent_model(text_model)
    second = agent_api.get_agent_model(text_model)
    assert isinstance(first, WardrobeQueryAgentModel) and first is not second
    assert first.text_model is text_model and second.text_model is text_model
    service = agent_api.get_agent_service(first)
    assert service.max_model_calls == 2 and service.max_tool_calls == MAX_QUERY_TYPES
    assert first.max_previews == second.max_previews == 3


async def test_local_api_uses_real_graph_bound_query_and_trusted_signed_display(local_context):
    ctx = local_context
    ctx.items[0].thumbnail_path = f"{ctx.user.id}/shirt_thumb.jpg"
    result = await post_local(ctx)
    assert result.status_code == 200
    data = AgentPreviewResponse.model_validate_json(result.content)
    assert data.status == "success"
    assert data.outfits[0].item_ids == [ctx.items[0].id]
    display = data.outfits[0].items[0]
    assert display.id == ctx.items[0].id and display.name == ctx.items[0].name
    assert display.type == "shirt"
    assert display.thumbnail_url.startswith(f"/api/v1/images/{ctx.user.id}/shirt_thumb.jpg?")
    ctx.service.get_list.assert_awaited_once()
    assert ctx.service.get_list.await_args.kwargs["user_id"] == ctx.user.id
    assert ctx.service.get_list.await_args.kwargs["filters"].type == "shirt"
    assert ctx.service.get_by_id.await_count == 2  # 模型前候选核验，模型后展示复核
    assert all(
        call.args == (ctx.items[0].id,) and call.kwargs == {"user_id": ctx.user.id}
        for call in ctx.service.get_by_id.await_args_list
    )
    assert len(ctx.local_requests) == len(ctx.tool_messages) == 1
    assert ctx.tool_messages[0].tool_call_id == "call_backend_wardrobe_47"
    assert ctx.tool_messages[0].name == "query_my_wardrobe_items"
    assert ctx.tool_messages[0].status == "success"
    assert ctx.local_paths.count("/v1/chat/completions") == 1
    assert all(call.args == (ctx.db,) for call in ctx.constructor.call_args_list)
    assert "/internal/" not in result.text and str(ctx.other.id) not in result.text
    assert ctx.db.mock_calls == []


async def test_local_api_observed_fenced_instruction_echo_uses_approved_defaults(local_context):
    from app.services.wardrobe_query_agent import DEFAULT_REASONING, DEFAULT_STYLE_NOTES

    ctx = local_context
    fence = chr(96) * 3
    draft = {
        "answer": "根据您的查询，我为您预览了真实查询的衬衫。",
        "reasoning": "基于您的查询事实，仅说明单品预览，不涉及其他信息。",
        "style_notes": "本次查询范围或后续需要补充的需求",
    }
    install_local_response(ctx, content=fence + "json\n" + json.dumps(draft) + "\n" + fence)
    result = await post_local(ctx)
    data = AgentPreviewResponse.model_validate_json(result.content)
    assert data.status == "success" and "本次查询到 1 件衬衫" in data.answer
    assert data.outfits[0].reasoning == DEFAULT_REASONING
    assert data.outfits[0].style_notes == DEFAULT_STYLE_NOTES
    assert "查询事实" not in data.outfits[0].reasoning
    assert len(ctx.local_requests) == 1


@pytest.mark.parametrize("message", ["查询我的未知类型", "不要查询我的衬衫", "周末徒步怎么搭配？"])
async def test_local_api_unknown_query_clarifies_without_ollama_or_database_query(local_context, message):
    ctx = local_context
    result = await post_local(ctx, message)
    assert result.status_code == 200 and result.json()["status"] == "needs_clarification"
    assert result.json()["outfits"] == [] and ctx.local_paths == []
    ctx.service.get_list.assert_not_awaited()
    ctx.service.get_by_id.assert_not_awaited()


@pytest.mark.parametrize("kind", ["empty", "wash", "processing"])
async def test_local_api_no_items_does_not_call_ollama_or_save(local_context, kind):
    ctx = local_context
    if kind == "empty":
        ctx.items = []
    elif kind == "wash":
        ctx.items[0].needs_wash = True
    else:
        ctx.items[0].status = ItemStatus.processing
    result = await post_local(ctx)
    assert result.status_code == 200 and result.json()["status"] == "no_items"
    assert result.json()["outfits"] == [] and ctx.local_paths == []
    ctx.service.get_list.assert_awaited_once()
    if kind == "empty":
        ctx.service.get_by_id.assert_not_awaited()
    else:
        ctx.service.get_by_id.assert_awaited_once_with(ctx.items[0].id, user_id=ctx.user.id)
    assert ctx.db.mock_calls == []


@pytest.mark.parametrize("kind", ["forged_id", "status", "null", "wrong_type", "too_long", "schema", "malformed", "length"])
async def test_local_api_illegal_model_output_is_rejected_without_final_display(local_context, kind):
    ctx = local_context
    draft = {"answer": "本次衬衫查询已完成。"}
    if kind == "forged_id":
        draft["item_ids"] = [str(ctx.items[2].id)]
    elif kind == "status":
        draft["status"] = "success"
    elif kind == "null":
        draft["answer"] = None
    elif kind == "wrong_type":
        draft["answer"] = 123
    elif kind == "too_long":
        draft["answer"] = "衣" * 1001
    elif kind == "schema":
        draft = {"$defs": {"TextDraft": {"type": "object"}}}
    content = '{"answer":' if kind == "malformed" else json.dumps(draft)
    install_local_response(ctx, content=content, finish="length" if kind == "length" else "stop")
    result = await post_local(ctx)
    assert result.status_code == 200 and result.json()["status"] == "invalid_result"
    assert result.json()["outfits"] == []
    assert len(ctx.local_requests) == len(ctx.tool_messages) == 1
    ctx.service.get_by_id.assert_awaited_once_with(ctx.items[0].id, user_id=ctx.user.id)
    assert str(ctx.items[2].id) not in result.text and ctx.db.mock_calls == []


@pytest.mark.parametrize("failure", ["timeout", "connection"])
async def test_local_api_model_failure_is_sanitized_without_retry(local_context, failure):
    ctx = local_context
    install_local_response(ctx, failure=failure)
    result = await post_local(ctx)
    assert result.status_code == 200 and result.json()["status"] == "invalid_result"
    assert result.json()["outfits"] == [] and len(ctx.local_requests) == 1
    assert "private" not in result.text
    ctx.service.get_by_id.assert_awaited_once_with(ctx.items[0].id, user_id=ctx.user.id)
    assert ctx.db.mock_calls == []


async def test_local_api_current_user_and_candidate_set_are_request_scoped(local_context):
    ctx = local_context
    first_user = ctx.user
    first = await post_local(ctx)
    ctx.user = ctx.other
    second = await post_local(ctx)
    assert first.json()["status"] == second.json()["status"] == "success"
    assert first.json()["outfits"][0]["item_ids"] == [str(ctx.items[0].id)]
    assert second.json()["outfits"][0]["item_ids"] == [str(ctx.items[2].id)]
    assert str(ctx.items[0].id) not in second.text
    assert [call.kwargs["user_id"] for call in ctx.service.get_list.await_args_list] == [
        first_user.id, ctx.other.id,
    ]
    assert [call.kwargs["user_id"] for call in ctx.service.get_by_id.await_args_list] == [
        first_user.id, first_user.id, ctx.other.id, ctx.other.id,
    ]
    assert len(ctx.local_requests) == len(ctx.tool_messages) == 2


async def test_local_api_rechecks_display_ownership_even_after_valid_tool_result(local_context):
    ctx = local_context
    foreign = SimpleNamespace(**vars(ctx.items[0]))
    foreign.user_id = ctx.other.id
    ctx.service.get_by_id.return_value = foreign
    ctx.service.get_by_id.side_effect = None
    result = await post_local(ctx)
    assert result.status_code == 200 and result.json()["status"] == "invalid_result"
    assert result.json()["outfits"] == [] and len(ctx.local_requests) == 0
    assert str(ctx.other.id) not in result.text


async def test_local_api_rechecks_owner_again_after_model_generation(local_context):
    ctx = local_context
    original = ctx.items[0]
    foreign = SimpleNamespace(**vars(original))
    foreign.user_id = ctx.other.id
    ctx.service.get_by_id.side_effect = [original, foreign]
    result = await post_local(ctx)
    assert result.status_code == 200 and result.json()["status"] == "invalid_result"
    assert result.json()["outfits"] == [] and len(ctx.local_requests) == 1
    assert ctx.service.get_by_id.await_count == 2
    assert str(ctx.other.id) not in result.text and ctx.db.mock_calls == []


async def test_local_api_missing_auth_has_no_tool_or_model_side_effects(local_context):
    ctx = local_context
    ctx.app.dependency_overrides.pop(get_current_user)
    result = await post_local(ctx)
    assert result.status_code == 401 and ctx.local_paths == []
    ctx.service.get_list.assert_not_awaited()
    assert ctx.db.mock_calls == []


async def test_runtime_status_checks_metadata_without_model_generation(local_context):
    ctx = local_context
    result = await ctx.client.get("/api/v1/agent/status")
    assert result.status_code == 200
    assert result.json() == {
        "mode": "local_ollama", "model": "qwen3:0.6b-cpu",
        "available": True, "cpu_only": True,
        "execution_mode": "local_cpu", "tunnel_status": "not_applicable", "gpu_verified": False,
    }
    assert ctx.local_paths == ["/api/show", "/api/ps"] and ctx.local_requests == []
    ctx.service.get_list.assert_not_awaited()
    assert ctx.db.mock_calls == []


async def test_runtime_status_does_not_claim_injected_mock_is_real(context):
    context.app.dependency_overrides[agent_api.get_agent_text_model] = lambda: mock_agent_model
    result = await context.client.get("/api/v1/agent/status")
    assert result.status_code == 200
    assert result.json() == {
        "mode": "unknown", "model": None, "available": False, "cpu_only": False,
        "execution_mode": "unknown", "tunnel_status": "unknown", "gpu_verified": "unknown",
    }


async def test_runtime_status_follows_preview_model_override_not_unused_text_dependency(context, monkeypatch):
    from app.services.local_agent_model import LocalAgentTextModel

    readiness = AsyncMock(side_effect=AssertionError("未用于预览的模型不应探测"))
    monkeypatch.setattr(LocalAgentTextModel, "check_ready", readiness)
    # 当前预览显式注入 Mock；未使用的默认文本依赖不能使页面声称已接入真实模型。
    result = await context.client.get("/api/v1/agent/status")
    assert result.status_code == 200 and result.json() == {
        "mode": "unknown", "model": None, "available": False, "cpu_only": False,
        "execution_mode": "unknown", "tunnel_status": "unknown", "gpu_verified": "unknown",
    }
    readiness.assert_not_awaited()


async def test_runtime_status_unavailable_is_explicit_without_generation(local_context, monkeypatch):
    from app.services.local_agent_model import LocalAgentTextModel

    monkeypatch.setattr(LocalAgentTextModel, "check_ready", AsyncMock(side_effect=RuntimeError("private")))
    result = await local_context.client.get("/api/v1/agent/status")
    assert result.status_code == 200
    assert result.json()["available"] is False and result.json()["cpu_only"] is False
    assert "private" not in result.text and local_context.local_requests == []


async def test_runtime_status_requires_authentication_without_model_side_effects(local_context):
    ctx = local_context
    ctx.app.dependency_overrides.pop(get_current_user)
    result = await ctx.client.get("/api/v1/agent/status")
    assert result.status_code == 401 and ctx.local_paths == []


async def test_ollama_unavailable_does_not_masquerade_as_success(local_context):
    ctx = local_context
    install_local_response(ctx, failure="connection")
    result = await post_local(ctx)
    assert result.status_code == 200 and result.json()["status"] == "invalid_result"
    assert "Ollama" in result.json()["answer"] and result.json()["outfits"] == []
    assert len(ctx.local_requests) == 1 and ctx.db.mock_calls == []


def audit_records(caplog):
    return [json.loads(record.getMessage()) for record in caplog.records
            if record.name == "app.services.agent_audit"]


async def test_audit_has_linked_safe_fields_cpu_and_actual_usage(local_context, caplog):
    from app.services.agent_audit import current_audit

    ctx = local_context
    ctx.items[0].thumbnail_path = f"{ctx.user.id}/private_thumb.jpg"
    install_local_response(ctx, usage={
        "prompt_tokens": 173, "completion_tokens": 64, "total_tokens": 237,
        "private_key": "USAGE-PRIVATE",
    })
    response = await ctx.client.post(
        "/api/v1/agent/preview", json={"message": "请查询我的衬衫"},
        headers={"Authorization": "Bearer TOKEN-PRIVATE", "Cookie": "session=COOKIE-PRIVATE",
                 "X-Request-ID": "UNTRUSTED-PRIVATE"},
    )
    assert response.status_code == 200
    records = audit_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert response.headers["X-Request-ID"] == record["request_id"]
    assert record["request_id"] != "UNTRUSTED-PRIVATE"
    assert record["authenticated"] is record["tool_scope_bound"] is True
    assert record["tool_count"] == record["successful_tool_count"] == 1
    assert record["tools"] == [{"type": "shirt", "returned_count": 1,
                                "tool_message_linked": True, "query_executed": True}]
    assert record["executed_tool_count"] == 1
    assert record["candidate_count"] == record["verified_candidate_count"] == record["display_count"] == 1
    assert record["candidate_scope_verified"] is True
    candidate, display = record["candidates"][0], record["display_items"][0]
    assert candidate["id"] == display["candidate_id"] == display["id"] == str(ctx.items[0].id)
    assert candidate["type"] == display["type"] == "shirt"
    assert candidate["status"] == "ready" and candidate["available"] is True
    assert display["name_consistent"] is display["image_associated"] is display["image_consistent"] is True
    assert candidate["image_ref"] == display["image_ref"]
    assert candidate["name_ref"] == display["name_ref"]
    assert record["schema_passed"] is True and record["final_status"] == "success"
    model = record["models"][0]
    assert record["model_request_count"] == 1 and model["transport"] == "mock"
    assert model["model"] == "qwen3:0.6b-cpu" and model["http_status"] == 200
    assert model["input_tokens"] == 173 and model["output_tokens"] == 64 and model["total_tokens"] == 237
    assert model["finish_reason"] == "stop" and model["seconds"] >= 0
    assert model["cpu_only_verified"] is model["cpu_config_verified"] is True
    assert model["gpu_vram_bytes"] == 0
    text = json.dumps(record, ensure_ascii=False)
    for forbidden in ("PRIVATE", ctx.items[0].name, ctx.items[0].thumbnail_path,
                      str(ctx.user.id), str(ctx.other.id), "sig=", "expires=", "ToolMessage"):
        assert forbidden not in text
    assert current_audit.get() is None


async def test_audit_unknown_query_and_empty_results_do_not_invent_model_metadata(local_context, caplog):
    ctx = local_context
    unknown = await post_local(ctx, "请查询我的未知类型")
    assert unknown.json()["status"] == "needs_clarification"
    first = audit_records(caplog)[-1]
    assert first["tool_count"] == first["model_request_count"] == first["display_count"] == 0
    assert first["candidate_count"] == "unavailable" and first["schema_passed"] is True
    ctx.items.clear()
    empty = await post_local(ctx)
    assert empty.json()["status"] == "no_items"
    second = audit_records(caplog)[-1]
    assert second["candidate_count"] == second["display_count"] == second["model_request_count"] == 0
    assert second["tool_count"] == 1 and second["models"] == []
    assert first["request_id"] != second["request_id"]
    assert ctx.local_requests == []


@pytest.mark.parametrize("failure", ["timeout", "connection", "length", "bad_json"])
async def test_failed_model_is_audited_without_retry_or_raw_output(local_context, caplog, failure):
    ctx = local_context
    if failure in ("timeout", "connection"):
        install_local_response(ctx, failure=failure)
    else:
        install_local_response(ctx, content='{"answer":"PRIVATE', finish="length" if failure == "length" else "stop")
    response = await post_local(ctx)
    assert response.status_code == 200 and response.json()["status"] == "invalid_result"
    record = audit_records(caplog)[-1]
    assert record["model_request_count"] == record["tool_count"] == 1
    assert record["schema_passed"] is True and record["display_count"] == 0
    model = record["models"][0]
    expected = {"timeout": "timeout", "connection": "connection_failed",
                "length": "response_truncated"}.get(failure, "invalid_response")
    if failure == "length":
        assert "截断" in response.json()["answer"]
    assert record["error"] == ("validation_failed" if failure == "bad_json" else expected)
    assert model["error"] == ("unavailable" if failure == "bad_json" else expected)
    assert model["input_tokens"] == model["output_tokens"] == "unavailable"
    assert model["cpu_only_verified"] == (True if failure in ("length", "bad_json") else "unavailable")
    if failure == "length":
        assert model["finish_reason"] == "length" and model["http_status"] == 200
    assert len(ctx.local_requests) == 1 and "PRIVATE" not in json.dumps(record)


@pytest.mark.parametrize("kind,code", [("unauthenticated", 401), ("invalid", 422), ("rate", 429), ("service", 503)])
async def test_rejected_api_requests_have_safe_single_audit(context, caplog, kind, code):
    ctx = context
    payload = {"message": "请查询我的衬衫"}
    if kind == "unauthenticated":
        ctx.app.dependency_overrides.pop(get_current_user)
    elif kind == "invalid":
        payload["message"] = " "
    elif kind == "rate":
        ctx.limit.side_effect = HTTPException(429, detail="RATE-PRIVATE")
    else:
        ctx.app.dependency_overrides[agent_api.get_agent_service] = lambda: SimpleNamespace(
            model=mock_agent_model, generate_preview=AsyncMock(side_effect=RuntimeError("ERROR-PRIVATE")),
        )
    response = await post(ctx, payload)
    assert response.status_code == code
    records = audit_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record["http_status"] == code
    assert record["schema_passed"] == "unavailable"
    assert record["model_request_count"] == record["tool_count"] == 0
    assert "PRIVATE" not in json.dumps(record)
    if kind == "unauthenticated":
        assert record["authenticated"] is False


async def test_audit_multi_type_counts_merges_and_links_without_claiming_complete_outfit(local_context, caplog):
    ctx = local_context
    for item in ctx.items[:2]:
        item.thumbnail_path = f"{ctx.user.id}/{item.type}_thumb.jpg"
    response = await post_local(ctx, "查询我的衬衫和裤子")
    assert response.status_code == 200 and response.json()["status"] == "success"
    record = audit_records(caplog)[-1]
    assert [t["type"] for t in record["tools"]] == ["shirt", "pants", "jeans", "shorts"]
    assert record["successful_tool_count"] == record["tool_count"] == 4
    assert record["executed_tool_count"] == 4
    assert record["candidate_count"] == record["display_count"] == 2
    assert {e["id"] for e in record["candidates"]} == {str(i.id) for i in ctx.items[:2]}
    assert all(e["name_consistent"] and e["image_consistent"] for e in record["display_items"])
    assert all(t["tool_message_linked"] for t in record["tools"])
    assert record["model_request_count"] == 1 and len(ctx.local_requests) == 1
    assert "前 2 件单品预览" in response.json()["answer"]
    assert ctx.db.mock_calls == []


async def test_audit_low_quality_fallback_and_absent_image_are_explicit(local_context, caplog):
    ctx = local_context
    install_local_response(ctx, content=json.dumps({
        "answer": "内部 Schema 说明", "reasoning": "请生成有效理由", "style_notes": "properties required",
    }))
    response = await post_local(ctx)
    assert response.json()["status"] == "success"
    record = audit_records(caplog)[-1]
    assert record["fallback_fields"] == ["answer", "reasoning", "style_notes"]
    display = record["display_items"][0]
    assert display["image_associated"] is False and display["image_consistent"] == "unavailable"
    assert record["models"][0]["total_tokens"] == "unavailable"


async def test_audit_candidate_scope_failure_hides_foreign_id(local_context, caplog):
    ctx = local_context
    foreign = ctx.items[-1]
    ctx.service.get_list.side_effect = AsyncMock(return_value=([foreign], 1))
    response = await post_local(ctx)
    assert response.json()["status"] == "invalid_result"
    record = audit_records(caplog)[-1]
    assert record["candidate_scope_verified"] is False and record["candidates"] == []
    assert record["model_request_count"] == 0 and ctx.local_requests == []
    assert str(foreign.id) not in json.dumps(record) and str(ctx.other.id) not in json.dumps(record)


async def test_audit_logging_failure_never_changes_business_or_context(local_context, monkeypatch):
    from app.services.agent_audit import current_audit, logger

    monkeypatch.setattr(logger, "info", Mock(side_effect=RuntimeError("private logging failure")))
    response = await post_local(local_context)
    assert response.status_code == 200 and response.json()["status"] == "success"
    assert current_audit.get() is None


async def test_audit_final_display_change_invalidates_output_without_stale_count(local_context, caplog):
    ctx = local_context
    original = ctx.service.get_by_id.side_effect
    reads = 0

    async def changed(item_id, *, user_id):
        nonlocal reads
        reads += 1
        item = await original(item_id, user_id=user_id)
        if reads == 2:
            return SimpleNamespace(**{**vars(item), "name": "PRIVATE changed name"})
        return item

    ctx.service.get_by_id.side_effect = changed
    response = await post_local(ctx)
    assert response.json()["status"] == "invalid_result"
    record = audit_records(caplog)[-1]
    assert record["error"] == "display_mismatch"
    assert record["schema_passed"] is True and record["display_count"] == 0
    assert record["candidate_count"] == 1 and record["model_request_count"] == 1
    assert "PRIVATE" not in json.dumps(record)


async def test_audit_rejected_final_schema_does_not_claim_validation_passed(context, caplog):
    invalid = AgentPreview.model_construct(status="invalid-status", answer="PRIVATE", outfits=[])
    context.app.dependency_overrides[agent_api.get_agent_service] = lambda: SimpleNamespace(
        model=mock_agent_model, generate_preview=AsyncMock(return_value=invalid),
    )
    response = await post(context)
    assert response.status_code == 503
    record = audit_records(caplog)[-1]
    assert record["schema_passed"] is False and record["final_status"] == "unavailable"
    assert record["display_count"] == "unavailable" and record["model_request_count"] == 0
    assert "PRIVATE" not in json.dumps(record)


async def test_audit_rejected_tool_arguments_never_claim_a_successful_query(context, caplog):
    model_for(context, [context.items[0].id], query_parameters={"user_id": str(context.other.id)})
    response = await post(context)
    assert response.json()["status"] == "invalid_result"
    record = audit_records(caplog)[-1]
    assert record["tool_count"] == 1 and record["successful_tool_count"] == 0
    assert record["executed_tool_count"] == 0
    assert record["candidate_count"] == "unavailable" and record["display_count"] == 0
    context.service.get_list.assert_not_awaited()
    assert str(context.other.id) not in json.dumps(record)


async def test_audit_failed_query_distinguishes_execution_from_success(local_context, caplog):
    ctx = local_context
    ctx.service.get_list.side_effect = RuntimeError("PRIVATE database details")
    response = await post_local(ctx)
    assert response.json()["status"] == "invalid_result"
    record = audit_records(caplog)[-1]
    assert record["executed_tool_count"] == record["tool_count"] == 1
    assert record["successful_tool_count"] == 0 and record["model_request_count"] == 0
    assert record["candidate_count"] == "unavailable"
    assert "PRIVATE" not in json.dumps(record) and ctx.local_requests == []


@pytest.mark.parametrize("message,types", [
    ("请查询我的裤子", {"pants", "jeans", "shorts"}),
    ("请查询我的鞋子", {"shoes", "sneakers", "boots", "sandals"}),
    ("请查询我的外套", {"jacket", "coat", "hoodie", "blazer"}),
])
async def test_api_queries_authoritative_type_groups_and_projects_multiple_verified_items(local_context, message, types):
    ctx = local_context
    ctx.items = [
        SimpleNamespace(
            id=uuid4(), user_id=ctx.user.id, name="当前用户单品", type=kind,
            status=ItemStatus.ready, needs_wash=False, is_archived=False,
        )
        for kind in sorted(types)
    ] + [
        SimpleNamespace(
            id=uuid4(), user_id=ctx.other.id, name="其他用户私有单品", type=next(iter(types)),
            status=ItemStatus.ready, needs_wash=False, is_archived=False,
        )
    ]
    result = await post_local(ctx, message)
    assert result.status_code == 200
    preview = AgentPreviewResponse.model_validate_json(result.content)
    assert preview.status == "success" and 1 < len(preview.outfits) <= 3
    assert {call.kwargs["filters"].type for call in ctx.service.get_list.await_args_list} == types
    assert len(ctx.tool_messages) == len(types)
    assert len({entry.tool_call_id for entry in ctx.tool_messages}) == len(types)
    own_ids = {item.id for item in ctx.items if item.user_id == ctx.user.id}
    selected_ids = [item_id for outfit in preview.outfits for item_id in outfit.item_ids]
    assert set(selected_ids) <= own_ids and len(selected_ids) == len(set(selected_ids))
    assert all(len(outfit.item_ids) == 1 for outfit in preview.outfits)
    assert "其他用户" not in result.text
    assert all(call.kwargs["user_id"] == ctx.user.id for call in ctx.service.get_list.await_args_list)
    assert len(ctx.local_requests) == 1 and ctx.db.mock_calls == []
