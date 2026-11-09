"""正式查询控制器的隔离测试；真实 AgentService/ToolNode，业务查询与 Session 为 Mock。"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app import agent_tools
from app.models.item import ItemStatus
from app.schemas.agent import AgentPreviewResponse
from app.services.agent_service import AgentService
from app.services.wardrobe_query_agent import (
    DEFAULT_REASONING,
    DEFAULT_STYLE_NOTES,
    WardrobeQueryAgentModel,
    default_query_answer,
    query_call,
    read_candidates,
)

pytestmark = pytest.mark.asyncio


def model_text(**changes):
    data = {
        "answer": "本次查询到两件衬衫，先展示第一件作为单品预览。",
        "reasoning": "这件衬衫来自本次查询结果。",
        "style_notes": "如需完整搭配，请补充使用场景或下装需求。",
    }
    data.update(changes)
    return AIMessage(content=json.dumps(data, ensure_ascii=False))


@pytest.fixture
def context(monkeypatch):
    db = Mock(spec=AsyncSession)
    user = SimpleNamespace(id=uuid4())
    items = [
        SimpleNamespace(
            id=uuid4(), name=name, type="shirt", status=ItemStatus.ready, needs_wash=False,
        )
        for name in ("测试白色衬衫", "测试蓝色衬衫")
    ]
    ctx = SimpleNamespace(user=user, db=db, items=items, events=[])

    async def lookup(*, user_id, filters, page, page_size):
        assert user_id == ctx.user.id
        assert filters.type == "shirt" and page == 1 and page_size == 20
        ctx.events.append("query")
        return ctx.items, len(ctx.items)

    ctx.service = Mock(spec=agent_tools.ItemService)
    ctx.service.get_list = AsyncMock(side_effect=lookup)
    ctx.constructor = Mock(return_value=ctx.service)
    monkeypatch.setattr(agent_tools, "ItemService", ctx.constructor)
    yield ctx
    assert db.mock_calls == [], "预览不能直接读写 Session、保存或提交事务"


async def run(context, text_model, message="请查询我的衬衫", **limits):
    controller = WardrobeQueryAgentModel(text_model)
    preview = await AgentService(model=controller, **limits).generate_preview(
        message, current_user=context.user, db=context.db,
    )
    response = AgentPreviewResponse.from_preview(preview, items=controller.display_items)
    return SimpleNamespace(controller=controller, preview=preview, response=response)


async def test_tool_executes_before_text_and_preview_uses_only_query_result(context):
    async def text_model(messages, *, tools):
        context.events.append("text")
        assert context.events == ["query", "text"]
        assert tools == []
        assert [m.type for m in messages] == ["system", "human"]
        data = json.loads(messages[-1].content)
        assert data["item"] == {"name": context.items[0].name, "type": "shirt"}
        for item in context.items:
            assert str(item.id) not in messages[-1].content
        assert str(context.user.id) not in messages[-1].content
        return model_text()

    outcome = await run(context, text_model)
    assert outcome.preview.status == outcome.response.status == "success"
    assert outcome.preview.outfits[0].item_ids == [context.items[0].id]
    assert outcome.response.outfits[0].items[0].name == context.items[0].name
    assert outcome.controller.text_calls == 1
    assert context.service.get_list.await_count == 1
    context.constructor.assert_called_once_with(context.db)
    query, final = outcome.controller.trace.graph_outputs
    results = [
        msg for msg in outcome.controller.trace.graph_inputs[-1] if isinstance(msg, ToolMessage)
    ]
    assert len(query.tool_calls) == len(results) == 1
    assert results[0].tool_call_id == query.tool_calls[0]["id"]
    assert results[0].status == "success" and not final.tool_calls


@pytest.mark.parametrize("message", [
    "你好", "查询我的未知类型", "查询我的上衣", "不要查询我的衬衫", "查询我的衬衫和未知类型",
])
async def test_unknown_query_never_calls_tool_or_model(context, message):
    model = AsyncMock()
    outcome = await run(context, model, message)
    assert outcome.preview.status == "needs_clarification"
    assert outcome.preview.outfits == []
    model.assert_not_awaited()
    context.service.get_list.assert_not_awaited()


@pytest.mark.parametrize("kind", ["empty", "processing", "needs_wash"])
async def test_no_eligible_candidates_returns_no_items_without_model(context, kind):
    if kind == "empty":
        context.items.clear()
    else:
        for item in context.items:
            if kind == "processing":
                item.status = ItemStatus.processing
            else:
                item.needs_wash = True
    model = AsyncMock()
    outcome = await run(context, model)
    assert outcome.preview.status == "no_items" and outcome.preview.outfits == []
    assert "本次衬衫查询" in outcome.preview.answer
    assert context.service.get_list.await_count == 1
    model.assert_not_awaited()


async def test_only_ready_candidate_is_selected_without_changing_list_query(context):
    context.items[0].needs_wash = True
    outcome = await run(context, AsyncMock(return_value=model_text()))
    assert outcome.preview.outfits[0].item_ids == [context.items[1].id]
    assert outcome.controller.display_items[0].name == context.items[1].name
    filters = context.service.get_list.await_args.kwargs["filters"]
    assert filters.needs_wash is None and filters.status is None


@pytest.mark.parametrize("extra", [
    {"status": "success"}, {"status": "wrong"}, {"item_ids": [str(uuid4())]},
    {"outfits": []}, {"items": [{"name": "伪造名称"}]},
    {"user_id": str(uuid4())}, {"db": "injected"},
])
async def test_model_cannot_supply_business_fields_or_cross_user_ids(context, extra):
    model = AsyncMock(return_value=model_text(**extra))
    outcome = await run(context, model)
    assert outcome.preview.status == "invalid_result" and outcome.response.outfits == []
    assert context.service.get_list.await_args.kwargs["user_id"] == context.user.id
    assert model.await_count == outcome.controller.text_calls == 1
    assert context.service.get_list.await_count == 1


@pytest.mark.parametrize("changes", [
    {"answer": "", "reasoning": "", "style_notes": ""},
    {"answer": "测试白色衬衫"},
    {"reasoning": "可省略；提供时必须为非空简短文本"},
    {"style_notes": "properties required additionalProperties"},
])
async def test_low_quality_text_uses_fact_based_defaults_without_retry(context, changes):
    outcome = await run(context, AsyncMock(return_value=model_text(**changes)))
    assert outcome.preview.status == "success" and outcome.controller.text_calls == 1
    expected = {
        "answer": default_query_answer(2),
        "reasoning": DEFAULT_REASONING,
        "style_notes": DEFAULT_STYLE_NOTES,
    }
    for field in changes:
        result = (
            outcome.preview.answer if field == "answer"
            else getattr(outcome.preview.outfits[0], field)
        )
        assert result == expected[field]
        assert field in outcome.controller.trace.text_fallbacks


@pytest.mark.parametrize("changes", [
    {"answer": None}, {"answer": 2}, {"reasoning": None},
    {"reasoning": []}, {"style_notes": 5}, {"style_notes": "x" * 1001},
])
async def test_invalid_text_types_and_lengths_still_reject(context, changes):
    outcome = await run(context, AsyncMock(return_value=model_text(**changes)))
    assert outcome.preview.status == "invalid_result" and outcome.response.outfits == []


async def test_single_json_fence_is_unwrapped_but_schema_is_not_accepted(context):
    good = model_text()
    good.content = "```json\n" + good.content + "\n```"
    outcome = await run(context, AsyncMock(return_value=good))
    assert outcome.preview.status == "success"
    bad = AIMessage(content='{"properties":{"answer":{"type":"string"}}}')
    outcome = await run(context, AsyncMock(return_value=bad))
    assert outcome.preview.status == "invalid_result"


@pytest.mark.parametrize("field", ["reasoning", "style_notes", "both"])
async def test_missing_optional_text_is_completed(context, field):
    data = json.loads(model_text().content)
    for key in ("reasoning", "style_notes") if field == "both" else (field,):
        data.pop(key)
    outcome = await run(
        context, AsyncMock(return_value=AIMessage(content=json.dumps(data))),
    )
    assert outcome.preview.status == "success"
    if field in ("reasoning", "both"):
        assert outcome.preview.outfits[0].reasoning == DEFAULT_REASONING
    if field in ("style_notes", "both"):
        assert outcome.preview.outfits[0].style_notes == DEFAULT_STYLE_NOTES


@pytest.mark.parametrize("kind", ["invalid_uuid", "duplicate", "wrong_type", "wrong_boolean"])
async def test_invalid_tool_candidates_reject_before_model(context, kind):
    if kind == "invalid_uuid":
        context.items[0].id = "not-uuid"
    elif kind == "duplicate":
        context.items[1].id = context.items[0].id
    elif kind == "wrong_type":
        context.items[0].type = "pants"
    else:
        context.items[0].needs_wash = "false"
    model = AsyncMock()
    outcome = await run(context, model)
    assert outcome.preview.status == "invalid_result"
    model.assert_not_awaited()


@pytest.mark.parametrize("change", [
    {"tool_call_id": "another"}, {"name": "another"}, {"status": "error"},
])
async def test_unlinked_tool_message_is_rejected(change):
    call = query_call("shirt")
    result = ToolMessage(
        content=json.dumps({
            "items": [], "total": 0, "page": 1, "page_size": 20, "has_more": False,
        }),
        tool_call_id=call["id"], name=call["name"],
    )
    for field, value in change.items():
        setattr(result, field, value)
    with pytest.raises(ValueError):
        read_candidates(result, call)


async def test_finished_controller_cannot_start_another_request(context):
    model = AsyncMock(return_value=model_text())
    outcome = await run(context, model)
    with pytest.raises(ValueError, match="本请求已结束"):
        await outcome.controller([HumanMessage(content="请查询我的衬衫")], tools=[])
    assert model.await_count == 1 and context.service.get_list.await_count == 1


async def test_text_failure_is_not_retried_with_same_controller(context):
    model = AsyncMock(side_effect=TimeoutError("fake timeout"))
    outcome = await run(context, model)
    assert outcome.preview.status == "invalid_result"
    with pytest.raises(ValueError):
        await outcome.controller(
            outcome.controller.trace.graph_inputs[-1], tools=[],
        )
    assert model.await_count == outcome.controller.text_calls == 1


@pytest.mark.parametrize("limits", [{"max_model_calls": 1}, {"max_tool_calls": 1}])
async def test_service_call_limits_remain_effective(context, limits):
    model = AsyncMock(return_value=model_text())
    outcome = await run(context, model, **limits)
    assert context.service.get_list.await_count == 1
    if limits.get("max_model_calls") == 1:
        assert outcome.preview.status == "invalid_result"
        model.assert_not_awaited()
    else:
        assert outcome.preview.status == "success"
        model.assert_awaited_once()


async def test_request_instances_do_not_share_candidates_or_identity(context):
    first = await run(context, AsyncMock(return_value=model_text()))
    first_id = first.preview.outfits[0].item_ids[0]
    context.user = SimpleNamespace(id=uuid4())
    context.items[0].id = uuid4()
    second = await run(context, AsyncMock(return_value=model_text()))
    assert first.controller is not second.controller
    assert first_id != second.preview.outfits[0].item_ids[0]
    assert context.service.get_list.await_args.kwargs["user_id"] == context.user.id
