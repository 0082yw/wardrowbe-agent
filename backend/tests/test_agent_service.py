"""Agent Service 独立测试；--noconftest 避开数据库和 Redis 夹具，全程使用 Mock。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app import agent_tools
from app.models.item import ItemStatus
from app.services.agent_service import AgentService
from app.services.item_service import ItemService

pytestmark = pytest.mark.asyncio


def tool_request(item_type="shirt", *, call_id="call_1", name="query_my_wardrobe_items", **extra):
    return {
        "name": name,
        "args": {"item_type": item_type, **extra},
        "id": call_id,
        "type": "tool_call",
    }


def request_message(*requests):
    return AIMessage(content="", tool_calls=list(requests))


def final_message(item_ids=None, *, status="success", answer="这套衣物可以作为穿搭预览。"):
    return AIMessage(
        content=json.dumps(
            {
                "status": status,
                "answer": answer,
                "outfits": (
                    [
                        {
                            "item_ids": [str(item_id) for item_id in item_ids],
                            "reasoning": "从已查询衣物中选择。",
                            "style_notes": "请按实际天气调整。",
                        }
                    ]
                    if item_ids is not None
                    else []
                ),
            },
            ensure_ascii=False,
        )
    )


@pytest.fixture
def context(monkeypatch):
    db = Mock(spec=AsyncSession)
    user = SimpleNamespace(id=uuid4())
    items = [
        SimpleNamespace(
            id=uuid4(), name="白色衬衫", type="shirt", status=ItemStatus.ready, needs_wash=False
        ),
        SimpleNamespace(
            id=uuid4(), name="长裤", type="pants", status=ItemStatus.ready, needs_wash=False
        ),
    ]
    ctx = SimpleNamespace(
        db=db, user=user, items=items, events=[], active_queries=0, max_active_queries=0
    )

    async def lookup(*, user_id, filters, page, page_size):
        ctx.active_queries += 1
        ctx.max_active_queries = max(ctx.max_active_queries, ctx.active_queries)
        ctx.events.append(("start", filters.type))
        try:
            await asyncio.sleep(0)
            matches = [item for item in ctx.items if item.type == filters.type]
            offset = (page - 1) * page_size
            return matches[offset : offset + page_size], len(matches)
        finally:
            ctx.events.append(("end", filters.type))
            ctx.active_queries -= 1

    ctx.service = Mock(spec=ItemService)
    ctx.service.get_list = AsyncMock(side_effect=lookup)
    ctx.constructor = Mock(return_value=ctx.service)
    monkeypatch.setattr(agent_tools, "ItemService", ctx.constructor)
    yield ctx
    assert db.mock_calls == [], "Service 不得直接执行数据库操作或管理 Session 生命周期"


async def run(context, model, **limits):
    return await AgentService(model=model, **limits).generate_preview(
        "请用我的衣橱生成穿搭预览。", current_user=context.user, db=context.db
    )


async def test_preview_uses_real_tool_results_and_executes_batch_sequentially(context):
    ids = [item.id for item in context.items]
    model = AsyncMock(
        side_effect=[
            request_message(tool_request(), tool_request("pants", call_id="call_2")),
            final_message(ids),
        ]
    )
    result = await run(context, model)

    assert result.status == "success"
    assert result.outfits[0].item_ids == ids
    assert result.outfits[0].reasoning
    assert result.outfits[0].style_notes
    assert result.model_dump(mode="json")["outfits"][0]["item_ids"] == list(map(str, ids))
    assert model.await_count == 2
    assert context.service.get_list.await_count == 2
    assert context.events == [
        ("start", "shirt"),
        ("end", "shirt"),
        ("start", "pants"),
        ("end", "pants"),
    ]
    assert context.max_active_queries == 1
    context.constructor.assert_called_once_with(context.db)
    for query in context.service.get_list.await_args_list:
        assert query.kwargs["user_id"] == context.user.id
        assert query.kwargs["filters"].is_archived is False
        assert query.kwargs["filters"].needs_wash is None

    messages = model.await_args_list[1].args[0]
    assert [entry.type for entry in messages] == ["system", "human", "ai", "tool", "tool"]
    results = [entry for entry in messages if isinstance(entry, ToolMessage)]
    assert [entry.tool_call_id for entry in results] == ["call_1", "call_2"]
    assert all(entry.name == "query_my_wardrobe_items" for entry in results)
    assert all(entry.status == "success" for entry in results)
    assert json.loads(results[0].content)["items"][0]["id"] == str(ids[0])
    assert json.loads(results[1].content)["items"][0]["id"] == str(ids[1])
    definition = model.await_args_list[0].kwargs["tools"][0]["function"]
    assert definition["name"] == "query_my_wardrobe_items"
    assert set(definition["parameters"]["properties"]) == {"item_type", "page", "page_size"}
    assert definition["parameters"]["additionalProperties"] is False


async def test_direct_clarification_does_not_execute_tools(context):
    model = AsyncMock(
        return_value=final_message(status="needs_clarification", answer="你准备参加什么场合？")
    )
    result = await run(context, model)
    assert result.status == "needs_clarification"
    assert result.answer == "你准备参加什么场合？"
    assert result.outfits == []
    assert model.await_count == 1
    context.service.get_list.assert_not_awaited()


async def test_empty_query_returns_no_items(context):
    model = AsyncMock(
        side_effect=[
            request_message(tool_request("hat")),
            final_message(status="no_items", answer="本次查询没有找到帽子。"),
        ]
    )
    result = await run(context, model)
    assert result.status == "no_items"
    assert result.outfits == []
    assert model.await_count == 2
    context.service.get_list.assert_awaited_once()


@pytest.mark.parametrize(
    "parameters",
    [
        {"page": 0},
        {"page_size": 101},
        {"item_type": ""},
        {"user_id": "another-user"},
        {"current_user": "another-user"},
        {"db": "another-session"},
        {"session": "another-session"},
    ],
)
async def test_invalid_tool_parameters_stop_before_business_query(context, parameters):
    model = AsyncMock(
        side_effect=[
            request_message(tool_request(**parameters)),
            final_message([context.items[0].id]),
        ]
    )
    result = await run(context, model)
    assert result.status == "invalid_result"
    assert result.outfits == []
    assert model.await_count == 1
    context.service.get_list.assert_not_awaited()


@pytest.mark.parametrize("forged_kind", ["unknown", "unqueried", "mixed", "bad_uuid"])
async def test_unconfirmed_ids_never_enter_valid_previews(context, forged_kind):
    forged = context.items[1].id if forged_kind == "unqueried" else uuid4()
    ids = [context.items[0].id, forged] if forged_kind == "mixed" else [forged]
    if forged_kind == "bad_uuid":
        ids = ["not-a-uuid"]
    model = AsyncMock(side_effect=[request_message(tool_request()), final_message(ids)])
    result = await run(context, model)
    assert result.status == "invalid_result"
    assert result.outfits == []
    assert model.await_count == 2
    context.service.get_list.assert_awaited_once()


async def test_success_without_query_is_invalid_even_if_item_exists(context):
    model = AsyncMock(return_value=final_message([context.items[0].id]))
    result = await run(context, model)
    assert result.status == "invalid_result"
    assert result.outfits == []
    context.service.get_list.assert_not_awaited()


async def test_multiple_model_rounds_accumulate_only_actual_tool_results(context):
    model = AsyncMock(
        side_effect=[
            request_message(tool_request()),
            request_message(tool_request("pants", call_id="call_2")),
            final_message([item.id for item in context.items]),
        ]
    )
    result = await run(context, model)
    assert result.status == "success"
    assert model.await_count == 3
    assert context.service.get_list.await_count == 2
    assert context.max_active_queries == 1
    messages = model.await_args_list[2].args[0]
    assert [entry.type for entry in messages] == ["system", "human", "ai", "tool", "ai", "tool"]
    assert [entry.tool_call_id for entry in messages if isinstance(entry, ToolMessage)] == [
        "call_1",
        "call_2",
    ]


async def test_model_loop_limit_returns_invalid_result(context):
    async def repeat(messages, *, tools):
        count = sum(isinstance(entry, AIMessage) for entry in messages)
        return request_message(tool_request(call_id=f"loop_{count}"))

    model = AsyncMock(side_effect=repeat)
    result = await run(context, model, max_model_calls=2, max_tool_calls=8)
    assert result.status == "invalid_result"
    assert "轮次达到上限" in result.answer
    assert result.outfits == []
    assert model.await_count == 2
    assert context.service.get_list.await_count == 2


async def test_tool_budget_rejects_entire_over_budget_batch(context):
    model = AsyncMock(
        return_value=request_message(tool_request(), tool_request("pants", call_id="call_2"))
    )
    result = await run(context, model, max_tool_calls=1)
    assert result.status == "invalid_result"
    assert "工具调用次数" in result.answer
    assert model.await_count == 1
    context.service.get_list.assert_not_awaited()


async def test_final_answer_never_triggers_another_tool_call(context):
    model = AsyncMock(
        side_effect=[
            request_message(tool_request()),
            final_message([context.items[0].id]),
            request_message(tool_request("pants", call_id="must_not_run")),
        ]
    )
    result = await run(context, model)
    assert result.status == "success"
    assert model.await_count == 2
    context.service.get_list.assert_awaited_once()


@pytest.mark.parametrize("failure_kind", ["unknown_tool", "duplicate_id", "empty_id"])
async def test_tool_name_and_call_ids_are_checked_before_execution(context, failure_kind):
    requests = [tool_request()]
    if failure_kind == "unknown_tool":
        requests = [tool_request(name="write_or_delete_items")]
    elif failure_kind == "duplicate_id":
        requests.append(tool_request("pants"))
    else:
        requests = [tool_request(call_id="")]
    model = AsyncMock(return_value=request_message(*requests))
    result = await run(context, model)
    assert result.status == "invalid_result"
    assert model.await_count == 1
    context.service.get_list.assert_not_awaited()


async def test_candidate_scope_does_not_leak_between_requests_or_users(context):
    model = AsyncMock(
        side_effect=[
            request_message(tool_request()),
            final_message([context.items[0].id]),
            request_message(tool_request("hat")),
            final_message([context.items[0].id]),
        ]
    )
    agent = AgentService(model=model)
    first = await agent.generate_preview("查询衬衫", current_user=context.user, db=context.db)
    other_user = SimpleNamespace(id=uuid4())
    other_db = Mock(spec=AsyncSession)
    second = await agent.generate_preview("查询帽子", current_user=other_user, db=other_db)

    assert first.status == "success"
    assert second.status == "invalid_result"
    assert second.outfits == []
    assert model.await_count == 4
    assert context.service.get_list.await_count == 2
    assert [
        entry.kwargs["user_id"] for entry in context.service.get_list.await_args_list
    ] == [context.user.id, other_user.id]
    assert context.constructor.call_args_list == [call(context.db), call(other_db)]
    assert other_db.mock_calls == []


async def test_model_cannot_forge_candidates_by_mutating_message_copy(context):
    forged = uuid4()

    async def mutate(messages, *, tools):
        if not isinstance(messages[-1], ToolMessage):
            return request_message(tool_request())
        payload = json.loads(messages[-1].content)
        payload["items"][0]["id"] = str(forged)
        messages[-1].content = json.dumps(payload)
        return final_message([forged])

    model = AsyncMock(side_effect=mutate)
    result = await run(context, model)
    assert result.status == "invalid_result"
    assert result.outfits == []
    assert model.await_count == 2
    context.service.get_list.assert_awaited_once()


@pytest.mark.parametrize("queried", [False, True])
async def test_no_items_requires_query_evidence_and_no_usable_candidates(context, queried):
    replies = [final_message(status="no_items")]
    if queried:
        replies.insert(0, request_message(tool_request()))
    model = AsyncMock(side_effect=replies)
    result = await run(context, model)
    assert result.status == "invalid_result"
    assert result.outfits == []
    assert context.service.get_list.await_count == int(queried)


@pytest.mark.parametrize("unusable", ["needs_wash", "processing"])
async def test_query_semantics_remain_but_unusable_items_are_not_recommended(context, unusable):
    if unusable == "needs_wash":
        context.items[0].needs_wash = True
    else:
        context.items[0].status = ItemStatus.processing
    model = AsyncMock(
        side_effect=[
            request_message(tool_request()),
            final_message([context.items[0].id]),
        ]
    )
    result = await run(context, model)
    assert result.status == "invalid_result"
    context.service.get_list.assert_awaited_once()
    payload = json.loads(model.await_args_list[1].args[0][-1].content)
    assert payload["items"][0]["id"] == str(context.items[0].id)


@pytest.mark.parametrize(
    "content",
    [
        "普通文本而非 JSON",
        '{"status":"success","answer":"好","outfits":[]}',
        '{"status":"unknown","answer":"好","outfits":[]}',
        '{"status":"needs_clarification","answer":"好","user_id":"forged"}',
        '{"status":"needs_clarification","answer":"   ","outfits":[]}',
    ],
)
async def test_malformed_final_results_are_rejected(context, content):
    model = AsyncMock(return_value=AIMessage(content=content))
    result = await run(context, model)
    assert result.status == "invalid_result"
    assert result.outfits == []
    assert model.await_count == 1
    context.service.get_list.assert_not_awaited()


@pytest.mark.parametrize("source", ["model", "tool"])
async def test_execution_errors_do_not_leak_internal_details(context, source):
    private_error = RuntimeError("/internal/private/path must not be exposed")
    if source == "model":
        model = AsyncMock(side_effect=private_error)
    else:
        model = AsyncMock(return_value=request_message(tool_request()))
        context.service.get_list.side_effect = private_error
    result = await run(context, model)
    assert result.status == "invalid_result"
    assert result.outfits == []
    assert "/internal/" not in result.model_dump_json()
    assert model.await_count == 1
    assert context.service.get_list.await_count == int(source == "tool")


async def test_synchronous_injected_model_can_request_clarification(context):
    model = Mock(return_value=final_message(status="needs_clarification"))
    result = await run(context, model)
    assert result.status == "needs_clarification"
    model.assert_called_once()
    context.service.get_list.assert_not_awaited()
