"""真实 ToolNode + 替身 ItemService/文本模型，禁止连接数据库和模型。"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app import agent_tools
from app.api import agent as api
from app.database import get_db
from app.schemas.agent import AgentItemDisplay, AgentPreviewResponse
from app.services.agent_service import AgentPreview, AgentService
from app.services.agent_audit import AgentAudit, current_audit
from app.services.outfit_engine import (
    OutfitCandidate, OutfitPlan, combine_outfits, outfit_defaults, parse_outfit_text, route_outfit,
)
from app.services.wardrobe_query_agent import WardrobeQueryAgentModel, route_query
from app.utils.auth import get_current_user


def row(user_id=None, kind="shirt", **changes):
    values = dict(
        id=uuid4(), user_id=user_id or uuid4(), name="真实衣物", type=kind, status="ready",
        needs_wash=False, is_archived=False, colors=[], primary_color=None, formality=None,
        thumbnail_path=None, image_path=None, processing_kind=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def candidate(kind="shirt", **changes):
    item = row(kind=kind, **changes)
    return OutfitCandidate.model_validate({k: getattr(item, k) for k in OutfitCandidate.model_fields})


def reply(**changes):
    return AIMessage(content=json.dumps({"answer": "已为你搭配", **changes}, ensure_ascii=False))


@pytest.mark.parametrize("message,scene,count", [
    ("帮我搭配一套日常穿搭", "daily", 1), ("帮我搭配三套穿搭", "daily", 3),
    ("请给我推荐两套通勤穿搭。", "commute", 2), ("生成3套日常搭配", "daily", 3),
    ("帮我搭配一套日常穿搭，包含衬衫和牛仔裤", "daily", 1),
    ("帮我搭配一套日常穿搭，包含蓝色单品", "daily", 1),
    ("帮我搭配一套日常穿搭，不要鞋子和外套", "daily", 1),
])
def test_explicit_intents(message, scene, count):
    plan = route_outfit(message)
    assert plan.scene == scene and plan.count == count
    assert route_query(message) is None


@pytest.mark.parametrize("message", [
    "帮我搭配", "我要穿什么", "帮我搭配四套穿搭", "帮我搭配一套雨天穿搭",
    "帮我搭配一套日常穿搭，纯棉", "帮我搭配一套日常穿搭，包含未知衣物",
    "帮我搭配一套日常穿搭，包含未知色单品", "不要帮我搭配一套日常穿搭",
    "查询我的衬衫", "查询我的裤子", "帮我搭配一套婚礼穿搭",
])
def test_unsupported_or_ambiguous_outfit_intent_is_not_guessed(message):
    assert route_outfit(message) is None


def test_two_real_candidates_form_one_complete_outfit_not_two_single_previews():
    top, bottom = candidate(colors=["white"]), candidate("jeans", colors=["light-blue"])
    groups, error = combine_outfits([top, bottom], OutfitPlan("daily", 3), limit=3)
    assert not error and len(groups) == 1 and groups[0] == (top, bottom)
    reason, notes = outfit_defaults(groups[0], OutfitPlan("daily", 1))
    assert "白色" in reason and "浅蓝色" in reason and "未包含鞋子" in notes


@pytest.mark.parametrize("kinds,missing", [([], "上衣和下装"), (["shirt"], "下装"), (["jeans"], "上衣")])
def test_missing_core_items_do_not_fabricate_outfit(kinds, missing):
    groups, error = combine_outfits([candidate(k) for k in kinds], OutfitPlan("daily", 1), limit=3)
    assert groups == [] and missing in error


def test_max_three_distinct_combinations_without_optional_omission_duplicates():
    items = [candidate("shirt"), candidate("top"), candidate("jeans"), candidate("pants"), candidate("shoes")]
    groups, _ = combine_outfits(items, OutfitPlan("daily", 3), limit=3)
    assert len(groups) == 3
    assert len({frozenset(i.id for i in g) for g in groups}) == 3
    assert all(len(g) == 3 and len({i.id for i in g}) == len(g) for g in groups)


@pytest.mark.parametrize("changes", [{"status": "error"}, {"status": "processing"},
    {"status": "archived"}, {"needs_wash": True}, {"is_archived": True}])
def test_unwearable_items_are_excluded(changes):
    groups, error = combine_outfits([candidate(**changes), candidate("jeans")], OutfitPlan("daily", 1), limit=3)
    assert not groups and "上衣" in error


def test_duplicate_candidates_are_rejected():
    top = candidate()
    with pytest.raises(ValueError, match="重复"):
        combine_outfits([top, top, candidate("jeans")], OutfitPlan("daily", 1), limit=3)


def test_commute_ranks_actual_formality_without_inventing_missing_attributes():
    formal = candidate(formality="business-casual")
    casual = candidate(formality="very-casual")
    groups, _ = combine_outfits([casual, formal, candidate("jeans")], OutfitPlan("commute", 1), limit=3)
    assert groups[0][0] == formal
    assert "工作场所" in outfit_defaults(groups[0], OutfitPlan("commute", 1))[1]


def test_explicit_type_color_and_optional_constraints_use_recorded_fields():
    items = [candidate(colors=["white"]), candidate("jeans", colors=["blue"]), candidate("shoes"), candidate("jacket")]
    plan = route_outfit("帮我搭配一套日常穿搭，包含衬衫和牛仔裤和蓝色单品，不要鞋子和外套")
    groups, _ = combine_outfits(items, plan, limit=3)
    assert len(groups[0]) == 2
    items[1].colors = []
    items[1].name = "蓝色牛仔裤"  # 名称不作为颜色证据。
    assert combine_outfits(items, plan, limit=3)[0] == []


@pytest.mark.parametrize("changes", [
    {"answer": ""}, {"answer": "   "}, {"reasoning": "可省略；提供时必须为非空简短文本"},
    {"style_notes": "JSON Schema properties type"}, {},
])
def test_low_quality_text_and_missing_optional_fields_fall_back(changes):
    items = [candidate(), candidate("jeans")]
    draft = parse_outfit_text(reply(**changes).content, items=items, plan=OutfitPlan("daily", 1), count=1)
    assert draft.answer and draft.reasoning and draft.style_notes and draft.fallback_fields
    assert "Schema" not in draft.style_notes and "可省略" not in draft.reasoning


@pytest.mark.parametrize("changes", [
    {"answer": None}, {"reasoning": None}, {"style_notes": 3}, {"answer": "x" * 1001},
    {"item_ids": [str(uuid4())]}, {"status": "success"}, {"items": []},
])
def test_invalid_text_types_or_forged_business_fields_fail(changes):
    with pytest.raises(ValueError):
        parse_outfit_text(reply(**changes).content, items=[candidate(), candidate("jeans")],
                          plan=OutfitPlan("daily", 1), count=1)


def test_grounded_model_text_is_preserved():
    reason, notes = outfit_defaults([candidate(), candidate("jeans")], OutfitPlan("daily", 1))
    value = parse_outfit_text(reply(answer="这套穿搭包含一件上衣和一件下装。", reasoning=reason,
        style_notes=notes).content, items=[candidate(), candidate("jeans")], plan=OutfitPlan("daily", 1), count=1)
    assert value.fallback_fields == []


@pytest.fixture
def context(monkeypatch):
    user = SimpleNamespace(id=uuid4(), family_id=None)
    ctx = SimpleNamespace(user=user, db=Mock(spec=AsyncSession), forced=None)
    ctx.items = [row(user.id, name="白色衬衫", colors=["white"]),
                 row(user.id, "jeans", name="浅蓝色牛仔裤", colors=["light-blue"])]
    async def get_list(*, user_id, filters, page, page_size):
        assert user_id == user.id and page == 1 and page_size in (20, 100)
        assert filters.is_archived is False
        matches = ctx.forced if ctx.forced is not None else [
            i for i in ctx.items if i.user_id == user_id and not i.is_archived
            and (filters.type is None or i.type == filters.type)
        ]
        return matches[:page_size], len(matches)
    async def get_by_id(item_id, *, user_id):
        assert user_id == user.id
        return next((i for i in ctx.items if i.id == item_id and i.user_id == user_id), None)
    ctx.service = Mock(spec=agent_tools.ItemService)
    ctx.service.get_list = AsyncMock(side_effect=get_list)
    ctx.service.get_by_id = AsyncMock(side_effect=get_by_id)
    monkeypatch.setattr(agent_tools, "ItemService", Mock(return_value=ctx.service))
    monkeypatch.setattr(api, "ItemService", Mock(return_value=ctx.service))
    monkeypatch.setattr(api, "rate_limit_by_user", AsyncMock())
    ctx.text_model = AsyncMock(return_value=reply())
    ctx.controller = WardrobeQueryAgentModel(ctx.text_model, max_previews=3)
    ctx.agent = api.get_agent_service(ctx.controller)
    yield ctx
    assert ctx.db.mock_calls == [], "不能直接执行 SQL、保存或提交"


async def post(ctx, message="帮我搭配一套日常穿搭"):
    app = FastAPI()
    app.include_router(api.router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = lambda: ctx.user
    async def db():
        yield ctx.db
    app.dependency_overrides[get_db] = db
    app.dependency_overrides[api.get_agent_service] = lambda: ctx.agent
    async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as c:
        return await c.post("/api/v1/agent/preview", json={"message": message})


@pytest.mark.asyncio
async def test_api_complete_outfit_real_toolnode_and_backend_id_projection(context):
    response = await post(context)
    assert response.status_code == 200
    result = AgentPreviewResponse.model_validate_json(response.content)
    assert result.status == "success" and result.kind == "outfit"
    assert len(result.outfits) == 1 and result.outfits[0].item_ids == [i.id for i in context.items]
    assert [i.name for i in result.outfits[0].items] == [i.name for i in context.items]
    context.service.get_list.assert_awaited_once()
    assert context.service.get_by_id.await_count == 4
    context.text_model.assert_awaited_once()
    messages = context.controller.trace.graph_inputs[-1]
    tool = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool.tool_call_id == context.controller.call["id"]
    assert json.loads(tool.content)["items"][0]["id"] == str(context.items[0].id)
    sent = repr(context.text_model.await_args)
    assert all(str(i.id) not in sent for i in context.items) and str(context.user.id) not in sent
    assert "尚未保存" in result.answer and context.controller.trace.text_fallbacks


@pytest.mark.asyncio
async def test_outfit_audit_links_all_candidates_and_two_display_items(context):
    # 路由自身也有审计；此处在 Service 范围直接验证新工具的真实执行标记。
    audit = AgentAudit()
    audit.bind_user(context.user.id)
    token = current_audit.set(audit)
    try:
        result = await context.agent.generate_preview(
            "帮我搭配一套日常穿搭", current_user=context.user, db=context.db,
        )
    finally:
        current_audit.reset(token)
    snapshot = audit.snapshot()
    assert result.status == "success"
    assert snapshot["tool_count"] == snapshot["executed_tool_count"] == 1
    assert snapshot["candidate_count"] == 2
    assert snapshot["tools"][0]["scope"] == "current_user_unarchived_candidates"


@pytest.mark.asyncio
@pytest.mark.parametrize("image_kind", ["own", "foreign", "absolute", "processing"])
async def test_outfit_images_only_use_current_user_paths(context, monkeypatch, image_kind):
    from app.utils import signed_urls
    monkeypatch.setattr(signed_urls, "get_settings", lambda: SimpleNamespace(secret_key="isolated-test-key"))
    item = context.items[0]
    item.thumbnail_path = f"{context.user.id}/shirt.jpg"
    if image_kind == "foreign":
        item.thumbnail_path = f"{uuid4()}/private.jpg"
    elif image_kind == "absolute":
        item.thumbnail_path = "/data/private.jpg"
    elif image_kind == "processing":
        item.processing_kind = "image"
    response = await post(context)
    result = response.json()
    assert result["status"] == "success"
    image = result["outfits"][0]["items"][0]["thumbnail_url"]
    if image_kind == "own":
        assert image.startswith(f"/api/v1/images/{context.user.id}/shirt.jpg?")
    else:
        assert image is None
    assert "/data/" not in response.text


@pytest.mark.asyncio
async def test_changed_color_after_model_invalidates_preview(context):
    async def mutate(*args, **kwargs):
        context.items[0].primary_color = "red"
        return reply()
    context.text_model.side_effect = mutate
    result = (await post(context)).json()
    assert result["status"] == "invalid_result" and result["outfits"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kinds", [[], ["shirt"], ["jeans"]])
async def test_api_missing_core_has_no_model_call(context, kinds):
    context.items = [row(context.user.id, k) for k in kinds]
    result = (await post(context)).json()
    assert result["status"] == "no_items" and result["outfits"] == [] and result["kind"] == "outfit"
    context.text_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_cannot_fill_three_by_repeating_one_pair(context):
    result = (await post(context, "帮我搭配三套穿搭")).json()
    assert result["status"] == "success" and len(result["outfits"]) == 1 and "未凑足 3 套" in result["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["foreign", "unknown", "archived", "changed", "duplicate"])
async def test_api_rejects_untrusted_candidates_before_model(context, bad):
    if bad == "foreign":
        context.forced = [row(uuid4()), context.items[1]]
    elif bad == "unknown":
        context.forced = [row(context.user.id), context.items[1]]
    elif bad == "archived":
        context.items[0].is_archived = True
        context.forced = context.items
    elif bad == "changed":
        context.forced = [row(context.user.id, id=context.items[0].id, colors=["red"]), context.items[1]]
    else:
        context.forced = [context.items[0], context.items[0], context.items[1]]
    result = (await post(context)).json()
    assert result["status"] == "invalid_result" and result["outfits"] == []
    context.text_model.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["{\"answer\":", '{"type":"object","properties":{}}',
    '{"answer":"ok","item_ids":["forged"]}', '{"answer":null}', '{"answer":"ok","status":"success"}'])
async def test_api_invalid_json_or_model_ids_not_accepted(context, output):
    context.text_model.return_value = AIMessage(content=output)
    result = (await post(context)).json()
    assert result["status"] == "invalid_result" and result["outfits"] == []
    context.text_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_timeout_does_not_retry(context):
    context.text_model.side_effect = TimeoutError("isolated timeout")
    result = (await post(context)).json()
    assert result["status"] == "invalid_result"
    context.text_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_request_clarifies_without_query_or_model(context):
    result = (await post(context, "帮我搭配一套雨天穿搭")).json()
    assert result["status"] == "needs_clarification"
    context.service.get_list.assert_not_awaited()
    context.text_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_formal_without_recorded_formality_clarifies_without_model(context):
    result = (await post(context, "帮我搭配一套正式穿搭")).json()
    assert result["status"] == "needs_clarification" and result["outfits"] == []
    context.text_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_formal_uses_verified_recorded_attributes(context):
    for i in context.items:
        i.formality = "formal"
    result = (await post(context, "帮我搭配一套正式穿搭")).json()
    assert result["status"] == "success" and result["scene"] == "formal"
    context.text_model.assert_awaited_once()


@pytest.mark.asyncio
async def test_single_query_is_not_complete_outfit(context):
    result = (await post(context, "查询我的衬衫")).json()
    assert result["status"] == "success" and result.get("kind", "items") == "items"
    assert len(result["outfits"][0]["items"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,model_limit", [(1, 1), (0, 2)])
async def test_tool_and_model_call_budgets_remain_enforced(context, limit, model_limit):
    if limit == 0:
        with pytest.raises(ValueError):
            AgentService(model=context.controller, max_tool_calls=0)
        return
    context.agent = AgentService(model=context.controller, max_tool_calls=limit, max_model_calls=model_limit)
    result = (await post(context)).json()
    assert result["status"] == "invalid_result"
    context.text_model.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["user_id", "current_user", "db", "item_ids"])
async def test_outfit_tool_parameters_cannot_override_bound_scope(context, key):
    tool = agent_tools.create_query_outfit_candidates_tool(current_user=context.user, db=context.db)
    assert set(tool.get_input_schema().model_fields) == {"page", "page_size"}
    with pytest.raises(ValidationError):
        await tool.ainvoke({key: str(uuid4())})
    context.service.get_list.assert_not_awaited()


@pytest.mark.parametrize("kinds", [["shirt"], ["shirt", "shoes"], ["jeans", "pants"], ["shirt", "jeans", "socks"]])
def test_response_schema_rejects_missing_or_invalid_roles(kinds):
    items = [AgentItemDisplay(id=uuid4(), type=k) for k in kinds]
    if len(items) == 1:
        with pytest.raises(ValueError):
            AgentPreview(kind="outfit", scene="daily", status="success", answer="ok", outfits=[{
                "item_ids": [i.id for i in items], "title": "方案", "reasoning": "ok", "style_notes": "ok"}])
    else:
        preview = AgentPreview(kind="outfit", scene="daily", status="success", answer="ok", outfits=[{
            "item_ids": [i.id for i in items], "title": "方案", "reasoning": "ok", "style_notes": "ok"}])
        with pytest.raises(ValueError):
            AgentPreviewResponse.from_preview(preview, items=items)

@pytest.mark.asyncio
async def test_copy_after_real_toolnode_is_model_text_not_default(context):
    model_copy = {"answer": "为你推荐1套日常搭配。",
                  "reasoning": "白色衬衫与浅蓝色牛仔裤搭配，整体简洁利落。",
                  "style_notes": "先试穿，再按实际场合调整。"}
    context.text_model.return_value = AIMessage(content=json.dumps(model_copy, ensure_ascii=False))
    response = await post(context)
    assert response.status_code == 200
    result = AgentPreviewResponse.model_validate_json(response.content)
    assert result.status == "success" and result.answer == model_copy["answer"]
    assert result.outfits[0].reasoning == model_copy["reasoning"]
    assert result.outfits[0].style_notes == model_copy["style_notes"]
    assert context.controller.trace.text_fallbacks == []
    assert len(context.controller.trace.graph_inputs) == 2
    context.service.get_list.assert_awaited_once()
    context.text_model.assert_awaited_once()
    tool = next(m for m in context.controller.trace.graph_inputs[-1] if isinstance(m, ToolMessage))
    assert tool.tool_call_id == context.controller.call["id"]


@pytest.mark.asyncio
async def test_model_failure_uses_safe_error_reply_without_fake_model_success(context):
    context.text_model.side_effect = TimeoutError("private exception must not leak")
    result = (await post(context)).json()
    assert result["status"] == "invalid_result" and result["outfits"] == []
    assert result["answer"] and "private exception" not in result["answer"]
    context.text_model.assert_awaited_once()
    context.service.get_list.assert_awaited_once()

@pytest.mark.asyncio
async def test_request_gpu_metadata_uses_only_trusted_matching_audit(context):
    import hashlib
    import time
    from app.services.gpu_evidence import GPU1_UUID, GpuObservation, GpuRequestEvidence
    model = "qwen3:4b-instruct-2507-q4_K_M"

    async def trusted_fixture(messages, *, tools):
        # Synthetic observer + HTTP metadata only; no network or GPU is used.
        audit = current_audit.get()
        audit.bind_execution("remote_gpu", model)
        index = audit.model_started(model, transport="local_http", execution_mode="remote_gpu")
        audit.model_response(index, 200, {"choices": [{"finish_reason": "stop"}]})
        audit.model_finished(index)
        now = time.time()
        observation = GpuObservation(model=model, model_digest="a" * 64, process_id=123,
            process_started_at=now-1, observed_at=now-0.00001, gpu_uuid=GPU1_UUID,
            layers_loaded=37, layers_total=37, model_size_bytes=3000, gpu_size_bytes=3000)
        canonical = json.dumps(observation.model_dump(), sort_keys=True).encode()
        proof = GpuRequestEvidence(**observation.model_dump(), request_id=audit.request_id,
            request_started_at=now-0.0001, request_finished_at=now,
            evidence_id=hashlib.sha256(canonical).hexdigest())
        audit.record_gpu_evidence(proof)
        return reply(answer="为你推荐1套日常搭配。")
    context.text_model.side_effect = trusted_fixture
    response = await post(context)
    assert response.status_code == 200
    result = AgentPreviewResponse.model_validate_json(response.content)
    assert result.execution.gpu_verified is True
    assert result.execution.request_id == response.headers["X-Request-ID"]
    assert result.execution.gpu_evidence.request_id == result.execution.request_id
    context.service.get_list.assert_awaited_once()
    context.text_model.assert_awaited_once()
