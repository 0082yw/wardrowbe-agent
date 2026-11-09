"""Agent 请求与响应契约测试；不注册路由、不发送 HTTP、不连接数据库或模型。"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app import agent_tools
from app.models.item import ItemStatus
from app.schemas.agent import (
    MAX_AGENT_MESSAGE_LENGTH,
    AgentItemDisplay,
    AgentPreviewResponse,
    AgentRequest,
    OutfitPreviewResponse,
)
from app.services.agent_service import AgentPreview, AgentService, OutfitPreview
from app.services.item_service import ItemService


def make_preview(item_ids):
    return AgentPreview(
        status="success",
        answer="已根据本次衣橱查询生成预览。",
        outfits=[
            OutfitPreview(
                item_ids=item_ids,
                reasoning="使用已查询确认的衣物。",
                style_notes="请结合实际天气决定是否增加外套。",
            )
        ],
    )


@pytest.fixture
def display_items():
    return [
        AgentItemDisplay(id=uuid4(), name="白色衬衫", type="shirt"),
        AgentItemDisplay(id=uuid4(), name=None, type="pants"),
    ]


@pytest.mark.parametrize(
    "message",
    [
        "我明天面试，该穿什么？",
        "周末徒步，希望活动方便。",
        "晚上聚餐，想搭配得轻松一些。",
        "我的衣橱有哪些上衣？",
        "这件衣服还能怎么搭？\n希望尽量使用已有衣物。",
    ],
)
def test_request_accepts_free_form_wardrobe_and_outfit_needs(message):
    request = AgentRequest.model_validate_json(json.dumps({"message": f"  {message}  "}))
    assert request.message == message
    assert request.model_dump() == {"message": message}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": " \t\n "},
        {"message": "\u200b\ufeff"},
        {"message": None},
        {"message": 1},
        {"message": True},
        {"message": ["穿搭需求"]},
        {"message": {"text": "穿搭需求"}},
        {"message": "衣" * (MAX_AGENT_MESSAGE_LENGTH + 1)},
        {"message": " " * MAX_AGENT_MESSAGE_LENGTH + "衣"},
    ],
)
def test_request_rejects_missing_blank_wrong_type_and_overlong_input(payload):
    with pytest.raises(ValidationError):
        AgentRequest.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("key", ["user_id", "current_user", "db", "session", "model", "endpoint"])
def test_request_rejects_identity_session_and_model_control_fields(key):
    with pytest.raises(ValidationError):
        AgentRequest.model_validate_json(json.dumps({"message": "查询衣橱", key: "untrusted"}))


def test_request_length_boundary_matches_existing_service():
    assert len(AgentRequest(message="衣" * MAX_AGENT_MESSAGE_LENGTH).message) == 10000
    with pytest.raises(ValidationError):
        AgentRequest(message=b"wardrobe")


def test_response_extends_existing_types_instead_of_replacing_them():
    assert issubclass(AgentPreviewResponse, AgentPreview)
    assert issubclass(OutfitPreviewResponse, OutfitPreview)
    for field in ("status", "answer"):
        assert (
            AgentPreviewResponse.model_fields[field].annotation
            == AgentPreview.model_fields[field].annotation
        )


def test_response_serializes_display_fields_in_preview_order(display_items):
    selected = list(reversed(display_items))
    preview = make_preview([item.id for item in selected])
    response = AgentPreviewResponse.from_preview(preview, items=display_items)
    payload = response.model_dump(mode="json")

    assert payload["status"] == "success"
    assert payload["answer"] == preview.answer
    outfit = payload["outfits"][0]
    assert outfit["item_ids"] == [str(item.id) for item in selected]
    assert outfit["items"] == [item.model_dump(mode="json") for item in selected]
    assert set(outfit) == {"item_ids", "reasoning", "style_notes", "items"}
    assert all(set(item) == {"id", "name", "type", "thumbnail_url"} for item in outfit["items"])
    assert AgentPreviewResponse.model_validate_json(response.model_dump_json()) == response
    assert "items" not in preview.model_dump()["outfits"][0], "不能原地改写模型输出契约"


def test_response_never_exposes_unreferenced_display_items(display_items):
    preview = make_preview([display_items[0].id])
    payload = AgentPreviewResponse.from_preview(
        preview, items=display_items
    ).model_dump(mode="json")
    assert len(payload["outfits"][0]["items"]) == 1
    assert str(display_items[1].id) not in json.dumps(payload)


@pytest.mark.parametrize("status", ["needs_clarification", "no_items", "invalid_result"])
def test_non_success_response_is_structured_and_has_no_outfits(status, display_items):
    preview = AgentPreview(status=status, answer="本次没有生成穿搭预览。")
    response = AgentPreviewResponse.from_preview(preview, items=display_items)
    assert response.model_dump(mode="json") == {
        "status": status,
        "answer": "本次没有生成穿搭预览。",
        "outfits": [],
    }


def test_empty_display_collection_cannot_complete_a_successful_preview(display_items):
    with pytest.raises(ValueError, match="缺少"):
        AgentPreviewResponse.from_preview(make_preview([display_items[0].id]))


def test_duplicate_display_metadata_is_rejected(display_items):
    with pytest.raises(ValueError, match="重复"):
        AgentPreviewResponse.from_preview(
            make_preview([display_items[0].id]),
            items=[display_items[0], display_items[0]],
        )


@pytest.mark.parametrize("change", ["unknown_id", "wrong_order", "empty", "extra_item"])
def test_display_ids_must_match_selected_ids_exactly(display_items, change):
    payload = AgentPreviewResponse.from_preview(
        make_preview([item.id for item in display_items]), items=display_items
    ).model_dump(mode="json")
    entries = payload["outfits"][0]["items"]
    if change == "unknown_id":
        entries[0]["id"] = str(uuid4())
    elif change == "wrong_order":
        entries.reverse()
    elif change == "empty":
        entries.clear()
    else:
        entries.append(AgentItemDisplay(id=uuid4(), name="未选择", type="hat").model_dump(mode="json"))
    with pytest.raises(ValidationError):
        AgentPreviewResponse.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "not-a-uuid"},
        {"name": "衣" * 101},
        {"name": 123},
        {"type": ""},
        {"type": "   "},
        {"type": "x" * 51},
        {"type": False},
        {"user_id": "untrusted"},
        {"image_path": "/internal/private.jpg"},
        {"ai_raw_response": {"internal": "not-for-client"}},
    ],
)
def test_display_rejects_invalid_fields_and_internal_metadata(changes):
    payload = {"id": str(uuid4()), "name": None, "type": "shirt", **changes}
    with pytest.raises(ValidationError):
        AgentItemDisplay.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "change",
    ["unknown_status", "blank_answer", "long_answer", "empty_success", "non_success_outfits",
     "duplicate_item", "too_many_items", "too_many_outfits", "blank_reasoning", "long_tip",
     "unexpected_field", "saved_outfit_id"],
)
def test_response_keeps_existing_status_text_size_and_extra_field_constraints(display_items, change):
    payload = AgentPreviewResponse.from_preview(
        make_preview([display_items[0].id]), items=display_items
    ).model_dump(mode="json")
    outfit = payload["outfits"][0]
    if change == "unknown_status":
        payload["status"] = "saved"
    elif change == "blank_answer":
        payload["answer"] = "   "
    elif change == "long_answer":
        payload["answer"] = "衣" * 1001
    elif change == "empty_success":
        payload["outfits"] = []
    elif change == "non_success_outfits":
        payload["status"] = "no_items"
    elif change == "duplicate_item":
        outfit["item_ids"] *= 2
        outfit["items"] *= 2
    elif change == "too_many_items":
        outfit["item_ids"] *= 21
        outfit["items"] *= 21
    elif change == "too_many_outfits":
        payload["outfits"] *= 4
    elif change == "blank_reasoning":
        outfit["reasoning"] = ""
    elif change == "long_tip":
        outfit["style_notes"] = "衣" * 1001
    elif change == "unexpected_field":
        payload["user_id"] = str(uuid4())
    else:
        outfit["id"] = str(uuid4())
    with pytest.raises(ValidationError):
        AgentPreviewResponse.model_validate_json(json.dumps(payload))


def test_model_contract_still_rejects_display_metadata(display_items):
    response = AgentPreviewResponse.from_preview(
        make_preview([display_items[0].id]), items=display_items
    )
    with pytest.raises(ValidationError):
        AgentPreview.model_validate_json(response.model_dump_json())


def test_json_schema_exposes_only_supported_request_and_response_fields():
    request = AgentRequest.model_json_schema()
    assert request["required"] == ["message"]
    assert set(request["properties"]) == {"message"}
    assert request["properties"]["message"]["maxLength"] == MAX_AGENT_MESSAGE_LENGTH
    assert request["additionalProperties"] is False

    response = AgentPreviewResponse.model_json_schema()
    assert set(response["properties"]) == {"status", "answer", "outfits", "kind", "scene", "execution"}
    assert "execution" not in response["required"]
    assert response["$defs"]["AgentRequestExecution"]["additionalProperties"] is False
    # Model-facing output stays unchanged; execution metadata is backend-only.
    assert "execution" not in AgentPreview.model_json_schema()["properties"]
    assert response["properties"]["kind"]["enum"] == ["items", "outfit"]
    assert response["additionalProperties"] is False
    outfit = response["$defs"]["OutfitPreviewResponse"]
    assert "items" in outfit["required"]
    assert outfit["properties"]["items"]["maxItems"] == 20
    item = response["$defs"]["AgentItemDisplay"]
    assert set(item["properties"]) == {"id", "name", "type", "thumbnail_url"}


@pytest.mark.asyncio
async def test_mock_service_preview_projects_trusted_item_data_without_saving(monkeypatch):
    db = Mock(spec=AsyncSession)
    user = SimpleNamespace(id=uuid4())
    item = SimpleNamespace(
        id=uuid4(),
        name="后端查询到的衬衫",
        type="shirt",
        status=ItemStatus.ready,
        needs_wash=False,
        image_path="/internal/private.jpg",
        user_id=user.id,
    )
    service = Mock(spec=ItemService)
    service.get_list = AsyncMock(return_value=([item], 1))
    constructor = Mock(return_value=service)
    monkeypatch.setattr(agent_tools, "ItemService", constructor)
    model = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "query_my_wardrobe_items",
                    "args": {"item_type": "shirt"},
                    "id": "contract_query",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content=make_preview([item.id]).model_dump_json()),
        ]
    )
    request = AgentRequest(message="看看衣橱里的衬衫，帮我搭配。")
    preview = await AgentService(model=model).generate_preview(
        request.message, current_user=user, db=db
    )
    response = AgentPreviewResponse.from_preview(
        preview,
        items=[AgentItemDisplay(id=item.id, name=item.name, type=item.type)],
    )

    assert response.status == "success"
    assert response.outfits[0].items[0].name == item.name
    assert response.outfits[0].item_ids == [item.id]
    assert "/internal/" not in response.model_dump_json()
    assert "user_id" not in response.model_dump_json()
    assert model.await_count == 2
    service.get_list.assert_awaited_once()
    assert service.get_list.await_args.kwargs["user_id"] == user.id
    constructor.assert_called_once_with(db)
    assert db.mock_calls == [], "Schema 和投影过程不能查询、提交、保存或关闭 Session"


def test_display_thumbnail_is_optional_and_accepts_only_local_signed_image_urls():
    item_id = uuid4()
    assert AgentItemDisplay(id=item_id, type="shirt").thumbnail_url is None
    url = f"/api/v1/images/{uuid4()}/shirt_thumb.jpg?expires=9999999999&sig={'a' * 32}"
    assert AgentItemDisplay(id=item_id, type="shirt", thumbnail_url=url).thumbnail_url == url


@pytest.mark.parametrize("url", [
    "https://example.com/private.jpg",
    "//example.com/private.jpg",
    "/home/developer/private.jpg",
    "/api/v1/images/other/../private.jpg",
    "/api/v1/images/00000000-0000-4000-8000-000000000001/shirt.jpg",
    123,
])
def test_display_rejects_unsigned_external_or_internal_image_locations(url):
    with pytest.raises(ValidationError):
        AgentItemDisplay(id=uuid4(), type="shirt", thumbnail_url=url)


@pytest.mark.parametrize("payload", [
    {"mode": "unknown", "model": "qwen3:0.6b-cpu", "available": False, "cpu_only": False},
    {"mode": "unknown", "model": None, "available": True, "cpu_only": True},
    {"mode": "local_ollama", "model": None, "available": False, "cpu_only": False},
    {"mode": "local_ollama", "model": "qwen3:0.6b-cpu", "available": True, "cpu_only": False},
    {"mode": "local_ollama", "model": "cloud-model", "available": True, "cpu_only": True},
])
def test_runtime_status_contract_rejects_false_mode_claims(payload):
    from pydantic import ValidationError

    from app.schemas.agent import AgentRuntimeStatus

    with pytest.raises(ValidationError):
        AgentRuntimeStatus.model_validate(payload)
