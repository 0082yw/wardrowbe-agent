"""工具封装单元测试；不使用数据库或 Redis。

独立运行时加 --noconftest，避免加载相邻 conftest.py 中的数据库/Redis 夹具。
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_tools import create_query_my_wardrobe_items_tool
from app.api.pagination import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.models.item import ItemStatus
from app.schemas.item import ItemFilter
from app.services.item_service import ItemService


@pytest.fixture
def bound_tool(monkeypatch):
    db = Mock(spec=AsyncSession)
    user = SimpleNamespace(id=uuid4())
    service = Mock(spec=ItemService)
    service.get_list = AsyncMock(return_value=([], 0))
    constructor = Mock(return_value=service)
    monkeypatch.setattr("app.agent_tools.ItemService", constructor)
    wardrobe_tool = create_query_my_wardrobe_items_tool(current_user=user, db=db)
    constructor.assert_called_once_with(db)
    yield wardrobe_tool, user, service
    assert db.mock_calls == [], "工具不能直接执行 SQL、提交、回滚或关闭 Session"


def test_model_schema_exposes_only_query_parameters(bound_tool):
    wardrobe_tool, _, _ = bound_tool
    schema = wardrobe_tool.get_input_schema().model_json_schema()
    assert wardrobe_tool.name == "query_my_wardrobe_items"
    assert set(schema["properties"]) == {"item_type", "page", "page_size"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["page_size"]["maximum"] == MAX_PAGE_SIZE


@pytest.mark.asyncio
async def test_normal_query_preserves_list_semantics_and_returns_safe_json(bound_tool):
    wardrobe_tool, user, service = bound_tool
    items = [
        SimpleNamespace(
            id=uuid4(),
            name="白色衬衫",
            type="shirt",
            status=ItemStatus.ready,
            needs_wash=True,
            image_path="/internal/wardrobe/private.jpg",
            user_id=user.id,
            ai_raw_response={"internal": "not-for-model"},
        ),
        SimpleNamespace(
            id=uuid4(),
            name=None,
            type="shirt",
            status=ItemStatus.processing,
            needs_wash=False,
            image_path="/internal/wardrobe/processing.jpg",
            user_id=user.id,
            ai_raw_response=None,
        ),
    ]
    service.get_list.return_value = (items, 5)
    result = await wardrobe_tool.ainvoke({"item_type": " shirt ", "page": 2, "page_size": 2})

    service.get_list.assert_awaited_once()
    kwargs = service.get_list.await_args.kwargs
    assert kwargs["user_id"] == user.id
    assert kwargs["page"] == 2
    assert kwargs["page_size"] == 2
    assert isinstance(kwargs["filters"], ItemFilter)
    assert kwargs["filters"].model_dump() == ItemFilter(type="shirt").model_dump()
    assert kwargs["filters"].needs_wash is None
    assert kwargs["filters"].status is None
    assert kwargs["filters"].is_archived is False
    assert result == {
        "items": [
            {
                "id": str(items[0].id),
                "name": "白色衬衫",
                "type": "shirt",
                "status": "ready",
                "needs_wash": True,
            },
            {
                "id": str(items[1].id),
                "name": None,
                "type": "shirt",
                "status": "processing",
                "needs_wash": False,
            },
        ],
        "total": 5,
        "page": 2,
        "page_size": 2,
        "has_more": True,
    }
    encoded = json.dumps(result, ensure_ascii=False)
    assert json.loads(encoded) == result
    assert "/internal/" not in encoded
    assert "ai_raw_response" not in encoded
    assert "user_id" not in encoded


@pytest.mark.asyncio
async def test_empty_query_uses_default_pagination(bound_tool):
    wardrobe_tool, user, service = bound_tool
    result = await wardrobe_tool.ainvoke({"item_type": "shirt"})
    assert result == {
        "items": [],
        "total": 0,
        "page": 1,
        "page_size": DEFAULT_PAGE_SIZE,
        "has_more": False,
    }
    service.get_list.assert_awaited_once()
    kwargs = service.get_list.await_args.kwargs
    assert kwargs["user_id"] == user.id
    assert kwargs["page"] == 1
    assert kwargs["page_size"] == DEFAULT_PAGE_SIZE


@pytest.mark.parametrize(
    "invalid_parameters",
    [
        {"item_type": ""},
        {"item_type": "   "},
        {"item_type": "x" * 51},
        {"item_type": None},
        {"item_type": 123},
        {"page": 0},
        {"page": -1},
        {"page": True},
        {"page": "1"},
        {"page_size": 0},
        {"page_size": MAX_PAGE_SIZE + 1},
        {"page_size": 1.5},
        {"page_size": False},
    ],
)
@pytest.mark.asyncio
async def test_invalid_parameters_fail_before_service_call(bound_tool, invalid_parameters):
    wardrobe_tool, _, service = bound_tool
    with pytest.raises(ValidationError):
        await wardrobe_tool.ainvoke({"item_type": "shirt", **invalid_parameters})
    service.get_list.assert_not_awaited()


@pytest.mark.parametrize("forbidden_parameter", ["user_id", "current_user", "db", "session"])
@pytest.mark.asyncio
async def test_model_cannot_override_bound_identity_or_session(bound_tool, forbidden_parameter):
    wardrobe_tool, _, service = bound_tool
    with pytest.raises(ValidationError):
        await wardrobe_tool.ainvoke({"item_type": "shirt", forbidden_parameter: str(uuid4())})
    service.get_list.assert_not_awaited()
