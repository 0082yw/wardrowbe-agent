"""只读衣橱工具；身份与 Session 必须由已完成认证的后端调用者绑定。"""

from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.pagination import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.models.user import User
from app.schemas.item import ItemFilter
from app.services.agent_audit import current_audit
from app.services.item_service import ItemService


class WardrobeQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    item_type: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)
    ] = Field(description="精确匹配衣物类型，例如 shirt 或 t-shirt；不使用中文大类")
    page: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int = Field(
        default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="每页衣物数量"
    )


def create_query_my_wardrobe_items_tool(
    *, current_user: User, db: AsyncSession
) -> BaseTool:
    """在请求内绑定可信身份和 Session；工具不接管认证或事务生命周期。

    每个请求单独创建工具，仅在所绑定 Session 有效期间使用；不要全局缓存。
    同一 AsyncSession 不能被多个并行工具调用共享，调用方应控制执行范围。
    """
    user_id = current_user.id
    item_service = ItemService(db)

    @tool("query_my_wardrobe_items", args_schema=WardrobeQueryInput)
    async def query_my_wardrobe_items(
        item_type: str, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE
    ) -> dict:
        """按类型查询当前用户未归档的衣物，包含待洗衣物和各处理状态。

        类型精确匹配，不自动将中文“上衣”映射为某个具体类型。
        """
        audit = current_audit.get()
        if audit is not None:
            audit.query_started(item_type, user_id=user_id)
        items, total = await item_service.get_list(
            user_id=user_id,
            filters=ItemFilter(type=item_type),
            page=page,
            page_size=page_size,
        )
        # 只投影必要字段；不复用会包含图片路径和内部元数据的完整 ItemResponse。
        return {
            "items": [
                {
                    "id": str(item.id),
                    "name": item.name,
                    "type": item.type,
                    "status": str(item.status),
                    "needs_wash": item.needs_wash,
                }
                for item in items
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": page * page_size < total,
        }

    return query_my_wardrobe_items


class OutfitQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=100, ge=1, le=MAX_PAGE_SIZE)


def create_query_outfit_candidates_tool(*, current_user: User, db: AsyncSession) -> BaseTool:
    """在同一请求内读取有限候选；复用列表权限和归档语义，不加载图片或写入。"""
    user_id = current_user.id
    service = ItemService(db)

    @tool("query_my_outfit_candidates", args_schema=OutfitQueryInput)
    async def query_my_outfit_candidates(page: int = 1, page_size: int = 100) -> dict:
        """读取当前用户未归档衣物，供后端组合引擎筛选；不由模型挑选 ID。"""
        audit = current_audit.get()
        if audit is not None:
            audit.query_started(None, user_id=user_id)
        items, total = await service.get_list(
            user_id=user_id, filters=ItemFilter(), page=page, page_size=page_size,
        )
        return {
            "items": [{
                "id": str(item.id), "name": item.name, "type": item.type,
                "status": str(item.status), "needs_wash": item.needs_wash,
                "is_archived": item.is_archived,
                "colors": item.colors or [], "primary_color": item.primary_color,
                "formality": item.formality,
            } for item in items],
            "total": total, "page": page, "page_size": page_size,
            "has_more": page * page_size < total,
        }

    return query_my_outfit_candidates
