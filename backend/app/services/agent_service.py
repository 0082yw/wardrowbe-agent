"""只读穿搭预览：注入模型，复用衣橱工具，不接管认证、事务或模型连接。"""

import json
from collections.abc import Awaitable
from copy import deepcopy
from inspect import isawaitable
from typing import Annotated, Literal, Protocol, Self
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_tools import create_query_my_wardrobe_items_tool, create_query_outfit_candidates_tool
from app.models.user import User
from app.services.agent_audit import current_audit

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class OutfitPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    item_ids: list[UUID] = Field(min_length=1, max_length=20)
    reasoning: ShortText
    style_notes: ShortText
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @field_validator("item_ids")
    @classmethod
    def unique_items(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("同一套预览不能重复包含同一衣物")
        return value


class AgentPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["success", "needs_clarification", "no_items", "invalid_result"]
    answer: ShortText
    outfits: list[OutfitPreview] = Field(default_factory=list, max_length=3)
    kind: Literal["items", "outfit"] = Field(default="items", exclude_if=lambda value: value == "items")
    scene: Literal["daily", "commute", "formal"] | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        if (self.status == "success") != bool(self.outfits):
            raise ValueError("只有成功结果可以包含非空穿搭预览")
        if self.kind == "outfit":
            if self.scene is None or any(len(o.item_ids) < 2 or o.title is None for o in self.outfits):
                raise ValueError("完整穿搭必须有场景、方案名和至少两件衣物")
            sets = [frozenset(o.item_ids) for o in self.outfits]
            if len(set(sets)) != len(sets):
                raise ValueError("不同穿搭不能完全重复")
        elif self.scene is not None or any(o.title is not None for o in self.outfits):
            raise ValueError("单品查询不能冒充完整穿搭")
        return self


class AgentModel(Protocol):
    """可信调用者注入的模型函数；只接收消息副本和工具定义，不接收用户或 Session。

    同步、异步函数均可。最终 AIMessage.content 必须是符合 AgentPreview 的 JSON。
    本节仅注入模拟模型；Service 自身不读取模型凭据、不创建 HTTP 客户端。
    """

    def __call__(
        self, messages: list[BaseMessage], *, tools: list[dict]
    ) -> AIMessage | Awaitable[AIMessage]: ...


def _invalid(answer: str) -> AgentPreview:
    return AgentPreview(status="invalid_result", answer=answer)


class AgentService:
    """每次 generate_preview 独立绑定工具和候选集，不保存会话或穿搭。

    调用者负责认证，并保证该请求的 Session 有效且不被其他请求并发使用。
    衣橱查询保持原列表语义；预览只选本次查到且 ready、无需清洗的衣物。
    """

    def __init__(
        self, *, model: AgentModel, max_model_calls: int = 6, max_tool_calls: int = 8
    ) -> None:
        for limit in (max_model_calls, max_tool_calls):
            if type(limit) is not int or limit < 1:
                raise ValueError("模型和工具调用上限必须是正整数")
        self.model = model
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls

    async def generate_preview(
        self, message: str, *, current_user: User, db: AsyncSession
    ) -> AgentPreview:
        """返回可用 model_dump(mode="json") 序列化的预览；不保存、不提交。"""
        if not isinstance(message, str) or not message.strip() or len(message) > 10000:
            raise ValueError("用户需求必须是 1 至 10000 字符的非空文本")

        from app.services.wardrobe_query_agent import WardrobeQueryAgentModel
        from app.services.outfit_engine import route_outfit

        outfit_mode = isinstance(self.model, WardrobeQueryAgentModel) and route_outfit(message) is not None
        factory = create_query_outfit_candidates_tool if outfit_mode else create_query_my_wardrobe_items_tool
        wardrobe_tool = factory(current_user=current_user, db=db)
        audit = current_audit.get()
        if audit is not None:
            audit.bind_user(current_user.id, tool=True)
        definitions = [convert_to_openai_tool(wardrobe_tool)]
        # 工具子集 Schema 可能丢失 extra=forbid，使用原输入模型保留完整约束。
        definitions[0]["function"]["parameters"] = wardrobe_tool.get_input_schema().model_json_schema()
        tool_node = ToolNode([wardrobe_tool], handle_tool_errors=True)
        # 只从本次 ToolNode 的成功结果登记候选；模型消息不能写入这个集合。
        candidates: dict[UUID, bool] = {}
        seen_call_ids: set[str] = set()
        model_calls = tool_calls = successful_queries = 0
        failure: AgentPreview | None = None

        def stop(answer: str) -> dict:
            nonlocal failure
            failure = _invalid(answer)
            return {"messages": []}

        async def model_node(state: MessagesState) -> dict:
            nonlocal model_calls
            if model_calls >= self.max_model_calls:
                return stop("模型执行轮次达到上限，本次未生成有效预览。")
            model_calls += 1
            try:
                # 防止注入的模型原地改写图内消息、工具结果或定义。
                response = self.model(
                    [entry.model_copy(deep=True) for entry in state["messages"]],
                    tools=deepcopy(definitions),
                )
                if isawaitable(response):
                    response = await response
                if not isinstance(response, AIMessage) or response.invalid_tool_calls:
                    return stop("模型没有返回有效的 AIMessage，本次结果无效。")
                return {"messages": [response.model_copy(deep=True)]}
            except Exception:
                return stop("模型节点执行失败，本次未生成有效预览。")

        async def tools_node(state: MessagesState) -> dict:
            nonlocal tool_calls, successful_queries
            requests = state["messages"][-1].tool_calls
            ids = [request["id"] for request in requests]
            if (
                any(not call_id or not call_id.strip() for call_id in ids)
                or len(ids) != len(set(ids))
                or seen_call_ids.intersection(ids)
                or any(request["name"] != wardrobe_tool.name for request in requests)
            ):
                return stop("工具名称或调用 ID 无效，本次未执行这批工具请求。")
            if tool_calls + len(requests) > self.max_tool_calls:
                return stop("工具调用次数达到上限，本次未生成有效预览。")
            seen_call_ids.update(ids)
            results = []
            for request in requests:
                tool_calls += 1
                tool_audit = (
                    audit.tool_started(request["args"].get("item_type"), wardrobe_scope=outfit_mode)
                    if audit is not None else None
                )
                try:
                    # ToolNode 会并发处理一批调用，因此每次只交给它一个请求并等待完成。
                    output = await tool_node.ainvoke(
                        {"messages": [AIMessage(content="", tool_calls=[request])]}
                    )
                    result = output["messages"][0]
                    if (
                        not isinstance(result, ToolMessage)
                        or result.tool_call_id != request["id"]
                        or result.name != wardrobe_tool.name
                        or result.status != "success"
                    ):
                        if audit is not None and tool_audit is not None:
                            audit.tool_result(tool_audit, None, linked=False)
                        return stop("工具参数或执行失败，本次未生成有效预览。")
                    payload = json.loads(result.content)
                    if audit is not None and tool_audit is not None:
                        audit.tool_result(tool_audit, payload, linked=True)
                    for item in payload["items"]:
                        candidates[UUID(item["id"])] = (
                            item["status"] == "ready" and item["needs_wash"] is False
                            and item.get("is_archived", False) is False
                        )
                    successful_queries += 1
                    results.append(result)
                except Exception:
                    # 不把异常详情、内部路径或数据库信息透传给模型及用户。
                    return stop("衣橱工具执行失败，本次未生成有效预览。")
            return {"messages": results}

        graph = StateGraph(MessagesState)
        graph.add_node("model", model_node)
        graph.add_node("tools", tools_node)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
        graph.add_conditional_edges(
            "tools",
            lambda state: END if failure is not None else "model",
            {"model": "model", END: END},
        )
        system = SystemMessage(
            content=(
                "你是只读衣橱助手。使用已提供工具查询真实衣物，不执行保存操作。"
                "工具参数不得包含用户身份或数据库 Session；item_type 使用项目具体类型。"
                "成功预览只能引用本次工具实际返回且 ready、无需清洗的衣物 ID。"
                "信息不足时追问；只有实际查询后没有可用候选才返回 no_items，"
                "此状态仅表示已查询范围，不得据此宣称整个衣橱为空。"
                "不推测未知颜色、材质或天气。最终只输出符合以下结构的 JSON："
                + json.dumps(AgentPreview.model_json_schema(), ensure_ascii=False)
            )
        )
        state = await graph.compile().ainvoke(
            {"messages": [system, HumanMessage(content=message.strip())]},
            config={"recursion_limit": self.max_model_calls * 2 + 2},
        )
        if failure is not None:
            return failure

        final = state["messages"][-1]
        if not isinstance(final, AIMessage) or final.tool_calls or not isinstance(final.content, str):
            return _invalid("模型未返回有效的最终回答。")
        try:
            preview = AgentPreview.model_validate_json(final.content)
        except ValueError:
            return _invalid("模型最终回答的结构无效，本次未生成穿搭预览。")
        if preview.status == "success" and any(
            not candidates.get(item_id, False)
            for outfit in preview.outfits
            for item_id in outfit.item_ids
        ):
            return _invalid("推荐包含未查询确认或当前不可穿的衣物，本次结果无效。")
        if outfit_mode:
            # 组合、状态和 ID 必须来自确定性控制器，不能由模型消息自行声明。
            if self.model.authoritative_preview is None or preview != self.model.authoritative_preview:
                return _invalid("穿搭结果与后端已验证组合不一致。")
        elif preview.kind != "items":
            return _invalid("未经后端组合引擎验证的完整穿搭无效。")
        if preview.status == "no_items" and (
            not successful_queries or (not outfit_mode and any(candidates.values()))
        ):
            return _invalid("没有足够的查询证据支持无可用衣物的结论。")
        return preview
