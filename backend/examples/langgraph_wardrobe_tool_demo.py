"""第 15 节：将业务衣橱工具接入 LangGraph，业务查询和 Session 均使用 Mock。

在 backend 目录运行：.venv/bin/python -B -m examples.langgraph_wardrobe_tool_demo
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy.ext.asyncio import AsyncSession

from app import agent_tools
from app.models.item import ItemStatus


def create_mock_bound_tool() -> tuple[BaseTool, SimpleNamespace, Mock, Mock]:
    """模拟可信后端调用者；不执行认证，也不创建真实数据库 Session。"""
    current_user = SimpleNamespace(id=uuid4())
    db = Mock(spec=AsyncSession)
    service = Mock(spec=agent_tools.ItemService)
    service.get_list = AsyncMock(
        return_value=(
            [
                SimpleNamespace(
                    id=uuid4(), name="白色衬衫", type="shirt",
                    status=ItemStatus.ready, needs_wash=True,
                    image_path="/internal/wardrobe/white.jpg",
                ),
                SimpleNamespace(
                    id=uuid4(), name="蓝色衬衫", type="shirt",
                    status=ItemStatus.processing, needs_wash=False,
                    image_path="/internal/wardrobe/blue.jpg",
                ),
            ],
            2,
        )
    )
    # 必须先替换构造器，再调用原有工具工厂；工具闭包将持有这个 Mock 服务。
    with patch.object(agent_tools, "ItemService", return_value=service) as constructor:
        wardrobe_tool = agent_tools.create_query_my_wardrobe_items_tool(
            current_user=current_user, db=db
        )
        constructor.assert_called_once_with(db)
    return wardrobe_tool, current_user, db, service


def build_workflow(
    wardrobe_tool: BaseTool, *, mock_tool_args: dict | None = None
) -> CompiledStateGraph:
    """保留真实工具及校验，只用本地规则模拟模型的选择和回答。

    mock_tool_args 用于测试不可信模型输出，不是注入身份或 Session 的入口。
    """
    arguments = {"item_type": "shirt"} if mock_tool_args is None else dict(mock_tool_args)

    def model_node(state: MessagesState) -> dict[str, list[AIMessage]]:
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            if last_message.status == "error":
                # 参数被拒绝后明确结束，不删除非法参数后偷偷重试。
                answer = "工具调用失败：参数或执行不符合要求，本次未取得衣物结果。"
            else:
                if not isinstance(last_message.content, str):
                    raise ValueError("本示例只接受 JSON 文本工具结果")
                payload = json.loads(last_message.content)
                if payload["items"]:
                    names = "、".join(item["name"] or item["type"] for item in payload["items"])
                    answer = f"本页衣物：{names}。"
                    if payload["has_more"]:
                        answer += "还有更多衣物，可继续分页查询。"
                else:
                    answer = "没有找到该类型的未归档衣物。"
            return {"messages": [AIMessage(content=answer)]}

        if not isinstance(last_message, HumanMessage) or not isinstance(last_message.content, str):
            raise ValueError("模拟模型需要文本用户消息或工具结果")
        if "衬衫" not in last_message.content:
            return {"messages": [AIMessage(content="你好！我可以帮你查询衬衫。")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": wardrobe_tool.name,
                            "args": arguments,
                            "id": "call_wardrobe_query_15",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("model", model_node)
    # 校验异常成为 status=error 的 ToolMessage；原工具的权限和参数约束保持不变。
    builder.add_node("tools", ToolNode([wardrobe_tool], handle_tool_errors=True))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")
    return builder.compile()


async def main() -> None:
    wardrobe_tool, current_user, db, service = create_mock_bound_tool()
    result = await build_workflow(wardrobe_tool).ainvoke(
        {"messages": [HumanMessage(content="我有哪些衬衫？")]},
        config={"recursion_limit": 6},
    )
    messages = result["messages"]
    assert [message.type for message in messages] == ["human", "ai", "tool", "ai"]
    request, tool_result, answer = messages[1:]
    assert isinstance(request, AIMessage)
    assert isinstance(tool_result, ToolMessage)
    assert isinstance(answer, AIMessage)
    assert sum(
        len(message.tool_calls) for message in messages if isinstance(message, AIMessage)
    ) == 1
    assert tool_result.tool_call_id == request.tool_calls[0]["id"]
    assert tool_result.name == wardrobe_tool.name
    assert tool_result.status == "success"
    service.get_list.assert_awaited_once()
    assert service.get_list.await_args.kwargs["user_id"] == current_user.id
    assert answer.content == "本页衣物：白色衬衫、蓝色衬衫。"
    assert not answer.tool_calls
    assert db.mock_calls == [], "不能查询数据库或改变 Session 生命周期"

    print("消息顺序：", " -> ".join(message.type for message in messages))
    print("工具名称：", wardrobe_tool.name)
    print("工具结果关联 ID：", tool_result.tool_call_id)
    print("工具结果：", tool_result.content)
    print("最终回答：", answer.content)
    print("验证通过：业务工具执行 1 次，Mock 服务查询 1 次，数据库操作 0 次。")


if __name__ == "__main__":
    asyncio.run(main())
