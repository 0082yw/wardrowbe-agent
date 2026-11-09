"""第 10 节：用模拟模型演示真实 ToolNode 工具调用循环，无需 LLM 或 API Key。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_tool_demo import query_wardrobe_items


def mock_model_node(state: MessagesState) -> dict[str, list[AIMessage]]:
    """用固定规则代替 LLM：提出工具请求，或根据工具结果生成回答。"""
    last_message = state["messages"][-1]
    if not isinstance(last_message.content, str):
        raise ValueError("本示例只处理纯文本消息")

    if isinstance(last_message, ToolMessage):
        # ToolNode 将工具返回的列表序列化为 JSON，模型节点从消息中读取它。
        items = json.loads(last_message.content)
        names = "、".join(item["name"] for item in items)
        return {"messages": [AIMessage(content=f"你的上衣有：{names}。")]}

    if not isinstance(last_message, HumanMessage):
        raise ValueError("模拟模型需要用户消息或工具结果")

    if "上衣" in last_message.content:
        # 这里只声明调用意图；工具尚未执行，id 用于关联后续 ToolMessage。
        request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": query_wardrobe_items.name,
                    "args": {"item_type": "上衣"},
                    "id": "call_wardrobe_tops_1",
                    "type": "tool_call",
                }
            ],
        )
        return {"messages": [request]}

    return {"messages": [AIMessage(content="你好！我可以帮你查询衣橱。")]}


def run_conversation(user_text: str) -> MessagesState:
    """每次使用独立消息 State；两条路径共享相同的图结构。"""
    # MessagesState 内置消息追加规则，节点只需返回新增消息。
    builder = StateGraph(MessagesState)
    builder.add_node("mock_model", mock_model_node)
    builder.add_node("tools", ToolNode([query_wardrobe_items], handle_tool_errors=False))
    builder.add_edge(START, "mock_model")
    builder.add_conditional_edges(
        "mock_model", tools_condition, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "mock_model")
    workflow = builder.compile()
    return workflow.invoke({"messages": [HumanMessage(content=user_text)]})


def test_wardrobe_tool_call() -> None:
    """上衣查询：模型提出请求，ToolNode 执行，模型读取结果后结束。"""
    # 单独检查模型输出请求的阶段，没有执行图，也没有取得任何工具结果。
    request_only = mock_model_node({"messages": [HumanMessage(content="我有哪些上衣？")]})
    assert len(request_only["messages"]) == 1
    assert len(request_only["messages"][0].tool_calls) == 1
    assert request_only["messages"][0].content == ""

    result = run_conversation("我有哪些上衣？")
    messages = result["messages"]
    assert [message.type for message in messages] == ["human", "ai", "tool", "ai"]
    request, tool_result, answer = messages[1:]
    assert isinstance(request, AIMessage)
    assert isinstance(tool_result, ToolMessage)
    assert isinstance(answer, AIMessage)

    requested_count = sum(
        len(message.tool_calls) for message in messages if isinstance(message, AIMessage)
    )
    executed_count = sum(isinstance(message, ToolMessage) for message in messages)
    assert requested_count == executed_count == 1, "应只请求并完成一次工具调用"
    assert request.tool_calls[0]["name"] == query_wardrobe_items.name
    assert request.tool_calls[0]["args"] == {"item_type": "上衣"}
    assert tool_result.tool_call_id == request.tool_calls[0]["id"]
    assert tool_result.name == query_wardrobe_items.name
    assert tool_result.status == "success"
    assert isinstance(tool_result.content, str)
    assert json.loads(tool_result.content) == [
        {"name": "白色衬衫", "type": "上衣"},
        {"name": "蓝色T恤", "type": "上衣"},
    ]
    assert answer.content == "你的上衣有：白色衬衫、蓝色T恤。"
    assert not answer.tool_calls, "最终回答不再请求工具，tools_condition 应结束流程"

    print("用例 1：上衣查询")
    print("消息顺序：", " -> ".join(message.type for message in messages))
    print("调用请求：", request.tool_calls)
    print("工具结果：", tool_result.content)
    print("最终回答：", answer.content)
    print(f"请求次数：{requested_count}；工具执行次数：{executed_count}")


def test_direct_reply() -> None:
    """普通问候：模型直接回答，tools_condition 不进入工具节点。"""
    result = run_conversation("你好！")
    messages = result["messages"]
    assert [message.type for message in messages] == ["human", "ai"]
    answer = messages[-1]
    assert isinstance(answer, AIMessage)
    assert answer.content == "你好！我可以帮你查询衣橱。"
    requested_count = sum(
        len(message.tool_calls) for message in messages if isinstance(message, AIMessage)
    )
    executed_count = sum(isinstance(message, ToolMessage) for message in messages)
    assert requested_count == executed_count == 0, "直接回答分支不应调用工具"

    print("用例 2：普通问候")
    print("消息顺序：", " -> ".join(message.type for message in messages))
    print("最终回答：", answer.content)
    print(f"请求次数：{requested_count}；工具执行次数：{executed_count}")


def main() -> None:
    test_wardrobe_tool_call()
    print()
    test_direct_reply()
    print("\n验证通过：两个独立用例的回答、调用次数和分支均正确。")


if __name__ == "__main__":
    main()
