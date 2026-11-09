"""第 12 节：用 MockTransport 验证 HTTP Tool Calling 消息适配，全程离线。"""

import json

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_tool_demo import query_wardrobe_items


def to_api_messages(messages: list[BaseMessage]) -> list[dict]:
    """把图内消息转换为 Chat Completions 请求中的 messages。"""
    converted = []
    for message in messages:
        if not isinstance(message.content, str):
            raise ValueError("本例只处理纯文本消息")
        if isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": message.content})
        elif isinstance(message, HumanMessage):
            converted.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            api_message = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                api_message["content"] = message.content or None
                api_message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"], ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            converted.append(api_message)
        elif isinstance(message, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                }
            )
        else:
            raise TypeError(f"不支持的消息类型：{type(message).__name__}")
    return converted


def from_api_response(data: dict) -> AIMessage:
    """保留工具请求；无工具请求时，只接受正常结束且有文本的回答。"""
    choice = data["choices"][0]
    message = choice["message"]
    if message["role"] != "assistant":
        raise ValueError("模型响应必须是 assistant 消息")
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("模型文本格式不正确")
    calls = []
    for call in message.get("tool_calls") or []:
        if call["type"] != "function" or not call["id"]:
            raise ValueError("工具调用必须包含 function 类型及调用 ID")
        arguments = json.loads(call["function"]["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        calls.append(
            {
                "id": call["id"],
                "name": call["function"]["name"],
                "args": arguments,
                "type": "tool_call",
            }
        )
    finish_reason = choice["finish_reason"]
    if calls:
        if finish_reason != "tool_calls":
            raise ValueError("工具请求的结束标记不正确")
    elif finish_reason != "stop" or not content.strip():
        raise ValueError("响应不是有效最终回答，可能为空或已截断")
    # 有 tool_calls 时 content 为空是合法的，不能当成空响应错误。
    return AIMessage(
        content=content,
        tool_calls=calls,
        response_metadata={"finish_reason": finish_reason},
    )


def main() -> None:
    mock_url = "https://mock-llm.invalid/v1/chat/completions"
    mock_model = "mock-wardrobe-model"
    call_id = "call_wardrobe_tops_12"
    # 从第 8 节工具导出 schema，不重新声明工具参数或复制查询实现。
    tools = [convert_to_openai_tool(query_wardrobe_items)]
    function = tools[0]["function"]
    assert tools[0]["type"] == "function"
    assert function["name"] == query_wardrobe_items.name
    assert function["description"] == query_wardrobe_items.description
    assert function["parameters"]["properties"]["item_type"]["type"] == "string"
    assert function["parameters"]["required"] == ["item_type"]

    initial_messages = [
        SystemMessage(content="你是衣橱助手，请根据工具查询结果回答。"),
        HumanMessage(content="我有哪些上衣？"),
    ]
    expected_initial_api = [
        {"role": "system", "content": "你是衣橱助手，请根据工具查询结果回答。"},
        {"role": "user", "content": "我有哪些上衣？"},
    ]
    expected_items = [
        {"name": "白色衬衫", "type": "上衣"},
        {"name": "蓝色T恤", "type": "上衣"},
    ]
    expected_answer = "你的上衣有：白色衬衫、蓝色T恤。"
    wire_call = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": query_wardrobe_items.name,
            "arguments": json.dumps({"item_type": "上衣"}, ensure_ascii=False),
        },
    }
    captured_requests: list[dict] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """这里只模拟模型响应；实际衣橱查询必须由图内 ToolNode 执行。"""
        assert request.method == "POST"
        assert str(request.url) == mock_url
        assert request.headers["content-type"] == "application/json"
        body = json.loads(request.content)
        assert body["model"] == mock_model
        assert body["stream"] is False
        assert body["tools"] == tools
        captured_requests.append(body)
        request_number = len(captured_requests)
        messages = body["messages"]

        if request_number == 1:
            assert messages == expected_initial_api
            response_message = {"role": "assistant", "content": None, "tool_calls": [wire_call]}
            finish_reason = "tool_calls"
        elif request_number == 2:
            assert [message["role"] for message in messages] == [
                "system", "user", "assistant", "tool"
            ]
            assert messages[:2] == expected_initial_api
            assert messages[2]["content"] is None
            assert messages[2]["tool_calls"] == [wire_call]
            arguments = messages[2]["tool_calls"][0]["function"]["arguments"]
            assert isinstance(arguments, str), "API 工具参数应为 JSON 字符串"
            assert json.loads(arguments) == {"item_type": "上衣"}
            tool_message = messages[3]
            assert tool_message["tool_call_id"] == call_id
            assert isinstance(tool_message["content"], str)
            items = json.loads(tool_message["content"])
            assert items == expected_items, "回传结果必须来自实际衣橱工具"
            names = "、".join(item["name"] for item in items)
            response_message = {"role": "assistant", "content": f"你的上衣有：{names}。"}
            finish_reason = "stop"
        else:
            raise AssertionError("本例应恰好发送两次 HTTP 请求")

        print(f"MockTransport 请求 {request_number}：", [m["role"] for m in messages])
        return httpx.Response(
            200,
            json={
                "id": f"mock-response-{request_number}",
                "object": "chat.completion",
                "created": 0,
                "model": mock_model,
                "choices": [
                    {"index": 0, "message": response_message, "finish_reason": finish_reason}
                ],
            },
        )

    # 客户端固定绑定 MockTransport；禁用环境代理，不创建真实网络传输。
    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport, trust_env=False, timeout=10) as client:

        def model_node(state: MessagesState) -> dict[str, list[AIMessage]]:
            response = client.post(
                mock_url,
                json={
                    "model": mock_model,
                    "messages": to_api_messages(state["messages"]),
                    "tools": tools,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return {"messages": [from_api_response(response.json())]}

        builder = StateGraph(MessagesState)
        builder.add_node("model", model_node)
        builder.add_node("tools", ToolNode([query_wardrobe_items], handle_tool_errors=False))
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
        builder.add_edge("tools", "model")
        result = builder.compile().invoke(
            {"messages": initial_messages}, config={"recursion_limit": 6}
        )

    messages = result["messages"]
    assert [message.type for message in messages] == ["system", "human", "ai", "tool", "ai"]
    request_message, tool_result, final_answer = messages[2:]
    assert isinstance(request_message, AIMessage)
    assert request_message.content == ""
    assert request_message.tool_calls == [
        {"name": query_wardrobe_items.name, "args": {"item_type": "上衣"},
         "id": call_id, "type": "tool_call"}
    ]
    assert request_message.response_metadata["finish_reason"] == "tool_calls"
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == call_id
    assert tool_result.name == query_wardrobe_items.name
    assert tool_result.status == "success"
    assert json.loads(tool_result.content) == expected_items
    assert sum(isinstance(message, ToolMessage) for message in messages) == 1
    assert isinstance(final_answer, AIMessage)
    assert final_answer.content == expected_answer
    assert not final_answer.tool_calls
    assert final_answer.response_metadata["finish_reason"] == "stop"
    assert len(captured_requests) == 2, "两次 HTTP 请求都必须到达 MockTransport"

    print("图内消息顺序：", " -> ".join(message.type for message in messages))
    print("工具调用 ID：", tool_result.tool_call_id)
    print("工具返回结果：", tool_result.content)
    print("最终回答：", final_answer.content)
    print("验证通过：HTTP 请求 2 次，工具执行 1 次，消息转换和断言全部通过。")


if __name__ == "__main__":
    main()
