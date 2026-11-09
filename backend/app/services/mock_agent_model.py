"""无网络、无状态的模拟模型；仅演示工具调用，不能代替真实自然语言理解。"""

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def _reply(status: str, answer: str, items: list | None = None) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "status": status,
                "answer": answer,
                "outfits": (
                    [{
                        "item_ids": [item["id"] for item in items],
                        "reasoning": "模拟模型仅选择本次工具查询到的可用衣物。",
                        "style_notes": "这是接口演示，尚未评估天气、颜色或完整搭配。",
                    }]
                    if items else []
                ),
            },
            ensure_ascii=False,
        )
    )


def mock_agent_model(messages: list[BaseMessage], *, tools: list[dict]) -> AIMessage:
    """识别衬衫、长裤或精确类型词；其他需求先追问，不假装具备通用理解能力。"""
    last = messages[-1]
    if isinstance(last, ToolMessage):
        items = [
            item for item in json.loads(last.content)["items"]
            if item["status"] == "ready" and item["needs_wash"] is False
        ]
        if not items:
            return _reply("no_items", "模拟查询：本次查询范围没有可用于预览的衣物。")
        return _reply("success", "模拟预览：已查询到可用衣物，以下不是完整穿搭建议。", items[:1])

    if not isinstance(last, HumanMessage) or not isinstance(last.content, str):
        raise ValueError("模拟模型需要文本用户消息或工具结果")
    text = last.content.strip().lower()
    item_type = None
    if "衬衫" in text or text == "shirt":
        item_type = "shirt"
    elif "长裤" in text or text == "pants":
        item_type = "pants"
    if item_type is None:
        return _reply(
            "needs_clarification",
            "当前为模拟模式。请说明要查询的具体衣物，例如衬衫或长裤。",
        )
    return AIMessage(
        content="",
        tool_calls=[{
            "name": tools[0]["function"]["name"],
            "args": {"item_type": item_type},
            "id": "mock_wardrobe_query",
            "type": "tool_call",
        }],
    )
