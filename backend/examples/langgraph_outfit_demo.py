"""第 7 节：使用关键词规则和固定回复的最小 LangGraph 工作流。"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class OutfitState(TypedDict):
    user_request: str
    request_type: str
    reply: str


def classify_request(state: OutfitState) -> dict[str, str]:
    """读取用户需求，只返回本节点更新的分类字段。"""
    request = state["user_request"]

    if "下雨" in request or "雨天" in request:
        request_type = "rainy"
    elif "面试" in request or "正式" in request or "商务" in request:
        request_type = "formal"
    else:
        request_type = "general"

    return {"request_type": request_type}


def generate_demo_reply(state: OutfitState) -> dict[str, str]:
    """根据分类选择固定回复，不调用大模型。"""
    replies = {
        "rainy": "建议穿轻便防水外套、舒适长裤和防滑鞋，通勤时带上雨伞。",
        "formal": "建议穿简洁的衬衫、深色长裤和干净的皮鞋，保持整洁得体。",
        "general": "建议穿舒适的上衣、休闲长裤和运动鞋，适合日常出行。",
    }
    return {"reply": replies[state["request_type"]]}


def main() -> None:
    builder = StateGraph(OutfitState)
    builder.add_node("classify_request", classify_request)
    builder.add_node("generate_demo_reply", generate_demo_reply)

    builder.add_edge(START, "classify_request")
    builder.add_edge("classify_request", "generate_demo_reply")
    builder.add_edge("generate_demo_reply", END)

    workflow = builder.compile()
    initial_state: OutfitState = {
        "user_request": "明天下雨，帮我搭配一套通勤穿搭",
        "request_type": "",
        "reply": "",
    }
    result = workflow.invoke(initial_state)

    print("输入：", initial_state["user_request"])
    print("分类结果：", result["request_type"])
    print("回复：", result["reply"])


if __name__ == "__main__":
    main()
