"""第 8 节：在 LangGraph 工作流中显式调用只读模拟衣橱工具。"""

from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph


MOCK_WARDROBE: list[dict[str, str]] = [
    {"name": "白色衬衫", "type": "上衣"},
    {"name": "蓝色T恤", "type": "上衣"},
    {"name": "黑色长裤", "type": "裤子"},
    {"name": "防水外套", "type": "外套"},
]


@tool
def query_wardrobe_items(item_type: str) -> list[dict[str, str]]:
    """按衣物类型查询模拟衣橱，例如上衣、裤子或外套；无匹配时返回空列表。"""
    # 返回匹配记录的副本，避免调用方修改查询结果时影响模拟数据。
    return [
        item.copy()
        for item in MOCK_WARDROBE
        if item["type"] == item_type
    ]


class WardrobeState(TypedDict):
    item_type: str
    items: list[dict[str, str]]


def query_wardrobe_node(state: WardrobeState) -> dict[str, list[dict[str, str]]]:
    items = query_wardrobe_items.invoke({"item_type": state["item_type"]})
    return {"items": items}


def main() -> None:
    builder = StateGraph(WardrobeState)
    builder.add_node("query_wardrobe", query_wardrobe_node)
    builder.add_edge(START, "query_wardrobe")
    builder.add_edge("query_wardrobe", END)
    workflow = builder.compile()

    test_cases = [
        (
            "上衣",
            [
                {"name": "白色衬衫", "type": "上衣"},
                {"name": "蓝色T恤", "type": "上衣"},
            ],
        ),
        ("鞋子", []),
    ]
    original_items = [item.copy() for item in MOCK_WARDROBE]

    for item_type, expected_items in test_cases:
        initial_state: WardrobeState = {"item_type": item_type, "items": []}
        result = workflow.invoke(initial_state)

        assert result["item_type"] == item_type, "查询类型应保持不变"
        assert result["items"] == expected_items, "查询结果与预期不一致"
        print("查询参数：", {"item_type": item_type})
        print("查询结果：", result["items"])
        print()

    assert MOCK_WARDROBE == original_items, "模拟衣橱数据不应被修改"
    print("验证通过：两种查询结果正确，模拟衣橱未被修改。")


if __name__ == "__main__":
    main()
