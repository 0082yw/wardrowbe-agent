"""LangGraph 集成测试；加 --noconftest 运行，避开数据库和 Redis 集成夹具。"""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.pagination import MAX_PAGE_SIZE
from examples.langgraph_wardrobe_tool_demo import build_workflow, create_mock_bound_tool

pytestmark = pytest.mark.asyncio


@pytest.fixture
def context():
    wardrobe_tool, current_user, db, service = create_mock_bound_tool()
    yield SimpleNamespace(tool=wardrobe_tool, user=current_user, db=db, service=service)
    assert db.mock_calls == [], "集成测试不能访问数据库、提交、回滚或关闭 Session"


async def run_workflow(context, *, arguments=None, text="我有哪些衬衫？"):
    return await build_workflow(context.tool, mock_tool_args=arguments).ainvoke(
        {"messages": [HumanMessage(content=text)]}, config={"recursion_limit": 6}
    )


async def test_business_tool_executes_and_returns_linked_result(context):
    result = await run_workflow(context)
    messages = result["messages"]
    assert [message.type for message in messages] == ["human", "ai", "tool", "ai"]
    request, tool_result, answer = messages[1:]
    assert isinstance(request, AIMessage)
    assert request.tool_calls[0]["name"] == context.tool.name
    assert request.tool_calls[0]["args"] == {"item_type": "shirt"}
    assert sum(
        len(message.tool_calls) for message in messages if isinstance(message, AIMessage)
    ) == 1
    assert sum(isinstance(message, ToolMessage) for message in messages) == 1
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == request.tool_calls[0]["id"]
    assert tool_result.name == context.tool.name
    assert tool_result.status == "success"
    assert isinstance(tool_result.content, str)
    payload = json.loads(tool_result.content)
    assert [item["name"] for item in payload["items"]] == ["白色衬衫", "蓝色衬衫"]
    assert [item["status"] for item in payload["items"]] == ["ready", "processing"]
    assert payload["items"][0]["needs_wash"] is True
    assert payload["total"] == 2
    assert payload["page"] == 1
    assert payload["page_size"] == 20
    assert payload["has_more"] is False
    assert all(
        set(item) == {"id", "name", "type", "status", "needs_wash"}
        for item in payload["items"]
    )
    assert "/internal/" not in tool_result.content
    assert isinstance(answer, AIMessage)
    assert answer.content == "本页衣物：白色衬衫、蓝色衬衫。"
    assert not answer.tool_calls

    context.service.get_list.assert_awaited_once()
    kwargs = context.service.get_list.await_args.kwargs
    assert kwargs["user_id"] == context.user.id
    assert kwargs["filters"].type == "shirt"
    assert kwargs["filters"].is_archived is False
    assert kwargs["filters"].needs_wash is None
    assert kwargs["filters"].status is None


async def test_greeting_ends_without_tool_execution(context):
    result = await run_workflow(context, text="你好！")
    messages = result["messages"]
    assert [message.type for message in messages] == ["human", "ai"]
    assert messages[-1].content == "你好！我可以帮你查询衬衫。"
    assert not messages[-1].tool_calls
    context.service.get_list.assert_not_awaited()


async def test_empty_tool_result_returns_clear_answer(context):
    context.service.get_list.return_value = ([], 0)
    result = await run_workflow(context)
    tool_result = result["messages"][2]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.status == "success"
    assert tool_result.tool_call_id == result["messages"][1].tool_calls[0]["id"]
    payload = json.loads(tool_result.content)
    assert payload["items"] == []
    assert payload["total"] == 0
    assert payload["has_more"] is False
    assert result["messages"][-1].content == "没有找到该类型的未归档衣物。"
    assert not result["messages"][-1].tool_calls
    context.service.get_list.assert_awaited_once()


@pytest.mark.parametrize("forbidden_parameter", ["user_id", "current_user", "db", "session"])
async def test_toolnode_rejects_identity_and_session_overrides(context, forbidden_parameter):
    result = await run_workflow(
        context, arguments={"item_type": "shirt", forbidden_parameter: str(uuid4())}
    )
    messages = result["messages"]
    assert [message.type for message in messages] == ["human", "ai", "tool", "ai"]
    tool_result = messages[2]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == messages[1].tool_calls[0]["id"]
    assert tool_result.status == "error"
    assert forbidden_parameter in tool_result.content
    assert messages[-1].content.startswith("工具调用失败：")
    assert not messages[-1].tool_calls
    context.service.get_list.assert_not_awaited()

    # 同一绑定工具仍可合法查询，且使用原有用户；非法参数没有改写绑定上下文。
    valid_result = await run_workflow(context)
    assert valid_result["messages"][2].status == "success"
    context.service.get_list.assert_awaited_once()
    assert context.service.get_list.await_args.kwargs["user_id"] == context.user.id


async def test_toolnode_keeps_existing_pagination_limit(context):
    result = await run_workflow(
        context, arguments={"item_type": "shirt", "page_size": MAX_PAGE_SIZE + 1}
    )
    tool_result = result["messages"][2]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.status == "error"
    assert "page_size" in tool_result.content
    assert result["messages"][-1].content.startswith("工具调用失败：")
    assert not result["messages"][-1].tool_calls
    context.service.get_list.assert_not_awaited()
