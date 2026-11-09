"""确定性查询集成回归；真实 Service/ToolNode，HTTP、Session 和查询服务均为 Mock。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest

from examples import langgraph_ollama_agent_demo as demo

pytestmark = pytest.mark.asyncio
ITEM_ID = "10000000-0000-4000-8000-000000000001"
OTHER_ID = "20000000-0000-4000-8000-000000000001"


def response(content=None, *, calls=None, finish=None):
    return httpx.Response(200, json={"choices": [{
        "message": {"role": "assistant", "content": content, "tool_calls": calls or []},
        "finish_reason": finish or ("tool_calls" if calls else "stop"),
    }]})


def text(**changes):
    data = {
        "answer": "本次查询到两件衬衫，先展示第一件作为单品预览。",
        "reasoning": "这件衬衫来自本次查询结果。",
        "style_notes": "如需完整搭配，请补充使用场景或下装需求。",
    }
    data.update(changes)
    return json.dumps(data)


def item(item_id=ITEM_ID, *, name="测试衬衫", status="ready", needs_wash=False):
    return SimpleNamespace(id=item_id, name=name, type="shirt", status=status, needs_wash=needs_wash)


def model(handler):
    def checked(request):
        body = json.loads(request.content)
        assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
        assert body["model"] == "qwen3:0.6b-cpu"
        assert body["reasoning_effort"] == "none" and body["max_tokens"] == 256
        assert body["tools"] == [] and "response_format" not in body
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        return handler(body)
    return demo.OllamaTextModel(transport=httpx.MockTransport(checked))


async def test_explicit_query_executes_tool_before_model_and_backend_builds_preview():
    from app import agent_tools

    seen = []

    def handle(body):
        assert agent_tools.ItemService.return_value.get_list.await_count == 1
        seen.append(body)
        context = json.loads(body["messages"][1]["content"])
        assert context["item"] == {"name": "测试白色衬衫", "type": "shirt"}
        assert ITEM_ID not in body["messages"][1]["content"]
        assert "output_fields" not in context
        assert context["matched_count"] == context["available_count"] == 2
        assert body["messages"][0]["content"] == demo.TEXT_MODEL_INSTRUCTIONS
        assert "可省略；提供时必须为非空简短文本" not in json.dumps(body, ensure_ascii=False)
        return response(text())

    cpu_check = AsyncMock()
    run = await demo.run_scenario(model(handle), "查询我的衬衫", after_request=cpu_check)
    demo.verify_roundtrip(run)
    assert len(seen) == 1 and cpu_check.await_count == 1
    assert run.preview.outfits[0].item_ids == [UUID(ITEM_ID)]
    assert run.response.outfits[0].items[0].name == "测试白色衬衫"
    assert run.response.outfits[0].items[0].thumbnail_url is None
    assert run.db.mock_calls == []


@pytest.mark.parametrize("message", [
    "查询我的未知类型", "查询我的上衣", "查询我的衬衫和未知类型",
    "不要查询我的衬衫", "你好", "请搭配我的外套",
])
async def test_unsupported_or_ambiguous_requests_clarify_without_model_or_tool(message):
    seen = []
    run = await demo.run_scenario(
        model(lambda body: seen.append(body) or response(text())), message,
    )
    assert run.preview.status == run.response.status == "needs_clarification"
    assert run.preview.outfits == [] and seen == []
    assert run.trace.toolnode_calls == 0
    run.service.get_list.assert_not_awaited()


@pytest.mark.parametrize("fixtures", [
    [], [item(status="processing")], [item(needs_wash=True)],
])
async def test_empty_or_unwearable_query_returns_no_items_without_model(fixtures):
    seen = []
    run = await demo.run_scenario(
        model(lambda body: seen.append(body) or response(text())),
        demo.QUERY, fixture_items=fixtures,
    )
    assert run.preview.status == "no_items" and run.preview.outfits == []
    assert "本次衬衫查询" in run.preview.answer
    assert seen == [] and run.trace.toolnode_calls == run.service.get_list.await_count == 1


async def test_backend_selects_only_wearable_candidates_from_this_query():
    fixtures = [item(status="processing"), item(OTHER_ID, name="可穿测试衬衫")]
    run = await demo.run_scenario(model(lambda _: response(text())), demo.QUERY, fixture_items=fixtures)
    demo.verify_roundtrip(run)
    assert run.preview.outfits[0].item_ids == [UUID(OTHER_ID)]
    assert run.response.outfits[0].items[0].name == "可穿测试衬衫"


@pytest.mark.parametrize("injected", [
    {"item_ids": [OTHER_ID]}, {"item_ids": "1234567890"},
    {"status": "done"}, {"status": "success"},
    {"outfits": [{"item_ids": [OTHER_ID]}]},
    {"items": [{"id": OTHER_ID, "name": "模型伪造名称"}]},
    {"user_id": "other-user"},
])
async def test_model_business_fields_are_rejected_and_never_used(injected):
    run = await demo.run_scenario(model(lambda _: response(text(**injected))), demo.QUERY)
    assert run.preview.status == "invalid_result" and run.response.outfits == []
    assert run.trace.toolnode_calls == run.service.get_list.await_count == 1
    assert len(run.trace.inputs) == 1 and run.trace.errors == ["ValidationError"]
    assert run.service.get_list.await_args.kwargs["user_id"] == run.user.id


@pytest.mark.parametrize("missing", [
    (), ("reasoning",), ("style_notes",), ("reasoning", "style_notes"),
])
async def test_optional_text_is_completed_only_when_missing(missing):
    data = json.loads(text())
    for field in missing:
        data.pop(field)
    run = await demo.run_scenario(model(lambda _: response(json.dumps(data))), demo.QUERY)
    demo.verify_roundtrip(run)
    assert run.preview.outfits[0].reasoning == (
        demo.DEFAULT_REASONING if "reasoning" in missing else data["reasoning"]
    )
    assert run.preview.outfits[0].style_notes == (
        demo.DEFAULT_STYLE_NOTES if "style_notes" in missing else data["style_notes"]
    )


@pytest.mark.parametrize("field", ["answer", "reasoning", "style_notes"])
@pytest.mark.parametrize("value", [None, 123, {}, "x" * 1001])
async def test_explicit_invalid_text_is_rejected(field, value):
    run = await demo.run_scenario(model(lambda _: response(text(**{field: value}))), demo.QUERY)
    assert run.preview.status == "invalid_result" and run.preview.outfits == []
    assert run.trace.errors == ["ValidationError"] and run.trace.toolnode_calls == 1


async def test_missing_answer_remains_invalid():
    run = await demo.run_scenario(model(lambda _: response('{"reasoning":"测试"}')), demo.QUERY)
    assert run.preview.status == "invalid_result" and run.trace.errors == ["ValidationError"]


async def test_valid_text_is_trimmed_and_length_boundary_preserved():
    meaningful = "当前仅完成衬衫查询。"
    boundary = meaningful + " " * (1000 - len(meaningful))
    run = await demo.run_scenario(
        model(lambda _: response(text(reasoning="  这件衬衫来自本次查询结果。  ", style_notes=boundary))),
        demo.QUERY,
    )
    demo.verify_roundtrip(run)
    assert run.preview.outfits[0].reasoning == "这件衬衫来自本次查询结果。"
    assert run.preview.outfits[0].style_notes == meaningful


@pytest.mark.parametrize("content", [text(), '{"answer":'])
async def test_length_finish_rejects_even_complete_json_without_retry(content):
    seen = []
    run = await demo.run_scenario(
        model(lambda body: seen.append(body) or response(content, finish="length")), demo.QUERY,
    )
    assert run.preview.status == "invalid_result"
    assert run.trace.errors == ["response_truncated"] and len(seen) == 1
    assert run.trace.toolnode_calls == 1


@pytest.mark.parametrize("content", [
    '{"$defs":{"Preview":{"type":"object"}}}', '{"$defs":',
    '{"answer":', 'not json',
])
async def test_schema_fragments_and_invalid_json_remain_rejected(content):
    run = await demo.run_scenario(model(lambda _: response(content)), demo.QUERY)
    assert run.preview.status == "invalid_result" and run.trace.errors == ["ValidationError"]
    assert run.response.outfits == []


@pytest.mark.parametrize("fixtures", [
    [item("invalid-uuid")], [item(), item()],
    [item(status="unknown")], [item(needs_wash="false")],
])
async def test_invalid_or_duplicate_tool_candidates_fail_before_model(fixtures):
    seen = []
    run = await demo.run_scenario(
        model(lambda body: seen.append(body) or response(text())),
        demo.QUERY, fixture_items=fixtures,
    )
    assert run.preview.status == "invalid_result" and run.response.outfits == []
    assert seen == [] and run.trace.toolnode_calls == 1


@pytest.mark.parametrize("field,value", [
    ("tool_call_id", "unrelated"), ("name", "wrong_tool"), ("status", "error"),
])
async def test_service_still_rejects_unlinked_or_failed_tool_messages(monkeypatch, field, value):
    from langgraph.prebuilt import ToolNode

    original = ToolNode.ainvoke

    async def altered(self, *args, **kwargs):
        output = await original(self, *args, **kwargs)
        setattr(output["messages"][0], field, value)
        return output

    monkeypatch.setattr(ToolNode, "ainvoke", altered)
    seen = []
    run = await demo.run_scenario(
        model(lambda body: seen.append(body) or response(text())), demo.QUERY,
    )
    assert run.preview.status == "invalid_result" and seen == []
    assert run.trace.toolnode_calls == 1


@pytest.mark.parametrize("extra", [
    {"user_id": "other-user"}, {"db": "other-session"}, {"page": 0}, {"page_size": 101},
])
async def test_even_defective_backend_arguments_cannot_bypass_tool_schema(monkeypatch, extra):
    original = demo.query_call

    def defective(item_type):
        call = original(item_type)
        call["args"].update(extra)
        return call

    monkeypatch.setattr(demo, "query_call", defective)
    seen = []
    run = await demo.run_scenario(
        model(lambda body: seen.append(body) or response(text())), demo.QUERY,
    )
    assert run.preview.status == "invalid_result" and seen == []
    assert run.trace.toolnode_calls == 1
    run.service.get_list.assert_not_awaited()


async def test_user_session_and_candidates_are_request_scoped():
    first = await demo.run_scenario(model(lambda _: response(text())), demo.QUERY)
    second_user = UUID("30000000-0000-4000-8000-000000000099")
    second = await demo.run_scenario(
        model(lambda _: response(text())), demo.QUERY,
        fixture_items=[item(OTHER_ID)], user_id=second_user,
    )
    for run in [first, second]:
        demo.verify_roundtrip(run)
        assert run.service.get_list.await_args.kwargs["user_id"] == run.user.id
        run.constructor.assert_called_once_with(run.db)
        assert run.db.mock_calls == []
    assert first.db is not second.db
    assert first.preview.outfits[0].item_ids != second.preview.outfits[0].item_ids


async def test_model_cannot_trigger_a_second_tool_call():
    calls = [{
        "id": "llm_requested_again", "type": "function",
        "function": {"name": "query_my_wardrobe_items", "arguments": '{"item_type":"shirt"}'},
    }]
    run = await demo.run_scenario(model(lambda _: response(calls=calls)), demo.QUERY)
    assert run.preview.status == "invalid_result"
    assert run.trace.errors == ["invalid_response"]
    assert run.trace.toolnode_calls == run.service.get_list.await_count == 1


async def test_graph_round_limit_stops_before_any_text_request():
    seen = []
    run = await demo.run_scenario(
        model(lambda body: seen.append(body) or response(text())), demo.QUERY, max_model_calls=1,
    )
    assert run.preview.status == "invalid_result" and "轮次达到上限" in run.preview.answer
    assert seen == [] and run.trace.toolnode_calls == 1


async def test_shared_real_request_budget_is_not_spent_on_backend_graph_nodes():
    seen = []
    budget = demo.RequestBudget(1)
    provider = model(lambda body: seen.append(body) or response(text()))
    first = await demo.run_scenario(provider, demo.QUERY, budget=budget)
    second = await demo.run_scenario(provider, demo.QUERY, budget=budget)
    assert first.preview.status == "success" and second.preview.status == "invalid_result"
    assert len(seen) == budget.used == 1
    assert first.trace.toolnode_calls == second.trace.toolnode_calls == 1


async def test_cpu_check_failure_prevents_success_and_no_further_tools_execute():
    cpu_check = AsyncMock(side_effect=RuntimeError("CPU status could not be confirmed"))
    run = await demo.run_scenario(
        model(lambda _: response(text())), demo.QUERY, after_request=cpu_check,
    )
    assert run.preview.status == "invalid_result" and run.trace.safety_failed
    assert run.trace.toolnode_calls == 1
    cpu_check.assert_awaited_once()


async def test_timeout_cancels_text_request_and_still_checks_runtime():
    async def slow(messages, *, tools):
        await asyncio.sleep(1)

    cpu_check = AsyncMock()
    run = await demo.run_scenario(slow, demo.QUERY, timeout=0.01, after_request=cpu_check)
    assert run.preview.status == "invalid_result" and run.trace.errors == ["TimeoutError"]
    assert run.trace.toolnode_calls == 1
    cpu_check.assert_awaited_once()


async def test_memory_guard_can_stop_before_backend_tool_execution():
    def safety():
        raise demo.DemoSafetyError("memory_pressure")

    run = await demo.run_scenario(model(lambda _: response(text())), demo.QUERY, safety_check=safety)
    assert run.preview.status == "invalid_result" and run.trace.safety_failed
    assert run.trace.errors == ["memory_pressure"] and run.trace.toolnode_calls == 0
    run.service.get_list.assert_not_awaited()


async def test_standalone_demo_keeps_original_psi_threshold(monkeypatch):
    original_read = demo.Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemAvailable: 4000000 kB\n"
        if str(path) == "/proc/pressure/memory":
            return "some avg10=3.62 avg60=0.2\nfull avg10=3.62 avg60=0.2\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(demo.Path, "read_text", read)
    with pytest.raises(demo.DemoSafetyError) as exc:
        demo.check_memory()
    assert exc.value.code == "wsl_memory_pressure"


async def test_cli_still_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr("sys.argv", ["demo"])
    with pytest.raises(SystemExit) as exc:
        demo.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("field", ["answer", "reasoning", "style_notes"])
@pytest.mark.parametrize("value", ["", " \n "])
async def test_blank_strings_use_verified_query_defaults(field, value):
    run = await demo.run_scenario(model(lambda _: response(text(**{field: value}))), demo.QUERY)
    demo.verify_roundtrip(run)
    expected = {
        "answer": demo.default_query_answer(2),
        "reasoning": demo.DEFAULT_REASONING,
        "style_notes": demo.DEFAULT_STYLE_NOTES,
    }
    actual = run.preview.answer if field == "answer" else getattr(run.preview.outfits[0], field)
    assert actual == expected[field]
    assert field in run.trace.text_fallbacks
    assert len(run.trace.inputs) == run.trace.toolnode_calls == 1


@pytest.mark.parametrize("field,value", [
    ("answer", "测试白色衬衫"),
    ("reasoning", "可省略；提供时必须为非空简短文本"),
    ("style_notes", "可省略；提供时必须为非空简短文本"),
    ("answer", "只输出包含 answer、reasoning、style_notes 的 JSON 对象"),
    ("reasoning", '{"$defs":{"TextDraft":{"type":"object"}}}'),
    ("style_notes", "properties required additionalProperties"),
    ("answer", "已为你完成整套穿搭推荐。"),
    ("reasoning", "这件衬衫是纯棉材质，适合25度的晴天。"),
    ("style_notes", "搭配蓝色牛仔裤更显瘦。"),
    ("answer", "本次查询到三件衬衫。"),
    ("answer", "哈哈哈哈"),
    ("style_notes", "好的"),
])
async def test_echoes_metadata_and_unsupported_claims_are_not_user_text(field, value):
    run = await demo.run_scenario(model(lambda _: response(text(**{field: value}))), demo.QUERY)
    demo.verify_roundtrip(run)
    actual = run.preview.answer if field == "answer" else getattr(run.preview.outfits[0], field)
    expected = {
        "answer": demo.default_query_answer(2),
        "reasoning": demo.DEFAULT_REASONING,
        "style_notes": demo.DEFAULT_STYLE_NOTES,
    }
    assert actual == expected[field] and actual != value
    assert field in run.trace.text_fallbacks
    # 单字段质量回退不能覆盖其他有事实依据的文本。
    original = json.loads(text())
    for other in {"answer", "reasoning", "style_notes"} - {field}:
        retained = run.preview.answer if other == "answer" else getattr(run.preview.outfits[0], other)
        assert retained == original[other]
    assert len(run.trace.inputs) == 1 and run.trace.toolnode_calls == 1


async def test_all_blank_text_uses_factual_single_item_preview_defaults():
    run = await demo.run_scenario(
        model(lambda _: response(text(answer="", reasoning=" ", style_notes="\n"))), demo.QUERY,
    )
    demo.verify_roundtrip(run)
    assert run.preview.answer == demo.default_query_answer(2)
    assert run.preview.outfits[0].reasoning == demo.DEFAULT_REASONING
    assert run.preview.outfits[0].style_notes == demo.DEFAULT_STYLE_NOTES
    assert set(run.trace.text_fallbacks) == {"answer", "reasoning", "style_notes"}
    assert "单品预览" in run.preview.answer
    assert "当前仅完成衬衫查询" in run.preview.outfits[0].style_notes


@pytest.mark.parametrize("draft", [
    {
        "answer": "本次查询到两件衬衫，先展示第一件作为单品预览。",
        "reasoning": "这件衬衫来自本次查询结果。",
        "style_notes": "如需完整搭配，请补充使用场景或下装需求。",
    },
    {
        "answer": "本次衬衫查询已完成，可以先查看这件衬衫的单品预览。",
        "reasoning": "已从本次查询结果中选出这件作为预览。",
        "style_notes": "当前仅完成衬衫查询；请补充下装需求。",
    },
])
async def test_grounded_natural_language_is_preserved(draft):
    run = await demo.run_scenario(model(lambda _: response(json.dumps(draft))), demo.QUERY)
    demo.verify_roundtrip(run)
    assert run.preview.answer == draft["answer"]
    assert run.preview.outfits[0].reasoning == draft["reasoning"]
    assert run.preview.outfits[0].style_notes == draft["style_notes"]
    assert run.trace.text_fallbacks == []


async def test_defaults_distinguish_query_count_from_wearable_count():
    fixtures = [item(status="processing"), item(OTHER_ID)]
    run = await demo.run_scenario(
        model(lambda _: response(text(answer="", reasoning="", style_notes=""))),
        demo.QUERY, fixture_items=fixtures,
    )
    demo.verify_roundtrip(run)
    assert "2 件衬衫" in run.preview.answer and "1 件可用" in run.preview.answer
    assert run.preview.outfits[0].item_ids == [UUID(OTHER_ID)]


@pytest.mark.parametrize("content", ["", None])
async def test_empty_http_content_is_not_misclassified_as_quality_fallback(content):
    run = await demo.run_scenario(model(lambda _: response(content)), demo.QUERY)
    assert run.preview.status == "invalid_result"
    assert run.trace.errors == ["invalid_response"]
    assert run.trace.text_fallbacks == [] and run.trace.toolnode_calls == 1


async def test_observed_fenced_response_reaches_quality_fallback_without_retry():
    # 复现真实响应：先处理展示包装，再拒绝照抄指令作为用户文案。
    draft = {
        "answer": "根据您的查询，我预览了测试白色衬衫。",
        "reasoning": "基于您提供的查询事实，说明单品预览，不涉及其他信息。",
        "style_notes": "本次查询范围或后续需要补充的需求",
    }
    seen = []
    fence = chr(96) * 3
    content = fence + "json\n" + json.dumps(draft, ensure_ascii=False, indent=2) + "\n" + fence
    run = await demo.run_scenario(
        model(lambda body: seen.append(body) or response(content)), demo.QUERY,
    )
    demo.verify_roundtrip(run)
    assert run.preview.answer == demo.default_query_answer(2)
    assert run.preview.outfits[0].reasoning == demo.DEFAULT_REASONING
    assert run.preview.outfits[0].style_notes == demo.DEFAULT_STYLE_NOTES
    assert set(run.trace.text_fallbacks) == {"answer", "reasoning", "style_notes"}
    assert len(seen) == len(run.trace.inputs) == run.trace.toolnode_calls == 1
    assert run.response.outfits[0].items[0].id == UUID(ITEM_ID)
    assert run.response.outfits[0].items[0].name == "测试白色衬衫"
    assert run.db.mock_calls == []


@pytest.mark.parametrize("label,newline", [("json", "\n"), ("", "\n"), ("json", "\r\n")])
async def test_single_json_fence_preserves_grounded_text(label, newline):
    fence = chr(96) * 3
    content = " \n" + fence + label + newline + text() + newline + fence + "\n "
    run = await demo.run_scenario(model(lambda _: response(content)), demo.QUERY)
    demo.verify_roundtrip(run)
    original = json.loads(text())
    assert run.preview.answer == original["answer"]
    assert run.preview.outfits[0].reasoning == original["reasoning"]
    assert run.preview.outfits[0].style_notes == original["style_notes"]
    assert run.trace.text_fallbacks == []


@pytest.mark.parametrize("wrapper", [
    "说明：\n~~~json\n{body}\n~~~",
    "~~~json\n{body}\n~~~\n说明结束",
    "~~~json\n{body}\n~~~\n~~~json\n{body}\n~~~",
    "~~~json\n{body}",
    "~~~python\n{body}\n~~~",
    "~~~json {body}\n~~~",
    "~~~json\n{body}\n~~~~",
    "~~~json\n{body}\n{body}\n~~~",
])
async def test_ambiguous_or_malformed_fences_remain_invalid(wrapper):
    # 只接受完整的单一包装，不能搜索或拼接其中的 JSON。
    content = wrapper.replace("~~~", chr(96) * 3).replace("{body}", text())
    run = await demo.run_scenario(model(lambda _: response(content)), demo.QUERY)
    assert run.preview.status == "invalid_result" and run.response.outfits == []
    assert run.trace.errors == ["ValidationError"] and run.trace.text_fallbacks == []
    assert len(run.trace.inputs) == run.trace.toolnode_calls == 1


@pytest.mark.parametrize("payload", [
    text(answer=None),
    text(reasoning=123),
    text(style_notes="x" * 1001),
    text(item_ids=[OTHER_ID]),
    '{"$defs":{"TextDraft":{"type":"object"}}}',
    '{"answer":',
    '{"reasoning":"缺少核心回答"}',
    "[]",
])
async def test_json_fence_does_not_relax_inner_contract(payload):
    fence = chr(96) * 3
    run = await demo.run_scenario(
        model(lambda _: response(fence + "json\n" + payload + "\n" + fence)), demo.QUERY,
    )
    assert run.preview.status == "invalid_result" and run.response.outfits == []
    assert run.trace.errors == ["ValidationError"] and run.trace.text_fallbacks == []
    assert len(run.trace.inputs) == run.trace.toolnode_calls == 1


@pytest.mark.parametrize("payload", [text(), '{"answer":'])
async def test_length_finish_rejects_fenced_json_before_unwrapping(payload):
    fence = chr(96) * 3
    run = await demo.run_scenario(
        model(lambda _: response(fence + "json\n" + payload + "\n" + fence, finish="length")),
        demo.QUERY,
    )
    assert run.preview.status == "invalid_result" and run.response.outfits == []
    assert run.trace.errors == ["response_truncated"] and run.trace.text_fallbacks == []
    assert len(run.trace.inputs) == run.trace.toolnode_calls == 1
