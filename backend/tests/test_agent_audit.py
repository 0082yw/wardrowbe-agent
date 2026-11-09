"""安全摘要单元测试；不读取凭据、不连接数据库或模型服务。"""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.agent_audit import UNAVAILABLE, AgentAudit


def candidate_context():
    audit = AgentAudit()
    user_id, item_id = uuid4(), uuid4()
    audit.bind_user(user_id)
    audit.bind_user(user_id, tool=True)
    item = SimpleNamespace(id=item_id, user_id=user_id, name="私有名称不可记录", type="shirt")
    path = f"{user_id}/private_thumb.jpg"
    tool = audit.tool_started("shirt")
    audit.query_started("shirt", user_id=user_id)
    audit.tool_result(tool, {"items": [{
        "id": str(item_id), "name": item.name, "type": "shirt",
        "status": "ready", "needs_wash": False,
    }]}, linked=True)
    audit.verify_candidate(item, valid=True, image_path=path)
    return audit, item, path


def test_summaries_link_candidates_names_and_images_without_private_material():
    audit, item, path = candidate_context()
    projected = SimpleNamespace(**vars(item), thumbnail_url=f"/api/v1/images/{path}?expires=1&sig=PRIVATE-SIGNATURE")
    audit.display(item, projected, valid=True, image_path=path)
    audit.schema_passed = True
    record = audit.snapshot()
    assert record["candidate_count"] == record["display_count"] == 1
    assert record["tool_scope_bound"] is True
    assert record["display_items"][0]["candidate_id"] == str(item.id)
    assert record["display_items"][0]["name_consistent"] is True
    assert record["display_items"][0]["image_consistent"] is True
    assert record["candidates"][0]["image_ref"] == record["display_items"][0]["image_ref"]
    text = json.dumps(record, ensure_ascii=False)
    for private in (item.name, str(item.user_id), path, "PRIVATE-SIGNATURE", "expires=", "sig="):
        assert private not in text
    another, _, _ = candidate_context()
    assert another.reference("name", item.name) != audit.reference("name", item.name)


@pytest.mark.parametrize("mismatch", ["name", "image"])
def test_summary_detects_display_inconsistency(mismatch):
    audit, item, path = candidate_context()
    projected = SimpleNamespace(id=item.id, name=item.name, type="shirt",
                                thumbnail_url=f"/api/v1/images/{path}?sig=HIDDEN")
    if mismatch == "name":
        projected.name = "改变的私有名称"
    else:
        projected.thumbnail_url = f"/api/v1/images/{item.user_id}/other.jpg?sig=HIDDEN"
    audit.display(item, projected, valid=True, image_path=path)
    audit.schema_passed = True
    assert audit.snapshot()["display_items"][0][f"{mismatch}_consistent"] is False


def test_unverified_or_foreign_candidates_are_not_published():
    audit, item, path = candidate_context()
    audit.verify_candidate(item, valid=False, image_path=None)
    record = audit.snapshot()
    assert record["candidate_scope_verified"] is False
    assert record["candidates"] == [] and str(item.id) not in json.dumps(record)
    assert record["error"] == "candidate_mismatch"


def test_missing_metadata_is_unavailable_and_usage_is_strictly_allowlisted():
    audit = AgentAudit()
    attempt = audit.model_started("secret-model-name")
    audit.model_response(attempt, 200, {
        "usage": {"prompt_tokens": True, "completion_tokens": "TOKEN-PRIVATE", "total_tokens": -1,
                  "api_key": "KEY-PRIVATE"},
        "choices": [{"finish_reason": "COOKIE-PRIVATE", "message": "RAW-PRIVATE"}],
    })
    audit.model_finished(attempt, error="PASSWORD-PRIVATE")
    record = audit.snapshot()
    entry = record["models"][0]
    assert entry["model"] == entry["finish_reason"] == entry["input_tokens"] == UNAVAILABLE
    assert entry["output_tokens"] == entry["total_tokens"] == entry["cpu_only_verified"] == UNAVAILABLE
    assert record["schema_passed"] == record["authenticated"] == record["candidate_count"] == UNAVAILABLE
    assert "PRIVATE" not in json.dumps(record) and "secret-model-name" not in json.dumps(record)


def test_valid_generation_metrics_do_not_imply_cpu_runtime_verification():
    audit = AgentAudit()
    attempt = audit.model_started("qwen3:0.6b-cpu", transport="mock")
    audit.model_response(attempt, 200, {
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        "choices": [{"finish_reason": "stop"}],
    })
    audit.model_finished(attempt)
    entry = audit.snapshot()["models"][0]
    assert entry["input_tokens"] == 10 and entry["output_tokens"] == 4
    assert entry["total_tokens"] == 14 and entry["finish_reason"] == "stop"
    assert entry["cpu_only_verified"] == UNAVAILABLE
    assert entry["transport"] == "mock" and entry["seconds"] >= 0


def test_unavailable_or_failed_tools_do_not_claim_empty_query():
    audit = AgentAudit()
    tool = audit.tool_started("secret-tool-type")
    audit.tool_result(tool, None, linked=False)
    record = audit.snapshot()
    assert record["successful_tool_count"] == 0
    assert record["candidate_count"] == UNAVAILABLE
    assert record["tools"][0]["type"] == UNAVAILABLE
    assert "secret-tool-type" not in json.dumps(record)


def test_absent_name_does_not_claim_two_unavailable_values_are_verified_equal():
    audit, item, path = candidate_context()
    audit.candidates[str(item.id)]["name_ref"] = UNAVAILABLE
    projected = SimpleNamespace(id=item.id, name=None, type="shirt", thumbnail_url=None)
    audit.display(item, projected, valid=True, image_path=path)
    audit.schema_passed = True
    record = audit.snapshot()
    assert record["display_items"][0]["name_consistent"] == UNAVAILABLE
    assert record["display_items"][0]["name_ref"] == UNAVAILABLE


def test_malformed_candidate_does_not_claim_an_empty_result():
    audit = AgentAudit()
    tool = audit.tool_started("shirt")
    audit.tool_result(tool, {"items": [{"id": "PRIVATE invalid id"}]}, linked=True)
    record = audit.snapshot()
    assert record["candidate_count"] == UNAVAILABLE
    assert record["tools"][0]["returned_count"] == 1
    assert "PRIVATE" not in json.dumps(record)


def copy_response():
    from app.schemas.agent import AgentPreviewResponse
    item_id = uuid4()
    return AgentPreviewResponse.model_validate({
        "status": "success", "answer": "面向用户的私有回答",
        "outfits": [{
            "item_ids": [item_id], "reasoning": "面向用户的私有理由",
            "style_notes": "面向用户的私有提示",
            "items": [{"id": item_id, "name": "私有名称", "type": "shirt"}],
        }],
    })


@pytest.mark.parametrize("reason,rule", [
    ("missing_field", "draft.optional_field"), ("blank", "copy.nonblank"),
    ("instruction_echo", "copy.instruction_echo"),
    ("unsupported_fact", "copy.grounded_facts"),
    ("unanchored_text", "copy.meaningful_anchoring"),
])
def test_copy_diagnostics_report_reason_rule_default_and_length_without_text(reason, rule):
    audit = AgentAudit()
    response = copy_response()
    audit.result(response, ["reasoning"], {"1": {"reasoning": reason}})
    records = audit.snapshot()["text_quality"]
    assert [r["field"] for r in records] == ["answer", "reasoning", "style_notes"]
    entry = records[1]
    assert entry == dict(preview=1, field="reasoning", fallback_reason=reason,
                        validation_rule=rule, used_default=True,
                        output_length=len(response.outfits[0].reasoning))
    assert records[0]["used_default"] is False
    assert records[0]["validation_rule"] == "copy.passed"
    assert "私有" not in json.dumps(records, ensure_ascii=False)


def test_copy_diagnostics_do_not_invent_missing_reasons_or_report_failed_requests_as_passed():
    audit = AgentAudit()
    audit.result(copy_response(), ["reasoning"], {"1": {"reasoning": "SECRET-RAW-TEXT"}})
    assert audit.snapshot()["text_quality"][1]["used_default"] == UNAVAILABLE
    assert "SECRET" not in json.dumps(audit.snapshot())
    audit.schema_passed = False
    assert audit.snapshot()["text_quality"] == UNAVAILABLE
    from app.schemas.agent import AgentPreviewResponse
    audit.result(AgentPreviewResponse(status="invalid_result", answer="请求失败"), [])
    assert audit.snapshot()["text_quality"] == UNAVAILABLE
