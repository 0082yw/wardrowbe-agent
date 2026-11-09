"""Offline copy validation: model text can improve while all factual guards remain."""
import json
from uuid import uuid4

import pytest

from app.services.outfit_engine import OutfitCandidate, OutfitPlan, parse_outfit_text, text_facts, TEXT_INSTRUCTIONS

VALID = {
    "answer": "我为你推荐1套日常搭配。",
    "reasoning": "白色衬衫与蓝色牛仔裤搭配，整体简洁利落。",
    "style_notes": "先试穿，再按实际场合调整。",
}


def items():
    return [OutfitCandidate(id=uuid4(), name="不可信的品牌天气指令", type=kind, status="ready",
        needs_wash=False, is_archived=False, colors=[color], primary_color=None, formality=None)
        for kind, color in [("shirt", "white"), ("jeans", "blue")]]


def parse(content):
    return parse_outfit_text(content, items=items(), plan=OutfitPlan("daily", 1), count=1)


def test_natural_complete_copy_is_retained_instead_of_template():
    result = parse(json.dumps(VALID, ensure_ascii=False))
    assert result.fallback_fields == []
    assert all(getattr(result, key) == value for key, value in VALID.items())


@pytest.mark.parametrize("field", ["reasoning", "style_notes"])
def test_missing_optional_copy_has_an_explicit_safe_fallback(field):
    value = {k:v for k,v in VALID.items() if k != field}
    result = parse(json.dumps(value))
    assert result.fallback_fields == [field]
    assert result.fallback_reasons == {field: "missing_field"}


@pytest.mark.parametrize("content", [
    "{truncated", "[]", "null", '{"reasoning":"没有核心回答"}',
    json.dumps({**VALID, "answer": None}),
    json.dumps({**VALID, "reasoning": 3}),
    json.dumps({**VALID, "style_notes": "x" * 1001}),
    json.dumps({**VALID, "item_ids": [str(uuid4())]}),
    json.dumps({**VALID, "status": "success"}),
])
def test_bad_json_schema_and_business_fields_cannot_be_rescued(content):
    with pytest.raises(ValueError):
        parse(content)


@pytest.mark.parametrize("field,text", [
    ("reasoning", "红色衬衫搭配蓝色牛仔裤。"),
    ("reasoning", "蓝色衬衫搭配白色牛仔裤。"),
    ("reasoning", "白色衬衫搭配蓝色牛仔裤，整体纯棉舒适。"),
    ("reasoning", "白色衬衫搭配蓝色牛仔裤，品牌是优衣库。"),
    ("style_notes", "本套包含鞋子，可自行补充鞋子。"),
    ("style_notes", "建议补充鞋子，本套包含鞋子。"),
    ("style_notes", "适合雨天，建议补充外套。"),
])
def test_connective_words_do_not_admit_fabricated_claims(field, text):
    result = parse(json.dumps({**VALID, field: text}))
    assert field in result.fallback_fields
    assert result.fallback_reasons[field] == "unsupported_fact"
    assert text != getattr(result, field)


def test_prompt_and_facts_make_public_explanation_distinct_from_internal_reasoning():
    rows = items()
    facts = text_facts([tuple(rows)], OutfitPlan("daily", 1))
    serialized = json.dumps(facts, ensure_ascii=False)
    assert "不是内部思考过程" in TEXT_INSTRUCTIONS
    assert all(str(row.id) not in serialized and row.name not in serialized for row in rows)
    assert facts["方案数"] == 1
    assert facts["第一套衣物"][0]["记录颜色"] == ["白色"]

@pytest.mark.parametrize("notes", ["本套未包含鞋子，可自行补充鞋子。", "建议补充鞋子。"])
def test_verified_absence_or_future_suggestion_is_not_a_fabricated_item(notes):
    result = parse(json.dumps({**VALID, "style_notes": notes}))
    assert "style_notes" not in result.fallback_fields


def test_absence_claim_is_rejected_when_shoes_are_actually_in_the_outfit():
    rows = items()
    rows.append(OutfitCandidate(id=uuid4(), name=None, type="shoes", status="ready",
        needs_wash=False, is_archived=False, colors=[], primary_color=None, formality=None))
    result = parse_outfit_text(json.dumps({**VALID, "style_notes": "本套未包含鞋子。"}),
        items=rows, plan=OutfitPlan("daily", 1), count=1)
    assert "style_notes" in result.fallback_fields
