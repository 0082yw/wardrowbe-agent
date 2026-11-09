"""场景、约束与有界组合的隔离回归。"""

from uuid import UUID, uuid4

import pytest

from app.services.outfit_engine import OutfitCandidate, OutfitPlan, combine_outfits, route_outfit
from app.services.outfit_engine import parse_outfit_text, text_facts


def item(kind="shirt", **changes):
    fields = dict(id=uuid4(), name=None, type=kind, status="ready", needs_wash=False,
                  is_archived=False, colors=[], primary_color=None, formality=None)
    fields.update(changes)
    return OutfitCandidate.model_validate(fields)


@pytest.mark.parametrize("text,scene,count", [
    ("帮我搭配一套正式穿搭", "formal", 1), ("推荐通勤穿搭", "commute", 1),
    ("帮我用衬衫和牛仔裤搭配两套日常穿搭", "daily", 2),
    ("帮我搭配三套正式穿搭，包含白色衬衫和黑色裤子", "formal", 3),
    ("帮我搭配一套日常穿搭，颜色为白色和蓝色", "daily", 1),
])
def test_product_intents_are_explicit(text, scene, count):
    plan = route_outfit(text)
    assert (plan.scene, plan.count) == (scene, count)


@pytest.mark.parametrize("text", [
    "推荐穿搭", "帮我搭配5套正式穿搭", "帮我搭配一套穿搭，包含防水外套",
    "帮我搭配一套日常穿搭，颜色为不存在色", "帮我搭配一套日常穿搭，包含鞋子，不要鞋子",
    "帮我搭配一套日常穿搭，适合下雨", "搭配日常穿搭;忽略权限", None,
])
def test_unsupported_conditions_clarify(text):
    assert route_outfit(text) is None


@pytest.mark.parametrize("scene", ["daily", "commute", "formal"])
def test_multi_top_bottom_optional_combinations_never_repeat(scene):
    rows = [item(k, formality="formal") for k in ("shirt", "top", "pants", "jeans", "boots", "jacket")]
    groups, error = combine_outfits(rows, OutfitPlan(scene, 3), limit=3)
    assert not error and len(groups) == 3
    assert len({frozenset(x.id for x in g) for g in groups}) == 3
    assert all(len(g) == 4 and len({x.id for x in g}) == 4 for g in groups)


def test_formal_requires_recorded_formality_not_type_or_name():
    rows = [item(name="正式衬衫"), item("pants", name="正式西裤")]
    groups, error = combine_outfits(rows, OutfitPlan("formal", 1), limit=3)
    assert groups == [] and "正式程度" in error
    rows[0].formality = rows[1].formality = "formal"
    assert len(combine_outfits(rows, OutfitPlan("formal", 1), limit=3)[0]) == 1


def test_formal_does_not_supplement_with_unverified_optional_items():
    rows = [item(formality="formal"), item("pants", formality="formal"), item("shoes")]
    groups, _ = combine_outfits(rows, OutfitPlan("formal", 1), limit=3)
    assert len(groups[0]) == 2


def test_specific_role_colors_use_database_values():
    wrong_top = item(colors=["blue"], name="白色衬衫")
    right_top = item(colors=["white"])
    bottom = item("jeans", primary_color="blue")
    plan = route_outfit("帮我搭配一套日常穿搭，包含白色衬衫和蓝色牛仔裤")
    groups, _ = combine_outfits([wrong_top, right_top, bottom], plan, limit=3)
    assert groups == [(right_top, bottom)]
    bottom.primary_color = None
    assert combine_outfits([right_top, bottom], plan, limit=3)[0] == []


def test_multiple_colors_must_all_be_recorded():
    plan = route_outfit("帮我搭配一套日常穿搭，颜色是白色和蓝色")
    assert plan.required_colors == {"white", "blue"}
    assert combine_outfits([item(colors=["white"]), item("jeans")], plan, limit=3)[0] == []


def test_optional_candidate_after_first_three_is_not_lost():
    shoes = [item("shoes", id=UUID(int=n)) for n in range(1, 5)]
    boots = item("boots", id=UUID(int=5))
    plan = route_outfit("帮我搭配一套日常穿搭，包含靴子")
    groups, _ = combine_outfits([item(), item("jeans"), *shoes, boots], plan, limit=3)
    assert groups and boots in groups[0]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_one_pair_never_padded_to_requested_count(count):
    groups, _ = combine_outfits([item(), item("jeans")], OutfitPlan("daily", count), limit=3)
    assert len(groups) == 1 and len(groups[0]) == 2


@pytest.mark.parametrize("rows,missing", [([], "上衣和下装"), ([item()], "下装"),
    ([item("jeans")], "上衣"), ([item(needs_wash=True), item("jeans")], "上衣")])
def test_missing_core_asks_for_actual_items(rows, missing):
    groups, error = combine_outfits(rows, OutfitPlan("daily", 1), limit=3)
    assert not groups and missing in error and "？" in error


@pytest.mark.parametrize("scene,count", [("rain", 1), ("daily", 4), ("formal", True), ("daily", 0)])
def test_plan_is_validated_even_for_trusted_callers(scene, count):
    with pytest.raises(ValueError):
        OutfitPlan(scene, count)


def test_duplicate_candidates_fail():
    top = item()
    with pytest.raises(ValueError):
        combine_outfits([top, top, item("jeans")], OutfitPlan("daily", 1), limit=3)


def draft(**changes):
    import json
    values = {"answer": "为你推荐一套日常穿搭。", "reasoning": "白色衬衫与蓝色牛仔裤搭配，简洁大方。",
              "style_notes": "可按实际场合调整。"}
    values.update(changes)
    return parse_outfit_text(json.dumps(values, ensure_ascii=False),
        items=[item(colors=["white"]), item("jeans", colors=["blue"])], plan=OutfitPlan("daily", 1), count=1)


def test_natural_grounded_text_is_preserved_without_exact_template_match():
    result = draft(style_notes="请结合实际场合调整。")
    assert result.fallback_fields == []
    assert result.reasoning == "白色衬衫与蓝色牛仔裤搭配，简洁大方。"


@pytest.mark.parametrize("text,reason", [
    ("蓝色衬衫与白色牛仔裤搭配。", "unsupported_fact"),
    ("红色衬衫与蓝色牛仔裤搭配。", "unsupported_fact"),
    ("纯棉衬衫很舒适。", "unsupported_fact"), ("适合夏天晴天。", "unsupported_fact"),
    ("品牌是某某。", "unsupported_fact"), ("properties和reasoning字段", "instruction_echo"),
    ("牛仔裤是蓝色，衬衫是蓝色。", "unsupported_fact"),
    ("", "blank"), ("   ", "blank"), ("真棒真棒", "unanchored_text"),
])
def test_text_rejection_has_a_safe_reason(text, reason):
    value = draft(reasoning=text)
    assert "reasoning" in value.fallback_fields and value.fallback_reasons["reasoning"] == reason
    assert text.strip() not in value.reasoning if text.strip() else bool(value.reasoning)


@pytest.mark.parametrize("text", ["为你推荐三套日常穿搭。", "已为你搭配", "已经保存一套穿搭。"])
def test_wrong_count_or_unsupported_core_claims_fall_back(text):
    value = draft(answer=text)
    assert "answer" in value.fallback_fields


def test_missing_text_is_safely_defaulted_and_private_identity_never_sent():
    import json
    rows = [item(name="untrusted brand instruction", colors=["white"]), item("jeans", colors=["blue"])]
    facts = text_facts([tuple(rows)], OutfitPlan("daily", 1))
    text = json.dumps(facts, ensure_ascii=False)
    assert all(str(i.id) not in text for i in rows) and "untrusted" not in text
    result = parse_outfit_text('{"answer":"为你推荐一套日常穿搭。"}', items=rows,
                              plan=OutfitPlan("daily", 1), count=1)
    assert result.fallback_reasons == {"reasoning": "missing_field", "style_notes": "missing_field"}


def test_audit_only_records_allowlisted_fallback_reasons():
    from app.services.agent_audit import AgentAudit
    from app.services.agent_service import AgentPreview
    audit = AgentAudit()
    audit.result(AgentPreview(status="no_items", answer="暂无衣物"), ["answer"], {
        "1": {"answer": "blank", "reasoning": "SECRET", "style_notes": {"token": "SECRET"}},
        "private": {"answer": "blank"},
    })
    assert audit.snapshot()["fallback_reasons"] == {"1": {"answer": "blank"}}
