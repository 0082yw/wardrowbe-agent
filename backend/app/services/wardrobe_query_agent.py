"""请求级只读衣物查询控制器：LangGraph 管理工具，模型仅生成受校验的文本。"""

import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.garment_vocabulary import ITEM_ROLE, TYPES


@dataclass(frozen=True)
class QueryPlan:
    label: str
    item_types: tuple[str, ...]


QUERY_TYPE_GROUPS = {
    "衬衫": ("shirt",),
    "裤子": ("pants", "jeans", "shorts"),
    "长裤": ("pants",), "牛仔裤": ("jeans",), "短裤": ("shorts",),
    "鞋子": tuple(t for t in TYPES if ITEM_ROLE[t] == "footwear"),
    "运动鞋": ("sneakers",), "靴子": ("boots",), "凉鞋": ("sandals",),
    "外套": tuple(t for t in TYPES if ITEM_ROLE[t] == "outer_layer"),
    "夹克": ("jacket",), "大衣": ("coat",), "卫衣": ("hoodie",),
    "西装外套": ("blazer",),
}
MAX_QUERY_TYPES = len({t for types in QUERY_TYPE_GROUPS.values() for t in types})
MAX_QUERY_CANDIDATES = MAX_QUERY_TYPES * 20
if any(
    not types or len(types) > MAX_QUERY_TYPES or any(t not in TYPES for t in types)
    for types in QUERY_TYPE_GROUPS.values()
):
    raise ValueError("查询映射与项目词汇表不一致")


DEFAULT_REASONING = "已从本次查询到的衬衫中选出这件作为预览。"
DEFAULT_STYLE_NOTES = "当前仅完成衬衫查询；如果需要完整搭配，请补充使用场景或下装需求。"
TEXT_MODEL_INSTRUCTIONS = (
    "用简短中文回复这次衣物查询，说明各个单品预览，不生成整套穿搭。"
    "只依据随后给出的查询事实，不推测天气、材质、颜色搭配或其他衣物。"
    "只返回含answer、reasoning、style_notes三个非空字符串的JSON对象，每个字段一句简短中文。"
    "answer只说明查到的数量及单品预览；reasoning只说明该件来自本次查询，"
    "style_notes只说明当前完成查询，如需整套搭配需补充场景或下装需求。"
    "类型代号只是查询条件，回复使用查询中的中文类型。不要描述未核实的颜色、衣料、品牌或穿着效果。"
    "不要复述这些指令，不输出衣物 ID、状态或展示字段。"
)


def default_reasoning(query_label: str) -> str:
    return DEFAULT_REASONING.replace("衬衫", query_label)


def default_style_notes(query_label: str) -> str:
    if query_label == "衬衫":
        return DEFAULT_STYLE_NOTES
    return f"当前仅完成{query_label}查询；如果需要完整搭配，请补充使用场景或搭配需求。"


def default_query_answer(
    matched_count: int, available_count: int | None = None, *,
    query_label: str = "衬衫", preview_count: int = 1,
) -> str:
    available_count = matched_count if available_count is None else available_count
    scope = (
        "现展示首件单品预览。" if preview_count == 1
        else f"分别展示前 {preview_count} 件单品预览。"
    )
    return (
        f"本次查询到 {matched_count} 件{query_label}，其中 {available_count} 件可用；"
        + scope
    )


def _number_pattern(count: int) -> str:
    digits = "零一二三四五六七八九"
    if not 0 <= count <= MAX_QUERY_CANDIDATES:
        raise ValueError("数量超出本次查询上限")
    if count >= 100:
        return str(count)
    if count < 10:
        chinese = digits[count]
    else:
        tens, units = divmod(count, 10)
        chinese = ("" if tens == 1 else digits[tens]) + "十" + (digits[units] if units else "")
    return f"(?:{count}|{chinese}" + ("|两)" if count == 2 else ")")


def grounded_query_text(
    value: str, field: str, *, selected_name, matched_count, available_count,
    query_label="衬衫", preview_count=1, preview_index=0,
) -> bool:
    """只保留可识别的查询事实/后续需求表述；不充当通用自然语言事实判定器。"""
    compact = re.sub(r"\s+", "", value)
    if field == "answer" and preview_count > 1 and re.search(r"首件|第一件|一件|这件", compact):
        return False
    if field == "reasoning" and preview_index > 0 and re.search(r"首件|第一件", compact):
        return False
    if selected_name:
        name = re.sub(r"\s+", "", selected_name)
        if name:
            compact = compact.replace(name, "这件衬衫")
    compact = compact.replace(query_label, "衬衫")
    clauses = [part for part in re.split(r"[，,。.;；!！?？]+", compact) if part]
    if not clauses:
        return False
    matched = _number_pattern(matched_count)
    available = _number_pattern(available_count)
    previews = _number_pattern(preview_count)
    facts = [
        rf"(?:分别)?(?:展示|查看)前{previews}件(?:衬衫)?(?:单品)?预览",
        rf"(?:本次)?(?:已)?(?:查询到|查到|找到)(?:了)?{matched}(?:件|款)衬衫",
        rf"(?:本次)?(?:已)?(?:查询到|查到|找到)(?:了)?{available}(?:件|款)(?:可用|可穿)的?衬衫",
        rf"(?:其中|本次查询中)?{available}件(?:可用|可穿)(?:的衬衫)?",
        r"(?:先|现|现在|可以先|可先|已)?(?:展示|查看)(?:其中的?)?"
        r"(?:首件|第一件|一件|这件)(?:衬衫)?的?(?:(?:作为)?(?:单品)?预览)?",
        r"(?:本次)?衬衫查询(?:已)?完成",
        r"(?:这件|该件)(?:衬衫)?(?:来自|选自)本次(?:衬衫)?查询(?:结果)?",
        r"(?:来自|选自)本次(?:衬衫)?查询(?:结果)?",
        r"(?:已)?从本次(?:查询到的衬衫|(?:衬衫)?查询结果)中"
        r"(?:选出|选取|选择)(?:了)?(?:这件|首件|第一件)(?:衬衫)?作为(?:单品)?预览",
        r"(?:已)?(?:根据|基于)(?:本次)?(?:衬衫)?查询结果(?:生成|展示)(?:单品)?预览",
    ]
    scope = [
        r"(?:当前|目前)?(?:仅|只)完成(?:了)?(?:本次)?衬衫查询",
        r"(?:本次)?衬衫查询(?:已)?完成",
        r"请?补充(?:使用场景|场景|下装需求|搭配需求)(?:或(?:使用场景|场景|下装需求|搭配需求))?",
    ]
    conditional = r"(?:如果|如)?(?:需要|需)(?:完整|整套)搭配"
    allowed = [*facts, *scope, conditional]
    if not all(any(re.fullmatch(pattern, clause) for pattern in allowed) for clause in clauses):
        return False
    meaningful = scope if field == "style_notes" else [*facts, scope[0]]
    return any(any(re.fullmatch(pattern, clause) for pattern in meaningful) for clause in clauses)


def unwrap_json_fence(content: str) -> str:
    """仅移除完整、单一的 JSON 展示围栏，不提取片段或修复 JSON。"""
    if not isinstance(content, str):
        return content
    fenced = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(?P<payload>.*?)\r?\n```",
        content.strip(), flags=re.DOTALL,
    )
    if fenced and "```" not in fenced["payload"]:
        return fenced["payload"]
    return content


def parse_model_text(
    content: str, *, selected_name, matched_count: int, available_count: int,
    query_label="衬衫", preview_count=1, preview_index=0,
):
    """结构非法仍拒绝；只有已核实查询后的低质量字符串允许按字段回退。"""
    from app.services.agent_service import ShortText

    if (
        any(label not in QUERY_TYPE_GROUPS for label in query_label.split("、"))
        or not 1 <= available_count <= matched_count <= MAX_QUERY_CANDIDATES
        or not 1 <= preview_count <= min(3, available_count)
    ):
        raise ValueError("缺少可支持默认文案的查询事实")
    defaults = {
        "answer": default_query_answer(
            matched_count, available_count, query_label=query_label, preview_count=preview_count,
        ),
        "reasoning": default_reasoning(query_label),
        "style_notes": default_style_notes(query_label),
    }
    fallback_fields = set()
    fallback_reasons = {}

    class TextDraft(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        answer: ShortText
        reasoning: ShortText = defaults["reasoning"]
        style_notes: ShortText = defaults["style_notes"]

        @field_validator("answer", "reasoning", "style_notes", mode="before")
        @classmethod
        def validate_user_text(cls, value, info):
            # 不把 null、数字等强制转为文本，也不让超长内容靠回退绕过长度限制。
            if not isinstance(value, str):
                return value
            if len(value) > 1000:
                raise ValueError("模型文案超出长度限制")
            cleaned = value.strip()
            if not grounded_query_text(
                cleaned, info.field_name, selected_name=selected_name,
                matched_count=matched_count, available_count=available_count,
                query_label=query_label, preview_count=preview_count, preview_index=preview_index,
            ):
                fallback_fields.add(info.field_name)
                fallback_reasons[info.field_name] = (
                    "blank" if not cleaned else
                    "instruction_echo" if re.search(
                        r"schema|properties|reasoning|style_notes|可省略|字段|指令|JSON", cleaned, re.I,
                    ) else "query_contract_mismatch"
                )
                return defaults[info.field_name]
            return cleaned

    draft = TextDraft.model_validate_json(unwrap_json_fence(content))
    missing = {"reasoning", "style_notes"} - draft.model_fields_set
    fallback_fields.update(missing)
    fallback_reasons.update({field: "missing_field" for field in missing})
    return SimpleNamespace(**draft.model_dump(), fallback_fields=sorted(fallback_fields),
                           fallback_reasons=fallback_reasons)


def route_query(message: str) -> QueryPlan | None:
    """只识别完整的查询句式和显式类型列表；否定、条件及搭配需求均追问。"""
    if not isinstance(message, str):
        return None
    normalized = message.strip().rstrip("。！？!?")
    match = re.fullmatch(
        r"(?:请)?(?:先)?(?:查询|查看|列出|看看)(?:一下)?我的(?P<label>[^。！？!?]+)",
        normalized,
    )
    if match is None:
        match = re.fullmatch(r"我有哪些(?P<label>[^，,。！？!?]+)", normalized)
    if match is None:
        match = re.fullmatch(r"我的(?P<label>[^，,。！？!?]+)有哪些", normalized)
    if match:
        labels = re.split(r"(?:和|与|及|、|，|,)", match["label"])
        if not labels or any(label not in QUERY_TYPE_GROUPS for label in labels):
            return None
        labels = list(dict.fromkeys(labels))
        types = tuple(dict.fromkeys(t for label in labels for t in QUERY_TYPE_GROUPS[label]))
        return QueryPlan("、".join(labels), types)
    return None


def route_item_types(message: str) -> tuple[str, ...] | None:
    plan = route_query(message)
    return plan.item_types if plan is not None else None


def route_item_type(message: str) -> str | None:
    """保留单类型 helper；类型组不伪装成不存在的 SQL type。"""
    types = route_item_types(message)
    return types[0] if types is not None and len(types) == 1 else None


def query_call(item_type: str) -> dict:
    from app.agent_tools import WardrobeQueryInput

    arguments = WardrobeQueryInput(
        item_type=item_type, page=1, page_size=20,
    ).model_dump()
    return {
        "id": "call_backend_wardrobe_47", "name": "query_my_wardrobe_items",
        "args": arguments, "type": "tool_call",
    }


def read_candidates(result: ToolMessage, call: dict, *, include_unavailable=False):
    """只读取本次真实 ToolNode 的结果；拒绝无效、重复或类型不一致的候选。"""
    from app.api.pagination import MAX_PAGE_SIZE
    from app.models.item import ItemStatus
    from app.schemas.agent import AgentItemDisplay

    class Candidate(AgentItemDisplay):
        status: ItemStatus
        needs_wash: bool

    class QueryResult(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        items: list[Candidate] = Field(max_length=20)
        total: int = Field(ge=0)
        page: int = Field(ge=1)
        page_size: int = Field(ge=1, le=MAX_PAGE_SIZE)
        has_more: bool

    if (
        result.status != "success" or result.tool_call_id != call["id"]
        or result.name != call["name"]
    ):
        raise ValueError("工具结果不属于本次后端查询")
    payload = QueryResult.model_validate_json(result.content)
    if (
        payload.page != call["args"]["page"]
        or payload.page_size != call["args"]["page_size"]
        or payload.total < len(payload.items)
        or payload.has_more != (payload.page * payload.page_size < payload.total)
        or any(item.type != call["args"]["item_type"] for item in payload.items)
        or any(item.thumbnail_url is not None for item in payload.items)
        or len({item.id for item in payload.items}) != len(payload.items)
    ):
        raise ValueError("候选衣物或分页信息无效")
    if include_unavailable:
        return payload.items
    return [
        item for item in payload.items
        if item.status == ItemStatus.ready and item.needs_wash is False
    ]



@dataclass
class QueryTrace:
    """请求内观察数据；不跨用户缓存消息或模型输出。"""

    graph_inputs: list = field(default_factory=list)
    graph_outputs: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    text_fallbacks: list = field(default_factory=list)
    text_fallback_reasons: dict = field(default_factory=dict)


class WardrobeQueryAgentModel:
    """作为 AgentService 的模型依赖，顺序查询明确类型，至多一次文本请求。

    只接受后端绑定的工具结果，不接收 Session 或用户身份。调用方必须每请求创建
    新实例，不能全局缓存。实际工具执行、权限与调用上限仍由 AgentService 管理。
    """

    def __init__(self, text_model, *, trace=None, safety_check=None, max_previews=1):
        if type(max_previews) is not int or not 1 <= max_previews <= 3:
            raise ValueError("max_previews 必须为 1 至 3 的整数")
        self.text_model = text_model
        self.trace = trace if trace is not None else QueryTrace()
        self.safety_check = safety_check
        self.call = None
        self.calls = []
        self.plan = None
        self.max_previews = max_previews
        # 正式 API 在请求内绑定复核函数；独立示例的工具继续使用隔离替身。
        self.candidate_validator = None
        self.display_items = []
        self.text_calls = 0
        self._finished = False
        self.authoritative_preview = None
        self.outfit_plan = None
        self.outfit_candidates = {}

    def _query_call(self, item_type):
        """仅供可信后端构造；最终工具输入仍由原工具 Schema 再校验。"""
        return query_call(item_type)

    async def __call__(self, messages, *, tools):
        self.trace.graph_inputs.append([m.model_copy(deep=True) for m in messages])
        try:
            if self._finished:
                raise ValueError("本请求已结束，不能重复执行")
            if self.safety_check is not None:
                self.safety_check()
            response = await self._step(messages)
            if not response.tool_calls:
                self._finished = True
            self.trace.graph_outputs.append(response.model_copy(deep=True))
            return response
        except Exception as exc:
            self._finished = True
            code = getattr(exc, "code", type(exc).__name__)
            if not self.trace.errors or self.trace.errors[-1] != code:
                self.trace.errors.append(code)
            raise

    async def _step(self, messages):
        from app.schemas.agent import AgentItemDisplay
        from app.services.agent_service import AgentPreview

        if self.call is None:
            if not messages or not isinstance(messages[-1], HumanMessage):
                raise ValueError("缺少用户查询")
            self.plan = route_query(messages[-1].content)
            from app.services.outfit_engine import route_outfit
            self.outfit_plan = route_outfit(messages[-1].content)
            if self.outfit_plan is not None:
                from app.agent_tools import OutfitQueryInput
                self.call = {
                    "id": "call_backend_outfit", "name": "query_my_outfit_candidates",
                    "args": OutfitQueryInput(page=1, page_size=100).model_dump(), "type": "tool_call",
                }
                self.calls = [self.call]
                return AIMessage(content="", tool_calls=self.calls)
            if self.plan is None:
                return AIMessage(content=AgentPreview(
                    status="needs_clarification",
                    answer="请明确要查询的衣物类型，或指定一至三套日常、通勤、正式穿搭。也可以补充要包含的衣物类型和记录颜色。",
                ).model_dump_json())
            for index, item_type in enumerate(self.plan.item_types):
                call = self._query_call(item_type)
                if index:
                    call = {**call, "id": f"{call['id']}_{index}"}
                self.calls.append(call)
            self.call = self.calls[0]
            return AIMessage(content="", tool_calls=self.calls)

        results = [m for m in messages if isinstance(m, ToolMessage)]
        if (
            len(results) != len(self.calls)
            or messages[-len(results):] != results
        ):
            raise ValueError("工具结果缺失或重复")
        if self.outfit_plan is not None:
            return await self._outfit_result(results[0])
        merged = {}
        partial_page = False
        for result, call in zip(results, self.calls, strict=True):
            entries = read_candidates(result, call, include_unavailable=True)
            partial_page |= json.loads(result.content)["has_more"]
            for item in entries:
                if item.id in merged and merged[item.id].model_dump() != item.model_dump():
                    raise ValueError("同一衣物的查询结果不一致")
                merged[item.id] = item
        # 包括未入选和不可穿衣物；先验证完整候选集，再把任何事实交给模型。
        if self.candidate_validator is not None:
            for item in merged.values():
                await self.candidate_validator(item)
        from app.models.item import ItemStatus

        candidates = [
            item for item in merged.values()
            if item.status == ItemStatus.ready and item.needs_wash is False
        ]
        if not candidates:
            return AIMessage(content=AgentPreview(
                status="no_items", answer=f"本次{self.plan.label}查询的当前页内暂无可穿衣物。",
            ).model_dump_json())

        selected = candidates[:self.max_previews]
        # UUID、Session、用户身份、业务状态和可执行工具均不发送给文本模型。
        matched_count = len(merged)
        context = {
            "query": f"查询我的{self.plan.label}",
            "item": {"name": selected[0].name, "type": selected[0].type},
            "items": [{"name": item.name, "type": item.type} for item in selected],
            "matched_count": matched_count,
            "available_count": len(candidates),
            "preview_count": len(selected),
        }
        if self.text_calls:
            raise ValueError("本次文本模型调用次数已达上限")
        self.text_calls += 1
        response = await self.text_model(
            [
                SystemMessage(content=TEXT_MODEL_INSTRUCTIONS),
                HumanMessage(content=json.dumps(context, ensure_ascii=False)),
            ],
            tools=[],
        )
        if not isinstance(response, AIMessage) or response.tool_calls or response.invalid_tool_calls:
            raise ValueError("文本模型不得发起工具调用")
        drafts = [
            parse_model_text(
                response.content, selected_name=item.name,
                matched_count=matched_count, available_count=len(candidates),
                query_label=self.plan.label, preview_count=len(selected), preview_index=index,
            )
            for index, item in enumerate(selected)
        ]
        self.trace.text_fallbacks = sorted({field for draft in drafts for field in draft.fallback_fields})
        self.trace.text_fallback_reasons = {
            str(index + 1): draft.fallback_reasons
            for index, draft in enumerate(drafts) if draft.fallback_reasons
        }
        answer = drafts[0].answer
        if partial_page:
            answer += " 本次仅查询各类型的第一页，尚有更多衣物未展示。"
        preview = AgentPreview.model_validate({
            "status": "success", "answer": answer,
            "outfits": [{
                "item_ids": [item.id], "reasoning": draft.reasoning,
                "style_notes": draft.style_notes,
            } for item, draft in zip(selected, drafts, strict=True)],
        })
        self.display_items = [
            AgentItemDisplay(id=item.id, name=item.name, type=item.type) for item in selected
        ]
        return AIMessage(content=preview.model_dump_json())

    async def _outfit_result(self, result):
        from app.schemas.agent import AgentItemDisplay
        from app.services.agent_service import AgentPreview
        from app.services.outfit_engine import (
            SCENE_LABELS, TEXT_INSTRUCTIONS, combine_outfits, parse_outfit_text,
            read_outfit_candidates, text_facts,
        )

        payload = read_outfit_candidates(result, self.call)
        self.outfit_candidates = {item.id: item for item in payload.items}
        if self.candidate_validator is not None:
            for item in payload.items:
                await self.candidate_validator(item)
        groups, missing = combine_outfits(payload.items, self.outfit_plan, limit=self.max_previews)
        if not groups:
            if payload.has_more:
                missing += " 本次仅使用前 100 件候选，其他衣物尚未核验。"
            preview = AgentPreview(
                kind="outfit", scene=self.outfit_plan.scene,
                status="needs_clarification" if self.outfit_plan.scene == "formal" else "no_items", answer=missing,
            )
        else:
            if self.text_calls:
                raise ValueError("本次文本模型调用次数已达上限")
            self.text_calls += 1
            # 仅提供类型、实际颜色、组合数量和场景，不提供 UUID/用户/图片/数据库。
            facts = text_facts(groups, self.outfit_plan)
            response = await self.text_model([
                SystemMessage(content=TEXT_INSTRUCTIONS),
                HumanMessage(content=json.dumps(facts, ensure_ascii=False)),
            ], tools=[])
            if not isinstance(response, AIMessage) or response.tool_calls or response.invalid_tool_calls:
                raise ValueError("穿搭文本模型不得执行工具")
            drafts = [parse_outfit_text(
                response.content, items=g, plan=self.outfit_plan, count=len(groups),
            ) for g in groups]
            self.trace.text_fallbacks = sorted({f for d in drafts for f in d.fallback_fields})
            self.trace.text_fallback_reasons = {
                str(index + 1): d.fallback_reasons for index, d in enumerate(drafts) if d.fallback_reasons
            }
            answer = drafts[0].answer
            if len(groups) < self.outfit_plan.count:
                answer += f" 当前候选仅能组成 {len(groups)} 套不同穿搭，未凑足 {self.outfit_plan.count} 套。"
            if payload.has_more:
                answer += " 本次仅使用前 100 件候选，尚有衣物未参与组合。"
            preview = AgentPreview.model_validate({
                "kind": "outfit", "scene": self.outfit_plan.scene, "status": "success", "answer": answer,
                "outfits": [{
                    "title": f"{SCENE_LABELS[self.outfit_plan.scene]}穿搭方案 {index + 1}",
                    "item_ids": [i.id for i in group], "reasoning": draft.reasoning,
                    "style_notes": draft.style_notes,
                } for index, (group, draft) in enumerate(zip(groups, drafts, strict=True))],
            })
            self.display_items = [AgentItemDisplay(id=i.id, name=i.name, type=i.type)
                                  for i in {i.id: i for group in groups for i in group}.values()]
        self.authoritative_preview = preview.model_copy(deep=True)
        return AIMessage(content=preview.model_dump_json())
