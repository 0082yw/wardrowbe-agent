"""有界、只读的基础穿搭组合：权限复核由调用者完成，模型不决定组合。"""

import heapq
import itertools
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.utils.garment_vocabulary import FORMALITY, ITEM_ROLE, OCCASION_FORMALITY, TYPES, normalize_color

COLOR_LABELS = {
    "black": "黑色", "white": "白色", "gray": "灰色", "navy": "藏蓝色",
    "blue": "蓝色", "light-blue": "浅蓝色", "red": "红色", "beige": "米色",
    "brown": "棕色", "green": "绿色", "pink": "粉色",
    "burgundy": "酒红色", "olive": "橄榄绿", "yellow": "黄色", "orange": "橙色",
    "purple": "紫色", "tan": "黄褐色", "cream": "奶油色", "gold": "金色", "silver": "银色",
}
ROLE_LABELS = {"base_top": "上衣", "bottom": "下装", "footwear": "鞋子", "outer_layer": "外套"}
SCENE_LABELS = {"daily": "日常", "commute": "通勤", "formal": "正式"}
NEUTRALS = {"black", "white", "gray", "navy", "beige", "cream"}


@dataclass(frozen=True)
class OutfitPlan:
    scene: str
    count: int
    required_types: tuple[tuple[str, ...], ...] = ()
    color: str | None = None
    excluded_roles: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    role_colors: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.scene not in SCENE_LABELS or type(self.count) is not int or not 1 <= self.count <= 3:
            raise ValueError("场景或方案数量无效")

    @property
    def required_colors(self):
        return set(self.colors) | ({self.color} if self.color else set())


def route_outfit(message: str) -> OutfitPlan | None:
    """完整匹配已支持的中文需求；不猜测天气、面试着装或未支持的条件。"""
    from app.services.wardrobe_query_agent import QUERY_TYPE_GROUPS

    if not isinstance(message, str):
        return None
    text = message.strip().rstrip("。！？!?")
    # 只转换完整的指定类型句式，名称不参与属性推断。
    named = re.fullmatch(r"(?:请)?(?:帮我|给我)?用([^，,。]+)搭配(.+)", text)
    if named:
        text = "搭配" + named[2] + "，包含" + named[1]
    match = re.fullmatch(
        r"(?:请)?(?:帮我|给我)?(?:搭配|推荐|生成)(?P<count>一|两|二|三|[1-3])?套?"
        r"(?P<scene>日常|通勤|正式)?(?:穿搭|搭配)(?P<clauses>(?:[，,].+)?)", text,
    )
    if not match:
        return None
    if not match["count"] and not match["scene"]:
        return None
    groups = {**QUERY_TYPE_GROUPS, "上衣": tuple(t for t in TYPES if ITEM_ROLE[t] == "base_top"),
              "下装": tuple(t for t in TYPES if ITEM_ROLE[t] == "bottom"), "T恤": ("t-shirt",)}
    required, colors, role_colors, excluded = [], [], [], []
    labels = {v: k for k, v in COLOR_LABELS.items()}
    for clause in re.split("[，,]", match["clauses"].lstrip("，,")):
        if not clause:
            continue
        if clause.startswith("不要"):
            terms = re.split("和|、", clause[2:])
            if any(t not in ("鞋子", "外套") for t in terms):
                return None
            excluded.extend({"鞋子": "footwear", "外套": "outer_layer"}[t] for t in terms)
        elif clause.startswith(("颜色为", "颜色是")):
            terms = re.split("和|、", clause[3:])
            if any(t not in labels for t in terms):
                return None
            colors.extend(labels[t] for t in terms)
        elif clause.startswith("包含"):
            for term in re.split("和|、", clause[2:]):
                if term.endswith("单品") and term[:-2] in labels:
                    colors.append(labels[term[:-2]])
                    continue
                color = next((v for k, v in sorted(labels.items(), key=lambda x: -len(x[0])) if term.startswith(k)), None)
                kind = term[len(COLOR_LABELS[color]):] if color else term
                if kind not in groups:
                    return None
                group = groups[kind]
                required.append(group)
                if color:
                    roles = {ITEM_ROLE[t] for t in group}
                    if len(roles) != 1:
                        return None
                    role_colors.append((next(iter(roles)), color))
        else:
            return None
    if any(ITEM_ROLE[t] in excluded for group in required for t in group):
        return None
    counts = {"一": 1, "两": 2, "二": 2, "三": 3, "1": 1, "2": 2, "3": 3}
    return OutfitPlan(
        {"通勤": "commute", "正式": "formal"}.get(match["scene"], "daily"), counts.get(match["count"], 1),
        tuple(required), None, tuple(dict.fromkeys(excluded)), tuple(dict.fromkeys(colors)), tuple(role_colors),
    )


class OutfitCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: UUID
    name: str | None = Field(max_length=100)
    type: str
    status: str
    needs_wash: bool
    is_archived: bool
    colors: list[str] = Field(max_length=50)
    primary_color: str | None
    formality: str | None

    def known_colors(self) -> set[str]:
        return {c for raw in [*self.colors, self.primary_color] if raw and (c := normalize_color(raw))}


def read_outfit_candidates(result, call):
    """只接受后端调用对应的完整 ToolMessage；拒绝重复/伪造字段和分页。"""
    class Result(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        items: list[OutfitCandidate] = Field(max_length=100)
        total: int = Field(ge=0)
        page: int = Field(ge=1)
        page_size: int = Field(ge=1, le=100)
        has_more: bool

    if result.status != "success" or result.name != call["name"] or result.tool_call_id != call["id"]:
        raise ValueError("穿搭查询结果关联无效")
    payload = Result.model_validate_json(result.content)
    if (
        payload.page != call["args"]["page"] or payload.page_size != call["args"]["page_size"]
        or payload.total < len(payload.items) or len(payload.items) > payload.page_size
        or payload.has_more != (payload.page * payload.page_size < payload.total)
        or len({i.id for i in payload.items}) != len(payload.items)
        or any(i.type not in TYPES for i in payload.items)
    ):
        raise ValueError("穿搭候选或分页信息无效")
    return payload


def combine_outfits(candidates: list[OutfitCandidate], plan: OutfitPlan, *, limit: int):
    """只从可穿候选组成上衣+下装，按真实正式程度/颜色偏好排序，至多三套。"""
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError("穿搭上限必须为 1 至 3")
    if len({i.id for i in candidates}) != len(candidates):
        raise ValueError("候选 ID 重复")
    pools = {role: [] for role in ROLE_LABELS}
    preferred = OCCASION_FORMALITY[{"daily": "casual", "commute": "office", "formal": "formal"}[plan.scene]]
    for item in candidates:
        role = ITEM_ROLE.get(item.type)
        if item.status == "ready" and not item.needs_wash and not item.is_archived and role in pools:
            pools[role].append(item)
    missing = [ROLE_LABELS[r] for r in ("base_top", "bottom") if not pools[r]]
    if missing:
        return [], "当前查询范围内缺少可穿的" + "和".join(missing) + "，不能生成完整穿搭。你是否有其他可用的对应衣物？"
    # 正式场景必须有实际的正式程度记录，不根据 shirt/jeans 或名称替代标签。
    if plan.scene == "formal":
        for role in pools:
            pools[role] = [i for i in pools[role] if i.formality in preferred]
        if not pools["base_top"] or not pools["bottom"]:
            return [], "没有已记录正式程度且符合本次正式场景的上衣和下装。请确认着装要求，或改选日常、通勤场景。"
    for role in pools:
        for group in plan.required_types:
            if {ITEM_ROLE[t] for t in group} == {role}:
                pools[role] = [i for i in pools[role] if i.type in group]
        for requested_role, color in plan.role_colors:
            if requested_role == role:
                pools[role] = [i for i in pools[role] if color in i.known_colors()]
    for pool in pools.values():
        pool.sort(key=lambda i: (i.formality not in preferred, str(i.id)))
    # 候选读取上限 100；所有可选件均参与，使用有界堆保留前三套而非存储整个笛卡尔积。
    choices = [pools["base_top"], pools["bottom"]]
    choices += [([None] if role in plan.excluded_roles else pools[role] or [None])
                for role in ("footwear", "outer_layer")]
    def ranked():
        for parts in itertools.product(*choices):
            items = tuple(i for i in parts if i is not None)
            if len({i.id for i in items}) != len(items):
                continue
            if any(not any(i.type in group for i in items) for group in plan.required_types):
                continue
            colors = set().union(*(i.known_colors() for i in items))
            if not plan.required_colors.issubset(colors):
                continue
            rank = (sum(i.formality not in preferred for i in items), max(0, len(colors) - 3),
                    not bool(colors & NEUTRALS), tuple(str(i.id) for i in items))
            yield rank, items
    selected = [items for _, items in heapq.nsmallest(min(plan.count, limit), ranked(), key=lambda x: x[0])]
    return selected, ("当前没有符合指定类型或已记录颜色的上衣与下装组合。是否愿意更换颜色或衣物类型？" if not selected else "")


def outfit_defaults(items, plan):
    colors = sorted(set().union(*(i.known_colors() for i in items)))
    color_fact = ""
    known_labels = [COLOR_LABELS[c] for c in colors if c in COLOR_LABELS]
    if known_labels:
        color_fact = " 已记录的颜色为" + "、".join(known_labels) + "。"
    reason = "用你的上衣和下装组成这套" + SCENE_LABELS[plan.scene] + "搭配。" + color_fact
    if len(colors) > 3:
        reason += " 颜色较多，可按个人喜好简化配饰。"
    notes = "先试试这套基础搭配，再结合实际场合调整。"
    if plan.scene in ("commute", "formal"):
        notes += " 请结合实际工作场所的着装要求确认。"
    if not any(ITEM_ROLE[i.type] == "footwear" for i in items):
        notes += " 本套未包含鞋子，可自行补充。"
    return reason, notes


TYPE_LABELS = {
    "shirt": "衬衫", "t-shirt": "T恤", "top": "上衣", "sweater": "毛衣", "polo": "Polo衫",
    "blouse": "女式衬衫", "tank-top": "背心", "pants": "长裤", "jeans": "牛仔裤",
    "shorts": "短裤", "skirt": "半身裙", "shoes": "鞋子", "sneakers": "运动鞋",
    "boots": "靴子", "sandals": "凉鞋", "jacket": "夹克", "coat": "大衣",
    "hoodie": "卫衣", "blazer": "西装外套",
}
TEXT_INSTRUCTIONS = (
    "你是穿搭助手，正在解释后端已经确定的穿搭，不重新选衣物。"
    "只返回一个JSON对象，恰好包含answer、reasoning、style_notes三个字符串字段。"
    "answer：用一句中文告诉用户有几套什么场景的搭配，明确写出方案数和场景。"
    "reasoning：给用户说明第一套上衣与下装如何组合，明确提到事实中的衣物类型；"
    "如提颜色，必须对应到那件衣物的记录颜色。这里写搭配说明，不是内部思考过程。"
    "style_notes：只建议先试穿、结合实际场合或个人喜好调整；不另行引入衣物或属性。"
    "只依据随后提供的事实。未记录的正式程度不作判断，未列出的单品不声称已搭配。"
    "不要谈天气、材质、季节、品牌、舒适度，不声称已经保存。"
    "每个字段一句不超过45字，不输出ID、Schema、字段说明或这些指令。"
)


def text_facts(groups, plan):
    """仅向模型提供可信词汇和已确认的第一套事实，不发送名称、UUID 或图片。"""
    first = groups[0]
    return {"场景": SCENE_LABELS[plan.scene], "方案数": len(groups), "第一套衣物": [
        {"角色": ROLE_LABELS[ITEM_ROLE[i.type]], "类型": TYPE_LABELS[i.type],
         "记录颜色": [COLOR_LABELS[c] for c in sorted(i.known_colors())],
         "正式程度": i.formality if i.formality in FORMALITY else "未记录"}
        for i in first
    ]}


def grounded_outfit_text(value, *, field, items, plan, count):
    """受控词汇覆盖 + 每个类型/颜色事实核验；不把任意自然文本当成可信事实。"""
    from app.services.wardrobe_query_agent import QUERY_TYPE_GROUPS

    if not value:
        return "blank"
    if re.search(r"schema|properties|reasoning|style_notes|可省略|必须|指令|字段|JSON|测试", value, re.I):
        return "instruction_echo"
    if re.search(r"天气|材质|季节|品牌|保暖|防水|弹性|纯棉|舒适|温度|夏天|冬天|雨天|晴天|已保存", value):
        return "unsupported_fact"
    compact = re.sub(r"\s+", "", value)
    colors = set().union(*(i.known_colors() for i in items))
    aliases = {**QUERY_TYPE_GROUPS, **{v: (k,) for k, v in TYPE_LABELS.items()},
               "上衣": tuple(t for t in TYPES if ITEM_ROLE[t] == "base_top"),
               "下装": tuple(t for t in TYPES if ITEM_ROLE[t] == "bottom")}
    color_tokens = sorted(COLOR_LABELS.values(), key=len, reverse=True)
    kind_tokens = sorted(aliases, key=len, reverse=True)
    for match in re.finditer("|".join(map(re.escape, color_tokens)), compact):
        color = next(k for k, v in COLOR_LABELS.items() if v == match[0])
        if color not in colors:
            return "unsupported_fact"
    # 拒绝将下装颜色说成上衣颜色，即使两个颜色都存在。
    pairs = re.compile("(?P<color>" + "|".join(map(re.escape, color_tokens)) + ")(?:的)?(?P<kind>" + "|".join(map(re.escape, kind_tokens)) + ")")
    reverse_pairs = re.compile("(?P<kind>" + "|".join(map(re.escape, kind_tokens)) + ")(?:是|为|的颜色是|的颜色为)(?P<color>" + "|".join(map(re.escape, color_tokens)) + ")")
    for match in [*pairs.finditer(compact), *reverse_pairs.finditer(compact)]:
        color = next(k for k, v in COLOR_LABELS.items() if v == match["color"])
        if not any(i.type in aliases[match["kind"]] and color in i.known_colors() for i in items):
            return "unsupported_fact"
    for scene, label in SCENE_LABELS.items():
        if label in compact and scene != plan.scene:
            return "unsupported_fact"
    for amount in re.finditer(r"([0-9一二两三四五六七八九十]+)套", compact):
        n = {"一": 1, "二": 2, "两": 2, "三": 3}.get(amount[1], int(amount[1]) if amount[1].isdigit() else -1)
        if n != (count if field == "answer" else 1):
            return "unsupported_fact"
    # 风格建议可明确建议今后补充鞋/外套；不声称它已在本套中。
    kind_pattern = "|".join(map(re.escape, kind_tokens))
    verified_missing_mentions = []
    for match in re.finditer(kind_pattern, compact):
        if not any(i.type in aliases[match[0]] for i in items):
            # A suggestion in another clause must not legitimize a false claim
            # such as “本套包含鞋子，可自行补充”. Verify this mention's clause.
            clause = re.split(r"[，,。.;；!！?？]+", compact[:match.start()])[-1]
            tail = re.split(r"[，,。.;；!！?？]+", compact[match.end():])[0]
            mention = clause + match[0] + tail
            optional_suggestion = field == "style_notes" and match[0] in ("鞋子", "外套") and bool(
                re.fullmatch(r"(?:可(?:以)?(?:自行)?|建议)补充" + re.escape(match[0]), mention)
                or re.fullmatch(r"(?:本套|这套)(?:未|没有)包含" + re.escape(match[0]), mention)
            )
            if not optional_suggestion:
                return "unsupported_fact"
            verified_missing_mentions.append(mention)
    remaining = compact
    for mention in verified_missing_mentions:
        remaining = remaining.replace(mention, "")
    remaining = re.sub(kind_pattern, "衣物", remaining)
    remaining = re.sub("|".join(map(re.escape, color_tokens)), "颜色", remaining)
    remaining = re.sub(r"[1-3一二两三]套", "套", remaining)
    tokens = (
        "这套", "本套", "本次", "已经", "已", "为你", "为您", "你的", "推荐", "生成", "提供", "展示", "一件", "两件",
        "搭配", "穿搭", "方案", "预览", "日常", "通勤", "正式", "基础", "简洁", "利落", "清爽", "协调", "大方", "适合",
        "组合", "搭配成", "包含", "采用", "配上", "配", "与", "和", "的", "以", "用", "了", "是", "为", "套",
        "衣物", "颜色", "记录", "已记录", "未记录", "可以", "可", "自行", "建议", "补充", "根据", "结合", "实际",
        "场合", "个人喜好", "调整", "选择", "确认", "着装要求", "工作场所", "试穿", "试试", "再", "先", "保持",
        "尚未保存", "未保存", "请", "避免", "过多配饰", "不同", "仅供参考",
        # Non-factual connective words only. Type/color/scene/attribute checks
        # above remain mandatory; unknown nouns and assertions still fail.
        "我", "整理", "整体", "作为", "按",
    )
    remaining = re.sub("|".join(map(re.escape, sorted(tokens, key=len, reverse=True))), "", remaining)
    if re.sub(r"[，,。.;；!！?？、：:]", "", remaining):
        return "unanchored_text"
    if field == "reasoning" and not re.search(r"搭配|组合|包含|配上", compact):
        return "unanchored_text"
    if field == "answer" and not re.search(r"穿搭|搭配|方案", compact):
        return "unanchored_text"
    if field == "answer" and not re.search(r"[1-3一二两三]套", compact) and len(list(re.finditer(kind_pattern, compact))) < 2:
        return "unanchored_text"
    return None


def parse_outfit_text(content, *, items, plan, count):
    """只接受文本草稿；错误结构拒绝，低质量字符串回退到可核实事实。"""
    from app.services.agent_service import ShortText
    from app.services.wardrobe_query_agent import unwrap_json_fence

    class Draft(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        answer: ShortText
        reasoning: ShortText | None = None
        style_notes: ShortText | None = None

    # 空白字符串可以回退；null/错误类型/超长/额外业务字段不能靠回退绕过校验。
    raw = json.loads(unwrap_json_fence(content))
    if not isinstance(raw, dict):
        raise ValueError("文案必须为 JSON 对象")
    defaults = dict(zip(("reasoning", "style_notes"), outfit_defaults(items, plan), strict=True))
    defaults["answer"] = f"为你整理了 {count} 套{SCENE_LABELS[plan.scene]}搭配，可以先看看效果，尚未保存。"
    fallbacks = set()
    fallback_reasons = {}
    for key in ("answer", "reasoning", "style_notes"):
        if key not in raw:
            if key == "answer":
                raise ValueError("缺少模型回答")
            fallbacks.add(key)
            fallback_reasons[key] = "missing_field"
            continue
        value = raw[key]
        if not isinstance(value, str) or len(value) > 1000:
            raise ValueError("模型文本字段无效")
        cleaned = value.strip()
        reason = None if cleaned == defaults[key] else grounded_outfit_text(
            cleaned, field=key, items=items, plan=plan, count=count,
        )
        if reason:
            raw[key] = defaults[key]
            fallbacks.add(key)
            fallback_reasons[key] = reason
        else:
            raw[key] = cleaned
    draft = Draft.model_validate(raw)
    return SimpleNamespace(
        answer=draft.answer, reasoning=draft.reasoning or defaults["reasoning"],
        style_notes=draft.style_notes or defaults["style_notes"], fallback_fields=sorted(fallbacks),
        fallback_reasons=fallback_reasons,
    )
