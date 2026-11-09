"""Agent API 契约；预览不等于已保存的穿搭，不定义路由或读取运行配置。"""

from collections.abc import Sequence
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from app.schemas.text import reject_blank
from app.services.agent_service import AgentPreview, OutfitPreview
from app.services.gpu_evidence import GpuRequestEvidence
from app.utils.garment_vocabulary import ITEM_ROLE

MAX_AGENT_MESSAGE_LENGTH = 10000


class AgentRuntimeStatus(BaseModel):
    """配置与管理接口状态，不表示本次推理已经成功。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    mode: Literal["local_ollama", "remote_ollama", "unknown"]
    model: str | None
    available: bool
    cpu_only: bool
    execution_mode: Literal["local_cpu", "remote_gpu", "unknown"] = "unknown"
    tunnel_status: Literal["connected", "disconnected", "not_applicable", "unknown"] = "unknown"
    # 全局就绪检查不代表请求级 GPU 执行；真实证明只随对应预览响应返回。
    gpu_verified: Literal[False, "unknown"] = "unknown"

    @model_validator(mode="before")
    @classmethod
    def legacy_local_metadata(cls, value):
        if isinstance(value, dict) and value.get("mode") == "local_ollama":
            value = {**value}
            value.setdefault("execution_mode", "local_cpu")
            value.setdefault("tunnel_status", "not_applicable")
            value.setdefault("gpu_verified", False)
        return value

    @model_validator(mode="after")
    def consistent_runtime(self) -> Self:
        if self.mode == "unknown":
            if (self.model is not None or self.available or self.cpu_only
                    or self.execution_mode != "unknown" or self.tunnel_status != "unknown"
                    or self.gpu_verified != "unknown"):
                raise ValueError("未知运行模式不能声称模型已确认")
        elif self.mode == "local_ollama":
            if (self.model != "qwen3:0.6b-cpu" or self.execution_mode != "local_cpu"
                    or self.tunnel_status != "not_applicable" or self.gpu_verified is not False
                    or self.available and not self.cpu_only):
                raise ValueError("本机模式必须保留 CPU 配置校验")
        else:
            from app.services.ollama_agent_model import validate_model_name

            validate_model_name(self.model)
            if (self.execution_mode != "remote_gpu" or self.cpu_only
                    or self.tunnel_status not in ("connected", "disconnected", "unknown")
                    or self.available and self.tunnel_status != "connected"):
                raise ValueError("远程模式与隧道状态不一致")
        return self


class AgentRequest(BaseModel):
    """单轮自由文本需求；用户身份、Session、模型端点均由可信后端提供。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    message: str = Field(
        min_length=1,
        max_length=MAX_AGENT_MESSAGE_LENGTH,
        description="自然语言穿搭或衣橱需求，不限制场景；本字段不表示模型已具备所有能力",
    )

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        # Field 先校验原始长度，与当前 Service 的上限一致；允许有意义的多行输入。
        return reject_blank(value.strip())


class AgentRequestExecution(BaseModel):
    """Trusted response metadata for one request, separate from global readiness."""

    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    execution_mode: Literal["local_cpu", "remote_gpu"]
    model: str = Field(min_length=1, max_length=128)
    gpu_verified: Literal[True, False, "unknown"]
    gpu_evidence_status: Literal["unavailable", "rejected", "verified"] = "unavailable"
    gpu_evidence: GpuRequestEvidence | None = None
    fallback_fields: list[Literal["answer", "reasoning", "style_notes"]] = Field(default_factory=list, max_length=3)

    @field_validator("gpu_verified", mode="before")
    @classmethod
    def strict_gpu_flag(cls, value):
        if type(value) is not bool and not (type(value) is str and value == "unknown"):
            raise ValueError("GPU 验证标记类型无效")
        return value

    @model_validator(mode="after")
    def evidence_matches_request(self) -> Self:
        from app.services.ollama_agent_model import validate_model_name

        validate_model_name(self.model)
        if len(self.fallback_fields) != len(set(self.fallback_fields)):
            raise ValueError("重复的文案回退字段")
        if self.gpu_verified is True:
            if (self.execution_mode != "remote_gpu" or self.gpu_evidence_status != "verified"
                    or self.gpu_evidence is None or self.gpu_evidence.request_id != self.request_id
                    or self.gpu_evidence.model != self.model):
                raise ValueError("GPU 证明与本次请求不一致")
        elif self.gpu_evidence is not None or self.gpu_evidence_status == "verified":
            raise ValueError("不能展示未经核验的 GPU 证明")
        if self.execution_mode == "local_cpu" and self.gpu_verified is not False:
            raise ValueError("本机 CPU 请求不能声称 GPU 已验证")
        return self


class AgentItemDisplay(BaseModel):
    """只投影工具已确认的展示字段；不接受用户 ID、内部路径或模型编造的元数据。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: UUID
    name: str | None = Field(default=None, max_length=100)
    thumbnail_url: str | None = Field(
        default=None,
        max_length=2048,
        pattern=(
            r"^/api/v1/images/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}/[a-zA-Z0-9_-]+\.(?:jpg|jpeg|png|webp)"
            r"\?expires=[0-9]+&sig=[0-9a-f]{32}$"
        ),
        description="可信后端生成的当前用户图片签名地址；缺图或处理中为 null",
    )
    type: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)
    ]


class OutfitPreviewResponse(OutfitPreview):
    """继承已有 ID、理由、提示及其约束，仅增加按穿搭顺序排列的展示信息。"""

    items: list[AgentItemDisplay] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def display_matches_ids(self) -> Self:
        if [item.id for item in self.items] != self.item_ids:
            raise ValueError("展示衣物必须与预览 item_ids 完全一致，顺序也必须一致")
        return self


class AgentPreviewResponse(AgentPreview):
    """继承原 Service 的状态与回答规则，补齐 API 展示内容。

    原 AgentPreview 仍是模型输出契约，禁止模型填写 items；本类型只用于后端响应。
    Schema 只能校验结构，不能证明归属。调用者必须先经过 AgentService 的请求内
    候选校验，并从同一用户获准访问的数据中补展示信息，不能信任客户端或模型提供的名称。
    不包含已保存的 outfit ID，也不会触发保存。
    """

    outfits: list[OutfitPreviewResponse] = Field(default_factory=list, max_length=3)
    execution: AgentRequestExecution | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def complete_outfit_roles(self) -> Self:
        if self.kind == "outfit":
            for outfit in self.outfits:
                roles = [ITEM_ROLE.get(item.type) for item in outfit.items]
                if (
                    roles.count("base_top") != 1 or roles.count("bottom") != 1
                    or any(r not in {"base_top", "bottom", "footwear", "outer_layer"} for r in roles)
                    or roles.count("footwear") > 1 or roles.count("outer_layer") > 1
                ):
                    raise ValueError("完整穿搭必须包含已验证的上衣与下装，可选一件鞋子和外套")
        return self

    @classmethod
    def from_preview(
        cls,
        preview: AgentPreview,
        *,
        items: Sequence[AgentItemDisplay] = (),
    ) -> Self:
        """用可信后端的最小字段投影补齐预览；缺失或重复元数据直接报错。"""
        by_id: dict[UUID, AgentItemDisplay] = {}
        for item in items:
            if item.id in by_id:
                raise ValueError("展示数据包含重复衣物 ID")
            by_id[item.id] = item

        outfits = []
        for outfit in preview.outfits:
            if any(item_id not in by_id for item_id in outfit.item_ids):
                raise ValueError("缺少已验证预览衣物的展示信息")
            outfits.append(
                {
                    **outfit.model_dump(),
                    "items": [by_id[item_id].model_dump() for item_id in outfit.item_ids],
                }
            )
        # 重新验证继承的状态规则和全部嵌套字段；仅输出被预览引用的展示信息。
        return cls.model_validate({**preview.model_dump(), "outfits": outfits})
