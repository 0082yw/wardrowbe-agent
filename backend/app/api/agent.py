"""Agent 只读预览；可信后端控制查询，CPU/远程文本模型与测试模型均可注入。"""

import re
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from app.database import DbSession
from app.models.item import ClothingItem, ItemStatus
from app.schemas.agent import (
    AgentItemDisplay,
    AgentPreviewResponse,
    AgentRequest,
    AgentRuntimeStatus,
)
from app.services.agent_audit import AgentAuditRoute, current_audit
from app.services.agent_service import AgentModel, AgentPreview, AgentService
from app.services.item_service import ItemService
from app.services.local_agent_model import LOCAL_MODEL, LocalAgentTextModel
from app.services.agent_model_config import create_agent_text_model
from app.services.remote_agent_model import RemoteAgentTextModel
from app.services.wardrobe_query_agent import MAX_QUERY_TYPES, WardrobeQueryAgentModel
from app.utils.auth import CurrentUser
from app.utils.rate_limit import rate_limit_by_user
from app.utils.signed_urls import sign_optional

router = APIRouter(prefix="/agent", tags=["Agent"], route_class=AgentAuditRoute)


def get_agent_text_model() -> AgentModel:
    """构造时不联网；仅由可信启动配置选择模型，测试可覆盖此依赖。"""
    try:
        return create_agent_text_model()
    except ValueError:
        raise HTTPException(status_code=503, detail={
            "error_code": "AGENT_MODEL_CONFIGURATION_INVALID",
            "message": "Agent 模型启动配置无效，本次不会切换或回退模型。",
        }) from None


def get_agent_model(
    text_model: Annotated[AgentModel, Depends(get_agent_text_model)],
) -> AgentModel:
    """每个请求独立控制工具、候选和状态；请求体不能选择身份或模型端点。"""
    return WardrobeQueryAgentModel(text_model, max_previews=3)


def get_agent_service(
    model: Annotated[AgentModel, Depends(get_agent_model)],
) -> AgentService:
    return AgentService(model=model, max_model_calls=2, max_tool_calls=MAX_QUERY_TYPES)


@router.get("/status", response_model=AgentRuntimeStatus, summary="检查 Agent 模型入口状态")
async def get_agent_status(
    current_user: CurrentUser,
    model: Annotated[AgentModel, Depends(get_agent_model)],
) -> AgentRuntimeStatus:
    if not isinstance(model, WardrobeQueryAgentModel) or not isinstance(
        model.text_model, (LocalAgentTextModel, RemoteAgentTextModel)
    ):
        return AgentRuntimeStatus(mode="unknown", model=None, available=False, cpu_only=False)
    text_model = model.text_model
    if isinstance(text_model, RemoteAgentTextModel):
        available = False
        try:
            await text_model.check_ready()
            available = True
        except Exception:
            pass
        return AgentRuntimeStatus(
            mode="remote_ollama", model=text_model.model_name, available=available, cpu_only=False,
            execution_mode="remote_gpu", tunnel_status=text_model.tunnel_status,
            gpu_verified="unknown",
        )
    try:
        await model.text_model.check_ready()
    except Exception:
        return AgentRuntimeStatus(
            mode="local_ollama", model=LOCAL_MODEL, available=False, cpu_only=False,
        )
    return AgentRuntimeStatus(
        mode="local_ollama", model=LOCAL_MODEL, available=True, cpu_only=True,
    )


def _preview_image_path(item: ClothingItem, user_id: UUID) -> str | None:
    """仅为当前用户的安全相对路径签名；图片不可用时不影响文字预览。"""
    if getattr(item, "processing_kind", None) is not None:
        return None
    for path in (getattr(item, "thumbnail_path", None), getattr(item, "image_path", None)):
        if isinstance(path, str) and re.fullmatch(
            rf"{user_id}/[a-zA-Z0-9_-]+\.(?:jpg|jpeg|png|webp)", path
        ):
            return path
    return None


def _preview_thumbnail_url(item: ClothingItem, user_id: UUID) -> str | None:
    return sign_optional(_preview_image_path(item, user_id))


@router.post(
    "/preview",
    response_model=AgentPreviewResponse,
    summary="生成只读衣物预览",
)
async def preview_agent(
    request: AgentRequest,
    current_user: CurrentUser,
    db: DbSession,
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> AgentPreviewResponse:
    audit = current_audit.get()
    if audit is not None:
        audit.bind_user(current_user.id)
        if isinstance(service.model, WardrobeQueryAgentModel):
            text = service.model.text_model
            if isinstance(text, LocalAgentTextModel):
                audit.bind_execution("local_cpu", LOCAL_MODEL)
            elif isinstance(text, RemoteAgentTextModel):
                audit.bind_execution("remote_gpu", text.model_name)

    def audited_result(preview, *, items=()):
        try:
            result = AgentPreviewResponse.from_preview(preview, items=items)
        except ValidationError:
            if audit is not None:
                audit.schema_passed = False
            raise
        if audit is not None:
            fallbacks = (
                service.model.trace.text_fallbacks
                if isinstance(service.model, WardrobeQueryAgentModel) else "unavailable"
            )
            if isinstance(service.model, WardrobeQueryAgentModel):
                audit.controller_errors(service.model.trace.errors)
            audit.result(result, fallbacks, (
                service.model.trace.text_fallback_reasons
                if isinstance(service.model, WardrobeQueryAgentModel) else None
            ))
            if audit.execution_mode in ("local_cpu", "remote_gpu"):
                try:
                    result = AgentPreviewResponse.model_validate({
                        **result.model_dump(), "execution": {
                            "request_id": audit.request_id, "execution_mode": audit.execution_mode,
                            "model": audit._configured_model, "gpu_verified": audit.gpu_verified,
                            "gpu_evidence_status": audit.gpu_evidence_status,
                            "gpu_evidence": audit.gpu_evidence,
                            "fallback_fields": fallbacks if isinstance(fallbacks, list) else [],
                        },
                    })
                except ValidationError:
                    audit.schema_passed = False
                    raise
        return result

    await rate_limit_by_user(current_user.id, "agent_preview", max_requests=10, window_seconds=60)
    try:
        if isinstance(service.model, WardrobeQueryAgentModel):
            candidate_service = ItemService(db)

            async def validate_candidate(candidate):
                item = await candidate_service.get_by_id(candidate.id, user_id=current_user.id)
                valid = not (
                    item is None or item.id != candidate.id or item.user_id != current_user.id
                    or item.is_archived is not False
                    or item.name != candidate.name or item.type != candidate.type
                    or item.status != candidate.status or item.needs_wash is not candidate.needs_wash
                )
                if valid and hasattr(candidate, "is_archived"):
                    valid = (
                        candidate.is_archived is False
                        and (item.colors or []) == candidate.colors
                        and item.primary_color == candidate.primary_color
                        and item.formality == candidate.formality
                    )
                if audit is not None:
                    audit.verify_candidate(
                        candidate, valid=valid,
                        image_path=_preview_image_path(item, current_user.id) if valid else None,
                    )
                if not valid:
                    raise ValueError("候选衣物已改变或不属于当前用户")

            service.model.candidate_validator = validate_candidate
        preview = await service.generate_preview(
            request.message, current_user=current_user, db=db
        )
        if preview.status != "success":
            if preview.status == "invalid_result" and isinstance(service.model, WardrobeQueryAgentModel):
                codes = set(service.model.trace.errors)
                remote = isinstance(service.model.text_model, RemoteAgentTextModel)
                if "response_truncated" in codes:
                    preview = AgentPreview(
                        status="invalid_result",
                        answer="模型输出达到 token 上限，截断结果已拒绝，本次未生成有效预览。",
                    )
                elif "thinking_not_disabled" in codes:
                    preview = AgentPreview(
                        status="invalid_result", answer="模型仍返回思考内容，本次结果已拒绝，未生成有效预览。",
                    )
                elif remote and "thinking_disable_unsupported" in codes:
                    preview = AgentPreview(
                        status="invalid_result",
                        answer="当前远程模型不支持关闭思考，本次未生成预览，也未回退本机模型。",
                    )
                elif remote and "model_metadata_blocked" in codes:
                    preview = AgentPreview(
                        status="invalid_result", answer="运行保护尚未允许检查远程模型兼容性，本次未生成预览。",
                    )
                elif remote and "thinking_metadata_unavailable" in codes:
                    preview = AgentPreview(
                        status="invalid_result", answer="远程模型缺少可核验的思考控制信息，本次未生成预览。",
                    )
                elif remote and codes.intersection({
                    "tunnel_unavailable", "remote_service_unavailable", "remote_model_missing",
                    "remote_metadata_invalid", "connection_failed", "http_error",
                }):
                    preview = AgentPreview(
                        status="invalid_result",
                        answer="SSH 隧道入口或远程模型服务不可用，本次未生成预览，也未回退本机模型。",
                    )
                elif codes.intersection({"windows_memory_low", "wsl_memory_low", "wsl_memory_pressure",
                                         "resources_unavailable", "swap_activity"}):
                    preview = AgentPreview(status="invalid_result", answer="本机系统资源保护未通过，本次未生成预览。")
                elif codes.intersection({"local_metadata_unavailable", "connection_failed", "cloud_not_disabled"}):
                    preview = AgentPreview(
                        status="invalid_result",
                        answer="本机 Ollama 当前不可用或未通过本地配置检查，本次未生成预览。",
                    )
                elif "timeout" in codes:
                    preview = AgentPreview(
                        status="invalid_result", answer=("远程模型请求超时，本次未生成预览，也未回退本机模型。"
                            if remote else "本机 Ollama 请求超时，本次未生成预览。"),
                    )
                elif codes.intersection({"local_model_busy", "remote_model_busy"}):
                    preview = AgentPreview(
                        status="invalid_result", answer="本机模型正在处理其他请求，请稍后再试。",
                    )
            return audited_result(preview)

        item_service = ItemService(db)
        # 同一请求内去重并顺序读取，继续使用当前用户过滤，不共享并发 Session。
        item_ids = dict.fromkeys(
            item_id for outfit in preview.outfits for item_id in outfit.item_ids
        )
        display = []
        expected_items = (
            {item.id: item for item in service.model.display_items}
            if isinstance(service.model, WardrobeQueryAgentModel) else None
        )
        for item_id in item_ids:
            item = await item_service.get_by_id(item_id, user_id=current_user.id)
            outfit_source = (
                service.model.outfit_candidates.get(item_id)
                if isinstance(service.model, WardrobeQueryAgentModel) and preview.kind == "outfit" else None
            )
            if (
                item is None
                or item.id != item_id
                or item.user_id != current_user.id
                or item.is_archived is not False
                or item.status != ItemStatus.ready
                or item.needs_wash is not False
                or expected_items is not None and (
                    item_id not in expected_items
                    or item.name != expected_items[item_id].name
                    or item.type != expected_items[item_id].type
                )
                or outfit_source is not None and (
                    (item.colors or []) != outfit_source.colors
                    or item.primary_color != outfit_source.primary_color
                    or item.formality != outfit_source.formality
                )
            ):
                # 查询后衣物可能已改变；整份结果作废，不泄露其他用户或缺失对象的信息。
                if audit is not None:
                    audit.error = "display_mismatch"
                return audited_result(
                    AgentPreview(
                        status="invalid_result",
                        answer="部分衣物已不可用，本次预览已失效，请重新生成。",
                    )
                )
            projected = AgentItemDisplay(
                    id=item.id,
                    name=item.name,
                    type=item.type,
                    thumbnail_url=_preview_thumbnail_url(item, current_user.id),
                )
            if audit is not None:
                audit.display(
                    item, projected, valid=True,
                    image_path=_preview_image_path(item, current_user.id),
                )
            display.append(projected)
        return audited_result(preview, items=display)
    except Exception:
        # 内部异常不能被全局 Pydantic handler 误报为用户输入错误，也不回显异常详情。
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "AGENT_PREVIEW_UNAVAILABLE",
                "message": "暂时无法生成穿搭预览，请稍后重试。",
            },
        ) from None
