"""请求级安全摘要，复用 app 日志；不保存消息、认证材料或图片地址。"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from app.utils.garment_vocabulary import TYPES

if TYPE_CHECKING:
    from app.schemas.agent import AgentPreviewResponse

UNAVAILABLE = "unavailable"
logger = logging.getLogger(__name__)
# 复用 app 的 handler；审计不随普通业务日志的 WARNING 配置一起被静默丢弃。
logger.setLevel(logging.INFO)
current_audit: ContextVar["AgentAudit | None"] = ContextVar("agent_audit", default=None)


def _uuid(value: object) -> str | None:
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def _type(value: object) -> str:
    return value if isinstance(value, str) and value in TYPES else UNAVAILABLE


def _integer(value: object) -> int | str:
    return value if type(value) is int and 0 <= value <= 10**9 else UNAVAILABLE


@dataclass
class AgentAudit:
    request_id: str = field(default_factory=lambda: uuid4().hex)
    authenticated: bool | str = UNAVAILABLE
    tool_scope_bound: bool | str = UNAVAILABLE
    http_status: int | str = UNAVAILABLE
    final_status: str = UNAVAILABLE
    schema_passed: bool | str = UNAVAILABLE
    error: str = UNAVAILABLE
    fallback_fields: list[str] | str = UNAVAILABLE
    fallback_reasons: dict[str, dict[str, str]] = field(default_factory=dict)
    text_quality: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    displays: list[dict[str, Any]] = field(default_factory=list)
    _user_id: str | None = field(default=None, repr=False)
    _key: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)
    _started: float = field(default_factory=time.monotonic, repr=False)
    _time: float = field(default_factory=time.time, repr=False)
    _model_started: dict[int, float] = field(default_factory=dict, repr=False)
    _cpu_config: bool | str = UNAVAILABLE
    _cpu_runtime: bool | str = UNAVAILABLE
    _candidate_shape_valid: bool = field(default=True, repr=False)
    execution_mode: str = UNAVAILABLE
    tunnel_status: str = UNAVAILABLE
    gpu_verified: bool | str = "unknown"
    gpu_evidence_status: str = "unavailable"
    gpu_evidence: Any = field(default=None, repr=False)
    _configured_model: str = field(default=UNAVAILABLE, repr=False)
    _preview_count: int | str = field(default=UNAVAILABLE, repr=False)

    def bind_execution(self, mode: str, model_name: str) -> None:
        from app.services.ollama_agent_model import validate_model_name

        if mode not in ("local_cpu", "remote_gpu"):
            return
        try:
            validate_model_name(model_name)
        except ValueError:
            return
        self.execution_mode = mode
        self._configured_model = model_name
        self.tunnel_status = "not_applicable" if mode == "local_cpu" else "unknown"
        self.gpu_verified = False if mode == "local_cpu" else "unknown"
        self.gpu_evidence = None
        self.gpu_evidence_status = "unavailable"

    def execution_error(self, code: str) -> None:
        allowed = {"tunnel_unavailable", "remote_service_unavailable", "remote_model_missing",
                   "remote_metadata_invalid", "remote_model_busy", "resources_unavailable",
                   "windows_memory_low", "wsl_memory_low", "wsl_memory_pressure", "swap_activity",
                   "timeout", "connection_failed", "http_error", "invalid_response",
                   "thinking_metadata_unavailable", "thinking_disable_unsupported", "model_metadata_blocked",
                   "response_truncated", "thinking_not_disabled"}
        if code in allowed:
            self.error = code

    def record_gpu_evidence(self, proof) -> None:
        """Only trusted evidence for this successful real model call can attest GPU."""
        from app.services.gpu_evidence import GpuObservation, GpuRequestEvidence, gpu_evidence_digest

        self.gpu_verified = "unknown"
        self.gpu_evidence = None
        try:
            proof = GpuRequestEvidence.model_validate(proof.model_dump())
            observation = GpuObservation.model_validate({
                key: getattr(proof, key) for key in GpuObservation.model_fields
            })
            expected_id = gpu_evidence_digest(observation, proof.memory_samples)
            entry = self.models[-1]
            if (self.execution_mode != "remote_gpu" or proof.request_id != self.request_id
                    or proof.model != self._configured_model or entry["model"] != proof.model
                    or len(self.models) != 1 or entry["transport"] != "local_http"
                    or entry["http_status"] != 200 or entry["finish_reason"] != "stop"
                    or entry["error"] != UNAVAILABLE or proof.evidence_id != expected_id
                    or not self._time <= proof.request_started_at <= proof.request_finished_at <= time.time()
                    or not 0 <= time.time() - proof.observed_at <= 10):
                raise ValueError
        except (ValueError, TypeError, AttributeError, IndexError, KeyError):
            self.gpu_evidence_status = "rejected"
            return
        self.gpu_verified = True
        self.gpu_evidence_status = "verified"
        self.gpu_evidence = proof
        entry.update(gpu_verified=True, gpu_vram_bytes=proof.gpu_size_bytes)

    def reference(self, kind: str, value: object) -> str:
        if not isinstance(value, str) or not value:
            return UNAVAILABLE
        return hmac.new(self._key, (kind + ":" + value).encode(), hashlib.sha256).hexdigest()

    def bind_user(self, user_id: object, *, tool: bool = False) -> None:
        value = _uuid(user_id)
        if tool:
            self.tool_scope_bound = value is not None and value == self._user_id
        else:
            self._user_id = value
            self.authenticated = value is not None

    def tool_started(self, item_type: object, *, wardrobe_scope: bool = False) -> dict[str, Any]:
        entry = {"type": _type(item_type), "returned_count": UNAVAILABLE,
                 "tool_message_linked": UNAVAILABLE, "query_executed": False}
        if wardrobe_scope:
            entry["scope"] = "current_user_unarchived_candidates"
        self.tools.append(entry)
        return entry

    def query_started(self, item_type: object, *, user_id: object) -> None:
        """在真实工具函数内记录；参数被 ToolNode 拒绝时不会到达此处。"""
        self.bind_user(user_id, tool=True)
        if self.tools and self.tools[-1]["type"] == _type(item_type):
            self.tools[-1]["query_executed"] = True

    def tool_result(self, entry: dict, payload: object, *, linked: bool) -> None:
        entry["tool_message_linked"] = linked
        if not linked or not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            self._candidate_shape_valid = False
            return
        entry["returned_count"] = len(payload["items"])
        for item in payload["items"]:
            if not isinstance(item, dict) or (item_id := _uuid(item.get("id"))) is None:
                self._candidate_shape_valid = False
                continue
            status = item.get("status")
            status = status if isinstance(status, str) and status in {"ready", "processing"} else UNAVAILABLE
            wash = item.get("needs_wash")
            if status == UNAVAILABLE or type(wash) is not bool or _type(item.get("type")) == UNAVAILABLE:
                self._candidate_shape_valid = False
            self.candidates.setdefault(item_id, {
                "id": item_id, "type": _type(item.get("type")), "status": status,
                "needs_wash": wash if type(wash) is bool else UNAVAILABLE,
                "available": (status == "ready" and wash is False)
                if status != UNAVAILABLE and type(wash) is bool else UNAVAILABLE,
                "scope_verified": UNAVAILABLE,
                "name_ref": self.reference("name", item.get("name")),
                "image_ref": UNAVAILABLE,
            })

    def verify_candidate(self, item: object, *, valid: bool, image_path: str | None) -> None:
        item_id = _uuid(getattr(item, "id", None))
        candidate = self.candidates.get(item_id or "")
        if candidate is not None:
            candidate["scope_verified"] = valid
            if valid:
                candidate["image_ref"] = self.reference("image", image_path)
        if not valid:
            self.error = "candidate_mismatch"

    def display(self, item: object, projected: object, *, valid: bool,
                image_path: str | None) -> None:
        item_id = _uuid(getattr(projected, "id", None))
        candidate = self.candidates.get(item_id or "")
        owned = _uuid(getattr(item, "user_id", None)) == self._user_id
        if candidate is not None and candidate["scope_verified"] == UNAVAILABLE:
            candidate["scope_verified"] = valid and owned
            if valid and owned:
                candidate["image_ref"] = self.reference("image", image_path)
        # 失败时不将其他用户的 ID、名称摘要或路径写入日志。
        if not valid or not owned or candidate is None:
            self.error = "display_mismatch"
            return
        thumbnail = getattr(projected, "thumbnail_url", None)
        path = urlsplit(thumbnail).path.removeprefix("/api/v1/images/") if thumbnail else None
        image_ref = self.reference("image", path)
        name_ref = self.reference("name", getattr(projected, "name", None))
        self.displays.append({
            "id": item_id, "candidate_id": candidate["id"],
            "type": _type(getattr(projected, "type", None)),
            "type_consistent": (
                _type(getattr(projected, "type", None)) == candidate["type"]
                if candidate["type"] != UNAVAILABLE
                and _type(getattr(projected, "type", None)) != UNAVAILABLE else UNAVAILABLE
            ),
            "available": candidate["available"],
            "name_ref": name_ref, "name_consistent": (
                name_ref == candidate["name_ref"]
                if name_ref != UNAVAILABLE and candidate["name_ref"] != UNAVAILABLE else UNAVAILABLE
            ),
            "image_associated": bool(thumbnail), "image_ref": image_ref,
            "image_consistent": image_ref == candidate["image_ref"] if thumbnail else UNAVAILABLE,
        })

    def model_started(self, model: str, *, transport: str = UNAVAILABLE,
                      execution_mode: str = "local_cpu") -> int:
        index = len(self.models)
        self._model_started[index] = time.monotonic()
        # 只记录可信启动者绑定的标识或原有 Qwen 标识，不回显响应里的 model 字段。
        model_id = model if model == self._configured_model or model in {
            "qwen3:0.6b-cpu", "qwen3:0.6b"
        } else UNAVAILABLE
        self.models.append({
            "model": model_id, "seconds": UNAVAILABLE, "http_status": UNAVAILABLE,
            "transport": transport if transport in ("mock", "local_http") else UNAVAILABLE,
            "input_tokens": UNAVAILABLE, "output_tokens": UNAVAILABLE,
            "total_tokens": UNAVAILABLE, "finish_reason": UNAVAILABLE,
            "cpu_config_verified": self._cpu_config, "cpu_only_verified": UNAVAILABLE,
            "gpu_vram_bytes": UNAVAILABLE, "error": UNAVAILABLE,
            "execution_mode": execution_mode if execution_mode in ("local_cpu", "remote_gpu") else UNAVAILABLE,
            "gpu_verified": self.gpu_verified,
        })
        return index

    def model_response(self, index: int, status: int, payload: object = None) -> None:
        entry = self.models[index]
        entry["http_status"] = status
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage")
        if isinstance(usage, dict):
            for target, source in (("input_tokens", "prompt_tokens"),
                                   ("output_tokens", "completion_tokens"),
                                   ("total_tokens", "total_tokens")):
                entry[target] = _integer(usage.get(source))
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish = choices[0].get("finish_reason")
            if finish in ("stop", "length", "tool_calls", "content_filter"):
                entry["finish_reason"] = finish

    def model_finished(self, index: int, *, error: str = UNAVAILABLE) -> None:
        self.models[index]["seconds"] = round(time.monotonic() - self._model_started[index], 4)
        self.models[index]["error"] = error if isinstance(error, str) and error in {
            "timeout", "connection_failed", "http_error", "invalid_response",
            "response_truncated", "thinking_not_disabled",
        } else UNAVAILABLE

    def cpu_verified(self, *, loaded: bool) -> None:
        if loaded:
            self._cpu_runtime = True
            if self.models:
                self.models[-1].update(cpu_only_verified=True, gpu_vram_bytes=0)

    def cpu_rejected(self, payload: dict) -> None:
        self._cpu_runtime = False
        if self.models:
            self.models[-1]["cpu_only_verified"] = False
            models = payload.get("models")
            if isinstance(models, list) and len(models) == 1 and isinstance(models[0], dict):
                if models[0].get("name") == self.models[-1]["model"]:
                    self.models[-1]["gpu_vram_bytes"] = _integer(models[0].get("size_vram"))

    def controller_errors(self, codes: object) -> None:
        if self.error != UNAVAILABLE or not isinstance(codes, list):
            return
        known = {
            "ValueError": "validation_failed", "ValidationError": "validation_failed",
            "timeout": "timeout", "connection_failed": "connection_failed",
            "invalid_response": "invalid_response", "http_error": "http_error",
            "cpu_runtime_unconfirmed": "cpu_runtime_unconfirmed",
            "cloud_not_disabled": "cloud_not_disabled",
            "local_metadata_unavailable": "local_metadata_unavailable",
            "response_truncated": "response_truncated",
            "thinking_not_disabled": "thinking_not_disabled",
        }
        for code in codes:
            if isinstance(code, str) and code in known:
                self.error = known[code]
                break

    def result(self, response: AgentPreviewResponse, fallback_fields: object = UNAVAILABLE,
               fallback_reasons: object = None) -> None:
        self.schema_passed = True
        self._preview_count = len(response.outfits)
        self.final_status = getattr(response, "status", UNAVAILABLE)
        if isinstance(fallback_fields, list):
            self.fallback_fields = sorted({x for x in fallback_fields
                                           if x in ("answer", "reasoning", "style_notes")})
        if isinstance(fallback_reasons, dict):
            allowed = {"missing_field", "blank", "instruction_echo", "unsupported_fact", "unanchored_text"}
            self.fallback_reasons = {
                index: {key: reason for key, reason in reasons.items()
                        if key in ("answer", "reasoning", "style_notes") and isinstance(reason, str) and reason in allowed}
                for index, reasons in fallback_reasons.items()
                if index in ("1", "2", "3") and isinstance(reasons, dict)
            }
        # 只记录规则和最终长度，不保存原始或最终文案。
        self.text_quality = []
        fields = [("1", "answer", response.answer)]
        for index, outfit in enumerate(response.outfits, 1):
            fields.extend((str(index), key, getattr(outfit, key))
                          for key in ("reasoning", "style_notes"))
        for index, key, value in fields:
            reason = self.fallback_reasons.get(index, {}).get(key)
            used_default = (True if reason else False if isinstance(self.fallback_fields, list)
                            and key not in self.fallback_fields else UNAVAILABLE)
            rule = {
                "missing_field": "draft.optional_field",
                "blank": "copy.nonblank",
                "instruction_echo": "copy.instruction_echo",
                "unsupported_fact": "copy.grounded_facts",
                "unanchored_text": "copy.meaningful_anchoring",
            }.get(reason, "copy.passed" if used_default is False else UNAVAILABLE)
            self.text_quality.append({
                "preview": int(index), "field": key,
                "fallback_reason": reason or ("none" if used_default is False else UNAVAILABLE),
                "validation_rule": rule, "used_default": used_default,
                "output_length": len(value),
            })
        # 只有真正返回的展示才计数；失败前累积的中间投影不得当成最终输出。
        final_ids = {str(i) for o in response.outfits for i in o.item_ids}
        self.displays = [entry for entry in self.displays if entry["id"] in final_ids]

    def snapshot(self) -> dict[str, Any]:
        verified = [c.copy() for c in self.candidates.values() if c["scope_verified"] is True]
        def total(key: str):
            return sum(m[key] for m in self.models) if self.models and all(
                type(m.get(key)) in (int, float) for m in self.models
            ) else UNAVAILABLE
        return {
            "event": "agent_preview_audit", "version": 1, "request_id": self.request_id,
            "time": self._time,
            "http_status": self.http_status, "seconds": round(time.monotonic() - self._started, 4),
            "authenticated": self.authenticated, "tool_scope_bound": self.tool_scope_bound,
            "model_request_count": len(self.models), "models": self.models,
            "tool_count": len(self.tools), "tools": self.tools,
            "successful_tool_count": sum(t["tool_message_linked"] is True for t in self.tools),
            "executed_tool_count": sum(t["query_executed"] for t in self.tools),
            "candidate_count": len(self.candidates) if self.tools and self._candidate_shape_valid and all(
                t["returned_count"] != UNAVAILABLE for t in self.tools
            ) else UNAVAILABLE,
            "candidate_scope_verified": (
                all(c["scope_verified"] is True for c in self.candidates.values())
                if self.candidates and all(c["scope_verified"] != UNAVAILABLE
                                           for c in self.candidates.values()) else UNAVAILABLE
            ),
            "cpu_config_verified": self._cpu_config,
            "cpu_only_verified": self._cpu_runtime,
            "verified_candidate_count": len(verified), "candidates": verified,
            "display_count": len(self.displays) if self.schema_passed is True else UNAVAILABLE,
            "display_items": self.displays if self.schema_passed is True else [],
            "final_status": self.final_status, "schema_passed": self.schema_passed,
            "fallback_fields": self.fallback_fields, "fallback_reasons": self.fallback_reasons,
            "text_quality": self.text_quality if self.schema_passed is True
            and self.final_status == "success" else UNAVAILABLE,
            "error": self.error,
            "execution_mode": self.execution_mode, "model_name": self._configured_model,
            "tunnel_status": self.tunnel_status, "gpu_verified": self.gpu_verified,
            "gpu_evidence_status": self.gpu_evidence_status,
            "gpu_evidence": self.gpu_evidence.model_dump() if self.gpu_evidence is not None else UNAVAILABLE,
            "inference_latency": total("seconds"),
            "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
            "tool_execution_count": sum(t["query_executed"] for t in self.tools),
            "preview_count": self._preview_count, "final_schema_status": self.schema_passed,
        }


class AgentAuditRoute(APIRoute):
    """在依赖/输入校验之前建立上下文，让 401/422/429/503 也有同一摘要。"""

    def get_route_handler(self):
        handler = super().get_route_handler()
        if not self.path.endswith("/agent/preview"):
            return handler

        async def audited(request: Request):
            audit = AgentAudit()
            token = current_audit.set(audit)
            try:
                response = await handler(request)
                audit.http_status = response.status_code
                response.headers["X-Request-ID"] = audit.request_id
                return response
            except HTTPException as exc:
                audit.http_status = exc.status_code
                audit.error = {401: "unauthenticated", 403: "forbidden", 429: "rate_limited",
                               503: "unavailable"}.get(exc.status_code, "http_error")
                if exc.status_code == 401:
                    audit.authenticated = False
                exc.headers = {**(exc.headers or {}), "X-Request-ID": audit.request_id}
                raise
            except RequestValidationError:
                audit.http_status = 422
                audit.error = "invalid_input"
                raise
            except Exception:
                audit.error = "request_failed"
                raise
            finally:
                try:
                    logger.info("%s", json.dumps(audit.snapshot(), ensure_ascii=True))
                except Exception:
                    # 日志后端故障不能改变认证或业务响应，也不回显原始记录。
                    pass
                finally:
                    current_audit.reset(token)

        return audited
