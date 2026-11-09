"""Synthetic trusted sampler records; no SSH, GPU, database or model requests."""
import json
import os
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.schemas.agent import AgentRequestExecution, AgentRuntimeStatus
from app.services.agent_audit import AgentAudit, current_audit
from app.services.gpu_evidence import GPU1_UUID, read_gpu_evidence
from app.services.remote_agent_model import RemoteAgentTextModel

MODEL = "qwen3:4b-instruct-2507-q4_K_M"
DIGEST = "a" * 64
REQUEST = "b" * 32


def observation(**changes):
    return dict(model=MODEL, model_digest=DIGEST, process_id=123,
        process_started_at=900.0, observed_at=995.0, gpu_uuid=GPU1_UUID,
        layers_loaded=37, layers_total=37, model_size_bytes=3000, gpu_size_bytes=3000, process_gpu_memory_bytes=4096, **changes)


def sample(tmp_path, row=None):
    path = tmp_path / "gpu.json"
    path.write_text(json.dumps({"observations": [row or observation()]}))
    path.chmod(0o600)
    return path


def read(path, **changes):
    args = dict(request_id=REQUEST, model=MODEL, digest=DIGEST,
                started_at=994.0, finished_at=996.0, now=997.0)
    args.update(changes)
    return read_gpu_evidence(str(path) if path is not None else None, **args)


def test_complete_trusted_record_is_bound_to_this_request(tmp_path):
    proof, state = read(sample(tmp_path))
    assert state == "verified" and proof.request_id == REQUEST
    assert proof.process_id == 123 and proof.gpu_uuid == GPU1_UUID
    assert len(proof.evidence_id) == 64
    result = AgentRequestExecution(request_id=REQUEST, model=MODEL, execution_mode="remote_gpu",
        gpu_verified=True, gpu_evidence_status=state, gpu_evidence=proof)
    assert result.gpu_verified is True


@pytest.mark.parametrize("changes", [
    {"model": "qwen3:4b"}, {"model_digest": "c" * 64},
    {"gpu_uuid": "GPU-7925e602-ae47-c052-60bb-9b9158b6132c"},
    {"observed_at": 993.0}, {"observed_at": 998.0},
    {"process_id": 0}, {"process_started_at": 999.0},
    {"layers_loaded": 36}, {"gpu_size_bytes": 0},
])
def test_wrong_identity_device_time_or_partial_loading_is_rejected(tmp_path, changes):
    row = observation(); row.update(changes)
    assert read(sample(tmp_path, row)) == (None, "rejected")


def test_stale_record_is_not_reused_for_later_inference(tmp_path):
    assert read(sample(tmp_path), now=1006.0) == (None, "rejected")


@pytest.mark.parametrize("bad", ["empty", "wrong_mode", "symlink", "malformed", "oversized", "fifo"])
def test_invalid_evidence_source_never_grants_gpu(tmp_path, bad):
    path = sample(tmp_path)
    if bad == "empty": path.write_text('{"observations":[]}')
    if bad == "wrong_mode": path.chmod(0o644)
    if bad == "symlink":
        link = tmp_path / "link"; link.symlink_to(path); path = link
    if bad == "malformed": path.write_text("secret invalid content")
    if bad == "oversized": path.write_text("x" * 65537)
    if bad == "fifo":
        path.unlink(); os.mkfifo(path,0o600)
    assert read(path) == (None, "rejected")


def test_missing_source_remains_unavailable_not_verified(tmp_path):
    assert read(None) == read(tmp_path / "missing") == (None, "unavailable")


@pytest.mark.parametrize("changes", [
    {"gpu_evidence": None}, {"request_id": "d" * 32}, {"model": "qwen3:4b"},
    {"execution_mode": "local_cpu"}, {"gpu_evidence_status": "unavailable"},
    {"gpu_verified": 1}, {"gpu_verified": "true"},
])
def test_response_cannot_forge_verified_flag(tmp_path, changes):
    proof, _ = read(sample(tmp_path))
    value = dict(request_id=REQUEST, model=MODEL, execution_mode="remote_gpu", gpu_verified=True,
                 gpu_evidence_status="verified", gpu_evidence=proof)
    value.update(changes)
    with pytest.raises(ValidationError):
        AgentRequestExecution.model_validate(value)


def test_global_readiness_never_claims_request_gpu_success():
    with pytest.raises(ValidationError):
        AgentRuntimeStatus(mode="remote_ollama", model=MODEL, available=True, cpu_only=False,
            execution_mode="remote_gpu", tunnel_status="connected", gpu_verified=True)


def test_audit_accepts_only_this_successful_real_inference(tmp_path, monkeypatch):
    import app.services.agent_audit as module
    monkeypatch.setattr(module.time, "time", lambda: 997.0)
    proof, _ = read(sample(tmp_path))
    audit = AgentAudit(request_id=REQUEST); audit._time = 993.0
    audit.bind_execution("remote_gpu", MODEL)
    idx = audit.model_started(MODEL, execution_mode="remote_gpu", transport="local_http")
    audit.model_response(idx,200,{"choices":[{"finish_reason":"stop"}]})
    audit.model_finished(idx)
    audit.record_gpu_evidence(proof)
    assert audit.gpu_verified is True
    assert audit.snapshot()["gpu_evidence"]["request_id"] == REQUEST
    audit.request_id = "e" * 32
    audit.record_gpu_evidence(proof)
    assert audit.gpu_verified == "unknown" and audit.gpu_evidence is None


@pytest.mark.parametrize("transport,finish,status", [("mock","stop",200),("local_http","length",200),
    ("local_http","stop",500)])
def test_http_or_mock_success_cannot_substitute_for_gpu_evidence(tmp_path, monkeypatch, transport,finish,status):
    import app.services.agent_audit as module
    monkeypatch.setattr(module.time,"time",lambda:997.0)
    proof,_=read(sample(tmp_path))
    audit=AgentAudit(request_id=REQUEST);audit._time=993.0;audit.bind_execution("remote_gpu",MODEL)
    idx=audit.model_started(MODEL,execution_mode="remote_gpu",transport=transport)
    audit.model_response(idx,status,{"choices":[{"finish_reason":finish}]});audit.model_finished(idx)
    audit.record_gpu_evidence(proof)
    assert audit.gpu_verified == "unknown"


@pytest.mark.asyncio
async def test_remote_mock_transport_never_reads_hardware_proof(monkeypatch):
    import app.services.gpu_evidence as module
    reader=Mock(side_effect=AssertionError("Mock cannot claim hardware"))
    monkeypatch.setattr(module,"read_gpu_evidence",reader)
    model=RemoteAgentTextModel(model=MODEL,transport=httpx.MockTransport(lambda r:httpx.Response(200)),
        gpu_evidence_file="/unread/fixture")
    model._check_service=AsyncMock();model._resources=Mock()
    model._adapter=AsyncMock(return_value=AIMessage(content='{"answer":"为你推荐一套日常搭配。"}'))
    audit=AgentAudit();token=current_audit.set(audit)
    try:await model([HumanMessage(content="fixture")],tools=[])
    finally:current_audit.reset(token)
    assert audit.gpu_verified == "unknown"
    reader.assert_not_called()

def test_users_and_model_outputs_cannot_supply_execution_evidence():
    from app.schemas.agent import AgentRequest
    from app.services.agent_service import AgentPreview
    with pytest.raises(ValidationError):
        AgentRequest.model_validate({"message": "搭配日常穿搭", "execution": {"gpu_verified": True}})
    with pytest.raises(ValidationError):
        AgentPreview.model_validate({"status": "no_items", "answer": "暂无",
                                     "execution": {"gpu_verified": True}})
