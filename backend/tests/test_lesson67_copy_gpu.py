"""New lesson67 copy diagnostics and GPU memory proof tests; all offline."""
import json
import pytest
from app.services.wardrobe_query_agent import parse_model_text, TEXT_MODEL_INSTRUCTIONS
from app.services.gpu_evidence import GPU1_UUID, read_gpu_evidence

def parse(data):
    return parse_model_text(json.dumps(data,ensure_ascii=False),selected_name="白衬衫",
                            matched_count=1,available_count=1)

def test_normal_query_copy_is_retained():
    data={"answer":"本次查询到1件衬衫。","reasoning":"这件衬衫来自本次查询结果。",
          "style_notes":"当前仅完成衬衫查询。"}
    draft=parse(data)
    assert draft.answer==data["answer"] and draft.reasoning==data["reasoning"]
    assert draft.fallback_reasons=={} and draft.fallback_fields==[]

@pytest.mark.parametrize("value,reason",[
    ("","blank"),("输出reasoning字段","instruction_echo"),
    ("这件纯棉衬衫适合晴天。","query_contract_mismatch"),
    ("本次查询到2件衬衫。","query_contract_mismatch"),
])
def test_per_field_reason_without_weakening_facts(value,reason):
    draft=parse({"answer":value})
    assert draft.fallback_reasons=={"answer":reason,"reasoning":"missing_field","style_notes":"missing_field"}
    assert draft.fallback_fields==["answer","reasoning","style_notes"]

@pytest.mark.parametrize("value",[None,12,[],{"status":"success"},"x"*1001])
def test_invalid_types_and_lengths_are_not_repaired(value):
    with pytest.raises(ValueError):parse({"answer":value})

def test_schema_fragment_and_extra_business_fields_are_rejected():
    with pytest.raises(ValueError):parse({"type":"object","properties":{}})
    with pytest.raises(ValueError):parse({"answer":"本次查询到1件衬衫。","item_ids":["forged"]})
    with pytest.raises(ValueError):
        parse_model_text('{"answer":',selected_name=None,matched_count=1,available_count=1)

def test_query_prompt_names_all_three_fields_without_allowing_speculation():
    assert all(field in TEXT_MODEL_INSTRUCTIONS for field in ["answer","reasoning","style_notes"])
    assert "不推测" in TEXT_MODEL_INSTRUCTIONS

MODEL="qwen3:4b-instruct-2507-q4_K_M"
def observe(at=995.0,memory=4096,pid=123):
    return dict(model=MODEL,model_digest="a"*64,process_id=pid,process_started_at=900.0,
                gpu_uuid=GPU1_UUID,observed_at=at,layers_loaded=37,layers_total=37,
                model_size_bytes=3000,gpu_size_bytes=3000,process_gpu_memory_bytes=memory)
def read(tmp_path,rows):
    p=tmp_path/"evidence.json";p.write_text(json.dumps({"observations":rows}));p.chmod(0o600)
    return read_gpu_evidence(str(p),request_id="b"*32,model=MODEL,digest="a"*64,
                             started_at=994.0,finished_at=996.0,now=997.0)

def test_pid_and_request_window_memory_history(tmp_path):
    proof,status=read(tmp_path,[observe(993.0,99999),observe(994.1,4096),observe(994.5,8192),observe(995.0,4096)])
    assert status=="verified"
    assert proof.request_id=="b"*32 and proof.process_id==123
    assert [s.observed_at for s in proof.memory_samples]==[994.1,994.5,995.0]
    assert max(s.process_gpu_memory_bytes for s in proof.memory_samples)==8192

def test_other_pid_sample_cannot_supply_peak(tmp_path):
    proof,status=read(tmp_path,[observe(994.1,99999,pid=456),observe()])
    assert status=="verified" and len(proof.memory_samples)==1

@pytest.mark.parametrize("memory",[None,0,-1])
def test_incomplete_or_invalid_process_memory_never_verifies(tmp_path,memory):
    proof,status=read(tmp_path,[observe(memory=memory)])
    assert proof is None and status=="rejected"

def test_unsorted_history_is_not_accepted(tmp_path):
    assert read(tmp_path,[observe(995.0),observe(994.5)])[1]=="rejected"

def test_evidence_digest_covers_memory_history(tmp_path):
    first,_=read(tmp_path,[observe(994.5,4096),observe()])
    second,_=read(tmp_path,[observe(994.5,8192),observe()])
    assert first.evidence_id!=second.evidence_id


def test_audit_and_reader_share_sample_digest(tmp_path,monkeypatch):
    from app.services.agent_audit import AgentAudit
    import app.services.agent_audit as module
    monkeypatch.setattr(module.time,"time",lambda:997.0)
    proof,_=read(tmp_path,[observe(994.5,8192),observe()])
    audit=AgentAudit(request_id="b"*32);audit._time=993.0
    audit.bind_execution("remote_gpu",MODEL)
    idx=audit.model_started(MODEL,execution_mode="remote_gpu",transport="local_http")
    audit.model_response(idx,200,{"choices":[{"finish_reason":"stop"}]})
    audit.model_finished(idx)
    audit.record_gpu_evidence(proof)
    assert audit.gpu_verified is True

def test_audit_rejects_tampered_history_with_original_digest(tmp_path,monkeypatch):
    from app.services.agent_audit import AgentAudit
    from app.services.gpu_evidence import GpuRequestEvidence
    import app.services.agent_audit as module
    monkeypatch.setattr(module.time,"time",lambda:997.0)
    proof,_=read(tmp_path,[observe(994.5,8192),observe()])
    payload=proof.model_dump()
    payload["memory_samples"][0]["process_gpu_memory_bytes"]+=1
    changed=GpuRequestEvidence.model_validate(payload)
    audit=AgentAudit(request_id="b"*32);audit._time=993.0
    audit.bind_execution("remote_gpu",MODEL)
    idx=audit.model_started(MODEL,execution_mode="remote_gpu",transport="local_http")
    audit.model_response(idx,200,{"choices":[{"finish_reason":"stop"}]})
    audit.model_finished(idx)
    audit.record_gpu_evidence(changed)
    assert audit.gpu_verified=="unknown" and audit.gpu_evidence_status=="rejected"
