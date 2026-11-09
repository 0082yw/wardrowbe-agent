"""实际公开元数据夹具 + 内存中的 Manifest 变体；无模型、网络及业务数据库。"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.services.ollama_agent_model import OllamaModelError, validate_non_thinking_metadata
from app.services.ollama_model_identity import (
    GGUF_TEMPLATE_DIGEST, INSTRUCT_MODEL, MANIFEST_MEDIA, OFFICIAL_DIGEST,
    forced_thinking_prefix, parse_manifest, validate_manifest, verify_blob,
)

FIXTURES = Path(__file__).parent / "fixtures"


def metadata():
    return json.loads((FIXTURES / "qwen3_instruct_2507_metadata.json").read_text())


def official():
    return (FIXTURES / "qwen3_instruct_2507_manifest.json").read_bytes()


def classify(data, *, name=INSTRUCT_MODEL, digest=OFFICIAL_DIGEST):
    return validate_non_thinking_metadata(data, model_name=name, model_digest=digest)


@pytest.mark.parametrize("thinking", [None, {"values": [True], "default": True},
    {"values": [False], "default": False}, {"values": [True, False], "default": True}])
def test_instruct_identity_outweighs_auxiliary_thinking_values(thinking):
    data = metadata()
    data["thinking"] = thinking
    assert "thinking" in data["capabilities"]
    assert "<think>" in data["template"]  # 历史 assistant 的处理，不是生成前缀。
    assert hashlib.sha256(data["template"].encode()).hexdigest() == GGUF_TEMPLATE_DIGEST
    assert not forced_thinking_prefix(data["template"])
    assert classify(data) == "instruct_only"


@pytest.mark.parametrize("change", ["name", "digest", "missing_digest", "architecture", "basename",
    "finetune", "version", "parameters", "quantization", "official_template", "runtime_template",
    "source", "forced_prefix", "system", "messages", "missing_modelfile"])
def test_inconsistent_instruct_proof_is_rejected(change):
    data = metadata(); name = INSTRUCT_MODEL; digest = OFFICIAL_DIGEST
    if change == "name": name = "qwen3:4b"
    elif change == "digest": digest = "0" * 64
    elif change == "missing_digest": digest = None
    elif change in {"architecture", "basename", "finetune", "version"}:
        data["model_info"]["general." + change] = "unknown"
    elif change == "parameters": data["model_info"]["general.parameter_count"] = 1
    elif change == "quantization": data["details"]["quantization_level"] = "Q8_0"
    elif change == "official_template": data["modelfile"] += '\nTEMPLATE """injected"""'
    elif change == "runtime_template": data["template"] += "modified"
    elif change == "source": data["template_source"] = "unknown"
    elif change == "forced_prefix": data["template"] += "<|im_start|>assistant\n<think>"
    elif change == "system": data["system"] = "injected"
    elif change == "messages": data["messages"] = [{"role": "user", "content": "injected"}]
    else: data.pop("modelfile")
    with pytest.raises(OllamaModelError): classify(data, name=name, digest=digest)


def test_official_go_template_is_supported_when_it_is_actually_selected():
    data = metadata()
    data["template"] = data["modelfile"].split('TEMPLATE """', 1)[1].split('"""', 1)[0]
    data["template_source"] = "go_template"
    assert classify(data) == "instruct_only"


@pytest.mark.parametrize("identity,prefix,values,code", [
    ("Thinking", "<think>", [True], "thinking_disable_unsupported"),
    ("Unknown", "<think>", [True], "remote_metadata_invalid"),
    ("Thinking", "assistant", [True], "remote_metadata_invalid"),
    ("Thinking", "<think>", [True, False], "remote_metadata_invalid"),
])
def test_only_agreeing_identity_and_template_are_thinking_only(identity, prefix, values, code):
    data = metadata()
    data["model_info"]["general.finetune"] = identity
    data["template"] = prefix
    data["thinking"] = {"values": values, "default": True}
    with pytest.raises(OllamaModelError) as error:
        classify(data, name="qwen3:4b", digest=None)
    assert error.value.code == code


@pytest.mark.parametrize("prefix", ["<think>", "<|im_start|>assistant\n<think>",
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>' }}{% endif %}"])
def test_forced_generation_prefix_is_detected(prefix):
    assert forced_thinking_prefix(prefix)


@pytest.mark.parametrize("provenance", [False, True])
def test_valid_manifest_accepts_only_the_original_provenance_record(provenance):
    raw = official(); local = parse_manifest(raw)
    if provenance:
        local["layers"].append({"mediaType": MANIFEST_MEDIA, "digest": "sha256:" + OFFICIAL_DIGEST,
                                "size": len(raw)})
    # JSON 格式变化不会改变模型身份。
    copied = json.loads(json.dumps(local, indent=4))
    assert validate_manifest(copied, raw) == [copied["config"], *copied["layers"]]


@pytest.mark.parametrize("change", ["config_digest", "config_size", "config_media", "weight_digest",
    "weight_size", "weight_media", "missing_weight", "duplicate_weight", "extra_weight",
    "provenance_digest", "provenance_size", "duplicate_provenance", "unknown_layer", "schema"])
def test_actual_manifest_descriptor_mismatch_never_becomes_a_provenance_exception(change):
    raw = official(); local = parse_manifest(raw)
    provenance = {"mediaType": MANIFEST_MEDIA, "digest": "sha256:" + OFFICIAL_DIGEST, "size": len(raw)}
    local["layers"].append(provenance)
    if change.startswith("config_"):
        key = {"config_digest": "digest", "config_size": "size", "config_media": "mediaType"}[change]
        local["config"][key] = 1 if key == "size" else "sha256:" + "0" * 64 if key == "digest" else "unknown"
    elif change.startswith("weight_"):
        key = {"weight_digest": "digest", "weight_size": "size", "weight_media": "mediaType"}[change]
        local["layers"][0][key] = 1 if key == "size" else "sha256:" + "0" * 64 if key == "digest" else MANIFEST_MEDIA
    elif change == "missing_weight": local["layers"].pop(0)
    elif change in {"duplicate_weight", "extra_weight"}:
        extra = deepcopy(local["layers"][0])
        if change == "extra_weight": extra["digest"] = "sha256:" + "0" * 64
        local["layers"].append(extra)
    elif change == "provenance_digest": provenance["digest"] = "sha256:" + "0" * 64
    elif change == "provenance_size": provenance["size"] += 1
    elif change == "duplicate_provenance": local["layers"].append(deepcopy(provenance))
    elif change == "unknown_layer": local["layers"].append({**provenance, "mediaType": "unknown"})
    else: local["schemaVersion"] = True
    with pytest.raises(ValueError): validate_manifest(local, raw)


def test_different_official_summary_is_rejected():
    raw = official()
    with pytest.raises(ValueError, match="official_model_digest_mismatch"):
        validate_manifest(parse_manifest(raw), raw + b" ")


@pytest.mark.parametrize("change", ["same_size_corruption", "truncated", "missing"])
def test_real_blob_content_mismatch_is_rejected(tmp_path, change):
    contents = b"a simulated model blob"
    desc = {"mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:" + hashlib.sha256(contents).hexdigest(), "size": len(contents)}
    p = tmp_path / "blob"
    p.write_bytes(contents)
    verify_blob(p, desc)
    if change == "same_size_corruption": p.write_bytes(b"b" + contents[1:])
    elif change == "truncated": p.write_bytes(contents[:-1])
    else: p = tmp_path / "missing"
    with pytest.raises(ValueError): verify_blob(p, desc)


def test_duplicate_manifest_keys_fail_closed():
    with pytest.raises(ValueError):
        parse_manifest(b'{"schemaVersion":2,"schemaVersion":2}')
