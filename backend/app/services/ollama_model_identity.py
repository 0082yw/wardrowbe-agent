"""固定官方模型的内容寻址与模板证据；纯标准库，不联网、不加载模型。"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

INSTRUCT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
OFFICIAL_DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
WEIGHT_DIGEST = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
OFFICIAL_TEMPLATE_DIGEST = "eade0a07cac7712787bbce23d12f9306adb4781d873d1df6e16f7840fa37afec"
# 从已验证权重的 tokenizer.chat_template 读取；Ollama 0.40.13 选择此 Jinja 模板。
GGUF_TEMPLATE_DIGEST = "40c21f34cf67d8c760ef72f8ad3ae5afad514299d4b06e91dd9a8d705af7b541"
MANIFEST_MEDIA = "application/vnd.docker.distribution.manifest.v2+json"
MODEL_MEDIA = "application/vnd.ollama.image.model"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_manifest_key")
        result[key] = value
    return result


def parse_manifest(raw: bytes) -> dict:
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if (not isinstance(value, dict) or type(value.get("schemaVersion")) is not int
            or value["schemaVersion"] != 2 or value.get("mediaType") != MANIFEST_MEDIA
            or not isinstance(value.get("layers"), list)):
        raise ValueError("invalid_manifest_structure")
    return value


def _descriptor(value: dict) -> tuple[str, str, int]:
    if (not isinstance(value, dict) or set(value) != {"mediaType", "digest", "size"}
            or not isinstance(value["mediaType"], str)
            or not isinstance(value["digest"], str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["digest"])
            or type(value["size"]) is not int or value["size"] <= 0):
        raise ValueError("invalid_blob_descriptor")
    return value["mediaType"], value["digest"], value["size"]


def validate_manifest(local: dict, official_raw: bytes) -> list[dict]:
    """比较官方结构与所有内容层；仅允许一层精确的原始 Manifest 来源记录。"""
    if hashlib.sha256(official_raw).hexdigest() != OFFICIAL_DIGEST:
        raise ValueError("official_model_digest_mismatch")
    official = parse_manifest(official_raw)
    # 本地 JSON 可重新序列化或附加来源层，因此不比较其整文件 SHA256。
    local = parse_manifest(json.dumps(local).encode())
    if _descriptor(local.get("config")) != _descriptor(official.get("config")):
        raise ValueError("config_descriptor_mismatch")
    expected = [_descriptor(layer) for layer in official["layers"]]
    actual = [_descriptor(layer) for layer in local["layers"]]
    provenance = (MANIFEST_MEDIA, "sha256:" + OFFICIAL_DIGEST, len(official_raw))
    extras = [entry for entry in actual if entry[0] == MANIFEST_MEDIA]
    if extras not in ([], [provenance]):
        raise ValueError("invalid_provenance_layer")
    if Counter(entry for entry in actual if entry[0] != MANIFEST_MEDIA) != Counter(expected):
        raise ValueError("model_layer_descriptor_mismatch")
    weights = [entry for entry in expected if entry[0] == MODEL_MEDIA]
    if weights != [(MODEL_MEDIA, "sha256:" + WEIGHT_DIGEST, 2497280480)]:
        raise ValueError("official_weight_mismatch")
    return [local["config"], *local["layers"]]


def verify_blob(path: Path, descriptor: dict) -> None:
    _, digest, size = _descriptor(descriptor)
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError("blob_size_mismatch")
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            checksum.update(block)
    if checksum.hexdigest() != digest.removeprefix("sha256:"):
        raise ValueError("blob_digest_mismatch")


def verify_model_store(models: Path) -> dict:
    """只读校验现有权重，不拉取、不注册、不写模型目录。"""
    models = models.resolve(strict=True)
    blobs = (models / "blobs").resolve(strict=True)
    if not blobs.is_relative_to(models):
        raise ValueError("blob_directory_outside_store")

    def blob(digest):
        path = (blobs / digest.replace(":", "-")).resolve(strict=True)
        if not path.is_relative_to(blobs):
            raise ValueError("blob_outside_store")
        return path

    manifest_path = (models / "manifests/registry.ollama.ai/library/qwen3/4b-instruct-2507-q4_K_M").resolve(strict=True)
    if not manifest_path.is_relative_to(models):
        raise ValueError("manifest_outside_store")
    local = parse_manifest(manifest_path.read_bytes())
    original = blob("sha256:" + OFFICIAL_DIGEST).read_bytes()
    descriptors = validate_manifest(local, original)
    for descriptor in descriptors:
        verify_blob(blob(descriptor["digest"]), descriptor)
    return {"model": INSTRUCT_MODEL, "official_digest": OFFICIAL_DIGEST,
            "config_digest": local["config"]["digest"], "layers": local["layers"],
            "weight_sha256": WEIGHT_DIGEST, "all_blobs_verified": True}


def forced_thinking_prefix(template: str) -> bool:
    """识别生成位置的显式思考前缀；历史消息中出现 think 标签不是强制思考证据。"""
    if template.strip() == "<think>":
        return True
    generation = template.rsplit("if add_generation_prompt", 1)[-1]
    if generation != template and re.search(r"<think>", generation, re.IGNORECASE):
        return True
    return bool(re.search(r"<\|im_start\|>assistant(?:\\n|\s)*<think>", template, re.IGNORECASE))


def verify_instruct_identity(data: dict, *, model_name: str | None, model_digest: str | None) -> None:
    """用名称、官方摘要、GGUF 身份及两种官方模板证据共同确认 Instruct。"""
    info = data.get("model_info", {})
    details = data.get("details", {})
    if (model_name != INSTRUCT_MODEL or model_digest != OFFICIAL_DIGEST
            or not isinstance(info, dict) or not isinstance(details, dict)
            or info.get("general.architecture") != "qwen3"
            or info.get("general.basename") != "Qwen3"
            or info.get("general.finetune") != "Instruct"
            or info.get("general.version") != "2507"
            or type(info.get("general.parameter_count")) is not int
            or info["general.parameter_count"] != 4022468096
            or details.get("format") != "gguf" or details.get("family") != "qwen3"
            or details.get("quantization_level") != "Q4_K_M"
            or data.get("system") or data.get("messages") not in (None, [])):
        raise ValueError("instruct_identity_mismatch")
    modelfile = data.get("modelfile")
    template = data.get("template")
    if not isinstance(modelfile, str) or not isinstance(template, str):
        raise ValueError("instruct_template_missing")
    matches = re.findall(r'^TEMPLATE\s+"""(.*?)"""', modelfile, re.MULTILINE | re.DOTALL)
    if len(matches) != 1 or hashlib.sha256(matches[0].encode()).hexdigest() != OFFICIAL_TEMPLATE_DIGEST:
        raise ValueError("official_template_mismatch")
    source = data.get("template_source")
    allowed = {"go_template": OFFICIAL_TEMPLATE_DIGEST, "gguf_chat_template": GGUF_TEMPLATE_DIGEST}
    if (source not in allowed or hashlib.sha256(template.encode()).hexdigest() != allowed[source]
            or forced_thinking_prefix(template)):
        raise ValueError("runtime_template_mismatch")
