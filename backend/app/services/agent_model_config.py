"""可信进程配置；API 请求、模型输出和 .env 均不能选择推理模式。"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from app.services.ollama_agent_model import validate_model_name

ExecutionMode = Literal["local_cpu", "remote_gpu"]
MODE_VARIABLE = "WARDROBE_AGENT_MODEL_MODE"
REMOTE_MODEL_VARIABLE = "WARDROBE_AGENT_REMOTE_MODEL"
DEFAULT_REMOTE_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


@dataclass(frozen=True)
class AgentModelConfig:
    execution_mode: ExecutionMode = "local_cpu"
    model_name: str = "qwen3:0.6b-cpu"

    def __post_init__(self) -> None:
        if self.execution_mode not in ("local_cpu", "remote_gpu"):
            raise ValueError("Agent 推理模式无效")
        validate_model_name(self.model_name)
        if self.execution_mode == "local_cpu" and self.model_name != "qwen3:0.6b-cpu":
            raise ValueError("CPU 模式必须保留原 CPU 专用模型")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AgentModelConfig":
        env = os.environ if environ is None else environ
        mode = env.get(MODE_VARIABLE, "local_cpu")
        if mode == "local_cpu":
            return cls()
        if mode != "remote_gpu":
            raise ValueError("Agent 推理模式无效")
        name = env.get(REMOTE_MODEL_VARIABLE, DEFAULT_REMOTE_MODEL)
        if not name:
            raise ValueError("远程模式需要可信启动配置中的模型名称")
        return cls(execution_mode="remote_gpu", model_name=name)


def create_agent_text_model():
    config = AgentModelConfig.from_env()
    if config.execution_mode == "local_cpu":
        from app.services.local_agent_model import LocalAgentTextModel

        return LocalAgentTextModel()
    from app.services.remote_agent_model import RemoteAgentTextModel

    return RemoteAgentTextModel(
        model=config.model_name,
        gpu_evidence_file=os.environ.get("WARDROBE_AGENT_GPU_EVIDENCE_FILE"),
    )
