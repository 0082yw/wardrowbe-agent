"""私有启动包使用的模式策略；只校验，不连接、启停服务或保存认证材料。"""

import json
from dataclasses import dataclass

from app.services.agent_model_config import AgentModelConfig
from app.services.agent_resources import read_resources, validate_resources


@dataclass(frozen=True)
class ModelRuntimePolicy:
    config: AgentModelConfig

    @property
    def model_port(self) -> int:
        return 11434 if self.config.execution_mode == "local_cpu" else 11435

    def network_guard(self, event: str, args: tuple) -> None:
        permitted = {("127.0.0.1", p) for p in (5432, 6379, self.model_port)}
        if event == "socket.connect" and (
            not isinstance(args[1], tuple) or args[1][:2] not in permitted
        ):
            raise PermissionError("Only fixed local dependencies and the selected model port are permitted")
        if event == "socket.getaddrinfo" and args[0] not in ("127.0.0.1", "localhost", None):
            raise PermissionError("External DNS prohibited")
        if event == "socket.bind" and args[1] != ("127.0.0.1", 8001):
            raise PermissionError("Only 127.0.0.1:8001 binding permitted")

    def check_resources(self, heartbeat: str):
        value = read_resources(heartbeat)
        validate_resources(value, self.config.execution_mode)
        return value

    def attest_model(self, model) -> None:
        from app.services.local_agent_model import LocalAgentTextModel
        from app.services.remote_agent_model import RemoteAgentTextModel

        expected = LocalAgentTextModel if self.config.execution_mode == "local_cpu" else RemoteAgentTextModel
        assert isinstance(model, expected) and model._transport is None
        assert model._adapter._transport is None
        assert model._adapter.execution_mode == self.config.execution_mode
        assert model._adapter.model == self.config.model_name
        assert model._adapter.disable_thinking and model._adapter.max_tokens == 256

    def validate_http(self, request) -> bool:
        """返回是否为生成请求；管理接口限定为现有只读操作。"""
        if (request.url.scheme != "http" or request.url.host != "127.0.0.1"
                or request.url.port != self.model_port or request.url.query or request.url.userinfo):
            raise PermissionError("Model HTTP destination prohibited")
        allowed = (
            {("POST", "/api/show"), ("GET", "/api/ps")}
            if self.config.execution_mode == "local_cpu" else {("GET", "/api/tags"), ("POST", "/api/show")}
        )
        generated = request.method == "POST" and request.url.path == "/v1/chat/completions"
        if not generated and (request.method, request.url.path) not in allowed:
            raise PermissionError("Model operation prohibited")
        if "authorization" in request.headers or "cookie" in request.headers:
            raise PermissionError("Authentication material must not be forwarded to a model")
        if self.config.execution_mode == "remote_gpu" and (request.method, request.url.path) == ("POST", "/api/show"):
            assert json.loads(request.content) == {"model": self.config.model_name}
        if generated:
            body = json.loads(request.content)
            assert body["model"] == self.config.model_name
            assert body["tools"] == []
            if self.config.execution_mode == "local_cpu" or "reasoning_effort" in body:
                assert body["reasoning_effort"] == "none"
            assert "think" not in body and "options" not in body
            assert body["max_tokens"] == 256 and body["stream"] is False
        return generated
