"""Bounded worker diagnostics and protected exit; no DB, Redis or subprocess."""
import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rotation_worker as entry
from app.services.ollama_agent_model import OllamaModelError


@pytest.mark.parametrize("code", [
    "windows_memory_low", "wsl_memory_low", "wsl_memory_pressure",
    "resources_unavailable", "swap_activity",
])
def test_resource_code_is_logged_without_exception_contents(code):
    result = entry.failure_diagnostic(OllamaModelError("password=secret /private/path", code=code))
    assert result["error_code"] == code
    assert result["resource_diagnostics"] == "unavailable"
    assert "secret" not in json.dumps(result) and "/private" not in json.dumps(result)


def test_snapshot_diagnostic_preserves_only_whitelisted_fields():
    error = OllamaModelError("hidden", code="resources_unavailable")
    error.resource_diagnostics = {
        "source": "windows_snapshot", "reason": "future_timestamp",
        "exception_type": "_InvalidSample", "snapshot_age_seconds": -0.0145678,
        "snapshot_open_attempts": 3, "errno": 2,
        "path": "/private/credential", "password": "secret", "snapshot": {"raw": True},
    }
    result = entry.failure_diagnostic(error)
    assert result["resource_diagnostics"] == {
        "source": "windows_snapshot", "reason": "future_timestamp",
        "exception_type": "_InvalidSample", "snapshot_age_seconds": -0.014568,
        "snapshot_open_attempts": 3, "errno": 2,
    }
    assert "secret" not in json.dumps(result) and "credential" not in json.dumps(result)


@pytest.mark.parametrize("age", [float("nan"), float("inf"), True, "private", 10**12])
def test_untrusted_diagnostic_values_are_omitted(age):
    error = OllamaModelError("hidden", code="password=secret")
    error.resource_diagnostics = {
        "source": "/private/path", "reason": "secret", "exception_type": "secret",
        "snapshot_age_seconds": age, "errno": True, "snapshot_open_attempts": 4,
    }
    result = entry.failure_diagnostic(error)
    assert result["error_code"] == "unavailable"
    assert result["resource_diagnostics"] == "unavailable"
    assert "secret" not in json.dumps(result)


def test_unknown_exception_message_is_never_echoed():
    result = entry.failure_diagnostic(RuntimeError("redis://secret@private/path"))
    assert result["error_type"] == "Exception"
    assert "redis" not in json.dumps(result)


def test_main_logs_safe_diagnostic_and_never_retries(monkeypatch, capsys):
    error = OllamaModelError("SECRET /private", code="resources_unavailable")
    error.resource_diagnostics = {"source": "windows_snapshot", "reason": "stale_snapshot",
                                  "snapshot_age_seconds": 12.25}
    run = AsyncMock(side_effect=error)
    monkeypatch.setattr(entry, "run", run)
    monkeypatch.setattr(logging, "disable", Mock())
    monkeypatch.setattr(sys, "argv", ["rotation_worker.py", "--check-only"])
    assert entry.main() == 1
    run.assert_awaited_once_with(check_only=True, config_path=None)
    text = capsys.readouterr().err
    result = json.loads(text.split("：", 1)[1])
    assert result["error_code"] == "resources_unavailable"
    assert result["resource_diagnostics"]["reason"] == "stale_snapshot"
    assert "SECRET" not in text and "/private" not in text


def test_resource_failure_stops_consumer_and_removes_owned_ready(tmp_path, monkeypatch):
    async def check():
        from arq import worker as arq_worker
        monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *args: None)
        config = tmp_path / "config.json"
        config.write_text("{}")
        ready = tmp_path / "ready.json"
        monkeypatch.setattr(entry, "READY", ready)
        monkeypatch.setattr(entry, "CONFIG", config)
        monkeypatch.setenv("WARDROWBE_MEMORY_FILE", "/mock-memory")
        instances = []

        class FakeWorker:
            def __init__(self, **kwargs):
                self.kwargs, self.tasks, self.allow_pick_jobs = kwargs, {}, True
                self.close = AsyncMock()
                instances.append(self)

            async def async_run(self):
                await self.kwargs["on_startup"]({})
                await asyncio.Event().wait()

        monkeypatch.setattr(arq_worker, "Worker", FakeWorker)
        settings = SimpleNamespace(functions=[], redis_settings=None, queue_name="arq:images",
                                   job_timeout=30, on_startup=AsyncMock(), on_shutdown=AsyncMock())
        error = OllamaModelError("hidden", code="windows_memory_low")
        resources = Mock(side_effect=[None, None, error])
        with pytest.raises(OllamaModelError) as caught:
            await entry.serve({}, settings, resources, object())
        assert caught.value is error and resources.call_count == 3
        assert len(instances) == 1 and instances[0].allow_pick_jobs is False
        instances[0].close.assert_awaited_once()
        assert not ready.exists()

    asyncio.run(check())
