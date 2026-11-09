"""Worker launcher security tests: no real process, Redis, DB or model requests."""
import asyncio
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import signal
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import local_agent as manager
import rotation_worker as entry


def options(**changes):
    values = dict(functions=[SimpleNamespace(__name__="authorized_rotate_item_image_job")],
                  queue_name="arq:images", max_jobs=1, max_tries=1, retry_jobs=False, cron_jobs=[],
                  redis_settings=SimpleNamespace(host="127.0.0.1", port=6379, database=15))
    return SimpleNamespace(**{**values, **changes})


@pytest.mark.parametrize("address,allowed", [
    (("127.0.0.1", 5432), True), (("127.0.0.1", 6379), True),
    (("127.0.0.1", 11435), False), (("127.0.0.1", 8001), False),
    (("inference.example.invalid", 11434), False), (("8.8.8.8", 443), False),
    (("localhost", 5432), False), ("/tmp/untrusted.sock", False),
])
def test_worker_network_is_dependency_only(address, allowed):
    if allowed:
        entry.network_guard("socket.connect", (object(), address))
    else:
        with pytest.raises(PermissionError):
            entry.network_guard("socket.connect", (object(), address))


@pytest.mark.parametrize("event,args", [
    ("socket.getaddrinfo", ("example.com", 443)), ("socket.bind", (None, ("127.0.0.1", 9999))),
    ("subprocess.Popen", ("anything",)), ("os.system", ("anything",)),
])
def test_no_dns_listening_or_subprocess_escape(event, args):
    with pytest.raises(PermissionError):
        entry.network_guard(event, args)


@pytest.mark.parametrize("change", [
    {"functions": [SimpleNamespace(__name__="tag_item_image")]},
    {"functions": []}, {"queue_name": "arq:queue"}, {"max_jobs": 2},
    {"max_tries": 2}, {"retry_jobs": True}, {"cron_jobs": ["task"]},
    {"redis_settings": SimpleNamespace(host="127.0.0.1", port=6379, database=0)},
])
def test_reject_broadened_worker_settings(change):
    with pytest.raises(manager.RuntimeErrorSafe):
        entry.validate_worker(options(**change))


def test_only_validated_rotation_registration():
    entry.validate_worker(options())


def test_readonly_profile_refused_before_reading_secrets(monkeypatch):
    monkeypatch.setattr(entry, "private_json", lambda path: {})
    monkeypatch.setattr(entry, "application_policy", lambda config: SimpleNamespace(mode="read_only_demo"))
    with pytest.raises(manager.RuntimeErrorSafe):
        entry.configure()


def test_failed_dependency_check_does_not_consume_or_write():
    async def run():
        class Context:
            async def __aenter__(self):
                return connection
            async def __aexit__(self, *args):
                pass
        connection = SimpleNamespace(
            execute=AsyncMock(side_effect=[None, SimpleNamespace(one=lambda: ("wardrowbe_agent_dev", "wardrowbe_agent_dev"))]),
            scalar=AsyncMock(return_value="a7d3c9e1f5b2"),
        )
        db = SimpleNamespace(connect=lambda: Context())
        redis = SimpleNamespace(ping=AsyncMock(return_value=True),
                                zcard=AsyncMock(return_value=1), exists=AsyncMock(return_value=False))
        with pytest.raises(manager.RuntimeErrorSafe, match="非空"):
            await entry.dependency_check(options(), database=db, redis=redis)
        redis.exists.assert_not_called()
        assert str(connection.execute.await_args_list[0].args[0]) == "SET TRANSACTION READ ONLY"
    asyncio.run(run())


def test_dependency_preflight_is_readonly():
    async def run():
        class Context:
            async def __aenter__(self):
                return connection
            async def __aexit__(self, *args):
                pass
        connection = SimpleNamespace(
            execute=AsyncMock(side_effect=[None, SimpleNamespace(one=lambda: ("wardrowbe_agent_dev", "wardrowbe_agent_dev"))]),
            scalar=AsyncMock(return_value="a7d3c9e1f5b2"),
        )
        redis = SimpleNamespace(ping=AsyncMock(return_value=True),
                                zcard=AsyncMock(return_value=0), exists=AsyncMock(return_value=False))
        await entry.dependency_check(options(), database=SimpleNamespace(connect=lambda: Context()), redis=redis)
        commands = [str(c.args[0]) for c in connection.execute.await_args_list]
        assert commands == ["SET TRANSACTION READ ONLY", "SELECT current_database(), current_user"]
    asyncio.run(run())


def test_manager_refuses_unowned_worker_and_never_signals(monkeypatch):
    obj = manager.Manager.__new__(manager.Manager)
    record = {"pid": 123, "command_sha256": "original"}
    obj.state = {"processes": {"rotation_worker": {"record": record, "owned": True}}}
    monkeypatch.setattr(manager, "same_process", lambda value: False)
    monkeypatch.setattr(manager, "process_record", lambda pid: {"pid": pid})
    kill = Mock()
    monkeypatch.setattr(manager.os, "kill", kill)
    with pytest.raises(manager.RuntimeErrorSafe):
        obj.stop_worker()
    kill.assert_not_called()


def test_worker_stop_and_global_stop_include_worker_first(monkeypatch):
    obj = manager.Manager.__new__(manager.Manager)
    calls = []
    obj.stop_one = calls.append
    obj.stop_worker()
    assert calls == ["rotation_worker"]
    calls.clear()
    obj.stop()
    assert calls == ["rotation_worker", "frontend", "backend", "ollama", "monitor"]


def test_manager_will_not_spawn_worker_in_readonly(monkeypatch):
    obj = manager.Manager.__new__(manager.Manager)
    obj.config = {}
    obj.spawn = Mock()
    monkeypatch.setattr(manager, "application_policy", lambda config: SimpleNamespace(mode="read_only_demo"))
    with pytest.raises(manager.RuntimeErrorSafe):
        obj.start_worker()
    obj.spawn.assert_not_called()


def test_duplicate_worker_never_spawns(tmp_path, monkeypatch):
    import hashlib
    obj = manager.Manager.__new__(manager.Manager)
    source = tmp_path / "backend/scripts/rotation_worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("verified")
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    backend = {"pid": 111}
    obj.config = {}
    obj.state = {"application_mode": "full_app", "processes": {
        "backend": {"record": backend, "owned": True},
        "rotation_worker": {"record": {"pid": 222}, "owned": True,
                            "entry_sha256": manager.digest(source), "config_sha256": manager.digest(cfg)},
    }}
    obj.spawn = Mock()
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "CONFIG", cfg)
    monkeypatch.setattr(manager, "application_policy", lambda config: SimpleNamespace(mode="full_app"))
    monkeypatch.setattr(manager, "same_process", lambda value: True)
    monkeypatch.setenv("WARDROBE_AGENT_MODEL_MODE", "remote_gpu")
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"WARDROBE_APP_MODE=full_app\0WARDROBE_AGENT_MODEL_MODE=remote_gpu\0"
                        if str(path) == "/proc/111/environ" else original(path))
    obj.start_worker()
    obj.spawn.assert_not_called()


def test_stale_readiness_only_removed_for_exited_owned_process(tmp_path, monkeypatch):
    obj = manager.Manager.__new__(manager.Manager)
    record = {"pid": 123, "ticks": "old"}
    monkeypatch.setattr(manager, "PRIVATE", tmp_path)
    monkeypatch.setattr(manager, "process_record", lambda pid: None)
    ready = tmp_path / "rotation-worker-ready.json"
    manager.write_private(ready, {"record": record})
    obj.clear_worker_ready(record)
    assert not ready.exists()


def test_unknown_readiness_never_removed(tmp_path, monkeypatch):
    obj = manager.Manager.__new__(manager.Manager)
    monkeypatch.setattr(manager, "PRIVATE", tmp_path)
    monkeypatch.setattr(manager, "process_record", lambda pid: None)
    ready = tmp_path / "rotation-worker-ready.json"
    manager.write_private(ready, {"record": {"pid": 999}})
    with pytest.raises(manager.RuntimeErrorSafe):
        obj.clear_worker_ready({"pid": 123})
    assert ready.exists()


def test_live_worker_readiness_never_removed(tmp_path, monkeypatch):
    obj = manager.Manager.__new__(manager.Manager)
    monkeypatch.setattr(manager, "PRIVATE", tmp_path)
    monkeypatch.setattr(manager, "process_record", lambda pid: {"pid": pid})
    ready = tmp_path / "rotation-worker-ready.json"
    manager.write_private(ready, {"record": {"pid": 123}})
    with pytest.raises(manager.RuntimeErrorSafe):
        obj.clear_worker_ready({"pid": 123})
    assert ready.exists()


def test_worker_serve_uses_signal_drain_and_cleans_owned_ready(tmp_path, monkeypatch):
    async def check():
        from arq import worker as arq_worker
        callbacks = {}
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: callbacks.update({sig: callback}))
        ready = tmp_path / "rotation-worker-ready.json"
        config = tmp_path / "config.json"
        config.write_text("{}")
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
                callbacks[signal.SIGTERM]()
        monkeypatch.setattr(arq_worker, "Worker", FakeWorker)
        settings = options()
        settings.on_startup = AsyncMock()
        settings.on_shutdown = AsyncMock()
        settings.job_timeout = 30
        resources = Mock()
        await entry.serve({}, settings, resources, object())
        assert instances[0].allow_pick_jobs is False
        instances[0].close.assert_awaited_once()
        settings.on_startup.assert_awaited_once()
        assert instances[0].kwargs["handle_signals"] is False
        assert instances[0].kwargs["max_jobs"] == 1
        assert not ready.exists()
    asyncio.run(check())


def test_alternate_config_cannot_start_consumer(monkeypatch):
    run = AsyncMock()
    monkeypatch.setattr(entry, "run", run)
    monkeypatch.setattr(sys, "argv", ["rotation_worker.py", "--config", "/tmp/candidate.json"])
    assert entry.main() == 1
    run.assert_not_called()


def test_candidate_config_only_runs_readonly_preflight(monkeypatch):
    run = AsyncMock()
    monkeypatch.setattr(entry, "run", run)
    monkeypatch.setattr(sys, "argv", ["rotation_worker.py", "--check-only", "--config", "/tmp/candidate.json"])
    assert entry.main() == 0
    run.assert_awaited_once_with(check_only=True, config_path=Path("/tmp/candidate.json"))
