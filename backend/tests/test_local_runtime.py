"""运行管理离线测试：替换进程、Docker、信号和网络，绝不操作实际服务。"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location("local_agent_runtime", Path(__file__).parents[1] / "scripts/local_agent.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "PRIVATE", tmp_path / "private")
    runtime.PRIVATE.mkdir(mode=0o700)
    build = tmp_path / "frontend/build"
    build.mkdir(parents=True)
    (build / "BUILD_ID").write_text("build-one")
    m = runtime.Manager({"build": str(build), "guard_hashes": {}, "legacy_heartbeat": str(tmp_path / "memory.json"),
                         "heartbeat": str(tmp_path / "memory-owned.json")})
    monkeypatch.setattr(m, "save", Mock())
    return m


def record(pid=123):
    return {"pid": pid, "ticks": "100", "exe": "/project/python", "cwd": "/project", "uid": os.getuid(), "command_sha256": "x"}


def test_identity_uses_start_ticks_executable_and_cwd_not_only_pid(monkeypatch):
    expected = record()
    for key, changed in (("pid", 99), ("ticks", "200"), ("exe", "/other/python"), ("cwd", "/other"), ("uid", os.getuid() + 1)):
        monkeypatch.setattr(runtime, "process_record", lambda pid, key=key, changed=changed: {**expected, key: changed})
        assert not runtime.same_process(expected)
    monkeypatch.setattr(runtime, "process_record", lambda pid: {**expected, "command_sha256": "next-title"})
    assert runtime.same_process(expected)


def test_pid_reuse_never_sends_signal(manager, monkeypatch):
    manager.state["processes"]["backend"] = {"owned": True, "record": record()}
    kill = Mock()
    monkeypatch.setattr(runtime.os, "kill", kill)
    monkeypatch.setattr(runtime, "same_process", lambda value: False)
    monkeypatch.setattr(runtime, "process_record", lambda pid: {**record(), "ticks": "200"})
    with pytest.raises(runtime.RuntimeErrorSafe):
        manager.stop_one("backend")
    kill.assert_not_called()


def test_borrowed_ollama_and_missing_processes_are_not_stopped(manager, monkeypatch):
    manager.state["processes"]["ollama"] = {"owned": False, "record": record()}
    kill = Mock()
    monkeypatch.setattr(runtime.os, "kill", kill)
    manager.stop()
    kill.assert_not_called()


def test_stop_only_sends_graceful_term_to_owned_identity(manager, monkeypatch):
    manager.state["processes"]["backend"] = {"owned": True, "record": record()}
    checks = iter([True, False])
    monkeypatch.setattr(runtime, "same_process", lambda value: next(checks))
    kill = Mock()
    monkeypatch.setattr(runtime.os, "kill", kill)
    manager.stop_one("backend")
    kill.assert_called_once_with(123, runtime.signal.SIGTERM)
    assert "backend" not in manager.state["processes"]


def test_unknown_process_on_port_blocks_start_without_kill_or_spawn(manager, monkeypatch):
    monkeypatch.setattr(runtime, "docker_dependencies", Mock())
    monkeypatch.setattr(runtime, "port_open", lambda p: True)
    spawn = Mock()
    monkeypatch.setattr(manager, "spawn", spawn)
    with pytest.raises(runtime.RuntimeErrorSafe, match="未登记"):
        manager.start()
    spawn.assert_not_called()


def test_repeated_start_has_no_duplicate_process_or_model_request(manager, monkeypatch):
    manager.state["processes"] = {n: {"owned": True, "record": record(i)} for i, n in enumerate(("backend", "frontend"))}
    manager.state["build_id"] = "build-one"
    monkeypatch.setattr(runtime, "docker_dependencies", Mock())
    monkeypatch.setattr(runtime, "port_open", lambda p: True)
    monkeypatch.setattr(runtime, "same_process", lambda p: True)
    spawn, ollama = Mock(), Mock()
    monkeypatch.setattr(manager, "spawn", spawn)
    monkeypatch.setattr(manager, "ensure_ollama", ollama)
    monkeypatch.setattr(manager, "ensure_monitor", Mock())
    manager.start()
    manager.start()
    spawn.assert_not_called()
    assert ollama.call_count == 2  # 只读检查或恢复缺失的本机服务，不生成模型文本。


def test_existing_build_is_not_silently_replaced(manager, monkeypatch):
    manager.state["processes"] = {n: {"owned": True, "record": record(i)} for i, n in enumerate(("backend", "frontend"))}
    manager.state["build_id"] = "old-build"
    monkeypatch.setattr(runtime, "docker_dependencies", Mock())
    monkeypatch.setattr(runtime, "port_open", lambda p: True)
    monkeypatch.setattr(runtime, "same_process", lambda p: True)
    with pytest.raises(runtime.RuntimeErrorSafe, match="reload"):
        manager.start()


@pytest.mark.parametrize("running", [True, False])
def test_docker_only_starts_existing_database_and_redis(monkeypatch, running):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "info":
            return SimpleNamespace(returncode=0, stdout="1")
        if args[1] == "inspect" and args[3].startswith("{{json"):
            port = 5432 if args[-1] == "wardrobe-db" else 6379
            return SimpleNamespace(returncode=0, stdout=json.dumps({"Running": running}) + "|" + json.dumps({f"{port}/tcp": [{"HostPort": str(port)}]}))
        return SimpleNamespace(returncode=0, stdout="healthy\n")
    monkeypatch.setattr(runtime.subprocess, "run", run)
    import redis
    client = Mock()
    client.ping.return_value = True
    monkeypatch.setattr(redis, "Redis", Mock(return_value=client))
    runtime.docker_dependencies()
    assert [c for c in calls if c[1] == "start"] == ([] if running else [["docker", "start", "wardrobe-db"], ["docker", "start", "wardrobe-redis"]])
    assert not any(any(x in c for x in ("compose", "rm", "prune", "down", "pull")) for c in calls)
    assert client.mock_calls == [__import__("unittest.mock").mock.call.ping(), __import__("unittest.mock").mock.call.close()]


@pytest.mark.parametrize("problem", ["daemon", "missing", "ports"])
def test_docker_failures_do_not_create_replacements(monkeypatch, problem):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "info":
            return SimpleNamespace(returncode=1 if problem == "daemon" else 0)
        return SimpleNamespace(returncode=1 if problem == "missing" else 0,
            stdout=json.dumps({"Running": True}) + "|" + json.dumps({"5432/tcp": [{"HostPort": "9999"}]}))
    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(runtime.RuntimeErrorSafe):
        runtime.docker_dependencies()
    assert not any(c[1] == "start" for c in calls)


def test_baseline_preserves_thresholds_and_stops_on_swap(manager, monkeypatch):
    entries = iter([{"pswpin": 0, "pswpout": 0}, {"pswpin": 0, "pswpout": 1}])
    monkeypatch.setattr(runtime, "resources", lambda p: next(entries))
    monkeypatch.setattr(runtime.time, "sleep", lambda s: None)
    with pytest.raises(runtime.RuntimeErrorSafe, match="Swap"):
        manager.baseline()


def test_complete_baseline_requires_thirteen_samples(manager, monkeypatch):
    sample = Mock(return_value={"pswpin": 0, "pswpout": 0})
    sleeper = Mock()
    monkeypatch.setattr(runtime, "resources", sample)
    monkeypatch.setattr(runtime.time, "sleep", sleeper)
    manager.baseline()
    assert sample.call_count == 13 and sleeper.call_args_list == [__import__("unittest.mock").mock.call(5)] * 12


def test_reuses_existing_fresh_memory_monitor(manager, monkeypatch):
    monkeypatch.setattr(runtime, "fresh_memory", lambda path: {"available_bytes": 4 * 2**30})
    spawn = Mock()
    monkeypatch.setattr(manager, "spawn", spawn)
    manager.ensure_monitor()
    manager.ensure_monitor()
    spawn.assert_not_called()


def test_failed_startup_self_check_does_not_stop_existing_services(manager, monkeypatch):
    monkeypatch.setattr(runtime, "docker_dependencies", Mock())
    monkeypatch.setattr(runtime, "port_open", lambda p: False)
    for name in ("ensure_monitor", "baseline", "ensure_ollama"):
        monkeypatch.setattr(manager, name, Mock())
    monkeypatch.setattr(manager, "preflight_backend", Mock(side_effect=runtime.RuntimeErrorSafe("自检失败")))
    stop = Mock()
    monkeypatch.setattr(manager, "stop_one", stop)
    with pytest.raises(runtime.RuntimeErrorSafe):
        manager.start(reload=True)
    stop.assert_not_called()


def test_private_configuration_permissions_and_atomic_write(tmp_path):
    path = tmp_path / "config.json"
    runtime.write_private(path, {"root": "/project", "credential_file": "/private/credentials.json"})
    assert path.stat().st_mode & 0o777 == 0o600
    assert runtime.private_json(path)["root"] == "/project"
    path.chmod(0o644)
    with pytest.raises(runtime.RuntimeErrorSafe):
        runtime.private_json(path)


def test_foreign_build_directory_is_rejected(tmp_path):
    with pytest.raises(runtime.RuntimeErrorSafe):
        runtime.build_path(tmp_path)


@pytest.mark.parametrize("problem", [None, "foreign", "cloud", "unknown_listener"])
def test_ollama_process_must_be_owned_local_and_cloud_disabled(monkeypatch, problem):
    output = "" if problem == "unknown_listener" else 'LISTEN 127.0.0.1:11434 users:(("ollama",pid=123,fd=3))'
    monkeypatch.setattr(runtime.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=output)))
    r = {**record(), "exe": "/fake/ollama"}
    if problem == "foreign":
        r["uid"] += 1
    monkeypatch.setattr(runtime, "process_record", lambda pid: r)
    original = Path.read_bytes
    payload = b"OLLAMA_NO_CLOUD=true\0OLLAMA_HOST=127.0.0.1:11434\0"
    if problem == "cloud":
        payload = b"OLLAMA_NO_CLOUD=false\0OLLAMA_HOST=127.0.0.1:11434\0"
    monkeypatch.setattr(Path, "read_bytes", lambda p: payload if str(p) == "/proc/123/environ" else original(p))
    if problem:
        with pytest.raises(runtime.RuntimeErrorSafe):
            runtime.verify_ollama_process("/fake/ollama")
    else:
        runtime.verify_ollama_process("/fake/ollama")


def test_low_memory_error_reports_public_metrics_without_lowering_threshold(monkeypatch):
    monkeypatch.setattr(runtime, "fresh_memory", lambda p: {"available_bytes": 2 * 2**30})
    original = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemAvailable: 5242880 kB\n"
        if str(path) == "/proc/pressure/memory":
            return "full avg10=0.00 avg60=0.00 total=0\n"
        if str(path) == "/proc/vmstat":
            return "pswpin 0\npswpout 0\n"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(runtime.RuntimeErrorSafe, match=r"Windows 2\.000 GiB.*需2\.5"):
        runtime.resources("unused")
