"""Resource parsing and diagnostics only; no model, database or network access."""

import errno
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.services import agent_resources as resources
from app.services.ollama_agent_model import OllamaModelError


@pytest.fixture
def sources(monkeypatch):
    values = {
        "/snapshot.json": json.dumps({"unix_seconds": 995, "available_bytes": 2**30}),
        "/proc/meminfo": "MemAvailable: 524288 kB\n",
        "/proc/pressure/memory": "full avg10=1.00 avg60=0.00\n",
        "/proc/vmstat": "pswpin 7\npswpout 8\n",
    }

    def read(path, *args, **kwargs):
        value = values[str(path)]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(resources.time, "time", lambda: 1000)
    monkeypatch.setattr(resources.time, "sleep", Mock())
    monkeypatch.setattr(resources.logger, "warning", Mock())
    return values


def failure(reason, source="windows_snapshot"):
    with pytest.raises(OllamaModelError) as caught:
        resources.read_resources("/snapshot.json")
    error = caught.value
    assert error.code == "resources_unavailable"
    assert error.resource_diagnostics["source"] == source
    assert error.resource_diagnostics["reason"] == reason
    return error


def test_valid_snapshot_keeps_remote_boundary_and_cpu_gate(sources):
    result = resources.read_resources("/snapshot.json")
    assert result == resources.ResourceSnapshot(2**30, 512 * 2**20, 1, 7, 8)
    resources.validate_resources(result, "remote_gpu")
    with pytest.raises(OllamaModelError) as caught:
        resources.validate_resources(result, "local_cpu")
    assert caught.value.code == "windows_memory_low"
    resources.logger.warning.assert_not_called()


@pytest.mark.parametrize("timestamp,reason", [
    (989.999, "stale_snapshot"), (1000.001, "future_timestamp"),
    (None, "invalid_timestamp"), (True, "invalid_timestamp"),
    ("995", "invalid_timestamp"), (float("nan"), "invalid_timestamp"),
    (float("inf"), "invalid_timestamp"),
])
def test_invalid_timestamp_is_diagnosed_and_never_retried(sources, timestamp, reason):
    sources["/snapshot.json"] = json.dumps({"unix_seconds": timestamp, "available_bytes": 2**30})
    failure(reason)
    assert resources.logger.warning.call_count == 1


@pytest.mark.parametrize("timestamp", [990, 1000])
def test_existing_ten_second_boundary_is_unchanged(sources, timestamp):
    sources["/snapshot.json"] = json.dumps({"unix_seconds": timestamp, "available_bytes": 2**30})
    assert resources.read_resources("/snapshot.json").windows_bytes == 2**30


@pytest.mark.parametrize("payload,reason", [
    ("", "invalid_json"), ("{truncated", "invalid_json"),
    ("[]", "invalid_snapshot_shape"), ("null", "invalid_snapshot_shape"),
    ('{"unix_seconds":995}', "missing_counter"),
    ('{"unix_seconds":995,"available_bytes":true}', "invalid_windows_counter"),
    ('{"unix_seconds":995,"available_bytes":-1}', "invalid_windows_counter"),
])
def test_malformed_snapshot_has_no_last_good_cache(sources, payload, reason):
    assert resources.read_resources("/snapshot.json")
    sources["/snapshot.json"] = payload
    failure(reason)


@pytest.mark.parametrize("path,source", [
    ("/snapshot.json", "windows_snapshot"), ("/proc/meminfo", "wsl_meminfo"),
    ("/proc/pressure/memory", "wsl_pressure"), ("/proc/vmstat", "wsl_vmstat"),
])
def test_io_error_reports_source_and_errno_without_sensitive_data(sources, path, source):
    secret = "private-token-must-not-appear"
    sources[path] = PermissionError(errno.EACCES, secret, "/sensitive/" + secret)
    error = failure("io_error", source)
    assert error.resource_diagnostics["errno"] == errno.EACCES
    log = str(resources.logger.warning.call_args)
    assert secret not in log and "/sensitive/" not in log
    assert secret not in str(error) and secret not in str(error.resource_diagnostics)


@pytest.mark.parametrize("path,value,source,reason", [
    ("/proc/meminfo", "", "wsl_meminfo", "missing_memory_counter"),
    ("/proc/pressure/memory", "", "wsl_pressure", "missing_pressure_counter"),
    ("/proc/vmstat", "pswpin 7\n", "wsl_vmstat", "missing_counter"),
    ("/proc/vmstat", "pswpin bad\npswpout 8\n", "wsl_vmstat", "invalid_counter"),
    ("/proc/vmstat", "pswpin -1\npswpout 8\n", "wsl_vmstat", "invalid_swap_counter"),
])
def test_proc_counter_failures_are_distinguishable(sources, path, value, source, reason):
    sources[path] = value
    failure(reason, source)


def test_missing_configuration_is_not_read_from_another_location(sources, monkeypatch):
    monkeypatch.delenv("WARDROWBE_MEMORY_FILE", raising=False)
    with pytest.raises(OllamaModelError) as caught:
        resources.read_resources()
    assert caught.value.resource_diagnostics["reason"] == "missing_path"


def test_new_swap_activity_still_stops(sources):
    baseline = resources.check_remote_resources(memory_file="/snapshot.json")
    sources["/proc/vmstat"] = "pswpin 8\npswpout 8\n"
    with pytest.raises(OllamaModelError) as caught:
        resources.check_remote_resources(baseline, memory_file="/snapshot.json")
    assert caught.value.code == "swap_activity"


@pytest.mark.parametrize("missing_reads", [1, 2])
def test_transient_snapshot_enoent_reopens_exact_path_and_checks_freshness(sources, monkeypatch, missing_reads):
    read = Path.read_text
    attempts = []

    def transient(path, *args, **kwargs):
        if str(path) == "/snapshot.json":
            attempts.append(str(path))
            if len(attempts) <= missing_reads:
                raise FileNotFoundError(errno.ENOENT, "temporary publication gap")
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient)
    result = resources.check_remote_resources(memory_file="/snapshot.json")
    assert result.windows_bytes == 2**30
    assert attempts == ["/snapshot.json"] * (missing_reads + 1)
    assert resources.time.sleep.call_count == missing_reads
    assert all(call.args == (0.01,) for call in resources.time.sleep.call_args_list)


def test_persistent_missing_snapshot_stops_after_three_opens_without_cache(sources, monkeypatch):
    assert resources.read_resources("/snapshot.json")
    attempts = []

    def missing(path, *args, **kwargs):
        attempts.append(str(path))
        raise FileNotFoundError(errno.ENOENT, "private content never logged")

    monkeypatch.setattr(Path, "read_text", missing)
    error = failure("io_error")
    assert error.resource_diagnostics["snapshot_open_attempts"] == 3
    assert attempts == ["/snapshot.json"] * 3
    assert resources.time.sleep.call_count == 2
    assert resources.logger.warning.call_count == 1


@pytest.mark.parametrize("payload,reason", [
    ('{"unix_seconds":989,"available_bytes":1073741824}', "stale_snapshot"),
    ('{"unix_seconds":1001,"available_bytes":1073741824}', "future_timestamp"),
    ("{truncated", "invalid_json"),
])
def test_reopening_cannot_bypass_data_validation(sources, monkeypatch, payload, reason):
    sources["/snapshot.json"] = payload
    read = Path.read_text
    attempts = 0

    def transient(path, *args, **kwargs):
        nonlocal attempts
        if str(path) == "/snapshot.json":
            attempts += 1
            if attempts == 1:
                raise FileNotFoundError(errno.ENOENT, "publication gap")
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient)
    failure(reason)
    assert attempts == 2


def test_permission_errors_are_never_reopened(sources, monkeypatch):
    attempts = Mock(side_effect=PermissionError(errno.EACCES, "sensitive message"))
    monkeypatch.setattr(Path, "read_text", attempts)
    failure("io_error")
    assert attempts.call_count == 1
    resources.time.sleep.assert_not_called()


def test_missing_proc_file_does_not_reopen_or_hide_diagnostics(sources):
    sources["/proc/meminfo"] = FileNotFoundError(errno.ENOENT, "missing proc file")
    error = failure("io_error", "wsl_meminfo")
    assert "snapshot_open_attempts" not in error.resource_diagnostics
    resources.time.sleep.assert_not_called()
