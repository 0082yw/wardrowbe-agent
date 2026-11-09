"""Exercise the real Windows sampler in isolated NTFS files, without app/DB access."""

import base64
import json
from pathlib import Path
import subprocess
import time

import pytest


SAMPLER = Path(__file__).parents[1] / "scripts/windows_memory.ps1"
POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
pytestmark = pytest.mark.skipif(not POWERSHELL.is_file(), reason="Requires Windows interop")


def ps_args(code):
    code = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);\n" + code
    encoded = base64.b64encode(code.encode("utf-16le")).decode()
    return [str(POWERSHELL), "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-EncodedCommand", encoded]


def run_ps(code):
    return subprocess.run(ps_args(code), capture_output=True, text=True, timeout=20)


def quoted(value):
    return "'" + str(value).replace("'", "''") + "'"


@pytest.fixture(scope="module")
def native_directory():
    result = run_ps("$p=Join-Path ([IO.Path]::GetTempPath()) ('wardrowbe-memory-tests-'+[guid]::NewGuid().ToString('N')); [IO.Directory]::CreateDirectory($p)|Out-Null; Write-Output $p")
    assert result.returncode == 0, result.stderr
    windows = result.stdout.strip()
    assert not windows.startswith("\\\\")
    local = subprocess.run(["wslpath", "-u", windows], capture_output=True, text=True, check=True).stdout.strip()
    # Keep these small files as failure evidence; never clean project/user files.
    return windows, Path(local)


def invocation(directory, name, *, once=True):
    windows, _ = directory
    output = windows + "\\" + name + ".json"
    stop = windows + "\\" + name + ".stop"
    code = "& {\n" + SAMPLER.read_text() + "\n} -OutputPath " + quoted(output) + " -StopPath " + quoted(stop)
    return code + (" -Once" if once else "")


def test_powershell_parser_and_no_cim_sampling():
    source = SAMPLER.read_text()
    assert "Get-CimInstance" not in source and "Get-WmiObject" not in source
    code = "$errors=$null; $tokens=$null; [void][Management.Automation.Language.Parser]::ParseInput(" + quoted(source) + ",[ref]$tokens,[ref]$errors); if($errors.Count){throw 'PowerShell syntax errors'}"
    result = run_ps(code)
    assert result.returncode == 0, result.stderr


def test_single_sample_preserves_format_and_matches_system(native_directory):
    result = run_ps(invocation(native_directory, "once") + "; if($LASTEXITCODE){exit $LASTEXITCODE}; $os=Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 5; @{total=[int64]$os.TotalVisibleMemorySize*1024;available=[int64]$os.FreePhysicalMemory*1024}|ConvertTo-Json -Compress")
    assert result.returncode == 0, result.stderr
    system = json.loads(result.stdout.strip())
    path = native_directory[1] / "once.json"
    sample = json.loads(path.read_text())
    diagnostic = json.loads(Path(str(path) + ".monitor.json").read_text())
    assert set(sample) == {"available_bytes", "unix_seconds"}
    assert type(sample["available_bytes"]) is int
    assert 0 <= time.time() - sample["unix_seconds"] < 10
    assert diagnostic["source"] == "GlobalMemoryStatusEx"
    assert diagnostic["exit_reason"] == "single_sample_complete"
    assert abs(diagnostic["total_physical_bytes"] - system["total"]) < 64 * 2**20
    assert abs(sample["available_bytes"] - system["available"]) < 512 * 2**20
    expected_load = 100 * (1 - sample["available_bytes"] / diagnostic["total_physical_bytes"])
    assert abs(expected_load - diagnostic["memory_load_percent"]) < 2


def test_atomic_first_write_and_replace(native_directory):
    for _ in range(2):
        result = run_ps(invocation(native_directory, "atomic"))
        assert result.returncode == 0, result.stderr
        path = native_directory[1] / "atomic.json"
        assert json.loads(path.read_text())["available_bytes"] > 0
        assert not Path(str(path) + ".next").exists()
        assert not Path(str(path) + ".monitor.json.next").exists()


def test_failed_write_exits_with_diagnostics_without_publishing_success(native_directory):
    windows, local = native_directory
    output = windows + "\\blocked"
    (local / "blocked").mkdir()
    code = "& {\n" + SAMPLER.read_text() + "\n} -OutputPath " + quoted(output) + " -StopPath " + quoted(windows + "\\blocked.stop") + " -Once"
    result = run_ps(code)
    assert result.returncode != 0
    diagnostic = json.loads((local / "blocked.monitor.json").read_text())
    assert diagnostic["state"] == "failed" and diagnostic["sample_count"] == 0
    assert diagnostic["last_success_unix_seconds"] is None
    for key in ("error_type", "error_message", "error_hresult", "error_id", "error_stack", "error_line"):
        assert diagnostic[key] is not None
    assert "wardrowbe_memory_monitor_failed" in result.stderr
    assert (local / "blocked").is_dir()


def test_stop_marker_exits_without_orphan(native_directory):
    process = subprocess.Popen(ps_args(invocation(native_directory, "stop", once=False)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    path = native_directory[1] / "stop.json"
    deadline = time.monotonic() + 15
    while not path.exists() and time.monotonic() < deadline:
        assert process.poll() is None
        time.sleep(0.2)
    assert path.is_file()
    (native_directory[1] / "stop.stop").touch()
    _, stderr = process.communicate(timeout=8)
    assert process.returncode == 0, stderr
    diagnostic = json.loads(Path(str(path) + ".monitor.json").read_text())
    assert diagnostic["state"] == "stopped" and diagnostic["exit_reason"] == "stop_marker"
    assert diagnostic["sample_count"] >= 1
