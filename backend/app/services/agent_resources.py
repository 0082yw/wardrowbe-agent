"""按模式校验真实快照；仅对跨系统发布时的短暂 ENOENT 有限重新打开。"""

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.services.ollama_agent_model import OllamaModelError

logger = logging.getLogger(__name__)


class _InvalidSample(ValueError):
    """只携带固定诊断代码，不携带文件内容或路径。"""


def _read_windows_snapshot(path: Path) -> str:
    # File.Replace is atomic on Windows, but WSL's view of the NTFS name can
    # briefly return ENOENT during publication. Reopen only this exact path;
    # never retry parsing, permissions, stale samples or any model request.
    for attempt in range(1, 4):
        try:
            result = path.read_text()
        except FileNotFoundError as exc:
            if attempt == 3:
                exc.resource_open_attempts = attempt
                raise
            time.sleep(0.01)
        else:
            if attempt > 1:
                logger.debug("agent_resource_snapshot_reopened attempts=%d", attempt)
            return result
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class ResourceSnapshot:
    windows_bytes: int
    wsl_bytes: int
    psi_full: float
    swap_in: int
    swap_out: int


def validate_resources(snapshot: ResourceSnapshot, mode: str) -> None:
    if mode not in ("local_cpu", "remote_gpu"):
        raise ValueError("资源保护模式无效")
    windows_min, wsl_min = (2.5, 1.25) if mode == "local_cpu" else (1, 0.5)
    if snapshot.windows_bytes < windows_min * 2**30:
        raise OllamaModelError("Windows 可用内存不足，本次未执行模型请求。", code="windows_memory_low")
    if snapshot.wsl_bytes < wsl_min * 2**30:
        raise OllamaModelError("WSL 可用内存不足，本次未执行模型请求。", code="wsl_memory_low")
    if not 0 <= snapshot.psi_full <= 1:
        raise OllamaModelError("WSL 内存压力过高，本次未执行模型请求。", code="wsl_memory_pressure")


def read_resources(memory_file: str | None = None) -> ResourceSnapshot:
    """只接受当前启动进程提供的采样位置；缺失/过期时关闭而非猜测余量。"""
    source = "configuration"
    age = None
    try:
        name = memory_file or os.environ.get("WARDROWBE_MEMORY_FILE")
        if not name:
            raise _InvalidSample("missing_path")
        source = "windows_snapshot"
        sample = json.loads(_read_windows_snapshot(Path(name)))
        if not isinstance(sample, dict):
            raise _InvalidSample("invalid_snapshot_shape")
        timestamp = sample["unix_seconds"]
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise _InvalidSample("invalid_timestamp")
        age = time.time() - timestamp
        if age < 0:
            raise _InvalidSample("future_timestamp")
        if age > 10:
            raise _InvalidSample("stale_snapshot")
        windows = sample["available_bytes"]
        if type(windows) is not int or windows < 0:
            raise _InvalidSample("invalid_windows_counter")
        source = "wsl_meminfo"
        mem = re.search(r"(?m)^MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())
        if mem is None:
            raise _InvalidSample("missing_memory_counter")
        source = "wsl_pressure"
        psi = re.search(r"(?m)^full avg10=([\d.]+)", Path("/proc/pressure/memory").read_text())
        if psi is None:
            raise _InvalidSample("missing_pressure_counter")
        pressure = float(psi[1])
        if not math.isfinite(pressure):
            raise _InvalidSample("invalid_pressure_counter")
        source = "wsl_vmstat"
        swaps = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
        swap_in, swap_out = int(swaps["pswpin"]), int(swaps["pswpout"])
        if swap_in < 0 or swap_out < 0:
            raise _InvalidSample("invalid_swap_counter")
        return ResourceSnapshot(windows, int(mem[1]) * 1024, pressure, swap_in, swap_out)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        # Never log raw exception messages, file paths, snapshots or environment values.
        reason = (
            str(exc) if isinstance(exc, _InvalidSample)
            else "io_error" if isinstance(exc, OSError)
            else "invalid_json" if isinstance(exc, json.JSONDecodeError)
            else "missing_counter" if isinstance(exc, KeyError)
            else "invalid_counter"
        )
        diagnostic = {"source": source, "reason": reason, "exception_type": type(exc).__name__}
        if isinstance(exc, OSError):
            diagnostic["errno"] = exc.errno
        attempts = getattr(exc, "resource_open_attempts", None)
        if type(attempts) is int:
            diagnostic["snapshot_open_attempts"] = attempts
        if age is not None and math.isfinite(age):
            diagnostic["snapshot_age_seconds"] = round(age, 6)
        logger.warning("agent_resource_read_failed %s", json.dumps(diagnostic, sort_keys=True))
        error = OllamaModelError("系统资源采样不可用，本次未执行模型请求。", code="resources_unavailable")
        error.resource_diagnostics = diagnostic
        raise error from None


def check_remote_resources(baseline: ResourceSnapshot | None = None, *, memory_file: str | None = None) -> ResourceSnapshot:
    value = read_resources(memory_file)
    validate_resources(value, "remote_gpu")
    if baseline is not None and (value.swap_in, value.swap_out) != (baseline.swap_in, baseline.swap_out):
        raise OllamaModelError("系统出现 Swap 活动，本次停止模型请求。", code="swap_activity")
    return value
