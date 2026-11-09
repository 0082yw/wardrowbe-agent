"""Verify trusted local sampler records; never probe SSH, infer GPU from HTTP, or load a model."""

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

GPU1_UUID = "GPU-11111111-1111-1111-1111-111111111111"


class GpuObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: str = Field(min_length=1, max_length=128)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    process_id: int = Field(gt=0)
    process_started_at: float = Field(gt=0, allow_inf_nan=False)
    gpu_uuid: str = Field(pattern=r"^GPU-[0-9a-f-]{36}$")
    observed_at: float = Field(gt=0, allow_inf_nan=False)
    layers_loaded: int = Field(gt=0)
    layers_total: int = Field(gt=0)
    model_size_bytes: int = Field(gt=0)
    gpu_size_bytes: int = Field(gt=0)
    process_gpu_memory_bytes: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def fully_loaded(self) -> Self:
        if (self.process_started_at > self.observed_at or self.layers_loaded != self.layers_total
                or self.gpu_size_bytes != self.model_size_bytes):
            raise ValueError("GPU residency or process lifetime is not verified")
        return self


class GpuMemorySample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    observed_at: float = Field(gt=0, allow_inf_nan=False)
    process_gpu_memory_bytes: int = Field(gt=0)


class GpuRequestEvidence(GpuObservation):
    memory_samples: list[GpuMemorySample] = Field(min_length=1, max_length=32)
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    request_started_at: float = Field(gt=0, allow_inf_nan=False)
    request_finished_at: float = Field(gt=0, allow_inf_nan=False)
    evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def within_request(self) -> Self:
        if not self.request_started_at <= self.observed_at <= self.request_finished_at:
            raise ValueError("GPU observation is outside this inference request")
        if self.process_gpu_memory_bytes is None:
            raise ValueError("Missing process VRAM measurement")
        times = [sample.observed_at for sample in self.memory_samples]
        if times != sorted(set(times)) or any(
            not self.request_started_at <= at <= self.observed_at for at in times
        ) or times[-1] != self.observed_at or (
            self.memory_samples[-1].process_gpu_memory_bytes != self.process_gpu_memory_bytes
        ):
            raise ValueError("Invalid request VRAM sampling window")
        if self.gpu_uuid != GPU1_UUID:
            raise ValueError("The approved physical GPU was not verified")
        return self


def gpu_evidence_digest(observation: GpuObservation, samples: list) -> str:
    """One canonical identity for the observation and its PID-bound sample history."""
    canonical = {
        "observation": observation.model_dump(),
        "memory_samples": [GpuMemorySample.model_validate(row).model_dump() for row in samples],
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def read_gpu_evidence(
    filename: str | None, *, request_id: str, model: str, digest: str | None,
    started_at: float, finished_at: float, now: float,
) -> tuple[GpuRequestEvidence | None, Literal["unavailable", "rejected", "verified"]]:
    """A trusted startup path only. No retries, historical cache or user supplied evidence.

    The existing external sampler must publish {"observations": [...]} atomically
    to a local, current-user-owned 0600 file. Each record combines the same model
    digest, runner PID/start time, physical UUID, full layer count and residency.
    The backend binds a fresh record to its own successful HTTP inference window.
    An absent producer remains unavailable; HTTP success alone never verifies GPU.
    """
    if filename is None:
        return None, "unavailable"
    try:
        path = Path(filename)
        if not path.is_absolute():
            raise ValueError
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "r") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 65536):
                raise ValueError
            content = stream.read(65537)
            if len(content.encode()) > 65536:
                raise ValueError
            payload = json.loads(content)
        if not isinstance(payload, dict) or set(payload) != {"observations"}:
            raise ValueError
        observations = payload["observations"]
        if not isinstance(observations, list) or not 1 <= len(observations) <= 32:
            raise ValueError
        validated = [GpuObservation.model_validate(raw) for raw in observations]
        times = [row.observed_at for row in validated]
        if times != sorted(set(times)):
            raise ValueError("Ambiguous observation chronology")
        for observation in reversed(validated):
            if (observation.process_gpu_memory_bytes is None or observation.model != model or observation.model_digest != digest
                    or observation.gpu_uuid != GPU1_UUID
                    or not started_at <= observation.observed_at <= finished_at
                    or not 0 <= now - observation.observed_at <= 10):
                continue
            identity = ("model", "model_digest", "process_id", "process_started_at", "gpu_uuid",
                        "layers_loaded", "layers_total", "model_size_bytes", "gpu_size_bytes")
            samples = [{
                "observed_at": row.observed_at,
                "process_gpu_memory_bytes": row.process_gpu_memory_bytes,
            } for row in validated if row.process_gpu_memory_bytes is not None
                and all(getattr(row, key) == getattr(observation, key) for key in identity)
                and started_at <= row.observed_at <= observation.observed_at]
            proof = GpuRequestEvidence.model_validate({
                **observation.model_dump(), "memory_samples": samples, "request_id": request_id,
                "request_started_at": started_at, "request_finished_at": finished_at,
                "evidence_id": gpu_evidence_digest(observation, samples),
            })
            return proof, "verified"
        return None, "rejected"
    except FileNotFoundError:
        return None, "unavailable"
    except (OSError, ValueError, TypeError):
        return None, "rejected"
