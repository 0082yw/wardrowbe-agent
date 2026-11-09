"""Read-only evidence sampler for the approved private Ollama. No inference or SSH creation.

Run on the server over an existing authenticated SSH channel. Only stdout is
written; the trusted local caller publishes validated observations atomically.
"""
import datetime
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/srv/wardrowbe/ollama")
GPU = "GPU-11111111-1111-1111-1111-111111111111"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"


def matched_layers(log, *, port, blob, started, observed):
    """Bind offload lines to this runner's exact port, weight and start time."""
    blocks = re.split(r"(?=time=\S+ .*msg=\"starting llama-server\")", log)
    matches = []
    for block in blocks:
        head = block.splitlines()[0] if block else ""
        if f"--port {port} " not in head or f"--model {blob} " not in head:
            continue
        timestamp = re.search(r"^time=(\S+)", head)
        if not timestamp:
            continue
        at = datetime.datetime.fromisoformat(timestamp[1]).timestamp()
        if not started - 2 <= at <= started + 2 or at > observed:
            continue
        counts = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", block)
        if len(counts) == 1:
            loaded, total = map(int, counts[0])
            if loaded > 0 and loaded == total:
                matches.append((loaded, total))
    return matches[0] if len(matches) == 1 else None


def sample(offset):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("Evidence endpoint redirects are forbidden")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open("http://127.0.0.1:11434/api/ps", timeout=3) as response:
        runtime = json.load(response)
    gpu_rows = subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True, timeout=3).splitlines()
    assignments = {}
    for row in gpu_rows:
        pid, uuid, memory = (part.strip() for part in row.split(","))
        assignments.setdefault(int(pid), []).append((uuid, int(memory)))
    loaded = [m for m in runtime["models"] if m.get("name") == MODEL]
    result = {"observed_at": time.time(), "loaded_count": len(loaded),
              "observation": None, "own_gpu_processes": []}
    manifest = json.loads((ROOT / "models/manifests/registry.ollama.ai/library/qwen3/4b-instruct-2507-q4_K_M").read_text())
    weights = [x["digest"] for x in manifest["layers"]
               if x["mediaType"] == "application/vnd.ollama.image.model"]
    if len(weights) != 1:
        raise ValueError("Ambiguous model weights")
    blob = str(ROOT / "models/blobs" / weights[0].replace(":", "-"))
    boot = int(re.search(r"(?m)^btime (\d+)$", Path("/proc/stat").read_text())[1])
    ticks = os.sysconf("SC_CLK_TCK")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid() or not proc.joinpath("exe").resolve().is_relative_to(ROOT / "bin"):
                continue
            args = proc.joinpath("cmdline").read_bytes().decode().split("\0")
            if "--model" not in args or args[args.index("--model")+1] != blob or "--port" not in args:
                continue
            pid = int(proc.name)
            devices = assignments.get(pid, [])
            result["own_gpu_processes"].append({"pid": pid, "devices": devices})
            if any(uuid != GPU for uuid, _ in devices):
                raise ValueError("Own runner is on an unapproved GPU")
            started = boot + int(proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]) / ticks
            with (ROOT / "logs/ollama-v0.40.13.log").open() as stream:
                stream.seek(offset)
                log = stream.read()
            layers = matched_layers(log, port=args[args.index("--port")+1], blob=blob,
                                    started=started, observed=result["observed_at"])
            if len(loaded) != 1 or len(devices) != 1 or not layers:
                continue
            model = loaded[0]
            if (model.get("digest") != DIGEST or not model.get("size")
                    or model["size"] != model.get("size_vram")):
                continue
            result["observation"] = {
                "model": MODEL, "model_digest": DIGEST, "process_id": pid,
                "process_started_at": float(started), "gpu_uuid": devices[0][0],
                "observed_at": float(result["observed_at"]),
                "layers_loaded": layers[0], "layers_total": layers[1],
                "model_size_bytes": model["size"], "gpu_size_bytes": model["size_vram"],
                "process_gpu_memory_bytes": devices[0][1] * 1024 * 1024,
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def publish_observations(path, rows):
    """Same-directory atomic, current-user 0600 publication; no historical cache."""
    import tempfile
    path = Path(path)
    if not path.is_absolute() or path.parent.stat().st_uid != os.getuid():
        raise ValueError("Untrusted evidence directory")
    fd, temporary = tempfile.mkstemp(prefix=".gpu-observation-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"observations": rows[-32:]}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collect(control, output, log, offset):
    """Consume only an existing SSH master. Never authenticate or open a tunnel."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.services.gpu_evidence import GpuObservation

    if Path(output).exists() or Path(log).exists():
        raise ValueError("Use new evidence and log paths; do not overwrite prior evidence")
    command = ["ssh", "-T", "-S", control, "-o", "ProxyCommand=/bin/false",
               "-o", "BatchMode=yes", "gpu-user@inference.example.invalid"]
    subprocess.run(command[:1] + ["-S", control, "-O", "check", "gpu-user@inference.example.invalid"],
                   check=True, capture_output=True, timeout=5)
    proc = subprocess.Popen(command + ["python3", "-B", "-", "--remote", str(int(offset))],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    proc.stdin.write(Path(__file__).read_text())
    proc.stdin.close()
    rows = []
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        for line in proc.stdout:
            packet = json.loads(line)
            stream.write(json.dumps(packet) + "\n")
            stream.flush()
            observation = packet["observation"]
            if observation is not None:
                row = GpuObservation.model_validate(observation)
                if row.gpu_uuid != GPU or row.model != MODEL or row.model_digest != DIGEST:
                    raise ValueError("Evidence identity mismatch")
                rows.append(row.model_dump())
                publish_observations(output, rows)
    if proc.wait(timeout=10) != 0:
        raise RuntimeError("Remote sampler stopped; no inference retry")
    print(json.dumps({"observations": len(rows)}))


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "--collect":
        collect(*sys.argv[2:])
    else:
        if sys.argv[1] != "--remote":
            raise ValueError("An explicit sampling mode is required")
        offset = int(sys.argv[2])
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            print(json.dumps(sample(offset)), flush=True)
            time.sleep(.15)
