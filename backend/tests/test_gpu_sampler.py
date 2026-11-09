"""Only pure log association is tested; no SSH, GPU, network or inference."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("gpu_sampler",
    Path(__file__).parents[1] / "scripts/sample_remote_gpu.py")
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)

START = 1760000000.0

def log(*, port=12345, blob="/weights/verified", layers="37/37", age=0):
    import datetime
    stamp = datetime.datetime.fromtimestamp(START + age, datetime.timezone.utc).isoformat()
    return (f'time={stamp} level=INFO msg="starting llama-server" '
            f'cmd="/bin/llama-server --model {blob} --port {port} --host 127.0.0.1"\n'
            f'load_tensors: offloaded {layers} layers to GPU\n')

def parse(value):
    return sampler.matched_layers(value, port="12345", blob="/weights/verified",
                                  started=START, observed=START+1)

def test_layers_are_bound_to_the_actual_runner_and_weight():
    assert parse(log()) == (37, 37)

@pytest.mark.parametrize("value", [
    log(port=12346), log(blob="/weights/other"), log(layers="36/37"),
    log(age=-60), log(age=60), "offloaded 37/37 layers to GPU",
    log()+log(),
])
def test_missing_stale_ambiguous_or_other_runner_evidence_cannot_attest(value):
    assert parse(value) is None


def test_publication_is_atomic_private_and_keeps_only_latest_32(tmp_path):
    import json
    import stat
    path = tmp_path / "evidence.json"
    sampler.publish_observations(path, [{"number": i} for i in range(40)])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["observations"] == [{"number": i} for i in range(8, 40)]
    sampler.publish_observations(path, [{"number": 41}])
    assert json.loads(path.read_text())["observations"] == [{"number": 41}]
    assert list(tmp_path.iterdir()) == [path]


def test_sampler_disables_environment_proxies_and_redirects(monkeypatch):
    import urllib.request
    handlers = []
    class NoNetwork:
        def open(self, *args, **kwargs):
            raise RuntimeError("offline boundary")
    def build(*values):
        handlers.extend(values)
        return NoNetwork()
    monkeypatch.setattr(sampler.urllib.request, "build_opener", build)
    with pytest.raises(RuntimeError, match="offline boundary"):
        sampler.sample(0)
    assert handlers[0].proxies == {}
    redirect = next(h for h in handlers if isinstance(h, urllib.request.HTTPRedirectHandler))
    with pytest.raises(ValueError, match="redirects are forbidden"):
        redirect.redirect_request(None, None, 302, "", {}, "https://external.invalid")


def test_collector_will_not_overwrite_previous_evidence_or_open_another_ssh(tmp_path):
    output = tmp_path / "old.json"
    output.write_text("previous evidence")
    with pytest.raises(ValueError, match="do not overwrite"):
        sampler.collect("/unavailable/control", str(output), str(tmp_path/"raw.log"), 0)
    assert output.read_text() == "previous evidence"
