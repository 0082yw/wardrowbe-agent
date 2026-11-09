"""Offline checks for the opt-in stability harness, no live dependencies."""
import importlib.util
import sys
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

path = Path(__file__).with_name("run_rotation_stability.py")
spec = importlib.util.spec_from_file_location("stability68", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_test_redis_database_change_preserves_authentication():
    original = make_url("redis://test-user:private%40value@127.0.0.1:16379/1")
    changed = make_url(module.isolated_redis_url(original.render_as_string(hide_password=False)))
    assert changed.database == "2"
    assert changed.username == original.username and changed.password == original.password
    assert changed.host == original.host and changed.port == original.port


@pytest.mark.parametrize("url", [
    "redis://127.0.0.1:6379/15", "redis://example.test:16379/1",
    "rediss://127.0.0.1:16379/1",
])
def test_test_endpoint_cannot_be_changed_to_production_or_external(url):
    with pytest.raises(ValueError):
        module.isolated_redis_url(url)


def test_safe_failure_diagnostic_has_location_without_exception_text():
    module.PHASE = "isolated_redis_connect"
    try:
        raise ValueError("password=secret redis://credentials/private")
    except ValueError as error:
        result = module.failure_details(error)
    import json
    text = json.dumps(result)
    assert result["phase"] == "isolated_redis_connect"
    assert result["error_type"] == "ValueError" and result["frames"]
    assert all(set(frame) == {"file", "line"} for frame in result["frames"])
    assert "secret" not in text and "redis://" not in text and "/private" not in text


def test_network_allows_only_isolated_dependencies():
    for port in (15432, 16379):
        module.network("socket.connect", (None, ("127.0.0.1", port)))
    for address in (("127.0.0.1", 5432), ("127.0.0.1", 6379), ("10.1.1.1", 15432)):
        with pytest.raises(AssertionError):
            module.network("socket.connect", (None, address))
