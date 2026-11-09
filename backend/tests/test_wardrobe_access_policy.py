"""New permission contract tests; no historical CRUD suite or production services."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

import pytest

from app.access_policy import (
    WardrobeAccessPolicy, manifest_bytes, manifest_digest, validate_launch_config,
)
from app.services.application_access import ApplicationAccessMiddleware

ROOT = Path(__file__).resolve().parents[2]
ID = str(uuid4())
CASES = [
    ("GET", "/api/v1/items", True, True),
    ("POST", "/api/v1/agent/preview", True, True),
    ("POST", "/api/v1/auth/sync", True, True),
    ("POST", "/api/v1/items", False, True),
    ("PATCH", f"/api/v1/items/{ID}", False, True),
    ("DELETE", f"/api/v1/items/{ID}", False, True),
    ("POST", f"/api/v1/items/{ID}/archive", False, True),
    ("POST", f"/api/v1/items/{ID}/restore", False, True),
    ("POST", f"/api/v1/items/{ID}/rotate", False, True),
    ("POST", "/api/v1/items/bulk", False, False),
    ("POST", "/api/v1/outfits", False, False),
    ("PATCH", "/api/v1/users/me", False, False),
    ("POST", f"/api/v1/items/{ID}/remove-background", False, False),
    ("POST", f"/api/v1/items/{ID}/analyze", False, False),
    ("PATCH", "/api/v1/items/not-a-uuid", False, False),
    ("POST", "/api/v1/items/", False, False),
]

@pytest.mark.parametrize("mode,column", [("read_only_demo", 2), ("full_app", 3)])
@pytest.mark.parametrize("case", CASES)
def test_http_policy_and_backend_middleware(case, mode, column, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "wardrobe_app_mode", mode)
    method, path = case[:2]
    expected = case[column]
    assert WardrobeAccessPolicy(mode).allows_http(method, path) == expected

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})
    messages = []
    async def send(message):
        messages.append(message)
    async def receive():
        return {"type": "http.request", "body": b""}
    asyncio.run(ApplicationAccessMiddleware(downstream)(
        {"type": "http", "path": path, "method": method}, receive, send,
    ))
    assert messages[0]["status"] == (204 if expected else 403)

def test_default_and_invalid_modes():
    assert WardrobeAccessPolicy().mode == "read_only_demo"
    with pytest.raises(ValueError):
        WardrobeAccessPolicy("browser-controlled")
    assert validate_launch_config({}, "read_only_demo", owner=os.getuid()).mode == "read_only_demo"
    with pytest.raises(ValueError):
        validate_launch_config({}, "full_app", owner=os.getuid())

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

@pytest.mark.parametrize("mode", ["read_only_demo", "full_app"])
def test_staged_adapters_and_launch_integrity(tmp_path, mode, monkeypatch):
    config_path = Path.home() / ".config/wardrowbe-agent/local-runtime.json"
    if not config_path.exists():
        pytest.skip("Existing private runtime bundle required; no service is created")
    initial = config_path.read_bytes()
    original = json.loads(initial)
    active = Path(original.get("guard_directory", Path.home() / ".local/state/wardrowbe-agent/local-runtime"))
    hashes = {name: hashlib.sha256((active / name).read_bytes()).hexdigest() for name in original["guard_hashes"]}
    stager = module(ROOT / "backend/scripts/stage_application_policy.py", "permission_stager")
    path = stager.stage(config_path, tmp_path / "bundle", mode)
    candidate = json.loads(path.read_text())
    assert validate_launch_config(candidate, mode, owner=os.getuid()).mode == mode
    launcher = module(ROOT / "backend/scripts/local_agent.py", "permission_launcher")
    monkeypatch.setenv("WARDROBE_APP_MODE", mode)
    assert launcher.application_policy(candidate).mode == mode

    directory = path.parent
    compile((directory / "backend.py").read_text(), "backend.py", "exec")
    compile((directory / "backend-entry.py").read_text(), "entry.py", "exec")
    node = original["node"]
    subprocess.run([node, "--check", str(directory / "frontend-guard.mjs")], check=True, capture_output=True)
    # Import the complete staged frontend guard: retains old network/login protection.
    subprocess.run([node, "--input-type=module", "-e",
                    "await import(process.argv[1]);", (directory / "frontend-guard.mjs").as_uri()],
                   check=True, capture_output=True, env={**os.environ, "WARDROBE_APP_MODE": mode})
    assert config_path.read_bytes() == initial
    assert hashes == {name: hashlib.sha256((active / name).read_bytes()).hexdigest() for name in hashes}

    broken = json.loads(path.read_text())
    broken["application_policy"]["source_hashes"] = {}
    with pytest.raises(ValueError):
        validate_launch_config(broken, mode, owner=os.getuid())
    (directory / "permissions.json").write_text("{}")
    with pytest.raises(ValueError):
        validate_launch_config(candidate, mode, owner=os.getuid())

def test_frontend_uses_identical_manifest(tmp_path):
    config_path = Path.home() / ".config/wardrowbe-agent/local-runtime.json"
    if not config_path.exists():
        pytest.skip("Existing local Node runtime configuration required")
    config = json.loads(config_path.read_text())
    path = tmp_path / "permissions.json"
    path.write_bytes(manifest_bytes())
    path.chmod(0o600)
    script = """
import {loadPolicy, allowsHttp, assertHttp} from './frontend/scripts/application-policy.mjs';
const [path, digest, cases] = JSON.parse(process.argv[1]);
const result = ['read_only_demo','full_app'].map(mode => {
  const p=loadPolicy(path,digest,mode);
  return cases.map(c=>allowsHttp(p,c[0],c[1]));
});
const p=loadPolicy(path,digest,'full_app');
let rejected=0;
for (const [url,headers] of [
 ['http://127.0.0.1:8001/api/v1/items',{}],
 ['http://example.com:8001/api/v1/items',{authorization:'Bearer unit-test'}],
 ['http://127.0.0.1:8000/api/v1/items',{authorization:'Bearer unit-test'}]
]) {try {assertHttp(p,new URL(url),'POST',headers)} catch {rejected++}}
assertHttp(p,new URL('http://127.0.0.1:8001/api/v1/items'),'POST',{authorization:'Bearer unit-test'});
console.log(JSON.stringify({result,rejected}));
"""
    r = subprocess.run([config["node"], "--input-type=module", "-e", script,
                        json.dumps([str(path), manifest_digest(), CASES])],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    assert data["result"] == [[c[2] for c in CASES], [c[3] for c in CASES]]
    assert data["rejected"] == 3
