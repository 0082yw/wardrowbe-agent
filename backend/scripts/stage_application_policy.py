"""Prepare a versioned permission bundle. Never changes active config or services."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.access_policy import WardrobeAccessPolicy, IMPLEMENTATION_FILES, manifest_bytes, manifest_digest, validate_launch_config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(data)


def stage(config_path: Path, output: Path, mode: str) -> Path:
    WardrobeAccessPolicy(mode)
    config = json.loads(config_path.read_text())
    active = Path(config.get("guard_directory", Path.home() / ".local/state/wardrowbe-agent/local-runtime"))
    original = {name: digest(active / name) for name in config["guard_hashes"]}
    if original != config["guard_hashes"]:
        raise ValueError("Active guard hashes changed")
    if output.exists() or output.parent.is_symlink():
        raise ValueError("Output must be a new owned directory")
    output.mkdir(mode=0o700)
    write(output / "permissions.json", manifest_bytes().decode())
    backend = (active / "backend.py").read_text()
    marker = "settings = get_settings()\n"
    assert backend.count(marker) == 1
    backend = backend.replace(marker, marker + (
        "from app.access_policy import WardrobeAccessPolicy\n"
        "from app.write_guard import install_write_guard, authorize_statement\n"
        f"_access_policy = WardrobeAccessPolicy({mode!r})\n"
        "assert settings.wardrobe_app_mode == _access_policy.mode\n"
    ), 1)
    marker = "from app.database import engine\n"
    assert backend.count(marker) == 1
    backend = backend.replace(marker, marker + "install_write_guard(engine, sql_listener=False)\n", 1)
    tree = ast.parse(backend)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RUNTIME" for t in node.targets):
            node.value = ast.Constant(str(active))
            node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[node.value], keywords=[])
        if isinstance(node, ast.FunctionDef) and node.name == "sql_guard":
            replacement = ast.parse('''
def sql_guard(conn, cursor, statement, parameters, context, executemany):
    assert (conn.engine.url.host, conn.engine.url.database, conn.engine.url.username) == ('127.0.0.1', 'wardrowbe_agent_dev', 'wardrowbe_agent_dev')
    record = ctx.get()
    kind = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ''
    try:
        authorize_statement(conn, statement, context)
    except PermissionError:
        if record is not None:
            record['blocked_sql'].append(kind)
        raise
    if record is not None:
        record['database_reads'] += kind == 'SELECT'
        if kind in ('INSERT', 'UPDATE', 'DELETE'):
            record['database_writes'] = record.get('database_writes', 0) + 1
''').body[0]
            node.body = replacement.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "observe_request":
            condition = next(n for n in node.body if isinstance(n, ast.If))
            condition.test = ast.parse("not _access_policy.allows_http(request.method, request.url.path)", mode="eval").body
    backend = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    compile(backend, str(output / "backend.py"), "exec")
    write(output / "backend.py", backend)
    auth = json.loads((active / "auth-state.json").read_text())
    auth["original_bootstrap"] = str(output / "backend.py")
    auth["original_bootstrap_sha256"] = digest(output / "backend.py")
    write(output / "auth-state.json", auth)

    entry = ast.parse((active / "backend-entry.py").read_text())
    for node in ast.walk(entry):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "STATE=Path(" in node.value:
            source = node.value.replace(str(active / "auth-state.json"), str(output / "auth-state.json"))
            assert source != node.value
            node.value = source
        if isinstance(node, ast.FunctionDef) and node.name == "_attest_loaded":
            checks = ast.parse(
                "from app.write_guard import guard_installed\n"
                "assert guard_installed(engine)\n"
                f"assert get_settings().wardrobe_app_mode == {mode!r}\n"
            ).body
            position = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "settings" for t in n.targets))
            node.body[position + 1:position + 1] = checks
    entry = ast.unparse(ast.fix_missing_locations(entry)) + "\n"
    compile(entry, str(output / "backend-entry.py"), "exec")
    write(output / "backend-entry.py", entry)

    frontend = (active / "frontend-guard.mjs").read_text()
    adapter = (ROOT / "frontend/scripts/application-policy.mjs").as_uri()
    prefix = (
        f"import {{ loadPolicy as _loadApplicationPolicy, assertHttp as _assertApplicationHttp }} from {json.dumps(adapter)};\n"
        f"const _applicationPolicy = _loadApplicationPolicy({json.dumps(str(output / 'permissions.json'))}, {json.dumps(manifest_digest())}, {json.dumps(mode)});\n"
        f"if ((process.env.WARDROBE_APP_MODE ?? 'read_only_demo') !== {json.dumps(mode)}) throw new Error('Application mode mismatch');\n"
    )
    lines = frontend.splitlines()
    indexes = [i for i, line in enumerate(lines) if "Read-only Agent validation prohibits other write requests" in line]
    assert len(indexes) == 1
    lines[indexes[0]] = (
        "  if (url.port==='8001') { const headers=new Headers(input?.headers); "
        "new Headers(init?.headers).forEach((v,k)=>headers.set(k,v)); "
        "_assertApplicationHttp(_applicationPolicy,url,method,headers); }"
    )
    write(output / "frontend-guard.mjs", prefix + "\n".join(lines) + "\n")
    sources = IMPLEMENTATION_FILES
    candidate = {**config, "guard_directory": str(output),
        "guard_hashes": {name: digest(output / name) for name in original},
        "application_policy": {"mode": mode, "manifest": str(output / "permissions.json"),
            "sha256": manifest_digest(), "source_hashes": {str(ROOT / f): digest(ROOT / f) for f in sources}}}
    write(output / "candidate-runtime.json", candidate)
    validate_launch_config(candidate, mode, owner=os.getuid())
    assert original == {name: digest(active / name) for name in original}
    return output / "candidate-runtime.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/wardrowbe-agent/local-runtime.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("read_only_demo", "full_app"), default="read_only_demo")
    args = parser.parse_args()
    print(stage(args.config, args.output, args.mode))


if __name__ == "__main__":
    main()
