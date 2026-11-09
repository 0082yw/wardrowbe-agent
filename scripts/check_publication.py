"""Dependency-free static checks for the sanitized publication snapshot."""
from pathlib import Path
import ast
import json
import re

root = Path(__file__).resolve().parents[1]
count = {"python": 0, "json": 0, "files": 0}
for path in root.rglob("*"):
    if not path.is_file() or any(p in {".git", "node_modules", ".venv", ".next", ".cache", "__pycache__"}
                                 for p in path.relative_to(root).parts):
        continue
    relative = path.relative_to(root)
    assert not (path.name.startswith(".env") and path.name != ".env.example"), relative
    assert relative.as_posix() != "k8s/secrets.yaml", relative
    assert path.stat().st_size < 50 * 1024 * 1024, relative
    count["files"] += 1
    if path.suffix == ".py":
        ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        count["python"] += 1
    elif path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
        count["json"] += 1
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    assert not re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", text), relative
    assert not re.search(r"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}", text), relative
assert (root/"LICENSE").is_file()
assert "Anyesh/wardrowbe" in (root/"README.md").read_text()
print(json.dumps({"result": "passed", **count}))
