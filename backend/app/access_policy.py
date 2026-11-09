"""One trusted policy for launch, HTTP adapters and scoped wardrobe SQL."""
from dataclasses import dataclass
from typing import Literal
from uuid import UUID
import hashlib
import json
import re

Mode = Literal["read_only_demo", "full_app"]
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
NON_BUSINESS_POSTS = frozenset({"/api/v1/auth/sync", "/api/v1/agent/preview"})
UUID_PATTERN = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
RULES = (
    ("POST", r"^/api/v1/items$", "create"),
    ("PATCH", rf"^/api/v1/items/({UUID_PATTERN})$", "update"),
    ("DELETE", rf"^/api/v1/items/({UUID_PATTERN})$", "delete"),
    ("POST", rf"^/api/v1/items/({UUID_PATTERN})/archive$", "archive"),
    ("POST", rf"^/api/v1/items/({UUID_PATTERN})/restore$", "restore"),
    ("POST", rf"^/api/v1/items/({UUID_PATTERN})/rotate$", "rotate"),
)


IMPLEMENTATION_FILES = (
    "backend/app/access_policy.py", "backend/app/write_guard.py",
    "backend/app/utils/auth.py", "backend/app/services/application_access.py",
    "backend/app/services/item_service.py", "backend/app/api/items.py",
    "backend/app/workers/authorized_rotation.py", "frontend/scripts/application-policy.mjs",
    "backend/scripts/local_agent.py", "backend/app/services/image_transactions.py",
    "backend/scripts/rotation_worker.py",
    "frontend/app/api/v1/[...path]/route.ts",
    "frontend/lib/application-access-context.ts",
    "frontend/components/application-access-provider.tsx",
    "frontend/components/ui/button.tsx",
)
GUARD_FILES = frozenset({"backend.py", "backend-entry.py", "auth-state.json", "frontend-guard.mjs"})


@dataclass(frozen=True)
class WardrobeAccessPolicy:
    mode: Mode = "read_only_demo"

    def __post_init__(self):
        if self.mode not in ("read_only_demo", "full_app"):
            raise ValueError("Invalid application mode")

    def operation(self, method: str, path: str) -> tuple[str, UUID | None] | None:
        for verb, pattern, operation in RULES:
            match = re.fullmatch(pattern, path)
            if method.upper() == verb and match:
                return operation, UUID(match[1]) if match.lastindex else None
        return None

    def allows_http(self, method: str, path: str) -> bool:
        method = method.upper()
        return (
            method in READ_METHODS
            or (method == "POST" and path in NON_BUSINESS_POSTS)
            or (self.mode == "full_app" and self.operation(method, path) is not None)
        )

    def permits_write(self, operation: str) -> bool:
        return self.mode == "full_app" and operation in {rule[2] for rule in RULES}


def policy_manifest() -> dict:
    return {
        "version": 1,
        "default_mode": "read_only_demo",
        "modes": ["read_only_demo", "full_app"],
        "read_methods": sorted(READ_METHODS),
        "non_business_posts": sorted(NON_BUSINESS_POSTS),
        "write_routes": [
            {"method": method, "pattern": pattern, "operation": operation}
            for method, pattern, operation in RULES
        ],
        "sql_tables": ["clothing_items"],
        "worker_functions": ["authorized_rotate_item_image_job"],
    }


def manifest_bytes() -> bytes:
    return (json.dumps(policy_manifest(), sort_keys=True, separators=(",", ":")) + "\n").encode()


def manifest_digest() -> str:
    return hashlib.sha256(manifest_bytes()).hexdigest()


def validate_manifest(data: dict) -> None:
    if data != policy_manifest():
        raise ValueError("Unrecognized application permission manifest")


def validate_launch_config(config: dict, mode: str, *, owner: int) -> WardrobeAccessPolicy:
    """Fail closed unless full_app has a separately staged and pinned adapter bundle."""
    from pathlib import Path
    import stat

    policy = WardrobeAccessPolicy(mode)
    bundle = config.get("application_policy")
    if bundle is None:
        if policy.mode != "read_only_demo":
            raise ValueError("full_app requires a validated permission bundle")
        return policy
    if bundle.get("mode") != mode:
        raise ValueError("Permission bundle mode mismatch")
    path = Path(bundle["manifest"])
    details = path.lstat()
    if path.is_symlink() or details.st_uid != owner or stat.S_IMODE(details.st_mode) & 0o077:
        raise ValueError("Permission manifest must be private and owned")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != bundle["sha256"] or data != manifest_bytes():
        raise ValueError("Permission manifest integrity failure")
    root = Path(__file__).resolve().parents[2]
    if set(bundle["source_hashes"]) != {str(root / name) for name in IMPLEMENTATION_FILES}:
        raise ValueError("Incomplete permission implementation pins")
    if set(config["guard_hashes"]) != GUARD_FILES:
        raise ValueError("Incomplete permission adapter pins")
    for name, expected in bundle["source_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Permission implementation changed")
    directory = Path(config["guard_directory"])
    if directory.is_symlink() or directory.stat().st_uid != owner or directory.stat().st_mode & 0o077:
        raise ValueError("Unsafe adapter directory")
    for name, expected in config["guard_hashes"].items():
        source = directory / name
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Permission adapter changed")
    return policy
