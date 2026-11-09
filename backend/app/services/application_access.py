"""Trusted process configuration; application capabilities never grant user permissions."""
from starlette.responses import JSONResponse
from app.config import get_settings
from app.access_policy import WardrobeAccessPolicy, RULES



def application_access() -> dict:
    mode = get_settings().wardrobe_app_mode
    return {
        "mode": mode,
        "business_write": mode == "full_app",
        "agent_preview": True,
        "write_operations": sorted({row[2] for row in RULES}) if mode == "full_app" else [],
        "reason_code": "APP_READ_ONLY" if mode == "read_only_demo" else None,
    }


class ApplicationAccessMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path, method = scope.get("path", ""), scope.get("method", "GET")
            policy = WardrobeAccessPolicy(get_settings().wardrobe_app_mode)
            allowed = policy.allows_http(method, path)
            if path.startswith("/api/v1/") and not allowed:
                response = JSONResponse(status_code=403, content={"detail": {
                    "error_code": "APP_READ_ONLY" if policy.mode == "read_only_demo" else "APP_OPERATION_FORBIDDEN",
                    "message": "This operation is not permitted by the trusted application policy.",
                }})
                return await response(scope, receive, send)
        # Authentication sync and Agent preview retain their existing authentication,
        # identity, rate-limit and SQL guards. This middleware never authorizes a user.
        await self.app(scope, receive, send)
