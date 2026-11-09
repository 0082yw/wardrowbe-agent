"""Lesson62 foundation checks; no DB/Redis/model/network calls or real images."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image

from app.services import application_access as access
from app.services import image_transactions as effects
from app.utils import rate_limit
from app.api import health
from app.database import get_db


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    monkeypatch.setattr(access, "get_settings", lambda: SimpleNamespace(wardrobe_app_mode="read_only_demo"))
    monkeypatch.setattr(health, "get_settings", lambda: SimpleNamespace(
        effective_ai_text_enabled=False, effective_ai_vision_enabled=False))


async def request(app, method, path, **kwargs):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://offline") as client:
        return await client.request(method, path, **kwargs)


def make_app(failure=False):
    app = FastAPI()
    app.include_router(health.router, prefix="/api/v1")
    async def database():
        yield SimpleNamespace(execute=AsyncMock(side_effect=RuntimeError("PRIVATE-DB-PASSWORD") if failure else None))
    app.dependency_overrides[get_db] = database
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,status", [(False, 200), (True, 503)])
async def test_readiness_and_liveness_are_distinct_and_private(failure, status):
    app = make_app(failure)
    ready = await request(app, "GET", "/api/v1/health/ready")
    live = await request(app, "GET", "/api/v1/health")
    assert ready.status_code == status and live.status_code == 200
    assert "PRIVATE" not in ready.text
    assert ready.json()["checks"]["database"] == ("unhealthy" if failure else "healthy")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,write", [("read_only_demo", False), ("full_app", True)])
async def test_capabilities_match_enforcement(monkeypatch, mode, write):
    monkeypatch.setattr(access, "get_settings", lambda: SimpleNamespace(wardrobe_app_mode=mode))
    app = make_app(); app.add_middleware(access.ApplicationAccessMiddleware)
    calls = []
    @app.post("/api/v1/items")
    async def mutate():
        calls.append(True); return {"ok": True}
    response = await request(app, "GET", "/api/v1/capabilities")
    assert response.json()["application"]["business_write"] is write
    assert response.json()["features"]["external_tagging"] is write
    changed = await request(app, "POST", "/api/v1/items", json={"mode": "full_app"})
    assert changed.status_code == (200 if write else 403)
    assert bool(calls) is write


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/v1/agent/preview", "/api/v1/auth/sync"])
async def test_non_business_routes_keep_their_authentication(path):
    app = make_app(); app.add_middleware(access.ApplicationAccessMiddleware)
    @app.post(path)
    async def protected():
        raise HTTPException(401, "authentication required")
    assert (await request(app, "POST", path)).status_code == 401


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.asyncio
async def test_readonly_denies_before_business_handler(method):
    app=make_app();app.add_middleware(access.ApplicationAccessMiddleware)
    response=await request(app,method,"/api/v1/items/private")
    assert response.status_code==403
    assert response.json()["detail"]["error_code"]=="APP_READ_ONLY"


def fake_redis(monkeypatch, *, count=1, broken=False):
    pipe = Mock()
    pipe.execute = AsyncMock(side_effect=RuntimeError("PRIVATE-SIGNED-URL") if broken else None,
                             return_value=[True, count])
    redis = SimpleNamespace(pipeline=Mock(return_value=pipe), ping=AsyncMock(
        side_effect=RuntimeError("PRIVATE-SIGNED-URL") if broken else None), aclose=AsyncMock())
    monkeypatch.setattr(rate_limit.Redis, "from_url", lambda *a, **kw: redis)
    return redis,pipe


@pytest.mark.asyncio
@pytest.mark.parametrize("count,status", [(1,None),(11,429)])
async def test_normal_rate_window_and_429_unchanged(monkeypatch,count,status):
    redis,pipe=fake_redis(monkeypatch,count=count)
    if status:
        with pytest.raises(HTTPException) as error:await rate_limit.check_rate_limit("test",10,60)
        assert error.value.status_code==status
    else:await rate_limit.check_rate_limit("test",10,60)
    pipe.set.assert_called_once_with("test",0,ex=60,nx=True)
    redis.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry",["limit","sensitive_preflight"])
async def test_redis_fault_fails_closed_without_leaking(monkeypatch,caplog,entry):
    redis,_=fake_redis(monkeypatch,broken=True)
    with pytest.raises(HTTPException) as error:
        if entry=="limit":await rate_limit.check_rate_limit("PRIVATE-key",10,60)
        else:await rate_limit.require_rate_limit_store()
    assert error.value.status_code==503
    assert "PRIVATE" not in str(error.value.detail)+caplog.text
    redis.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,commit,cleanup",[
    ("created",True,False),("created",False,True),
    ("deleted",True,True),("deleted",False,False),
])
async def test_image_effects_follow_transaction(kind,commit,cleanup):
    db=AsyncSession()
    service=SimpleNamespace(delete_images=Mock())
    getattr(effects,kind+"_images")(db,service,{"image_path":"owned/new.jpg"})
    service.delete_images.assert_not_called()
    if commit:await db.commit()
    else:await db.rollback()
    assert service.delete_images.called is cleanup
    await db.close()


@pytest.mark.asyncio
async def test_committed_new_file_is_not_deleted_by_a_later_failure():
    db=AsyncSession();service=SimpleNamespace(delete_images=Mock())
    effects.created_images(db,service,{"image_path":"owned/new.jpg"})
    await db.commit()
    await db.begin();await db.rollback()
    service.delete_images.assert_not_called()
    await db.close()


@pytest.mark.asyncio
async def test_failed_commit_is_compensated_by_get_db(monkeypatch):
    import app.database as database
    db=AsyncSession()
    service=SimpleNamespace(delete_images=Mock())
    effects.created_images(db,service,{"image_path":"owned/new.jpg"})
    monkeypatch.setattr(db,"commit",AsyncMock(side_effect=RuntimeError("commit failed")))
    class Maker:
        async def __aenter__(self):return db
        async def __aexit__(self,*args):await db.close()
    monkeypatch.setattr(database,"async_session_maker",lambda:Maker())
    generator=database.get_db()
    await anext(generator)
    with pytest.raises(RuntimeError):await anext(generator)
    service.delete_images.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_failure_never_turns_committed_delete_into_false_500(caplog):
    db=AsyncSession()
    effects.deleted_images(db,SimpleNamespace(delete_images=Mock(side_effect=OSError("PRIVATE path"))),
                           {"image_path":"owned/new.jpg"})
    await db.commit()
    assert "image_compensation_cleanup_failed" in caplog.text and "PRIVATE" not in caplog.text
    await db.close()


@pytest.mark.asyncio
async def test_partial_image_encoding_cleans_only_new_files(tmp_path,monkeypatch):
    from app.services.image_service import ImageService
    service=ImageService.__new__(ImageService);service.storage_path=tmp_path
    monkeypatch.setattr(service,"_open_bounded",lambda *a:Image.new("RGB",(2,2)))
    monkeypatch.setattr(service,"_generate_filename",lambda *a:"new.jpg")
    monkeypatch.setattr(service,"encode_variant",Mock(side_effect=[b"ok",RuntimeError("encoding failed")]))
    user=uuid4()
    with pytest.raises(RuntimeError):await service.process_and_store(user,b"mock","photo.png")
    assert list((tmp_path/str(user)).iterdir())==[]


@pytest.mark.asyncio
async def test_filename_collision_does_not_overwrite_existing_image(tmp_path,monkeypatch):
    from app.services.image_service import ImageService
    service=ImageService.__new__(ImageService);service.storage_path=tmp_path
    user=uuid4();directory=tmp_path/str(user);directory.mkdir()
    target=directory/"existing.jpg";target.write_bytes(b"existing")
    monkeypatch.setattr(service,"_open_bounded",lambda *a:Image.new("RGB",(2,2)))
    monkeypatch.setattr(service,"_generate_filename",lambda *a:"existing.jpg")
    monkeypatch.setattr(service,"encode_variant",lambda *a:b"new")
    with pytest.raises(FileExistsError):await service.process_and_store(user,b"mock","photo.png")
    assert target.read_bytes()==b"existing"


def test_unsafe_image_cleanup_path_is_rejected(tmp_path):
    from app.services.image_service import ImageService
    service=ImageService.__new__(ImageService);service.storage_path=tmp_path
    with pytest.raises(ValueError):service.delete_images({"image_path":"../private.jpg"})


@pytest.mark.asyncio
async def test_sensitive_preflight_does_not_consume_a_successful_quota(monkeypatch):
    redis,pipe=fake_redis(monkeypatch)
    redis.ping.return_value=True
    await rate_limit.require_rate_limit_store()
    redis.ping.assert_awaited_once()
    pipe.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_redis_failure_precedes_file_processing(monkeypatch):
    from app.api import items
    from app.utils.auth import get_current_user
    fake_redis(monkeypatch,broken=True)
    service=Mock(side_effect=AssertionError("No file work before protection"))
    monkeypatch.setattr(items,"ImageService",service)
    app=make_app();app.include_router(items.router,prefix="/api/v1")
    app.dependency_overrides[get_current_user]=lambda:SimpleNamespace(id=uuid4())
    response=await request(app,"POST","/api/v1/items",files={"image":("test.jpg",b"synthetic","image/jpeg")})
    assert response.status_code==503
    service.assert_not_called()


@pytest.mark.asyncio
async def test_auth_sync_redis_failure_precedes_user_sync(monkeypatch):
    from app.api import auth
    fake_redis(monkeypatch,broken=True)
    service=Mock(side_effect=AssertionError("No user mutation before protection"))
    monkeypatch.setattr(auth,"UserService",service)
    app=make_app();app.include_router(auth.router,prefix="/api/v1")
    response=await request(app,"POST","/api/v1/auth/sync",json={
        "external_id":"isolated","email":"test@example.com","display_name":"Test"})
    assert response.status_code==503
    service.assert_not_called()


def test_manager_cannot_use_full_mode_to_bypass_private_protection(monkeypatch):
    from scripts.local_agent import Manager,RuntimeErrorSafe
    monkeypatch.setenv("WARDROBE_APP_MODE","full_app")
    with pytest.raises(RuntimeErrorSafe):
        Manager.__new__(Manager).start()


def test_unknown_mode_is_not_a_full_mode_default():
    from app.config import Settings
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(wardrobe_app_mode="invalid")
