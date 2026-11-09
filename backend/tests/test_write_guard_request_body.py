"""Regression for the real client's JSON header + bodyless mutation combination."""
import asyncio
from types import SimpleNamespace
from uuid import uuid4
import json

import httpx
import pytest
from fastapi import Depends, FastAPI, Request
from app.config import get_settings
from app.schemas.item import ArchiveRequest
from app.write_guard import bind_authenticated_request, KEY

ID, ACTOR = uuid4(), uuid4()

async def exercise(operation, content, headers=None):
    db = SimpleNamespace(sync_session=SimpleNamespace(info={}))
    app = FastAPI()
    executed = []
    async def guard(request: Request):
        await bind_authenticated_request(db, request, SimpleNamespace(id=ACTOR))
    @app.post("/api/v1/items/{item_id}/rotate", dependencies=[Depends(guard)])
    @app.post("/api/v1/items/{item_id}/restore", dependencies=[Depends(guard)])
    async def action(item_id: str, request: Request):
        executed.append(item_id)
        return {"operation": db.sync_session.info[KEY].operation,
                "body": (await request.body()).decode()}
    @app.post("/api/v1/items/{item_id}/archive", dependencies=[Depends(guard)])
    async def archive(item_id: str, payload: ArchiveRequest):
        executed.append(item_id)
        return {"accepted": True}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://isolated.test") as client:
        response = await client.post(f"/api/v1/items/{ID}/{operation}", content=content,
                                     headers=headers or {"Content-Type": "application/json"})
    return response, executed

@pytest.fixture(autouse=True)
def full_policy(monkeypatch):
    monkeypatch.setattr(get_settings(), "wardrobe_app_mode", "full_app")

@pytest.mark.parametrize("operation", ["rotate", "restore"])
@pytest.mark.parametrize("content", [b"", b"{}"])
def test_client_bodyless_json_header_is_supported(operation, content):
    response, executed = asyncio.run(exercise(operation, content))
    assert response.status_code == 200
    assert response.json() == {"operation": operation, "body": content.decode()}
    assert executed == [str(ID)]

@pytest.mark.parametrize("operation", ["rotate", "restore"])
def test_empty_body_with_charset_header(operation):
    response, executed = asyncio.run(exercise(operation, b"", {"Content-Type": "application/json; charset=utf-8"}))
    assert response.status_code == 200
    assert executed == [str(ID)]

@pytest.mark.parametrize("operation", ["rotate", "restore"])
@pytest.mark.parametrize("content", [b" ", b"{", b"\xff", b"null", b"[]", b'"text"'])
def test_invalid_nonempty_or_nonobject_body_rejected(operation, content):
    response, executed = asyncio.run(exercise(operation, content))
    assert response.status_code == 422
    assert executed == []

@pytest.mark.parametrize("field", ["id","user_id","owner_id","family_id","mode","model_name","endpoint"])
def test_identity_and_mode_forgery_still_rejected(field):
    response, executed = asyncio.run(exercise("rotate", json.dumps({field: "DO_NOT_ECHO_PRIVATE_INPUT"}).encode()))
    assert response.status_code == 422
    assert executed == []
    assert "DO_NOT_ECHO_PRIVATE_INPUT" not in response.text

def test_required_archive_schema_is_not_relaxed():
    response, executed = asyncio.run(exercise("archive", b""))
    assert response.status_code == 422
    assert executed == []

def test_archive_object_and_cached_body_still_work():
    response, executed = asyncio.run(exercise("archive", b"{}"))
    assert response.status_code == 200
    assert executed == [str(ID)]
