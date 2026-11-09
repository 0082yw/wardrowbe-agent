"""Lesson67 only: tags, lifecycle, signed variants, two-account isolation."""
import asyncio
from io import BytesIO
import json
import os
from pathlib import Path
from uuid import uuid4
import httpx
from PIL import Image, ImageDraw
from sqlalchemy import delete
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

def test_isolated_tags_lifecycle_images_and_cross_account(monkeypatch):
    assert os.environ.get("WARDROBE_L67_ISOLATED") == "1"
    asyncio.run(scenario(monkeypatch))

async def scenario(monkeypatch):
    from fastapi import FastAPI
    from app.api import items, images
    from app.api.auth import create_access_token
    from app.config import get_settings
    from app.database import engine
    from app.models.user import User
    from app.models.identity import UserIdentity
    from app.models.item import ClothingItem, ItemStatus
    from app.services.application_access import ApplicationAccessMiddleware
    from app.services.image_service import ImageService
    from app.write_guard import install_write_guard
    from redis.asyncio import Redis
    u=make_url(os.environ["TEST_DATABASE_URL"]);r=make_url(os.environ["TEST_REDIS_URL"])
    assert (u.host,u.port,u.database)==("127.0.0.1",15432,"wardrowbe_browser_test")
    assert (r.host,r.port,r.database)==("127.0.0.1",16379,"1")
    settings=get_settings()
    assert str(settings.database_url)==os.environ["TEST_DATABASE_URL"]
    assert settings.redis_url==os.environ["TEST_REDIS_URL"]
    assert Path(settings.storage_path).name.startswith("images-67-")
    redis=Redis.from_url(settings.redis_url,socket_connect_timeout=3,socket_timeout=3)
    assert await redis.ping()
    await redis.aclose()
    monkeypatch.setattr(settings,"wardrobe_app_mode","full_app")
    seed_engine=create_async_engine(os.environ["TEST_DATABASE_URL"])
    seed=async_sessionmaker(seed_engine,expire_on_commit=False)
    a,b,ia,ib,item_id=[uuid4() for _ in range(5)]
    paths={}
    service=ImageService()
    try:
        picture=Image.new("RGB",(1800,1200),"white")
        ImageDraw.Draw(picture).rectangle((100,100,900,800),fill="blue")
        raw=BytesIO();picture.save(raw,"JPEG")
        paths=await service.process_and_store(a,raw.getvalue(),"lesson67.jpg")
        async with seed() as db:
            db.add_all([User(id=a,email=f"l67-{a}@example.com",display_name="L67 A"),
                        User(id=b,email=f"l67-{b}@example.com",display_name="L67 B")])
            await db.flush()
            db.add_all([UserIdentity(id=ia,user_id=a,provider="dev",subject=str(a)),
                        UserIdentity(id=ib,user_id=b,provider="dev",subject=str(b)),
                        ClothingItem(id=item_id,user_id=a,type="shirt",status=ItemStatus.ready,**paths)])
            await db.commit()
        install_write_guard(engine)
        app=FastAPI();app.add_middleware(ApplicationAccessMiddleware)
        app.include_router(items.router,prefix="/api/v1");app.include_router(images.router,prefix="/api/v1")
        ha={"Authorization":"Bearer "+create_access_token(a,ia)}
        hb={"Authorization":"Bearer "+create_access_token(b,ib)}
        target=f"/api/v1/items/{item_id}"
        checks=[]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://isolated.test") as client:
            for payload,wanted in [({"size":"M"},"M"),({"size":"L"},"L"),({"size":None},None)]:
                res=await client.patch(target,headers=ha,json={"tags":payload})
                assert res.status_code==200
                assert res.json()["tags"].get("size")==wanted
            checks.extend(["tag_add","tag_edit","tag_remove"])
            res=await client.post(target+"/archive",headers=ha,json={})
            assert res.status_code==200 and res.json()["status"]=="archived" and res.json()["is_archived"]
            res=await client.post(target+"/restore",headers={**ha,"Content-Type":"application/json"})
            assert res.status_code==200 and res.json()["status"]=="ready" and not res.json()["is_archived"]
            item=res.json();checks.extend(["archive","restore_empty_json_header"])
            for field in ["image_url","medium_url","thumbnail_url"]:
                res=await client.get(item[field])
                assert res.status_code==200 and res.headers["Content-Type"].startswith("image/")
                with Image.open(BytesIO(res.content)) as decoded:assert decoded.width>0 and decoded.height>0
            checks.extend(["signed_original_200","signed_medium_200","signed_thumbnail_200"])
            # A signed URL is a short-lived bearer capability, not a user-session cookie.
            # B must not obtain A's item or a signed URL through item APIs.
            for method,payload in [("GET",None),("PATCH",{"tags":{"size":"XL"}}),("DELETE",None)]:
                res=await client.request(method,target,headers=hb,json=payload)
                assert res.status_code==404
            res=await client.post(target+"/archive",headers=hb,json={})
            assert res.status_code==404
            res=await client.get("/api/v1/images/"+paths["image_path"],headers=hb)
            assert res.status_code==401
            signed=httpx.URL(item["image_url"])
            bad=signed.copy_set_param("sig","0"*32)
            assert (await client.get(bad)).status_code==401
            changed_path=str(signed).replace(str(a),str(b))
            assert (await client.get(changed_path)).status_code==401
            checks.extend(["cross_user_item_read_update_delete_archive_denied",
                           "cross_user_unsigned_image_denied","tampered_signature_and_owner_denied"])
            res=await client.get(target,headers=ha)
            assert res.status_code==200 and "size" not in res.json()["tags"] and res.json()["status"]=="ready"
        Path(os.environ["WARDROBE_L67_REPORT"]).write_text(json.dumps({"checks":checks,"count":len(checks)},indent=2))
    finally:
        async with seed() as db:
            await db.execute(delete(User).where(User.id.in_([a,b])));await db.commit()
        if paths:service.delete_images(paths)
        await engine.dispose();await seed_engine.dispose()
