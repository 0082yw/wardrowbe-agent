"""Opt-in real PostgreSQL + Redis tests. Never run against the development DB."""
import asyncio
from datetime import date
from io import BytesIO
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from PIL import Image
import pytest
from sqlalchemy import delete, event, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

def test_real_isolated_wardrobe_permissions_and_compensation(monkeypatch):
    if os.environ.get("WARDROBE_POLICY_ISOLATED_TEST") != "1":
        pytest.skip("Explicit isolated test runner required")
    asyncio.run(scenario(monkeypatch))

async def scenario(monkeypatch):
    from fastapi import FastAPI
    from app.access_policy import WardrobeAccessPolicy
    from app.api import items, images
    from app.api.auth import create_access_token
    from app.config import get_settings
    from app.database import engine, async_session_maker
    from app.models.user import User
    from app.models.identity import UserIdentity
    from app.models.item import ClothingItem, ItemHistory, ItemStatus
    from app.services.application_access import ApplicationAccessMiddleware
    from app.services.image_service import ImageService
    from app.workers.authorized_rotation import (
        rotation_receipt, verify_rotation, authorized_rotate_item_image_job,
        WardrobeRotationWorkerSettings,
    )
    from app.write_guard import bind_session, install_write_guard

    settings = get_settings()
    url = make_url(os.environ["TEST_DATABASE_URL"])
    assert (url.host, url.port, url.database) == ("127.0.0.1", 15432, "wardrowbe_browser_test")
    redis_url = make_url(os.environ["TEST_REDIS_URL"])
    assert (redis_url.host, redis_url.port, redis_url.database) == ("127.0.0.1", 16379, "1")
    assert str(settings.database_url) == os.environ["TEST_DATABASE_URL"]
    assert settings.redis_url == os.environ["TEST_REDIS_URL"]
    storage = Path(settings.storage_path)
    assert storage.name.startswith("images-65-") and storage.is_dir()
    assert not settings.effective_ai_vision_enabled
    assert not settings.effective_ai_text_enabled
    seed_engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    seed = async_sessionmaker(seed_engine, expire_on_commit=False)
    a, b, ia, ib, protected = [uuid4() for _ in range(5)]
    marker = "permission-" + uuid4().hex
    checks = []
    def passed(name):
        checks.append(name)
        Path(os.environ["WARDROBE_POLICY_TEST_REPORT"]).write_text(json.dumps({"checks": checks, "count": len(checks), "isolated": True}, indent=2))
    install_write_guard(engine)
    app = FastAPI()
    app.add_middleware(ApplicationAccessMiddleware)
    app.include_router(items.router, prefix="/api/v1")
    app.include_router(images.router, prefix="/api/v1")
    image = Image.new("RGB", (73, 101), "white")
    for x in range(20):
        for y in range(40):
            image.putpixel((x, y), (x * 9, y * 5, 100))
    stream = BytesIO()
    image.save(stream, "JPEG")
    files = {"image": ("permission.jpg", stream.getvalue(), "image/jpeg")}
    payload = {"type": "shirt", "name": marker, "skip_ai": "true"}
    try:
        async with seed() as db:
            # Only this run's unique users; existing manual A/B accounts remain untouched.
            db.add_all([
                User(id=a, email=f"{marker}-a@example.com", display_name="Policy A", email_verified=True),
                User(id=b, email=f"{marker}-b@example.com", display_name="Policy B", email_verified=True),
            ])
            await db.flush()
            db.add_all([
                UserIdentity(id=ia, user_id=a, provider="dev", subject=f"{marker}-a"),
                UserIdentity(id=ib, user_id=b, provider="dev", subject=f"{marker}-b"),
                ClothingItem(id=protected, user_id=b, type="shirt", status=ItemStatus.ready,
                             image_path=f"{b}/protected.jpg", name=marker+"-protected"),
            ])
            await db.commit()
        ha = {"Authorization": "Bearer " + create_access_token(a, ia)}
        hb = {"Authorization": "Bearer " + create_access_token(b, ib)}
        monkeypatch.setattr(settings, "wardrobe_app_mode", "full_app")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://isolated.test",
        ) as client:
            r = await client.post("/api/v1/items", data=payload, files=files)
            assert r.status_code == 401
            passed("unauthenticated_write_denied")
            r = await client.post("/api/v1/items", headers=ha, data=payload, files=files)
            assert r.status_code == 201, f"create status {r.status_code}"
            item = r.json()
            item_id = item["id"]
            target = "/api/v1/items/" + item_id
            assert item["user_id"] == str(a) and item["status"] == "ready"
            before = {p.relative_to(storage) for p in storage.rglob("*") if p.is_file()}
            assert len(before) == 3
            assert all(p.parts[0] == str(a) for p in before)
            passed("authenticated_create_three_owned_images")
            for field in ("image_url", "medium_url", "thumbnail_url"):
                image_r = await client.get(item[field])
                assert image_r.status_code == 200
            passed("three_signed_image_variants_accessible")
            r = await client.get(target, headers=ha)
            assert r.status_code == 200 and r.json()["id"] == item_id
            passed("owner_read")
            for method, body in (("GET", None), ("PATCH", {"name": "intrusion"}), ("DELETE", None)):
                r = await client.request(method, target, headers=hb, json=body)
                assert r.status_code == 404
            r = await client.get("/api/v1/images/" + item["image_path"], headers=hb)
            assert r.status_code == 401
            passed("cross_user_read_update_delete_image_denied")
            r = await client.post("/api/v1/items", headers=ha, data={**payload, "user_id": str(b)}, files=files)
            assert r.status_code == 422
            r = await client.patch(target, headers=ha, json={"name": "x", "mode": "full_app"})
            assert r.status_code == 422
            passed("forged_identity_and_browser_mode_denied")
            for path in ("/api/v1/outfits", target+"/remove-background", "/api/v1/items/bulk"):
                r = await client.post(path, headers=ha, json={})
                assert r.status_code == 403
            passed("unrelated_http_and_unverified_jobs_denied")
            monkeypatch.setattr(settings, "wardrobe_app_mode", "read_only_demo")
            r = await client.patch(target, headers=ha, json={"name": "x"})
            assert r.status_code == 403
            assert {p.relative_to(storage) for p in storage.rglob("*") if p.is_file()} == before
            passed("read_only_mode_still_denies_write")
            monkeypatch.setattr(settings, "wardrobe_app_mode", "full_app")
            r = await client.patch(target, headers=ha, json={
                "name": marker+"-edited", "type": "jeans",
                "tags": {"colors": ["blue"], "primary_color": "blue", "size": "M"},
            })
            assert r.status_code == 200
            assert r.json()["type"] == "jeans" and r.json()["tags"]["size"] == "M"
            passed("category_and_tag_update")
            r = await client.post(target+"/archive", headers=ha, json={})
            assert r.status_code == 200 and r.json()["status"] == "archived"
            r = await client.post(target+"/restore", headers=ha)
            assert r.status_code == 200 and r.json()["status"] == "ready"
            passed("archive_restore_state")
            r = await client.post("/api/v1/items", headers=ha, data=payload,
                                  files={"image": ("bad.jpg", b"invalid", "image/jpeg")})
            assert r.status_code == 400
            assert {p.relative_to(storage) for p in storage.rglob("*") if p.is_file()} == before
            passed("invalid_upload_no_file_or_row")
            # A real PG unique-constraint failure after new image files were saved.
            def duplicate_id(mapper, connection, obj):
                if obj.name == marker+"-db-failure":
                    obj.id = protected
            event.listen(ClothingItem, "before_insert", duplicate_id)
            try:
                # Avoid the duplicate-image fast path, not any permission check.
                with monkeypatch.context() as m:
                    m.setattr(ImageService, "compute_phash", lambda *args: uuid4().hex[:16])
                    r = await client.post("/api/v1/items", headers=ha,
                                          data={**payload, "name": marker+"-db-failure"}, files=files)
                assert r.status_code == 500
            finally:
                event.remove(ClothingItem, "before_insert", duplicate_id)
            assert {p.relative_to(storage) for p in storage.rglob("*") if p.is_file()} == before
            passed("real_database_failure_cleans_only_new_images")
            original_encode = ImageService.encode_variant
            count = 0
            def broken_encode(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise OSError("isolated injected image encoder failure")
                return original_encode(*args, **kwargs)
            with monkeypatch.context() as m:
                m.setattr(ImageService, "compute_phash", lambda *args: uuid4().hex[:16])
                m.setattr(ImageService, "encode_variant", broken_encode)
                r = await client.post("/api/v1/items", headers=ha, data=payload, files=files)
                assert r.status_code == 500
            assert {p.relative_to(storage) for p in storage.rglob("*") if p.is_file()} == before
            passed("partial_image_upload_compensated")

            full = WardrobeAccessPolicy("full_app")
            async def denied(change):
                async with async_session_maker() as db:
                    bind_session(db, full, a, "PATCH", target)
                    with pytest.raises(PermissionError):
                        await change(db)
                    await db.rollback()
            async def user_mutation(db):
                user = await db.get(User, a)
                user.display_name = "forbidden"
                await db.flush()
            await denied(user_mutation)
            async def foreign_item(db):
                obj = await db.get(ClothingItem, protected)
                obj.name = "forbidden"
                await db.flush()
            await denied(foreign_item)
            async def ownership(db):
                obj = await db.get(ClothingItem, UUID(item_id))
                obj.user_id = b
                await db.flush()
            await denied(ownership)
            async def paths(db):
                obj = await db.get(ClothingItem, UUID(item_id))
                obj.image_path = f"{b}/protected.jpg"
                await db.flush()
            await denied(paths)
            async def core(db):
                await db.execute(update(ClothingItem).where(ClothingItem.id == UUID(item_id)).values(name="bad"))
            await denied(core)
            async def raw(db):
                await db.execute(text("UPDATE users SET display_name='forbidden' WHERE id=:id"), {"id": a})
            await denied(raw)
            async with async_session_maker() as db:
                with pytest.raises(PermissionError):
                    await db.execute(text("UPDATE clothing_items SET name='forbidden' WHERE id=:id"), {"id": a})
                await db.rollback()
            passed("sql_denies_unrelated_cross_owner_image_reassignment_raw_bulk_unscoped")
            async with async_session_maker() as db:
                bind_session(db, full, a, "PATCH", target)
                with pytest.raises(PermissionError):
                    bind_session(db, full, b, "PATCH", target)
            passed("session_actor_cannot_change")
            # Queue failure must restore state, without invoking an unverified worker.
            async def queue_failure(*args, **kwargs):
                raise ConnectionError("isolated queue unavailable")
            with monkeypatch.context() as m:
                m.setattr(items, "create_pool", queue_failure)
                r = await client.post(target+"/rotate", headers=ha)
                assert r.status_code == 500
            async with seed() as db:
                obj = await db.get(ClothingItem, UUID(item_id))
                assert obj.status == ItemStatus.ready and obj.processing_kind is None and obj.ai_job_id is None
            passed("rotation_enqueue_failure_restores_state")

            # Actual signed worker execution, restricted SQL, actual files; no AI/jobs besides rotation.
            job_id = uuid4().hex
            receipt = rotation_receipt(a, item_id, "cw", job_id)
            assert verify_rotation(receipt, item_id, "cw", job_id) == a
            with pytest.raises(PermissionError):
                verify_rotation(receipt, item_id, "ccw", job_id)
            with pytest.raises(PermissionError):
                verify_rotation(receipt, item_id, "cw", "different-job")
            assert [f.__name__ for f in WardrobeRotationWorkerSettings.functions] == ["authorized_rotate_item_image_job"]
            assert WardrobeRotationWorkerSettings.cron_jobs == []
            assert WardrobeRotationWorkerSettings.max_tries == 1
            async with seed() as db:
                obj = await db.get(ClothingItem, UUID(item_id))
                obj.status, obj.processing_kind, obj.ai_job_id = ItemStatus.processing, "rotate", job_id
                await db.commit()
            worker_ctx = {"db_session_factory": async_session_maker, "job_id": job_id}
            result = await authorized_rotate_item_image_job(worker_ctx, item_id, "cw", receipt)
            assert result["status"] == "success"
            with pytest.raises(PermissionError):
                await authorized_rotate_item_image_job(worker_ctx, item_id, "cw", receipt)
            passed("signed_rotation_only_worker_rejects_tamper_and_replay")

            # Linked records cannot be cascaded into unrelated tables by item-delete permission.
            history_id = uuid4()
            async with seed() as db:
                db.add(ItemHistory(id=history_id, item_id=UUID(item_id), worn_at=date.today()))
                await db.commit()
            r = await client.delete(target, headers=ha)
            assert r.status_code == 409
            assert all((storage / p).is_file() for p in before)
            async with seed() as db:
                await db.execute(delete(ItemHistory).where(ItemHistory.id == history_id))
                await db.commit()
            passed("linked_delete_refused_without_cascade")
            r = await client.delete(target, headers=ha)
            assert r.status_code == 204
            assert not any(p.is_file() for p in storage.rglob("*"))
            async with seed() as db:
                assert await db.get(ClothingItem, UUID(item_id)) is None
                obj = await db.get(ClothingItem, protected)
                assert obj.name == marker+"-protected" and obj.user_id == b
                assert (await db.get(User, a)).display_name == "Policy A"
            passed("owner_delete_files_and_row_without_other_mutation")
        report = Path(os.environ["WARDROBE_POLICY_TEST_REPORT"])
        report.write_text(json.dumps({"checks": checks, "count": len(checks), "isolated": True}, indent=2))
    finally:
        # Cleanup is scoped to this run's two fresh UUIDs, never global table/Redis cleanup.
        async with seed() as db:
            await db.execute(delete(User).where(User.id.in_([a, b])))
            await db.commit()
        await engine.dispose()
        await seed_engine.dispose()


def test_real_isolated_rotation_api_queue_and_worker(monkeypatch):
    if os.environ.get("WARDROBE_POLICY_ISOLATED_TEST") != "1":
        pytest.skip("Explicit isolated test runner required")
    asyncio.run(queue_scenario(monkeypatch))

async def queue_scenario(monkeypatch):
    from arq import create_pool
    from arq.worker import Worker
    from fastapi import FastAPI
    from app.api import items
    from app.api.auth import create_access_token
    from app.config import get_settings
    from app.database import engine, async_session_maker
    from app.models.user import User
    from app.models.identity import UserIdentity
    from app.models.item import ClothingItem, ItemStatus
    from app.services.application_access import ApplicationAccessMiddleware
    from app.services.image_service import ImageService
    from app.workers.authorized_rotation import authorized_rotate_item_image_job
    from app.workers.settings import get_redis_settings
    from app.write_guard import install_write_guard

    assert make_url(os.environ["TEST_DATABASE_URL"]).port == 15432
    assert make_url(os.environ["TEST_REDIS_URL"]).port == 16379
    settings = get_settings()
    monkeypatch.setattr(settings, "wardrobe_app_mode", "full_app")
    owner, identity, item_id = uuid4(), uuid4(), uuid4()
    queue = "arq:policy65:" + uuid4().hex
    monkeypatch.setattr(items, "IMAGE_QUEUE", queue)
    seed_engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    seed = async_sessionmaker(seed_engine, expire_on_commit=False)
    redis = None
    files = {}
    try:
        picture = BytesIO()
        Image.new("RGB", (45, 93), "blue").save(picture, "JPEG")
        service = ImageService()
        files = await service.process_and_store(owner, picture.getvalue(), "queue.jpg")
        async with seed() as db:
            db.add(User(id=owner, email=f"policy65-{owner}@example.com", display_name="Queue test"))
            await db.flush()
            db.add(UserIdentity(id=identity, user_id=owner, provider="dev", subject=f"policy65-{owner}"))
            db.add(ClothingItem(id=item_id, user_id=owner, type="jeans", status=ItemStatus.ready,
                                **files))
            await db.commit()
        install_write_guard(engine)
        app = FastAPI()
        app.add_middleware(ApplicationAccessMiddleware)
        app.include_router(items.router, prefix="/api/v1")
        headers = {"Authorization": "Bearer " + create_access_token(owner, identity)}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://isolated.test") as client:
            r = await client.post(f"/api/v1/items/{item_id}/rotate?direction=cw",
                                  headers={**headers, "Content-Type": "application/json"})
            assert r.status_code == 200 and r.json()["status"] == "processing"
        options = get_redis_settings()
        options.conn_retries = 0
        redis = await create_pool(options)
        assert await redis.zcard(queue) == 1
        worker = Worker(
            functions=[authorized_rotate_item_image_job], redis_pool=redis, queue_name=queue,
            burst=True, max_burst_jobs=1, max_jobs=1, max_tries=1, retry_jobs=False,
            handle_signals=False, job_timeout=30, poll_delay=0.01, keep_result=0, log_results=False,
            health_check_key=queue+":health", ctx={"db_session_factory": async_session_maker},
        )
        try:
            await worker.async_run()
            assert worker.jobs_complete == 1 and worker.jobs_failed == 0
            assert await redis.zcard(queue) == 0
            async with seed() as db:
                obj = await db.get(ClothingItem, item_id)
                assert obj.user_id == owner and obj.status == ItemStatus.ready and obj.processing_kind is None
                assert obj.image_path == files["image_path"]
            with Image.open(service.get_image_path(files["image_path"])) as rotated:
                assert rotated.width > rotated.height
        finally:
            await worker.close()
        Path(os.environ["WARDROBE_POLICY_TEST_REPORT"]).write_text(json.dumps({
            "queue_unique": True, "worker_jobs_complete": 1, "worker_jobs_failed": 0,
            "rotation": "real_api_to_arq_to_guarded_worker",
        }))
    finally:
        if redis is not None:
            # Only this run's unique empty queue and its own health/result keys.
            keys = [key async for key in redis.scan_iter(match=queue+"*")]
            if keys:
                await redis.delete(*keys)
            await redis.aclose()
        async with seed() as db:
            await db.execute(delete(User).where(User.id == owner))
            await db.commit()
        if files:
            service.delete_images(files)
        await engine.dispose()
        await seed_engine.dispose()
