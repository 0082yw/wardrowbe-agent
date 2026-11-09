"""Opt-in 30-minute real rotation stability run. Never uses production resources."""
import asyncio
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "backend/scripts")]


PHASE = "initializing"


def failure_details(exc):
    """Keep location and stage, never exception text, locals, URLs or credentials."""
    import traceback
    from rotation_worker import failure_diagnostic
    result = failure_diagnostic(exc)
    allowed = {"AssertionError", "ConnectionError", "TimeoutError", "RuntimeError",
               "PermissionError", "ValueError", "TypeError", "KeyError", "FileNotFoundError"}
    if type(exc).__name__ in allowed:
        result["error_type"] = type(exc).__name__
    result["phase"] = PHASE
    result["frames"] = [
        {"file": Path(frame.filename).name, "line": frame.lineno}
        for frame in traceback.extract_tb(exc.__traceback__)[-8:]
    ]
    return result


def isolated_redis_url(value):
    """Select test DB2 without losing the existing test account credentials."""
    from sqlalchemy.engine import make_url
    url = make_url(value)
    if (url.drivername, url.host, url.port) != ("redis", "127.0.0.1", 16379):
        raise ValueError("Only the isolated Redis endpoint is allowed")
    return url.set(database="2").render_as_string(hide_password=False)


def isolation():
    from sqlalchemy.engine import make_url
    db = make_url(os.environ["TEST_DATABASE_URL"])
    redis = make_url(os.environ["TEST_REDIS_URL"])
    assert (db.host, db.port, db.database) == ("127.0.0.1", 15432, "wardrowbe_browser_test")
    assert (redis.host, redis.port, redis.database) == ("127.0.0.1", 16379, "2")
    assert os.environ["DATABASE_URL"] == os.environ["TEST_DATABASE_URL"]
    assert os.environ["REDIS_URL"] == os.environ["TEST_REDIS_URL"]
    stage = Path(os.environ["WARDROBE_STABILITY_STAGE"]).resolve()
    storage = Path(os.environ["STORAGE_PATH"]).resolve()
    assert stage.name.startswith("wardrowbe-lesson68-") and storage == stage / "images"
    assert os.environ["WARDROBE_STABILITY_RUN"] == "1"
    from app.config import Settings, get_settings
    Settings.model_config["env_file"] = None
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.wardrobe_app_mode == "full_app"
    assert not settings.effective_ai_text_enabled and not settings.effective_ai_vision_enabled
    return stage


def network(event, args):
    if event == "socket.connect":
        assert isinstance(args[1], tuple) and args[1][:2] in {
            ("127.0.0.1", 15432), ("127.0.0.1", 16379)
        }, "Only isolated test dependencies allowed"
    if event == "socket.getaddrinfo":
        assert args[0] in ("127.0.0.1", None), "External DNS denied"


def write(path, data):
    temporary = path.with_suffix(".new")
    with temporary.open("w") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    temporary.chmod(0o600)
    os.replace(temporary, path)


async def child(stage):
    from types import SimpleNamespace
    import rotation_worker as entry
    from app.workers.authorized_rotation import WardrobeRotationWorkerSettings as settings
    from app.services.agent_resources import check_remote_resources
    import logging
    logging.disable(logging.CRITICAL)
    entry.READY = stage / "worker-ready.json"
    entry.CONFIG = stage / "worker-config.json"
    options = SimpleNamespace(functions=settings.functions, queue_name=os.environ["WARDROBE_STABILITY_QUEUE"],
                              cron_jobs=[], max_jobs=1, max_tries=1, retry_jobs=False,
                              job_timeout=30, redis_settings=settings.redis_settings,
                              on_startup=settings.on_startup, on_shutdown=settings.on_shutdown)
    options.redis_settings.conn_retries = 0
    baseline = check_remote_resources(memory_file=os.environ["WARDROWBE_MEMORY_FILE"])
    await entry.serve({}, options, check_remote_resources, baseline)


async def scenario(stage):
    global PHASE
    PHASE = "api_imports"
    import httpx
    from fastapi import FastAPI
    from PIL import Image
    from sqlalchemy import delete, select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from arq import create_pool
    from arq.jobs import Job
    from app.config import get_settings
    from app.api import items, images
    from app.api.auth import create_access_token
    from app.models.user import User
    from app.models.identity import UserIdentity
    from app.models.item import ClothingItem
    from app.database import engine
    from app.services.application_access import ApplicationAccessMiddleware
    from app.workers.authorized_rotation import rotation_receipt
    from app.workers.settings import get_redis_settings
    from app.services.agent_resources import check_remote_resources
    from app.write_guard import install_write_guard

    queue = os.environ["WARDROBE_STABILITY_QUEUE"]
    assert queue.startswith("arq:lesson68:rotation:") and queue != "arq:images"
    items.IMAGE_QUEUE = queue  # Isolated process only; production constants unchanged.
    install_write_guard(engine)
    seed_engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    seed = async_sessionmaker(seed_engine, expire_on_commit=False)
    actor, identity = uuid4(), uuid4()
    marker = "l68-" + uuid4().hex
    options = get_redis_settings()
    options.conn_retries = 0
    PHASE = "isolated_redis_connect"
    redis = await create_pool(options)
    assert await redis.zcard(queue) == 0 and not await redis.exists(queue + ":health-check")
    app = FastAPI()
    app.add_middleware(ApplicationAccessMiddleware)
    app.include_router(items.router, prefix="/api/v1")
    app.include_router(images.router, prefix="/api/v1")
    stream = BytesIO()
    picture = Image.new("RGB", (73, 101), "white")
    for x in range(20):
        for y in range(40):
            picture.putpixel((x, y), (x * 9, y * 5, 100))
    picture.save(stream, "JPEG")
    worker = None
    item_id = None
    report = {"duration_required_seconds": 1800, "queue": queue, "jobs": [],
              "resources": [], "passed": False, "failures": [], "duplicate_rejected": False}
    PHASE = "resource_preflight"
    baseline = check_remote_resources(memory_file=os.environ["WARDROWBE_MEMORY_FILE"])
    if "--preflight" in sys.argv:
        await redis.aclose()
        await seed_engine.dispose()
        await engine.dispose()
        print(json.dumps({"isolated_preflight": "passed", "business_writes": 0}), flush=True)
        return
    before_time = time.time()
    try:
        PHASE = "seed_test_identity"
        async with seed() as db:
            db.add(User(id=actor, email=marker+"@example.com", display_name="Stability Test", email_verified=True))
            await db.flush()
            db.add(UserIdentity(id=identity, user_id=actor, provider="dev", subject=marker))
            await db.commit()
        headers = {"Authorization": "Bearer " + create_access_token(actor, identity)}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://isolated.test") as client:
            PHASE = "upload_test_image"
            response = await client.post("/api/v1/items", headers=headers,
                                         data={"type": "shirt", "name": marker, "skip_ai": "true"},
                                         files={"image": ("stability.jpg", stream.getvalue(), "image/jpeg")})
            assert response.status_code == 201, f"isolated upload HTTP {response.status_code}"
            item_id = response.json()["id"]
            target = "/api/v1/items/" + item_id
            write(stage/"worker-config.json", {"isolated": True, "queue": queue})
            PHASE = "worker_start"
            with (stage/"worker.log").open("wb") as log:
                worker = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
                                          cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(40):
                assert worker.poll() is None, "isolated worker exited at startup"
                if (stage/"worker-ready.json").exists():
                    break
                await asyncio.sleep(.25)
            assert (stage/"worker-ready.json").exists(), "isolated worker not ready"
            from local_agent import process_record, same_process
            record = process_record(worker.pid)
            report.update(worker_pid=worker.pid, worker_started_at=time.time())
            start = time.monotonic()
            completed = 0
            while time.monotonic()-start < 1800:
                assert worker.poll() is None and same_process(record), "isolated worker exited"
                value = check_remote_resources(baseline, memory_file=os.environ["WARDROWBE_MEMORY_FILE"])
                status = Path(f"/proc/{worker.pid}/status").read_text()
                rss = int(next(l.split()[1] for l in status.splitlines() if l.startswith("VmRSS:"))) * 1024
                stat = Path(f"/proc/{worker.pid}/stat").read_text().rsplit(")", 1)[1].split()
                report["resources"].append({"at": time.time(), "rss_bytes": rss, "cpu_ticks": int(stat[11])+int(stat[12]),
                                            "windows_bytes": value.windows_bytes, "wsl_bytes": value.wsl_bytes,
                                            "psi_full": value.psi_full, "swap_in": value.swap_in, "swap_out": value.swap_out})
                elapsed = time.monotonic()-start
                if completed < 30 and elapsed >= completed * 60:
                    PHASE = "rotation_task"
                    direction = "cw" if completed % 2 == 0 else "ccw"
                    call_start = time.monotonic()
                    response = await client.post(target+"/rotate?direction="+direction,
                                                 headers={**headers, "Content-Type": "application/json"})
                    assert response.status_code == 200, f"rotation HTTP {response.status_code}"
                    async with seed() as db:
                        item = await db.get(ClothingItem, UUID(item_id))
                        job_id = item.ai_job_id
                    assert job_id and job_id not in [x["job_id"] for x in report["jobs"]]
                    if completed == 0:
                        duplicate = await redis.enqueue_job(
                            "authorized_rotate_item_image_job", item_id, direction,
                            rotation_receipt(actor, item_id, direction, job_id), _job_id=job_id, _queue_name=queue)
                        assert duplicate is None, "duplicate job accepted"
                        report["duplicate_rejected"] = True
                    result = await Job(job_id, redis, _queue_name=queue).result(timeout=30, poll_delay=.1)
                    assert result == {"status": "success", "item_id": item_id}
                    response = await client.get(target, headers=headers)
                    assert response.status_code == 200 and response.json()["status"] == "ready"
                    expected = (101, 73) if completed % 2 == 0 else (73, 101)
                    for key in ["image_path", "medium_path", "thumbnail_path"]:
                        with Image.open(Path(os.environ["STORAGE_PATH"])/response.json()[key]) as image:
                            assert image.size == expected, "duplicate execution or wrong rotation"
                    completed += 1
                    report["jobs"].append({"job_id": job_id, "direction": direction, "result": "success",
                                           "seconds": round(time.monotonic()-call_start, 3)})
                    write(stage/"stability-progress.json", {**report, "elapsed_seconds": round(time.monotonic()-start, 2)})
                    print(json.dumps({"completed_jobs": completed, "elapsed_seconds": round(time.monotonic()-start, 1)}), flush=True)
                await asyncio.sleep(5)
            assert len(report["jobs"]) == 30
            health = (await redis.get(queue+":health-check")).decode()
            assert "j_complete=30" in health and "j_failed=0" in health, "worker counts mismatch"
            report.update(passed=True, elapsed_seconds=round(time.monotonic()-start, 2), health_counts=health)
            # Stop only our worker by its verified identity; no force kill.
            assert same_process(record)
            worker.send_signal(signal.SIGTERM)
            await asyncio.to_thread(worker.wait, 35)
            assert worker.returncode == 0 and not (stage/"worker-ready.json").exists()
            response = await client.delete(target, headers=headers)
            assert response.status_code == 204, f"isolated cleanup HTTP {response.status_code}"
            item_id = None
        async with seed() as db:
            await db.execute(delete(UserIdentity).where(UserIdentity.id == identity))
            await db.execute(delete(User).where(User.id == actor))
            await db.commit()
        # Delete only this run's unique queue/result keys, never flush any Redis database.
        keys = [queue, queue+":health-check"]
        for job in report["jobs"]:
            keys.extend("arq:"+suffix+":"+job["job_id"] for suffix in ["job", "result", "retry", "in-progress"])
        if keys:
            await redis.delete(*keys)
        report["temporary_image_files_remaining"] = sum(1 for p in Path(os.environ["STORAGE_PATH"]).rglob("*") if p.is_file())
        assert report["temporary_image_files_remaining"] == 0
        report["cleanup_complete"] = True
    except Exception as exc:
        from rotation_worker import failure_diagnostic
        report["passed"] = False
        report["failures"].append(failure_details(exc))
        if worker is not None and worker.poll() is None:
            from local_agent import process_record
            if process_record(worker.pid):
                worker.send_signal(signal.SIGTERM)
                await asyncio.to_thread(worker.wait, 35)
        # Failed evidence and only this run's isolated data are retained; no force cleanup.
        raise
    finally:
        report["wall_seconds"] = round(time.time()-before_time, 2)
        write(stage/"stability-result.json", report)
        await redis.aclose()
        await seed_engine.dispose()
        await engine.dispose()


if __name__ == "__main__":
    stage = None
    try:
        PHASE = "isolation"
        stage = isolation()
        sys.addaudithook(network)
        asyncio.run(child(stage) if "--child" in sys.argv else scenario(stage))
    except Exception as exc:
        diagnostic = failure_details(exc)
        if stage is not None:
            write(stage / ("worker-failure.json" if "--child" in sys.argv else "startup-failure.json"), diagnostic)
        print(json.dumps(diagnostic), file=sys.stderr)
        raise SystemExit(1)
