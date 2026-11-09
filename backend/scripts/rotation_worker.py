"""Restricted rotation worker entry. No model, migrations or general task consumer."""
import asyncio
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

from local_agent import (
    CONFIG, PRIVATE, ROOT, RuntimeErrorSafe, application_policy, digest,
    private_json, process_record, require, write_private,
)

READY = PRIVATE / "rotation-worker-ready.json"
MODE = "remote_gpu"


def network_guard(event, args):
    if event == "socket.connect":
        address = args[1]
        if not isinstance(address, tuple) or address[:2] not in {
            ("127.0.0.1", 5432), ("127.0.0.1", 6379),
        }:
            raise PermissionError("Rotation worker network target denied")
    elif event == "socket.getaddrinfo" and args[0] != "127.0.0.1":
        raise PermissionError("Rotation worker DNS target denied")
    elif event in {"socket.bind", "subprocess.Popen", "os.system"}:
        raise PermissionError("Rotation worker cannot listen or launch subprocesses")


def configure(config_path=None):
    """Read existing private credentials; never put them in argv, state or logs."""
    config = private_json(config_path or CONFIG)
    require(application_policy(config).mode == "full_app", "旋转Worker仅允许可信full_app配置")
    require(os.environ.get("WARDROBE_AGENT_MODEL_MODE") == MODE, "旋转Worker必须保留remote_gpu模式")
    credentials = private_json(Path.home() / ".config/wardrowbe-agent/development-postgres.json")
    require((credentials["host"], credentials["port"], credentials["database"], credentials["username"]) ==
            ("127.0.0.1", 5432, "wardrowbe_agent_dev", "wardrowbe_agent_dev"), "Worker数据库身份不符")
    from dotenv import dotenv_values
    from urllib.parse import quote
    trusted = dotenv_values(ROOT / ".env")
    require(bool(trusted.get("SECRET_KEY")), "Worker签发密钥不存在")
    storage = Path.home() / ".local/share/wardrowbe-agent/images"
    require(storage.is_dir() and storage.stat().st_uid == os.getuid() and not storage.is_symlink(),
            "Worker图片目录身份不符")
    require(bool(os.environ.get("WARDROWBE_MEMORY_FILE")), "Worker资源快照未配置")
    for key in list(os.environ):
        if key.startswith(("AI_", "OIDC_", "FORWARD_AUTH_", "TEST_")):
            os.environ.pop(key)
    os.environ.update(
        DATABASE_URL="postgresql+asyncpg://" + quote(credentials["username"], safe="") + ":" +
        quote(credentials["password"], safe="") + "@127.0.0.1:5432/wardrowbe_agent_dev",
        REDIS_URL="redis://127.0.0.1:6379/15", SECRET_KEY=trusted["SECRET_KEY"],
        STORAGE_PATH=str(storage), AI_INTERNAL_ENABLED="false", AI_TEXT_ENABLED="false",
        AI_VISION_ENABLED="false", DATABASE_ECHO="false", LOG_LEVEL="WARNING",
    )
    return config


def validate_worker(settings):
    require([f.__name__ for f in settings.functions] == ["authorized_rotate_item_image_job"],
            "Worker函数注册不符")
    require(settings.queue_name == "arq:images" and settings.max_jobs == 1
            and settings.max_tries == 1 and settings.retry_jobs is False
            and settings.cron_jobs == [], "Worker队列或并发/重试配置不符")
    require(settings.redis_settings.host == "127.0.0.1" and settings.redis_settings.port == 6379
            and settings.redis_settings.database == 15, "Worker必须使用独立Redis DB15")


async def dependency_check(settings, *, database=None, redis=None):
    """Explicit read-only DB transaction and Redis reads; never consume queued jobs."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from redis.asyncio import Redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry
    owned_db, owned_redis = database is None, redis is None
    database = database if database is not None else create_async_engine(os.environ["DATABASE_URL"])
    redis = redis if redis is not None else Redis.from_url(
        os.environ["REDIS_URL"], socket_connect_timeout=3, socket_timeout=3, retry=Retry(NoBackoff(), 0),
    )
    try:
        async with database.connect() as conn:
            await conn.execute(text("SET TRANSACTION READ ONLY"))
            row = (await conn.execute(text("SELECT current_database(), current_user"))).one()
            require(tuple(row) == ("wardrowbe_agent_dev", "wardrowbe_agent_dev"), "Worker实际数据库身份不符")
            version = await conn.scalar(text("SELECT version_num FROM alembic_version"))
            require(version == "a7d3c9e1f5b2", "Worker数据库版本不符")
        require(await redis.ping() is True, "Worker Redis不可用")
        require(await redis.zcard(settings.queue_name) == 0,
                "旋转队列非空；拒绝直接消费历史任务，不删除或重试")
        # A second project consumer on this isolated DB/queue is not adopted.
        require(not await redis.exists(settings.queue_name + ":health-check"),
                "旋转队列存在其他消费者健康记录；停止核对，不清理")
    finally:
        if owned_db:
            await database.dispose()
        if owned_redis:
            await redis.aclose()


async def serve(config, settings, resource_check, baseline):
    from arq.worker import Worker
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    worker = None
    def request_stop():
        if worker is not None:
            worker.allow_pick_jobs = False
        stop.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, request_stop)

    async def startup(ctx):
        await settings.on_startup(ctx)
        # init_db has installed the same SQL guard before any task is accepted.
        resource_check(baseline, memory_file=os.environ["WARDROWBE_MEMORY_FILE"])
        write_private(READY, {
            "record": process_record(os.getpid()), "config_sha256": digest(CONFIG),
            "entry_sha256": digest(Path(__file__)), "queue": settings.queue_name,
        })

    async def before_job(ctx):
        try:
            resource_check(baseline, memory_file=os.environ["WARDROWBE_MEMORY_FILE"])
        except Exception:
            request_stop()
            raise

    worker = Worker(
        functions=settings.functions, redis_settings=settings.redis_settings,
        queue_name=settings.queue_name, cron_jobs=[], max_jobs=1, max_tries=1,
        retry_jobs=False, allow_abort_jobs=False, handle_signals=False,
        job_timeout=settings.job_timeout, job_completion_wait=settings.job_timeout,
        on_startup=startup, on_shutdown=settings.on_shutdown, on_job_start=before_job,
        health_check_interval=30, log_results=False,
    )
    task = asyncio.create_task(worker.async_run())
    try:
        while not stop.is_set() and not task.done():
            resource_check(baseline, memory_file=os.environ["WARDROWBE_MEMORY_FILE"])
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass
    finally:
        # Stop accepting work first; let current rotation finish within its existing timeout.
        worker.allow_pick_jobs = False
        try:
            while any(not t.done() for t in worker.tasks.values()):
                await asyncio.sleep(0.1)
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            await worker.close()
            if READY.exists():
                data = private_json(READY)
                if data.get("record", {}).get("pid") == os.getpid():
                    READY.unlink()


async def run(check_only=False, config_path=None):
    config = configure(config_path)
    sys.path.insert(0, str(ROOT / "backend"))
    sys.addaudithook(network_guard)
    from app.config import Settings, get_settings
    Settings.model_config["env_file"] = None
    get_settings.cache_clear()
    from app.services.agent_resources import check_remote_resources
    from app.workers.authorized_rotation import WardrobeRotationWorkerSettings
    settings = WardrobeRotationWorkerSettings
    settings.redis_settings.conn_retries = 0
    validate_worker(settings)
    baseline = check_remote_resources(memory_file=os.environ["WARDROWBE_MEMORY_FILE"])
    await dependency_check(settings)
    check_remote_resources(baseline, memory_file=os.environ["WARDROWBE_MEMORY_FILE"])
    if check_only:
        print(json.dumps({"worker_preflight": "passed", "mode": "check_only", "business_writes": 0}))
        return
    await serve(config, settings, check_remote_resources, baseline)


def failure_diagnostic(exc):
    """Expose bounded resource codes, never exception messages or task contents."""
    import math

    codes = {
        "windows_memory_low", "wsl_memory_low", "wsl_memory_pressure",
        "swap_activity", "resources_unavailable",
    }
    sources = {"configuration", "windows_snapshot", "wsl_meminfo", "wsl_pressure", "wsl_vmstat"}
    reasons = {
        "missing_path", "invalid_snapshot_shape", "invalid_timestamp", "future_timestamp",
        "stale_snapshot", "invalid_windows_counter", "missing_memory_counter",
        "missing_pressure_counter", "invalid_pressure_counter", "invalid_swap_counter",
        "io_error", "invalid_json", "missing_counter", "invalid_counter",
    }
    types = {
        "OllamaModelError", "RuntimeErrorSafe", "PermissionError", "OSError",
        "FileNotFoundError", "ValueError", "TypeError", "KeyError", "OverflowError",
        "JSONDecodeError", "_InvalidSample",
    }
    code = getattr(exc, "code", None)
    result = {
        "event": "rotation_worker_stopped",
        "error_type": type(exc).__name__ if type(exc).__name__ in types else "Exception",
        "error_code": code if type(code) is str and code in codes else "unavailable",
        "resource_diagnostics": "unavailable",
    }
    raw = getattr(exc, "resource_diagnostics", None)
    if type(raw) is dict:
        safe = {}
        for key, allowed in (("source", sources), ("reason", reasons), ("exception_type", types)):
            value = raw.get(key)
            if type(value) is str and value in allowed:
                safe[key] = value
        for key, upper in (("errno", 4095), ("snapshot_open_attempts", 3)):
            value = raw.get(key)
            if type(value) is int and 0 <= value <= upper:
                safe[key] = value
        age = raw.get("snapshot_age_seconds")
        if type(age) in (int, float) and math.isfinite(age) and abs(age) <= 10**9:
            safe["snapshot_age_seconds"] = round(age, 6)
        if safe:
            result["resource_diagnostics"] = safe
    return result


def main():
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--check-only", action="store_true")
        parser.add_argument("--config", type=Path, help="只读预检专用候选配置")
        args = parser.parse_args()
        require(args.config is None or args.check_only, "运行Worker只允许活动可信配置")
        # No subprocess environment, exceptions or job payloads go to stdout/stderr.
        import logging
        logging.disable(logging.CRITICAL)
        if args.check_only:
            asyncio.run(run(check_only=True, config_path=args.config))
        else:
            # Separate from the manager lock: also prevents direct duplicate entry execution.
            fd = os.open(PRIVATE / "rotation-worker.lock",
                         os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "r+") as lock:
                require(os.fstat(lock.fileno()).st_uid == os.getuid()
                        and os.fstat(lock.fileno()).st_mode & 0o777 == 0o600,
                        "Worker锁文件归属或权限异常")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                asyncio.run(run())
    except Exception as exc:
        print("受保护旋转Worker停止：" + json.dumps(failure_diagnostic(exc), sort_keys=True),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
