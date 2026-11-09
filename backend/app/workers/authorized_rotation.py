"""Only signed, owner-bound rotation jobs may enter the restricted worker."""
import hashlib
import hmac
import json
from uuid import UUID

from sqlalchemy import select

from app.access_policy import WardrobeAccessPolicy
from app.config import get_settings
from app.models.item import ClothingItem, ItemStatus
from app.workers.db import init_db, close_db
from app.workers.rotation import rotate_item_image_job
from app.workers.queues import IMAGE_QUEUE
from app.workers.settings import get_redis_settings
from app.write_guard import bind_session, install_write_guard


def rotation_receipt(user_id, item_id, direction: str, job_id: str) -> str:
    if direction not in ("cw", "ccw") or not job_id:
        raise ValueError("Invalid rotation job")
    payload = json.dumps([str(UUID(str(user_id))), str(UUID(str(item_id))), direction, job_id], separators=(",", ":"))
    signature = hmac.new(get_settings().secret_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return payload + "." + signature


def verify_rotation(receipt: str, item_id: str, direction: str, job_id: str) -> UUID:
    try:
        payload, _signature = receipt.rsplit(".", 1)
        values = json.loads(payload)
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError
        user = UUID(values[0])
        expected = rotation_receipt(user, item_id, direction, job_id)
        if not hmac.compare_digest(expected, receipt):
            raise ValueError
        return user
    except (ValueError, TypeError, AttributeError, IndexError):
        raise PermissionError("Invalid authenticated rotation receipt") from None


async def authorized_rotate_item_image_job(ctx, item_id: str, direction: str, receipt: str):
    actor = verify_rotation(receipt, item_id, direction, ctx.get("job_id", ""))
    policy = WardrobeAccessPolicy(get_settings().wardrobe_app_mode)
    if not policy.permits_write("rotate"):
        raise PermissionError("Rotation worker disabled in this application mode")
    factory = ctx["db_session_factory"]
    async with factory() as db:
        item = await db.scalar(select(ClothingItem).where(ClothingItem.id == UUID(item_id)))
        if (item is None or item.user_id != actor or item.status != ItemStatus.processing
                or item.processing_kind != "rotate" or item.ai_job_id != ctx["job_id"]):
            raise PermissionError("Rotation job does not match current ownership and state")

    def scoped_session():
        db = factory()
        bind_session(db, policy, actor, "POST", f"/api/v1/items/{item_id}/rotate")
        return db

    return await rotate_item_image_job({**ctx, "db_session_factory": scoped_session}, item_id, direction)


async def startup(ctx):
    if get_settings().wardrobe_app_mode != "full_app":
        raise PermissionError("Restricted image worker requires trusted full_app")
    await init_db(ctx)
    install_write_guard(ctx["db_engine"])


class WardrobeRotationWorkerSettings:
    functions = [authorized_rotate_item_image_job]
    cron_jobs = []
    on_startup = startup
    on_shutdown = close_db
    redis_settings = get_redis_settings()
    queue_name = IMAGE_QUEUE
    max_jobs = 1
    max_tries = 1
    retry_jobs = False
    allow_abort_jobs = True
    job_timeout = get_settings().image_job_timeout
    health_check_interval = 30
