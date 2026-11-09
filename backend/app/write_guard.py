"""Session-bound ownership checks; raw/core DML is never a wardrobe grant."""
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID
from pathlib import PurePosixPath
from weakref import WeakKeyDictionary
import re

from fastapi import HTTPException
from sqlalchemy import event, inspect, select, exists
from sqlalchemy.orm import Session

from app.access_policy import WardrobeAccessPolicy

KEY = "wardrobe_write_scope"
_FLUSH = ContextVar("wardrobe_trusted_flush", default=None)
_ENGINES = WeakKeyDictionary()
_SESSION_LISTENERS = False
FORBIDDEN_FIELDS = frozenset({"id", "user_id", "owner_id", "family_id", "session", "mode", "execution_mode", "model", "model_name", "endpoint"})


@dataclass(frozen=True)
class WriteScope:
    policy: WardrobeAccessPolicy
    user_id: UUID
    operation: str | None
    item_id: UUID | None


def bind_session(db, policy: WardrobeAccessPolicy, user_id: UUID, method: str, path: str) -> None:
    match = policy.operation(method, path)
    scope = WriteScope(policy, UUID(str(user_id)), match[0] if match else None, match[1] if match else None)
    session = db.sync_session
    previous = session.info.get(KEY)
    if previous is not None and previous != scope:
        raise PermissionError("Session identity cannot change")
    session.info[KEY] = scope
    session.info.setdefault(KEY + "_created", set())


async def bind_authenticated_request(db, request, user) -> None:
    from app.config import get_settings

    policy = WardrobeAccessPolicy(get_settings().wardrobe_app_mode)
    # Read scopes are also bound, so incidental writes cannot borrow a previous grant.
    bind_session(db, policy, user.id, request.method, request.url.path)
    if policy.operation(request.method, request.url.path):
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            data = await request.form()
        elif request.headers.get("content-type", "").startswith("application/json"):
            # The existing client keeps this header on bodyless rotate/restore POSTs.
            # Only a genuinely absent body is empty; malformed non-empty JSON still fails.
            body = await request.body()
            if not body:
                data = {}
            else:
                try:
                    data = await request.json()
                except (ValueError, UnicodeDecodeError):
                    raise HTTPException(422, "Invalid JSON request body") from None
                if not isinstance(data, dict):
                    raise HTTPException(422, "JSON request body must be an object")
        else:
            data = {}
        if isinstance(data, dict) or hasattr(data, "keys"):
            if FORBIDDEN_FIELDS.intersection(data.keys()):
                raise HTTPException(422, "Server-controlled identity and configuration fields are forbidden")


def _before_flush(session, flush_context, instances):
    engine = session.get_bind()
    if engine not in _ENGINES:
        return
    scope = session.info.get(KEY)
    if scope is None:
        return  # The cursor guard denies unscoped DML; the preserved login adapter has its own exact exception.
    if not scope.policy.permits_write(scope.operation or ""):
        raise PermissionError("This request has no wardrobe write grant")
    new, deleted = set(session.new), set(session.deleted)
    changed = new | deleted | {o for o in session.dirty if session.is_modified(o, include_collections=False)}
    for obj in changed:
        if getattr(getattr(obj, "__table__", None), "name", None) != "clothing_items":
            raise PermissionError("Unrelated table mutation prohibited")
        if getattr(obj, "user_id", None) != scope.user_id:
            raise PermissionError("Cross-user wardrobe mutation prohibited")
        history = inspect(obj).attrs.user_id.history
        if history.deleted and any(v != scope.user_id for v in history.deleted):
            raise PermissionError("Clothing ownership is immutable")
        if scope.operation == "create":
            created = session.info[KEY + "_created"]
            if obj in new:
                created.add(obj)
            if obj in deleted or obj not in created:
                raise PermissionError("Create is limited to this request's new items")
        elif obj in new or getattr(obj, "id", None) != scope.item_id:
            raise PermissionError("Mutation is outside the authenticated target")
        for name in ("image_path", "medium_path", "thumbnail_path", "original_image_path"):
            value = getattr(obj, name, None)
            if value:
                path = PurePosixPath(value)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != str(scope.user_id):
                    raise PermissionError("Image reference is outside the authenticated user")
            if obj not in new and scope.operation != "create" and inspect(obj).attrs[name].history.has_changes():
                raise PermissionError("Image references cannot be reassigned")
        if obj in deleted and scope.operation != "delete":
            raise PermissionError("This operation cannot delete")
        if scope.operation == "delete" and obj not in deleted:
            raise PermissionError("Delete cannot modify another record")
    _FLUSH.set((engine, scope, id(session)))


def _clear_flush(*args):
    _FLUSH.set(None)


def _orm_execute(state):
    if state.session.get_bind() not in _ENGINES or KEY not in state.session.info:
        return
    # Bulk/core statements do not pass ORM object ownership validation.
    if state.is_insert or state.is_update or state.is_delete:
        raise PermissionError("Core/bulk DML has no wardrobe object grant")


def authorize_statement(conn, statement: str, context=None) -> None:
    kind = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ""
    if kind in {"SELECT", "SHOW", "BEGIN", "COMMIT", "ROLLBACK"}:
        return
    if re.fullmatch(r"SET TRANSACTION READ ONLY", statement.strip(), re.I):
        return
    grant = _FLUSH.get()
    if kind not in {"INSERT", "UPDATE", "DELETE"} or grant is None or grant[0] is not conn.engine:
        raise PermissionError("Unscoped database mutation prohibited")
    scope = grant[1]
    compiled = getattr(context, "compiled", None)
    table = getattr(getattr(compiled, "statement", None), "table", None)
    if getattr(table, "name", None) != "clothing_items" or not scope.policy.permits_write(scope.operation or ""):
        raise PermissionError("SQL is outside the wardrobe policy")
    if kind == "INSERT" and scope.operation != "create":
        raise PermissionError("Insert requires a create grant")
    if kind == "DELETE" and scope.operation != "delete":
        raise PermissionError("Delete requires a delete grant")


def _cursor(conn, cursor, statement, parameters, context, executemany):
    authorize_statement(conn, statement, context)


def install_write_guard(engine, *, sql_listener: bool = True) -> None:
    global _SESSION_LISTENERS
    engine = getattr(engine, "sync_engine", engine)
    if engine not in _ENGINES:
        _ENGINES[engine] = True
    if not _SESSION_LISTENERS:
        event.listen(Session, "before_flush", _before_flush)
        event.listen(Session, "after_flush_postexec", _clear_flush)
        event.listen(Session, "after_soft_rollback", _clear_flush)
        event.listen(Session, "do_orm_execute", _orm_execute)
        _SESSION_LISTENERS = True
    if sql_listener and not event.contains(engine, "before_cursor_execute", _cursor):
        event.listen(engine, "before_cursor_execute", _cursor)


def guard_installed(engine) -> bool:
    return getattr(engine, "sync_engine", engine) in _ENGINES


async def refuse_linked_delete(db, item_id) -> None:
    """Do not silently cascade a wardrobe permission into outfit/history tables."""
    from app.database import Base

    if KEY not in db.sync_session.info:
        return
    from app.models.item import ClothingItem, ItemStatus
    item = await db.get(ClothingItem, item_id)
    if item is not None and item.status == ItemStatus.processing:
        raise HTTPException(409, "Wait for the active image operation before deleting")
    for table in Base.metadata.tables.values():
        for foreign in table.foreign_keys:
            if foreign.target_fullname == "clothing_items.id":
                present = await db.scalar(select(exists().where(foreign.parent == item_id)))
                if present:
                    raise HTTPException(409, "This item has linked records; deletion requires a separate supported workflow")
