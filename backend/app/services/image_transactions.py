"""File compensation scoped to the actual outer SQLAlchemy transaction."""
import logging
from sqlalchemy import event

logger = logging.getLogger(__name__)
KEY = "wardrowbe_image_effects"


def _cleanup(actions):
    for service, paths in actions:
        try:
            service.delete_images(paths)
        except (OSError, ValueError):
            # A committed database write must not become a false failure response.
            # Do not log paths, original exception messages or signed URLs.
            logger.error("image_compensation_cleanup_failed")


def _commit(session):
    if session.in_nested_transaction():
        return
    state = session.info.pop(KEY, None)
    if state:
        _cleanup(state["commit"])


def _rollback(session):
    if session.in_nested_transaction():
        return
    state = session.info.pop(KEY, None)
    if state:
        _cleanup(state["rollback"])


def _register(db, service, paths, phase):
    session = db.sync_session
    if session.in_nested_transaction():
        raise ValueError("Image effects require an outer transaction")
    if not session.in_transaction():
        session.begin()
    if not session.info.get(KEY + "_listeners"):
        event.listen(session, "after_commit", _commit)
        event.listen(session, "after_rollback", _rollback)
        session.info[KEY + "_listeners"] = True
    state = session.info.setdefault(KEY, {"commit": [], "rollback": []})
    state[phase].append((service, dict(paths)))


def created_images(db, service, paths):
    try:
        _register(db, service, paths, "rollback")
    except Exception:
        _cleanup([(service, paths)])
        raise


def deleted_images(db, service, paths):
    _register(db, service, paths, "commit")
