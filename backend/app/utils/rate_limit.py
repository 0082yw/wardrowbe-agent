import logging
from ipaddress import IPv4Address, IPv6Address, ip_address, ip_network
from uuid import UUID

from fastapi import HTTPException, Request, status
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from app.config import get_settings

logger = logging.getLogger(__name__)


IPV6_CLIENT_PREFIX = 64


def _parse_address(value: str) -> IPv4Address | IPv6Address | None:
    # An address is tried whole first, because an unbracketed IPv6 address such as
    # 2001:db8::7:4711 would otherwise lose its last group as a port.
    try:
        return ip_address(value)
    except ValueError:
        pass
    if value.startswith("["):
        host, separator, port = value[1:].partition("]:")
        if not separator or not _is_port(port):
            return None
        try:
            return IPv6Address(host)
        except ValueError:
            return None
    if value.count(":") == 1:
        host, port = value.split(":")
        if not _is_port(port):
            return None
        try:
            return IPv4Address(host)
        except ValueError:
            return None
    return None


def _is_port(value: str) -> bool:
    return value.isascii() and value.isdigit() and int(value) <= 65535


def _client_bucket(value: str) -> str | None:
    address = _parse_address(value)
    if address is None:
        return None
    if isinstance(address, IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        # One subscriber is handed a whole /64, so per-address buckets would give them billions
        # of fresh limits. ULA and link-local stay per address so docker peers remain distinct.
        if address.is_global:
            return str(ip_network(f"{address}/{IPV6_CLIENT_PREFIX}", strict=False))
    return str(address)


def _get_client_ip(request: Request) -> str:
    raw_peer = request.client.host if request.client else "unknown"
    peer = _client_bucket(raw_peer) or raw_peer
    hops = get_settings().trusted_proxy_count
    if hops == 0:
        return peer

    # Repeated header lines are the same list as one comma-joined line, so a proxy that adds its
    # own line instead of appending to the client's still counts from the right.
    entries = [
        entry.strip()
        for line in request.headers.getlist("x-forwarded-for")
        for entry in line.split(",")
        if entry.strip()
    ]
    # Fewer entries than trusted proxies means the request skipped part of the chain, so any
    # entry may have been written by the client; the socket peer is the one address it cannot forge.
    if len(entries) < hops:
        return peer
    return _client_bucket(entries[-hops]) or peer


async def check_rate_limit(key: str, limit: int, window_seconds: int, *, fail_closed: bool = True) -> None:
    settings = get_settings()
    try:
        redis = Redis.from_url(settings.redis_url, socket_connect_timeout=3, socket_timeout=3,
                               retry=Retry(NoBackoff(), 0))
        try:
            # The window is fixed when the key is first created, because re-arming the expiry on
            # every hit would let a steady trickle keep the counter alive forever. SET NX rather
            # than EXPIRE NX so it works on Redis 6, where EXPIRE NX errors and limiting is lost.
            pipe = redis.pipeline(transaction=True)
            pipe.set(key, 0, ex=window_seconds, nx=True)
            pipe.incr(key)
            results = await pipe.execute()
            count = results[1]
            if count > limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please try again later.",
                )
        finally:
            await redis.aclose()
    except HTTPException:
        raise
    except Exception:
        logger.warning("rate_limit_store_unavailable")
        if fail_closed:
            raise HTTPException(status_code=503, detail={
                "error_code": "RATE_LIMIT_UNAVAILABLE",
                "message": "Request protection is temporarily unavailable; no operation was performed.",
            }) from None


async def rate_limit_by_ip(request: Request, action: str, limit: int, window_seconds: int) -> None:
    ip = _get_client_ip(request)
    key = f"rate_limit:{action}:ip:{ip}"
    await check_rate_limit(key, limit, window_seconds)


async def rate_limit_by_user(
    user_id: UUID, action: str, max_requests: int, window_seconds: int
) -> None:
    key = f"rate_limit:{action}:user:{user_id}"
    await check_rate_limit(key, max_requests, window_seconds)


async def require_rate_limit_store() -> None:
    """Sensitive entry points fail closed without adding a new request quota."""
    redis = None
    try:
        redis = Redis.from_url(get_settings().redis_url, socket_connect_timeout=3, socket_timeout=3,
                               retry=Retry(NoBackoff(), 0))
        if await redis.ping() is not True:
            raise RuntimeError("Rate-limit store did not confirm availability")
    except Exception:
        logger.warning("rate_limit_store_unavailable")
        raise HTTPException(status_code=503, detail={
            "error_code": "RATE_LIMIT_UNAVAILABLE",
            "message": "Request protection is temporarily unavailable; no operation was performed.",
        }) from None
    finally:
        if redis is not None:
            try:
                await redis.aclose()
            except Exception:
                logger.warning("rate_limit_store_close_failed")
