"""Shared async Redis connection pool for optional application features."""

import redis.asyncio as redis

from core.config import settings


redis_client = redis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)


async def check_redis() -> bool:
    try:
        return bool(await redis_client.ping())
    except Exception:
        return False
