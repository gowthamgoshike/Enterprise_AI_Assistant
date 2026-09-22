"""Best-effort JSON cache for public aggregate analytics."""

import json
import logging

from core.observability.metrics import (
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    REDIS_ERRORS_TOTAL,
)
from core.redis import redis_client


logger = logging.getLogger(__name__)
CACHE_TYPES = frozenset({
    "top_products", "revenue_summary", "transaction_metrics",
})


def _cache_type(key: str) -> str:
    parts = key.split(":", 3)
    if (
        len(parts) > 2
        and parts[0] == "v1"
        and parts[1] == "analytics"
        and parts[2] in CACHE_TYPES
    ):
        return parts[2]
    return "other"


async def cache_get(key: str):
    cache_type = _cache_type(key)
    try:
        value = await redis_client.get(key)
        if value is None:
            CACHE_MISSES_TOTAL.labels(cache_type=cache_type).inc()
            logger.info("cache_lookup", extra={
                "event": "cache_lookup", "cache_type": cache_type,
                "result": "miss",
            })
            return None
        decoded = json.loads(value)
        CACHE_HITS_TOTAL.labels(cache_type=cache_type).inc()
        logger.info("cache_lookup", extra={
            "event": "cache_lookup", "cache_type": cache_type,
            "result": "hit",
        })
        return decoded
    except Exception:
        REDIS_ERRORS_TOTAL.labels(operation="get").inc()
        logger.warning("redis_cache_get_unavailable", extra={
            "event": "redis_error", "cache_type": cache_type,
            "operation": "get",
        })
        return None


async def cache_set(key: str, value, ttl_seconds: int) -> None:
    try:
        await redis_client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
    except Exception:
        REDIS_ERRORS_TOTAL.labels(operation="set").inc()
        logger.warning("redis_cache_set_unavailable", extra={
            "event": "redis_error", "cache_type": _cache_type(key),
            "operation": "set",
        })
