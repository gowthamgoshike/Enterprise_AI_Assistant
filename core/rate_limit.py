"""Endpoint-scoped fixed-window limits backed by Redis."""

import logging

from core.config import settings
from core.observability.metrics import RATE_LIMIT_DENIALS_TOTAL, REDIS_ERRORS_TOTAL
from core.redis import redis_client


logger = logging.getLogger(__name__)

# Keep the first increment and expiry atomic. A process stopping between two
# separate commands would otherwise leave a counter without an expiry.
INCREMENT_WITH_EXPIRY = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


async def allow_request(
    subject: str,
    *,
    scope: str = "chat",
    limit: int | None = None,
    window_seconds: int | None = None,
) -> bool:
    request_limit = limit or settings.rate_limit_requests
    window = window_seconds or settings.rate_limit_window_seconds
    key = f"v1:rate:{scope}:{subject}"
    try:
        count = await redis_client.eval(
            INCREMENT_WITH_EXPIRY, 1, key, window,
        )
    except Exception:
        REDIS_ERRORS_TOTAL.labels(operation="rate_limit").inc()
        logger.warning("redis_rate_limit_unavailable", extra={
            "event": "redis_rate_limit_unavailable",
        })
        return True

    allowed = int(count) <= request_limit
    if not allowed:
        RATE_LIMIT_DENIALS_TOTAL.inc()
        logger.info("api_rate_limit_denied", extra={
            "event": "api_rate_limit", "result": "denied",
            "operation": scope,
        })
    return allowed
