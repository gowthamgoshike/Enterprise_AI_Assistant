"""Shared fixed-window chat limit backed by Redis."""

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


async def allow_request(user_id: str) -> bool:
    key = f"v1:rate:chat:{user_id}"
    try:
        count = await redis_client.eval(
            INCREMENT_WITH_EXPIRY, 1, key, settings.rate_limit_window_seconds,
        )
    except Exception:
        REDIS_ERRORS_TOTAL.labels(operation="rate_limit").inc()
        logger.warning("redis_rate_limit_unavailable", extra={
            "event": "redis_rate_limit_unavailable",
        })
        return True

    allowed = int(count) <= settings.rate_limit_requests
    if not allowed:
        RATE_LIMIT_DENIALS_TOTAL.inc()
        logger.info("chat_rate_limit_denied", extra={
            "event": "chat_rate_limit", "result": "denied",
        })
    return allowed
