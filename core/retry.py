"""Bounded asynchronous retries with exponential backoff and jitter."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from core.config import settings
from core.observability.metrics import RETRIES_TOTAL


logger = logging.getLogger(__name__)
T = TypeVar("T")


def calculate_backoff(
    attempt: int,
    base_delay: float | None = None,
    max_delay: float | None = None,
) -> float:
    """Return capped exponential delay plus up to twenty percent jitter."""
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    if base_delay is None:
        base_delay = settings.retry_base_delay_seconds
    if max_delay is None:
        max_delay = settings.retry_max_delay_seconds
    delay = min(base_delay * (2 ** attempt), max_delay)
    return delay + random.uniform(0, delay * 0.2)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    retry_on: tuple[type[Exception], ...],
    operation_name: str = "operation",
    request_id: str | None = None,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    """Retry only explicitly classified exceptions, never arbitrary failures."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except retry_on as exc:
            if attempt >= max_retries:
                raise
            delay = calculate_backoff(attempt)
            logger.warning(
                "operation_retry",
                extra={
                    "event": "retry",
                    "request_id": request_id,
                    "operation": operation_name,
                    "attempt": attempt + 2,
                    "max_attempts": max_retries + 1,
                    "retry_delay": delay,
                    "status": "retrying",
                    "error_code": getattr(exc, "code", type(exc).__name__),
                },
            )
            RETRIES_TOTAL.labels(operation=operation_name).inc()
            if on_retry is not None:
                on_retry(attempt + 2, delay, exc)
            await asyncio.sleep(delay)


async def run_with_timeout(
    coroutine: Awaitable[T],
    timeout_seconds: float,
) -> T:
    """Bound an async tool operation and translate its timeout."""
    from core.errors import ToolTimeoutError

    try:
        return await asyncio.wait_for(coroutine, timeout=timeout_seconds)
    except TimeoutError as exc:
        raise ToolTimeoutError() from exc
