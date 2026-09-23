"""Reliable invocation boundary for model providers."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from ollama import RequestError, ResponseError

from ai.models.concurrency import model_semaphore
from ai.models.errors import ModelTimeoutError
from core.config import settings
from core.errors import ModelUnavailableError
from core.retry import retry_async
from core.observability.metrics import (
    MODEL_LATENCY,
    MODEL_FAILURES_TOTAL,
    MODEL_QUEUE_WAIT,
    MODEL_REQUESTS_TOTAL,
    MODEL_TIMEOUTS_TOTAL,
)


T = TypeVar("T")
logger = logging.getLogger(__name__)
MODEL_NAME = settings.ollama_model
MODEL_PROVIDER = settings.model_provider


def _is_transient_response_error(exc: ResponseError) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


async def invoke_model_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    request_id: str | None = None,
    operation_name: str = "model_call",
    on_retry: Callable[[int, float, Exception], None] | None = None,
    semaphore: asyncio.Semaphore = model_semaphore,
    timeout_seconds: float | None = None,
    model_name: str = MODEL_NAME,
    provider_name: str = MODEL_PROVIDER,
) -> T:
    """Apply a timeout and retry only known transient provider failures."""

    async def invoke_once() -> T:
        queued_at = time.perf_counter()
        async with semaphore:
            queue_wait_seconds = time.perf_counter() - queued_at
            MODEL_QUEUE_WAIT.labels(
                provider=provider_name, model=model_name
            ).observe(queue_wait_seconds)
            logger.info(
                "model_slot_acquired",
                extra={
                    "event": "model_queue",
                    "model": model_name,
                    "provider": provider_name,
                    "model_queue_wait_ms": round(queue_wait_seconds * 1000, 2),
                },
            )
            started = time.perf_counter()
            status = "failure"
            try:
                result = await asyncio.wait_for(
                    operation(),
                    timeout=(timeout_seconds or settings.model_timeout_seconds),
                )
                status = "success"
                return result
            except ModelUnavailableError:
                raise
            except ModelTimeoutError:
                MODEL_TIMEOUTS_TOTAL.labels(
                    provider=provider_name, model=model_name
                ).inc()
                raise
            except (TimeoutError, httpx.TimeoutException) as exc:
                MODEL_TIMEOUTS_TOTAL.labels(
                    provider=provider_name, model=model_name
                ).inc()
                raise ModelTimeoutError() from exc
            except (httpx.NetworkError, ConnectionError, RequestError) as exc:
                raise ModelUnavailableError() from exc
            except ResponseError as exc:
                if _is_transient_response_error(exc):
                    raise ModelUnavailableError() from exc
                raise
            finally:
                duration_seconds = time.perf_counter() - started
                MODEL_REQUESTS_TOTAL.labels(
                    provider=provider_name,
                    model=model_name,
                    status=status,
                ).inc()
                MODEL_LATENCY.labels(
                    provider=provider_name, model=model_name
                ).observe(duration_seconds)
                if status == "failure":
                    MODEL_FAILURES_TOTAL.labels(
                        provider=provider_name, model=model_name
                    ).inc()
                logger.info(
                    "model_call_completed",
                    extra={
                        "event": "model_call",
                        "model": model_name,
                        "provider": provider_name,
                        "status": status,
                        "latency_ms": round(duration_seconds * 1000, 2),
                    },
                )

    return await retry_async(
        invoke_once,
        max_retries=settings.model_max_retries,
        retry_on=(ModelUnavailableError, ModelTimeoutError),
        operation_name=operation_name,
        request_id=request_id,
        on_retry=on_retry,
    )
