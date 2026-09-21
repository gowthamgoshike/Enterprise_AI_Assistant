"""Reliable invocation boundary for model providers."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from ollama import RequestError, ResponseError

from core.config import settings
from core.errors import ModelUnavailableError
from core.retry import retry_async
from core.observability.metrics import MODEL_LATENCY, MODEL_REQUESTS_TOTAL


T = TypeVar("T")
logger = logging.getLogger(__name__)
MODEL_NAME = settings.ollama_model


def _is_transient_response_error(exc: ResponseError) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


async def invoke_model_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    request_id: str | None = None,
    operation_name: str = "model_call",
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    """Apply a timeout and retry only known transient provider failures."""

    async def invoke_once() -> T:
        started = time.perf_counter()
        status = "failure"
        try:
            result = await asyncio.wait_for(
                operation(),
                timeout=settings.model_timeout_seconds,
            )
            status = "success"
            return result
        except ModelUnavailableError:
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelUnavailableError("Model request timed out") from exc
        except (httpx.NetworkError, ConnectionError, RequestError) as exc:
            raise ModelUnavailableError() from exc
        except ResponseError as exc:
            if _is_transient_response_error(exc):
                raise ModelUnavailableError() from exc
            raise
        finally:
            duration_seconds = time.perf_counter() - started
            MODEL_REQUESTS_TOTAL.labels(
                model=MODEL_NAME,
                status=status,
            ).inc()
            MODEL_LATENCY.labels(model=MODEL_NAME).observe(duration_seconds)
            logger.info(
                "model_call_completed",
                extra={
                    "event": "model_call",
                    "model": MODEL_NAME,
                    "status": status,
                    "latency_ms": round(duration_seconds * 1000, 2),
                },
            )

    return await retry_async(
        invoke_once,
        max_retries=settings.model_max_retries,
        retry_on=(ModelUnavailableError,),
        operation_name=operation_name,
        request_id=request_id,
        on_retry=on_retry,
    )
