"""Reliable invocation boundary for model providers."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from ollama import RequestError, ResponseError

from core.config import MODEL_MAX_RETRIES, MODEL_TIMEOUT_SECONDS
from core.errors import ModelUnavailableError
from core.retry import retry_async


T = TypeVar("T")


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
        try:
            return await asyncio.wait_for(
                operation(),
                timeout=MODEL_TIMEOUT_SECONDS,
            )
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

    return await retry_async(
        invoke_once,
        max_retries=MODEL_MAX_RETRIES,
        retry_on=(ModelUnavailableError,),
        operation_name=operation_name,
        request_id=request_id,
        on_retry=on_retry,
    )
