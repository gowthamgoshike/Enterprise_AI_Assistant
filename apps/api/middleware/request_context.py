"""Request correlation, HTTP logs, and low-cardinality HTTP metrics."""

import logging
import time
import uuid

from fastapi import Request

from core.observability.context import (
    reset_request_context,
    set_request_context,
)
from core.observability.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
)


logger = logging.getLogger(__name__)


def _route_template(request: Request) -> str:
    # Current API endpoints are static. When parameterized routes are added,
    # resolve their templates here instead of labeling raw identifiers.
    return request.url.path or "unmatched"


async def request_context_middleware(request: Request, call_next):
    request_id = f"REQ-{uuid.uuid4()}"
    tokens = set_request_context(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    response = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_seconds = time.perf_counter() - started
        path = _route_template(request)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=path,
            status_code=str(status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=path,
        ).observe(duration_seconds)
        logger.info(
            "http_request_completed",
            extra={
                "event": "http_request",
                "method": request.method,
                "path": path,
                "status": status_code,
                "latency_ms": round(duration_seconds * 1000, 2),
            },
        )
        reset_request_context(tokens)
