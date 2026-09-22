"""Validated request correlation plus HTTP metrics and completion logs."""

import logging
import re
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware

from core.observability.context import reset_request_context, set_request_context
from core.observability.metrics import HTTP_REQUEST_DURATION, HTTP_REQUESTS_TOTAL


logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
JOB_ID_PATH_SEGMENT = re.compile(
    r"^/api/v1/jobs/(?!failed-transactions(?:/|$))[^/]+(?=/|$)"
)


def _request_id(value: str | None) -> str:
    if value and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return f"REQ-{uuid4()}"


def _metric_path(request) -> str:
    path = request.url.path or "unmatched"
    return JOB_ID_PATH_SEGMENT.sub("/api/v1/jobs/{job_id}", path)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = _request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        tokens = set_request_context(request_id)
        started = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_seconds = time.perf_counter() - started
            path = _metric_path(request)
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
