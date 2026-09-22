"""Central translation from internal failures to a stable, safe API contract."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from apps.api.schemas.errors import ErrorDetail, ErrorResponse
from core.errors import (
    AppError,
    AuthenticationRequiredError,
    InvalidCredentialsError,
    InvalidTokenError,
    RateLimitError,
)
from services.auth_service.exceptions import AuthorizationError


logger = logging.getLogger(__name__)


STATUS_DEFAULTS: dict[int, tuple[str, str, bool]] = {
    400: ("bad_request", "The request could not be processed.", False),
    401: ("authentication_required", "Authentication is required.", False),
    403: (
        "permission_denied",
        "You do not have permission to perform this action.",
        False,
    ),
    404: ("resource_not_found", "The requested resource was not found.", False),
    409: ("conflict", "The request conflicts with the current state.", False),
    413: ("request_too_large", "The request is too large.", False),
    422: (
        "validation_error",
        "The request contains invalid fields.",
        False,
    ),
    429: (
        "rate_limit_exceeded",
        "Too many requests. Please try again shortly.",
        True,
    ),
    502: ("upstream_unavailable", "An upstream service is unavailable.", True),
    503: ("service_unavailable", "The service is temporarily unavailable.", True),
    504: ("request_timeout", "The operation timed out.", True),
}


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool = False,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    payload = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
        ),
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


def _app_error_status(error: AppError) -> int:
    if error.code in {"model_unavailable", "database_unavailable"}:
        return 503
    if error.code in {"tool_timeout"}:
        return 504
    if error.code.endswith("_not_found"):
        return 404
    return 400


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AuthenticationRequiredError)
    async def authentication_required_handler(request: Request, exc):
        return error_response(
            request,
            status_code=401,
            code=exc.code,
            message=exc.safe_message,
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(InvalidTokenError)
    async def invalid_token_handler(request: Request, exc):
        return error_response(
            request,
            status_code=401,
            code=exc.code,
            message=exc.safe_message,
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(InvalidCredentialsError)
    async def invalid_credentials_handler(request: Request, exc):
        return error_response(
            request,
            status_code=401,
            code=exc.code,
            message=exc.safe_message,
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_handler(request: Request, exc):
        return error_response(
            request,
            status_code=403,
            code="permission_denied",
            message="You do not have permission to perform this action.",
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc):
        return error_response(
            request,
            status_code=429,
            code=exc.code,
            message=exc.safe_message,
            retryable=True,
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc):
        return error_response(
            request,
            status_code=422,
            code="validation_error",
            message="The request contains invalid fields.",
        )

    @app.exception_handler(AppError)
    async def application_error_handler(request: Request, exc: AppError):
        return error_response(
            request,
            status_code=_app_error_status(exc),
            code=exc.code,
            message=exc.safe_message,
            retryable=exc.retryable,
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        default = STATUS_DEFAULTS.get(
            exc.status_code,
            ("internal_error", "The request could not be completed.", False),
        )
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return error_response(
            request,
            status_code=exc.status_code,
            code=str(detail.get("code", default[0])),
            message=str(detail.get("message", default[1])),
            retryable=bool(detail.get("retryable", default[2])),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.exception(
            "unhandled_api_error",
            extra={"event": "api_error", "error_code": "internal_error"},
        )
        return error_response(
            request,
            status_code=500,
            code="internal_error",
            message="The request could not be completed.",
        )
