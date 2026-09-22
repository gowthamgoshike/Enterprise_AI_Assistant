"""Apply endpoint-specific limits at the cheapest trusted boundary."""

from fastapi import Depends, Request

from apps.api.dependencies.auth import get_current_user
from apps.api.schemas.auth import AuthenticatedUser
from core.config import settings
from core.errors import RateLimitError
from core.rate_limit import allow_request


async def _enforce(
    subject: str,
    *,
    scope: str,
    limit: int | None = None,
) -> None:
    if not await allow_request(subject, scope=scope, limit=limit):
        raise RateLimitError(settings.rate_limit_window_seconds)


async def get_rate_limited_chat_user(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    await _enforce(
        current_user.user_id,
        scope="chat",
    )
    return current_user


async def get_rate_limited_approval_user(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    await _enforce(
        current_user.user_id,
        scope="approval",
        limit=settings.approval_rate_limit_requests,
    )
    return current_user


async def get_rate_limited_job_user(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    await _enforce(
        current_user.user_id,
        scope="jobs",
        limit=settings.job_rate_limit_requests,
    )
    return current_user


async def rate_limit_login(request: Request) -> None:
    # Do not trust X-Forwarded-For until a known proxy topology is configured.
    client_host = request.client.host if request.client else "unknown"
    await _enforce(
        client_host,
        scope="login",
        limit=settings.login_rate_limit_requests,
    )
