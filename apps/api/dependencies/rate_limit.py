"""Apply the chat limit only after verified authentication."""

from fastapi import Depends, HTTPException

from apps.api.dependencies.auth import get_current_user
from apps.api.schemas.auth import AuthenticatedUser
from core.config import settings
from core.rate_limit import allow_request


async def get_rate_limited_chat_user(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    if not await allow_request(current_user.user_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again shortly.",
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )
    return current_user
