import asyncio

import jwt
from fastapi import (
    Depends,
    Request,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from apps.api.schemas.auth import AuthenticatedUser
from services.auth_service.jwt_service import decode_access_token
from services.auth_service.repository import find_user_by_id
from core.observability.context import set_user_context
from core.errors import AuthenticationRequiredError, InvalidTokenError


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    request: Request = None,
) -> AuthenticatedUser:
    if credentials is None:
        raise AuthenticationRequiredError()
    token = credentials.credentials

    try:
        payload = decode_access_token(
            token
        )
        user_id = payload.get("sub")

        if not user_id:
            raise ValueError(
                "Missing subject"
            )

    except (
        jwt.InvalidTokenError,
        ValueError,
    ) as exc:
        raise InvalidTokenError() from exc

    user = await asyncio.to_thread(find_user_by_id, user_id)

    if (
        not user
        or not user["is_active"]
    ):
        raise InvalidTokenError()

    authenticated_user = AuthenticatedUser(
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
        customer_id=user["customer_id"],
    )
    if request is not None:
        set_user_context(authenticated_user.user_id)
    return authenticated_user
