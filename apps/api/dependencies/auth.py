import jwt
from fastapi import (
    Depends,
    HTTPException,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from apps.api.schemas.auth import AuthenticatedUser
from services.auth_service.jwt_service import decode_access_token
from services.auth_service.repository import find_user_by_id


bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
) -> AuthenticatedUser:
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = find_user_by_id(
        user_id
    )

    if (
        not user
        or not user["is_active"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthenticatedUser(
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
        customer_id=user["customer_id"],
    )
