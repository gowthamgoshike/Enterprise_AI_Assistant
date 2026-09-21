from datetime import (
    datetime,
    timedelta,
    timezone,
)

import jwt
from core.config import settings


JWT_SECRET_KEY = settings.jwt_secret
JWT_ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_MINUTES = settings.jwt_access_token_expire_minutes


def create_access_token(
    user_id: str,
) -> str:
    now = datetime.now(
        timezone.utc
    )

    payload = {
        "sub": user_id,
        "iat": now,
        "exp": (
            now
            + timedelta(
                minutes=ACCESS_TOKEN_MINUTES
            )
        ),
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(
    token: str,
) -> dict:
    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[
            JWT_ALGORITHM
        ],
    )
