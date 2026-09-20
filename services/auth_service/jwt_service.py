import os

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import jwt
from dotenv import load_dotenv


load_dotenv()


JWT_SECRET_KEY = os.environ[
    "JWT_SECRET_KEY"
]

JWT_ALGORITHM = os.getenv(
    "JWT_ALGORITHM",
    "HS256",
)

ACCESS_TOKEN_MINUTES = int(
    os.getenv(
        "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
        "30",
    )
)


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
