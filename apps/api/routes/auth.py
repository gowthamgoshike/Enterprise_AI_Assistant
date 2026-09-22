import asyncio
import logging
import time
import uuid

from fastapi import (
    APIRouter,
    Depends,
)

from apps.api.dependencies.auth import get_current_user
from apps.api.dependencies.rate_limit import rate_limit_login
from apps.api.schemas.auth import (
    AuthenticatedUser,
    LoginRequest,
    TokenResponse,
)
from services.auth_service.auth_service import authenticate_user
from services.auth_service.jwt_service import create_access_token
from core.observability.context import get_request_id, set_user_context
from core.errors import InvalidCredentialsError


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
)


@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(
    request: LoginRequest,
    _: None = Depends(rate_limit_login),
):
    request_id = get_request_id() or f"REQ-{uuid.uuid4()}"
    start = time.perf_counter()
    user = await asyncio.to_thread(
        authenticate_user,
        email=request.email,
        password=request.password,
    )

    if not user:
        logger.info(
            "auth_login request_id=%s endpoint=/api/v1/auth/login "
            "result=failure latency=%.3f",
            request_id,
            time.perf_counter() - start,
            extra={
                "event": "auth_login",
                "status": "failure",
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        raise InvalidCredentialsError()

    access_token = create_access_token(
        user_id=user["user_id"],
    )
    set_user_context(user["user_id"])
    logger.info(
        "auth_login request_id=%s user_id=%s endpoint=/api/v1/auth/login "
        "result=success latency=%.3f",
        request_id,
        user["user_id"],
        time.perf_counter() - start,
        extra={
            "event": "auth_login",
            "status": "success",
            "role": (
                user["role"].value
                if hasattr(user.get("role"), "value")
                else user.get("role")
            ),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    )

    return TokenResponse(
        access_token=access_token
    )


@router.get(
    "/me",
    response_model=AuthenticatedUser,
)
async def me(
    current_user: AuthenticatedUser = Depends(
        get_current_user
    ),
):
    return current_user
