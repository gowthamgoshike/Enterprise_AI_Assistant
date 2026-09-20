import logging
import time
import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from apps.api.dependencies.auth import get_current_user
from apps.api.schemas.auth import (
    AuthenticatedUser,
    LoginRequest,
    TokenResponse,
)
from services.auth_service.auth_service import authenticate_user
from services.auth_service.jwt_service import create_access_token


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
):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    user = authenticate_user(
        email=request.email,
        password=request.password,
    )

    if not user:
        logger.info(
            "auth_login request_id=%s endpoint=/api/v1/auth/login "
            "result=failure latency=%.3f",
            request_id,
            time.perf_counter() - start,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token(
        user_id=user["user_id"],
    )
    logger.info(
        "auth_login request_id=%s user_id=%s endpoint=/api/v1/auth/login "
        "result=success latency=%.3f",
        request_id,
        user["user_id"],
        time.perf_counter() - start,
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
