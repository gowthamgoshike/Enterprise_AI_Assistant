from pydantic import (
    BaseModel,
    EmailStr,
)

from services.auth_service.models import UserRole


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthenticatedUser(BaseModel):
    user_id: str
    email: EmailStr
    role: UserRole
    customer_id: str | None = None
