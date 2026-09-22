from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
)

from services.auth_service.models import UserRole


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthenticatedUser(BaseModel):
    user_id: str
    email: EmailStr
    role: UserRole
    customer_id: str | None = None
