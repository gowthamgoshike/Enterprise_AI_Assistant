from dataclasses import dataclass

from services.auth_service.models import UserRole


@dataclass(frozen=True)
class ToolExecutionContext:
    user_id: str
    role: UserRole
    customer_id: str | None
    request_id: str
