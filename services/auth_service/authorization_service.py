from services.auth_service.exceptions import AuthorizationError
from services.auth_service.models import UserRole
from services.auth_service.permissions import Permission, has_permission


def require_permission(
    role: UserRole,
    permission: Permission,
) -> None:
    if not has_permission(
        role,
        permission,
    ):
        raise AuthorizationError(
            "Permission denied"
        )
