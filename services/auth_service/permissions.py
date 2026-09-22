from enum import Enum

from services.auth_service.models import UserRole


class Permission(str, Enum):
    READ_OWN_TRANSACTIONS = "read_own_transactions"
    READ_TRANSACTION = "read_transaction"
    READ_ANALYTICS = "read_analytics"
    READ_REVENUE = "read_revenue"
    PROPOSE_REFUND = "propose_refund"
    EXECUTE_REFUND = "execute_refund"
    GENERATE_ANALYTICS_REPORT = "generate_analytics_report"
    ADMIN_ACCESS = "admin_access"


ROLE_PERMISSIONS: dict[UserRole, set[Permission]] = {
    UserRole.CUSTOMER: {
        Permission.READ_OWN_TRANSACTIONS,
    },
    UserRole.ANALYST: {
        Permission.READ_TRANSACTION,
        Permission.READ_ANALYTICS,
        Permission.READ_REVENUE,
        Permission.GENERATE_ANALYTICS_REPORT,
    },
    UserRole.ADMIN: {
        Permission.READ_TRANSACTION,
        Permission.READ_ANALYTICS,
        Permission.READ_REVENUE,
        Permission.PROPOSE_REFUND,
        Permission.EXECUTE_REFUND,
        Permission.GENERATE_ANALYTICS_REPORT,
        Permission.ADMIN_ACCESS,
    },
}


def has_permission(
    role: UserRole,
    permission: Permission,
) -> bool:
    permissions = ROLE_PERMISSIONS.get(
        role,
        set(),
    )

    return permission in permissions
