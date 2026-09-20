import pytest

from services.auth_service.authorization_service import require_permission
from services.auth_service.exceptions import AuthorizationError
from services.auth_service.models import UserRole
from services.auth_service.permissions import (
    ROLE_PERMISSIONS,
    Permission,
    has_permission,
)


def test_customer_can_read_only_own_transactions():
    assert ROLE_PERMISSIONS[UserRole.CUSTOMER] == {
        Permission.READ_OWN_TRANSACTIONS,
    }


def test_customer_can_read_own():
    assert has_permission(
        UserRole.CUSTOMER,
        Permission.READ_OWN_TRANSACTIONS,
    )


def test_analyst_has_read_permissions_without_admin_access():
    assert ROLE_PERMISSIONS[UserRole.ANALYST] == {
        Permission.READ_TRANSACTION,
        Permission.READ_ANALYTICS,
        Permission.READ_REVENUE,
    }
    assert not has_permission(
        UserRole.ANALYST,
        Permission.ADMIN_ACCESS,
    )


def test_admin_has_analytics_revenue_and_admin_permissions():
    assert ROLE_PERMISSIONS[UserRole.ADMIN] == {
        Permission.READ_TRANSACTION,
        Permission.READ_ANALYTICS,
        Permission.READ_REVENUE,
        Permission.ADMIN_ACCESS,
    }


def test_analyst_can_read_revenue():
    assert has_permission(
        UserRole.ANALYST,
        Permission.READ_REVENUE,
    )


def test_customer_cannot_read_revenue():
    assert not has_permission(
        UserRole.CUSTOMER,
        Permission.READ_REVENUE,
    )


def test_require_permission_allows_authorized_role():
    require_permission(
        UserRole.ANALYST,
        Permission.READ_ANALYTICS,
    )


def test_require_permission_raises_authorization_error():
    with pytest.raises(
        AuthorizationError,
        match="Permission denied",
    ):
        require_permission(
            UserRole.CUSTOMER,
            Permission.READ_REVENUE,
        )
