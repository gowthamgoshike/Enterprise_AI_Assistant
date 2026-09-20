import pytest

from services.auth_service.models import UserRole
from services.auth_service.permissions import Permission, has_permission


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (UserRole.CUSTOMER, False),
        (UserRole.ANALYST, False),
        (UserRole.ADMIN, True),
    ],
)
def test_refund_proposal_authorization(role, expected):
    assert has_permission(role, Permission.PROPOSE_REFUND) is expected
