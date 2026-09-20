from services.auth_service.password_service import verify_password
from services.auth_service.repository import find_user_by_email


def authenticate_user(
    email: str,
    password: str,
):
    user = find_user_by_email(
        email
    )

    if not user:
        return None

    if not user["is_active"]:
        return None

    if not verify_password(
        password,
        user["password_hash"],
    ):
        return None

    return user
