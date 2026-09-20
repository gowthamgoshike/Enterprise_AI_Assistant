from sqlalchemy import text

from data.database.connection import engine
from data.database.reliability import translate_database_errors
from core.observability.timing import observe_database_query


FIND_USER_BY_EMAIL = text("""
    SELECT
        user_id,
        email,
        password_hash,
        role,
        customer_id,
        is_active
    FROM users
    WHERE email = :email
    LIMIT 1
""")


FIND_USER_BY_ID = text("""
    SELECT
        user_id,
        email,
        role,
        customer_id,
        is_active
    FROM users
    WHERE user_id = :user_id
    LIMIT 1
""")


def find_user_by_email(
    email: str,
):
    with (
        observe_database_query("find_user_by_email"),
        translate_database_errors(),
        engine.connect() as connection,
    ):
        result = connection.execute(
            FIND_USER_BY_EMAIL,
            {
                "email": email.lower(),
            },
        ).mappings().first()

    return result


def find_user_by_id(
    user_id: str,
):
    with (
        observe_database_query("find_user_by_id"),
        translate_database_errors(),
        engine.connect() as connection,
    ):
        return connection.execute(
            FIND_USER_BY_ID,
            {
                "user_id": user_id,
            },
        ).mappings().first()
