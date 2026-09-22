"""Central settings behavior without relying on the developer's .env file."""

import pytest
from pydantic import ValidationError

from core.config import Settings, get_settings, settings


def test_settings_are_cached_and_hide_secrets():
    assert get_settings() is settings
    configured = Settings(
        _env_file=None,
        DATABASE_URL="postgresql+psycopg2://user:password@localhost/test",
        JWT_SECRET_KEY="test-secret",
    )
    assert configured.database_url.endswith("/test")
    assert configured.async_database_url == (
        "postgresql+asyncpg://user:password@localhost/test"
    )
    assert configured.jwt_secret == "test-secret"
    assert "test-secret" not in repr(configured)
    assert "password" not in repr(configured)


def test_legacy_database_parts_and_jwt_name_work():
    configured = Settings(
        _env_file=None,
        DATABASE_USER="user",
        DATABASE_PASSWORD="p@ss word",
        DATABASE_NAME="test",
        JWT_SECRET_KEY="test-secret",
    )
    assert configured.database_url == (
        "postgresql+psycopg2://user:p%40ss%20word@localhost:5432/test"
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MODEL_TIMEOUT_SECONDS", 0),
        ("MODEL_MAX_RETRIES", 6),
        ("MODEL_MAX_CONCURRENCY", 9),
        ("MAX_TOOL_CALLS_PER_REQUEST", 21),
        ("DB_POOL_SIZE", 0),
        ("DB_MAX_OVERFLOW", 51),
        ("DB_POOL_TIMEOUT_SECONDS", 0),
        ("APP_ENV", "unknown"),
    ],
)
def test_invalid_settings_are_rejected(name, value):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            DATABASE_URL="postgresql+psycopg2://user:password@localhost/test",
            JWT_SECRET_KEY="test-secret",
            **{name: value},
        )
