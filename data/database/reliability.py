"""Translate transient SQLAlchemy failures at the database boundary."""

from contextlib import contextmanager

from sqlalchemy.exc import OperationalError, TimeoutError as SQLAlchemyTimeoutError

from core.errors import DatabaseUnavailableError


@contextmanager
def translate_database_errors():
    try:
        yield
    except (OperationalError, SQLAlchemyTimeoutError) as exc:
        raise DatabaseUnavailableError() from exc
