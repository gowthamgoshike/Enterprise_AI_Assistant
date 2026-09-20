"""Translate transient SQLAlchemy failures at the database boundary."""

from contextlib import contextmanager

import psycopg
import psycopg2
from psycopg_pool import PoolClosed, PoolTimeout
from sqlalchemy.exc import OperationalError, TimeoutError as SQLAlchemyTimeoutError

from core.errors import DatabaseUnavailableError


DATABASE_UNAVAILABLE_EXCEPTIONS = (
    DatabaseUnavailableError,
    OperationalError,
    SQLAlchemyTimeoutError,
    psycopg.OperationalError,
    psycopg2.OperationalError,
    PoolClosed,
    PoolTimeout,
)


def is_database_unavailable_error(error: BaseException) -> bool:
    """Inspect wrapped/aggregated exceptions without matching error strings."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, DATABASE_UNAVAILABLE_EXCEPTIONS):
            return True
        pending.extend(getattr(current, "exceptions", ()))
        cause = current.__cause__ or current.__context__
        if cause is not None:
            pending.append(cause)
    return False


@contextmanager
def translate_database_errors():
    try:
        yield
    except (OperationalError, SQLAlchemyTimeoutError) as exc:
        raise DatabaseUnavailableError() from exc
