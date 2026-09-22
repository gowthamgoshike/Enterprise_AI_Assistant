"""Bounded async connection pool for read-only PostgreSQL access."""

from sqlalchemy.ext.asyncio import create_async_engine

from core.config import settings


async_engine = create_async_engine(
    settings.async_database_url,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout_seconds,
)
