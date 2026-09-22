"""Async, parameterized transaction reads against the existing PostgreSQL DB."""

from core.async_database import async_engine
from core.observability.timing import observe_database_query
from data.database.reliability import translate_database_errors
from services.transaction_service.repository import (
    FIND_CUSTOMER_TRANSACTION,
    GET_TRANSACTION,
)


async def get_transaction_async(transaction_id: str) -> dict | None:
    with observe_database_query("get_transaction_async"), translate_database_errors():
        async with async_engine.connect() as connection:
            result = await connection.execute(
                GET_TRANSACTION,
                {"transaction_id": transaction_id},
            )
            row = result.mappings().first()
            return dict(row) if row else None


async def find_customer_transaction_async(
    transaction_id: str,
    customer_id: str,
) -> dict | None:
    with (
        observe_database_query("find_customer_transaction_async"),
        translate_database_errors(),
    ):
        async with async_engine.connect() as connection:
            result = await connection.execute(
                FIND_CUSTOMER_TRANSACTION,
                {
                    "transaction_id": transaction_id,
                    "customer_id": customer_id,
                },
            )
            row = result.mappings().first()
            return dict(row) if row else None
