"""Async transaction-read service; writes remain synchronous and transactional."""

from services.transaction_service.async_repository import (
    find_customer_transaction_async as find_scoped_customer_transaction_async,
    get_transaction_async,
)


async def find_transaction_async(transaction_id: str) -> dict:
    transaction = await get_transaction_async(transaction_id)
    return {"found": transaction is not None, "transaction": transaction}


async def find_customer_transaction_async(
    transaction_id: str,
    customer_id: str,
) -> dict:
    transaction = await find_scoped_customer_transaction_async(
        transaction_id=transaction_id,
        customer_id=customer_id,
    )
    return {"found": transaction is not None, "transaction": transaction}
