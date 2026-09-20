from services.transaction_service.repository import (
    find_customer_transaction as find_scoped_customer_transaction,
    get_transaction,
    get_transactions_by_customer,
)


def find_transaction(
    transaction_id: str
):

    transaction = get_transaction(
        transaction_id
    )

    if transaction is None:

        return {
            "found": False,
            "transaction": None,
        }

    return {
        "found": True,
        "transaction": transaction,
    }


def fetch_customer_transactions(
    customer_id: str,
    limit: int = 20,
) -> dict:
    transactions = get_transactions_by_customer(
        customer_id=customer_id,
        limit=limit,
    )

    return {
        "customer_id": customer_id,
        "count": len(transactions),
        "transactions": transactions,
    }


def find_customer_transaction(
    transaction_id: str,
    customer_id: str,
) -> dict:
    transaction = find_scoped_customer_transaction(
        transaction_id=transaction_id,
        customer_id=customer_id,
    )

    if transaction is None:
        return {
            "found": False,
            "transaction": None,
        }

    return {
        "found": True,
        "transaction": transaction,
    }
