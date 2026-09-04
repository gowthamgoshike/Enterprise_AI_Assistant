from services.transaction_service.repository import (
    get_transaction,
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