from services.transaction_service.repository import get_transaction


def test_transaction_lookup():
    transaction = get_transaction("TX-TEST")

    if transaction:
        assert transaction["transaction_id"] == "TX-TEST"
