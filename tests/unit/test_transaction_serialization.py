import json

from streaming.producer.generator import (
    generate_transaction,
)

from data.schemas.transaction import (
    TransactionEvent,
)


def test_transaction_json_round_trip():

    original = generate_transaction()

    json_data = (
        original.model_dump_json()
    )

    dictionary = json.loads(
        json_data
    )

    reconstructed = (
        TransactionEvent.model_validate(
            dictionary
        )
    )

    assert (
        reconstructed.transaction_id
        ==
        original.transaction_id
    )

    assert (
        reconstructed.customer_id
        ==
        original.customer_id
    )