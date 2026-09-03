import json
from pydantic import ValidationError

from data.schemas.transaction import (
    TransactionEvent,
)
from kafka import KafkaConsumer

from streaming.kafka.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TRANSACTION_TOPIC,
)


consumer = KafkaConsumer(
    TRANSACTION_TOPIC,

    bootstrap_servers=
        KAFKA_BOOTSTRAP_SERVERS,

    group_id=
        "transaction-debug-consumer",

    auto_offset_reset=
        "earliest",

    enable_auto_commit=
        True,

    key_deserializer=
        lambda key:
        key.decode("utf-8")
        if key else None,

    value_deserializer=
        lambda value:
        json.loads(
            value.decode("utf-8")
        ),
)


print(
    "Waiting for transactions..."
)


for message in consumer:
    try:
        transaction = TransactionEvent.model_validate(
            message.value
        )

        print(transaction)

    except ValidationError as exc:
        print("Invalid transaction:")
        print(exc)
