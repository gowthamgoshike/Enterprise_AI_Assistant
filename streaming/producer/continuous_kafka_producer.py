import json
import logging
import time

from kafka import KafkaProducer

from streaming.kafka.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TRANSACTION_TOPIC,
)

from streaming.producer.generator import generate_transaction


EVENT_INTERVAL_SECONDS = 1


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,

    key_serializer=lambda key:
        key.encode("utf-8"),

    value_serializer=lambda value:
        json.dumps(value).encode("utf-8"),

    acks="all",

    retries=5,
)


def run():

    try:

        while True:

            transaction = generate_transaction()

            payload = transaction.model_dump(
                mode="json"
            )

            try:
                future = producer.send(
                    TRANSACTION_TOPIC,
                    key=transaction.customer_id,
                    value=payload,
                )

                metadata = future.get(
                    timeout=10
                )

            except Exception:
                logger.exception(
                    "failed_to_send_transaction",
                    extra={
                        "transaction_id": transaction.transaction_id,
                        "customer_id": transaction.customer_id,
                    },
                )
                time.sleep(EVENT_INTERVAL_SECONDS)
                continue

            logger.info(
                "transaction_sent",
                extra={
                    "transaction_id": transaction.transaction_id,
                    "customer_id": transaction.customer_id,
                    "partition": metadata.partition,
                    "offset": metadata.offset,
                },
            )

            time.sleep(
                EVENT_INTERVAL_SECONDS
            )

    except KeyboardInterrupt:

        print(
            "\nProducer stopped."
        )

    finally:

        producer.flush()
        producer.close()


if __name__ == "__main__":
    run()
