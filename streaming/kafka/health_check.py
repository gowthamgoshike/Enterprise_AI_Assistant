from kafka import KafkaAdminClient

from streaming.kafka.config import (
    KAFKA_BOOTSTRAP_SERVERS,
)


def check_kafka():

    admin = KafkaAdminClient(
        bootstrap_servers=
            KAFKA_BOOTSTRAP_SERVERS,
    )

    print(
        "Kafka connection successful"
    )

    admin.close()


if __name__ == "__main__":
    check_kafka()