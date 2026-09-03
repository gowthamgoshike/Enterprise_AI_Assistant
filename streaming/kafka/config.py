import os

from dotenv import load_dotenv


load_dotenv()


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)


TRANSACTION_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "transactions",
)