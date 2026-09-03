import time

from streaming.producer.generator import generate_transaction


def run():

    while True:

        transaction = generate_transaction()

        print(
            transaction.model_dump_json()
        )

        time.sleep(1)


if __name__ == "__main__":
    run()