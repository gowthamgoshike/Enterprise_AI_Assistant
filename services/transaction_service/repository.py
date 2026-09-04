from sqlalchemy import text

from data.database.connection import engine


UPSERT_TRANSACTION = text("""
    INSERT INTO transactions (
        transaction_id,
        latest_event_id,
        customer_id,
        product_id,
        amount_cents,
        currency,
        status,
        transaction_type,
        source,
        event_time
    )
    VALUES (
        :transaction_id,
        :latest_event_id,
        :customer_id,
        :product_id,
        :amount_cents,
        :currency,
        :status,
        :transaction_type,
        :source,
        :event_time
    )
    ON CONFLICT (transaction_id)
    DO UPDATE SET
        latest_event_id = EXCLUDED.latest_event_id,
        customer_id = EXCLUDED.customer_id,
        product_id = EXCLUDED.product_id,
        amount_cents = EXCLUDED.amount_cents,
        currency = EXCLUDED.currency,
        status = EXCLUDED.status,
        transaction_type = EXCLUDED.transaction_type,
        source = EXCLUDED.source,
        event_time = EXCLUDED.event_time,
        updated_at = NOW()
""")


GET_TRANSACTION = text("""
    SELECT
        transaction_id,
        customer_id,
        product_id,
        amount_cents,
        currency,
        status,
        transaction_type,
        source,
        event_time
    FROM transactions
    WHERE transaction_id = :transaction_id
""")


GET_CUSTOMER_TRANSACTIONS = text("""
    SELECT
        transaction_id,
        product_id,
        amount_cents,
        currency,
        status,
        event_time
    FROM transactions
    WHERE customer_id = :customer_id
    ORDER BY event_time DESC
    LIMIT :limit
""")


def upsert_transaction(transaction: dict) -> None:
    with engine.begin() as connection:
        connection.execute(UPSERT_TRANSACTION, transaction)


def get_transaction(transaction_id: str) -> dict | None:
    with engine.connect() as connection:
        result = connection.execute(
            GET_TRANSACTION,
            {"transaction_id": transaction_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None


def get_customer_transactions(
    customer_id: str,
    limit: int = 20,
) -> list[dict]:
    with engine.connect() as connection:
        result = connection.execute(
            GET_CUSTOMER_TRANSACTIONS,
            {
                "customer_id": customer_id,
                "limit": limit,
            },
        )
        return [dict(row) for row in result.mappings()]
