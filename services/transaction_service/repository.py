from sqlalchemy import text

from data.database.connection import engine
from data.database.reliability import translate_database_errors
from core.observability.timing import observe_database_query


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


FIND_CUSTOMER_TRANSACTION = text("""
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
      AND customer_id = :customer_id
    LIMIT 1
""")


GET_TRANSACTIONS_BY_CUSTOMER = text("""
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
    WHERE customer_id = :customer_id
    ORDER BY event_time DESC
    LIMIT :limit
""")


FIND_FAILED_TRANSACTIONS = text("""
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
    WHERE status = 'failed'
      AND event_time >= NOW() - make_interval(days => :days)
    ORDER BY event_time DESC
""")


SET_LOCAL_STATEMENT_TIMEOUT = text("""
    SELECT set_config('statement_timeout', :timeout_ms, true)
""")


def upsert_transaction(transaction: dict) -> None:
    with observe_database_query("upsert_transaction"), translate_database_errors():
        with engine.begin() as connection:
            connection.execute(UPSERT_TRANSACTION, transaction)


def get_transaction(transaction_id: str) -> dict | None:
    with observe_database_query("get_transaction"), translate_database_errors():
        with engine.connect() as connection:
            result = connection.execute(
                GET_TRANSACTION,
                {"transaction_id": transaction_id},
            )
            row = result.mappings().first()
            return dict(row) if row else None


def find_customer_transaction(
    transaction_id: str,
    customer_id: str,
) -> dict | None:
    with (
        observe_database_query("find_customer_transaction"),
        translate_database_errors(),
    ):
        with engine.connect() as connection:
            result = connection.execute(
                FIND_CUSTOMER_TRANSACTION,
                {
                    "transaction_id": transaction_id,
                    "customer_id": customer_id,
                },
            )
            row = result.mappings().first()
            return dict(row) if row else None


def get_transactions_by_customer(
    customer_id: str,
    limit: int = 20,
) -> list[dict]:
    with (
        observe_database_query("get_transactions_by_customer"),
        translate_database_errors(),
    ):
        with engine.connect() as connection:
            result = connection.execute(
                GET_TRANSACTIONS_BY_CUSTOMER,
                {
                    "customer_id": customer_id,
                    "limit": limit,
                },
            )
            return [dict(row) for row in result.mappings()]


def find_failed_transactions(
    days: int,
    *,
    timeout_ms: int = 120_000,
) -> list[dict]:
    with (
        observe_database_query("find_failed_transactions"),
        translate_database_errors(),
        engine.begin() as connection,
    ):
        connection.execute(
            SET_LOCAL_STATEMENT_TIMEOUT,
            {"timeout_ms": str(timeout_ms)},
        )
        result = connection.execute(
            FIND_FAILED_TRANSACTIONS,
            {"days": days},
        )
        return [dict(row) for row in result.mappings()]


# Backwards-compatible name for callers written before the scoped tool flow.
get_customer_transactions = get_transactions_by_customer
