from sqlalchemy import text


FIND_REFUND_BY_ACTION_ID = text("""
    SELECT
        refund_id,
        action_id,
        transaction_id,
        amount_cents,
        currency,
        status
    FROM refunds
    WHERE action_id = :action_id
    LIMIT 1
""")


FIND_TRANSACTION_FOR_UPDATE = text("""
    SELECT
        transaction_id,
        customer_id,
        product_id,
        amount_cents,
        currency,
        status,
        transaction_type,
        event_time
    FROM transactions
    WHERE transaction_id = :transaction_id
    FOR UPDATE
""")


INSERT_REFUND = text("""
    INSERT INTO refunds (
        refund_id,
        action_id,
        transaction_id,
        amount_cents,
        currency,
        reason,
        status,
        requested_by,
        approved_by,
        completed_at
    )
    VALUES (
        :refund_id,
        :action_id,
        :transaction_id,
        :amount_cents,
        :currency,
        :reason,
        'completed',
        :requested_by,
        :approved_by,
        NOW()
    )
    ON CONFLICT (action_id) DO NOTHING
""")


MARK_TRANSACTION_REFUNDED = text("""
    UPDATE transactions
    SET
        status = 'refunded',
        updated_at = NOW()
    WHERE transaction_id = :transaction_id
      AND status = 'completed'
""")


INSERT_AUDIT_LOG = text("""
    INSERT INTO audit_log (
        audit_id,
        request_id,
        action_id,
        user_id,
        action_type,
        resource_type,
        resource_id,
        decision,
        result,
        metadata
    )
    VALUES (
        :audit_id,
        :request_id,
        :action_id,
        :user_id,
        'refund_transaction',
        'transaction',
        :transaction_id,
        'approved',
        'completed',
        CAST(:metadata AS JSONB)
    )
""")


def find_refund_by_action_id(connection, action_id: str):
    return connection.execute(
        FIND_REFUND_BY_ACTION_ID,
        {"action_id": action_id},
    ).mappings().first()


def find_transaction_for_update(connection, transaction_id: str):
    return connection.execute(
        FIND_TRANSACTION_FOR_UPDATE,
        {"transaction_id": transaction_id},
    ).mappings().first()


def insert_refund(connection, values: dict) -> bool:
    return connection.execute(INSERT_REFUND, values).rowcount == 1


def mark_transaction_refunded(connection, transaction_id: str) -> int:
    return connection.execute(
        MARK_TRANSACTION_REFUNDED,
        {"transaction_id": transaction_id},
    ).rowcount


def insert_audit_log(connection, values: dict) -> None:
    connection.execute(INSERT_AUDIT_LOG, values)
