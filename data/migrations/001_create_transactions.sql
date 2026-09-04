CREATE TABLE IF NOT EXISTS transactions (

    transaction_id VARCHAR(100) PRIMARY KEY,

    latest_event_id VARCHAR(100) NOT NULL,

    customer_id VARCHAR(100) NOT NULL,

    product_id VARCHAR(100) NOT NULL,

    amount_cents BIGINT NOT NULL,

    currency VARCHAR(10) NOT NULL,

    status VARCHAR(30) NOT NULL,

    transaction_type VARCHAR(30) NOT NULL,

    source VARCHAR(30) NOT NULL,

    event_time TIMESTAMPTZ NOT NULL,

    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CHECK (amount_cents > 0),

    CONSTRAINT valid_transaction_status
    CHECK (
        status IN (
            'pending',
            'completed',
            'failed',
            'refunded'
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_transactions_customer_id
ON transactions(customer_id);

CREATE INDEX IF NOT EXISTS idx_transactions_event_time
ON transactions(event_time);

CREATE INDEX IF NOT EXISTS idx_transactions_customer_time
ON transactions(
    customer_id,
    event_time DESC
);

CREATE INDEX IF NOT EXISTS idx_transactions_status
ON transactions(status);

