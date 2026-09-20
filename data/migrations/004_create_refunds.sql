CREATE TABLE IF NOT EXISTS refunds (
    refund_id UUID PRIMARY KEY,
    action_id UUID NOT NULL UNIQUE,
    transaction_id VARCHAR(64) NOT NULL,
    amount_cents BIGINT NOT NULL,
    currency VARCHAR(8) NOT NULL,
    reason TEXT,
    status VARCHAR(32) NOT NULL,
    requested_by VARCHAR(64) NOT NULL,
    approved_by VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    CONSTRAINT fk_refund_transaction
        FOREIGN KEY (transaction_id)
        REFERENCES transactions(transaction_id)
);
