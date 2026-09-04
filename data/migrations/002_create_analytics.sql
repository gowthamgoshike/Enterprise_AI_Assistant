CREATE TABLE IF NOT EXISTS product_revenue (

    product_id VARCHAR(100) PRIMARY KEY,

    total_revenue NUMERIC(18, 2) NOT NULL,

    transaction_count BIGINT NOT NULL,

    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transaction_metrics (

    status VARCHAR(30) PRIMARY KEY,

    transaction_count BIGINT NOT NULL,

    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS customer_daily_spend (

    customer_id VARCHAR(100) NOT NULL,

    transaction_date DATE NOT NULL,

    daily_spend NUMERIC(18, 2) NOT NULL,

    PRIMARY KEY (
        customer_id,
        transaction_date
    )
);

CREATE INDEX IF NOT EXISTS
idx_customer_daily_spend_date
ON customer_daily_spend(
    transaction_date
);