CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64)
        PRIMARY KEY,

    email VARCHAR(255)
        UNIQUE NOT NULL,

    password_hash TEXT
        NOT NULL,

    role VARCHAR(32)
        NOT NULL,

    customer_id VARCHAR(64),

    is_active BOOLEAN
        NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ
        NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_user_role
        CHECK (
            role IN (
                'CUSTOMER',
                'ANALYST',
                'ADMIN'
            )
        )
);
