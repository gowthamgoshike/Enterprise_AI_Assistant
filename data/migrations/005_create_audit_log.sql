CREATE TABLE IF NOT EXISTS audit_log (
    audit_id UUID PRIMARY KEY,
    request_id VARCHAR(128),
    action_id UUID,
    user_id VARCHAR(64) NOT NULL,
    action_type VARCHAR(64) NOT NULL,
    resource_type VARCHAR(64) NOT NULL,
    resource_id VARCHAR(128) NOT NULL,
    decision VARCHAR(32),
    result VARCHAR(32) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
