CREATE TABLE IF NOT EXISTS background_jobs (
    job_id UUID PRIMARY KEY,
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(30) NOT NULL,
    requested_by VARCHAR(255) NOT NULL,
    request_id VARCHAR(255),
    parameters JSONB NOT NULL,
    result JSONB,
    error_code VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_background_job_status
        CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_background_jobs_requested_by_created_at
ON background_jobs(requested_by, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_background_jobs_status
ON background_jobs(status);
