ALTER TABLE background_jobs
ADD COLUMN IF NOT EXISTS stage VARCHAR(100);

UPDATE background_jobs
SET stage = CASE status
    WHEN 'queued' THEN 'queued'
    WHEN 'running' THEN 'querying_transactions'
    WHEN 'completed' THEN 'completed'
    WHEN 'failed' THEN 'failed'
    WHEN 'cancelled' THEN 'cancelled'
    ELSE 'queued'
END
WHERE stage IS NULL;

ALTER TABLE background_jobs
ALTER COLUMN stage SET DEFAULT 'queued';

ALTER TABLE background_jobs
ALTER COLUMN stage SET NOT NULL;
