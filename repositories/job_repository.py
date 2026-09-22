import json
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text

from core.observability.timing import observe_database_query
from data.database.connection import engine
from data.database.reliability import translate_database_errors
from services.job_service.models import JobStage, JobType


CREATE_JOB = text("""
    INSERT INTO background_jobs (
        job_id,
        job_type,
        status,
        requested_by,
        request_id,
        parameters
    )
    VALUES (
        :job_id,
        :job_type,
        'queued',
        :requested_by,
        :request_id,
        CAST(:parameters AS JSONB)
    )
""")


AUDIT_JOB_CREATION = text("""
    INSERT INTO audit_log (
        audit_id,
        request_id,
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
        :requested_by,
        'create_background_job',
        'background_job',
        :job_id,
        'allowed',
        'queued',
        CAST(:metadata AS JSONB)
    )
""")


GET_JOB = text("""
    SELECT
        job_id,
        job_type,
        status,
        stage,
        requested_by,
        request_id,
        parameters,
        result,
        error_code,
        created_at,
        started_at,
        completed_at,
        updated_at
    FROM background_jobs
    WHERE job_id = :job_id
    LIMIT 1
""")


GET_JOB_FOR_USER = text("""
    SELECT
        job_id,
        job_type,
        status,
        stage,
        requested_by,
        request_id,
        parameters,
        result,
        error_code,
        created_at,
        started_at,
        completed_at,
        updated_at
    FROM background_jobs
    WHERE job_id = :job_id
      AND requested_by = :requested_by
    LIMIT 1
""")


MARK_RUNNING = text("""
    UPDATE background_jobs
    SET
        status = 'running',
        stage = 'querying_transactions',
        started_at = NOW(),
        updated_at = NOW()
    WHERE job_id = :job_id
      AND status = 'queued'
""")


MARK_COMPLETED = text("""
    UPDATE background_jobs
    SET
        status = 'completed',
        stage = 'completed',
        result = CAST(:result AS JSONB),
        error_code = NULL,
        completed_at = NOW(),
        updated_at = NOW()
    WHERE job_id = :job_id
      AND status = 'running'
""")


MARK_FAILED = text("""
    UPDATE background_jobs
    SET
        status = 'failed',
        stage = 'failed',
        result = NULL,
        error_code = :error_code,
        completed_at = NOW(),
        updated_at = NOW()
    WHERE job_id = :job_id
      AND status = 'running'
""")


UPDATE_STAGE = text("""
    UPDATE background_jobs
    SET
        stage = :stage,
        updated_at = NOW()
    WHERE job_id = :job_id
      AND status = 'running'
""")


CANCEL_QUEUED_JOB_FOR_USER = text("""
    UPDATE background_jobs
    SET
        status = 'cancelled',
        stage = 'cancelled',
        completed_at = NOW(),
        updated_at = NOW()
    WHERE job_id = :job_id
      AND requested_by = :requested_by
      AND status = 'queued'
""")


def create_job(
    *,
    job_id: UUID,
    job_type: JobType,
    requested_by: str,
    request_id: str | None,
    parameters: dict[str, Any],
) -> None:
    with (
        observe_database_query("create_background_job"),
        translate_database_errors(),
        engine.begin() as connection,
    ):
        values = {
            "job_id": str(job_id),
            "job_type": job_type.value,
            "requested_by": requested_by,
            "request_id": request_id,
            "parameters": json.dumps(parameters),
        }
        connection.execute(CREATE_JOB, values)
        connection.execute(AUDIT_JOB_CREATION, {
            **values,
            "audit_id": str(uuid4()),
            "metadata": json.dumps({
                "job_type": job_type.value,
                "parameters": parameters,
                "parameters_classification": "non_sensitive",
            }),
        })


def get_job(job_id: UUID) -> dict[str, Any] | None:
    with (
        observe_database_query("get_background_job"),
        translate_database_errors(),
        engine.connect() as connection,
    ):
        row = connection.execute(
            GET_JOB,
            {"job_id": str(job_id)},
        ).mappings().first()
        return dict(row) if row else None


def get_job_for_user(
    job_id: UUID,
    *,
    requested_by: str,
) -> dict[str, Any] | None:
    with (
        observe_database_query("get_background_job_for_user"),
        translate_database_errors(),
        engine.connect() as connection,
    ):
        row = connection.execute(
            GET_JOB_FOR_USER,
            {
                "job_id": str(job_id),
                "requested_by": requested_by,
            },
        ).mappings().first()
        return dict(row) if row else None


def mark_running(job_id: UUID) -> bool:
    with (
        observe_database_query("mark_background_job_running"),
        translate_database_errors(),
        engine.begin() as connection,
    ):
        result = connection.execute(MARK_RUNNING, {"job_id": str(job_id)})
        return result.rowcount == 1


def mark_completed(job_id: UUID, *, result: dict[str, Any]) -> bool:
    with (
        observe_database_query("mark_background_job_completed"),
        translate_database_errors(),
        engine.begin() as connection,
    ):
        update = connection.execute(MARK_COMPLETED, {
            "job_id": str(job_id),
            "result": json.dumps(result),
        })
        return update.rowcount == 1


def mark_failed(job_id: UUID, *, error_code: str) -> bool:
    with (
        observe_database_query("mark_background_job_failed"),
        translate_database_errors(),
        engine.begin() as connection,
    ):
        update = connection.execute(MARK_FAILED, {
            "job_id": str(job_id),
            "error_code": error_code,
        })
        return update.rowcount == 1


def update_stage(job_id: UUID, stage: JobStage) -> bool:
    with (
        observe_database_query("update_background_job_stage"),
        translate_database_errors(),
        engine.begin() as connection,
    ):
        update = connection.execute(UPDATE_STAGE, {
            "job_id": str(job_id),
            "stage": stage.value,
        })
        return update.rowcount == 1


def cancel_queued_job_for_user(job_id: UUID, *, requested_by: str) -> bool:
    with (
        observe_database_query("cancel_queued_background_job"),
        translate_database_errors(),
        engine.begin() as connection,
    ):
        update = connection.execute(CANCEL_QUEUED_JOB_FOR_USER, {
            "job_id": str(job_id),
            "requested_by": requested_by,
        })
        return update.rowcount == 1
