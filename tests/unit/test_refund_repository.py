from pathlib import Path

from services.transaction_service import refund_repository


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_transaction_lookup_uses_row_lock():
    assert "FOR UPDATE" in str(
        refund_repository.FIND_TRANSACTION_FOR_UPDATE
    )


def test_refund_insert_has_database_conflict_guard():
    assert "ON CONFLICT (action_id) DO NOTHING" in str(
        refund_repository.INSERT_REFUND
    )


def test_refund_and_audit_migrations_enforce_required_contracts():
    refunds_sql = (
        PROJECT_ROOT / "data/migrations/004_create_refunds.sql"
    ).read_text()
    audit_sql = (
        PROJECT_ROOT / "data/migrations/005_create_audit_log.sql"
    ).read_text()

    assert "action_id UUID NOT NULL UNIQUE" in refunds_sql
    assert "REFERENCES transactions(transaction_id)" in refunds_sql
    assert "metadata JSONB" in audit_sql
