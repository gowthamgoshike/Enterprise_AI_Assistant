from contextlib import nullcontext
from unittest.mock import MagicMock

from services.analytics_service import repository


def test_transaction_metrics_are_aggregated_from_current_transactions(
    monkeypatch,
):
    rows = [
        {"status": "completed", "transaction_count": 8},
        {"status": "failed", "transaction_count": 3},
    ]
    result = MagicMock()
    result.mappings.return_value = rows
    connection = MagicMock()
    connection.execute.return_value = result
    monkeypatch.setattr(
        repository.engine,
        "connect",
        lambda: nullcontext(connection),
    )

    assert repository.get_transaction_metrics() == rows

    query = str(connection.execute.call_args.args[0])
    assert "FROM transactions" in query
    assert "COUNT(*) AS transaction_count" in query
    assert "GROUP BY status" in query
    assert "FROM transaction_metrics" not in query
