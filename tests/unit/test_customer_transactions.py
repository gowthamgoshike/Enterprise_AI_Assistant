from unittest.mock import Mock

from services.transaction_service import repository, service


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def __iter__(self):
        return iter(self.rows)

    def first(self):
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.execute = Mock(side_effect=self._execute)

    def _execute(self, query, parameters):
        return FakeResult(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def test_customer_transaction_query_is_scoped_and_parameterized(monkeypatch):
    rows = [{
        "transaction_id": "TX-1",
        "customer_id": "CUST-1",
        "transaction_type": "purchase",
        "source": "web",
    }]
    connection = FakeConnection(rows)
    monkeypatch.setattr(
        repository.engine,
        "connect",
        lambda: connection,
    )

    result = repository.get_transactions_by_customer(
        customer_id="CUST-1",
        limit=7,
    )

    assert result == rows
    query, parameters = connection.execute.call_args.args
    assert "WHERE customer_id = :customer_id" in str(query)
    assert "LIMIT :limit" in str(query)
    assert parameters == {"customer_id": "CUST-1", "limit": 7}


def test_customer_transaction_service_returns_scoped_envelope(monkeypatch):
    transactions = [{"transaction_id": "TX-1"}]
    lookup = Mock(return_value=transactions)
    monkeypatch.setattr(
        service,
        "get_transactions_by_customer",
        lookup,
    )

    result = service.fetch_customer_transactions(
        customer_id="CUST-1",
        limit=5,
    )

    assert result == {
        "customer_id": "CUST-1",
        "count": 1,
        "transactions": transactions,
    }
    lookup.assert_called_once_with(
        customer_id="CUST-1",
        limit=5,
    )


def test_specific_transaction_query_enforces_customer_ownership(monkeypatch):
    row = {
        "transaction_id": "TX-A",
        "customer_id": "CUST-A",
    }
    connection = FakeConnection([row])
    monkeypatch.setattr(
        repository.engine,
        "connect",
        lambda: connection,
    )

    result = repository.find_customer_transaction(
        transaction_id="TX-A",
        customer_id="CUST-A",
    )

    assert result == row
    query, parameters = connection.execute.call_args.args
    assert "transaction_id = :transaction_id" in str(query)
    assert "customer_id = :customer_id" in str(query)
    assert "LIMIT 1" in str(query)
    assert parameters == {
        "transaction_id": "TX-A",
        "customer_id": "CUST-A",
    }
