import pytest
from pydantic import ValidationError

from ai.tools import transaction_tools


def test_get_transaction_normalizes_transaction_id(monkeypatch):

    seen = {}

    def fake_find_transaction(transaction_id):
        seen["transaction_id"] = transaction_id
        return {
            "found": True,
            "transaction": {
                "transaction_id": transaction_id,
            },
        }

    monkeypatch.setattr(
        transaction_tools,
        "find_transaction",
        fake_find_transaction,
    )

    result = transaction_tools.get_transaction_tool(
        "  tx-123  "
    )

    assert seen["transaction_id"] == "TX-123"
    assert result["transaction"]["transaction_id"] == "TX-123"


def test_get_transaction_rejects_blank_normalized_id():

    with pytest.raises(ValidationError):
        transaction_tools.get_transaction_tool(
            "   "
        )