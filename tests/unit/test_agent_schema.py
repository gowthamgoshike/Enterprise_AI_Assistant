import pytest
from pydantic import ValidationError

from ai.schemas.agent import (
    agent_decision_adapter,
)


def test_tool_decision_requires_tool_name_and_arguments():

    with pytest.raises(ValidationError):
        agent_decision_adapter.validate_python(
            {
                "action": "tool",
            }
        )


def test_respond_decision_rejects_tool_name():

    with pytest.raises(ValidationError):
        agent_decision_adapter.validate_python(
            {
                "action": "respond",
                "tool_name": "get_transaction",
            }
        )


def test_tool_decision_normalizes_transaction_id():

    decision = agent_decision_adapter.validate_python(
        {
            "action": "tool",
            "tool_name": "get_transaction",
            "arguments": {
                "transaction_id": " tx-123 ",
            },
        },
    )

    assert decision.arguments.transaction_id == "TX-123"