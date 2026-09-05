import pytest
from pydantic import ValidationError

from ai.schemas.agent import (
    agent_decision_adapter,
)


def test_respond_decision():

    decision = agent_decision_adapter.validate_python(
        {
            "action": "respond",
        }
    )

    assert decision.action == "respond"


def test_transaction_decision():

    decision = agent_decision_adapter.validate_python(
        {
            "action": "tool",
            "tool_name": "get_transaction",
            "arguments": {
                "transaction_id": "TX-123",
            },
        }
    )

    assert decision.arguments.transaction_id == "TX-123"


def test_unknown_tool_rejected():

    with pytest.raises(ValidationError):
        agent_decision_adapter.validate_python(
            {
                "action": "tool",
                "tool_name": "delete_database",
                "arguments": {},
            }
        )


def test_missing_arguments_rejected():

    with pytest.raises(ValidationError):
        agent_decision_adapter.validate_python(
            {
                "action": "tool",
                "tool_name": "get_transaction",
            }
        )


def test_long_transaction_id_rejected():

    with pytest.raises(ValidationError):
        agent_decision_adapter.validate_python(
            {
                "action": "tool",
                "tool_name": "get_transaction",
                "arguments": {
                    "transaction_id": "X" * 500,
                },
            }
        )
