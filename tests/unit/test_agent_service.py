import asyncio
import pytest

from ai.exceptions import (
    AgentDecisionError,
)
from ai.tools.exceptions import (
    ToolExecutionError,
)
from ai.schemas.agent import (
    RespondDecision,
    TransactionToolDecision,
)
from apps.api.services import agent_service


def test_invalid_model_json_retries_then_raises(monkeypatch):

    calls = 0

    async def fake_chat_with_model(*args, **kwargs):
        nonlocal calls
        calls += 1
        return "not-json"

    monkeypatch.setattr(
        agent_service,
        "chat_with_model",
        fake_chat_with_model,
    )

    with pytest.raises(AgentDecisionError):
        asyncio.run(
            agent_service.decide_action(
                "What is the status of TX-123?"
            )
        )

    assert calls == 2


def test_invalid_decision_is_repaired_on_second_attempt(monkeypatch):

    responses = iter(
        [
            '{"action":"respond","tool_name":"get_transaction"}',
            '{"action":"respond"}',
        ]
    )
    captured_messages = []

    async def fake_chat_with_model(messages, **kwargs):
        captured_messages.append(messages)
        return next(responses)

    monkeypatch.setattr(
        agent_service,
        "chat_with_model",
        fake_chat_with_model,
    )

    decision = asyncio.run(
        agent_service.decide_action(
            "Explain transaction monitoring."
        )
    )

    assert decision.action == "respond"
    assert len(captured_messages) == 2
    assert "did not match" in captured_messages[1][-1]["content"]


def test_unknown_tool_is_rejected():

    async def execute():
        return await agent_service.execute_tool(
            "delete_database",
            {},
        )

    try:
        asyncio.run(execute())
    except ValueError as exc:
        assert str(exc) == "Unknown tool requested"
    else:
        raise AssertionError("Unknown tools must be rejected")


def test_allowlisted_tool_is_executed(monkeypatch):

    def fake_tool(transaction_id: str):
        return {
            "transaction_id": transaction_id,
            "status": "completed",
        }

    monkeypatch.setitem(
        agent_service.TOOL_REGISTRY,
        "test_tool",
        fake_tool,
    )

    result = asyncio.run(
        agent_service.execute_tool(
            "test_tool",
            {"transaction_id": "TX-123"},
        )
    )

    assert result == {
        "transaction_id": "TX-123",
        "status": "completed",
    }


def test_tool_execution_logs_request_id_without_result(
    monkeypatch,
    caplog,
):

    def fake_tool():
        return {
            "internal_risk_score": "sensitive-value",
        }

    monkeypatch.setitem(
        agent_service.TOOL_REGISTRY,
        "test_tool",
        fake_tool,
    )

    with caplog.at_level("INFO"):

        result = asyncio.run(
            agent_service.execute_tool(
                "test_tool",
                {},
                "request-123",
            )
        )

    assert result["internal_risk_score"] == "sensitive-value"
    assert "request_id=request-123" in caplog.text
    assert "tool=test_tool" in caplog.text
    assert "latency=" in caplog.text
    assert "sensitive-value" not in caplog.text


def test_process_user_request_without_tool(monkeypatch):

    async def fake_decide_action(user_message):
        return RespondDecision(action="respond")

    async def fake_chat_with_model(messages, **kwargs):
        assert messages == [
            {
                "role": "user",
                "content": "Hello",
            }
        ]
        return "direct response"

    monkeypatch.setattr(
        agent_service,
        "decide_action",
        fake_decide_action,
    )
    monkeypatch.setattr(
        agent_service,
        "chat_with_model",
        fake_chat_with_model,
    )

    result = asyncio.run(
        agent_service.process_user_request(
            "Hello"
        )
    )

    assert result.response == "direct response"
    assert result.tool_used is None
    assert result.success is True


def test_process_user_request_falls_back_after_invalid_decisions(
    monkeypatch,
):

    async def failing_decide_action(user_message):
        raise AgentDecisionError(
            "Unable to obtain valid agent decision"
        )

    monkeypatch.setattr(
        agent_service,
        "decide_action",
        failing_decide_action,
    )

    result = asyncio.run(
        agent_service.process_user_request(
            "What is the status of TX-123?"
        )
    )

    assert result.response == (
        "I couldn't reliably determine "
        "the correct action for that request."
    )
    assert result.tool_used is None
    assert result.success is False


def test_process_user_request_with_tool(monkeypatch):

    async def fake_decide_action(user_message):
        return TransactionToolDecision(
            action="tool",
            tool_name="get_transaction",
            arguments={
                "transaction_id": "TX-123",
            },
        )

    def fake_tool(transaction_id):
        return {
            "transaction_id": transaction_id,
            "status": "completed",
        }

    async def fake_chat_with_model(messages, **kwargs):
        assert kwargs["include_default_system_prompt"] is False
        assert messages[0]["content"] == agent_service.TOOL_RESULT_PROMPT
        assert "TX-123" in messages[-1]["content"]
        return "Transaction is completed."

    monkeypatch.setattr(
        agent_service,
        "decide_action",
        fake_decide_action,
    )
    monkeypatch.setattr(
        agent_service,
        "chat_with_model",
        fake_chat_with_model,
    )
    monkeypatch.setitem(
        agent_service.TOOL_REGISTRY,
        "get_transaction",
        fake_tool,
    )

    result = asyncio.run(
        agent_service.process_user_request(
            "What is the status of TX-123?"
        )
    )

    assert result.response == "Transaction is completed."
    assert result.tool_used == "get_transaction"
    assert result.success is True


def test_process_user_request_passes_request_id_to_tool(
    monkeypatch,
):

    async def fake_decide_action(user_message):
        return TransactionToolDecision(
            action="tool",
            tool_name="get_transaction",
            arguments={
                "transaction_id": "TX-123",
            },
        )

    captured = {}

    async def fake_execute_tool(
        tool_name,
        arguments,
        request_id,
    ):
        captured["request_id"] = request_id
        return {
            "found": False,
            "transaction": None,
        }

    async def fake_chat_with_model(messages, **kwargs):
        return "Transaction not found."

    monkeypatch.setattr(
        agent_service,
        "decide_action",
        fake_decide_action,
    )
    monkeypatch.setattr(
        agent_service,
        "execute_tool",
        fake_execute_tool,
    )
    monkeypatch.setattr(
        agent_service,
        "chat_with_model",
        fake_chat_with_model,
    )

    result = asyncio.run(
        agent_service.process_user_request(
            "What is the status of TX-123?",
            "request-123",
        )
    )

    assert captured["request_id"] == "request-123"
    assert result.tool_used == "get_transaction"


def test_execute_tool_sanitizes_tool_execution_error(
    monkeypatch,
):

    def failing_tool():
        raise ToolExecutionError(
            code="DATABASE_UNAVAILABLE",
            message="The database is unavailable.",
            retryable=True,
        )

    monkeypatch.setitem(
        agent_service.TOOL_REGISTRY,
        "test_tool",
        failing_tool,
    )

    result = asyncio.run(
        agent_service.execute_tool(
            "test_tool",
            {},
        )
    )

    assert result.success is False
    assert result.tool_name == "test_tool"
    assert result.error.code == "DATABASE_UNAVAILABLE"
    assert result.error.retryable is True