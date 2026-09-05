import asyncio

from apps.api.services import agent_service


def test_invalid_model_json_returns_respond_action(monkeypatch):

    async def fake_chat_with_model(*args, **kwargs):
        return "not-json"

    monkeypatch.setattr(
        agent_service,
        "chat_with_model",
        fake_chat_with_model,
    )

    decision = asyncio.run(
        agent_service.decide_action(
            "What is the status of TX-123?"
        )
    )

    assert decision == {
        "action": "respond",
        "error": "invalid_model_output",
    }


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
        return {
            "action": "respond",
        }

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

    assert result == {
        "response": "direct response",
        "tool_used": None,
    }


def test_process_user_request_with_tool(monkeypatch):

    async def fake_decide_action(user_message):
        return {
            "action": "tool",
            "tool_name": "test_tool",
            "arguments": {
                "transaction_id": "TX-123",
            },
        }

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
        "test_tool",
        fake_tool,
    )

    result = asyncio.run(
        agent_service.process_user_request(
            "What is the status of TX-123?"
        )
    )

    assert result == {
        "response": "Transaction is completed.",
        "tool_used": "test_tool",
        "tool_result": {
            "transaction_id": "TX-123",
            "status": "completed",
        },
    }


def test_process_user_request_passes_request_id_to_tool(
    monkeypatch,
):

    async def fake_decide_action(user_message):
        return {
            "action": "tool",
            "tool_name": "get_transaction",
            "arguments": {
                "transaction_id": "TX-123",
            },
        }

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
    assert result["tool_used"] == "get_transaction"