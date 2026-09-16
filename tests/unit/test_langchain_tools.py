import asyncio
import json

from langchain_core.messages import (
    ToolMessage,
)

from ai.agents import (
    tool_calling_agent,
)
from ai.schemas.agent import (
    RouterDecision,
)
from ai.tools.transaction_tools import (
    get_transaction,
)
from apps.api.services import (
    langchain_agent_service,
)


class FakeResponse:

    def __init__(
        self,
        content="",
        tool_calls=None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeModel:

    def __init__(
        self,
        responses,
    ):
        self.responses = iter(responses)
        self.messages = []

    async def ainvoke(
        self,
        messages,
    ):
        self.messages.append(messages)
        return next(self.responses)


class FakeSyncModel:

    def __init__(
        self,
        responses,
    ):
        self.responses = iter(responses)
        self.messages = []

    def invoke(
        self,
        messages,
    ):
        self.messages.append(messages)
        return next(self.responses)


class FakeTool:

    name = "get_transaction"

    def __init__(self):
        self.arguments = None

    def invoke(
        self,
        arguments,
    ):
        self.arguments = arguments
        return {
            "found": True,
            "transaction": {
                "transaction_id": arguments[
                    "transaction_id"
                ],
                "status": "completed",
                "internal_secret": "must-not-leak",
            },
            "internal_metadata": "must-not-leak",
        }


def test_transaction_tool_metadata():

    assert get_transaction.name == "get_transaction"

    schema = (
        get_transaction
        .args_schema
        .model_json_schema()
    )

    assert "transaction_id" in schema[
        "properties"
    ]


def test_router_schema():

    result = RouterDecision(
        action="get_transaction",
        transaction_id="TX-123",
    )

    assert result.transaction_id == "TX-123"


def test_langchain_service_answers_general_question(
    monkeypatch,
):

    fake_model = FakeModel([
        FakeResponse(
            content="Kafka consumer groups share work across consumers."
        )
    ])
    monkeypatch.setattr(
        langchain_agent_service,
        "model_with_tools",
        fake_model,
    )

    result = asyncio.run(
        langchain_agent_service.process_with_langchain(
            "Explain Kafka consumer groups."
        )
    )

    assert result == {
        "response": "Kafka consumer groups share work across consumers.",
        "tools_used": [],
    }
    assert len(fake_model.messages) == 1


def test_manual_tool_loop_answers_general_question(
    monkeypatch,
):

    fake_model = FakeSyncModel([
        FakeResponse(
            content="Kafka uses consumer groups to share partitions."
        )
    ])
    monkeypatch.setattr(
        tool_calling_agent,
        "model_with_tools",
        fake_model,
    )

    result = tool_calling_agent.run_agent(
        "Explain Kafka consumer groups."
    )

    assert result == (
        "Kafka uses consumer groups to share partitions."
    )
    assert len(fake_model.messages) == 1


def test_manual_tool_loop_validates_and_executes_tool(
    monkeypatch,
):

    fake_model = FakeSyncModel([
        FakeResponse(
            tool_calls=[
                {
                    "name": "get_transaction",
                    "args": {
                        "transaction_id": " tx-404 ",
                    },
                    "id": "call-2",
                }
            ]
        ),
        FakeResponse(
            content="No transaction was found."
        ),
    ])
    fake_tool = FakeTool()

    monkeypatch.setattr(
        tool_calling_agent,
        "model_with_tools",
        fake_model,
    )
    monkeypatch.setattr(
        tool_calling_agent,
        "tool_map",
        {"get_transaction": fake_tool},
    )

    result = tool_calling_agent.run_agent(
        "What happened to TX-DOES-NOT-EXIST?"
    )

    assert result == "No transaction was found."
    assert fake_tool.arguments == {
        "transaction_id": "TX-404",
    }


def test_langchain_service_executes_validated_tool(
    monkeypatch,
):

    fake_model = FakeModel([
        FakeResponse(
            tool_calls=[
                {
                    "name": "get_transaction",
                    "args": {
                        "transaction_id": " tx-123 ",
                    },
                    "id": "call-1",
                }
            ]
        ),
        FakeResponse(
            content="Transaction TX-123 is completed."
        ),
    ])
    fake_tool = FakeTool()

    monkeypatch.setattr(
        langchain_agent_service,
        "model_with_tools",
        fake_model,
    )
    monkeypatch.setattr(
        langchain_agent_service,
        "TOOL_MAP",
        {"get_transaction": fake_tool},
    )

    result = asyncio.run(
        langchain_agent_service.process_with_langchain(
            "What is the status of TX-123?"
        )
    )

    assert result == {
        "response": "Transaction TX-123 is completed.",
        "tools_used": [
            "get_transaction",
        ],
    }
    assert fake_tool.arguments == {
        "transaction_id": "TX-123",
    }

    tool_message = fake_model.messages[1][-1]
    assert isinstance(tool_message, ToolMessage)

    payload = json.loads(
        tool_message.content
    )
    assert payload == {
        "found": True,
        "transaction": {
            "transaction_id": "TX-123",
            "status": "completed",
        },
    }
    assert "internal_metadata" not in tool_message.content


def test_langchain_service_executes_multiple_tools(
    monkeypatch,
):

    fake_model = FakeModel([
        FakeResponse(
            tool_calls=[
                {
                    "name": "get_transaction",
                    "args": {
                        "transaction_id": "TX-100",
                    },
                    "id": "call-1",
                },
                {
                    "name": "get_transaction",
                    "args": {
                        "transaction_id": "TX-200",
                    },
                    "id": "call-2",
                },
            ]
        ),
        FakeResponse(
            content="TX-100 and TX-200 are completed."
        ),
    ])
    fake_tool = FakeTool()

    monkeypatch.setattr(
        langchain_agent_service,
        "model_with_tools",
        fake_model,
    )
    monkeypatch.setattr(
        langchain_agent_service,
        "TOOL_MAP",
        {"get_transaction": fake_tool},
    )

    result = asyncio.run(
        langchain_agent_service.process_with_langchain(
            "Compare TX-100 and TX-200.",
            request_id="request-123",
        )
    )

    assert result == {
        "response": "TX-100 and TX-200 are completed.",
        "tools_used": [
            "get_transaction",
            "get_transaction",
        ],
    }
    assert len(fake_model.messages) == 2
    assert len(fake_model.messages[1][-2:]) == 2
    assert all(
        isinstance(message, ToolMessage)
        for message in fake_model.messages[1][-2:]
    )


def test_langchain_service_handles_followup_tool_calls(monkeypatch):
    fake_model = FakeModel([
        FakeResponse(tool_calls=[{
            "name": "get_revenue_summary", "args": {}, "id": "revenue",
        }]),
        FakeResponse(tool_calls=[{
            "name": "get_transaction_metrics", "args": {}, "id": "metrics",
        }]),
        FakeResponse(content="Revenue is 100; 2 transactions completed."),
    ])
    executed = []

    def execute(call, request_id):
        executed.append(call["name"])
        return {"total_revenue": 100} if len(executed) == 1 else {"completed": 2}

    monkeypatch.setattr(langchain_agent_service, "model_with_tools", fake_model)
    monkeypatch.setattr(langchain_agent_service, "execute_tool_call", execute)
    result = asyncio.run(langchain_agent_service.process_with_langchain(
        "How is the business doing?"
    ))
    assert result["tools_used"] == executed == [
        "get_revenue_summary", "get_transaction_metrics",
    ]
    assert result["response"] == "Revenue is 100; 2 transactions completed."
    assert len(fake_model.messages) == 3


def test_langchain_service_bounds_tool_loop(monkeypatch):
    fake_model = FakeModel([
        FakeResponse(tool_calls=[{
            "name": "get_revenue_summary", "args": {}, "id": str(index),
        }]) for index in range(2)
    ])
    executed = []
    monkeypatch.setattr(langchain_agent_service, "model_with_tools", fake_model)
    monkeypatch.setattr(langchain_agent_service, "MAX_AGENT_STEPS", 1)
    monkeypatch.setattr(
        langchain_agent_service, "execute_tool_call",
        lambda call, request_id: executed.append(call["name"]) or {},
    )
    result = asyncio.run(langchain_agent_service.process_with_langchain(
        "How is the business doing?"
    ))
    assert executed == ["get_revenue_summary"]
    assert "tool-call limit" in result["response"]
