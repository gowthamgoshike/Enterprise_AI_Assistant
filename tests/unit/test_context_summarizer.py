import asyncio
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai.context import summarizer
from ai.context.config import SUMMARY_TRIGGER_MESSAGES
from ai.prompts.memory_prompts import CONVERSATION_SUMMARY_PROMPT


def make_state(message_count: int) -> dict:
    return {
        "messages": [HumanMessage(content="message") for _ in range(message_count)],
        "request_id": "REQ-summary",
        "route": None,
        "current_turn_tools": [],
        "conversation_summary": None,
        "error": None,
    }


def test_should_summarize_only_after_threshold() -> None:
    assert not summarizer.should_summarize(make_state(SUMMARY_TRIGGER_MESSAGES))
    assert summarizer.should_summarize(make_state(SUMMARY_TRIGGER_MESSAGES + 1))


def test_should_summarize_counts_only_messages_added_after_last_summary() -> None:
    state = make_state(SUMMARY_TRIGGER_MESSAGES * 2)
    state["summarized_message_count"] = SUMMARY_TRIGGER_MESSAGES

    assert not summarizer.should_summarize(state)

    state["messages"].append(HumanMessage(content="one more message"))
    assert summarizer.should_summarize(state)


def test_summarize_messages_includes_existing_and_new_context(monkeypatch) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="Updated summary")
    monkeypatch.setattr(summarizer, "summary_model", model)

    result = asyncio.run(summarizer.summarize_messages(
        [HumanMessage(content="Check TX-123")],
        existing_summary="The user is investigating a transaction.",
    ))

    assert result == "Updated summary\n\nMain entities discussed:\n- TX-123"
    prompt_messages = model.ainvoke.await_args.args[0]
    assert prompt_messages == [
        SystemMessage(content=CONVERSATION_SUMMARY_PROMPT),
        prompt_messages[1],
    ]
    assert "The user is investigating a transaction." in prompt_messages[1].content
    assert "HumanMessage: Check TX-123" in prompt_messages[1].content


def test_summarize_messages_does_not_duplicate_retained_transaction_id(monkeypatch) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content="Main entities discussed: TX-123"
    )
    monkeypatch.setattr(summarizer, "summary_model", model)

    result = asyncio.run(summarizer.summarize_messages(
        [HumanMessage(content="Check TX-123")],
        existing_summary=None,
    ))

    assert result.count("TX-123") == 1
