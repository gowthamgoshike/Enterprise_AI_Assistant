from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai.context.config import MAX_CONTEXT_TOKENS, MAX_RECENT_MESSAGES
from ai.context.manager import (
    build_model_context,
    estimate_message_tokens,
    estimate_tokens,
    get_recent_messages,
    select_messages_by_budget,
    truncate_tool_result,
)


def test_truncate_tool_result_keeps_content_within_limit() -> None:
    assert truncate_tool_result("short result", max_chars=20) == "short result"


def test_truncate_tool_result_marks_truncated_content() -> None:
    assert truncate_tool_result("abcdefgh", max_chars=4) == "abcd\n...[truncated]"


def test_context_token_budget_is_a_conservative_development_default() -> None:
    assert MAX_CONTEXT_TOKENS == 6000


def test_estimate_tokens_has_a_minimum_cost() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_message_tokens_handles_structured_content() -> None:
    messages = [
        HumanMessage(content="abcdefgh"),
        AIMessage(content=[{"type": "text", "text": "hello"}]),
    ]

    assert estimate_message_tokens(messages) == (
        estimate_tokens("abcdefgh")
        + estimate_tokens(str(messages[1].content))
    )


def test_select_messages_by_budget_keeps_newest_messages_in_order() -> None:
    messages = [
        HumanMessage(content="a" * 8),
        AIMessage(content="b" * 12),
        HumanMessage(content="c" * 4),
    ]

    selected = select_messages_by_budget(messages, max_tokens=4)

    assert selected == messages[-2:]


def test_select_messages_by_budget_returns_empty_when_newest_does_not_fit() -> None:
    messages = [HumanMessage(content="a" * 8)]

    assert select_messages_by_budget(messages, max_tokens=1) == []


def test_build_model_context_uses_recent_message_window() -> None:
    messages = [
        message
        for turn in range(MAX_RECENT_MESSAGES)
        for message in (
            HumanMessage(content=f"Question {turn}"),
            AIMessage(content=f"Answer {turn}"),
        )
    ]
    state = {
        "messages": messages,
        "request_id": "REQ-context",
        "route": None,
        "current_turn_tools": [],
        "conversation_summary": "Existing summary",
        "error": None,
    }

    assert build_model_context(state) == messages[-MAX_RECENT_MESSAGES:]


def test_returns_original_short_history() -> None:
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi")]

    assert get_recent_messages(messages) is messages


def test_keeps_recent_messages_within_configured_limit() -> None:
    messages = [
        message
        for turn in range(MAX_RECENT_MESSAGES)
        for message in (
            HumanMessage(content=f"Question {turn}"),
            AIMessage(content=f"Answer {turn}"),
        )
    ]

    recent = get_recent_messages(messages)

    assert len(recent) == MAX_RECENT_MESSAGES
    assert recent == messages[-MAX_RECENT_MESSAGES:]


def test_does_not_keep_an_orphaned_tool_message() -> None:
    messages = [
        HumanMessage(content="Find transaction TX-1"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_transaction",
                "args": {"transaction_id": "TX-1"},
                "id": "call-1",
            }],
        ),
        ToolMessage(content="not found", tool_call_id="call-1"),
        AIMessage(content="Transaction TX-1 was not found."),
    ]
    messages.extend(
        message
        for turn in range(5)
        for message in (
            HumanMessage(content=f"Question {turn}"),
            AIMessage(content=f"Answer {turn}"),
        )
    )

    recent = get_recent_messages(messages)

    assert isinstance(recent[0], HumanMessage)
    assert all(not isinstance(message, ToolMessage) for message in recent)


def test_preserves_initial_system_message() -> None:
    system_message = SystemMessage(content="Follow enterprise policy.")
    messages = [system_message]
    messages.extend(
        message
        for turn in range(7)
        for message in (
            HumanMessage(content=f"Question {turn}"),
            AIMessage(content=f"Answer {turn}"),
        )
    )

    recent = get_recent_messages(messages)

    assert recent[0] is system_message
    assert isinstance(recent[1], HumanMessage)
    assert len(recent) <= MAX_RECENT_MESSAGES
