from langchain_core.messages import BaseMessage, trim_messages

from ai.context.config import MAX_RECENT_MESSAGES
from ai.schemas.graph_state import EnterpriseAgentState


def truncate_tool_result(content: str, max_chars: int) -> str:
    """Bound a serialized tool result as a last-resort context safeguard."""
    if len(content) <= max_chars:
        return content

    return content[:max_chars] + "\n...[truncated]"


def estimate_tokens(text: str) -> int:
    """Return a rough development-time token estimate for text."""
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: list[BaseMessage]) -> int:
    """Estimate the combined content tokens for a sequence of messages."""
    total = 0

    for message in messages:
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        total += estimate_tokens(content)

    return total


def select_messages_by_budget(
    messages: list[BaseMessage],
    max_tokens: int,
) -> list[BaseMessage]:
    """Select the newest messages whose estimated cost fits the budget."""
    selected = []
    total = 0

    for message in reversed(messages):
        cost = estimate_tokens(str(message.content))

        if total + cost > max_tokens:
            break

        selected.append(message)
        total += cost

    return list(reversed(selected))


def get_recent_messages(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Return a valid window containing the most recent conversation turns."""
    if len(messages) <= MAX_RECENT_MESSAGES:
        return messages

    return trim_messages(
        messages,
        max_tokens=MAX_RECENT_MESSAGES,
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )


def build_model_context(state: EnterpriseAgentState) -> list[BaseMessage]:
    """Build the recent-message portion of the model context."""
    return get_recent_messages(state["messages"])
