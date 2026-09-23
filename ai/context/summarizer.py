import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ai.context.config import SUMMARY_TRIGGER_MESSAGES
from ai.models.dependencies import get_model_gateway
from ai.models.gateway import ModelGateway
from ai.models.reliability import invoke_model_with_retry
from ai.prompts.memory_prompts import CONVERSATION_SUMMARY_PROMPT
from ai.schemas.graph_state import EnterpriseAgentState


# Optional injection seam; avoid retaining an async client across event loops.
summary_model = None
TRANSACTION_ID_PATTERN = re.compile(r"TX-[A-Za-z0-9][A-Za-z0-9-]*", re.IGNORECASE)
MAX_SUMMARY_TRANSACTION_IDS = 20


def _transaction_ids(
    messages: list[BaseMessage],
    existing_summary: str | None,
) -> list[str]:
    """Extract a bounded, ordered set of transaction identifiers."""
    text = "\n".join([
        existing_summary or "",
        *(str(message.content) for message in messages),
    ])
    identifiers = []
    seen = set()
    for match in TRANSACTION_ID_PATTERN.finditer(text):
        identifier = match.group(0)
        normalized = identifier.casefold()
        if normalized not in seen:
            seen.add(normalized)
            identifiers.append(identifier)
        if len(identifiers) == MAX_SUMMARY_TRANSACTION_IDS:
            break
    return identifiers


async def summarize_messages(
    messages: list[BaseMessage],
    existing_summary: str | None,
) -> str:
    """Create a concise summary that combines prior and new context."""
    conversation_text = "\n".join(
        f"{type(message).__name__}: {message.content}"
        for message in messages
    )

    prompt = f"""
Existing summary:

{existing_summary or "None"}

New conversation:

{conversation_text}

Create an updated concise summary.
"""

    model_messages = [
        SystemMessage(content=CONVERSATION_SUMMARY_PROMPT),
        HumanMessage(content=prompt),
    ]
    active_model = summary_model if summary_model is not None else get_model_gateway()
    if isinstance(active_model, ModelGateway):
        response = await active_model.ainvoke(
            model_messages,
            operation_name="conversation_summary",
        )
    else:
        response = await invoke_model_with_retry(
            lambda: active_model.ainvoke(model_messages),
            operation_name="conversation_summary",
        )

    summary = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    missing_identifiers = [
        identifier
        for identifier in _transaction_ids(messages, existing_summary)
        if identifier.casefold() not in summary.casefold()
    ]
    if missing_identifiers:
        summary = (
            f"{summary.rstrip()}\n\n"
            "Main entities discussed:\n"
            + "\n".join(f"- {identifier}" for identifier in missing_identifiers)
        )

    return summary


def should_summarize(state: EnterpriseAgentState) -> bool:
    """Return whether history exceeds the development message threshold."""
    summarized_message_count = state.get("summarized_message_count", 0)
    unsummarized_message_count = len(state["messages"]) - summarized_message_count
    return unsummarized_message_count > SUMMARY_TRIGGER_MESSAGES
