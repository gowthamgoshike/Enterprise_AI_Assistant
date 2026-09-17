import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ai.models.langchain_ollama import get_chat_model
from ai.tools.langchain_registry import TRANSACTION_TOOLS, TRANSACTION_TOOL_MAP


model = get_chat_model()
transaction_model = model.bind_tools(TRANSACTION_TOOLS)

TRANSACTION_SYSTEM_PROMPT = """
You are the transaction-data component
of an enterprise AI assistant.

Use approved tools for factual information
about specific transaction IDs.

Never invent transaction status, amount,
product, customer, source, or timestamps.

If a transaction cannot be found,
state that clearly.
"""


async def handle_transaction_request(user_message: str) -> dict:
    messages = [
        SystemMessage(content=TRANSACTION_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = await transaction_model.ainvoke(messages)
    if not response.tool_calls:
        return {"response": response.content, "tools_used": []}

    messages.append(response)
    tools_used = []
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        if tool_name not in TRANSACTION_TOOL_MAP:
            raise ValueError("Transaction tool not permitted")

        result = await TRANSACTION_TOOL_MAP[tool_name].ainvoke(tool_call["args"])
        tools_used.append(tool_name)
        messages.append(ToolMessage(
            content=json.dumps(result, default=str),
            tool_call_id=tool_call["id"],
        ))

    final_response = await transaction_model.ainvoke(messages)
    return {"response": final_response.content, "tools_used": tools_used}
