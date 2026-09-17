import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ai.models.langchain_ollama import get_chat_model
from ai.tools.langchain_registry import ANALYTICS_TOOLS, ANALYTICS_TOOL_MAP


model = get_chat_model()
analytics_model = model.bind_tools(ANALYTICS_TOOLS)

ANALYTICS_SYSTEM_PROMPT = """
You are the enterprise analytics
component of an AI assistant.

Use approved analytics tools whenever
the answer requires enterprise metrics.

Available analytics cover:

- highest-revenue products
- aggregate revenue summary
- transaction counts by status

Never invent enterprise metrics.

Do not reinterpret aggregate revenue as
date-specific revenue unless the tool
explicitly supplies date information.
"""


async def handle_analytics_request(user_message: str) -> dict:
    messages = [
        SystemMessage(content=ANALYTICS_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = await analytics_model.ainvoke(messages)
    if not response.tool_calls:
        return {"response": response.content, "tools_used": []}

    messages.append(response)
    tools_used = []
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        if tool_name not in ANALYTICS_TOOL_MAP:
            raise ValueError("Analytics tool not permitted")

        result = await ANALYTICS_TOOL_MAP[tool_name].ainvoke(tool_call["args"])
        tools_used.append(tool_name)
        messages.append(ToolMessage(
            content=json.dumps(result, default=str),
            tool_call_id=tool_call["id"],
        ))

    final_response = await analytics_model.ainvoke(messages)
    return {"response": final_response.content, "tools_used": tools_used}
