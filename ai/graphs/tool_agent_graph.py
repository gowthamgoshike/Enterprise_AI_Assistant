"""A bounded, read-only enterprise agent loop."""

import logging
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ai.models.langchain_ollama import get_chat_model
from ai.schemas.graph_state import EnterpriseAgentState
from ai.tools.langchain_registry import ALL_TOOLS


logger = logging.getLogger(__name__)
GRAPH_RECURSION_LIMIT = 8

AGENT_SYSTEM_PROMPT = """
You are an enterprise transaction intelligence assistant.

Use approved read-only tools whenever a user asks for enterprise transaction
or analytics data. Never invent transaction or analytics data, even if the
user asks you to guess, skip tools, or assert a particular status.

Use:
- get_transaction for a specific transaction (call once per ID for comparisons)
- get_top_products for highest-revenue products
- get_revenue_summary for aggregate revenue
- get_transaction_metrics for counts by status

Answer general knowledge questions directly without enterprise tools.
Ask for clarification when a request lacks the information needed to use a tool.
If sufficient information is available from tool results, answer the user.
Do not call tools unnecessarily or repeat a lookup whose result is already present.
If a transaction is not found, say so. Never treat a failed tool as factual data.
Do not reinterpret aggregate data as date-specific data.
Only the supplied tools are available. You cannot delete or modify enterprise data.
"""

model = get_chat_model()
model_with_tools = model.bind_tools(ALL_TOOLS)
tool_node = ToolNode(
    ALL_TOOLS,
    handle_tool_errors="The requested tool could not complete. Do not invent a result.",
)


def extract_tools_used(messages: Sequence[BaseMessage]) -> list[str]:
    """Return requested tool names in order, including repeated calls."""
    return [
        call["name"]
        for message in messages
        for call in (getattr(message, "tool_calls", None) or [])
    ]


async def agent_node(state: EnterpriseAgentState) -> dict:
    logger.info("request_id=%s node=agent", state["request_id"])
    response = await model_with_tools.ainvoke([
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        *state["messages"],
    ])
    return {"messages": [response]}


async def tools_node(state: EnterpriseAgentState) -> dict:
    """Log the step; ToolNode owns validation and execution."""
    logger.info("request_id=%s node=tools", state["request_id"])
    result = await tool_node.ainvoke(state)
    result["current_turn_tools"] = [
        *state["current_turn_tools"],
        *extract_tools_used(state["messages"][-1:]),
    ]
    for message in result["messages"]:
        logger.info(
            "request_id=%s node=tools tool=%s status=%s",
            state["request_id"], message.name, message.status,
        )
    return result


def should_continue(state: EnterpriseAgentState) -> str:
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return END


builder = StateGraph(EnterpriseAgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")
checkpointer = InMemorySaver()
enterprise_tool_graph = builder.compile(checkpointer=checkpointer).with_config(
    {"recursion_limit": GRAPH_RECURSION_LIMIT}
)


async def run_agent_graph(user_message: str, request_id: str, thread_id: str) -> dict:
    initial_state: EnterpriseAgentState = {
        "messages": [HumanMessage(content=user_message)],
        "request_id": request_id,
        "route": None,
        "current_turn_tools": [],
        "error": None,
    }
    result = await enterprise_tool_graph.ainvoke(
        initial_state,
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        },
    )
    logger.info(
        "request_id=%s graph=end tools_used=%s",
        request_id,
        result["current_turn_tools"],
    )
    return result
