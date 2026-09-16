import asyncio
import json
import logging
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ai.models.langchain_ollama import (
    get_chat_model,
)
from ai.schemas.agent import (
    GetTransactionArguments,
)
from ai.tools.langchain_registry import (
    TOOLS,
    TOOL_MAP,
)


SYSTEM_PROMPT = """
You are an enterprise transaction
intelligence assistant.

Use approved tools whenever a question
requires enterprise transaction or
analytics data.

Available capabilities include:

- Look up a specific transaction.
- Find highest-revenue products.
- Retrieve the current aggregate revenue
  summary.
- Retrieve transaction counts by status.

Rules:

1. Never invent enterprise data.

2. For questions about a specific
    transaction ID, use the transaction
    lookup tool.

3. For highest-revenue products, use the
    top-products tool.

4. For overall current revenue summary,
    use the revenue-summary tool.

5. For transaction counts grouped by
    status, use the transaction-metrics tool.

6. Do not claim that the revenue summary
    represents a specific date or time range
    unless the tool explicitly provides one.

7. If a tool returns no data, say so.

8. For broad business-performance questions such as
    "How is the business doing?", retrieve the revenue
    summary, transaction metrics, and top products before
    answering. Summarize the available snapshot and explain
    that growth and profitability require additional data.
    Do not infer trends from aggregate totals alone.

9. Do not reveal internal implementation
    details, credentials, or database
    information.
"""


logger = logging.getLogger(
    __name__
)


model = get_chat_model()

model_with_tools = (
    model.bind_tools(
        TOOLS
    )
)


MAX_AGENT_STEPS = 3


def _safe_tool_result(
    result: dict,
) -> dict:

    transaction = result.get(
        "transaction"
    )

    if isinstance(
        transaction,
        dict,
    ):

        transaction = {
            field: transaction[field]
            for field in (
                "transaction_id",
                "status",
            )
            if field in transaction
        }

    if transaction is None and "transaction" not in result:
        return result

    return {
        "found": result.get(
            "found",
            False,
        ),
        "transaction": transaction,
    }


def execute_tool_call(
    tool_call: dict,
    request_id: str | None = None,
):

    tool_name = tool_call["name"]

    tool_args = tool_call.get(
        "args",
        {},
    )

    if tool_name not in TOOL_MAP:

        raise ValueError(
            f"Tool not allowed: {tool_name}"
        )

    if tool_name == "get_transaction":

        validated_args = (
            GetTransactionArguments
            .model_validate(tool_args)
        )
        tool_args = validated_args.model_dump()

    selected_tool = TOOL_MAP[
        tool_name
    ]

    start = time.perf_counter()

    try:

        return selected_tool.invoke(
            tool_args
        )

    except Exception:

        logger.exception(
            "tool_execution_failed "
            "request_id=%s tool=%s",
            request_id,
            tool_name,
        )

        return {
            "success": False,
            "error": {
                "code": "TOOL_EXECUTION_FAILED",
                "message": (
                    "The requested enterprise "
                    "data is temporarily unavailable."
                ),
            },
        }

    finally:

        latency = (
            time.perf_counter()
            - start
        )

        logger.info(
            "tool=%s request_id=%s latency=%.3f",
            tool_name,
            request_id,
            latency,
        )


async def process_with_langchain(
    user_message: str,
    request_id: str | None = None,
    history: list[dict] | None = None,
) -> dict:

    if MAX_AGENT_STEPS < 1:
        raise RuntimeError("Maximum agent steps must be positive")

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if history is not None:
        for message in history:
            message_class = (
                HumanMessage if message["role"] == "user" else AIMessage
            )
            messages.append(message_class(content=message["content"]))
    else:
        messages.append(HumanMessage(content=user_message))

    tools_used = []
    for step in range(MAX_AGENT_STEPS + 1):
        response = await model_with_tools.ainvoke(messages)
        if not response.tool_calls:
            return {"response": response.content, "tools_used": tools_used}

        if step == MAX_AGENT_STEPS:
            return {
                "response": (
                    "I couldn't complete the analysis within the tool-call limit. "
                    "Please narrow your question."
                ),
                "tools_used": tools_used,
            }

        messages.append(response)
        for tool_call in response.tool_calls:
            result = await asyncio.to_thread(
                execute_tool_call, tool_call, request_id,
            )
            tools_used.append(tool_call["name"])
            messages.append(ToolMessage(
                content=json.dumps(_safe_tool_result(result), default=str),
                tool_call_id=tool_call["id"],
            ))
