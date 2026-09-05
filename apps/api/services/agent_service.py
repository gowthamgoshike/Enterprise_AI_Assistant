import asyncio
import json
import logging
import time

from ai.prompts.tool_prompts import (
    TOOL_SELECTION_PROMPT,
    TOOL_RESULT_PROMPT,
)

from ai.tools.registry import (
    TOOL_REGISTRY,
)

from apps.api.services.llm_service import (
    chat_with_model,
)


logger = logging.getLogger(
    __name__
)


async def decide_action(
    user_message: str,
) -> dict:

    messages = [
        {
            "role": "system",
            "content":
                TOOL_SELECTION_PROMPT,
        },
        {
            "role": "user",
            "content":
                user_message,
        },
    ]

    response = await chat_with_model(
        messages,
        include_default_system_prompt=False,
        response_format="json",
    )

    try:

        return json.loads(
            response
        )

    except json.JSONDecodeError:

        return {
            "action": "respond",
            "error": "invalid_model_output",
        }


async def execute_tool(
    tool_name: str,
    arguments: dict,
    request_id: str | None = None,
):

    if tool_name not in TOOL_REGISTRY:

        raise ValueError(
            "Unknown tool requested"
        )

    tool_function = TOOL_REGISTRY[
        tool_name
    ]

    logger.info(
        "tool_execution_started "
        "request_id=%s tool=%s",
        request_id,
        tool_name,
    )

    start = time.perf_counter()

    try:

        tool_result = await asyncio.to_thread(
            tool_function,
            **arguments,
        )

    finally:

        tool_latency = (
            time.perf_counter()
            -
            start
        )

        logger.info(
            "tool=%s request_id=%s latency=%.3f",
            tool_name,
            request_id,
            tool_latency,
        )

    logger.info(
        "tool_execution_completed "
        "request_id=%s tool=%s",
        request_id,
        tool_name,
    )

    return tool_result


async def process_user_request(
    user_message: str,
    request_id: str | None = None,
) -> dict:

    decision = await decide_action(
        user_message
    )

    if decision.get("action") == "respond":

        response = await chat_with_model(
            [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        )

        return {
            "response": response,
            "tool_used": None,
        }

    tool_name = decision.get(
        "tool_name"
    )

    arguments = decision.get(
        "arguments",
        {},
    )

    tool_result = await execute_tool(
        tool_name,
        arguments,
        request_id,
    )

    messages = [
        {
            "role": "system",
            "content": TOOL_RESULT_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "user",
            "content": (
                "Tool result:\n"
                f"{json.dumps(tool_result, default=str)}"
            ),
        },
    ]

    response = await chat_with_model(
        messages,
        include_default_system_prompt=False,
    )

    return {
        "response": response,
        "tool_used": tool_name,
        "tool_result": tool_result,
    }


async def generate_agent_response(
    messages: list[dict],
) -> str:

    user_message = next(
        (
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )

    decision = await decide_action(
        user_message
    )

    if decision.get("action") != "tool":

        return await chat_with_model(
            messages
        )

    tool_name = decision.get(
        "tool_name"
    )

    arguments = decision.get(
        "arguments",
        {},
    )

    tool_result = await execute_tool(
        tool_name,
        arguments,
    )

    tool_context = (
        "Use the following approved enterprise "
        "tool result to answer the user's request. "
        "Do not invent or alter any values.\n\n"
        f"{json.dumps(tool_result, default=str)}"
    )

    return await chat_with_model(
        [
            *messages,
            {
                "role": "system",
                "content": tool_context,
            },
        ]
    )
