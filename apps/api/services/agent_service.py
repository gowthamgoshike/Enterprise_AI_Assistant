import asyncio
import json
import logging
import time

from ai.prompts.tool_prompts import (
    TOOL_SELECTION_PROMPT,
    TOOL_RESULT_PROMPT,
)
from ai.exceptions import (
    AgentDecisionError,
)
from ai.schemas.agent import (
    AgentDecision,
    AgentResponse,
    agent_decision_adapter,
)
from pydantic import (
    ValidationError,
)

from ai.tools.registry import (
    TOOL_REGISTRY,
)
from ai.tools.exceptions import (
    ToolExecutionError,
)
from ai.schemas.tools import (
    ToolError,
    ToolResult,
)

from apps.api.services.llm_service import (
    chat_with_model,
)


logger = logging.getLogger(
    __name__
)


MAX_DECISION_ATTEMPTS = 2


async def decide_action(
    user_message: str,
) -> AgentDecision:

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

    for attempt in range(
        MAX_DECISION_ATTEMPTS
    ):

        response = await chat_with_model(
            messages,
            include_default_system_prompt=False,
            response_format="json",
        )

        try:

            raw_decision = json.loads(
                response
            )

            return agent_decision_adapter.validate_python(
                raw_decision
            )

        except (
            json.JSONDecodeError,
            ValidationError,
        ) as exc:

            if attempt == MAX_DECISION_ATTEMPTS - 1:

                raise AgentDecisionError(
                    "Unable to obtain valid agent decision"
                ) from exc

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output did not "
                        "match the required JSON schema. "
                        "Return valid JSON only."
                    ),
                }
            )


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

    except ToolExecutionError as exc:

        tool_result = ToolResult(
            success=False,
            tool_name=tool_name,
            error=ToolError(
                code=exc.code,
                message=exc.message,
                retryable=exc.retryable,
            ),
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
) -> AgentResponse:

    try:

        decision = await decide_action(
            user_message
        )

    except AgentDecisionError:

        return AgentResponse(
            response=(
                "I couldn't reliably determine "
                "the correct action for that request."
            ),
            tool_used=None,
            success=False,
        )

    if decision.action == "respond":
        response = await chat_with_model(
            [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        )

        return AgentResponse(
            response=response,
            tool_used=None,
        )

    tool_result = await execute_tool(
        decision.tool_name,
        decision.arguments.model_dump(),
        request_id,
    )

    serialized_tool_result = (
        tool_result.model_dump(mode="json")
        if hasattr(tool_result, "model_dump")
        else tool_result
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
                f"{json.dumps(serialized_tool_result)}"
            ),
        },
    ]

    response = await chat_with_model(
        messages,
        include_default_system_prompt=False,
    )

    return AgentResponse(
        response=response,
        tool_used=decision.tool_name,
    )


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

    decision = await decide_action(user_message)

    if decision.action == "respond":

        response = await chat_with_model(
            messages
        )

        return response

    tool_result = await execute_tool(
        decision.tool_name,
        decision.arguments.model_dump(),
    )

    serialized_tool_result = (
        tool_result.model_dump(mode="json")
        if hasattr(tool_result, "model_dump")
        else tool_result
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
                f"{json.dumps(serialized_tool_result)}"
            ),
        },
    ]

    return await chat_with_model(
        messages,
        include_default_system_prompt=False,
    )


