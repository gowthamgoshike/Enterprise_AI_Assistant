import json

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ai.models.dependencies import (
    get_model_gateway,
)

from ai.tools.transaction_tools import (
    get_transaction,
)
from ai.schemas.agent import (
    GetTransactionArguments,
)


model = get_model_gateway()

tools = [
    get_transaction,
]

model_with_tools = (
    model.bind_tools(
        tools
    )
)

tool_map = {
    get_transaction.name:
        get_transaction,
}

SYSTEM_PROMPT = """
You are an enterprise transaction
assistant.

Rules:

1. Use approved tools for enterprise
   transaction information.

2. Never invent transaction data.

3. If a tool returns no result, state
   that the transaction was not found.

4. Do not claim that a tool succeeded
   unless a tool result is present.

5. Do not reveal internal implementation
   details.
"""


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

    return {
        "found": result.get(
            "found",
            False,
        ),
        "transaction": transaction,
    }


def run_agent(
    user_message: str,
) -> str:

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=user_message
        ),
    ]

    response = model_with_tools.invoke(
        messages
    )

    if not response.tool_calls:

        return response.content

    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name not in tool_map:

        raise ValueError(
            "Requested tool is not allowed"
        )

    if tool_name == "get_transaction":

        validated_args = (
            GetTransactionArguments
            .model_validate(tool_args)
        )
        tool_args = validated_args.model_dump()

    tool_result = tool_map[
        tool_name
    ].invoke(
        tool_args
    )

    tool_message = ToolMessage(
        content=json.dumps(
            _safe_tool_result(
                tool_result
            ),
            default=str,
        ),
        tool_call_id=tool_call["id"],
    )

    final_response = model_with_tools.invoke(
        [
            *messages,
            response,
            tool_message,
        ]
    )

    return final_response.content


if __name__ == "__main__":

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=(
                "What is the status "
                "of TX-123?"
            )
        ),
    ]

    response = model_with_tools.invoke(
        messages
    )

    print(
        response.tool_calls
    )
