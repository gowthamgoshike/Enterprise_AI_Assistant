from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from ai.models.langchain_ollama import (
    get_chat_model,
)

from ai.schemas.agent import (
    RouterDecision,
)


model = get_chat_model()

structured_model = (
    model.with_structured_output(
        RouterDecision
    )
)


def route_request(
    user_message: str,
) -> RouterDecision:

    messages = [
        SystemMessage(
            content="""
You are an enterprise AI router.

If the user asks about a specific
transaction ID, choose:

action = get_transaction

Otherwise choose:

action = respond

Do not invent enterprise data.
"""
        ),
        HumanMessage(
            content=user_message
        ),
    ]

    return structured_model.invoke(
        messages
    )


def validate_decision(
    decision: RouterDecision,
) -> RouterDecision:

    if (
        decision.action
        == "get_transaction"
        and not decision.transaction_id
    ):

        raise ValueError(
            "Missing transaction_id"
        )

    return decision


transaction_decision = validate_decision(
    route_request(
    "What is the status of TX-123?"
    )
)

general_decision = validate_decision(
    route_request(
        "Explain Kafka."
    )
)

print(transaction_decision)
print(general_decision)