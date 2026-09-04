from apps.api.services.llm_service import (
    chat_with_model,
)


async def generate_chat_response(
    messages: list[dict],
) -> str:

    return await chat_with_model(
        messages
    )