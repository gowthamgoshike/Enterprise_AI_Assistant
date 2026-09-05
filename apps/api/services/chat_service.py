from apps.api.services.agent_service import (
    generate_agent_response,
)


async def generate_chat_response(
    messages: list[dict],
) -> str:

    return await generate_agent_response(
        messages
    )