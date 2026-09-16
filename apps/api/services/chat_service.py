from apps.api.services.langchain_agent_service import process_with_langchain


async def generate_chat_response(messages: list[dict]) -> str:
    user_message = next(
        (message["content"] for message in reversed(messages)
         if message["role"] == "user"),
        "",
    )
    result = await process_with_langchain(user_message, history=messages)
    return result["response"]
