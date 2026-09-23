from ai.models.dependencies import get_model_gateway
from ai.prompts.system_prompts import (
    ENTERPRISE_ASSISTANT_SYSTEM_PROMPT,
)
from core.config import settings

MODEL_NAME = settings.model_name
# Compatibility for evaluation/reporting imports during the provider migration.
OLLAMA_MODEL = MODEL_NAME

LLM_REQUEST_TIMEOUT_SECONDS = settings.model_timeout_seconds

async def chat_with_model(
    messages: list[dict],
    include_default_system_prompt: bool = True,
    response_format: str | None = None,
) -> str:

    if include_default_system_prompt:

        full_messages = [
            {
                "role": "system",
                "content":
                    ENTERPRISE_ASSISTANT_SYSTEM_PROMPT,
            },
            *messages,
        ]

    else:

        full_messages = messages

    kwargs = {"format": response_format} if response_format is not None else {}
    response = await get_model_gateway().ainvoke(full_messages, **kwargs)
    return response.content if isinstance(response.content, str) else str(response.content)


async def stream_chat_with_model(
    messages: list[dict],
):

    full_messages = [
        {
            "role": "system",
            "content":
                ENTERPRISE_ASSISTANT_SYSTEM_PROMPT,
        },
        *messages,
    ]

    async for chunk in get_model_gateway().astream(full_messages):
        content = getattr(chunk, "content", chunk)
        if content:
            yield content if isinstance(content, str) else str(content)
