import httpx

from ai.prompts.system_prompts import (
    ENTERPRISE_ASSISTANT_SYSTEM_PROMPT,
)
from ai.models.reliability import invoke_model_with_retry
from core.config import MODEL_TIMEOUT_SECONDS
from core.app_settings import settings
from core.errors import ModelUnavailableError

OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_MODEL = settings.ollama_model

LLM_REQUEST_TIMEOUT_SECONDS = MODEL_TIMEOUT_SECONDS

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

    request_payload = {
        "model": OLLAMA_MODEL,
        "messages": full_messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    if response_format is not None:

        request_payload["format"] = response_format

    async def operation() -> str:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(LLM_REQUEST_TIMEOUT_SECONDS)
        ) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=request_payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if response.status_code == 429 or response.status_code >= 500:
                    raise ModelUnavailableError() from exc
                raise
            data = response.json()
            return data["message"]["content"]

    return await invoke_model_with_retry(operation)


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

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(MODEL_TIMEOUT_SECONDS)
    ) as client:

        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": full_messages,
                "stream": True,
                "options": {
                    "temperature": 0.1,
                },
            },
        ) as response:

            response.raise_for_status()

            async for line in response.aiter_lines():

                if line:
                    yield line
