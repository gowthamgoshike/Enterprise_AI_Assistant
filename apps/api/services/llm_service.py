import os

import httpx

from dotenv import load_dotenv
from ai.prompts.system_prompts import (
    ENTERPRISE_ASSISTANT_SYSTEM_PROMPT,
)

load_dotenv()


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
)

LLM_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv(
        "LLM_REQUEST_TIMEOUT_SECONDS",
        "120",
    )
)

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

    timeout = httpx.Timeout(
        LLM_REQUEST_TIMEOUT_SECONDS
    )

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

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=request_payload,
        )

        response.raise_for_status()

        data = response.json()

        return data[
            "message"
        ][
            "content"
        ]


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
        timeout=None
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