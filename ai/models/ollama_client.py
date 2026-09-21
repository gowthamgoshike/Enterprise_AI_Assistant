import requests

from core.app_settings import settings

from ai.prompts.system_prompts import (
    ENTERPRISE_ASSISTANT_SYSTEM_PROMPT,
)

OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_MODEL = settings.ollama_model


def generate(
    prompt: str,
) -> str:

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
        "temperature": 0.1,
    },
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]



def chat(
    messages: list[dict],
) -> str:

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data[
        "message"
    ][
        "content"
    ]
messages = [
    {
        "role": "system",
        "content":
            ENTERPRISE_ASSISTANT_SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content":
            "Tell me the exact revenue for Product P-50 today.",
    },
]

print(
    chat(messages)
)
