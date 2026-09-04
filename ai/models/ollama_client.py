import os

import requests

from dotenv import load_dotenv

from ai.prompts.system_prompts import (
    ENTERPRISE_ASSISTANT_SYSTEM_PROMPT,
)

load_dotenv()


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL"
)


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL"
)


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