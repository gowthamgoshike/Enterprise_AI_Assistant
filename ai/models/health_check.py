import requests

from ai.models.ollama_client import (
    OLLAMA_BASE_URL,
)


def check_ollama():

    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=10,
    )

    response.raise_for_status()

    print(
        "Ollama connection successful"
    )


if __name__ == "__main__":
    check_ollama()