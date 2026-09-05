import json

import requests

from apps.ui.config import API_BASE_URL


def get_api_health() -> dict:

    response = requests.get(
        f"{API_BASE_URL}/api/v1/health",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()

def get_model_health() -> dict:

    response = requests.get(
        f"{API_BASE_URL}/api/v1/health/model",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()

def send_chat(
    messages: list[dict],
) -> dict:

    response = requests.post(
        f"{API_BASE_URL}/api/v1/chat",
        json={
            "messages": messages,
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


def stream_chat(
    messages: list[dict],
):

    with requests.post(
        (
            f"{API_BASE_URL}"
            "/api/v1/chat/stream"
        ),
        json={
            "messages": messages,
        },
        stream=True,
        timeout=180,
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines():

            if line:

                yield json.loads(
                    line.decode(
                        "utf-8"
                    )
                )