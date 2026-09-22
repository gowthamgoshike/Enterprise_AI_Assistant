"""Parse the application's authenticated chat SSE protocol."""

import json
from collections.abc import Iterator

import httpx

from apps.ui.config import API_BASE_URL
from apps.ui.services.api_client import SessionExpiredError


def stream_chat(
    *,
    message: str,
    session_id: str,
    access_token: str,
) -> Iterator[dict]:
    with httpx.stream(
        "POST",
        f"{API_BASE_URL}/api/v1/chat/stream",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "text/event-stream",
        },
        json={
            "messages": [{"role": "user", "content": message}],
            "session_id": session_id,
        },
        timeout=httpx.Timeout(180, connect=10),
    ) as response:
        if response.status_code == 401:
            raise SessionExpiredError("Session expired")
        response.raise_for_status()

        event_type = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                payload = json.loads(line.removeprefix("data:").strip())
                if payload.get("type") == event_type and isinstance(
                    payload.get("data"), dict
                ):
                    yield payload
                event_type = None
