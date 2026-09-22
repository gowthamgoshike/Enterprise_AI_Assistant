"""Send five authenticated, independent general-question chat requests.

Set LOAD_TEST_TOKEN in the shell; never put a token in this file.
"""

import asyncio
import os
import time
import uuid

import httpx


URL = os.getenv("LOAD_TEST_URL", "http://localhost:8000/api/v1/chat")
REQUEST_COUNT = 5


async def send_request(client: httpx.AsyncClient, index: int, token: str) -> None:
    started = time.perf_counter()
    try:
        response = await client.post(
            URL,
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [{
                    "role": "user",
                    "content": f"Explain Kafka briefly. Request {index}.",
                }],
                "session_id": f"load-test-{uuid.uuid4()}",
            },
        )
        status = str(response.status_code)
    except httpx.RequestError as exc:
        status = type(exc).__name__
    print(f"request={index} status={status} elapsed={time.perf_counter() - started:.2f}s")


async def main() -> None:
    token = os.getenv("LOAD_TEST_TOKEN")
    if not token:
        raise SystemExit("Set LOAD_TEST_TOKEN to a development access token")
    async with httpx.AsyncClient(timeout=180) as client:
        await asyncio.gather(*(
            send_request(client, index, token)
            for index in range(1, REQUEST_COUNT + 1)
        ))


if __name__ == "__main__":
    asyncio.run(main())
