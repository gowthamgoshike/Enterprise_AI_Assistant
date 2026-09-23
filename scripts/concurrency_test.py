"""Send authenticated, independent chat requests concurrently.

Set LOAD_TEST_TOKEN in the shell; never put a token in this file.
"""

import asyncio
import os
import time
import uuid

import httpx


URL = os.getenv("LOAD_TEST_URL", "http://localhost:8000/api/v1/chat")
REQUEST_COUNT = int(os.getenv("LOAD_TEST_REQUESTS", "4"))
TIMEOUT_SECONDS = float(os.getenv("LOAD_TEST_TIMEOUT_SECONDS", "180"))


async def send_request(
    client: httpx.AsyncClient,
    index: int,
    token: str,
) -> dict[str, object]:
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
                "session_id": f"vllm-load-{index}-{uuid.uuid4()}",
            },
        )
        result: dict[str, object] = {
            "request": index,
            "status": response.status_code,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        if response.status_code != 200:
            result["error"] = response.text[:300]
    except httpx.RequestError as exc:
        result = {
            "request": index,
            "status": type(exc).__name__,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
    print(
        f"request={result['request']} status={result['status']} "
        f"elapsed={result['elapsed_seconds']}s"
    )
    return result


async def main() -> None:
    token = os.getenv("LOAD_TEST_TOKEN")
    if not token:
        raise SystemExit("Set LOAD_TEST_TOKEN to a development access token")
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        results = await asyncio.gather(*(
            send_request(client, index, token)
            for index in range(1, REQUEST_COUNT + 1)
        ))
    print(f"statuses={[result['status'] for result in results]}")
    print(f"total_elapsed={time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
