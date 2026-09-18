import asyncio
import uuid

from ai.graphs.enterprise_graph import (
    run_enterprise_graph,
)


async def main():
    result = await run_enterprise_graph(
        user_message="Show me the top 5 products.",
        request_id=str(uuid.uuid4()),
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
