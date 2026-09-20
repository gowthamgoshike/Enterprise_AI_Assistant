"""Exercise two graph turns that share the same in-memory checkpoint."""

import asyncio
import uuid

from functools import partial
from langgraph.checkpoint.memory import InMemorySaver
from ai.graphs.tool_agent_graph import build_enterprise_tool_graph, run_agent_graph


async def main() -> None:
    enterprise_tool_graph = build_enterprise_tool_graph(InMemorySaver())
    run_turn = partial(
        run_agent_graph,
        graph=enterprise_tool_graph,
        user_id="USER-EXAMPLE",
        user_role="CUSTOMER",
        customer_id="CUST-EXAMPLE",
    )
    thread_id = str(uuid.uuid4())

    result1 = await run_turn(
        user_message="What is the status of TX-003e57d4-cb8c-45d0-8dff-78aa10303636?",
        request_id=str(uuid.uuid4()),
        thread_id=thread_id,
    )
    print(result1["messages"][-1].content)

    result2 = await run_turn(
        user_message="How much was it?",
        request_id=str(uuid.uuid4()),
        thread_id=thread_id,
    )
    print(result2["messages"][-1].content)

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    snapshot = enterprise_tool_graph.get_state(config)
    print(snapshot.values)

    for message in snapshot.values["messages"]:
        print(type(message).__name__, ":", message.content)

    history = list(enterprise_tool_graph.get_state_history(config))
    print("Checkpoint count:", len(history))


if __name__ == "__main__":
    asyncio.run(main())
