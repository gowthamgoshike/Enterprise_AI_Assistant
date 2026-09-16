"""Inspect initial tool selection without executing tools.

Run from the project root:
    python -m ai.evaluations.tool_routing_eval

Requires the application's configured Ollama model to be running.
This evaluates routing only, not database results or final answer quality.
"""

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage

from apps.api.services.langchain_agent_service import (
    SYSTEM_PROMPT,
    model_with_tools,
)


TEST_CASES = [
    {
        "question": "Explain Kafka.",
        "expected_tool": None,
    },
    {
        "question": "What is the status of TX-123?",
        "expected_tool": "get_transaction",
    },
    {
        "question": "Show me the top 5 products.",
        "expected_tool": "get_top_products",
    },
    {
        "question": "Give me the revenue summary.",
        "expected_tool": "get_revenue_summary",
    },
    {
        "question": "How many transactions failed?",
        "expected_tool": "get_transaction_metrics",
    },
]


async def main() -> None:
    print("Tool routing evaluation (no tools are executed)")

    for case in TEST_CASES:
        print(f"\nQuestion: {case['question']}")
        print(f"Expected tool: {case['expected_tool'] or '(none)'}")

        try:
            response = await model_with_tools.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=case["question"]),
            ])
        except Exception as exc:
            print(f"Routing error: {type(exc).__name__}: {exc}")
            continue

        selected_tools = [call["name"] for call in response.tool_calls]
        print(f"Selected tools: {json.dumps(selected_tools)}")
        for call in response.tool_calls:
            print(f"  {call['name']} arguments: {json.dumps(call['args'], default=str)}")
        if response.invalid_tool_calls:
            print(f"Invalid tool calls: {response.invalid_tool_calls}")
        if response.content:
            print(f"Model response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
