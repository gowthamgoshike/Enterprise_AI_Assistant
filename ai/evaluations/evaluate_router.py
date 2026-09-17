"""Run with: python -m ai.evaluations.evaluate_router

Requires the configured Ollama model. Evaluates route accuracy only;
confidence is not a calibrated probability.
"""
import asyncio

from ai.agents.intent_router import route_request
from ai.evaluations.routing_cases import ROUTING_CASES


async def evaluate():
    correct = 0
    for case in ROUTING_CASES:
        result = await route_request(case["question"])
        passed = result.route == case["expected"]
        if passed:
            correct += 1
        print(case["question"])
        print("Expected:", case["expected"])
        print("Actual:", result.route)
        print("Confidence:", result.confidence)
        print("PASS" if passed else "FAIL")
        print()

    accuracy = correct / len(ROUTING_CASES)
    print(f"Routing accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    asyncio.run(evaluate())
