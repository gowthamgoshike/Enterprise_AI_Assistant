"""Exact evaluators for tool choice and structured arguments."""

from typing import Any

from ai.evaluations.schemas import ToolCallTrace


def evaluate_tool_selection(
    expected_tool: str | None,
    actual_tools: list[str],
) -> bool:
    if expected_tool is None:
        return not actual_tools
    return expected_tool in actual_tools


def _contains(expected: Any, actual: Any) -> bool:
    """Recursively compare an expected subset with actual arguments."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            _contains(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual)
        )
    return expected == actual


def evaluate_tool_arguments(
    expected_tool: str | None,
    expected_arguments: dict[str, Any] | None,
    actual_tool_calls: list[ToolCallTrace],
) -> bool:
    if expected_arguments is None:
        return True
    return any(
        call.name == expected_tool
        and _contains(expected_arguments, call.arguments)
        for call in actual_tool_calls
    )
