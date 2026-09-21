"""Conversation-level aggregation helpers."""

from ai.evaluations.schemas import EvaluationResult


def evaluate_conversation(turns: list[EvaluationResult]) -> tuple[bool, list[str]]:
    failures = [
        f"turn {index}: {reason}"
        for index, turn in enumerate(turns, start=1)
        for reason in turn.failure_reasons
    ]
    return all(turn.passed for turn in turns), failures
