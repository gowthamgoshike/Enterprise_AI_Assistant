"""Conservative lexical checks for grounded response facts."""

from ai.evaluations.schemas import ExpectedFact


def evaluate_response(
    response: str | None,
    *,
    expected_contains: list[str],
    forbidden_contains: list[str],
    expected_facts: list[ExpectedFact],
) -> tuple[bool, list[str]]:
    normalized = (response or "").casefold()
    failures = []
    for phrase in expected_contains:
        if phrase.casefold() not in normalized:
            failures.append(f"response missing expected text: {phrase!r}")
    for phrase in forbidden_contains:
        if phrase.casefold() in normalized:
            failures.append(f"response contained forbidden text: {phrase!r}")
    for fact in expected_facts:
        if fact.value.casefold() not in normalized:
            failures.append(
                f"response missing expected fact {fact.name}={fact.value!r}"
            )
    return not failures, failures
