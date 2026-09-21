"""Deterministic workflow and error-state checks."""


def evaluate_workflow(
    expected_status: str | None,
    actual_status: str | None,
    expected_error_code: str | None,
    actual_error_code: str | None,
    expected_approval_status: str | None = None,
    actual_approval_status: str | None = None,
) -> tuple[bool, list[str]]:
    failures = []
    if expected_status is not None and actual_status != expected_status:
        failures.append(
            f"expected status {expected_status!r}, got {actual_status!r}"
        )
    if expected_error_code is not None and actual_error_code != expected_error_code:
        failures.append(
            f"expected error {expected_error_code!r}, got {actual_error_code!r}"
        )
    if (
        expected_approval_status is not None
        and actual_approval_status != expected_approval_status
    ):
        failures.append(
            "expected approval status "
            f"{expected_approval_status!r}, got {actual_approval_status!r}"
        )
    return not failures, failures
