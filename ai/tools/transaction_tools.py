from services.transaction_service.identifiers import normalize_transaction_id
from sqlalchemy.exc import (
    SQLAlchemyError,
)
from ai.tools.exceptions import (
    ToolExecutionError,
)
from ai.schemas.tools import (
    TransactionLookupResult,
    ToolResult,
)
from langchain_core.tools import tool

from services.transaction_service.service import find_transaction


def get_transaction_tool(
    transaction_id: str,
) -> ToolResult:

    normalized_transaction_id = (
        normalize_transaction_id(transaction_id)
    )

    if not normalized_transaction_id:

        raise ValueError(
            "transaction_id must not be blank"
        )

    try:

        raw_result = find_transaction(
            normalized_transaction_id
        )

    except SQLAlchemyError as exc:

        raise ToolExecutionError(
            code="DATABASE_UNAVAILABLE",
            message="The transaction database is unavailable.",
            retryable=True,
        ) from exc

    lookup_result = TransactionLookupResult.model_validate(
        raw_result
    )

    return ToolResult(
        success=True,
        tool_name="get_transaction",
        data=lookup_result.model_dump(mode="json"),
    )

@tool
def get_transaction(
    transaction_id: str,
) -> dict:
    """
    Retrieve the current state of a
    transaction by transaction ID.

    Use this tool only when the user asks
    about one specific transaction ID, not aggregate analytics.
    The transaction_id is required; UUID IDs use a TX- prefix and lowercase UUID.
    Returns found and the matching transaction, or no match.
    """

    normalized_id = (
        normalize_transaction_id(transaction_id)
    )

    raw_result = find_transaction(
        normalized_id
    )
    return TransactionLookupResult.model_validate(
        raw_result
    ).model_dump(mode="json")


@tool
def get_my_transactions(
    limit: int = 10,
) -> dict:
    """
    Return the authenticated customer's most recent transactions.

    Use this tool when a customer asks for their transactions or transaction
    history. The limit defaults to 10 and is clamped to 1 through 20. Customer
    identity is supplied by the trusted application runtime, not tool arguments.
    """

    raise RuntimeError(
        "get_my_transactions requires trusted execution context"
    )
