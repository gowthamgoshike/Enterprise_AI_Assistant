from services.transaction_service.service import (
    find_transaction,
)
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

def get_transaction_tool(
    transaction_id: str,
) -> ToolResult:

    normalized_transaction_id = (
        transaction_id.strip().upper()
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

