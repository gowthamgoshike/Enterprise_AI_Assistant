from ai.schemas.tools import TransactionLookupResult
from ai.tools.context import ToolExecutionContext
from services.analytics_service.service import (
    fetch_revenue_summary,
    fetch_top_products,
    fetch_transaction_metrics,
)
from services.auth_service.authorization_service import require_permission
from services.auth_service.exceptions import AuthorizationError
from services.auth_service.models import UserRole
from services.auth_service.permissions import Permission
from services.transaction_service.identifiers import normalize_transaction_id
from services.transaction_service.service import (
    fetch_customer_transactions,
    find_customer_transaction,
    find_transaction,
)
from services.transaction_service.refund_service import prepare_refund_proposal


def execute_get_my_transactions(
    context: ToolExecutionContext,
    limit: int = 10,
) -> dict:
    require_permission(
        context.role,
        Permission.READ_OWN_TRANSACTIONS,
    )
    if not context.customer_id:
        raise AuthorizationError(
            "Customer identity unavailable"
        )

    bounded_limit = max(1, min(limit, 20))
    return fetch_customer_transactions(
        customer_id=context.customer_id,
        limit=bounded_limit,
    )


def execute_get_transaction(
    context: ToolExecutionContext,
    transaction_id: str,
) -> dict:
    normalized_id = normalize_transaction_id(transaction_id)
    if not normalized_id:
        raise ValueError(
            "transaction_id must not be blank"
        )

    if context.role is UserRole.CUSTOMER:
        require_permission(
            context.role,
            Permission.READ_OWN_TRANSACTIONS,
        )
        if not context.customer_id:
            raise AuthorizationError(
                "Customer identity unavailable"
            )
        result = find_customer_transaction(
            transaction_id=normalized_id,
            customer_id=context.customer_id,
        )
    else:
        require_permission(
            context.role,
            Permission.READ_TRANSACTION,
        )
        result = find_transaction(normalized_id)

    return TransactionLookupResult.model_validate(
        result
    ).model_dump(mode="json")


def execute_get_top_products(
    context: ToolExecutionContext,
    limit: int = 5,
) -> dict:
    require_permission(
        context.role,
        Permission.READ_ANALYTICS,
    )
    bounded_limit = max(1, min(limit, 20))
    return fetch_top_products(limit=bounded_limit)


def execute_get_revenue_summary(
    context: ToolExecutionContext,
) -> dict:
    require_permission(
        context.role,
        Permission.READ_REVENUE,
    )
    return fetch_revenue_summary()


def execute_get_transaction_metrics(
    context: ToolExecutionContext,
) -> dict:
    require_permission(
        context.role,
        Permission.READ_ANALYTICS,
    )
    return fetch_transaction_metrics()


def execute_propose_refund(
    context: ToolExecutionContext,
    transaction_id: str,
    reason: str,
) -> dict:
    require_permission(
        context.role,
        Permission.PROPOSE_REFUND,
    )
    proposal = prepare_refund_proposal(
        transaction_id=transaction_id,
        reason=reason,
        requested_by=context.user_id,
    )
    return proposal.model_dump(mode="json")
