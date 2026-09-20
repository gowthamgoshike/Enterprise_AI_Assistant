from collections.abc import Callable
from dataclasses import dataclass

from ai.tools.secure_executors import (
    execute_get_my_transactions,
    execute_get_revenue_summary,
    execute_get_top_products,
    execute_get_transaction,
    execute_get_transaction_metrics,
    execute_propose_refund,
)
from services.auth_service.permissions import Permission


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    function: Callable
    permission: Permission | None
    requires_approval: bool = False
    retry_safe: bool = False


SECURE_TOOL_REGISTRY = {
    "get_my_transactions": ToolDefinition(
        name="get_my_transactions",
        function=execute_get_my_transactions,
        permission=Permission.READ_OWN_TRANSACTIONS,
        retry_safe=True,
    ),
    "get_transaction": ToolDefinition(
        name="get_transaction",
        function=execute_get_transaction,
        # Authorization is role-sensitive: customers receive ownership-scoped
        # access while analysts and admins require READ_TRANSACTION.
        permission=None,
        retry_safe=True,
    ),
    "get_top_products": ToolDefinition(
        name="get_top_products",
        function=execute_get_top_products,
        permission=Permission.READ_ANALYTICS,
        retry_safe=True,
    ),
    "get_revenue_summary": ToolDefinition(
        name="get_revenue_summary",
        function=execute_get_revenue_summary,
        permission=Permission.READ_REVENUE,
        retry_safe=True,
    ),
    "get_transaction_metrics": ToolDefinition(
        name="get_transaction_metrics",
        function=execute_get_transaction_metrics,
        permission=Permission.READ_ANALYTICS,
        retry_safe=True,
    ),
    "propose_refund": ToolDefinition(
        name="propose_refund",
        function=execute_propose_refund,
        permission=Permission.PROPOSE_REFUND,
        requires_approval=True,
    ),
}
