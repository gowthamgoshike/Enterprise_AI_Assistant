from collections.abc import Callable
from dataclasses import dataclass

from ai.tools.secure_executors import (
    execute_get_my_transactions,
    execute_get_revenue_summary,
    execute_get_top_products,
    execute_get_transaction,
    execute_get_transaction_metrics,
)
from services.auth_service.permissions import Permission


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    function: Callable
    permission: Permission | None


SECURE_TOOL_REGISTRY = {
    "get_my_transactions": ToolDefinition(
        name="get_my_transactions",
        function=execute_get_my_transactions,
        permission=Permission.READ_OWN_TRANSACTIONS,
    ),
    "get_transaction": ToolDefinition(
        name="get_transaction",
        function=execute_get_transaction,
        # Authorization is role-sensitive: customers receive ownership-scoped
        # access while analysts and admins require READ_TRANSACTION.
        permission=None,
    ),
    "get_top_products": ToolDefinition(
        name="get_top_products",
        function=execute_get_top_products,
        permission=Permission.READ_ANALYTICS,
    ),
    "get_revenue_summary": ToolDefinition(
        name="get_revenue_summary",
        function=execute_get_revenue_summary,
        permission=Permission.READ_REVENUE,
    ),
    "get_transaction_metrics": ToolDefinition(
        name="get_transaction_metrics",
        function=execute_get_transaction_metrics,
        permission=Permission.READ_ANALYTICS,
    ),
}
