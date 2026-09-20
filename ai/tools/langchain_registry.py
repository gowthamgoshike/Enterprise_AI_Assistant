from ai.tools.analytics_tools import (
    get_revenue_summary,
    get_top_products,
    get_transaction_metrics,
)
from ai.tools.transaction_tools import (
    get_my_transactions,
    get_transaction,
)
from ai.tools.refund_tools import propose_refund
from services.auth_service.models import UserRole


CUSTOMER_TOOLS = [get_my_transactions]

TRANSACTION_TOOLS = [get_transaction]

ANALYTICS_TOOLS = [
    get_top_products,
    get_revenue_summary,
    get_transaction_metrics,
]
ACTION_TOOLS = [propose_refund]

TRANSACTION_TOOL_MAP = {tool.name: tool for tool in TRANSACTION_TOOLS}
ANALYTICS_TOOL_MAP = {tool.name: tool for tool in ANALYTICS_TOOLS}
ALL_TOOLS = [*TRANSACTION_TOOLS, *ANALYTICS_TOOLS, *ACTION_TOOLS]

ROLE_TOOLS = {
    UserRole.CUSTOMER: [*CUSTOMER_TOOLS, *TRANSACTION_TOOLS],
    UserRole.ANALYST: [*TRANSACTION_TOOLS, *ANALYTICS_TOOLS],
    UserRole.ADMIN: [*TRANSACTION_TOOLS, *ANALYTICS_TOOLS, *ACTION_TOOLS],
}


def get_tools_for_role(
    role: UserRole,
) -> list:
    return list(ROLE_TOOLS.get(role, []))

# Compatibility for the existing chat service.
TOOLS = [*CUSTOMER_TOOLS, *ALL_TOOLS]
TOOL_MAP = {tool.name: tool for tool in TOOLS}
