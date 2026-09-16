from ai.tools.analytics_tools import (
    get_revenue_summary,
    get_top_products,
    get_transaction_metrics,
)
from ai.tools.transaction_tools import (
    get_transaction,
)


TOOLS = [
    get_transaction,
    get_top_products,
    get_revenue_summary,
    get_transaction_metrics,
]

TOOL_MAP = {
    tool.name: tool
    for tool in TOOLS
}