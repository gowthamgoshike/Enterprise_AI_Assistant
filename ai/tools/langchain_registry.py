from ai.tools.analytics_tools import (
    get_revenue_summary,
    get_top_products,
    get_transaction_metrics,
)
from ai.tools.transaction_tools import (
    get_transaction,
)


TRANSACTION_TOOLS = [get_transaction]

ANALYTICS_TOOLS = [
    get_top_products,
    get_revenue_summary,
    get_transaction_metrics,
]

TRANSACTION_TOOL_MAP = {tool.name: tool for tool in TRANSACTION_TOOLS}
ANALYTICS_TOOL_MAP = {tool.name: tool for tool in ANALYTICS_TOOLS}
ALL_TOOLS = [*TRANSACTION_TOOLS, *ANALYTICS_TOOLS]

# Compatibility for the existing chat service.
TOOLS = ALL_TOOLS
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}
