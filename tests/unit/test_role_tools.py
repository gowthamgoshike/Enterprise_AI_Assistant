from ai.tools.langchain_registry import get_tools_for_role
from services.auth_service.models import UserRole


def tool_names(role):
    return {tool.name for tool in get_tools_for_role(role)}


def test_customer_sees_only_customer_safe_transaction_tools():
    names = tool_names(UserRole.CUSTOMER)

    assert names == {
        "get_my_transactions",
        "get_transaction",
    }
    assert "get_revenue_summary" not in names
    assert "get_top_products" not in names


def test_analyst_sees_transaction_and_analytics_tools():
    assert tool_names(UserRole.ANALYST) == {
        "get_transaction",
        "get_top_products",
        "get_revenue_summary",
        "get_transaction_metrics",
    }


def test_only_admin_sees_refund_proposal_tool():
    assert "propose_refund" in tool_names(UserRole.ADMIN)
    assert "propose_refund" not in tool_names(UserRole.ANALYST)
    assert "propose_refund" not in tool_names(UserRole.CUSTOMER)
