import logging

from langgraph.graph import END, START, StateGraph

from ai.agents.analytics_agent import handle_analytics_request
from ai.agents.general_agent import handle_general_request
from ai.agents.intent_router import route_request
from ai.agents.transaction_agent import handle_transaction_request
from ai.schemas.graph_state import EnterpriseState


logger = logging.getLogger(__name__)
ROUTING_CONFIDENCE_THRESHOLD = 0.65


async def router_node(state: EnterpriseState) -> dict:
    routing = await route_request(state["user_message"])
    logger.info(
        "request_id=%s node=router route=%s confidence=%.2f",
        state["request_id"],
        routing.route,
        routing.confidence,
    )
    return {
        "route": routing.route,
        "route_confidence": routing.confidence,
    }


async def general_node(state: EnterpriseState) -> dict:
    logger.info("request_id=%s node=general", state["request_id"])
    response = await handle_general_request(state["user_message"])
    return {"response": response, "tools_used": []}


async def transaction_node(state: EnterpriseState) -> dict:
    logger.info("request_id=%s node=transaction", state["request_id"])
    try:
        result = await handle_transaction_request(state["user_message"])
        return {
            "response": result["response"],
            "tools_used": result.get("tools_used", []),
        }
    except Exception:
        logger.error(
            "request_id=%s node=transaction error=transaction_handler_failed",
            state["request_id"],
        )
        return {
            "error": "transaction_handler_failed",
            "response": "Transaction data is temporarily unavailable.",
        }


async def analytics_node(state: EnterpriseState) -> dict:
    logger.info("request_id=%s node=analytics", state["request_id"])
    result = await handle_analytics_request(state["user_message"])
    return {
        "response": result["response"],
        "tools_used": result.get("tools_used", []),
    }


async def clarification_node(state: EnterpriseState) -> dict:
    logger.info("request_id=%s node=clarify", state["request_id"])
    return {
        "response": (
            "I’m not certain which capability you need. "
            "Could you clarify whether you want transaction details "
            "or business analytics?"
        ),
        "tools_used": [],
    }


async def error_node(state: EnterpriseState) -> dict:
    """Provide a safe response for future explicit error paths."""
    logger.info("request_id=%s node=error", state["request_id"])
    return {"response": "The requested service is temporarily unavailable."}


def choose_route(state: EnterpriseState) -> str:
    confidence = state.get("route_confidence") or 0.0
    if confidence < ROUTING_CONFIDENCE_THRESHOLD:
        return "clarify"

    route = state["route"]
    if route == "transaction":
        return "transaction"
    if route == "analytics":
        return "analytics"
    return "general"


graph_builder = StateGraph(EnterpriseState)
graph_builder.add_node("router", router_node)
graph_builder.add_node("transaction", transaction_node)
graph_builder.add_node("analytics", analytics_node)
graph_builder.add_node("general", general_node)
graph_builder.add_node("clarify", clarification_node)

graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges(
    "router",
    choose_route,
    {
        "transaction": "transaction",
        "analytics": "analytics",
        "general": "general",
        "clarify": "clarify",
    },
)
graph_builder.add_edge("transaction", END)
graph_builder.add_edge("analytics", END)
graph_builder.add_edge("general", END)
graph_builder.add_edge("clarify", END)

enterprise_graph = graph_builder.compile()


async def run_enterprise_graph(user_message: str, request_id: str) -> dict:
    initial_state: EnterpriseState = {
        "user_message": user_message,
        "request_id": request_id,
        "route": None,
        "route_confidence": None,
        "response": None,
        "tools_used": [],
        "error": None,
    }
    result = await enterprise_graph.ainvoke(initial_state)
    logger.info("request_id=%s graph=end", request_id)
    return result
