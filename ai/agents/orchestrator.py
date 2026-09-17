import logging
import time
import uuid

from ai.agents.intent_router import route_request
from ai.agents.transaction_agent import handle_transaction_request
from ai.agents.analytics_agent import handle_analytics_request
from ai.agents.general_agent import handle_general_request


logger = logging.getLogger(__name__)

# Experimental: model-generated confidence is not a calibrated probability.
ROUTING_CONFIDENCE_THRESHOLD = 0.65


async def process_request(user_message: str, request_id: str | None = None) -> dict:
    request_id = request_id or str(uuid.uuid4())
    start = time.perf_counter()
    route = await route_request(user_message)
    route_latency = time.perf_counter() - start
    logger.info(
        "request_id=%s route=%s confidence=%.2f route_latency=%.2fs",
        request_id, route.route, route.confidence, route_latency,
    )
    if route.confidence < ROUTING_CONFIDENCE_THRESHOLD:
        logger.info("request_id=%s status=clarification tools=[]", request_id)
        return {
            "response": (
                "I’m not certain which enterprise capability "
                "you need. Could you clarify your request?"
            ),
            "tools_used": [],
            "route": route.route,
            "route_confidence": route.confidence,
        }

    if route.route == "transaction":
        result = await handle_transaction_request(user_message)
    elif route.route == "analytics":
        result = await handle_analytics_request(user_message)
    else:
        response = await handle_general_request(user_message)
        result = {"response": response, "tools_used": []}

    return {
        **result,
        "route": route.route,
        "route_confidence": route.confidence,
    }
