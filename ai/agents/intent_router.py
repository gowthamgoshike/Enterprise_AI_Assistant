from langchain_core.messages import HumanMessage, SystemMessage

from ai.models.dependencies import get_model_gateway
from ai.prompts.router_prompts import INTENT_ROUTER_PROMPT
from ai.schemas.agent import IntentRoute


model = get_model_gateway()
router_model = model.with_structured_output(IntentRoute)


async def route_request(user_message: str) -> IntentRoute:
    messages = [
        SystemMessage(content=INTENT_ROUTER_PROMPT),
        HumanMessage(content=user_message),
    ]
    result = await router_model.ainvoke(messages)
    return result
