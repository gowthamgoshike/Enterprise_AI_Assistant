from langchain_core.messages import HumanMessage, SystemMessage

from ai.models.langchain_ollama import get_chat_model
from ai.prompts.router_prompts import INTENT_ROUTER_PROMPT
from ai.schemas.agent import IntentRoute


model = get_chat_model()
router_model = model.with_structured_output(IntentRoute)


async def route_request(user_message: str) -> IntentRoute:
    messages = [
        SystemMessage(content=INTENT_ROUTER_PROMPT),
        HumanMessage(content=user_message),
    ]
    result = await router_model.ainvoke(messages)
    return result
