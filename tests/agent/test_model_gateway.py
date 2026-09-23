import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from ai.models.dependencies import get_model_gateway
from ai.prompts.router_prompts import INTENT_ROUTER_PROMPT
from ai.schemas.agent import IntentRoute


@pytest.mark.asyncio(loop_scope="module")
async def test_router_structured_output_through_gateway():
    router = get_model_gateway().with_structured_output(IntentRoute)
    result = await router.ainvoke([
        SystemMessage(content=INTENT_ROUTER_PROMPT),
        HumanMessage(content="Explain Kafka in one sentence."),
    ])

    assert isinstance(result, IntentRoute)
