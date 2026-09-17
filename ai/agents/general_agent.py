from langchain_core.messages import HumanMessage, SystemMessage

from ai.models.langchain_ollama import get_chat_model


model = get_chat_model()

GENERAL_SYSTEM_PROMPT = """
You are an enterprise AI engineering
assistant.

Answer general conceptual questions
clearly and concisely.

Do not invent enterprise-specific
transaction or analytics data.
"""


async def handle_general_request(user_message: str) -> str:
    response = await model.ainvoke([
        SystemMessage(content=GENERAL_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    return response.content
