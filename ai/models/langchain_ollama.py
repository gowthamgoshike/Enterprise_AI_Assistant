from langchain_ollama import ChatOllama

from core.config import settings

OLLAMA_MODEL = settings.ollama_model


def get_chat_model():

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        reasoning=False,
        client_kwargs={"timeout": settings.model_timeout_seconds},
    )
