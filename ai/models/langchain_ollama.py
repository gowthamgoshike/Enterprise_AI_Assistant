from langchain_ollama import ChatOllama

from core.app_settings import settings
from core.config import MODEL_TIMEOUT_SECONDS

OLLAMA_MODEL = settings.ollama_model or "replace-with-model-name"


def get_chat_model():

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        client_kwargs={"timeout": MODEL_TIMEOUT_SECONDS},
    )
