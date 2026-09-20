import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from core.config import MODEL_TIMEOUT_SECONDS


load_dotenv()


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "replace-with-model-name",
)


def get_chat_model():

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,
        client_kwargs={"timeout": MODEL_TIMEOUT_SECONDS},
    )
