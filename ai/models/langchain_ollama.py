import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "replace-with-model-name",
)


def get_chat_model():

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,
    )