from ai.models.dependencies import get_model_gateway
from core.config import settings


OLLAMA_MODEL = settings.model_name


def get_chat_model():
    """Compatibility shim; new code should import get_model_gateway directly."""
    return get_model_gateway()
