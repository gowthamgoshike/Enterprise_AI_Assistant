from functools import lru_cache

from ai.models.factory import create_model_provider
from ai.models.gateway import ModelGateway
from core.config import settings


@lru_cache
def get_model_gateway() -> ModelGateway:
    return ModelGateway(create_model_provider(settings.model_provider), settings)
