"""Configuration for persistent conversation checkpoints."""

from core.config import settings


def get_checkpoint_database_url() -> str:
    database_url = settings.langgraph_database_url
    if not database_url or not database_url.strip():
        raise RuntimeError("LANGGRAPH_DATABASE_URL is not configured")
    return database_url
