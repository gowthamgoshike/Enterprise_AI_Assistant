"""Configuration for persistent conversation checkpoints."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_checkpoint_database_url() -> str:
    database_url = os.getenv("LANGGRAPH_DATABASE_URL")
    if not database_url or not database_url.strip():
        raise RuntimeError("LANGGRAPH_DATABASE_URL is not configured")
    return database_url
