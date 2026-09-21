"""Connection settings shared by the API and UI."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str | None = None
    api_base_url: str = "http://localhost:8000"


settings = Settings()
