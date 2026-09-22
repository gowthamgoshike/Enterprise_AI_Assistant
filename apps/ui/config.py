"""UI-only settings; the UI container does not receive API secrets."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class UISettings(BaseSettings):
    api_base_url: str = "http://localhost:8000"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


API_BASE_URL = UISettings().api_base_url
