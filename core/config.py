"""Validated application settings, loaded once per process."""

from functools import lru_cache
from typing import Literal
from urllib.parse import quote

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: Literal["development", "testing", "staging", "production"] = (
        "development"
    )
    database_url: str = Field(default="", repr=False)
    database_host: str = "localhost"
    database_port: int = Field(default=5432, ge=1, le=65535)
    database_name: str = "enterprise_ai"
    database_user: str = "enterprise_user"
    database_password: str | None = Field(default=None, repr=False)
    langgraph_database_url: str | None = Field(default=None, repr=False)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "replace-with-model-name"
    jwt_secret: str = Field(
        validation_alias=AliasChoices("JWT_SECRET_KEY", "JWT_SECRET"),
        repr=False,
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = Field(default=30, ge=1)
    api_base_url: str = "http://localhost:8000"

    model_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=300,
        validation_alias=AliasChoices(
            "MODEL_TIMEOUT_SECONDS", "LLM_REQUEST_TIMEOUT_SECONDS"
        ),
    )
    model_max_retries: int = Field(default=1, ge=0, le=5)
    max_tool_calls_per_request: int = Field(default=5, ge=1, le=20)
    tool_max_retries: int = Field(default=2, ge=0, le=5)
    database_max_retries: int = Field(default=2, ge=0, le=5)
    tool_timeout_seconds: float = Field(default=30.0, gt=0)
    retry_base_delay_seconds: float = Field(default=0.5, gt=0)
    retry_max_delay_seconds: float = Field(default=4.0, gt=0)
    health_check_timeout_seconds: float = Field(default=5.0, gt=0)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_connections_and_delays(self) -> "Settings":
        # Older local .env files use DATABASE_* parts instead of DATABASE_URL.
        if not self.database_url:
            if not self.database_password:
                raise ValueError("DATABASE_URL or DATABASE_PASSWORD is required")
            self.database_url = (
                "postgresql+psycopg2://"
                f"{quote(self.database_user, safe='')}:"
                f"{quote(self.database_password, safe='')}@"
                f"{self.database_host}:{self.database_port}/"
                f"{quote(self.database_name, safe='')}"
            )
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError(
                "RETRY_MAX_DELAY_SECONDS must be greater than or equal to "
                "RETRY_BASE_DELAY_SECONDS"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Keep existing imports working while consumers migrate to settings.*.
MODEL_TIMEOUT_SECONDS = settings.model_timeout_seconds
MODEL_MAX_RETRIES = settings.model_max_retries
TOOL_MAX_RETRIES = settings.tool_max_retries
DATABASE_MAX_RETRIES = settings.database_max_retries
TOOL_TIMEOUT_SECONDS = settings.tool_timeout_seconds
RETRY_BASE_DELAY_SECONDS = settings.retry_base_delay_seconds
RETRY_MAX_DELAY_SECONDS = settings.retry_max_delay_seconds
HEALTH_CHECK_TIMEOUT_SECONDS = settings.health_check_timeout_seconds
