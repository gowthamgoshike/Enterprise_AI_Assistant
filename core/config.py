"""Validated application settings, loaded once per process."""

from functools import lru_cache
from typing import Literal
from urllib.parse import quote

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    app_env: Literal["development", "testing", "staging", "production"] = (
        "development"
    )
    database_url: str = Field(default="", repr=False)
    async_database_url: str | None = Field(default=None, repr=False)
    database_host: str = "localhost"
    database_port: int = Field(default=5432, ge=1, le=65535)
    database_name: str = "enterprise_ai"
    database_user: str = "enterprise_user"
    database_password: str | None = Field(default=None, repr=False)
    langgraph_database_url: str | None = Field(default=None, repr=False)

    model_provider: Literal["ollama", "vllm"] = Field(
        default="ollama",
        validation_alias=AliasChoices("MODEL_PROVIDER", "LLM_PROVIDER"),
    )
    model_name: str = Field(
        default="replace-with-model-name",
        validation_alias=AliasChoices("MODEL_NAME", "OLLAMA_MODEL", "LLM_MODEL"),
    )
    model_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("MODEL_BASE_URL", "OLLAMA_BASE_URL"),
    )
    model_api_key: str | None = Field(default=None, repr=False)
    jwt_secret: str = Field(
        validation_alias=AliasChoices("JWT_SECRET_KEY", "JWT_SECRET"),
        repr=False,
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = Field(default=30, ge=1)
    api_base_url: str = "http://localhost:8000"
    allowed_origins: list[str] = ["http://localhost:8501"]
    max_request_bytes: int = Field(default=64 * 1024, ge=1024, le=1024 * 1024)
    redis_url: str = "redis://redis:6379/0"
    redis_cache_ttl_seconds: int = Field(default=30, ge=1)
    rate_limit_requests: int = Field(default=20, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    login_rate_limit_requests: int = Field(default=5, ge=1)
    job_rate_limit_requests: int = Field(default=5, ge=1)
    approval_rate_limit_requests: int = Field(default=10, ge=1)

    model_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=300,
        validation_alias=AliasChoices(
            "MODEL_TIMEOUT_SECONDS", "LLM_REQUEST_TIMEOUT_SECONDS"
        ),
    )
    model_max_retries: int = Field(default=1, ge=0, le=5)
    model_max_concurrency: int = Field(default=2, ge=1, le=8)
    max_tool_calls_per_request: int = Field(default=5, ge=1, le=20)
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=50)
    db_pool_timeout_seconds: int = Field(default=10, ge=1, le=60)
    tool_max_retries: int = Field(default=2, ge=0, le=5)
    database_max_retries: int = Field(default=2, ge=0, le=5)
    tool_timeout_seconds: float = Field(default=30.0, gt=0)
    retry_base_delay_seconds: float = Field(default=0.5, gt=0)
    retry_max_delay_seconds: float = Field(default=4.0, gt=0)
    health_check_timeout_seconds: float = Field(default=5.0, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )

    @property
    def ollama_model(self) -> str:
        """Backward-compatible name for older call sites."""
        return self.model_name

    @property
    def ollama_base_url(self) -> str:
        """Backward-compatible name for older call sites."""
        return self.model_base_url

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
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
        if not self.async_database_url:
            sync_url = make_url(self.database_url)
            if not sync_url.drivername.startswith("postgresql"):
                raise ValueError("DATABASE_URL must use PostgreSQL")
            self.async_database_url = sync_url.set(
                drivername="postgresql+asyncpg"
            ).render_as_string(hide_password=False)
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
