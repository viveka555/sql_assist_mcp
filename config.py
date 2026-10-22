
"""
Configuration settings for the AI SQL Assistant.

Reads application settings from environment variables and .env.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Groq LLM settings.
    groq_api_key: SecretStr
    groq_model: str = "openai/gpt-oss-20b"

    # PostgreSQL connection.
    database_url: SecretStr

    # API authentication.
    api_token: SecretStr

    # Query safety limits.
    sql_max_rows: int = 100
    sql_timeout_seconds: int = 10

    def validate_limits(self) -> None:
        """Validate SQL execution limits."""

        if self.sql_max_rows < 1:
            raise ValueError("SQL_MAX_ROWS must be greater than zero.")

        if self.sql_timeout_seconds < 1:
            raise ValueError(
                "SQL_TIMEOUT_SECONDS must be greater than zero."
            )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings()
    settings.validate_limits()

    return settings
