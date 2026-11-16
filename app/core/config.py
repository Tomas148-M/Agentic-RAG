from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment and an optional .env file.

    Attributes:
        app_name (str): Application name exposed by the API.
        debug (bool): Whether FastAPI debug mode is enabled.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENTIC_RAG_", extra="ignore")

    app_name: str = "Agentic RAG"
    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings for this process.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()
