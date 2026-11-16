from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the environment and an optional .env file.

    Attributes:
        app_name (str): Application name exposed by the API.
        debug (bool): Whether FastAPI debug mode is enabled.
        llm_provider (str): Selected text generation provider.
        ollama_base_url (str): Ollama server address.
        ollama_model (str): Installed Ollama model name.
        codex_executable (str): Codex CLI executable on the server.
        codex_model (str | None): Optional Codex model override.
        llm_timeout_seconds (float): Maximum wait for provider responses.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENTIC_RAG_", extra="ignore")

    app_name: str = "Agentic RAG"
    debug: bool = False
    llm_provider: Literal["ollama", "codex"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    codex_executable: str = "codex"
    codex_model: str | None = None
    llm_timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings for this process.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()
