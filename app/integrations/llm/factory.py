from app.core.config import Settings
from app.integrations.llm.base import LLMProvider
from app.integrations.llm.codex import CodexProvider
from app.integrations.llm.ollama import OllamaProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Create the provider selected by the validated application settings.

    Args:
        settings (Settings): Provider selection and connection settings.

    Returns:
        LLMProvider: Configured text generation provider.
    """
    if settings.llm_provider == "codex":
        return CodexProvider(settings.codex_executable, settings.codex_model, settings.llm_timeout_seconds)
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
