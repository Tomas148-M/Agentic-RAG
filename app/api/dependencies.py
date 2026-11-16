from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.integrations.llm.base import LLMProvider
from app.integrations.llm.factory import create_llm_provider

__all__ = ["get_llm_provider", "get_settings"]


def get_llm_provider(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    """Create a provider using settings supplied by dependency injection.

    Args:
        settings (Settings): Configured provider settings.

    Returns:
        LLMProvider: Configured text generation provider.
    """
    return create_llm_provider(settings)
