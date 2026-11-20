from app.core.config import (
    DEFAULT_EMBEDDING_PROVIDER,
    EMBEDDING_ID_PREFIX_SEPARATOR,
    EMBEDDING_ID_SEPARATOR,
    EMBEDDING_PROVIDER_UNSUPPORTED_ERROR,
    SENTENCE_TRANSFORMERS_PROVIDER,
    Settings,
)
from app.integrations.embeddings.base import EmbeddingProvider
from app.integrations.embeddings.ollama import OllamaEmbeddingProvider


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the embedding adapter selected by application settings."""
    if settings.embedding_provider == DEFAULT_EMBEDDING_PROVIDER:
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model,
            timeout_seconds=settings.embedding_timeout_seconds,
            document_prefix=settings.embedding_document_prefix,
            query_prefix=settings.embedding_query_prefix,
        )
    if settings.embedding_provider == SENTENCE_TRANSFORMERS_PROVIDER:
        from app.integrations.embeddings.local import LocalEmbeddingProvider

        return LocalEmbeddingProvider(
            settings.embedding_model,
            document_prefix=settings.embedding_document_prefix,
            query_prefix=settings.embedding_query_prefix,
        )
    raise ValueError(EMBEDDING_PROVIDER_UNSUPPORTED_ERROR)


def get_embedding_identifier(settings: Settings) -> str:
    """Return the provider and model identity stored with Chroma vectors."""
    model_id = EMBEDDING_ID_SEPARATOR.join(
        (settings.embedding_provider, settings.embedding_model)
    )
    return EMBEDDING_ID_PREFIX_SEPARATOR.join(
        (
            model_id,
            settings.embedding_document_prefix,
            settings.embedding_query_prefix,
        )
    )
