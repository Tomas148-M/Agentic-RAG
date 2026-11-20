from functools import lru_cache
from threading import Lock
from typing import Annotated

from fastapi import Depends, HTTPException

from app.agents.tools.registry import ToolRegistry
from app.core.config import (
    EMBEDDING_PROVIDER_CACHE_SIZE,
    Settings,
    get_settings,
)
from app.integrations.embeddings.base import EmbeddingProvider
from app.integrations.embeddings.factory import (
    create_embedding_provider,
    get_embedding_identifier,
)
from app.integrations.llm.base import LLMProvider
from app.integrations.llm.factory import create_llm_provider
from app.observability.phoenix import PhoenixObservability
from app.rag.retriever import DocumentRetriever
from app.rag.vector_store import ChromaVectorStore
from app.services.chat_service import ChatService

__all__ = ["get_chat_service", "get_llm_provider", "get_settings"]
_embedding_model_lock = Lock()


def get_llm_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMProvider:
    """Create a provider using settings supplied by dependency injection.

    Args:
        settings (Settings): Configured provider settings.

    Returns:
        LLMProvider: Configured text generation provider.
    """
    observability = PhoenixObservability(settings)
    return observability.instrument(create_llm_provider(settings))


@lru_cache(maxsize=EMBEDDING_PROVIDER_CACHE_SIZE)
def _load_embedding_provider(
    provider_name: str,
    model_name: str,
    ollama_base_url: str,
    timeout_seconds: float,
    document_prefix: str,
    query_prefix: str,
) -> EmbeddingProvider:
    """Cache embedding clients and local model instances per configuration."""
    settings = Settings(
        embedding_provider=provider_name,
        embedding_model=model_name,
        ollama_base_url=ollama_base_url,
        embedding_timeout_seconds=timeout_seconds,
        embedding_document_prefix=document_prefix,
        embedding_query_prefix=query_prefix,
        _env_file=None,
    )
    return create_embedding_provider(settings)


class _LazyEmbeddings(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Create and cache the configured provider when vectors are needed."""
        return self._provider().embed(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Forward document-specific encoding to the configured provider."""
        return self._provider().embed_documents(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Forward query-specific encoding to the configured provider."""
        return self._provider().embed_queries(texts)

    def _provider(self) -> EmbeddingProvider:
        """Create and cache the configured provider when vectors are needed."""
        # lru_cache alone does not prevent concurrent cache misses from loading
        # the same large model multiple times.
        with _embedding_model_lock:
            return _load_embedding_provider(
                self.settings.embedding_provider,
                self.settings.embedding_model,
                self.settings.ollama_base_url,
                self.settings.embedding_timeout_seconds,
                self.settings.embedding_document_prefix,
                self.settings.embedding_query_prefix,
            )


def get_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentRetriever:
    """Create a document retriever using configured local storage.

    Args:
        settings (Settings): Storage and embedding settings.

    Returns:
        DocumentRetriever: Retriever for the configured collection.
    """
    try:
        store = ChromaVectorStore(
            settings.chroma_path, settings.chroma_collection,
            get_embedding_identifier(settings),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Document store unavailable"
        ) from exc
    return DocumentRetriever(store, _LazyEmbeddings(settings))


def get_chat_service(
    retriever: Annotated[DocumentRetriever, Depends(get_retriever)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatService:
    """Build a chat service from the retriever and selected LLM provider.

    Args:
        retriever (DocumentRetriever): Document search dependency.
        llm (LLMProvider): Configured text-generation provider.
        settings (Settings): Selected chat mode and retry settings.

    Returns:
        ChatService: Service configured for the current request.
    """
    return ChatService(
        retriever,
        llm,
        settings.rag_mode,
        settings.agentic_max_retries,
        ToolRegistry(retriever, settings),
        PhoenixObservability(settings),
    )
