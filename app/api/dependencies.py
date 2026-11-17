from functools import lru_cache
from threading import Lock
from typing import Annotated

from fastapi import Depends, HTTPException

from app.core.config import Settings, get_settings
from app.integrations.embeddings.base import EmbeddingProvider
from app.integrations.llm.base import LLMProvider
from app.integrations.llm.factory import create_llm_provider
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
    return create_llm_provider(settings)


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str) -> EmbeddingProvider:
    from app.integrations.embeddings.local import LocalEmbeddingProvider

    return LocalEmbeddingProvider(model_name)


class _LazyEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Load the configured model when vectors are actually needed."""
        # lru_cache alone does not prevent concurrent cache misses from loading
        # the same large model multiple times.
        with _embedding_model_lock:
            model = _load_embedding_model(self.model_name)
        return model.embed(texts)


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
            settings.embedding_model,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Document store unavailable"
        ) from exc
    return DocumentRetriever(store, _LazyEmbeddings(settings.embedding_model))


def get_chat_service(
    retriever: Annotated[DocumentRetriever, Depends(get_retriever)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> ChatService:
    """Build a chat service from the retriever and selected LLM provider.

    Args:
        retriever (DocumentRetriever): Document search dependency.
        llm (LLMProvider): Configured text-generation provider.

    Returns:
        ChatService: Service configured for the current request.
    """
    return ChatService(retriever, llm)
