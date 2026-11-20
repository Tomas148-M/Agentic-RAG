from unittest.mock import Mock

import httpx
import pytest

from app.api.dependencies import _LazyEmbeddings
from app.core.config import (
    DEFAULT_EMBEDDING_DOCUMENT_PREFIX,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_QUERY_PREFIX,
    EMBEDDING_ID_PREFIX_SEPARATOR,
    EMBEDDING_ID_SEPARATOR,
    OLLAMA_EMBED_COUNT_ERROR,
    Settings,
)
from app.integrations.embeddings.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
)
from app.integrations.embeddings.factory import (
    create_embedding_provider,
    get_embedding_identifier,
)
from app.integrations.embeddings.ollama import OllamaEmbeddingProvider


def test_default_embeddings_use_nomic_through_ollama() -> None:
    """Use the selected Nomic model and Ollama backend by default."""
    settings = Settings(_env_file=None)

    provider = create_embedding_provider(settings)

    assert settings.embedding_provider == DEFAULT_EMBEDDING_PROVIDER
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert isinstance(provider, OllamaEmbeddingProvider)
    assert get_embedding_identifier(settings) == EMBEDDING_ID_PREFIX_SEPARATOR.join(
        (
            EMBEDDING_ID_SEPARATOR.join(
                (DEFAULT_EMBEDDING_PROVIDER, DEFAULT_EMBEDDING_MODEL)
            ),
            DEFAULT_EMBEDDING_DOCUMENT_PREFIX,
            DEFAULT_EMBEDDING_QUERY_PREFIX,
        )
    )


def test_ollama_embedding_batches_inputs_and_returns_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send one batch to Ollama and preserve returned vector order."""
    response = Mock()
    response.json.return_value = {
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(httpx, "post", post)
    provider = OllamaEmbeddingProvider(
        "http://localhost:11434/", "nomic-embed-text", 90.0
    )

    vectors = provider.embed(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    post.assert_called_once_with(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": ["first", "second"]},
        timeout=90.0,
    )
    response.raise_for_status.assert_called_once()


def test_empty_embedding_batch_skips_ollama_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an empty batch without contacting Ollama."""
    post = Mock()
    monkeypatch.setattr(httpx, "post", post)
    provider = OllamaEmbeddingProvider(
        "http://localhost:11434", "nomic-embed-text", 90.0
    )

    assert provider.embed([]) == []
    post.assert_not_called()


def test_ollama_embedding_uses_distinct_nomic_task_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefix document and query vectors for their retrieval task."""
    response = Mock()
    response.json.return_value = {"embeddings": [[0.1, 0.2]]}
    post = Mock(return_value=response)
    monkeypatch.setattr(httpx, "post", post)
    provider = OllamaEmbeddingProvider(
        "http://localhost:11434", "nomic-embed-text", 90.0
    )

    provider.embed_documents(["a passage"])
    provider.embed_queries(["a question"])

    assert post.call_args_list[0].kwargs["json"]["input"] == [
        "search_document: a passage"
    ]
    assert post.call_args_list[1].kwargs["json"]["input"] == [
        "search_query: a question"
    ]


def test_lazy_embedding_provider_forwards_task_specific_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the provider's document and Nomic query prefixes in the API."""
    provider = Mock(spec=EmbeddingProvider)
    provider.embed_documents.return_value = [[0.1, 0.2]]
    provider.embed_queries.return_value = [[0.3, 0.4]]
    load_provider = Mock(return_value=provider)
    monkeypatch.setattr(
        "app.api.dependencies._load_embedding_provider", load_provider
    )
    embeddings = _LazyEmbeddings(Settings(_env_file=None))

    assert embeddings.embed_documents(["document"]) == [[0.1, 0.2]]
    assert embeddings.embed_queries(["query"]) == [[0.3, 0.4]]

    provider.embed_documents.assert_called_once_with(["document"])
    provider.embed_queries.assert_called_once_with(["query"])


def test_ollama_embedding_rejects_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject responses that omit vectors for any input text."""
    response = Mock()
    response.json.return_value = {"embeddings": [[0.1, 0.2]]}
    monkeypatch.setattr(httpx, "post", Mock(return_value=response))
    provider = OllamaEmbeddingProvider(
        "http://localhost:11434", "nomic-embed-text", 90.0
    )

    with pytest.raises(EmbeddingProviderError, match=OLLAMA_EMBED_COUNT_ERROR):
        provider.embed(["first", "second"])


def test_ollama_embedding_translates_http_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose Ollama request failures through the embedding error type."""
    monkeypatch.setattr(
        httpx,
        "post",
        Mock(side_effect=httpx.ConnectError("offline")),
    )
    provider = OllamaEmbeddingProvider(
        "http://localhost:11434", "nomic-embed-text", 90.0
    )

    with pytest.raises(EmbeddingProviderError, match="Ollama embedding request failed"):
        provider.embed(["text"])
