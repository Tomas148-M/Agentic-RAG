import math

import httpx

from app.core.config import (
    DEFAULT_EMBEDDING_DOCUMENT_PREFIX,
    DEFAULT_EMBEDDING_QUERY_PREFIX,
    OLLAMA_EMBED_COUNT_ERROR,
    OLLAMA_EMBED_DIMENSION_ERROR,
    OLLAMA_EMBED_INPUT_FIELD,
    OLLAMA_EMBED_MODEL_FIELD,
    OLLAMA_EMBED_PATH,
    OLLAMA_EMBED_REQUEST_ERROR,
    OLLAMA_EMBED_RESPONSE_ERROR,
    OLLAMA_EMBEDDINGS_FIELD,
)
from app.integrations.embeddings.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate normalized vectors through Ollama's native embedding API."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
        document_prefix: str = DEFAULT_EMBEDDING_DOCUMENT_PREFIX,
        query_prefix: str = DEFAULT_EMBEDDING_QUERY_PREFIX,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}{OLLAMA_EMBED_PATH}"
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed stored chunks with the model's document task prefix."""
        return self.embed([f"{self.document_prefix}{text}" for text in texts])

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed searches with the model's query task prefix."""
        return self.embed([f"{self.query_prefix}{text}" for text in texts])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one Ollama embedding for each input text.

        Args:
            texts: Text strings to embed in one batch.

        Returns:
            Normalized vectors in input order.

        Raises:
            EmbeddingProviderError: The request or returned vectors are invalid.
        """
        if not texts:
            return []

        try:
            response = httpx.post(
                self.endpoint,
                json={
                    OLLAMA_EMBED_MODEL_FIELD: self.model_name,
                    OLLAMA_EMBED_INPUT_FIELD: texts,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingProviderError(OLLAMA_EMBED_REQUEST_ERROR) from exc

        if not isinstance(payload, dict):
            raise EmbeddingProviderError(OLLAMA_EMBED_RESPONSE_ERROR)
        embeddings = payload.get(OLLAMA_EMBEDDINGS_FIELD)
        if not isinstance(embeddings, list):
            raise EmbeddingProviderError(OLLAMA_EMBED_RESPONSE_ERROR)
        if len(embeddings) != len(texts):
            raise EmbeddingProviderError(OLLAMA_EMBED_COUNT_ERROR)

        vectors: list[list[float]] = []
        expected_dimensions: int | None = None
        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise EmbeddingProviderError(OLLAMA_EMBED_RESPONSE_ERROR)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            ):
                raise EmbeddingProviderError(OLLAMA_EMBED_RESPONSE_ERROR)
            if expected_dimensions is None:
                expected_dimensions = len(vector)
            elif len(vector) != expected_dimensions:
                raise EmbeddingProviderError(OLLAMA_EMBED_DIMENSION_ERROR)
            vectors.append([float(value) for value in vector])
        return vectors
