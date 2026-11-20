from sentence_transformers import SentenceTransformer

from app.core.config import EMPTY_EMBEDDING_PREFIX
from app.integrations.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """Generate normalized vectors with a local Sentence Transformers model."""

    def __init__(
        self,
        model_name: str,
        document_prefix: str = EMPTY_EMBEDDING_PREFIX,
        query_prefix: str = EMPTY_EMBEDDING_PREFIX,
    ) -> None:
        self.model = SentenceTransformer(model_name, device="cpu")
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode stored document text with the configured task prefix."""
        return self.embed([f"{self.document_prefix}{text}" for text in texts])

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Encode search query text with the configured task prefix."""
        return self.embed([f"{self.query_prefix}{text}" for text in texts])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized embedding per input text.

        Args:
            texts (list[str]): Texts to encode.

        Returns:
            list[list[float]]: Normalized embedding vectors in input order.
        """
        if not texts:
            return []
        return self.model.encode(texts, normalize_embeddings=True).tolist()
