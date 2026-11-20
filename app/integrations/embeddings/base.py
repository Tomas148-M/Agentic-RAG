from abc import ABC, abstractmethod


class EmbeddingProviderError(Exception):
    """An embedding backend failed to produce valid vectors."""


class EmbeddingProvider(ABC):
    """Interface for generating vectors from text."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts in order.

        Args:
            texts (list[str]): Texts to encode.

        Returns:
            list[list[float]]: One vector for each input text.
        """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode stored document text, defaulting to the general encoder."""
        return self.embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Encode search queries, defaulting to the general encoder."""
        return self.embed(texts)
