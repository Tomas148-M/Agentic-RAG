from abc import ABC, abstractmethod


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
