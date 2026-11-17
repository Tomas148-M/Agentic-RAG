from sentence_transformers import SentenceTransformer

from app.integrations.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """Generate normalized vectors with a local Sentence Transformers model."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name, device="cpu")

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
