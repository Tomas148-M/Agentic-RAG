from dataclasses import dataclass

from app.integrations.embeddings.base import EmbeddingProvider
from app.rag.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved document chunk and its provenance."""

    text: str
    source: str
    chunk_index: int
    distance: float


class RetrievalError(Exception):
    """Document retrieval failed."""


class DocumentRetriever:
    """Retrieve relevant document chunks from the local vector store."""

    def __init__(
        self,
        store: ChromaVectorStore,
        embeddings: EmbeddingProvider,
        limit: int = 5,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.limit = limit

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve the closest stored chunks.

        Args:
            query (str): User question to search for.

        Returns:
            list[RetrievedChunk]: Retrieved chunks in similarity order.

        Raises:
            RetrievalError: If vector search or embedding fails.
        """
        try:
            rows = self.store.search(query, self.embeddings, self.limit)
            return [RetrievedChunk(**row) for row in rows]
        except Exception as exc:
            raise RetrievalError("Document retrieval failed") from exc
