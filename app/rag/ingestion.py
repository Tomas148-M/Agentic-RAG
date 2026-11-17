from pathlib import Path

from app.integrations.embeddings.base import EmbeddingProvider
from app.rag.chunking import chunk_text
from app.rag.loaders import load_document
from app.rag.vector_store import ChromaVectorStore


def ingest_document(
    path: Path, store: ChromaVectorStore, embeddings: EmbeddingProvider, size: int, overlap: int
) -> int:
    """Load, split, embed, and store one document.

    Args:
        path (Path): Source document path.
        store (ChromaVectorStore): Destination vector store.
        embeddings (EmbeddingProvider): Local embedding provider.
        size (int): Chunk size in characters.
        overlap (int): Character overlap between chunks.

    Returns:
        int: Number of chunks stored.
    """
    document = load_document(path)
    return store.replace_document(document, chunk_text(document.text, size, overlap), embeddings)
