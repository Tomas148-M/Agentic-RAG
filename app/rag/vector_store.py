import hashlib

import chromadb

from app.integrations.embeddings.base import EmbeddingProvider
from app.rag.loaders import Document


class ChromaVectorStore:
    """Store document chunks and caller-generated embeddings in persistent Chroma."""

    def __init__(self, path: str, collection: str, model_name: str) -> None:
        client = chromadb.PersistentClient(path=path)
        self.collection = client.get_or_create_collection(
            name=collection, embedding_function=None, metadata={"embedding_model": model_name}
        )
        if (self.collection.metadata or {}).get("embedding_model") != model_name:
            raise ValueError("Collection embedding model differs; use a new collection")

    def replace_document(
        self, document: Document, chunks: list[str], embeddings: EmbeddingProvider
    ) -> int:
        """Replace a source's stored chunks after embedding the new content.

        Args:
            document (Document): Loaded document and source identity.
            chunks (list[str]): Non-empty chunks in source order.
            embeddings (EmbeddingProvider): Provider for chunk vectors.

        Returns:
            int: Number of chunks stored.
        """
        if not chunks:
            raise ValueError("Cannot ingest a document without chunks")
        vectors = embeddings.embed_documents(chunks)
        if len(vectors) != len(chunks):
            raise ValueError("Embedding count does not match chunk count")
        source_hash = hashlib.sha256(document.source.encode("utf-8")).hexdigest()
        ids = [f"{source_hash}:{index}" for index in range(len(chunks))]
        previous = self.collection.get(where={"source": document.source}, include=[])["ids"]
        self.collection.upsert(
            ids=ids, embeddings=vectors, documents=chunks,
            metadatas=[{"source": document.source, "chunk_index": index} for index in range(len(chunks))],
        )
        stale = list(set(previous) - set(ids))
        if stale:
            self.collection.delete(ids=stale)
        return len(chunks)

    def search(self, query: str, embeddings: EmbeddingProvider, limit: int = 5) -> list[dict]:
        """Find stored chunks similar to the query.

        Args:
            query (str): Text to embed and search for.
            embeddings (EmbeddingProvider): Provider matching the collection's model.
            limit (int): Maximum number of results.

        Returns:
            list[dict]: Results with text, source, chunk index, and distance.
        """
        if not query.strip() or limit < 1:
            raise ValueError("Query must be non-empty and limit must be positive")
        if self.collection.count() == 0:
            return []
        vectors = embeddings.embed_queries([query])
        if len(vectors) != 1:
            raise ValueError("Expected exactly one query embedding")
        result = self.collection.query(
            query_embeddings=vectors, n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": text, "source": metadata["source"], "chunk_index": metadata["chunk_index"], "distance": distance}
            for text, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
