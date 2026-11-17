from pathlib import Path

import pytest

from app.integrations.embeddings.base import EmbeddingProvider
from app.integrations.embeddings.local import LocalEmbeddingProvider
from app.rag.ingestion import ingest_document
from app.rag.vector_store import ChromaVectorStore


class FakeEmbeddings(EmbeddingProvider):
    """Provide deterministic vectors without downloading an embedding model."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one fixed-size vector per text."""
        return [[float(len(text)), 1.0] for text in texts]


def test_ingest_replaces_chunks_and_persists(tmp_path: Path) -> None:
    """Reingestion removes stale chunks while preserving other sources."""
    path = tmp_path / "chroma"
    source = tmp_path / "one.txt"
    other = tmp_path / "two.txt"
    source.write_text("abcdefghij", encoding="utf-8")
    other.write_text("second", encoding="utf-8")
    store = ChromaVectorStore(str(path), "documents", "test-model")
    embedding = FakeEmbeddings()
    assert ingest_document(source, store, embedding, 4, 1) == 3
    assert ingest_document(other, store, embedding, 20, 0) == 1
    source.write_text("new", encoding="utf-8")
    assert ingest_document(source, store, embedding, 4, 1) == 1
    reopened = ChromaVectorStore(str(path), "documents", "test-model")
    assert reopened.collection.count() == 2
    results = reopened.search("new", embedding)
    assert {result["text"] for result in results} == {"new", "second"}
    assert {result["source"] for result in results} == {str(source.resolve()), str(other.resolve())}
    with pytest.raises(ValueError, match="embedding model"):
        ChromaVectorStore(str(path), "documents", "other-model")


def test_local_sentence_transformer_with_chroma(tmp_path: Path) -> None:
    """A real local Sentence Transformers model can ingest and search without downloads."""
    from sentence_transformers import SentenceTransformer, models

    model_path = tmp_path / "model"
    SentenceTransformer(modules=[models.BoW(vocab=["apple", "pear"])]).save(str(model_path))
    provider = LocalEmbeddingProvider(str(model_path))
    store = ChromaVectorStore(str(tmp_path / "chroma"), "local-test", str(model_path))
    source = tmp_path / "fruit.txt"
    source.write_text("apple pear", encoding="utf-8")
    assert ingest_document(source, store, provider, 100, 0) == 1
    assert store.search("apple", provider)[0]["text"] == "apple pear"
