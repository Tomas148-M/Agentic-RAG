from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.dependencies import get_llm_provider, get_retriever
from app.integrations.embeddings.base import EmbeddingProvider
from app.main import create_app
from app.rag.ingestion import ingest_document
from app.rag.retriever import DocumentRetriever
from app.rag.vector_store import ChromaVectorStore


class TestEmbeddings(EmbeddingProvider):
    """Produce deterministic vectors for the Chroma integration test."""

    __test__ = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for each supplied text."""
        return [[float(len(text)), 1.0] for text in texts]


def test_chat_endpoint_with_persistent_chroma(tmp_path: Path) -> None:
    """An ingested document is retrieved and cited by the chat endpoint."""
    source = tmp_path / "document.txt"
    source.write_text("Czech capital is Prague.", encoding="utf-8")
    embeddings = TestEmbeddings()
    store = ChromaVectorStore(str(tmp_path / "db"), "documents", "fake")
    ingest_document(source, store, embeddings, 100, 0)
    retriever = DocumentRetriever(store, embeddings)
    llm = Mock()
    llm.generate.return_value = "Prague is the capital. [S1]"
    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm_provider] = lambda: llm
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat", json={"question": "What is the capital?"}
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Prague is the capital. [S1]"
    assert payload["citations"] == [
        {"id": "S1", "source": str(source.resolve()), "chunk_index": 0,
         "excerpt": "Czech capital is Prague."}
    ]
    assert "Czech capital is Prague." in llm.generate.call_args.args[0]
