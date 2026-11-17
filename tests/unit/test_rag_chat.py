from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.integrations.llm.base import LLMProviderError
from app.main import create_app
from app.rag.retriever import DocumentRetriever, RetrievalError, RetrievedChunk
from app.services.chat_service import ChatService


@pytest.fixture
def chunks() -> list[RetrievedChunk]:
    """Return two distinguishable retrieved passages."""
    return [
        RetrievedChunk("First passage", "/one.txt", 0, 0.1),
        RetrievedChunk("Second passage", "/two.txt", 3, 0.2),
    ]


def test_retriever_preserves_source_metadata() -> None:
    """Search results retain their exact text, provenance and distance."""
    store = Mock()
    store.search.return_value = [
        {"text": "hello", "source": "file.txt", "chunk_index": 2,
         "distance": 0.25}
    ]
    embeddings = Mock()
    retriever = DocumentRetriever(store, embeddings, limit=3)
    result = retriever.retrieve("question")
    store.search.assert_called_once_with("question", embeddings, 3)
    assert result == [RetrievedChunk("hello", "file.txt", 2, 0.25)]


def test_retriever_wraps_store_errors() -> None:
    """Search failures are surfaced as retrieval errors."""
    store = Mock()
    store.search.side_effect = RuntimeError("private detail")
    with pytest.raises(RetrievalError):
        DocumentRetriever(store, Mock()).retrieve("question")


def test_empty_retrieval_skips_generation() -> None:
    """No model call occurs when no documents are available."""
    retriever = Mock()
    retriever.retrieve.return_value = []
    llm = Mock()
    response = ChatService(retriever, llm).answer("question")
    assert response.citations == []
    assert "No relevant documents" in response.answer
    llm.generate.assert_not_called()


def test_citations_map_only_referenced_chunks(chunks: list[RetrievedChunk]) -> None:
    """Repeated citation markers map to one correctly attributed source."""
    retriever = Mock()
    retriever.retrieve.return_value = chunks
    llm = Mock()
    llm.generate.return_value = "Answer [S2]. Another fact [S2]."
    response = ChatService(retriever, llm).answer("Question?")
    assert response.answer == "Answer [S2]. Another fact [S2]."
    assert [citation.model_dump() for citation in response.citations] == [
        {"id": "S2", "source": "/two.txt", "chunk_index": 3,
         "excerpt": "Second passage"}
    ]
    prompt = llm.generate.call_args.args[0]
    assert '"id": "S1"' in prompt
    assert '"id": "S2"' in prompt
    assert "Question?" in prompt


@pytest.mark.parametrize("answer", [
    "A claim without a citation.",
    "Invented [S3].",
    "Invalid [S0].",
    "Valid [S1], invented [T9].",
])
def test_missing_or_invalid_citation_fails(
    chunks: list[RetrievedChunk], answer: str,
) -> None:
    """Do not return uncited answers or nonexistent source references."""
    retriever = Mock()
    retriever.retrieve.return_value = chunks
    llm = Mock()
    llm.generate.return_value = answer
    with pytest.raises(LLMProviderError):
        ChatService(retriever, llm).answer("question")


def test_brackets_that_are_not_citations_are_preserved(
    chunks: list[RetrievedChunk],
) -> None:
    """Ordinary bracketed text does not become a bogus source reference."""
    retriever = Mock()
    retriever.retrieve.return_value = chunks
    llm = Mock()
    llm.generate.return_value = "[Summary] In [2026], this happened [S1]."
    response = ChatService(retriever, llm).answer("question")
    assert response.answer == llm.generate.return_value


@pytest.mark.parametrize("question", ["", "   ", "x" * 4001])
def test_chat_rejects_invalid_questions(question: str) -> None:
    """Blank or oversized questions return a validation error."""
    app = create_app()
    service = Mock()
    app.dependency_overrides[get_chat_service] = lambda: service
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"question": question})
    assert response.status_code == 422
    service.answer.assert_not_called()


@pytest.mark.parametrize("error,status", [
    (RetrievalError("internal detail"), 503),
    (LLMProviderError("internal detail"), 502),
])
def test_chat_maps_backend_failures(error: Exception, status: int) -> None:
    """Failures map to stable HTTP errors without leaking internal details."""
    app = create_app()
    service = Mock()
    service.answer.side_effect = error
    app.dependency_overrides[get_chat_service] = lambda: service
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"question": "hello"})
    assert response.status_code == status
    assert "internal detail" not in response.text
